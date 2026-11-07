"use strict";
/* Whisper hallucination demo: front end. No external libraries, so it also works offline. */

const $ = (id) => document.getElementById(id);
const SR = 16000;
const MAX_REC_S = 30;

const S = {
  samples: [], methods: [], status: "loading",
  pref: "live",            // live | saved
  current: null,           // {kind: "sample", sample} | {kind: "audio", pcm, label}
  results: null, busy: false,
};

let audioCtx = null;
const ctx = () => (audioCtx ||= new (window.AudioContext || window.webkitAudioContext)());

/* ------------------------------------------------------------------ helpers */
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function toast(msg, ms = 3200) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), ms);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || `request failed (${r.status})`);
  return j;
}

/* ------------------------------------------------------------------ status and source */
async function pollStatus() {
  try {
    const s = await api("/api/status");
    S.methods = s.methods;
    setStatus(s.status, s.device, s.error);
    if (s.status === "loading") setTimeout(pollStatus, 1500);
  } catch {
    setStatus("failed", null, "server not reachable");
  }
}

function setStatus(status, device, error) {
  const prev = S.status;
  S.status = status;
  const el = $("status");
  el.dataset.state = status;
  el.querySelector("span").textContent =
    status === "ready" ? `Whisper small · ${device}` : status === "loading" ? "Loading Whisper…" : "Model unavailable · saved results only";
  if (error) el.title = error;
  $("src-live").disabled = status !== "ready";
  if (status !== "ready") setPref("saved", false);
  else if (prev !== "ready") {
    setPref("live", false);
    toast("Model ready. Results are now computed live.");
    if (S.current?.kind === "sample") run();
  }
}

function setPref(p, rerun = true) {
  S.pref = p;
  $("src-live").setAttribute("aria-pressed", p === "live");
  $("src-saved").setAttribute("aria-pressed", p === "saved");
  if (rerun && S.current?.kind === "sample") run();
}

/* ------------------------------------------------------------------ samples list */
async function loadSamples() {
  S.samples = await api("/api/samples");
  const ol = $("samples");
  ol.innerHTML = "";
  let lastCase = null;
  S.samples.forEach((s, i) => {
    if (s.case !== lastCase) {
      lastCase = s.case;
      const h = document.createElement("li");
      h.className = "case";
      h.textContent = s.case;
      ol.appendChild(h);
    }
    const li = document.createElement("li");
    li.innerHTML = `<button class="sample" data-id="${s.id}">
        <span class="sno">S${i + 1}</span>
        <span class="sname ${s.speech ? "quote" : ""}">${esc(sampleName(s))}</span>
        <span class="smeta">${esc(s.dataset)} · ${s.seconds.toFixed(1)} s</span>
      </button>`;
    li.firstElementChild.addEventListener("click", () => selectSample(s));
    ol.appendChild(li);
  });
}

/* non-speech: what the sound is; speech: its opening words */
function sampleName(s) {
  if (!s.speech) return s.sound.charAt(0).toUpperCase() + s.sound.slice(1);
  const w = s.ref.toLowerCase().split(" ");
  return `“${w[0].charAt(0).toUpperCase() + w[0].slice(1)} ${w.slice(1, 4).join(" ")}…”`;
}

async function selectSample(s) {
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-current", b.dataset.id === s.id));
  S.current = { kind: "sample", sample: s };
  const n = S.samples.indexOf(s) + 1;
  setClipHeader(s.speech ? `Speech: ${sampleName(s)}` : sampleName(s), `S${n} · ${s.dataset} · ${s.seconds.toFixed(1)} s`);
  setReference(s.speech ? s.ref : "");
  setAudioSource(`/samples/${s.file}`);
  run();
}

function setClipHeader(title, meta) {
  $("clip-title").textContent = title;
  $("clip-meta").textContent = meta;
}

function setReference(text) {
  $("reference").hidden = !text;
  $("ref-text").textContent = text ? text.charAt(0) + text.slice(1).toLowerCase() : "";
}

/* ------------------------------------------------------------------ running */
function truth() {
  if (S.current?.kind === "sample") return S.current.sample.speech ? "speech" : "nonspeech";
  return document.querySelector('input[name="truth"]:checked').value;
}

async function run() {
  if (!S.current) return;
  const token = (run._n = (run._n || 0) + 1);
  showLoadingRows();
  try {
    let res;
    if (S.current.kind === "sample") {
      const mode = S.pref === "live" ? "auto" : "saved";
      res = await api(`/api/run/sample/${S.current.sample.id}?mode=${mode}`, { method: "POST" });
    } else {
      res = await api("/api/run/audio", {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: S.current.pcm.buffer,
      });
    }
    if (token !== run._n) return;           // a newer request superseded this one
    S.results = res;
    drawWave(res.wave);
    drawSpec(res.spectrogram);
    renderRows(res);
  } catch (e) {
    if (token !== run._n) return;
    $("rows").innerHTML = `<div class="empty-rows">${esc(e.message)}</div>`;
    $("source").hidden = true;
  }
}

/* ------------------------------------------------------------------ signal: waveform, spectrogram, playback */
/* CSS fixes the displayed size; the backing store follows it at the screen's pixel ratio. */
function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

let lastWave = null, lastSpec = null;

/* waveform from per-column min/max (sent by the server; computed here only for audio the server has not seen) */
function peaksFrom(samples, cols = 900) {
  const min = [], max = [];
  let pk = 1e-4;
  for (let c = 0; c < cols; c++) {
    const a = Math.floor((c * samples.length) / cols), b = Math.max(a + 1, Math.floor(((c + 1) * samples.length) / cols));
    let lo = 0, hi = 0;
    for (let i = a; i < b && i < samples.length; i++) { const v = samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    min.push(lo); max.push(hi); pk = Math.max(pk, -lo, hi);
  }
  return { min: min.map((v) => v / pk), max: max.map((v) => v / pk) };
}

function drawWave(p) {
  lastWave = p;
  $("scope-empty").hidden = true;
  const [g, w, h] = sizeCanvas($("wave"));
  g.clearRect(0, 0, w, h);
  g.strokeStyle = "#ddd7ca";
  g.beginPath(); g.moveTo(0, h / 2 + 0.5); g.lineTo(w, h / 2 + 0.5); g.stroke();
  g.fillStyle = "#45423c";
  const n = p.min.length, amp = h / 2 - 6;
  for (let x = 0; x < w; x++) {
    const i = Math.min(n - 1, Math.floor((x / w) * n));
    const y1 = h / 2 - p.max[i] * amp, y2 = h / 2 - p.min[i] * amp;
    g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
}

/* sequential ramp, one hue: paper -> accent blue -> deep ink blue */
const RAMP = [[255, 253, 248], [214, 224, 238], [139, 165, 204], [29, 79, 145], [12, 30, 58]];
function rampColor(t) {
  const x = Math.min(0.9999, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.floor(x), f = x - i, a = RAMP[i], b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function drawSpec(spec) {
  lastSpec = spec;
  const [g, w, h] = sizeCanvas($("spec"));
  const raw = Uint8Array.from(atob(spec.data), (c) => c.charCodeAt(0));
  const img = new ImageData(spec.cols, spec.rows);
  for (let r = 0; r < spec.rows; r++) {
    for (let c = 0; c < spec.cols; c++) {
      const [R, G, B] = rampColor(raw[r * spec.cols + c] / 255);
      const o = ((spec.rows - 1 - r) * spec.cols + c) * 4;     // low frequencies at the bottom
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = spec.cols; off.height = spec.rows;
  off.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true;
  g.drawImage(off, 0, 0, w, h);
}

function setAudioSource(url) {
  const a = $("audio");
  a.pause();
  a.src = url;
  $("play").disabled = false;
  updateTime();
}

function updateTime() {
  const a = $("audio");
  const d = isFinite(a.duration) ? a.duration : 0;
  $("time").textContent = `${fmtTime(a.currentTime)} / ${fmtTime(d)}`;
  const ph = $("playhead");
  if (d > 0) {
    ph.style.opacity = a.currentTime > 0 ? 1 : 0;
    ph.style.left = `${(a.currentTime / d) * 100}%`;
  }
  $("play").innerHTML = a.paused
    ? '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>'
    : '<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';
}

function tick() {
  updateTime();
  if (!$("audio").paused) requestAnimationFrame(tick);
}

function togglePlay() {
  const a = $("audio");
  if (!a.src) return;
  if (a.paused) a.play().then(tick).catch(() => {}); else a.pause();
}

/* ------------------------------------------------------------------ decision rows */
function showLoadingRows() {
  $("source").hidden = true;
  const sk = '<div class="skeleton" style="width:%w%"></div>';
  $("rows").innerHTML = headRow() + S.methods.map((m) => `
    <div class="row">
      <div class="m-name">${esc(m.name)}<span class="m-detail">${esc(m.detail)}</span></div>
      <div>${sk.replace("%w", 80)}</div><div>${sk.replace("%w", 70)}</div><div>${sk.replace("%w", 60)}</div>
    </div>`).join("");
}

function headRow() {
  const norm = truth() === "speech" ? " (normalised, compared with the reference)" : "";
  return `<div class="row head"><span>Method</span><span>No-speech probability, 0 → 1</span><span>Verdict</span><span class="out">Output${norm}</span></div>`;
}

function verdict(r, kind) {
  const hasText = !!r.text;
  if (kind === "nonspeech") {
    if (hasText) return ["critical", "✗ Hallucinated", ""];
    return ["good", "✓ Correct", r.rejected ? "rejected" : "no text"];
  }
  if (kind === "speech") {
    if (r.rejected) return ["warning", "! Falsely rejected", "speech thrown away"];
    if (r.diff) {
      const wer = r.diff.words ? (100 * r.diff.errors) / r.diff.words : 0;
      if (r.diff.errors === 0) return ["good", "✓ Correct", "no word errors"];
      return ["neutral", `${r.diff.errors} word error${r.diff.errors > 1 ? "s" : ""}`, `WER ${wer.toFixed(0)}%`];
    }
    return ["good", "✓ Kept", ""];
  }
  return r.rejected ? ["neutral", "Rejected", "no output"] : ["neutral", hasText ? "Kept" : "Empty", ""];
}

function outputCell(r, kind) {
  if (r.diff && !r.rejected) {
    return `<div class="diff">${r.diff.tokens.map((t) =>
      t.op === "eq" ? esc(t.word)
        : t.op === "sub" ? `<span class="d-sub" title="reference: ${esc(t.ref)}">${esc(t.word)}</span>`
        : t.op === "del" ? `<span class="d-del" title="missing">${esc(t.ref)}</span>`
        : `<span class="d-ins" title="extra word">${esc(t.word)}</span>`).join(" ")}</div>`;
  }
  if (r.text) return `<q>${esc(r.text)}</q>`;
  let html = `<span class="none">no output</span>`;
  if (r.rejected && r.raw_text) html += `<span class="would">would have written <q>${esc(r.raw_text)}</q></span>`;
  if (r.rejected && kind === "speech" && r.diff) html += `<span class="would">all ${r.diff.words} words count as deleted</span>`;
  return html;
}

function gauge(m, r) {
  const W = 230, x0 = 7, x1 = 190, y = 16;
  const X = (p) => x0 + (x1 - x0) * Math.min(1, Math.max(0, p));
  const tx = X(m.tau);
  let s = `<svg viewBox="0 0 ${W} 38" role="img" aria-label="no-speech probability ${r.p_ns.toFixed(2)}, threshold ${m.tau}">`;
  s += `<rect x="${tx}" y="${y - 9}" width="${x1 - tx}" height="18" fill="var(--reject)"/>`;
  s += `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" stroke="var(--rule-2)" stroke-width="1.5"/>`;
  s += `<line x1="${tx}" x2="${tx}" y1="${y - 10}" y2="${y + 10}" stroke="var(--critical)" stroke-width="1.5"/>`;
  s += `<text class="lab" x="${tx}" y="${y + 21}" text-anchor="middle">τ ${m.tau}</text>`;
  if (m.layer) {
    const gx = X(m.gamma);
    s += `<line x1="${gx}" x2="${gx}" y1="${y - 8}" y2="${y + 8}" stroke="var(--muted)" stroke-width="1.2" stroke-dasharray="2 2"/>`;
    s += `<text class="lab" x="${gx}" y="${y + 21}" text-anchor="middle">γ</text>`;
  }
  const bx = X(r.p_ns_base), ex = X(r.p_ns);
  if (r.projected) {
    s += `<line class="arrow" x1="${bx}" x2="${bx}" y1="${y}" y2="${y}" stroke="var(--accent)" stroke-width="1.5" data-to="${ex}"/>`;
    s += `<circle cx="${bx}" cy="${y}" r="4.5" fill="var(--card)" stroke="var(--ink-2)" stroke-width="1.5"/>`;
    s += `<circle class="dot-edit" cx="${bx}" cy="${y}" r="5.5" fill="var(--accent)" stroke="var(--card)" stroke-width="2" data-to="${ex}"/>`;
  } else {
    s += `<circle cx="${bx}" cy="${y}" r="5" fill="var(--ink)" stroke="var(--card)" stroke-width="2"/>`;
  }
  s += `<text class="num" x="${W - 2}" y="${y + 4}" text-anchor="end">${r.p_ns.toFixed(2)}</text>`;
  return s + "</svg>";
}

function animateGauges() {
  const t0 = performance.now(), D = 650;
  const items = [...document.querySelectorAll(".rows [data-to]")].map((el) => ({
    el, from: parseFloat(el.getAttribute(el.tagName === "line" ? "x2" : "cx")), to: parseFloat(el.dataset.to),
  }));
  const step = (now) => {
    const k = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - k, 3);
    for (const it of items) it.el.setAttribute(it.el.tagName === "line" ? "x2" : "cx", it.from + (it.to - it.from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderRows(res) {
  const kind = truth();
  const byKey = Object.fromEntries(res.results.map((r) => [r.key, r]));
  $("rows").innerHTML = headRow() + S.methods.map((m) => {
    const r = byKey[m.key];
    const [cls, label, sub] = verdict(r, kind);
    const gate = m.layer
      ? `<span class="m-gate ${r.projected ? "open" : ""}">${r.projected ? "gate open, edited" : "gate closed, not edited"}</span>`
      : "";
    return `<div class="row ${m.key === "tau" ? "key" : ""}">
      <div class="m-name">${esc(m.name)}<span class="m-detail">${esc(m.detail)}</span>${gate}</div>
      <div class="gauge">${gauge(m, r)}</div>
      <div class="verdict"><span class="chip ${cls}">${label}</span>${sub ? `<small>${sub}</small>` : ""}</div>
      <div class="out">${outputCell(r, kind)}</div>
    </div>`;
  }).join("");
  const src = $("source");
  src.hidden = false;
  src.className = `source ${res.source}`;
  src.textContent = res.source === "live" ? "computed live" : "from the saved run";
  const note = $("case-note");
  if (S.current.kind === "sample") {
    note.hidden = false;
    note.innerHTML = `<b>${esc(S.current.sample.case)}.</b> ${esc(S.current.sample.caption)}`;
  } else note.hidden = true;
  animateGauges();
}

/* ------------------------------------------------------------------ your audio: microphone and files */
const R = { rec: null, stream: null, chunks: [], t0: 0, timer: null, analyser: null, raf: null };

async function toResampled(arrayBuffer) {
  const decoded = await ctx().decodeAudioData(arrayBuffer);
  const len = Math.max(1, Math.ceil(decoded.duration * SR));
  const off = new OfflineAudioContext(1, len, SR);           // also mixes down to mono
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice(0, MAX_REC_S * SR);
}

async function useAudio(blob, label) {
  try {
    const pcm = await toResampled(await blob.arrayBuffer());
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-current", "false"));
    S.current = { kind: "audio", pcm, label };
    setClipHeader(label, `${(pcm.length / SR).toFixed(1)} s · scored as: ${{ none: "not scored", speech: "speech", nonspeech: "no speech" }[truth()]}`);
    setReference("");
    setAudioSource(URL.createObjectURL(blob));
    drawWave(peaksFrom(pcm));
    run();
  } catch (e) {
    console.error(e);
    toast("Could not read that audio.");
  }
}

async function startRec() {
  try {
    // Browser noise suppression would remove exactly the sounds we want Whisper to hear.
    R.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch {
    toast("Microphone access was blocked.");
    return;
  }
  R.chunks = [];
  R.rec = new MediaRecorder(R.stream);
  R.rec.ondataavailable = (e) => e.data.size && R.chunks.push(e.data);
  R.rec.onstop = () => {
    R.stream.getTracks().forEach((t) => t.stop());          // release the microphone
    cancelAnimationFrame(R.raf);
    const blob = new Blob(R.chunks, { type: R.rec.mimeType });
    useAudio(blob, "Your recording");
  };
  const src = ctx().createMediaStreamSource(R.stream);
  R.analyser = ctx().createAnalyser();
  R.analyser.fftSize = 2048;
  src.connect(R.analyser);
  R.rec.start();
  R.t0 = performance.now();
  $("rec-btn").classList.add("on");
  $("rec-btn").setAttribute("aria-label", "Stop recording");
  $("rec-hint").textContent = "Recording… press again to stop.";
  liveMeter();
  R.timer = setInterval(() => {
    const s = (performance.now() - R.t0) / 1000;
    $("rec-time").textContent = fmtTime(s);
    if (s >= MAX_REC_S) stopRec();
  }, 200);
}

function stopRec() {
  if (!R.rec || R.rec.state !== "recording") return;
  clearInterval(R.timer);
  R.rec.stop();
  $("rec-btn").classList.remove("on");
  $("rec-btn").setAttribute("aria-label", "Start recording");
  $("rec-hint").textContent = "Record up to 30 s. Silence, typing or music near the microphone make good tests.";
}

function liveMeter() {
  const buf = new Float32Array(R.analyser.fftSize);
  const hist = [];
  $("scope-empty").hidden = true;
  setClipHeader("Recording…", "");
  const draw = () => {
    R.analyser.getFloatTimeDomainData(buf);
    let pk = 0;
    for (const v of buf) pk = Math.max(pk, Math.abs(v));
    hist.push(pk);
    const [g, w, h] = sizeCanvas($("wave"));
    g.clearRect(0, 0, w, h);
    g.fillStyle = "#b3261e";
    const n = Math.floor(w / 3), start = Math.max(0, hist.length - n);
    for (let i = start; i < hist.length; i++) {
      const a = Math.min(1, hist[i] * 2.5) * (h / 2 - 4) + 1;
      g.fillRect((i - start) * 3, h / 2 - a, 2, 2 * a);
    }
    R.raf = requestAnimationFrame(draw);
  };
  draw();
}

/* ------------------------------------------------------------------ wiring */
function setTab(which) {
  const rec = which === "record";
  $("tab-samples").setAttribute("aria-selected", !rec);
  $("tab-record").setAttribute("aria-selected", rec);
  $("pane-samples").hidden = rec;
  $("pane-record").hidden = !rec;
}

function stepSample(d) {
  if (!S.samples.length) return;
  const i = S.current?.kind === "sample" ? S.samples.indexOf(S.current.sample) : -1;
  const j = Math.min(S.samples.length - 1, Math.max(0, i + d));
  setTab("samples");
  selectSample(S.samples[j]);
  document.querySelector(`.sample[data-id="${S.samples[j].id}"]`)?.scrollIntoView({ block: "nearest" });
}

function wire() {
  $("src-live").onclick = () => setPref("live");
  $("src-saved").onclick = () => setPref("saved");
  $("tab-samples").onclick = () => setTab("samples");
  $("tab-record").onclick = () => setTab("record");
  $("rec-btn").onclick = () => (R.rec?.state === "recording" ? stopRec() : startRec());
  $("play").onclick = togglePlay;
  const a = $("audio");
  a.addEventListener("loadedmetadata", updateTime);
  a.addEventListener("pause", updateTime);
  a.addEventListener("ended", () => { a.currentTime = 0; updateTime(); });
  $("scope").addEventListener("click", (e) => {
    if (!a.src || !isFinite(a.duration)) return;
    const r = $("scope").getBoundingClientRect();
    a.currentTime = ((e.clientX - r.left) / r.width) * a.duration;
    updateTime();
    if (a.paused) togglePlay();
  });
  document.querySelectorAll('input[name="truth"]').forEach((el) =>
    el.addEventListener("change", () => S.current?.kind === "audio" && S.results && renderRows(S.results)));

  const drop = $("drop");
  $("file").onchange = (e) => e.target.files[0] && useAudio(e.target.files[0], e.target.files[0].name);
  ["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  document.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) { setTab("record"); useAudio(f, f.name); }
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); stepSample(1); }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); stepSample(-1); }
    else if (e.key === "r") { setTab("record"); $("rec-btn").click(); }
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(redraw, 120); });
}

function redraw() {
  if (lastWave) drawWave(lastWave);
  if (lastSpec) drawSpec(lastSpec);
}

(async function init() {
  wire();
  await pollStatus();                       // method list first, then the samples
  await loadSamples();
  const want = new URLSearchParams(location.search).get("s");          // e.g. ?s=s08 opens sample 8
  if (S.samples.length) selectSample(S.samples.find((s) => s.id === want) || S.samples[0]);
  if (new URLSearchParams(location.search).has("howto")) $("howto").open = true;
})();
