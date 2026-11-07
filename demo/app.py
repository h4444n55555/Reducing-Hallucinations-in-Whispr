"""Interactive demo: run a sample clip or your own recording through the five methods of the report.

    python demo/app.py            (opens http://127.0.0.1:7860)

The model loads in the background. Until it is ready, or if it cannot load, sample clips show the saved results
of our run (results/small/eval); recordings need the live model.
"""
import argparse
import base64
import json
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import whisper
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from whisper.normalizers import EnglishTextNormalizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from halluspace import METHODS, SR, Runner, load_audio  # noqa: E402

CALIB = ROOT / "results" / "small" / "calibration" / "calib.pt"
SAMPLES = json.loads((HERE / "samples" / "samples.json").read_text(encoding="utf8"))
BY_ID = {s["id"]: s for s in SAMPLES}
MAX_SECONDS = 30
normalize = EnglishTextNormalizer()

state = {"status": "loading", "error": None, "runner": None, "device": None}
lock = threading.Lock()          # one clip at a time on the GPU


def load_model():
    try:
        runner = Runner(CALIB)
        runner.run(np.zeros(SR, dtype=np.float32))           # warm-up
        dev = torch.cuda.get_device_name(0) if runner.device == "cuda" else "CPU"
        state.update(runner=runner, status="ready", device=dev)
        print(f"[demo] model ready on {dev}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        state.update(status="failed", error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------------------------------- helpers

def spectrogram(audio, max_cols=900):
    """Whisper's 80-band log-Mel input for the clip's own length, as uint8 [80, T] (low bands first)."""
    audio = audio[: MAX_SECONDS * SR]
    if len(audio) < SR // 10:
        audio = np.pad(audio, (0, SR // 10 - len(audio)))
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio)).numpy()       # Whisper's own scaling
    if mel.shape[1] > max_cols:
        idx = np.linspace(0, mel.shape[1] - 1, max_cols).astype(int)
        mel = mel[:, idx]
    lo, hi = np.percentile(mel, 1), mel.max()
    img = np.clip((mel - lo) / max(hi - lo, 1e-6), 0, 1)
    return dict(rows=img.shape[0], cols=img.shape[1],
                data=base64.b64encode((img * 255).astype(np.uint8).tobytes()).decode())


def peaks(audio, cols=900):
    """Min/max per column for drawing the waveform, scaled to the clip's peak."""
    audio = audio[: MAX_SECONDS * SR]
    edges = np.linspace(0, len(audio), cols + 1).astype(int)
    lo = [float(audio[a:b].min()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    hi = [float(audio[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    pk = max(1e-4, max(map(abs, lo + hi)))
    return dict(min=[round(v / pk, 3) for v in lo], max=[round(v / pk, 3) for v in hi])


def signal(audio):
    return dict(spectrogram=spectrogram(audio), wave=peaks(audio), seconds=round(min(len(audio), MAX_SECONDS * SR) / SR, 2))


def word_diff(ref, hyp):
    """Alignment of normalised hypothesis against the reference: tokens tagged eq / sub / del / ins."""
    import jiwer
    r, h = normalize(ref), normalize(hyp)
    rw = r.split()
    if not h.strip():
        return dict(tokens=[dict(op="del", ref=w) for w in rw], errors=len(rw), words=len(rw))
    out = jiwer.process_words(r, h)
    hw = h.split()
    toks, errors = [], 0
    for ch in out.alignments[0]:
        if ch.type == "equal":
            toks += [dict(op="eq", word=w) for w in hw[ch.hyp_start_idx:ch.hyp_end_idx]]
            continue
        if ch.type == "substitute":
            pairs = zip(rw[ch.ref_start_idx:ch.ref_end_idx], hw[ch.hyp_start_idx:ch.hyp_end_idx])
            toks += [dict(op="sub", ref=a, word=b) for a, b in pairs]
        elif ch.type == "delete":
            toks += [dict(op="del", ref=w) for w in rw[ch.ref_start_idx:ch.ref_end_idx]]
        elif ch.type == "insert":
            toks += [dict(op="ins", word=w) for w in hw[ch.hyp_start_idx:ch.hyp_end_idx]]
        errors += max(ch.ref_end_idx - ch.ref_start_idx, ch.hyp_end_idx - ch.hyp_start_idx)
    return dict(tokens=toks, errors=errors, words=len(rw))


def finish(results, ref, source):
    by_key = {r["key"]: r for r in results}
    rows = []
    for m in METHODS:
        r = dict(by_key[m.key])
        if ref:
            r["diff"] = word_diff(ref, r["text"])
        rows.append(r)
    return rows


def live(audio):
    with lock:
        return state["runner"].run(audio)


# ------------------------------------------------------------------------------------------- app

app = FastAPI(title="Whisper hallucination demo")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/samples", StaticFiles(directory=HERE / "samples"), name="samples")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/method.svg")
def method_svg():
    return FileResponse(ROOT / "report" / "method_overview.svg", media_type="image/svg+xml")


@app.get("/report.pdf")
def report_pdf():
    return FileResponse(ROOT / "report" / "Whisper_HalluSpace_Report.pdf", media_type="application/pdf")


@app.get("/api/status")
def status():
    return dict(status=state["status"], error=state["error"], device=state["device"],
                methods=[asdict(m) for m in METHODS])


@app.get("/api/samples")
def samples():
    keep = ("id", "file", "case", "caption", "dataset", "sound", "speech", "ref", "seconds")
    return [{k: s[k] for k in keep} for s in SAMPLES]


@app.post("/api/run/sample/{sid}")
def run_sample(sid: str, mode: str = "auto"):
    s = BY_ID.get(sid)
    if s is None:
        raise HTTPException(404, "unknown sample")
    audio = load_audio(HERE / "samples" / s["file"])
    use_live = mode != "saved" and state["status"] == "ready"
    if use_live:
        results, source = live(audio), "live"
    else:
        results = [dict(key=k, **v) for k, v in s["saved"].items()]
        source = "saved"
    return dict(source=source, **signal(audio), results=finish(results, s["ref"], source))


@app.post("/api/run/audio")
async def run_audio(request: Request):
    """Body: raw float32 little-endian PCM, 16 kHz mono (resampled in the browser)."""
    if state["status"] != "ready":
        msg = "The model is still loading." if state["status"] == "loading" else f"The model failed to load: {state['error']}"
        return JSONResponse(dict(error=msg), status_code=503)
    body = await request.body()
    audio = np.frombuffer(body, dtype="<f4").astype(np.float32)[: MAX_SECONDS * SR]
    if len(audio) < SR // 4:
        return JSONResponse(dict(error="The recording is shorter than a quarter of a second."), status_code=400)
    return dict(source="live", **signal(audio), results=finish(live(audio), "", "live"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    threading.Thread(target=load_model, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    print(f"[demo] {url}  (model loading in the background)")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
