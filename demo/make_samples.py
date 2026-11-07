"""Pick the demo's sample clips from the saved evaluation results and copy them to demo/samples/.

Needs data/ (the original audio) once; the demo itself only needs demo/samples/. Run from the project root:
    python demo/make_samples.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from halluspace import SR, load_audio  # noqa: E402

EVAL = ROOT / "results" / "small" / "eval"
OUT = ROOT / "demo" / "samples"
TAU_TUNED = 0.17
SETS = {
    "ESC_50__folds_4_5_": ("ESC-50", False),
    "UrbanSound8K": ("UrbanSound8K", False),
    "FSD50K__non_speech_": ("FSD50K", False),
    "LibriSpeech_test_clean": ("LibriSpeech test-clean", True),
    "LibriSpeech_test_other": ("LibriSpeech test-other", True),
}
SAVED_MODES = {"plain": "original", "paper": "gated_paper", "ours": "gated", "affine": "gated_affine"}
FIELDS = ("text", "raw_text", "p_ns_base", "p_ns", "projected", "rejected")


def load(mode, name):
    return [json.loads(l) for l in open(EVAL / mode / f"{name}.jsonl", encoding="utf8")]


def clips():
    """One dict per evaluation clip with the saved result of every demo method."""
    for name, (label, speech) in SETS.items():
        recs = {k: load(m, name) for k, m in SAVED_MODES.items()}
        for i, orig in enumerate(recs["plain"]):
            res = {k: {f: r[i][f] for f in FIELDS} for k, r in recs.items()}
            assert all(r[i]["path"] == orig["path"] for r in recs.values())
            rej = orig["p_ns"] > TAU_TUNED      # plain Whisper re-thresholded: same decode, other tau
            res["tau"] = dict(text="" if rej else orig["raw_text"], raw_text=orig["raw_text"], p_ns_base=orig["p_ns"],
                              p_ns=orig["p_ns"], projected=False, rejected=rej)
            yield dict(path=orig["path"], ref=orig["ref"], dataset=label, speech=speech, saved=res,
                       errors=orig.get("errors"), ref_words=orig.get("ref_words"))


def duration(path):
    i = sf.info(path)
    return i.frames / i.samplerate


# ------------------------------------------------------------ cases: (title, caption, test, count, max seconds)
H = lambda c, k: bool(c["saved"][k]["text"])          # non-speech clip still has text = hallucination
R = lambda c, k: c["saved"][k]["rejected"]
PROJ = ("paper", "ours")

CASES = [
    ("Hallucination removed by every method",
     "Plain Whisper writes text for a non-speech sound; all four alternatives stay silent.",
     lambda c: not c["speech"] and H(c, "plain") and not any(H(c, k) for k in ("paper", "ours", "affine", "tau")), 3, 10),
    ("Only the tuned threshold helps",
     "Both projection settings still hallucinate; lowering the threshold to 0.17 removes it.",
     lambda c: not c["speech"] and H(c, "plain") and all(H(c, k) for k in PROJ) and not H(c, "tau"), 2, 10),
    ("Only the projection helps",
     "The edit removes a hallucination that the tuned threshold keeps; the rarer case.",
     lambda c: not c["speech"] and H(c, "plain") and not any(H(c, k) for k in PROJ) and H(c, "tau"), 1, 10),
    ("The affine variant misses",
     "Eq. (1) and the threshold remove the hallucination; the affine edit does not.",
     lambda c: not c["speech"] and H(c, "plain") and not any(H(c, k) for k in (*PROJ, "tau")) and H(c, "affine"), 1, 10),
    ("Speech lost to the projection",
     "Real speech that a projection rejects while the tuned threshold keeps it.",
     lambda c: c["speech"] and any(R(c, k) for k in PROJ) and not R(c, "tau") and not R(c, "plain"), 2, 14),
    ("Speech lost to the threshold",
     "The threshold has a cost too: this utterance is rejected at 0.17 but kept by our projection.",
     lambda c: c["speech"] and R(c, "tau") and not R(c, "ours"), 1, 14),
    ("Clean speech",
     "Every method transcribes it without a single error.",
     lambda c: c["speech"] and not any(R(c, k) for k in c["saved"]) and c["errors"] == 0 and 10 <= c["ref_words"] <= 30, 2, 14),
]


TYPICAL = ("thank", "watching", "subscribe", "bye", "you")
US8K_CLASSES = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine idling", "gun shot",
                "jackhammer", "siren", "street music"]


def sound_label(c):
    """What the clip actually contains, from the dataset's own metadata."""
    p = Path(c["path"])
    if c["speech"]:
        return "read speech"
    if c["dataset"] == "UrbanSound8K":
        return US8K_CLASSES[int(p.stem.split("-")[1])]
    if c["dataset"] == "ESC-50":
        import csv
        meta = next(p.parents[1].glob("meta/esc50.csv"))
        return next(r["category"] for r in csv.DictReader(open(meta)) if r["filename"] == p.name).replace("_", " ")
    import csv
    gt = p.parents[1] / "FSD50K.ground_truth" / "eval.csv"
    labels = next(r["labels"] for r in csv.DictReader(open(gt, encoding="utf8")) if r["fname"] == p.stem).split(",")
    return labels[0].replace("_", " ").replace(" and ", " & ")


def rank(c):
    if c["speech"]:
        return (abs(c["ref_words"] - 18),)
    t = c["saved"]["plain"]["raw_text"].strip()
    letters = sum(ch.isalpha() for ch in t)
    return (letters < 3, not any(w in t.lower() for w in TYPICAL), abs(len(t) - 20))


def main():
    all_clips = list(clips())
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.wav"):
        old.unlink()
    chosen, used = [], set()
    for title, caption, test, count, max_s in CASES:
        cands = [c for c in all_clips if c["path"] not in used and test(c)]
        # prefer recognisable hallucinations ("Thank you.", real phrases) and medium-length utterances, varied datasets
        cands.sort(key=rank)
        picked, seen = [], set()
        for c in cands:
            if len(picked) == count:
                break
            if c["dataset"] in seen and len({x["dataset"] for x in cands}) > len(seen):
                continue
            if duration(c["path"]) > max_s:
                continue
            picked.append(c)
            seen.add(c["dataset"])
        print(f"{title}: {len(cands)} candidates, picked {len(picked)}")
        for c in picked:
            used.add(c["path"])
            chosen.append(dict(c, case=title, caption=caption))

    samples = []
    for n, c in enumerate(chosen, 1):
        audio = load_audio(c["path"])[: 30 * SR]
        fname = f"s{n:02d}.wav"
        sf.write(OUT / fname, audio, SR, subtype="PCM_16")
        samples.append(dict(id=f"s{n:02d}", file=fname, case=c["case"], caption=c["caption"], dataset=c["dataset"],
                            sound=sound_label(c),
                            speech=c["speech"], ref=c["ref"], source=Path(c["path"]).name,
                            seconds=round(len(audio) / SR, 2), saved=c["saved"]))
        print(f"  s{n:02d}  {c['dataset']:24s} {samples[-1]['sound']:22s} plain: {c['saved']['plain']['raw_text'][:50]!r}")
    (OUT / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf8")
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 2**20
    print(f"{len(samples)} samples, {size:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
