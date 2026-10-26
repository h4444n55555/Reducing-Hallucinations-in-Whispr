"""Analysis A1: the full threshold-only trade-off of unmodified Whisper, recomputed
offline from eval/original/*.jsonl (no new decoding), against the measured projection points.

The paper sweeps tau only in [0.5, 0.9] (Table 8). Its gate gamma is 0.05-0.15, so the
question "is the gate alone enough?" needs tau down to ~0.01. Also reports AUROC of the
no-speech probability for separating non-speech from speech, before and after projection.
"""
import argparse
import json
from pathlib import Path

import torch

from halluspace.metrics import load_jsonl

NONSPEECH = ["ESC-50", "UrbanSound8K", "FSD50K"]
SPEECH = ["test-clean", "test-other"]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--eval", required=True, help="runs/<model>/eval folder from evaluate.py")
p.add_argument("--plot", default=None, help="optional output .png")
args = p.parse_args()
ev = Path(args.eval)


def load(mode):
    return {n: load_jsonl(ev / mode / f"{n}.jsonl") for n in NONSPEECH + SPEECH
            if (ev / mode / f"{n}.jsonl").exists()}


def at_tau(sets, tau):
    """HR / WER / FRR of original Whisper had it used threshold tau (rejects p_ns > tau)."""
    hr = [100 * sum(r["raw_text"] != "" and r["p_ns"] <= tau for r in sets[n]) / len(sets[n])
          for n in NONSPEECH if n in sets]
    frr, wer = [], []
    for n in (s for s in SPEECH if s in sets):
        rs = sets[n]
        frr.append(100 * sum(r["p_ns"] > tau for r in rs) / len(rs))
        err = sum(r["ref_words"] if r["p_ns"] > tau else r["raw_errors"] for r in rs)
        wer.append(100 * err / sum(r["ref_words"] for r in rs))
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return mean(hr), mean(wer), mean(frr)


def auroc(sets, key):
    """P(score of a non-speech clip > score of a speech clip), ties count 1/2."""
    pos = torch.tensor([r[key] for n in NONSPEECH if n in sets for r in sets[n]])
    neg = torch.tensor([r[key] for n in SPEECH if n in sets for r in sets[n]])
    ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1   # ties: ignored (rare)
    return ((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


orig = load("original")
taus = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
curve = [(t, *at_tau(orig, t)) for t in taus]
print(f"{'tau':>6} {'avg HR':>8} {'avg WER':>8} {'avg FRR':>8}   (original Whisper, threshold only)")
for t, hr, wer, frr in curve:
    print(f"{t:>6} {hr:>8.2f} {wer:>8.2f} {frr:>8.2f}")
print(f"\nAUROC of p_ns (non-speech vs speech), original: {auroc(orig, 'p_ns'):.4f}")

summary = json.loads((ev / "summary.json").read_text())
points = {}
for mode in summary:
    if mode == "original":
        continue
    s = summary[mode]
    avg = lambda key, names: (sum(s[n][key] for n in names if n in s)
                              / max(1, sum(n in s for n in names)))
    hr, wer, frr = avg("HR", NONSPEECH), avg("WER", SPEECH), avg("FRR", SPEECH)
    points[mode] = (hr, wer, frr)
    extra = (f", AUROC of p_ns {auroc(load(mode), 'p_ns'):.4f}"
             if mode in ("always", "gated") else "")
    print(f"{mode:>10}: avg HR {hr:.2f}  avg WER {wer:.2f}  avg FRR {frr:.2f}{extra}")

if args.plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([c[3] for c in curve], [c[1] for c in curve], "-o", ms=3, color="#2a6fb0",
            label="original Whisper, sweeping tau")
    for t, hr, _, frr in curve:
        if t in (0.01, 0.05, 0.1, 0.6):
            ax.annotate(f"tau={t}", (frr, hr), textcoords="offset points", xytext=(6, 4), fontsize=8)
    for mode, (hr, _, frr) in points.items():
        ax.plot(frr, hr, "s", ms=7, label=mode)
    ax.set_xlabel("speech false-rejection rate, avg % (lower is better)")
    ax.set_ylabel("non-speech hallucination rate, avg % (lower is better)")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=200)
    print("saved", args.plot)
