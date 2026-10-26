"""Held-out evaluation (Sec. 5.2-5.4): reproduces Tables 1-3 and Fig. 4.
Writes one JSONL of per-input records per (mode, dataset) plus summary.json."""
import json
from pathlib import Path

from halluspace import data
from halluspace.baselines import apply_bag, apply_vad
from halluspace.cli import engine, load, parser
from halluspace.inference import MODES
from halluspace.metrics import records, save_jsonl, summarize

p = parser(__doc__)
p.add_argument("--modes", nargs="+", default=["original", "always", "gated"],
               choices=[*MODES, "boh", "vad"])
p.add_argument("--esc50")
p.add_argument("--us8k")
p.add_argument("--fsd50k")
p.add_argument("--librispeech")
p.add_argument("--tag", default="", help="suffix for the output folder, e.g. _affine")
args = p.parse_args()

sets = {}
if args.esc50:
    sets["ESC-50"] = data.esc50(args.esc50, (4, 5))
if args.us8k:
    sets["UrbanSound8K"] = data.urbansound8k(args.us8k)
if args.fsd50k:
    sets["FSD50K"] = data.fsd50k_nonspeech(args.fsd50k)
if args.librispeech:
    for subset in ("test-clean", "test-other"):
        sets[subset] = data.librispeech(args.librispeech, subset)

model, calib = load(args)
out_dir = Path(args.run, "eval" + args.tag)
cache, summary = {}, {}
for mode in args.modes:
    base = "original" if mode in ("boh", "vad") else mode   # BoH / VAD post-process Whisper
    eng = engine(args, model, calib, base)
    for name, items in sets.items():
        if (base, name) not in cache:
            cache[base, name] = eng.run(items, args.batch_size)
        res = cache[base, name]
        if mode == "boh":
            res = apply_bag(res, calib["bag"])
        if mode == "vad":
            res = apply_vad(res, [path for path, _ in items])
        recs = records(res, items)
        save_jsonl(recs, out_dir / mode / f"{name}.jsonl")
        summary.setdefault(mode, {})[name] = summarize(recs)
        print(mode, name, {k: (round(v, 2) if isinstance(v, float) else v)
                           for k, v in summary[mode][name].items()})

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
cols = list(sets)
print("\n" + "mode".ljust(10) + "".join(c[:14].rjust(26) for c in cols))
for mode, row in summary.items():
    cells = [", ".join(f"{k} {v:.2f}" for k, v in row[c].items() if k in ("HR", "WER", "FRR"))
             for c in cols]
    print(mode.ljust(10) + "".join(c.rjust(26) for c in cells))
