"""Offline parameter selection (Sec. 5.1, Figs. 1-2 and 5-8). Grid over layer, rank, alpha
and gamma on the development splits: ESC-50 folds 1-3 (HR) and LibriSpeech dev-clean (WER, FRR).

Stage 1: --mode always --alphas 1 --layers 4 8 12 ... --ranks 1 2 4 8
Stage 2: --mode gated  --layer 28 --rank 4 --alphas .25 .5 .75 1 --gammas .03 .05 .1 .15
"""
import csv
import itertools
from pathlib import Path

from halluspace import data
from halluspace.cli import config, engine, load, parser
from halluspace.metrics import records, summarize

p = parser(__doc__)
p.add_argument("--esc50", required=True)
p.add_argument("--librispeech", required=True)
p.add_argument("--mode", default="always", choices=["always", "gated"])
p.add_argument("--layers", type=int, nargs="+")
p.add_argument("--ranks", type=int, nargs="+")
p.add_argument("--alphas", type=float, nargs="+")
p.add_argument("--gammas", type=float, nargs="+")
p.add_argument("--speech-limit", type=int, help="subsample dev-clean for speed")
args = p.parse_args()

model, calib = load(args)
base = config(args)
dev_ns = data.esc50(args.esc50, (1, 2, 3))
dev_sp = data.librispeech(args.librispeech, "dev-clean", args.speech_limit)
grid = itertools.product(args.layers or [base.layer], args.ranks or [base.rank],
                         args.alphas or [base.alpha], args.gammas or [base.gamma])

with open(Path(args.run, f"sweep_{args.mode}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "rank", "alpha", "gamma", "HR", "WER", "FRR"])
    for layer, rank, alpha, gamma in grid:
        eng = engine(args, model, calib, args.mode, layer=layer, rank=rank, alpha=alpha, gamma=gamma)
        hr = summarize(records(eng.run(dev_ns, args.batch_size), dev_ns))["HR"]
        sp = summarize(records(eng.run(dev_sp, args.batch_size), dev_sp))
        row = [layer, rank, alpha, gamma, round(hr, 2), round(sp["WER"], 2), round(sp["FRR"], 2)]
        w.writerow(row)
        f.flush()
        print(dict(zip(["layer", "rank", "alpha", "gamma", "HR", "WER", "FRR"], row)))
