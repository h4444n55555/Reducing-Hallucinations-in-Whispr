"""Step 1 (Sec. 3.1): run Whisper on ESC-50 folds 1-3, label each clip by its own outcome
(hallucinated vs. correctly empty) and store the SOT decoder state after every block."""
from pathlib import Path

import torch
import whisper

from halluspace import data
from halluspace.baselines import build_bag
from halluspace.cli import config, parser
from halluspace.inference import HallucinationProjector

p = parser(__doc__)
p.add_argument("--esc50", required=True)
p.add_argument("--librispeech", help="also store dev-clean speech states (for --protect)")
p.add_argument("--speech-limit", type=int, default=500)
p.add_argument("--limit", type=int, help="random subset of calibration clips (analysis A3)")
args = p.parse_args()

model = whisper.load_model(args.model, device=args.device)
original = HallucinationProjector(model, config(args), mode="original")

items = data.esc50(args.esc50, folds=(1, 2, 3))
if args.limit:
    items = [items[i] for i in torch.randperm(len(items), generator=torch.Generator().manual_seed(0))[:args.limit]]
results, states = original.run(items, args.batch_size, capture=True)
hall = torch.tensor([bool(r.text) for r in results])
out = {"model": args.model, "states": states.half(), "hallucinated": hall,
       "texts": [r.raw_text for r in results], "bag": build_bag(r.text for r in results)}

if args.librispeech:
    speech = data.librispeech(args.librispeech, "dev-clean", args.speech_limit)
    out["speech_states"] = original.run(speech, args.batch_size, capture=True)[1].half()

Path(args.run).mkdir(parents=True, exist_ok=True)
torch.save(out, Path(args.run, "calib.pt"))
print(f"{int(hall.sum())} hallucinated / {int((~hall).sum())} correctly empty "
      f"(dev HR {100 * hall.float().mean():.1f}%), {states.shape[1]} layers, d={states.shape[2]}")
print("BoH phrases:", out["bag"])
