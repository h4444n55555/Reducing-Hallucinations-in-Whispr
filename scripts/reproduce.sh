#!/usr/bin/env bash
# Full pipeline for one model size. Usage: bash scripts/reproduce.sh large-v3
# Expects datasets under data/ (see README "Data").
set -euo pipefail
MODEL=${1:-large-v3}
RUN=runs/$MODEL
DATA="--esc50 data/ESC-50 --us8k data/UrbanSound8K --fsd50k data/FSD50K --librispeech data/LibriSpeech"

# 1. Calibration states on ESC-50 folds 1-3 (+ dev-clean speech states for --protect)
python scripts/calibrate.py --model "$MODEL" --run "$RUN" --esc50 data/ESC-50 --librispeech data/LibriSpeech

# 2. Optional: parameter selection (slow; the paper's choices are the defaults)
# python scripts/sweep.py --model "$MODEL" --run "$RUN" --esc50 data/ESC-50 --librispeech data/LibriSpeech \
#     --mode always --alphas 1 --layers 4 8 12 18 20 24 28 30 --ranks 1 2 4 8 --speech-limit 500

# 3. Held-out evaluation: Tables 1-3 and Fig. 4, plus the gate-only ablation
python scripts/evaluate.py --model "$MODEL" --run "$RUN" $DATA --modes original always gated gate_only boh

# 4. Full threshold-only trade-off of original Whisper vs. the projection points
python scripts/tradeoff.py --eval "$RUN/eval" --plot "$RUN/tradeoff.png"
