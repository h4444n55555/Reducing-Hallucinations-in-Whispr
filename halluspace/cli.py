"""Command-line plumbing shared by the scripts."""
import argparse
from dataclasses import replace
from pathlib import Path

import torch
import whisper

from .config import PAPER_CONFIGS
from .inference import HallucinationProjector
from .subspace import fit_basis, principal_directions, protect_speech


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default="large-v3", choices=list(PAPER_CONFIGS))
    p.add_argument("--run", required=True, help="run folder (written by calibrate.py)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    g = p.add_argument_group("projection (default: the paper's setting for --model)")
    g.add_argument("--layer", type=int)
    g.add_argument("--rank", type=int)
    g.add_argument("--alpha", type=float)
    g.add_argument("--gamma", type=float)
    g.add_argument("--tau", type=float)
    e = p.add_argument_group("extensions (not in the paper)")
    e.add_argument("--basis", default="paired", choices=["paired", "diffmean"])
    e.add_argument("--pair-seed", type=int, help="random pairing of H and F rows")
    e.add_argument("--affine", action="store_true", help="project toward the empty-class mean")
    e.add_argument("--sot-only", action="store_true", help="edit the SOT position only")
    e.add_argument("--protect", type=int, default=0, help="remove top-k speech PCs from B")
    return p


def config(args, **override):
    given = {k: getattr(args, k) for k in ("layer", "rank", "alpha", "gamma", "tau")
             if getattr(args, k) is not None}
    return replace(PAPER_CONFIGS[args.model], **{**given, **override})


def load(args):
    model = whisper.load_model(args.model, device=args.device)
    return model, torch.load(Path(args.run, "calib.pt"))


def engine(args, model, calib, mode, **override):
    """Fit B for the chosen layer from the stored calibration states and wrap the model."""
    cfg = config(args, **override)
    states, hall = calib["states"][:, cfg.layer - 1].float(), calib["hallucinated"]
    B = fit_basis(states[hall], states[~hall], cfg.rank, args.basis, args.pair_seed)
    if args.protect:
        speech = calib["speech_states"][:, cfg.layer - 1].float()
        B = protect_speech(B, principal_directions(speech, args.protect))
    center = states[~hall].mean(0) if args.affine else None
    return HallucinationProjector(model, cfg, B, mode, center=center, sot_only=args.sot_only)
