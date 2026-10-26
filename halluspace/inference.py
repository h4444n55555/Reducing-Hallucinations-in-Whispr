"""Gated low-rank decoder projection (Sec. 3.4, Algorithm 1)."""
import contextlib
from dataclasses import dataclass

import torch
import whisper
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer

from .data import load_mels
from .hooks import first_step, project

MODES = ("original", "always", "gated", "gate_only")


@dataclass
class Result:
    text: str         # final transcript ("" if rejected as no-speech)
    raw_text: str     # decoded text before the no-speech filter
    p_ns_base: float  # unprojected no-speech probability (the gate input)
    p_ns: float       # no-speech probability used for the final decision
    projected: bool   # was the projection applied?
    rejected: bool    # p_ns > tau


class HallucinationProjector:
    """mode="original" : plain Whisper + no-speech filter (tau)
    mode="always"   : projection on every input
    mode="gated"    : projection only if p_ns_base >= gamma (the paper's final method)
    mode="gate_only": reject every input with p_ns_base >= gamma, no projection.
                      Not in the paper (analysis A1): how much of "gated" the gate alone does.
    """

    def __init__(self, model, cfg, basis=None, mode="gated", center=None, sot_only=False,
                 language="en", **decode_kwargs):
        assert mode in MODES, mode
        self.model, self.cfg, self.mode = model, cfg, mode
        self.basis, self.center, self.sot_only = basis, center, sot_only
        self.fp16 = model.device.type == "cuda"
        self.tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                 language=language, task="transcribe")
        self.opts = whisper.DecodingOptions(language=language, without_timestamps=True,
                                            fp16=self.fp16, **decode_kwargs)

    def _projection(self):
        return project(self.model, self.cfg.layer, self.basis, self.cfg.alpha,
                       center=self.center, sot_only=self.sot_only)

    @torch.no_grad()
    def __call__(self, mel, capture=False):
        """mel: [batch, n_mels, 3000]. Returns list[Result] (and decoder states if capture)."""
        mel = mel.to(self.model.device)
        feats = self.model.embed_audio(mel.half() if self.fp16 else mel)  # encode once
        p_base = first_step(self.model, feats, self.tok, capture=capture)
        if capture:
            p_base, states = p_base

        if self.mode == "original":
            gate = torch.zeros(len(mel), dtype=torch.bool)
        elif self.mode == "always":
            gate = torch.ones(len(mel), dtype=torch.bool)
        else:
            gate = p_base >= self.cfg.gamma

        # Algorithm 1 decodes everything once unprojected and re-decodes gated inputs.
        # p_ns_base is fixed by the first step alone, so we read it from `first_step`
        # and decode each input once, with or without the hook: same outputs, less compute.
        out = [None] * len(mel)
        for flag in (False, True):
            idx = torch.nonzero(gate == flag).flatten().tolist()
            if not idx:
                continue
            if flag and self.mode == "gate_only":
                for i in idx:
                    p = p_base[i].item()
                    out[i] = Result("", "", p, p, False, True)
                continue
            ctx = self._projection() if flag else contextlib.nullcontext()
            with ctx:
                decoded = whisper.decode(self.model, feats[idx], self.opts)
            for i, r in zip(idx, decoded):
                rejected = r.no_speech_prob > self.cfg.tau
                out[i] = Result("" if rejected else r.text.strip(), r.text.strip(),
                                p_base[i].item(), r.no_speech_prob, flag, rejected)
        return (out, states) if capture else out

    def run(self, items, batch_size=16, capture=False):
        """Transcribe a manifest [(audio_path, ref), ...] in batches."""
        results, states = [], []
        for i in tqdm(range(0, len(items), batch_size), desc=self.mode, leave=False):
            mel = load_mels([p for p, _ in items[i:i + batch_size]], self.model.dims.n_mels)
            res = self(mel, capture=capture)
            if capture:
                res, s = res
                states.append(s)
            results += res
        return (results, torch.cat(states)) if capture else results
