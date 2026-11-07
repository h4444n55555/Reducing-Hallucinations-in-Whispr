"""Hallucination space projection for Whisper, for single clips (used by the demo).

The functions below are the same as in the notebook's core cells (section 2). `Runner` loads Whisper and the saved
calibration states, rebuilds the bases, and runs one clip through the five methods compared in the report.
"""
import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import resample_poly
from whisper.tokenizer import get_tokenizer

SR = whisper.audio.SAMPLE_RATE


# ---------------------------------------------------------------- core (identical to the notebook)

@torch.no_grad()
def first_step(model, feats, tokenizer):
    """No-speech probability at the SOT position of the first decoding step (what whisper.decode reports)."""
    prompt = torch.tensor([tokenizer.sot_sequence_including_notimestamps], device=feats.device).repeat(len(feats), 1)
    logits = model.decoder(prompt, feats)
    return logits[:, 0].float().softmax(-1)[:, tokenizer.no_speech].cpu()


@contextlib.contextmanager
def project(model, layer, basis, alpha=1.0, center=None):
    """Edit the output of decoder block `layer` (1-indexed): h <- h - alpha * ((h - c) B^T) B.
    center=None is Eq. (1) (c = 0); the affine variant uses c = mean state of the correctly-empty clips."""
    block = model.decoder.blocks[layer - 1]
    dev = next(block.parameters()).device
    B = basis.to(dev).float()
    c = None if center is None else center.to(dev).float()

    def hook(_m, _i, out):
        x = out.float()
        coord = (x if c is None else x - c) @ B.T
        return (x - alpha * coord @ B).to(out.dtype)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def half_weights(model):
    """Cast Linear/Conv/Embedding weights to fp16 once; LayerNorms stay fp32."""
    from whisper.model import Linear, Conv1d
    for m in model.modules():
        if isinstance(m, (Linear, Conv1d, torch.nn.Embedding)):
            m.half()
    return model


def fit_basis(H, F, rank):
    """Rows of Delta are h_hall_i - h_empty_i (paired in order); B = top-`rank` right singular vectors."""
    n = min(len(H), len(F))
    delta = H[:n].double() - F[:n].double()
    return torch.linalg.svd(delta, full_matrices=False).Vh[:rank].float()


def load_audio(path):
    """16 kHz mono float32."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------- the five methods of the report

@dataclass(frozen=True)
class Method:
    key: str
    name: str
    detail: str
    tau: float
    layer: int = 0          # 0 = no projection
    rank: int = 0
    alpha: float = 1.0
    gamma: float = 0.0
    affine: bool = False


METHODS = [
    Method("plain", "Plain Whisper", "default threshold, no edit", tau=0.6),
    Method("paper", "Projection, paper setting", "Eq. (1), layer 10, rank 1", tau=0.6, layer=10, rank=1, gamma=0.15),
    Method("ours", "Projection, our setting", "Eq. (1), layer 11, rank 8", tau=0.6, layer=11, rank=8, gamma=0.05),
    Method("affine", "Affine projection", "Eq. (2), layer 10, rank 2", tau=0.6, layer=10, rank=2, gamma=0.05, affine=True),
    Method("tau", "Plain Whisper, tuned threshold", "threshold chosen on dev, no edit", tau=0.17),
]


class Runner:
    """Whisper small + calibration bases; `run(audio)` gives every method's decision for one clip."""

    def __init__(self, calib_path, model_name="small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = self.device == "cuda"
        torch.set_grad_enabled(False)
        model = whisper.load_model(model_name, device="cpu").eval().requires_grad_(False)
        if self.fp16:
            half_weights(model)
        self.model = model.to(self.device)
        self.tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
        self.opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=self.fp16)

        ck = torch.load(calib_path, weights_only=False)
        states, hall = ck["states"].float(), ck["hallucinated"]
        self.edit = {}
        for m in METHODS:
            if m.layer:
                s = states[:, m.layer - 1]
                basis = fit_basis(s[hall], s[~hall], m.rank)
                center = s[~hall].mean(0) if m.affine else None
                self.edit[m.key] = (basis, center)

    def _decode(self, feats):
        return whisper.decode(self.model, feats, self.opts)[0].text.strip()

    @torch.no_grad()
    def run(self, audio):
        """audio: 16 kHz mono float32. Every clip is decoded by every method (also rejected ones, to show what
        Whisper would have written); rejection is p_ns > tau, as in the notebook."""
        mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)).to(self.device), self.model.dims.n_mels)
        feats = self.model.embed_audio(mel[None].half() if self.fp16 else mel[None])
        p_base = float(first_step(self.model, feats, self.tok)[0])
        plain_text = None
        out = []
        for m in METHODS:
            gate = bool(m.layer) and p_base >= m.gamma
            if gate:
                basis, center = self.edit[m.key]
                with project(self.model, m.layer, basis, m.alpha, center):
                    p = float(first_step(self.model, feats, self.tok)[0])
                    text = self._decode(feats)
            else:
                if plain_text is None:
                    plain_text = self._decode(feats)
                p, text = p_base, plain_text
            rejected = p > m.tau
            out.append(dict(key=m.key, p_ns_base=p_base, p_ns=p, projected=gate, rejected=rejected,
                            raw_text=text, text="" if rejected else text))
        return out
