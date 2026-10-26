"""Forward hooks on Whisper's text decoder: read activations, or edit them with Eq. (1)."""
import contextlib

import torch


@torch.no_grad()
def first_step(model, feats, tokenizer, capture=False):
    """One decoder forward over the prompt <|sot|><|en|><|transcribe|><|notimestamps|>.

    Whisper reads no_speech_prob from the logits at the SOT position of the very first
    decoding step, so this single forward gives exactly the value `whisper.decode` reports
    (tests/test_core.py checks this). With capture=True it also returns the SOT-position
    hidden state after every decoder block: tensor [batch, n_layers, d].
    """
    prompt = torch.tensor([tokenizer.sot_sequence_including_notimestamps], device=feats.device)
    prompt = prompt.repeat(len(feats), 1)
    states, handles = [], []
    if capture:
        save = lambda _m, _i, out: states.append(out[:, 0].float().cpu())
        handles = [blk.register_forward_hook(save) for blk in model.decoder.blocks]
    try:
        logits = model.decoder(prompt, feats)
    finally:
        for h in handles:
            h.remove()
    p_ns = logits[:, 0].float().softmax(-1)[:, tokenizer.no_speech].cpu()
    return (p_ns, torch.stack(states, 1)) if capture else p_ns


@contextlib.contextmanager
def project(model, layer, basis, alpha=1.0, center=None, sot_only=False):
    """While active, replace the output h of decoder block `layer` (1-indexed) by

        h - alpha * ((h - c) B^T) B          (Eq. 1 when c = 0)

    basis : [r, d] tensor with orthonormal rows (B_{l,r}).
    center: optional [d] vector c. None reproduces the paper. Passing the mean state of
            the correctly-empty class turns zeroing into "move to the empty-class value"
            (affine variant, extension M2 in the improvements report).
    sot_only: edit only the SOT position of the first step (extension M3); later
            decoding steps pass through unchanged.
    """
    ref = next(model.parameters())
    B = basis.to(device=ref.device, dtype=ref.dtype)
    c = None if center is None else center.to(device=ref.device, dtype=ref.dtype)

    def hook(_module, _inputs, h):
        if sot_only and h.shape[1] == 1:          # incremental (kv-cached) steps
            return None
        x = h if c is None else h - c
        delta = alpha * (x @ B.T) @ B
        if sot_only:
            delta[:, 1:] = 0
        return h - delta

    handle = model.decoder.blocks[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
