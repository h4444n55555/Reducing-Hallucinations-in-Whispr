"""Unit tests on a tiny randomly initialised Whisper: no downloads, CPU, ~20 s."""
import pytest
import torch
import whisper
from whisper.model import ModelDimensions, Whisper

from halluspace import HallucinationProjector, ProjConfig, first_step, fit_basis, project
from halluspace.subspace import protect_speech

torch.manual_seed(0)
DIMS = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=64, n_audio_head=4,
                       n_audio_layer=2, n_vocab=51865, n_text_ctx=448, n_text_state=64,
                       n_text_head=4, n_text_layer=4)


@pytest.fixture(scope="module")
def model():
    return Whisper(DIMS).eval()


@pytest.fixture(scope="module")
def mel():
    return torch.randn(3, 80, 3000)


def engine(model, mode, basis=None, **cfg):
    cfg = ProjConfig(**{**dict(layer=3, rank=2, alpha=1.0, gamma=0.0, tau=0.6), **cfg})
    return HallucinationProjector(model, cfg, basis, mode, sample_len=6)


def block_output(model, feats, tok, layer):
    seen = []
    h = model.decoder.blocks[layer - 1].register_forward_hook(lambda m, i, o: seen.append(o.clone()))
    first_step(model, feats, tok)
    h.remove()
    return seen[0]


def test_basis_is_orthonormal():
    H, F = torch.randn(30, 64) + 2, torch.randn(50, 64)
    B = fit_basis(H, F, rank=4)
    assert torch.allclose(B @ B.T, torch.eye(4), atol=1e-5)
    assert fit_basis(H, F, rank=4, method="diffmean").shape == (1, 64)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_projection_removes_component(model, mel, alpha):
    eng = engine(model, "original")
    feats = model.embed_audio(mel)
    B = fit_basis(torch.randn(20, 64) + 1, torch.randn(20, 64), rank=2)
    h = block_output(model, feats, eng.tok, 3)
    with project(model, 3, B, alpha):
        h2 = block_output(model, feats, eng.tok, 3)
    assert torch.allclose(h2 @ B.T, (1 - alpha) * (h @ B.T), atol=1e-4)   # in-subspace part scaled
    resid = (h2 - h) - ((h2 - h) @ B.T) @ B
    assert resid.abs().max() < 1e-4                                       # nothing else changed
    assert torch.allclose(block_output(model, feats, eng.tok, 3), h)     # hook removed afterwards


def test_affine_center_moves_to_center(model, mel):
    eng = engine(model, "original")
    feats = model.embed_audio(mel)
    B, c = fit_basis(torch.randn(20, 64) + 1, torch.randn(20, 64), rank=2), torch.randn(64)
    with project(model, 3, B, 1.0, center=c):
        h2 = block_output(model, feats, eng.tok, 3)
    assert torch.allclose(h2 @ B.T, (c @ B.T).expand_as(h2 @ B.T), atol=1e-4)


def test_first_step_matches_decode(model, mel):
    eng = engine(model, "original")
    feats = model.embed_audio(mel)
    p = first_step(model, feats, eng.tok)
    decoded = whisper.decode(model, feats, eng.opts)
    assert torch.allclose(p, torch.tensor([r.no_speech_prob for r in decoded]), atol=1e-6)


def test_modes(model, mel):
    B = fit_basis(torch.randn(20, 64) + 1, torch.randn(20, 64), rank=2)
    orig = engine(model, "original")(mel)
    same = engine(model, "always", B, alpha=0.0)(mel)            # alpha = 0 is a no-op
    assert [r.raw_text for r in orig] == [r.raw_text for r in same]
    always = engine(model, "always", B)(mel)
    assert all(r.projected for r in always)
    gamma = sorted(r.p_ns_base for r in orig)[1]                  # gate opens for 2 of 3
    gated = engine(model, "gated", B, gamma=gamma)(mel)
    assert [r.projected for r in gated] == [r.p_ns_base >= gamma for r in orig]
    gate_only = engine(model, "gate_only", B, gamma=gamma)(mel)
    assert [r.rejected for r in gate_only] == [r.p_ns_base >= gamma for r in orig]
    rej = engine(model, "original", tau=0.0)(mel)                  # tau = 0 rejects everything
    assert all(r.rejected and r.text == "" for r in rej)


def test_capture_shapes(model, mel):
    res, states = engine(model, "original")(mel, capture=True)
    assert states.shape == (3, DIMS.n_text_layer, DIMS.n_text_state) and len(res) == 3


def test_protect_speech_is_orthogonal():
    B = fit_basis(torch.randn(20, 64) + 1, torch.randn(20, 64), rank=3)
    U = torch.linalg.qr(torch.randn(64, 5)).Q.T
    B2 = protect_speech(B, U)
    assert (B2 @ U.T).abs().max() < 1e-5
    assert torch.allclose(B2 @ B2.T, torch.eye(3), atol=1e-5)
