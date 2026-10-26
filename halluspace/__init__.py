"""Hallucination space projection for Whisper (arXiv:2609.04561), minimal re-implementation."""
from .config import PAPER_CONFIGS, ProjConfig
from .hooks import first_step, project
from .inference import MODES, HallucinationProjector, Result
from .subspace import fit_basis, principal_directions, protect_speech

__all__ = ["PAPER_CONFIGS", "ProjConfig", "first_step", "project", "MODES",
           "HallucinationProjector", "Result", "fit_basis", "principal_directions",
           "protect_speech"]
