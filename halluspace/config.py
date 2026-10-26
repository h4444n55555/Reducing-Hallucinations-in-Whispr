"""Operating points selected in the paper (Sec. 5.1 and Appendix C)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjConfig:
    layer: int        # decoder block whose output is edited, 1-indexed (see README "Assumptions")
    rank: int         # r: number of hallucination directions removed
    alpha: float      # projection strength in Eq. (1)
    gamma: float      # gate threshold on the unprojected no-speech probability
    tau: float = 0.6  # Whisper's no-speech rejection threshold


PAPER_CONFIGS = {
    "small":    ProjConfig(layer=10, rank=1, alpha=1.0,  gamma=0.15),
    "medium":   ProjConfig(layer=24, rank=2, alpha=0.75, gamma=0.10),
    "large-v3": ProjConfig(layer=28, rank=4, alpha=1.0,  gamma=0.05),
}

MAX_RANK = 8  # bases are stored up to this rank; any r <= MAX_RANK can be selected later
