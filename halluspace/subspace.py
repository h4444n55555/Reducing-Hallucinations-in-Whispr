"""Low-rank hallucination subspace estimation (Sec. 3.2) plus two optional variants."""
import torch


def fit_basis(H, F, rank, method="paired", seed=None):
    """Return B [rank, d] with orthonormal rows.

    H: [N_h, d] states of non-speech clips that hallucinated.
    F: [N_f, d] states of non-speech clips that were correctly left empty.

    method="paired"  : paper. Rows of Delta are h_hall_i - h_empty_i, n = min(N_h, N_f);
                       B = top-r right singular vectors of Delta. seed=None pairs clips in
                       index order; an integer seed pairs them at random (analysis A3:
                       sensitivity of the basis to the arbitrary pairing).
    method="diffmean": extension M1. Rank-1 difference of class means, the estimator
                       used for steering/refusal directions in LLMs.
    """
    if len(H) == 0 or len(F) == 0:
        raise ValueError(f"need both classes, got {len(H)} hallucinated / {len(F)} empty clips")
    H, F = H.double(), F.double()
    if method == "diffmean":
        D = (H.mean(0) - F.mean(0))[None]
    elif method == "paired":
        n = min(len(H), len(F))
        if seed is None:
            hi, fi = torch.arange(n), torch.arange(n)
        else:
            g = torch.Generator().manual_seed(seed)
            hi = torch.randperm(len(H), generator=g)[:n]
            fi = torch.randperm(len(F), generator=g)[:n]
        D = H[hi] - F[fi]
    else:
        raise ValueError(method)
    Vh = torch.linalg.svd(D, full_matrices=False).Vh
    return Vh[:rank].float()


def protect_speech(B, speech_pcs):
    """Extension M4: drop from span(B) the top principal directions of *speech* states,
    then re-orthonormalise, so the edit avoids directions real speech relies on."""
    U = speech_pcs.to(B)
    B = B - (B @ U.T) @ U
    return torch.linalg.qr(B.T).Q.T.contiguous()


def principal_directions(X, k):
    """Top-k principal directions of states X [N, d] (rows of the result are orthonormal)."""
    X = X.double()
    return torch.linalg.svd(X - X.mean(0), full_matrices=False).Vh[:k].float()
