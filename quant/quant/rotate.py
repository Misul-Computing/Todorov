"""Walsh-Hadamard transforms and randomized incoherence rotation.

Two rotation granularities are used by the quantizers:

  * group-local: a G x G normalized Hadamard applied per (row, group) block
    (the QuaRot/QuIP block-incoherence trick, data-free with a fixed H).
  * full-dimension: a randomized sign flip + full-dim FWHT over the whole
    input dimension (pads to next power of 2).  Spreads outliers across the
    ENTIRE dimension, which is what makes low-bit quant work, the lever
    `runs/whlm/fullrot.json` identified as the real win over group-local.

All transforms are orthonormal and self-inverse (H @ H.T == I, H == H.T).
"""
import math
import torch

_SIGNS = {}


def hadamard_matrix(G, device, dtype=torch.float32):
    """G x G normalized Walsh-Hadamard matrix.  G must be a power of 2."""
    assert G >= 1 and (G & (G - 1)) == 0, f"G must be a power of 2, got {G}"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < G:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(G)


def fwht(x):
    """In-place-style normalized Fast Walsh-Hadamard Transform along the last dim.

    n = x.shape[-1] must be a power of 2.  Self-inverse: fwht(fwht(x)) == x.
    """
    n = x.shape[-1]
    assert n >= 1 and (n & (n - 1)) == 0, f"last dim must be a power of 2, got {n}"
    shape = x.shape
    x = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.cat([(a + b).unsqueeze(2), (a - b).unsqueeze(2)], dim=2).reshape(-1, n)
        h *= 2
    return (x / math.sqrt(n)).reshape(shape)


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def signs_for(npad, device, seed=1234):
    """Deterministic +/-1 sign vector of length npad (seeded, cached per npad+seed)."""
    key = (npad, seed)
    if key not in _SIGNS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        _SIGNS[key] = (torch.randint(0, 2, (npad,), generator=g).to(device) * 2 - 1).float()
    return _SIGNS[key]


def optimal_signs_for(W, npad, device, K=8, sample_rows=128):
    """Find the sign vector that produces the most Gaussian rotated weights.

    Tries K random sign vectors, applies FWHT on a subset of rows, and picks
    the one with kurtosis closest to 3 (Gaussian). This is data-free and
    per-matrix, each weight matrix gets its own optimized rotation.

    Returns (signs, seed).
    """
    d_out, d_in = W.shape
    n_sample = min(sample_rows, d_out)
    Wsub = W[:n_sample].float()
    Wp = torch.zeros(n_sample, npad, device=device)
    Wp[:, :d_in] = Wsub

    best_kurt = float('inf')
    best_seed = 1234 + npad

    for k in range(K):
        seed = 1234 + npad * 100 + k
        s = signs_for(npad, device, seed)
        Wr = fwht(Wp * s)
        # Kurtosis: (mean(x^4) / mean(x^2)^2), 3.0 for Gaussian
        m2 = (Wr ** 2).mean()
        m4 = (Wr ** 4).mean()
        kurt = (m4 / (m2 ** 2 + 1e-12)).item()
        # Distance from Gaussian kurtosis (3.0)
        kurt_excess = abs(kurt - 3.0)
        if kurt_excess < best_kurt:
            best_kurt = kurt_excess
            best_seed = seed

    return signs_for(npad, device, best_seed), best_seed
