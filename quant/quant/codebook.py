"""Weight codebooks and per-block quantizers.

Three codebook families, all data-free:

  * NF4           , the QLoRA 4-bit normal-floating codebook (fixed, absmax-scale).
  * Lloyd-Max     , optimal centroids for N(0,1), per-block std-scale.  Beats
                      uniform at low bit-widths because real (rotated) weights are
                      roughly Gaussian.
  * data-fit Lloyd, Lloyd-Max run on the REAL absmax-normalized weight
                      distribution (heavy-tailed); the OF4 codebook from quant4_v2.

And the uniform symmetric baseline (per-block absmax-scale).
"""
import torch

# QLoRA NF4 codebook (absmax-normalized, 16 levels).
NF4 = torch.tensor([
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.28444138, -0.18477343,
    -0.09105004, 0.0, 0.07958030, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261700, 0.72295684, 1.0,
])


def lloyd_max_gaussian(b, device, n_samples=1_000_000, iters=40, seed=0):
    """Lloyd-Max optimal centroids for a standard normal; 2**b levels, sorted."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_samples, generator=g).to(device)
    k = 2 ** b
    c = torch.distributions.Normal(0.0, 1.0).icdf(
        torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=device))
    for _ in range(iters):
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        for j in range(k):
            s = x[idx == j]
            if s.numel():
                c[j] = s.mean()
        c, _ = torch.sort(c)
    return c


def lloyd_on_data(x, k=16, iters=40, init_frac=0.03):
    """Lloyd-Max fit to a real (already normalized) weight distribution.

    Used to build OF4: Lloyd on the absmax-normalized weights (heavy-tailed),
    which beats NF4's Gaussian-quantile assumption.
    """
    c = torch.quantile(x[:200000].sort().values,
                       torch.linspace(init_frac, 1 - init_frac, k, device=x.device))
    for _ in range(iters):
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        for j in range(k):
            s = x[idx == j]
            if s.numel():
                c[j] = s.mean()
        c, _ = torch.sort(c)
    return c


def quant_codebook_std(blocks, centroids):
    """Per-block std-normalize -> nearest centroid -> de-scale.  blocks: [..., G]."""
    scale = blocks.std(dim=-1, keepdim=True).clamp(min=1e-8)
    xn = blocks / scale
    bounds = (centroids[1:] + centroids[:-1]) / 2
    idx = torch.bucketize(xn, bounds)
    return centroids[idx] * scale


def quant_uniform_sym(blocks, bits):
    """Per-block symmetric uniform quant.  blocks: [..., G]."""
    qmax = 2 ** (bits - 1) - 1
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(blocks / scale), -qmax - 1, qmax)
    return q * scale


def quant_nf4(blocks, nf4):
    """Per-block absmax-normalize -> NF4 nearest level -> de-scale.  blocks: [..., G]."""
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    xn = (blocks / scale).clamp(nf4[0].item(), nf4[-1].item())
    return nf4[torch.bucketize(xn, (nf4[1:] + nf4[:-1]) / 2)] * scale
