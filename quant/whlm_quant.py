"""Walsh-Hadamard + Lloyd-Max weight quantization (data-free, no training).

Pure quantization: deterministic, no calibration set, no gradients, no PTQ
reconstruction, no distillation. For each Linear weight W [d_out, d_in] we split
d_in into groups of G and, within each (row, group) block of G weights:

  1. rotate the block by a G x G normalized Hadamard H  (spreads outliers,
     Gaussianizes the per-block distribution, the QuIP/QuaRot incoherence
     trick, but data-free with a fixed structured H)
  2. quantize the rotated block with a b-bit codebook:
       - 'whlm':  Lloyd-Max-optimal centroids for N(0,1), per-block std scale
       - 'whrtn': uniform symmetric,                    per-block absmax scale
  3. de-quantize and un-rotate (H is orthonormal: H^-1 = H^T = H here)

  'rtn' = uniform symmetric per-block, NO rotation, the honest baseline.

Effective bits/weight = b + 16/G  (b-bit index per weight + one FP16 scale per
group). The Hadamard is structured (free) and the Lloyd-Max codebook is shared
across all weights (free). This is the engine's TurboQuant idea (WH + Lloyd-Max)
pointed at weights instead of the KV cache.
"""
import math
import time

import torch
import torch.nn as nn


def hadamard_matrix(G, device, dtype=torch.float32):
    assert (G & (G - 1)) == 0, f"group size {G} must be a power of 2, got {G}"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < G:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(G)  # orthonormal: H @ H.T == I, and H symmetric


def lloyd_max_gaussian(b, device, n_samples=1_000_000, iters=50, seed=0):
    """Lloyd-Max optimal centroids for a standard normal, 2**b levels (sorted)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_samples, generator=gen).to(device)
    k = 2 ** b
    qs = torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=device)
    c = torch.distributions.Normal(0.0, 1.0).icdf(qs)  # quantile init
    for _ in range(iters):
        bounds = (c[1:] + c[:-1]) / 2
        idx = torch.bucketize(x, bounds)
        for j in range(k):
            sel = x[idx == j]
            if sel.numel() > 0:
                c[j] = sel.mean()
        c, _ = torch.sort(c)
    return c


def _quant_codebook(blocks, centroids):
    """Per-block std-normalize -> nearest centroid -> de-scale. blocks: [..., G]."""
    scale = blocks.std(dim=-1, keepdim=True).clamp(min=1e-8)
    xn = blocks / scale
    bounds = (centroids[1:] + centroids[:-1]) / 2          # [k-1], sorted
    idx = torch.bucketize(xn, bounds)                      # memory-light 1D nearest
    return centroids[idx] * scale


def _quant_uniform(blocks, b):
    """Per-block symmetric uniform quant. blocks: [..., G]."""
    qmax = 2 ** (b - 1) - 1
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(blocks / scale), -qmax - 1, qmax)
    return q * scale


def quant_dequant(W, bits, group_size=128, method="whlm", centroids=None):
    """Quantize+dequantize a weight matrix; return (Wq, recon_rel_err, bpw, G)."""
    d_out, d_in = W.shape
    G = group_size
    while d_in % G != 0 and G > 1:
        G //= 2
    n_groups = d_in // G
    Wf = W.float()
    blocks = Wf.reshape(d_out, n_groups, G)
    H = None
    if method in ("whlm", "whrtn"):
        H = hadamard_matrix(G, W.device)
        blocks = blocks @ H
    if method == "whlm":
        deq = _quant_codebook(blocks, centroids)
    else:
        deq = _quant_uniform(blocks, bits)
    if H is not None:
        deq = deq @ H  # un-rotate
    Wq = deq.reshape(d_out, d_in).to(W.dtype)
    err = ((Wq.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = bits + 16.0 / G
    return Wq, err, bpw, G


def quantize_model_inplace(model, bits, group_size=128, method="whlm",
                           log_every=40, verbose=True):
    """Overwrite every nn.Linear weight in place with its quantized version.

    Returns a per-tensor log of {name, shape, recon_rel_err, bpw, group}.
    """
    device = next(model.parameters()).device
    centroids = lloyd_max_gaussian(bits, device) if method == "whlm" else None
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    log = []
    t0 = time.time()
    for i, (name, mod) in enumerate(targets):
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant(mod.weight.data, bits, group_size,
                                            method, centroids)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "shape": list(mod.weight.shape),
                    "recon_rel_err": err, "bpw": bpw, "group": G})
        if verbose and (i % log_every == 0 or i == len(targets) - 1):
            print(f"    [{method} b{bits}] {i + 1}/{len(targets)} {name} "
                  f"err={err:.4f} bpw={bpw:.3f}", flush=True)
    if verbose:
        print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
              f"({method}, b={bits})", flush=True)
    return log


if __name__ == "__main__":
    # Self-test: rotation+codebook should beat plain RTN on a synthetic weight.
    torch.manual_seed(0)
    W = torch.randn(1536, 1536) * 0.02
    W[:, ::97] += torch.randn(1536, 16) * 0.3  # inject outlier channels
    for b in (4, 3, 2):
        cb = lloyd_max_gaussian(b, W.device)
        _, e_rtn, bpw, _ = quant_dequant(W, b, method="rtn")
        _, e_wh, _, _ = quant_dequant(W, b, method="whlm", centroids=cb)
        print(f"b={b} bpw~{bpw:.2f}  rtn_err={e_rtn:.4f}  whlm_err={e_wh:.4f}  "
              f"improvement={100 * (1 - e_wh / e_rtn):.1f}%")
