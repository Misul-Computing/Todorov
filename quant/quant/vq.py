"""Vector quantization in the Hadamard-rotated domain.

Codebook families:
  * D4 lattice      , d=4, root lattice D4
  * E8 lattice      , d=8, root lattice E8 (used by QuIP#)
  * Gaussian k-means, d=4/8, codebook from k-means on N(0,I) samples
  * Datafit k-means , d=4/8, codebook from k-means on ACTUAL rotated weights
  * Weighted k-means, d=4/8, activation-energy-weighted k-means on actual weights

Residual VQ (RVQ): M stages, each quantizes the residual of the previous.
"""
import torch
import torch.nn.functional as F


def d4_lattice_points(radius=3):
    pts = []
    r = int(radius)
    for z0 in range(-r, r + 1):
        for z1 in range(-r, r + 1):
            for z2 in range(-r, r + 1):
                for z3 in range(-r, r + 1):
                    if (z0 + z1 + z2 + z3) % 2 == 0:
                        if z0**2 + z1**2 + z2**2 + z3**2 <= radius**2:
                            pts.append([z0, z1, z2, z3])
    return torch.tensor(pts, dtype=torch.float32)


def e8_lattice_points(radius=3):
    r = int(radius)

    def _enum_shell(coords, dim, max_norm_sq, out, prefix, prefix_sq):
        if dim == 0:
            out.append(prefix[:])
            return
        for c in coords:
            new_sq = prefix_sq + c * c
            if new_sq > max_norm_sq:
                if c > 0:
                    break
                continue
            prefix.append(c)
            _enum_shell(coords, dim - 1, max_norm_sq, out, prefix, new_sq)
            prefix.pop()

    int_pts = []
    _enum_shell(list(range(-r, r + 1)), 8, radius ** 2, int_pts, [], 0)
    int_pts = [p for p in int_pts if sum(p) % 2 == 0]

    half_pts = []
    _enum_shell([c + 0.5 for c in range(-r, r + 1)], 8, radius ** 2, half_pts, [], 0)
    half_pts = [p for p in half_pts if sum(p) % 2 == 0]

    all_pts = int_pts + half_pts
    if not all_pts:
        return torch.zeros(0, 8)
    return torch.tensor(all_pts, dtype=torch.float32)


def kmeans(data, k, iters=15, seed=0, weights=None, chunk=8192, max_samples=50000):
    """k-means on actual data. If weights given, weighted k-means (AAAC-style).

    data: [N, d]  weights: [N] (per-sample activation energy) or None
    Subsamples to max_samples for speed. Returns centroids [k, d].
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, d = data.shape
    if N > max_samples:
        perm = torch.randperm(N, generator=g)[:max_samples].to(data.device)
        data = data[perm]
        if weights is not None:
            weights = weights[perm]
        N = max_samples
    perm = torch.randperm(N, generator=g)[:k].to(data.device)
    centroids = data[perm].clone()
    for _ in range(iters):
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=data.device)
        for i in range(0, N, chunk):
            dist = torch.cdist(data[i:i + chunk], centroids)
            if weights is not None:
                dist = dist * weights[i:i + chunk].unsqueeze(1)
            idx = dist.argmin(dim=1)
            one_hot = F.one_hot(idx, k).float()  # [chunk, k]
            if weights is not None:
                w = weights[i:i + chunk].unsqueeze(1)  # [chunk, 1]
                new_c += (data[i:i + chunk].unsqueeze(1) * one_hot.unsqueeze(2) * w.unsqueeze(2)).sum(0)
                counts += (one_hot * w).sum(0)
            else:
                new_c += one_hot.T @ data[i:i + chunk]  # [k, d]
                counts += one_hot.sum(0)
        mask = counts > 0
        centroids[mask] = new_c[mask] / counts[mask].unsqueeze(1)
    return centroids


def gaussian_kmeans_codebook(d, k, device, n_samples=100000, iters=20, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_samples, d, generator=g).to(device)
    return kmeans(x, k, iters=iters, seed=seed)


def build_codebook(codebook_type, d, bits, device, rvq_stage=0,
                   data_blocks=None, act_weights=None):
    """Build a VQ codebook. If data_blocks given, uses data-fit k-means."""
    k = int(2 ** (bits * d))
    if codebook_type == "d4":
        pts = d4_lattice_points(radius=max(3, bits + 1)).to(device)
        if pts.shape[0] >= k:
            norms = (pts ** 2).sum(dim=1)
            _, order = norms.sort()
            return pts[order[:k]]
        extra = gaussian_kmeans_codebook(d, k - pts.shape[0], device, seed=rvq_stage)
        return torch.cat([pts, extra], dim=0)

    elif codebook_type == "e8":
        pts = e8_lattice_points(radius=max(4, bits + 2)).to(device)
        if pts.shape[0] >= k:
            norms = (pts ** 2).sum(dim=1)
            _, order = norms.sort()
            return pts[order[:k]]
        extra = gaussian_kmeans_codebook(d, k - pts.shape[0], device, seed=rvq_stage)
        return torch.cat([pts, extra], dim=0)

    elif codebook_type in ("gaussian", "gaussian8"):
        return gaussian_kmeans_codebook(d, k, device, seed=rvq_stage)

    elif codebook_type in ("datafit", "datafit8"):
        if data_blocks is None:
            return gaussian_kmeans_codebook(d, k, device, seed=rvq_stage)
        return kmeans(data_blocks, k, iters=30, seed=rvq_stage,
                      weights=act_weights)

    elif codebook_type in ("weighted", "weighted8"):
        if data_blocks is None or act_weights is None:
            return gaussian_kmeans_codebook(d, k, device, seed=rvq_stage)
        return kmeans(data_blocks, k, iters=30, seed=rvq_stage,
                      weights=act_weights)

    else:
        raise ValueError(f"unknown codebook type {codebook_type!r}")


def quant_vq(vectors, codebook, chunk=4096):
    orig_shape = vectors.shape[:-1]
    d = vectors.shape[-1]
    flat = vectors.reshape(-1, d)
    n = flat.shape[0]
    idx = torch.empty(n, dtype=torch.long, device=flat.device)
    for i in range(0, n, chunk):
        dist = torch.cdist(flat[i:i + chunk], codebook)
        idx[i:i + chunk] = dist.argmin(dim=1)
    quantized = codebook[idx].reshape(*orig_shape, d)
    return quantized, idx.reshape(*orig_shape)


def quant_rvq(vectors, codebooks):
    residual = vectors
    total_q = torch.zeros_like(vectors)
    all_indices = []
    for cb in codebooks:
        q, idx = quant_vq(residual, cb)
        total_q = total_q + q
        residual = vectors - total_q
        all_indices.append(idx)
    return total_q, all_indices


def dequant_rvq(indices_list, codebooks):
    total = None
    for idx, cb in zip(indices_list, codebooks):
        q = cb[idx.reshape(-1)]
        if total is None:
            total = q
        else:
            total = total + q
    return total


def bpw_for_rvq(d_in, codebooks, d):
    bits_per_group = sum(int(torch.log2(torch.tensor(float(cb.shape[0]))).item())
                         for cb in codebooks)
    n_groups = d_in // d
    total_index_bits = n_groups * bits_per_group
    scale_bits = 32
    return (total_index_bits + scale_bits) / d_in


_codebook_cache = {}

def get_or_build_codebook(key, builder_fn):
    if key not in _codebook_cache:
        _codebook_cache[key] = builder_fn()
    return _codebook_cache[key]

def clear_cache():
    _codebook_cache.clear()
