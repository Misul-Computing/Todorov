"""Per-tensor and whole-model weight quantization (data-free, in-place).

Methods (the ones that produced the headline numbers in runs/):

  rtn          , per-group uniform symmetric (the honest baseline).
  whlm         , group-local WH + Lloyd-Max (std-scale).
  whrtn        , group-local WH + uniform.
  fullrot_whlm , full-dim randomized WH + Lloyd-Max  (best data-free: 40.6 ppl @ 3.1 bpw).
  fullrot_whrtn, full-dim randomized WH + uniform.
  nf4          , per-group NF4 codebook (absmax-scale).

bits/weight = bits + 16/group  (b-bit index per weight + one FP16 scale per group).
The Hadamard is structured (free) and the Lloyd-Max codebook is shared (free).
"""
import time
import torch
import torch.nn as nn

from .rotate import hadamard_matrix, fwht, next_pow2, signs_for, optimal_signs_for
from .codebook import (lloyd_max_gaussian, quant_codebook_std,
                       quant_uniform_sym, quant_nf4, NF4)
from .vq import build_codebook, quant_rvq, bpw_for_rvq, get_or_build_codebook, clear_cache


def _adjust_group(d_in, G):
    while d_in % G and G > 1:
        G //= 2
    return G


def quant_dequant(W, bits, group_size=128, method="fullrot_whlm",
                  centroids=None, nf4=None, fullrot_chunk=8192):
    """Quantize + dequantize a single weight matrix.

    Returns (Wq, recon_rel_err, bpw, group_used).
    recon_rel_err = ||Wq - W||_F / ||W||_F.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    if method == "nf4":
        G = _adjust_group(d_in, group_size)
        blocks = Wf.reshape(d_out, d_in // G, G)
        deq = quant_nf4(blocks, nf4 if nf4 is not None else NF4.to(W.device))
        bpw = bits + 16.0 / G
    elif method == "rtn":
        G = _adjust_group(d_in, group_size)
        blocks = Wf.reshape(d_out, d_in // G, G)
        deq = quant_uniform_sym(blocks, bits)
        bpw = bits + 16.0 / G
    elif method in ("whlm", "whrtn"):
        G = _adjust_group(d_in, group_size)
        blocks = Wf.reshape(d_out, d_in // G, G)
        H = hadamard_matrix(G, W.device)
        blocks = blocks @ H
        if method == "whlm":
            deq = quant_codebook_std(blocks, centroids)
        else:
            deq = quant_uniform_sym(blocks, bits)
        deq = deq @ H
        bpw = bits + 16.0 / G
    elif method in ("fullrot_whlm", "fullrot_whrtn", "fullrot_whlm_opt", "fullrot_whrtn_opt"):
        npad = next_pow2(d_in)
        if "opt" in method:
            s, _ = optimal_signs_for(W, npad, W.device)
        else:
            s = signs_for(npad, W.device)
        bounds = (centroids[1:] + centroids[:-1]) / 2 if centroids is not None else None
        out = torch.empty_like(W)
        for i in range(0, d_out, fullrot_chunk):
            Wc = Wf[i:i + fullrot_chunk]
            Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
            Wp[:, :d_in] = Wc
            Wr = fwht(Wp * s)
            if method == "fullrot_whlm":
                sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
                q = centroids[torch.bucketize(Wr / sc, bounds)] * sc
            else:
                q = quant_uniform_sym(Wr, bits)
            out[i:i + fullrot_chunk] = (fwht(q) * s)[:, :d_in].to(W.dtype)
        G = group_size  # bpw reported at the nominal group; rotation is free
        bpw = bits + 16.0 / G
        err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
        return out, err, bpw, G
    elif method.startswith("fullrot_vq"):
        parts = method.split(":")
        cb_type = parts[1] if len(parts) > 1 else "gaussian"
        vq_bits = int(parts[2]) if len(parts) > 2 else 2
        n_stages = int(parts[3]) if len(parts) > 3 else 1
        outlier_pct = float(parts[4]) if len(parts) > 4 else 0.0
        d = 8 if cb_type in ("e8", "datafit8", "weighted8", "gaussian8") else 4
        bits_per_stage = vq_bits / n_stages
        assert bits_per_stage * d == int(bits_per_stage * d), \
            f"bits_per_stage*d must be integer: {bits_per_stage}*{d}"
        bps = bits_per_stage
        npad = next_pow2(d_in)
        assert npad % d == 0, f"padded dim {npad} not divisible by vq_dim {d}"
        s = signs_for(npad, W.device)
        out = torch.empty_like(W)
        all_blocks = []
        for i in range(0, d_out, fullrot_chunk):
            Wc = Wf[i:i + fullrot_chunk]
            Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
            Wp[:, :d_in] = Wc
            Wr = fwht(Wp * s)
            sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
            Wrn = Wr / sc
            n_groups = npad // d
            blocks = Wrn.reshape(Wc.shape[0], n_groups, d)
            all_blocks.append(blocks)
        all_blocks = torch.cat(all_blocks, dim=0).reshape(-1, d)
        if outlier_pct > 0:
            norms = all_blocks.norm(dim=1)
            k = max(1, int(norms.numel() * outlier_pct / 100))
            outlier_thresh = norms.kthvalue(norms.numel() - k + 1).values
            outlier_mask = norms > outlier_thresh
            normal_blocks = all_blocks[~outlier_mask]
            all_blocks[outlier_mask]
        else:
            outlier_thresh = None
            normal_blocks = all_blocks
        if cb_type in ("datafit", "datafit8", "weighted", "weighted8"):
            cache_key = f"{cb_type}:{d}:{bps}:{n_stages}:{outlier_pct}"
            def _build():
                return [build_codebook(cb_type, d, bps, W.device, rvq_stage=s,
                        data_blocks=normal_blocks, act_weights=None)
                        for s in range(n_stages)]
            codebooks = get_or_build_codebook(cache_key, _build)
        else:
            codebooks = [build_codebook(cb_type, d, bps, W.device, rvq_stage=s)
                         for s in range(n_stages)]
        for i in range(0, d_out, fullrot_chunk):
            Wc = Wf[i:i + fullrot_chunk]
            Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
            Wp[:, :d_in] = Wc
            Wr = fwht(Wp * s)
            sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
            Wrn = Wr / sc
            n_groups = npad // d
            blocks = Wrn.reshape(Wc.shape[0], n_groups, d)
            if outlier_pct > 0:
                norms = blocks.norm(dim=-1)
                is_outlier = norms > outlier_thresh
                q_normal, _ = quant_rvq(blocks[~is_outlier], codebooks) if (~is_outlier).any() else (torch.empty(0, d, device=W.device), [])
                q = torch.zeros_like(blocks)
                if (~is_outlier).any():
                    q[~is_outlier] = q_normal
                if is_outlier.any():
                    q[is_outlier] = blocks[is_outlier]
            else:
                q, _ = quant_rvq(blocks, codebooks)
            q = q.reshape(Wc.shape[0], npad) * sc
            out[i:i + fullrot_chunk] = (fwht(q) * s)[:, :d_in].to(W.dtype)
        G = group_size
        bpw = bpw_for_rvq(npad, codebooks, d)
        if outlier_pct > 0:
            n_outliers = int(outlier_mask.sum().item()) if outlier_pct > 0 else 0
            outlier_bits = n_outliers * 16 * d
            bpw += outlier_bits / (d_out * d_in)
        err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
        return out, err, bpw, G

    else:
        raise ValueError(f"unknown method {method!r}")

    Wq = deq.reshape(d_out, d_in).to(W.dtype)
    err = ((Wq.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    return Wq, err, bpw, G


def quantize_model_inplace(model, bits, group_size=128, method="fullrot_whlm",
                           skip_names=(), log_every=40, verbose=True):
    """Overwrite every nn.Linear weight in place with its quantized version.

    skip_names: iterable of substrings; tensors whose name contains any are
    left at full precision (used by the picker to protect sensitive tensors
    and by quant4_v2's keep-emb ablation).

    Returns a per-tensor log of {name, shape, recon_rel_err, bpw, group}.
    """
    clear_cache()
    device = next(model.parameters()).device
    centroids = None
    if method in ("whlm", "fullrot_whlm", "fullrot_whlm_opt"):
        centroids = lloyd_max_gaussian(bits, device)
    nf4 = NF4.to(device) if method == "nf4" else None
    skip = tuple(skip_names)
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    log = []
    t0 = time.time()
    for i, (name, mod) in enumerate(targets):
        if any(s in name for s in skip):
            log.append({"name": name, "shape": list(mod.weight.shape),
                        "recon_rel_err": 0.0, "bpw": 16.0, "group": 0,
                        "skipped": True})
            continue
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant(mod.weight.data, bits, group_size,
                                            method, centroids, nf4)
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


def snapshot(model):
    """CPU copy of every nn.Linear weight (for restore between configs)."""
    return {n: m.weight.data.detach().to("cpu", copy=True)
            for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def restore(model, snap):
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and n in snap:
                m.weight.data.copy_(snap[n].to(m.weight.device))
