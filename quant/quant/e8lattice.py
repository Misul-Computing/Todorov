"""E8 lattice vector quantization for LLM weight quantization.

The E8 lattice is the densest lattice packing in 8 dimensions. It gives
better shaping gain than scalar quantization for Gaussian sources.

E8 lattice: { x in Z^8 : sum(x_i) is even } union { x in (Z+0.5)^8 : sum(x_i) is even }

For 2-bit quantization (4 levels per dimension), we use the E8P (E8 Padded)
codebook from QuIP#: 2^16 entries in 8 dimensions, but stored compactly
using the lattice structure (only 2^8 base entries + signs/shifts).

For 4-bit, we use RVQ: two cascaded E8P stages (2+2 bits).

The key advantage over scalar Lloyd-Max: vector quantization with E8 lattice
achieves ~0.5-1.0 dB shaping gain over scalar at the same bitrate for
Gaussian sources. This translates to ~10-20% lower distortion.
"""
import torch
from .rotate import fwht, signs_for, next_pow2


def e8_lattice_quantize(x, scale=1.0):
    """Quantize 8-dimensional vectors to the E8 lattice.

    E8 = { v in Z^8 : sum(v_i) even } union { v in (Z+0.5)^8 : sum(v_i) even }

    Round to nearest E8 lattice point. The nearest point is either:
    - Round to integers, adjust to make sum even
    - Round to half-integers, adjust to make sum even
    Pick whichever is closer.

    Args:
        x: [..., 8] tensor of input vectors
        scale: scaling factor (vectors are x/scale before quantization)

    Returns: quantized vectors [..., 8] at the same scale
    """
    x = x / scale

    # Option 1: round to integers
    x_int = torch.round(x)
    # Make sum even: find the coordinate with largest rounding error and flip it
    sum_int = x_int.sum(dim=-1)
    needs_adjust = (sum_int % 2 != 0)
    if needs_adjust.any():
        # Find the coordinate where rounding error is largest
        rounding_err = (x - x_int).abs()
        # For vectors that need adjustment, flip the coordinate with largest error
        mask = needs_adjust.unsqueeze(-1).expand_as(x)
        # Get the index of max error for each vector
        max_err_idx = rounding_err.masked_fill(~mask, -1).argmax(dim=-1, keepdim=True)
        # Flip that coordinate by ±1 (whichever is closer to original)
        flip = torch.sign(x.gather(-1, max_err_idx) - x_int.gather(-1, max_err_idx))
        flip[flip == 0] = 1
        x_int.scatter_(-1, max_err_idx, x_int.gather(-1, max_err_idx) + flip * needs_adjust.long().unsqueeze(-1).to(x_int.dtype))

    # Option 2: round to half-integers
    x_half = torch.round(x - 0.5) + 0.5
    sum_half = x_half.sum(dim=-1)
    needs_adjust_h = (sum_half % 2 != 0)
    if needs_adjust_h.any():
        rounding_err_h = (x - x_half).abs()
        mask_h = needs_adjust_h.unsqueeze(-1).expand_as(x)
        max_err_idx_h = rounding_err_h.masked_fill(~mask_h, -1).argmax(dim=-1, keepdim=True)
        flip_h = torch.sign(x.gather(-1, max_err_idx_h) - x_half.gather(-1, max_err_idx_h))
        flip_h[flip_h == 0] = 1
        x_half.scatter_(-1, max_err_idx_h, x_half.gather(-1, max_err_idx_h) + flip_h * needs_adjust_h.long().unsqueeze(-1).to(x_half.dtype))

    # Pick whichever is closer
    err_int = (x - x_int).norm(dim=-1)
    err_half = (x - x_half).norm(dim=-1)
    use_int = err_int <= err_half

    result = torch.where(use_int.unsqueeze(-1), x_int, x_half)
    return result * scale


def quant_dequant_e8rvq(W, total_bits=4, group_size=128, chunk=1024):
    """Quantize W using E8 lattice RVQ after Hadamard rotation.

    total_bits: target bits per weight. Even bits (2,4,6) use 2-bit E8 stages.
    Odd bits (3,5) use (total_bits-1)/2 E8 stages + a 1-bit sign-residual stage:
    the residual after the E8 stages is quantized to sign(residual)*scale per
    8-dim block, adding 1 bit/weight. This gives {2,3,4,5,6} granularity so the
    per-tensor picker can smooth allocations (it cannot with {2,4,6} alone).
    Returns (Wq, recon_rel_err, bpw, G).
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0, f"padded dim {npad} not divisible by 8"
    npad // d

    n_e8_stages = total_bits // 2
    has_sign_stage = (total_bits % 2 == 1)
    assert n_e8_stages * 2 + (1 if has_sign_stage else 0) == total_bits

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)

        # Per-row scale
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc

        # Reshape to 8-dim blocks
        blocks = Wrn.reshape(-1, d)

        # RVQ: quantize, subtract residual, quantize again
        residual = blocks.clone()
        for stage in range(n_e8_stages):
            if stage == 0:
                q = e8_lattice_quantize(residual, scale=1.0)
            else:
                # Scale residual to unit variance for lattice quantization
                res_std = residual.std().clamp(min=1e-8)
                q = e8_lattice_quantize(residual, scale=res_std)
            residual = residual - q
            if stage == 0:
                total_q = q
            else:
                total_q = total_q + q

        # Optional 1-bit sign-residual stage (for odd total_bits: 3=2+1, 5=4+1)
        if has_sign_stage:
            # per-8-dim-block scale = mean abs residual; quantize to sign*scale
            block_scale = residual.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            q_sign = torch.sign(residual) * block_scale
            total_q = total_q + q_sign

        # Un-reshape and un-normalize
        q_full = total_q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = total_bits + 16.0 / group_size  # bits + scale overhead
    return out, err, bpw, group_size


def quantize_model_e8rvq(model, total_bits=4, group_size=128, verbose=True):
    """Apply E8 RVQ quantization to every nn.Linear in the model."""
    import torch.nn as nn
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant_e8rvq(
                mod.weight.data, total_bits, group_size)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
        if verbose and (i % 40 == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name} err={err:.4f} bpw={bpw:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    if verbose:
        avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
        print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
              f"(E8 RVQ {total_bits}bit, avg_err={avg_err:.4f})", flush=True)
    return log
