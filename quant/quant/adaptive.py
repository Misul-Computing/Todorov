"""Per-dimension bit allocation in the Hadamard domain.

After Hadamard rotation, dimensions have different variances. Instead of
uniform bit allocation, assign bits per-dimension based on variance.
High-variance dimensions get more bits, low-variance get fewer.

This is data-free (uses only weight statistics) and per-matrix. The key
insight: the Hadamard rotation makes dimensions approximately Gaussian
but with different variances. Optimal bit allocation for Gaussian sources
follows the water-filling principle: bits_i = max(0, log2(var_i / theta))
where theta is a threshold determined by the total bit budget.

We use a simpler greedy approach: sort dimensions by variance, assign
bits from a discrete set {2, 3, 4, 5, 6} to minimize total MSE given
a target average BPW.
"""
import torch
from .rotate import fwht, signs_for, next_pow2
from .codebook import lloyd_max_gaussian, quant_codebook_std


def _centroids_for_bits(bits, device):
    """Get Lloyd-Max centroids for given bit width."""
    return lloyd_max_gaussian(bits, device)


def quant_dequant_adaptive(W, target_bpw, group_size=128,
                           device="cuda", verbose=False):
    """Quantize W with per-dimension adaptive bit allocation.

    After Hadamard rotation, each dimension gets a bit-width from {2,3,4,5,6}
    such that the average BPW matches target_bpw and total MSE is minimized.

    Returns (Wq, recon_rel_err, actual_bpw, dim_bits).
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)

    # Rotate
    Wp = torch.zeros(d_out, npad, device=W.device)
    Wp[:, :d_in] = Wf
    Wr = fwht(Wp * s)
    sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
    Wrn = Wr / sc  # normalized per row

    # Per-dimension variance (across all rows)
    dim_var = Wrn.var(dim=0)  # [npad]
    # Theoretical MSE for b-bit Lloyd-Max on Gaussian: ~ 2^(-2b) * var
    # So MSE_i = 2^(-2*bits_i) * dim_var_i
    # We want to minimize sum(MSE_i) subject to sum(bits_i) = target_bpw * npad

    # Greedy bit allocation: start all at min_bits, give extra bits to highest-MSE dims
    min_bits = 2
    max_bits = 6

    total_bit_budget = int(round(target_bpw * npad))
    dim_bits = torch.full((npad,), min_bits, dtype=torch.long, device=W.device)
    remaining = total_bit_budget - min_bits * npad

    if remaining > 0:
        # Use water-filling: the optimal allocation for Gaussian sources
        # with MSE = 2^(-2b) * var is to equalize the MSE across dimensions.
        # bits_i = max(min_bits, min(max_bits, 0.5 * log2(var_i / theta)))
        # where theta is chosen so sum(bits_i) = total_bit_budget.
        # We binary search on theta.
        log_var = torch.log2(dim_var.clamp(min=1e-12))

        def alloc_for_theta(theta):
            bits = 0.5 * (log_var - torch.log2(torch.tensor(theta, device=W.device)))
            bits = bits.clamp(min_bits, max_bits)
            return bits

        # Binary search on theta
        lo, hi = 1e-12, 1e6
        for _ in range(50):
            mid = (lo * hi) ** 0.5  # geometric mean
            bits_est = alloc_for_theta(mid)
            total = bits_est.sum().item()
            if total > total_bit_budget:
                lo = mid  # increase theta to reduce bits
            else:
                hi = mid  # decrease theta to increase bits
        dim_bits = alloc_for_theta((lo * hi) ** 0.5).round().clamp(min_bits, max_bits).long()

        # Fine-tune: adjust to hit exact budget
        diff = total_bit_budget - dim_bits.sum().item()
        while diff > 0:
            mask = dim_bits < max_bits
            if not mask.any():
                break
            current_mse = (2.0 ** (-2.0 * dim_bits.float())) * dim_var
            current_mse[~mask] = -1
            best = current_mse.argmax()
            dim_bits[best] += 1
            diff -= 1
        while diff < 0:
            mask = dim_bits > min_bits
            if not mask.any():
                break
            current_mse = (2.0 ** (-2.0 * dim_bits.float())) * dim_var
            current_mse[~mask] = 1e12
            worst = current_mse.argmin()
            dim_bits[worst] -= 1
            diff += 1

    # Now quantize each dimension with its assigned bit-width
    # Precompute centroids for each bit-width
    centroids_cache = {}
    for b in set(dim_bits.tolist()):
        centroids_cache[b] = _centroids_for_bits(b, W.device)

    out_rotated = torch.zeros_like(Wrn)
    for b in set(dim_bits.tolist()):
        mask = dim_bits == b
        if mask.any():
            c = centroids_cache[b]
            out_rotated[:, mask] = quant_codebook_std(Wrn[:, mask], c)

    # Un-rotate
    out = (fwht(out_rotated * sc) * s)[:, :d_in].to(W.dtype)
    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    actual_bpw = dim_bits.float().mean().item() + 16.0 / npad  # +scale overhead

    if verbose:
        bits_dist = {}
        for b in dim_bits.tolist():
            bits_dist[b] = bits_dist.get(b, 0) + 1
        print(f"  adaptive: bits_dist={bits_dist}  bpw={actual_bpw:.3f}  err={err:.4f}",
              flush=True)

    return out, err, actual_bpw, dim_bits


def quantize_model_adaptive(model, target_bpw, skip_names=(),
                            verbose=True):
    """Apply adaptive per-dimension quantization to every nn.Linear."""
    import torch.nn as nn
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        if any(s in name for s in skip_names):
            log.append({"name": name, "shape": list(mod.weight.shape),
                        "recon_rel_err": 0.0, "bpw": 16.0, "skipped": True})
            continue
        with torch.no_grad():
            Wq, err, bpw, dim_bits = quant_dequant_adaptive(
                mod.weight.data, target_bpw, device=mod.weight.device)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "shape": list(mod.weight.shape),
                    "recon_rel_err": err, "bpw": bpw, "skipped": False})
        if verbose and (i % 40 == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name} err={err:.4f} bpw={bpw:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    if verbose:
        print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s", flush=True)
    return log
