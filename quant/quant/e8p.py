"""E8P codebook: proper implementation following QuIP#.

The E8P codebook contains 2^16 entries in 8D, built from the E8 lattice.
Grid construction (from QuIP# source):
1. Generate all D8 lattice points: cartesian_prod of integers, +1/2
2. Filter: even sum, norm^2 <= 10
3. Take absolute values, unique entries -> 227 entries
4. Add 29 padding entries with norm^2 = 12 -> 256 total
5. Each grid entry is a specific 8D vector (NOT sorted)

Quantization:
1. Try X+1/4 and X-1/4 (two parity shifts)
2. For each: take abs(X), find nearest grid entry via dot product
3. Apply signs with parity constraint
4. Pick whichever shift gives lower error
"""
import torch
from quant.rotate import fwht, signs_for, next_pow2


def _build_e8p_grid(device):
    """Build the 256-entry absolute-value grid for E8P.

    Following QuIP# exactly: cartesian product of half-integers,
    filtered by D8 parity and norm, unique absolute values.
    """
    # Generate all D8 lattice points: (Z+1/2)^8 with even sum
    # cartesian_prod of [0,1,2,3] gives all integer combinations
    intr = torch.tensor([0, 1, 2, 3], device=device)
    d8 = torch.cartesian_prod(intr, intr, intr, intr,
                              intr, intr, intr, intr).float() + 0.5  # [65536, 8]

    # Filter: sum must be even (D8_hat parity)
    # sum of (k+0.5) for 8 coords = sum(k) + 4, so sum is even iff sum(k) is even
    d8_sum = d8.sum(dim=-1)
    mask_parity = (d8_sum % 2 == 0)

    # Filter: norm^2 <= 10
    mask_norm = (d8.norm(dim=-1) ** 2 <= 10)

    # Apply both filters
    d8_filtered = d8[mask_parity & mask_norm]

    # Take absolute values and unique entries
    d8abs = torch.unique(d8_filtered.abs(), dim=0)

    # Should have 227 entries
    d8abs.shape[0]

    # Add 29 padding entries with norm^2 = 12
    # Find D8 points with norm^2 = 12
    mask_norm12 = (d8.norm(dim=-1) ** 2 - 12).abs() < 0.01
    d8_norm12 = d8[mask_parity & mask_norm12]
    d8abs12 = torch.unique(d8_norm12.abs(), dim=0)

    # Sort by norm and take 29
    norms12 = d8abs12.norm(dim=-1)
    sorted_idx = norms12.argsort()
    d8abs12 = d8abs12[sorted_idx]
    padding = d8abs12[:29]

    # Combine: 227 + 29 = 256
    grid = torch.cat([d8abs, padding], dim=0)

    # Ensure exactly 256
    if grid.shape[0] < 256:
        # Pad with last entry
        grid = torch.cat([grid, grid[-1:].repeat(256 - grid.shape[0], 1)], dim=0)
    grid = grid[:256]

    return grid


_E8P_GRID = None
_E8P_GRID_NORM = None


def _get_e8p_grid(device):
    global _E8P_GRID, _E8P_GRID_NORM
    if _E8P_GRID is None or _E8P_GRID.device != device:
        _E8P_GRID = _build_e8p_grid(device)
        _E8P_GRID_NORM = (_E8P_GRID ** 2).sum(dim=1)
    return _E8P_GRID, _E8P_GRID_NORM


def _fast_quantize_part(X, grid, grid_norm):
    """Quantize X to D8_hat grid.

    X is already shifted (X+1/4 or X-1/4).
    Returns (quantized_values, error).
    """
    N = X.shape[0]

    # Take absolute values
    X_abs = X.abs()

    # Determine sign pattern
    X_signs = torch.sign(X)
    X_signs[X_signs == 0] = 1

    # Parity constraint: number of negative signs must be even
    # (for D8_hat: sum of signed half-integers must be even)
    n_neg = (X_signs < 0).sum(dim=1)
    needs_flip = (n_neg % 2 != 0)

    if needs_flip.any():
        # Flip the sign of the coordinate with smallest |X|
        X_abs_for_flip = X_abs.clone()
        X_abs_for_flip[~needs_flip] = float('inf')
        flip_idx = X_abs_for_flip.argmin(dim=1)
        X_signs[torch.arange(N, device=X.device), flip_idx] *= -1 * needs_flip.long()

    # Nearest neighbor: maximize 2*X_abs@g^T - ||g||^2
    scores = 2 * X_abs @ grid.T - grid_norm.unsqueeze(0)
    grid_idx = scores.argmax(dim=1)

    # Reconstruct
    grid_vals = grid[grid_idx]
    vals = grid_vals * X_signs

    err = (X - vals).norm(dim=1)
    return vals, err


def e8p_quantize(X, scale=1.0):
    """Quantize 8D vectors using E8P codebook.

    E8P = (D8_hat - 1/4) ∪ (D8_hat + 1/4)
    """
    X = X / scale
    orig_shape = X.shape
    X_flat = X.reshape(-1, 8)
    X_flat.shape[0]

    grid, grid_norm = _get_e8p_grid(X.device)

    # Try X + 1/4 (quantize to D8_hat, result is vals - 1/4)
    X_plus = X_flat + 0.25
    vals_plus, err_plus = _fast_quantize_part(X_plus, grid, grid_norm)
    vals_plus = vals_plus - 0.25

    # Try X - 1/4 (quantize to D8_hat, result is vals + 1/4)
    X_minus = X_flat - 0.25
    vals_minus, err_minus = _fast_quantize_part(X_minus, grid, grid_norm)
    vals_minus = vals_minus + 0.25

    # Pick better
    use_plus = err_plus <= err_minus
    final_vals = torch.where(use_plus.unsqueeze(-1), vals_plus, vals_minus)

    return (final_vals * scale).reshape(orig_shape)


def quant_dequant_e8p_rvq(W, total_bits=4, group_size=128, chunk=4096):
    """Quantize W using E8P codebook with RVQ after Hadamard rotation."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    n_stages = total_bits // 2
    assert n_stages * 2 == total_bits

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)

        # RVQ
        residual = blocks.clone()
        for stage in range(n_stages):
            if stage == 0:
                q = e8p_quantize(residual, scale=1.0)
            else:
                res_std = residual.std().clamp(min=1e-8)
                q = e8p_quantize(residual, scale=res_std)
            residual = residual - q
            if stage == 0:
                total_q = q
            else:
                total_q = total_q + q

        q_full = total_q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = total_bits + 16.0 / group_size
    return out, err, bpw, group_size


def quantize_model_e8p(model, total_bits=4, group_size=128, verbose=True):
    """Apply E8P RVQ quantization to every nn.Linear."""
    import torch.nn as nn
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant_e8p_rvq(
                mod.weight.data, total_bits, group_size)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
        if verbose and (i % 40 == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name} err={err:.4f} bpw={bpw:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    if verbose:
        avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
        print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
              f"(E8P RVQ {total_bits}bit, avg_err={avg_err:.4f})", flush=True)
    return log
