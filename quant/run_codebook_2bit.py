"""THE definitive 2-bit codebook test.

CRITICAL FINDING: All previous 2-bit results used `e8_lattice_quantize`,
which maps to the NEAREST E8 LATTICE POINT, an unbounded codebook with
~99,000 points in the typical range. That's more than 2^16 = 65,536,
meaning our "2-bit" E8 may have been using >2 bits/weight.

The E8P codebook (e8p.py, from QuIP#) is strictly bounded to 65,536
entries (256 grid × 128 sign patterns × 2 parity shifts). It was
implemented but NEVER TESTED. This script tests it.

For true AQ at 2-bit, we need two 256-entry codebooks (1 bit each),
jointly optimized via coordinate descent. Each stage uses 8 bits per
8D block = 1 bit/weight. Total: 2 bits/weight. The 256-entry codebooks
are initialized from the E8P grid and optionally refined with Lloyd-Max.

Tests (all at TRUE 2-bit, Qwen2.5-1.5B, baseline 9.48):
1. E8 lattice (unbounded, current, might be >2 bits)     → 27.70 (known)
2. E8P codebook (bounded, exactly 65536 entries = 2 bits)  → ?
3. AQ: 2× 256-entry E8P grid codebooks, joint optimization  → ?
4. AQ + Lloyd-Max: learned 256-entry codebooks              → ?
5. E8P + Lloyd-Max: refine the full 65536-entry codebook    → ?
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import snapshot, restore
from eval import ppl_wt2
from rotate import fwht, signs_for, next_pow2
from e8lattice import e8_lattice_quantize, quantize_model_e8rvq
from e8p import e8p_quantize, _get_e8p_grid

GROUP = 128


# ============ Helper: count distinct codewords ============
def count_distinct_lattice(blocks, scale=1.0):
    """Count distinct E8 lattice points used for a set of blocks."""
    q = e8_lattice_quantize(blocks, scale=scale)
    # Round to avoid float comparison issues
    q_key = (q * 4).round().long()  # E8 points are at multiples of 0.25
    unique = torch.unique(q_key, dim=0)
    return unique.shape[0]


def count_distinct_e8p(blocks, scale=1.0):
    """Count distinct E8P codewords used for a set of blocks."""
    q = e8p_quantize(blocks, scale=scale)
    q_key = (q * 4).round().long()
    unique = torch.unique(q_key, dim=0)
    return unique.shape[0]


# ============ E8P 2-bit quantization ============
def quant_e8p_2bit(W, group_size=GROUP, chunk=4096):
    """E8P codebook at 2-bit (exactly 65536 entries, bounded)."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)
        q = e8p_quantize(blocks, scale=1.0)
        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


# ============ AQ with 256-entry codebooks (true 2-bit) ============
def quant_aq_256(W, group_size=GROUP, chunk=4096, n_aq_iter=5,
                 n_lloyd_iter=0, codebook_size=256):
    """AQ at true 2-bit: two 256-entry codebooks, jointly optimized.

    Stage 1: 256 entries (8 bits per 8D block = 1 bit/weight)
    Stage 2: 256 entries (8 bits per 8D block = 1 bit/weight)
    Total: 2 bits/weight

    Codebooks initialized from E8P grid (256 absolute-value entries).
    Optionally refined with Lloyd-Max (n_lloyd_iter > 0).

    Storage: 2 × 256 × 8 × 2 bytes = 8KB, shared globally.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    # Get E8P grid for codebook initialization (256 absolute-value entries)
    grid, _ = _get_e8p_grid(W.device)
    # Stage 1 codebook: E8P grid entries (absolute values, will be signed during quantization)
    cb1 = grid.clone()  # [256, 8]

    # Collect all blocks for codebook training
    all_blocks = []
    all_sc = []
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        all_blocks.append(Wrn.reshape(-1, d))
        all_sc.append(sc)
    blocks = torch.cat(all_blocks, dim=0)  # [N, 8]

    # Initialize stage 2 codebook from residual after stage 1
    # Assign blocks to nearest cb1 entry (with sign handling)
    def assign_signed(blocks, cb):
        """Assign blocks to nearest codebook entry with sign handling.
        For each block, try all sign patterns (with even parity) and find nearest.
        Simplified: just use brute-force over cb entries with sign = sign(block)."""
        N = blocks.shape[0]
        cb.shape[0]
        # For each block, find nearest cb entry with best sign pattern
        # Simplified: take abs(blocks), find nearest abs(cb), then apply sign
        blocks_abs = blocks.abs()
        # Score: 2 * |blocks| @ |cb|^T - |cb|^2
        cb_norm = (cb ** 2).sum(dim=1)  # [K]
        scores = 2 * blocks_abs @ cb.T - cb_norm.unsqueeze(0)  # [N, K]
        idx = scores.argmax(dim=1)
        # Apply signs
        signs = torch.sign(blocks)
        signs[signs == 0] = 1
        # Enforce even parity (even number of negative signs)
        n_neg = (signs < 0).sum(dim=1)
        needs_flip = (n_neg % 2 != 0)
        if needs_flip.any():
            abs_blocks = blocks.abs()
            abs_blocks[~needs_flip] = float('inf')
            flip_idx = abs_blocks.argmin(dim=1)
            signs[torch.arange(N, device=blocks.device), flip_idx] *= -1 * needs_flip.long()
        q = cb[idx] * signs
        return q, idx, signs

    # Initial stage 1 assignment
    q1, idx1, signs1 = assign_signed(blocks, cb1)
    residual = blocks - q1
    res_std = residual.std().clamp(min=1e-8)

    # Stage 2 codebook: E8P grid scaled to residual distribution
    cb2 = grid.clone() * res_std

    # Lloyd-Max + AQ training
    for lloyd_iter in range(n_lloyd_iter):
        # AQ coordinate descent for assignment
        q2, idx2, signs2 = assign_signed(residual, cb2 / res_std) * res_std
        for _ in range(n_aq_iter):
            # Fix q2, re-optimize q1
            target1 = blocks - q2
            q1, idx1, signs1 = assign_signed(target1, cb1)
            # Fix q1, re-optimize q2
            target2 = blocks - q1
            res_std = target2.std().clamp(min=1e-8)
            q2, idx2, signs2 = assign_signed(target2 / res_std, cb2 / res_std) * res_std

        # Update codebook entries (Lloyd-Max)
        for k in range(codebook_size):
            mask1 = (idx1 == k)
            if mask1.any():
                # Centroid of blocks assigned to this entry (minus q2 contribution)
                target = blocks[mask1] - q2[mask1]
                cb1[k] = target.abs().mean(dim=0)  # absolute value centroid
            mask2 = (idx2 == k)
            if mask2.any():
                target = (blocks[mask2] - q1[mask2]) / res_std
                cb2[k] = target.abs().mean(dim=0) * res_std

    # Final quantization with trained codebooks
    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blk = Wrn.reshape(-1, d)

        # AQ with trained codebooks
        q1, idx1, signs1 = assign_signed(blk, cb1)
        target2 = blk - q1
        rs = target2.std().clamp(min=1e-8)
        q2, idx2, signs2 = assign_signed(target2 / rs, cb2 / rs) * rs
        # One more iteration
        q1, idx1, signs1 = assign_signed(blk - q2, cb1)
        target2 = blk - q1
        rs = target2.std().clamp(min=1e-8)
        q2, idx2, signs2 = assign_signed(target2 / rs, cb2 / rs) * rs

        total_q = (q1 + q2).reshape(Wc.shape[0], npad)
        q_full = total_q * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size, cb1, cb2


# ============ E8P + Lloyd-Max refinement ============
def quant_e8p_lloyd(W, group_size=GROUP, chunk=4096, n_lloyd_iter=3):
    """E8P 2-bit + Lloyd-Max refinement of the full 65536-entry codebook.

    Initialize with E8P codebook, then refine via Lloyd-Max on actual
    weight distribution. Captures non-Gaussian structure.

    Storage: 65536 × 8 × 2 bytes = 1MB, shared globally (0.27% of 375MB model).
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    # Collect all blocks
    all_blocks = []
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        all_blocks.append(Wrn.reshape(-1, d))
    blocks = torch.cat(all_blocks, dim=0)  # [N, 8]

    # Initialize codebook: quantize all blocks with E8P, collect unique codewords
    q_init = e8p_quantize(blocks, scale=1.0)
    # Build codebook from unique quantized values
    q_key = (q_init * 4).round().long()
    unique_keys, inverse_idx = torch.unique(q_key, dim=0, return_inverse=True)
    codebook = torch.zeros(unique_keys.shape[0], d, device=W.device, dtype=torch.float32)
    # Initialize codebook entries from the unique quantized values
    for u in range(unique_keys.shape[0]):
        mask = (inverse_idx == u)
        if mask.any():
            codebook[u] = q_init[mask].mean(dim=0)
    n_cb = codebook.shape[0]
    print(f"    E8P initial codebook: {n_cb} unique codewords", flush=True)

    # Lloyd-Max refinement
    for iteration in range(n_lloyd_iter):
        # Assignment: nearest codeword (brute force)
        # Process in chunks to avoid OOM
        batch_size = 65536
        assignments = torch.empty(blocks.shape[0], dtype=torch.long, device=W.device)
        for start in range(0, blocks.shape[0], batch_size):
            end = min(start + batch_size, blocks.shape[0])
            dist = torch.cdist(blocks[start:end], codebook)
            assignments[start:end] = dist.argmin(dim=1)

        # Update: centroids
        new_codebook = codebook.clone()
        for k in range(n_cb):
            mask = (assignments == k)
            if mask.any():
                new_codebook[k] = blocks[mask].mean(dim=0)
        codebook = new_codebook
        if n_cb <= 65536:
            print(f"    Lloyd-Max iter {iteration+1}: {n_cb} codewords", flush=True)

    # Final quantization with refined codebook
    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blk = Wrn.reshape(-1, d)

        # Nearest codeword
        batch = 65536
        q = torch.empty_like(blk)
        for start in range(0, blk.shape[0], batch):
            end = min(start + batch, blk.shape[0])
            dist = torch.cdist(blk[start:end], codebook)
            idx = dist.argmin(dim=1)
            q[start:end] = codebook[idx]

        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    # Effective bits = log2(n_cb) / 8
    eff_bits = torch.log2(torch.tensor(float(n_cb))).item() / 8
    return out, err, bpw, group_size, n_cb, eff_bits


def apply_quant(model, quant_fn, **kwargs):
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            result = quant_fn(mod.weight.data, **kwargs)
            Wq = result[0]
            err = result[1]
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err})
    avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
    return log, avg_err


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16,
        device_map=dev, low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)
    snap = snapshot(model)
    base_ppl = ppl_wt2(model, tok, device=dev)
    print(f"baseline PPL: {base_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # First: measure distinct codeword counts (honesty check)
    print(f"\n=== CODEBOOK HONESTY CHECK ===", flush=True)
    # Get a representative sample of blocks
    first_linear = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            first_linear = mod
            break
    W = first_linear.weight.data
    npad = next_pow2(W.shape[1])
    s = signs_for(npad, dev)
    Wp = torch.zeros(W.shape[0], npad, device=dev)
    Wp[:, :W.shape[1]] = W.float()
    Wr = fwht(Wp * s)
    sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
    Wrn = (Wr / sc).reshape(-1, 8)
    sample = Wrn[:50000]
    n_lattice = count_distinct_lattice(sample)
    n_e8p = count_distinct_e8p(sample)
    print(f"  {first_linear.weight.data.shape[1]}→{first_linear.weight.data.shape[0]} tensor:", flush=True)
    print(f"  E8 lattice: {n_lattice} distinct codewords (2^{torch.log2(torch.tensor(float(n_lattice))).item():.1f} = {n_lattice/8:.2f} bits/weight)", flush=True)
    print(f"  E8P:        {n_e8p} distinct codewords (2^{torch.log2(torch.tensor(float(n_e8p))).item():.1f} = {n_e8p/8:.2f} bits/weight)", flush=True)
    if n_lattice > 65536:
        print(f"  WARNING: E8 lattice uses {n_lattice} > 65536 codewords! Previous '2-bit' results used >2 bits/weight!", flush=True)

    results = []

    # 1. E8 lattice 2-bit (current, unbounded)
    print(f"\n=== 1. E8 lattice 2-bit (unbounded, current) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    quantize_model_e8rvq(model, total_bits=2, group_size=GROUP, verbose=False)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "e8_lattice_2bit", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 2. E8P 2-bit (bounded, exactly 65536 entries)
    print(f"\n=== 2. E8P 2-bit (bounded, exactly 65536 entries) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log, avg_err = apply_quant(model, quant_e8p_2bit)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "e8p_2bit", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr, "avg_err": avg_err})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 3. AQ with 256-entry E8P grid codebooks (joint optimization, no Lloyd-Max)
    print(f"\n=== 3. AQ 256-entry (E8P grid init, joint opt, no Lloyd-Max) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log, avg_err = apply_quant(model, quant_aq_256, n_aq_iter=5, n_lloyd_iter=0)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "aq_256_e8p_init", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr, "avg_err": avg_err})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 4. AQ with 256-entry learned codebooks (Lloyd-Max + joint optimization)
    print(f"\n=== 4. AQ 256-entry (learned codebooks, Lloyd-Max + joint opt) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log, avg_err = apply_quant(model, quant_aq_256, n_aq_iter=5, n_lloyd_iter=5)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "aq_256_learned", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr, "avg_err": avg_err})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 5. E8P + Lloyd-Max (refine full 65536-entry codebook)
    print(f"\n=== 5. E8P + Lloyd-Max (refine full 65536-entry codebook) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    n_cb_total = 0
    eff_bits_total = 0
    log_lm = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _, n_cb, eff_bits = quant_e8p_lloyd(
                mod.weight.data, n_lloyd_iter=3)
            mod.weight.data.copy_(Wq)
            log_lm.append({"name": name, "recon_rel_err": err, "n_cb": n_cb, "eff_bits": eff_bits})
            n_cb_total += n_cb
            eff_bits_total += eff_bits
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    avg_err = sum(e["recon_rel_err"] for e in log_lm) / len(log_lm)
    avg_eff_bits = eff_bits_total / len(log_lm)
    results.append({"cfg": "e8p_lloyd_2bit", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr,
                    "avg_err": avg_err, "avg_eff_bits": avg_eff_bits})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} eff_bits={avg_eff_bits:.2f} ({time.time()-tq:.0f}s)", flush=True)

    # Summary
    print(f"\n{'='*75}", flush=True)
    print(f"{'cfg':>24s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s} {'avg_err':>8s}", flush=True)
    print(f"{'-'*75}", flush=True)
    for r in results:
        ae = r.get("avg_err", "")
        ae_str = f"{ae:.4f}" if isinstance(ae, float) else ""
        print(f"{r['cfg']:>24s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f} {ae_str:>8s}", flush=True)

    print(f"\n=== improvement over E8 lattice ({results[0]['ppl']:.3f}) ===", flush=True)
    for r in results[1:]:
        delta = results[0]["ppl"] - r["ppl"]
        pct = delta / results[0]["ppl"] * 100
        print(f"  {r['cfg']:>24s}: {delta:+.3f} PPL ({pct:+.1f}%)", flush=True)

    out_dir = "/root/novelquant/runs/codebook_2bit" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "codebook_2bit")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results,
                   "honesty_check": {"e8_lattice_codewords": n_lattice,
                                     "e8p_codewords": n_e8p}}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
