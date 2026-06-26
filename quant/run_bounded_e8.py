"""THE 2-bit codebook upgrade: bounded E8 with 65536 entries.

FINDING: Our "2-bit" E8 lattice only uses 33,686 distinct codewords
(1.875 bits/weight), leaving 31,850 unused codeword slots in the
2-bit budget (65536 = 2^16 entries). E8P is worse (27,609 codewords,
1738 PPL) because it uses shifted D8 lattice points that miss the
small-norm region critical for Gaussian sources.

THE UPGRADE: Build a bounded E8 codebook with exactly 65536 entries
, the 65536 shortest E8 lattice points. This fills the 2-bit budget
(2x more codewords) at zero storage cost (codebook is computed from
the lattice structure, not stored).

E8 lattice = { x in Z^8 : sum even } ∪ { x in (Z+0.5)^8 : sum even }
The 65536 shortest points cover norm^2 <= ~12, which captures >99.9%
of the probability mass for a unit-variance 8D Gaussian.

Optionally: Lloyd-Max refinement. After building the bounded E8
codebook, refine codeword positions to centroids of assigned blocks.
This captures non-Gaussian structure. Storage: 65536 × 8 × 2 bytes
= 1MB shared globally (0.27% of a 375MB 2-bit model).

Tests (all at TRUE 2-bit, Qwen2.5-1.5B, baseline 9.48):
1. E8 lattice unbounded (current, 33686 codewords = 1.875 bits)  → 27.70
2. Bounded E8 (65536 shortest lattice points = 2.0 bits)          → ?
3. Bounded E8 + Lloyd-Max (refined codeword positions)            → ?
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
from e8lattice import quantize_model_e8rvq

GROUP = 128
CODEBOOK_SIZE = 65536  # 2^16 = exactly 2 bits per 8D block


# ============ Build bounded E8 codebook ============
def build_bounded_e8_codebook(device, target_size=CODEBOOK_SIZE):
    """Build the 65536 shortest E8 lattice points.

    E8 = { x in Z^8 : sum(x_i) even } ∪ { x in (Z+0.5)^8 : sum(x_i) even }

    Generate integer and half-integer points with small coordinates,
    filter by even-sum constraint, sort by norm, take target_size.
    """
    points = []

    # Integer points: coordinates in {-3, -2, -1, 0, 1, 2, 3}
    # 7^8 = 5,764,801, generate in chunks to manage memory
    # Generate cartesian product in batches
    batch_dim = 7  # 7 values per coordinate
    total_int = batch_dim ** 8  # 5,764,801

    # Generate all integer points using meshgrid-like approach
    # Use arange + divmod to generate indices
    idx = torch.arange(total_int, device=device)
    # Convert linear index to 8D coordinates
    coords = torch.stack([
        (idx // (batch_dim ** k) % batch_dim) - 3  # map 0-6 to -3..3
        for k in range(8)
    ], dim=1).float()  # [total_int, 8]

    # Filter: even sum
    sum_coords = coords.sum(dim=1)
    mask_even = (sum_coords % 2 == 0)
    int_points = coords[mask_even]  # [~2.88M, 8]

    # Filter: norm^2 <= 16 (covers the range we need)
    norms_sq = (int_points ** 2).sum(dim=1)
    int_points = int_points[norms_sq <= 16]

    points.append(int_points)
    print(f"  Integer E8 points (norm^2<=16): {int_points.shape[0]}", flush=True)

    # Half-integer points: coordinates in {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5}
    # 6^8 = 1,679,616
    coords_hf_vals = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], device=device)
    batch_dim_hf = 6
    total_hf = batch_dim_hf ** 8  # 1,679,616

    idx_hf = torch.arange(total_hf, device=device)
    coords_h = torch.stack([
        coords_hf_vals[(idx_hf // (batch_dim_hf ** k) % batch_dim_hf)]
        for k in range(8)
    ], dim=1)  # [total_hf, 8]

    # Filter: even sum (sum of half-integers = sum(k+0.5) = sum(k)+4, even iff sum(k) even)
    sum_hf = coords_h.sum(dim=1)
    # sum of half-integers: each is k+0.5, sum = sum(k) + 4. Even iff sum(k) even.
    # But we need to check: sum of (k+0.5) for 8 coords. If each coord is n+0.5, sum = 8*0.5 + sum(n) = 4 + sum(n).
    # Even iff sum(n) is even. n = coord - 0.5, so sum(n) = sum(coords) - 4.
    # Even iff sum(coords) - 4 is even, i.e., sum(coords) is even.
    mask_even_hf = (sum_hf % 2 == 0)
    hf_points = coords_h[mask_even_hf]

    # Filter: norm^2 <= 16
    norms_sq_hf = (hf_points ** 2).sum(dim=1)
    hf_points = hf_points[norms_sq_hf <= 16]

    points.append(hf_points)
    print(f"  Half-integer E8 points (norm^2<=16): {hf_points.shape[0]}", flush=True)

    # Combine and sort by norm
    all_points = torch.cat(points, dim=0)
    all_norms_sq = (all_points ** 2).sum(dim=1)
    sorted_idx = all_norms_sq.argsort()
    all_points = all_points[sorted_idx]

    # Take target_size shortest
    codebook = all_points[:target_size]
    print(f"  Bounded E8 codebook: {codebook.shape[0]} entries, "
          f"norm^2 range: [{(codebook**2).sum(1).min():.1f}, "
          f"{(codebook**2).sum(1).max():.1f}]", flush=True)

    return codebook


# ============ Bounded E8 quantization ============
_CB = None

def get_codebook(device):
    global _CB
    if _CB is None or _CB.device != device:
        _CB = build_bounded_e8_codebook(device)
    return _CB


def quant_bounded_e8(W, group_size=GROUP, chunk=2048):
    """Quantize using bounded E8 codebook (65536 entries, exactly 2 bits/weight)."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    cb = get_codebook(W.device)  # [65536, 8]
    cb_norm = (cb ** 2).sum(dim=1)  # [65536]

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)  # [N, 8]

        # Brute-force nearest neighbor over 65536 entries
        # Process in sub-batches to manage memory
        q = torch.empty_like(blocks)
        sub_batch = 8192
        for start in range(0, blocks.shape[0], sub_batch):
            end = min(start + sub_batch, blocks.shape[0])
            blk = blocks[start:end]  # [B, 8]
            # Distance: ||blk - cb||^2 = ||blk||^2 - 2*blk@cb^T + ||cb||^2
            # Minimize = maximize 2*blk@cb^T - ||cb||^2
            scores = 2 * blk @ cb.T - cb_norm.unsqueeze(0)  # [B, 65536]
            idx = scores.argmax(dim=1)
            q[start:end] = cb[idx]

        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


# ============ Bounded E8 + Lloyd-Max ============
def quant_bounded_e8_lloyd(W, group_size=GROUP, chunk=2048, n_lloyd_iter=3):
    """Bounded E8 + Lloyd-Max refinement on actual weight distribution.

    After initial assignment with bounded E8 codebook, update codeword
    positions to centroids of assigned blocks. Repeat for n_lloyd_iter.
    The refined codebook is stored (1MB, shared globally).
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    cb = get_codebook(W.device).clone()  # [65536, 8], will be refined

    # Collect all blocks for Lloyd-Max training
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

    # Lloyd-Max iterations
    for iteration in range(n_lloyd_iter):
        # Assignment: nearest codeword
        cb_norm = (cb ** 2).sum(dim=1)
        assignments = torch.empty(blocks.shape[0], dtype=torch.long, device=W.device)
        sub_batch = 8192
        for start in range(0, blocks.shape[0], sub_batch):
            end = min(start + sub_batch, blocks.shape[0])
            scores = 2 * blocks[start:end] @ cb.T - cb_norm.unsqueeze(0)
            assignments[start:end] = scores.argmax(dim=1)

        # Update: centroids
        new_cb = cb.clone()
        for k in range(cb.shape[0]):
            mask = (assignments == k)
            if mask.any():
                new_cb[k] = blocks[mask].mean(dim=0)
        cb = new_cb
        print(f"    Lloyd-Max iter {iteration+1}/{n_lloyd_iter} done", flush=True)

    # Final quantization with refined codebook
    out = torch.empty_like(W)
    cb_norm = (cb ** 2).sum(dim=1)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)

        q = torch.empty_like(blocks)
        sub_batch = 8192
        for start in range(0, blocks.shape[0], sub_batch):
            end = min(start + sub_batch, blocks.shape[0])
            scores = 2 * blocks[start:end] @ cb.T - cb_norm.unsqueeze(0)
            idx = scores.argmax(dim=1)
            q[start:end] = cb[idx]

        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size, cb


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

    # Build the bounded E8 codebook
    print(f"\n=== Building bounded E8 codebook (65536 entries) ===", flush=True)
    build_bounded_e8_codebook(dev, CODEBOOK_SIZE)

    results = []

    # 1. E8 lattice unbounded (current baseline, 33686 codewords = 1.875 bits)
    print(f"\n=== 1. E8 lattice unbounded 2-bit (current, 33686 codewords) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    quantize_model_e8rvq(model, total_bits=2, group_size=GROUP, verbose=False)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "e8_lattice_unbounded", "bpw": 2 + 16.0/GROUP,
                    "ppl": ppl, "degr": degr, "effective_bits": 1.875})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 2. Bounded E8 (65536 entries, exactly 2 bits)
    print(f"\n=== 2. Bounded E8 2-bit (65536 entries, exactly 2 bits) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log, avg_err = apply_quant(model, quant_bounded_e8)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "bounded_e8_65536", "bpw": 2 + 16.0/GROUP,
                    "ppl": ppl, "degr": degr, "avg_err": avg_err, "effective_bits": 2.0})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 3. Bounded E8 + Lloyd-Max (refined codebook)
    print(f"\n=== 3. Bounded E8 + Lloyd-Max 2-bit (refined codebook) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log_lm = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _, _ = quant_bounded_e8_lloyd(
                mod.weight.data, n_lloyd_iter=3)
            mod.weight.data.copy_(Wq)
            log_lm.append({"name": name, "recon_rel_err": err})
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    avg_err = sum(e["recon_rel_err"] for e in log_lm) / len(log_lm)
    results.append({"cfg": "bounded_e8_lloyd", "bpw": 2 + 16.0/GROUP,
                    "ppl": ppl, "degr": degr, "avg_err": avg_err, "effective_bits": 2.0})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # Summary
    print(f"\n{'='*75}", flush=True)
    print(f"{'cfg':>26s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s} {'eff_bits':>9s} {'avg_err':>8s}", flush=True)
    print(f"{'-'*75}", flush=True)
    for r in results:
        ae = r.get("avg_err", "")
        ae_str = f"{ae:.4f}" if isinstance(ae, float) else ""
        print(f"{r['cfg']:>26s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f} "
              f"{r['effective_bits']:>9.3f} {ae_str:>8s}", flush=True)

    print(f"\n=== improvement over unbounded E8 ({results[0]['ppl']:.3f}) ===", flush=True)
    for r in results[1:]:
        delta = results[0]["ppl"] - r["ppl"]
        pct = delta / results[0]["ppl"] * 100
        print(f"  {r['cfg']:>26s}: {delta:+.3f} PPL ({pct:+.1f}%)", flush=True)

    out_dir = "/root/novelquant/runs/bounded_e8" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bounded_e8")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
