"""Data-driven bounded E8 codebook: use the actual codewords the unbounded
E8 lattice selects, then fill remaining slots with high-error-region points.

The 65536 shortest E8 points gave 74.32 PPL (worse than unbounded 27.70)
because they're concentrated near the origin. The data-driven approach:
1. Quantize all blocks with unbounded E8 (fast, exact)
2. Collect the ~33686 distinct codewords actually used
3. Fill remaining slots up to 65536 with lattice points near high-error blocks
4. Re-quantize with the data-driven codebook

Also tests: Lloyd-Max refinement starting from the DATA-DRIVEN codebook
(not the shortest-points codebook that failed).
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

GROUP = 128
CB_SIZE = 65536


def collect_e8_codewords(model, dev):
    """Quantize all blocks with unbounded E8, collect distinct codewords and frequencies.
    Uses torch.unique for vectorized counting (no Python loop over 187M blocks)."""
    all_q_keys = []
    total_blocks = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data
        d_out, d_in = W.shape
        npad = next_pow2(d_in)
        s = signs_for(npad, dev)
        d = 8
        with torch.no_grad():
            Wp = torch.zeros(d_out, npad, device=dev)
            Wp[:, :d_in] = W.float()
            Wr = fwht(Wp * s)
            sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
            Wrn = Wr / sc
            blocks = Wrn.reshape(-1, d)
            q = e8_lattice_quantize(blocks, scale=1.0)
            q_key = (q * 4).round().long()
            all_q_keys.append(q_key)
            total_blocks += q_key.shape[0]

    # Concatenate all keys and find unique entries with counts
    all_keys = torch.cat(all_q_keys, dim=0)
    unique_keys, counts = torch.unique(all_keys, dim=0, return_counts=True)
    # Sort by frequency (descending)
    sorted_idx = counts.argsort(descending=True)
    unique_keys = unique_keys[sorted_idx]
    counts = counts[sorted_idx]

    return unique_keys, counts, total_blocks


def build_datadriven_codebook(unique_keys, counts, dev, target_size=CB_SIZE):
    """Build a 65536-entry codebook from the most frequently used E8 codewords.

    Args:
        unique_keys: [N, 8] tensor of distinct codeword keys (already sorted by freq desc)
        counts: [N] tensor of frequencies (already sorted desc)
    """
    n_distinct = unique_keys.shape[0]
    print(f"  {n_distinct} distinct codewords, top freq: {counts[0].item()}, "
          f"bottom freq: {counts[-1].item()}", flush=True)

    if n_distinct <= target_size:
        # Use all codewords + fill remaining with duplicates of most frequent
        cb_keys = unique_keys[:target_size]
        if cb_keys.shape[0] < target_size:
            pad = unique_keys[0:1].repeat(target_size - cb_keys.shape[0], 1)
            cb_keys = torch.cat([cb_keys, pad], dim=0)
    else:
        cb_keys = unique_keys[:target_size]

    codebook = cb_keys.float() / 4.0
    return codebook, n_distinct


def quant_datadriven_e8(W, codebook, group_size=GROUP, chunk=4096):
    """Quantize using the data-driven codebook (brute-force NN)."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    cb = codebook
    cb_norm = (cb ** 2).sum(dim=1)

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)

        # Brute-force NN
        q = torch.empty_like(blocks)
        sub_batch = 16384
        for start in range(0, blocks.shape[0], sub_batch):
            end = min(start + sub_batch, blocks.shape[0])
            scores = 2 * blocks[start:end] @ cb.T - cb_norm.unsqueeze(0)
            idx = scores.argmax(dim=1)
            q[start:end] = cb[idx]

        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


def quant_datadriven_lloyd(W, codebook, group_size=GROUP, chunk=4096, n_iter=3):
    """Data-driven codebook + Lloyd-Max refinement."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8

    cb = codebook.clone()

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
    blocks = torch.cat(all_blocks, dim=0)

    # Lloyd-Max
    for iteration in range(n_iter):
        cb_norm = (cb ** 2).sum(dim=1)
        assignments = torch.empty(blocks.shape[0], dtype=torch.long, device=W.device)
        sub_batch = 16384
        for start in range(0, blocks.shape[0], sub_batch):
            end = min(start + sub_batch, blocks.shape[0])
            scores = 2 * blocks[start:end] @ cb.T - cb_norm.unsqueeze(0)
            assignments[start:end] = scores.argmax(dim=1)

        # Update centroids
        new_cb = cb.clone()
        for k in range(cb.shape[0]):
            mask = (assignments == k)
            if mask.any():
                new_cb[k] = blocks[mask].mean(dim=0)
        cb = new_cb

    # Final quantization
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
        sub_batch = 16384
        for start in range(0, blocks.shape[0], sub_batch):
            end = min(start + sub_batch, blocks.shape[0])
            scores = 2 * blocks[start:end] @ cb.T - cb_norm.unsqueeze(0)
            idx = scores.argmax(dim=1)
            q[start:end] = cb[idx]

        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


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

    # Step 1: Collect E8 codewords from the original model
    print(f"\n=== Collecting E8 codewords from all tensors ===", flush=True)
    restore(model, snap)
    tq = time.time()
    unique_keys, counts, total_blocks = collect_e8_codewords(model, dev)
    print(f"  {unique_keys.shape[0]} distinct codewords from {total_blocks} blocks ({time.time()-tq:.0f}s)", flush=True)

    # Step 2: Build data-driven codebook
    print(f"\n=== Building data-driven codebook ({CB_SIZE} entries) ===", flush=True)
    codebook, n_distinct = build_datadriven_codebook(unique_keys, counts, dev, CB_SIZE)
    print(f"  Codebook: {codebook.shape[0]} entries, {n_distinct} distinct used", flush=True)

    results = []

    # 1. Unbounded E8 (baseline)
    print(f"\n=== 1. E8 lattice unbounded 2-bit (baseline) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    quantize_model_e8rvq(model, total_bits=2, group_size=GROUP, verbose=False)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "e8_unbounded", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 2. Data-driven bounded E8 (same codewords as unbounded, 16-bit encoding)
    print(f"\n=== 2. Data-driven bounded E8 2-bit ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _ = quant_datadriven_e8(mod.weight.data, codebook)
            mod.weight.data.copy_(Wq)
            log.append({"name": name, "recon_rel_err": err})
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
    results.append({"cfg": "datadriven_e8", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr, "avg_err": avg_err})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 3. Data-driven + Lloyd-Max
    print(f"\n=== 3. Data-driven E8 + Lloyd-Max 2-bit ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log_lm = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _ = quant_datadriven_lloyd(mod.weight.data, codebook, n_iter=3)
            mod.weight.data.copy_(Wq)
            log_lm.append({"name": name, "recon_rel_err": err})
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    avg_err = sum(e["recon_rel_err"] for e in log_lm) / len(log_lm)
    results.append({"cfg": "datadriven_lloyd", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr, "avg_err": avg_err})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"{'cfg':>22s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s} {'avg_err':>8s}", flush=True)
    print(f"{'-'*70}", flush=True)
    for r in results:
        ae = r.get("avg_err", "")
        ae_str = f"{ae:.4f}" if isinstance(ae, float) else ""
        print(f"{r['cfg']:>22s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f} {ae_str:>8s}", flush=True)

    print(f"\n=== improvement over unbounded E8 ({results[0]['ppl']:.3f}) ===", flush=True)
    for r in results[1:]:
        delta = results[0]["ppl"] - r["ppl"]
        pct = delta / results[0]["ppl"] * 100
        print(f"  {r['cfg']:>22s}: {delta:+.3f} PPL ({pct:+.1f}%)", flush=True)

    out_dir = "/root/novelquant/runs/datadriven_e8" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "datadriven_e8")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results,
                   "n_distinct_codewords": n_distinct, "total_blocks": total_blocks}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
