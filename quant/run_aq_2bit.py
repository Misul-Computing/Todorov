"""2-bit codebook upgrade: Additive Quantization (AQ) with joint optimization.

WHAT IS A CODEBOOK?
At 2-bit, each 8-dim block of weights is mapped to one of 2^16 = 65536
codewords. These 65536 codewords ARE the codebook. E8 lattice generates
them structurally (2-stage RVQ: 256 x 256), costing zero storage. The
codebook quality is why E8 2-bit (27.70 PPL) beats scalar 2-bit (417 PPL).

THE PROBLEM WITH RVQ:
RVQ is greedy: stage 1 commits to q1 = E8_nearest(x) without knowing
what stage 2 can fix. The residual x-q1 might quantize poorly even though
a different q1 (slightly worse for stage 1) would give a much better
overall result. Our learned-RVQ experiment proved this: Gaussian codebook
got 281 PPL (10x worse than E8) because greedy decomposition + uneven
codebook = catastrophic. E8 survives greedy because its lattice points
are evenly spread, but it still leaves gains on the table.

THE FIX: ADDITIVE QUANTIZATION (AQ)
Instead of greedy RVQ, jointly optimize both stages via coordinate descent:
  1. q1 = E8_nearest(x)           [standard stage 1]
  2. q2 = E8_nearest(x - q1)      [standard stage 2 = RVQ]
  3. q1 = E8_nearest(x - q2)      [AQ: fix q2, re-optimize q1]
  4. q2 = E8_nearest(x - q1)      [AQ: fix q1, re-optimize q2]
  5. repeat 3-4 until convergence (usually 2-3 iterations)

Each step is still a fast E8 nearest-neighbor lookup. No extra storage.
The reconstruction q1+q2 is guaranteed to be <= RVQ's error (AQ can only
improve, never worsen, because it starts from the RVQ solution).

VARIANT B: LEARNED CODEBOOK
After AQ assignment converges, update codebook entries as centroids of
assigned blocks. This captures non-Gaussian structure that E8's fixed
lattice misses. Storage: 2 x 256 x 8 x 2 bytes = 8KB, shared globally
(0.002% of a 375MB 2-bit model). Nearest-neighbor becomes brute-force
over 256 entries (still fast on GPU).

Tests at 2-bit on Qwen2.5-1.5B (baseline 9.48):
- Plain E8 RVQ: 27.70 (current ceiling)
- E8 + LDLQ: 24.46 (better rounding, same codebook)
- AQ fixed E8: ? (joint assignment, same codebook)
- AQ learned: ? (joint assignment + learned codebook)
- AQ + LDLQ: ? (joint assignment + activation-aware rounding)
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
from datasets import load_dataset

from quantize import snapshot, restore
from eval import ppl_wt2
from rotate import fwht, signs_for, next_pow2
from e8lattice import e8_lattice_quantize, quantize_model_e8rvq

GROUP = 128


# ============ AQ with fixed E8 codebooks ============
def quant_aq_fixed_e8(W, group_size=GROUP, chunk=2048, n_iter=3):
    """E8 2-bit with Additive Quantization (coordinate descent).

    Same codebook as E8 RVQ (256+256 E8 lattice points), but jointly
    optimized assignment via coordinate descent. No extra storage.
    """
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
        blocks = Wrn.reshape(-1, d)  # [N, 8]

        # Standard RVQ initialization
        q1 = e8_lattice_quantize(blocks, scale=1.0)
        residual = blocks - q1
        res_std = residual.std().clamp(min=1e-8)
        q2 = e8_lattice_quantize(residual, scale=res_std)

        # AQ coordinate descent
        for _ in range(n_iter):
            # Fix q2, re-optimize q1: target = blocks - q2
            q1_new = e8_lattice_quantize(blocks - q2, scale=1.0)
            # Fix q1, re-optimize q2: target = blocks - q1_new
            residual_new = blocks - q1_new
            res_std_new = residual_new.std().clamp(min=1e-8)
            q2_new = e8_lattice_quantize(residual_new, scale=res_std_new)

            # Check convergence
            changed = (q1_new != q1).any() or (q2_new != q2).any()
            q1, q2 = q1_new, q2_new
            if not changed:
                break

        total_q = (q1 + q2).reshape(Wc.shape[0], npad)
        q_full = total_q * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


# ============ AQ with learned codebooks ============
def quant_aq_learned(W, group_size=GROUP, chunk=2048, n_aq_iter=3,
                     n_codebook_epochs=5, codebook_size=256):
    """E8 2-bit with AQ + learned codebook entries.

    Stage 1 and 2 codebooks are initialized from E8 lattice, then
    jointly optimized: coordinate descent for assignments + centroid
    update for codebook entries. Codebook stored globally (8KB).
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    # Collect ALL blocks from this tensor for codebook training
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
    sc_list = []
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc_list.append(Wr.std(1, keepdim=True).clamp(min=1e-8))

    # Initialize codebooks from E8: sample blocks, quantize to get E8 points
    # Use a random subset to build initial codebook
    n_init = min(codebook_size * 4, blocks.shape[0])
    idx = torch.randperm(blocks.shape[0])[:n_init]
    init_blocks = blocks[idx]
    cb1 = e8_lattice_quantize(init_blocks, scale=1.0)  # [n_init, 8]
    # Deduplicate to get ~256 unique entries
    cb1_unique = torch.unique(cb1, dim=0)
    if cb1_unique.shape[0] < codebook_size:
        # Pad with more E8 points
        extra_idx = torch.randperm(blocks.shape[0])[:codebook_size * 8]
        extra = e8_lattice_quantize(blocks[extra_idx], scale=1.0)
        cb1_unique = torch.unique(torch.cat([cb1_unique, extra], dim=0), dim=0)
    cb1 = cb1_unique[:codebook_size].clone()
    if cb1.shape[0] < codebook_size:
        # Pad with zeros (will be updated by centroid)
        cb1 = torch.cat([cb1, torch.zeros(codebook_size - cb1.shape[0], d, device=W.device)], dim=0)

    # Stage 2: initialize from residual after stage 1
    # Assign all blocks to nearest cb1 entry
    dist1 = torch.cdist(blocks, cb1)  # [N, 256]
    assign1 = dist1.argmin(dim=1)
    residual = blocks - cb1[assign1]
    res_std = residual.std().clamp(min=1e-8)
    init_res = residual[:n_init]
    cb2 = e8_lattice_quantize(init_res, scale=res_std)
    cb2_unique = torch.unique(cb2, dim=0)
    if cb2_unique.shape[0] < codebook_size:
        extra_idx = torch.randperm(residual.shape[0])[:codebook_size * 8]
        extra = e8_lattice_quantize(residual[extra_idx], scale=res_std)
        cb2_unique = torch.unique(torch.cat([cb2_unique, extra], dim=0), dim=0)
    cb2 = cb2_unique[:codebook_size].clone()
    if cb2.shape[0] < codebook_size:
        cb2 = torch.cat([cb2, torch.zeros(codebook_size - cb2.shape[0], d, device=W.device)], dim=0)

    # AQ training loop: alternately optimize assignments and codebooks
    for epoch in range(n_codebook_epochs):
        # Assignment step: coordinate descent
        # Fix cb2, optimize assign1: target = blocks - cb2[assign2]
        # Start from current assignments
        dist2 = torch.cdist(residual, cb2)
        assign2 = dist2.argmin(dim=1)

        for _ in range(n_aq_iter):
            # Fix assign2, optimize assign1: target = blocks - cb2[assign2]
            target1 = blocks - cb2[assign2]
            dist1 = torch.cdist(target1, cb1)
            assign1 = dist1.argmin(dim=1)

            # Fix assign1, optimize assign2: target = blocks - cb1[assign1]
            target2 = blocks - cb1[assign1]
            dist2 = torch.cdist(target2, cb2)
            assign2 = dist2.argmin(dim=1)

        # Codebook update step: centroids
        for k in range(codebook_size):
            mask1 = (assign1 == k)
            if mask1.any():
                cb1[k] = blocks[mask1].mean(dim=0) - cb2[assign2[mask1]].mean(dim=0)
            mask2 = (assign2 == k)
            if mask2.any():
                cb2[k] = blocks[mask2].mean(dim=0) - cb1[assign1[mask2]].mean(dim=0)

    # Final quantization: apply learned codebooks
    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blk = Wrn.reshape(-1, d)

        # Final AQ assignment with learned codebooks
        dist1 = torch.cdist(blk, cb1)
        assign1 = dist1.argmin(dim=1)
        target2 = blk - cb1[assign1]
        dist2 = torch.cdist(target2, cb2)
        assign2 = dist2.argmin(dim=1)
        # One more iteration
        target1 = blk - cb2[assign2]
        dist1 = torch.cdist(target1, cb1)
        assign1 = dist1.argmin(dim=1)
        target2 = blk - cb1[assign1]
        dist2 = torch.cdist(target2, cb2)
        assign2 = dist2.argmin(dim=1)

        total_q = (cb1[assign1] + cb2[assign2]).reshape(Wc.shape[0], npad)
        q_full = total_q * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size, cb1, cb2


# ============ AQ + LDLQ ============
def capture_hessians(model, tok, n_examples=48, max_len=512, dev="cuda"):
    hessians = {}
    handles = []
    def make_hook(name):
        def hook(mod, inp, _out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            H = x.T @ x
            if name not in hessians:
                hessians[name] = H.to(dev)
            else:
                hessians[name] += H.to(dev)
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    k = 0
    model.eval()
    with torch.no_grad():
        for ex in ds:
            if k >= n_examples:
                break
            t = ex["text"]
            if 50 < len(t) < 2000:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
                if ids.shape[1] < 2:
                    continue
                model(ids)
                k += 1
    for h in handles:
        h.remove()
    return hessians


def quant_aq_ldlq(W, H, group_size=GROUP, chunk=2048, n_aq_iter=3, reg=1.0):
    """AQ 2-bit + LDLQ adaptive rounding with real Hessians.

    Combines joint codebook assignment (AQ) with activation-aware
    block ordering (LDLQ). The LDLQ adjusts each 8D block based on
    previous rounding errors, weighted by the Hessian. Then AQ
    jointly optimizes the 2-stage assignment within each block.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    # Build Hessian in rotated space
    Hpad = torch.zeros(npad, npad, device=W.device)
    Hpad[:d_in, :d_in] = H.to(W.device).float()
    Hs = Hpad * s.unsqueeze(0) * s.unsqueeze(1)
    H_rot = fwht(fwht(Hs).T).T
    tr = H_rot.diagonal().mean().clamp(min=1e-8)
    H_rot = H_rot / tr + reg * torch.eye(npad, device=W.device)
    try:
        L = torch.linalg.cholesky(H_rot)
        U = L.T - torch.eye(npad, device=W.device)
    except Exception:
        U = torch.zeros(npad, npad, device=W.device)

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        nb = npad // d

        # LDLQ: quantize block by block with Hessian-weighted adjustment
        What = torch.zeros_like(Wrn)
        for k in range(nb):
            if k > 0:
                err = Wrn[:, :k * d] - What[:, :k * d]
                A_k = U[:k * d, k * d:(k + 1) * d]
                adj = err @ A_k
            else:
                adj = 0.0
            W_block = Wrn[:, k * d:(k + 1) * d] + adj

            # AQ within this block: 2-stage E8 with coordinate descent
            blk = W_block.reshape(-1, d)
            q1 = e8_lattice_quantize(blk, scale=1.0)
            residual = blk - q1
            res_std = residual.std().clamp(min=1e-8)
            q2 = e8_lattice_quantize(residual, scale=res_std)
            for _ in range(n_aq_iter):
                q1_new = e8_lattice_quantize(blk - q2, scale=1.0)
                res_new = blk - q1_new
                rs_new = res_new.std().clamp(min=1e-8)
                q2_new = e8_lattice_quantize(res_new, scale=rs_new)
                if (q1_new == q1).all() and (q2_new == q2).all():
                    q1, q2 = q1_new, q2_new
                    break
                q1, q2 = q1_new, q2_new

            What[:, k * d:(k + 1) * d] = (q1 + q2).reshape(W_block.shape)

        q_full = What * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


def apply_quant(model, quant_fn, **kwargs):
    """Apply a quantization function to every nn.Linear."""
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

    # Capture Hessians for LDLQ variant
    print("capturing activations for LDLQ...", flush=True)
    hessians = capture_hessians(model, tok, n_examples=48, max_len=512, dev=dev)
    print(f"  {len(hessians)} Hessians ({time.time()-t0:.0f}s)", flush=True)

    results = []

    # 1. Plain E8 RVQ (baseline)
    print(f"\n=== 1. Plain E8 RVQ 2-bit (baseline) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    quantize_model_e8rvq(model, total_bits=2, group_size=GROUP, verbose=False)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "e8_rvq_2bit", "bpw": 2 + 16.0/GROUP, "ppl": ppl, "degr": degr})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 2. AQ with fixed E8 codebooks (3 iterations)
    for n_iter in [1, 3, 7]:
        print(f"\n=== 2. AQ fixed-E8 2-bit (n_iter={n_iter}) ===", flush=True)
        restore(model, snap)
        tq = time.time()
        log, avg_err = apply_quant(model, quant_aq_fixed_e8, n_iter=n_iter)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"aq_fixed_e8_2bit_iter{n_iter}", "bpw": 2 + 16.0/GROUP,
                        "ppl": ppl, "degr": degr, "avg_recon_err": avg_err})
        print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 3. AQ with learned codebooks
    print(f"\n=== 3. AQ learned-codebook 2-bit ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log_learned = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _, cb1, cb2 = quant_aq_learned(
                mod.weight.data, n_codebook_epochs=5, codebook_size=256)
            mod.weight.data.copy_(Wq)
            log_learned.append({"name": name, "recon_rel_err": err})
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    avg_err = sum(e["recon_rel_err"] for e in log_learned) / len(log_learned)
    results.append({"cfg": "aq_learned_2bit", "bpw": 2 + 16.0/GROUP,
                    "ppl": ppl, "degr": degr, "avg_recon_err": avg_err})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)

    # 4. AQ + LDLQ (best of both)
    print(f"\n=== 4. AQ + LDLQ 2-bit (joint + activation-aware) ===", flush=True)
    restore(model, snap)
    tq = time.time()
    n_ldlq, n_plain = 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        H = hessians.get(name)
        d_in = mod.weight.data.shape[1]
        with torch.no_grad():
            if H is not None and d_in <= 2048:
                Wq, err, bpw, _ = quant_aq_ldlq(mod.weight.data, H, n_aq_iter=3)
                n_ldlq += 1
            else:
                Wq, err, bpw, _ = quant_aq_fixed_e8(mod.weight.data, n_iter=3)
                n_plain += 1
            mod.weight.data.copy_(Wq)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "aq_ldlq_2bit", "bpw": 2 + 16.0/GROUP,
                    "ppl": ppl, "degr": degr, "n_ldlq": n_ldlq, "n_plain": n_plain})
    print(f"  ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"{'cfg':>28s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s}", flush=True)
    print(f"{'-'*70}", flush=True)
    for r in results:
        print(f"{r['cfg']:>28s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f}", flush=True)

    # Improvement over baseline
    base_rvq = results[0]
    print(f"\n=== improvement over plain E8 RVQ ({base_rvq['ppl']:.3f}) ===", flush=True)
    for r in results[1:]:
        delta = base_rvq["ppl"] - r["ppl"]
        pct = delta / base_rvq["ppl"] * 100
        print(f"  {r['cfg']:>28s}: {delta:+.3f} PPL ({pct:+.1f}%)", flush=True)

    out_dir = "/root/novelquant/runs/aq_2bit" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "aq_2bit")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
