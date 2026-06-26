"""LDLQ debug + fix: the 211k PPL was caused by 100x too-weak regularization
(0.01 vs 1.0 in working ldlq.py) and wrong Hessian normalization.

This script tests:
1. Data-free LDLQ at 2-bit (control, should match or beat plain E8 2-bit)
2. Real-Hessian LDLQ at 2-bit with fixed regularization (1.0) and normalization
3. Same at 3-bit and 4-bit for comparison
4. Diagnostics: condition number, max adjustment magnitude, recon error

The LDLQ adjustment: when quantizing block k, adjust it by the Hessian-weighted
error from previous blocks: W_block += (W_prev - Q_prev) @ U[:k*d, k*d:(k+1)*d]
where U = L^T - I from Cholesky of the rotated Hessian. With weak regularization,
U has huge values and the adjustment explodes.
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
from e8lattice import e8_lattice_quantize, quant_dequant_e8rvq, quantize_model_e8rvq

GROUP = 128


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


def quant_e8_ldlq(W, H, total_bits=2, group_size=GROUP, chunk=2048, reg=1.0,
                  use_real_hessian=True, verbose_diag=False):
    """E8 RVQ with LDLQ adaptive rounding.

    If use_real_hessian: transform the real activation Hessian to Hadamard space.
    Else: use data-free proxy H = Wrn^T Wrn (like ldlq.py).
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0
    n_e8_stages = total_bits // 2
    has_sign_stage = (total_bits % 2 == 1)

    out = torch.empty_like(W)
    diag_info = {"max_adj": 0.0, "cond": 0.0, "n_chunks": 0}

    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        nb = npad // d

        # Build Hessian in rotated space
        if use_real_hessian and H is not None:
            # Transform real Hessian to Hadamard space: H_rot = D^T H D
            Hpad = torch.zeros(npad, npad, device=W.device)
            Hpad[:d_in, :d_in] = H.to(W.device).float()
            Hs = Hpad * s.unsqueeze(0) * s.unsqueeze(1)
            H_rot = fwht(fwht(Hs).T).T
            # Normalize: divide by trace/n so diagonal ~ O(1)
            tr = H_rot.diagonal().mean().clamp(min=1e-8)
            H_rot = H_rot / tr
        else:
            # Data-free proxy: H = Wrn^T Wrn (from this chunk's normalized rows)
            H = Wrn[:min(128, Wrn.shape[0])].T @ Wrn[:min(128, Wrn.shape[0])]
            H = H / H.shape[0]
            H_rot = H

        # Regularize (key fix: 1.0 not 0.01)
        H_rot = H_rot + reg * torch.eye(npad, device=W.device)

        if verbose_diag and diag_info["n_chunks"] == 0:
            try:
                eigs = torch.linalg.eigvalsh(H_rot)
                diag_info["cond"] = (eigs.max() / eigs.min().clamp(min=1e-12)).item()
            except Exception:
                diag_info["cond"] = float("inf")

        try:
            L = torch.linalg.cholesky(H_rot)
            U = L.T - torch.eye(npad, device=W.device)
        except Exception:
            U = torch.zeros(npad, npad, device=W.device)

        # LDLQ: quantize block by block with adjustment
        What = torch.zeros_like(Wrn)
        max_adj_this = 0.0
        for k in range(nb):
            if k > 0:
                err = Wrn[:, :k * d] - What[:, :k * d]
                A_k = U[:k * d, k * d:(k + 1) * d]
                adj = err @ A_k
                max_adj_this = max(max_adj_this, adj.abs().max().item())
            else:
                adj = 0.0
            W_block = Wrn[:, k * d:(k + 1) * d] + adj
            # E8 quantize this block
            q = e8_lattice_quantize(W_block.reshape(-1, d), scale=1.0)
            What[:, k * d:(k + 1) * d] = q.reshape(W_block.shape)

        # Sign-residual stage for odd bits
        if has_sign_stage:
            residual = Wrn - What
            block_scale = residual.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            q_sign = torch.sign(residual) * block_scale
            What = What + q_sign

        # Additional E8 stages for >2-bit
        for stage in range(1, n_e8_stages):
            residual = Wrn - What
            res_std = residual.std().clamp(min=1e-8)
            blocks_res = residual.reshape(-1, d)
            q = e8_lattice_quantize(blocks_res, scale=res_std)
            What = What + q.reshape(Wrn.shape)

        q_full = What * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

        diag_info["max_adj"] = max(diag_info["max_adj"], max_adj_this)
        diag_info["n_chunks"] += 1

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = total_bits + 16.0 / group_size
    return out, err, bpw, group_size, diag_info


def run_ldlq_sweep(model, tok, snap, base_ppl, dev, hessians, results, t0):
    for bits in [2, 3, 4]:
        # 1. Plain E8 (baseline)
        restore(model, snap)
        quantize_model_e8rvq(model, total_bits=bits, group_size=GROUP, verbose=False)
        ppl_plain = ppl_wt2(model, tok, device=dev)
        degr_plain = (ppl_plain - base_ppl) / base_ppl
        results.append({"cfg": f"e8_{bits}bit_plain", "bpw": bits + 16.0/GROUP,
                        "ppl": ppl_plain, "degr": degr_plain})
        print(f"  E8 {bits}bit plain:      ppl={ppl_plain:.3f} degr={degr_plain:+.4f}", flush=True)

        # 2. Data-free LDLQ (control)
        restore(model, snap)
        tq = time.time()
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            with torch.no_grad():
                Wq, err, bpw, _, _ = quant_e8_ldlq(
                    mod.weight.data, None, total_bits=bits, use_real_hessian=False)
                mod.weight.data.copy_(Wq)
        ppl_df = ppl_wt2(model, tok, device=dev)
        degr_df = (ppl_df - base_ppl) / base_ppl
        results.append({"cfg": f"e8_ldlq_df_{bits}bit", "bpw": bits + 16.0/GROUP,
                        "ppl": ppl_df, "degr": degr_df})
        print(f"  E8+LDLQ(df) {bits}bit:    ppl={ppl_df:.3f} degr={degr_df:+.4f} ({time.time()-tq:.0f}s)", flush=True)

        # 3. Real-Hessian LDLQ (the fix)
        restore(model, snap)
        tq = time.time()
        n_ldlq, n_plain = 0, 0
        first_diag = None
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            H = hessians.get(name)
            d_in = mod.weight.data.shape[1]
            with torch.no_grad():
                if H is not None and d_in <= 2048:
                    Wq, err, bpw, _, diag = quant_e8_ldlq(
                        mod.weight.data, H, total_bits=bits, use_real_hessian=True,
                        verbose_diag=(first_diag is None))
                    if first_diag is None:
                        first_diag = diag
                    n_ldlq += 1
                else:
                    # Fallback to plain E8 for large MLP tensors
                    Wq, err, bpw, _ = quant_dequant_e8rvq(
                        mod.weight.data, total_bits=bits, group_size=GROUP)
                    n_plain += 1
                mod.weight.data.copy_(Wq)
        ppl_real = ppl_wt2(model, tok, device=dev)
        degr_real = (ppl_real - base_ppl) / base_ppl
        results.append({"cfg": f"e8_ldlq_real_{bits}bit", "bpw": bits + 16.0/GROUP,
                        "ppl": ppl_real, "degr": degr_real,
                        "n_ldlq": n_ldlq, "n_plain": n_plain,
                        "max_adj": first_diag["max_adj"] if first_diag else None,
                        "cond": first_diag["cond"] if first_diag else None})
        print(f"  E8+LDLQ(real) {bits}bit:   ppl={ppl_real:.3f} degr={degr_real:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        if first_diag:
            print(f"    diag: max_adj={first_diag['max_adj']:.4f} cond={first_diag['cond']:.1f}", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None


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

    print("capturing activations...", flush=True)
    hessians = capture_hessians(model, tok, n_examples=48, max_len=512, dev=dev)
    print(f"  {len(hessians)} Hessians ({time.time()-t0:.0f}s)", flush=True)

    results = []
    print(f"\n=== LDLQ sweep (reg=1.0, proper normalization) ===", flush=True)
    run_ldlq_sweep(model, tok, snap, base_ppl, dev, hessians, results, t0)

    print(f"\n{'='*70}", flush=True)
    print(f"{'cfg':>24s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s}", flush=True)
    print(f"{'-'*70}", flush=True)
    for r in results:
        print(f"{r['cfg']:>24s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f}", flush=True)

    out_dir = "/root/novelquant/runs/ldlq_debug" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ldlq_debug")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
