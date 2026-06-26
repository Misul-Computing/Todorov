"""Test novel data-free techniques to push 4-bit below 5% degradation.

Techniques:
1. Outlier protection: keep top-k% highest-variance groups at FP16
2. Optimized Hadamard signs (kurtosis minimization)
3. Variable group sizes (smaller groups = more scales = better quality)

All data-free, per-matrix. Tested on 1.5B and 3B.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sys
import time
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant.quantize import quant_dequant, snapshot, restore
from quant.eval import ppl_wt2
from quant.codebook import lloyd_max_gaussian
from quant.rotate import fwht, signs_for, next_pow2


def quant_dequant_outlier(W, bits, group_size=128, outlier_pct=5,
                          method="fullrot_whlm"):
    """Quantize with outlier protection: keep top-k% groups at FP16.

    After Hadamard rotation, identify the groups with highest variance.
    Keep those at FP16, quantize the rest at `bits`. This is data-free
    (uses only weight statistics).

    BPW = bits * (1 - k) + 16 * k + index overhead
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)

    # Chunked rotation to avoid OOM
    chunk = 4096
    G = group_size
    n_groups = npad // G

    # First pass: compute group variance (need full rotation for variance)
    group_var = torch.zeros(n_groups, device=W.device)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        blocks = Wr.reshape(Wc.shape[0], n_groups, G)
        group_var += blocks.var(dim=(0, 2)) * Wc.shape[0]
    group_var /= d_out

    # Top-k% groups by variance
    k = max(1, int(n_groups * outlier_pct / 100))
    _, top_idx = group_var.topk(k)
    outlier_mask = torch.zeros(n_groups, dtype=torch.bool, device=W.device)
    outlier_mask[top_idx] = True

    # Second pass: quantize
    centroids = lloyd_max_gaussian(bits, W.device)
    bounds = (centroids[1:] + centroids[:-1]) / 2
    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        out_rotated = torch.empty_like(Wr)
        for g in range(n_groups):
            start, end = g * G, (g + 1) * G
            if outlier_mask[g]:
                out_rotated[:, start:end] = Wr[:, start:end]
            else:
                block = Wrn[:, start:end]
                q = centroids[torch.bucketize(block, bounds)]
                out_rotated[:, start:end] = q * sc
        out[i:i + chunk] = (fwht(out_rotated) * s)[:, :d_in].to(W.dtype)
    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()

    # BPW: bits for non-outlier elements, 16 for outlier elements
    n_outlier = int(outlier_mask.sum().item())
    n_normal = n_groups - n_outlier
    bpw = (bits * n_normal * G + 16 * n_outlier * G) / npad  # per-element
    bpw += 16.0 / npad  # row scale (one FP16 per row, negligible)
    bpw += 1.0 / G      # outlier flag per group
    return out, err, bpw, G


def quantize_model_with_outliers(model, bits, outlier_pct, group_size=128):
    """Apply outlier-protected quantization to every nn.Linear."""
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant_outlier(
                mod.weight.data, bits, group_size, outlier_pct)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
    avg_bpw = sum(e["bpw"] for e in log) / len(log)
    avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
    print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
          f"(b={bits}, outlier={outlier_pct}%, avg_bpw={avg_bpw:.3f}, "
          f"avg_err={avg_err:.4f})", flush=True)
    return log, avg_bpw


def quantize_model_opt_signs(model, bits, group_size=128):
    """Apply quantization with optimized Hadamard signs."""
    log = []
    t0 = time.time()
    device = next(model.parameters()).device
    centroids = lloyd_max_gaussian(bits, device)
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant(
                mod.weight.data, bits, group_size, "fullrot_whlm_opt",
                centroids=centroids)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
    avg_bpw = sum(e["bpw"] for e in log) / len(log)
    avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
    print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
          f"(opt_signs, b={bits}, avg_bpw={avg_bpw:.3f}, "
          f"avg_err={avg_err:.4f})", flush=True)
    return log, avg_bpw


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    all_results = []

    models = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
        ("Qwen/Qwen2.5-3B-Instruct",   "3B"),
    ]

    for model_id, label in models:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {label}", flush=True)
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map=dev).eval()
        snap = snapshot(model)
        base_ppl = ppl_wt2(model, tok, device=dev)
        print(f"  fp16_ppl={base_ppl:.4f}", flush=True)

        results = []

        # Baseline: uniform 4-bit and 5-bit
        for bits in [4, 5]:
            restore(model, snap)
            from quant.quantize import quantize_model_inplace
            log = quantize_model_inplace(model, bits, 128, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            print(f"  uniform b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f}", flush=True)
            results.append({"cfg": f"uniform_{bits}bit", "ppl": ppl, "bpw": bpw, "degr": degr})

        # Technique 1: outlier protection at 4-bit
        for pct in [2, 5, 10, 15, 20]:
            restore(model, snap)
            _, bpw = quantize_model_with_outliers(model, 4, pct, 128)
            ppl = ppl_wt2(model, tok, device=dev)
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  outlier b4 pct={pct}%: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})", flush=True)
            results.append({"cfg": f"outlier4_{pct}pct", "ppl": ppl, "bpw": bpw, "degr": degr})

        # Technique 3: smaller group size at 4-bit (more scales, better quality)
        for gs in [64, 32]:
            restore(model, snap)
            from quant.quantize import quantize_model_inplace
            log = quantize_model_inplace(model, 4, gs, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  uniform b4 gs={gs}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})", flush=True)
            results.append({"cfg": f"uniform4_gs{gs}", "ppl": ppl, "bpw": bpw, "degr": degr})

        all_results.append({"model": label, "fp16_ppl": base_ppl, "results": results})
        del model, snap
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}", flush=True)
    for r in all_results:
        print(f"\n{r['model']} (fp16={r['fp16_ppl']:.4f}):", flush=True)
        print(f"  {'cfg':25s} {'bpw':>8s} {'ppl':>10s} {'degr':>8s} {'<5%':>5s}", flush=True)
        for res in r["results"]:
            ok = "YES" if res["degr"] < 0.05 else "no"
            print(f"  {res['cfg']:25s} {res['bpw']:>8.3f} {res['ppl']:>10.4f} {res['degr']:>+8.4f} {ok:>5s}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/novel4bit" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "novel4bit")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "Novel 4-bit techniques", "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
