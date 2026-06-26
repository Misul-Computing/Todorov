"""Dense-sparse decomposition + Hadamard rotation (novel combination).

Extract outliers from W in the original domain, store them at FP16 (sparse).
Apply Hadamard rotation + Lloyd-Max quantization to the residual.

The residual has smaller variance and is more Gaussian, so the rotation
works better on it. The outliers are stored exactly.

BPW = bits * (1-k) + 16 * k + sparse_index_overhead
For k=0.5%: ~4.3 bpw (vs 4.125 for uniform 4-bit)

This is data-free (outliers identified by weight magnitude only).
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
from quant.quantize import snapshot, restore
from quant.eval import ppl_wt2
from quant.codebook import lloyd_max_gaussian
from quant.rotate import fwht, signs_for, next_pow2


def quant_dequant_dense_sparse(W, bits, group_size=128, outlier_pct=0.5,
                               chunk=4096):
    """Dense-sparse decomposition + Hadamard rotation.

    1. Extract top-k% weights by absolute value (outliers)
    2. W_residual = W - W_outliers
    3. Apply Hadamard rotation + Lloyd-Max to W_residual
    4. Store outliers at FP16 (sparse)

    Returns (Wq, recon_rel_err, bpw, n_outliers).
    """
    d_out, d_in = W.shape
    Wf = W.float()

    # Extract outliers (top-k% by absolute value)
    n_elements = d_out * d_in
    k = max(1, int(n_elements * outlier_pct / 100))
    Wflat = Wf.abs().flatten()
    _, top_flat_idx = Wflat.topk(k)
    # Convert flat indices to 2D
    outlier_rows = top_flat_idx // d_in
    outlier_cols = top_flat_idx % d_in

    # Create outlier mask and extract
    mask = torch.zeros_like(Wf, dtype=torch.bool)
    mask[outlier_rows, outlier_cols] = True
    W_outliers = Wf * mask  # sparse, only outliers
    W_residual = Wf - W_outliers  # dense, no outliers

    # Quantize residual with Hadamard + Lloyd-Max
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    centroids = lloyd_max_gaussian(bits, W.device)
    bounds = (centroids[1:] + centroids[:-1]) / 2

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = W_residual[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        q = centroids[torch.bucketize(Wrn, bounds)]
        q = q * sc
        out_rotated = (fwht(q) * s)[:, :d_in]
        # Add back outliers
        out[i:i + chunk] = (out_rotated + W_outliers[i:i + chunk]).to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()

    # BPW: bits for non-outlier elements + 16 for outlier values + 32 for indices
    bpw = bits * (1 - outlier_pct / 100)  # quantized elements
    bpw += 16 * outlier_pct / 100  # outlier values at FP16
    bpw += 32 * outlier_pct / 100  # outlier indices (32-bit int per outlier)
    bpw += 16.0 / group_size  # scale per group
    return out, err, bpw, k


def quantize_model_dense_sparse(model, bits, outlier_pct, group_size=128):
    """Apply dense-sparse + Hadamard quantization to every nn.Linear."""
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    total_outliers = 0
    for name, mod in targets:
        with torch.no_grad():
            Wq, err, bpw, n_out = quant_dequant_dense_sparse(
                mod.weight.data, bits, group_size, outlier_pct)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
        total_outliers += n_out
    avg_bpw = sum(e["bpw"] for e in log) / len(log)
    avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
    print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
          f"(b={bits}, outlier={outlier_pct}%, avg_bpw={avg_bpw:.3f}, "
          f"avg_err={avg_err:.4f}, total_outliers={total_outliers})", flush=True)
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

        # Baselines
        for bits in [4, 5]:
            restore(model, snap)
            from quant.quantize import quantize_model_inplace
            log = quantize_model_inplace(model, bits, 128, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  uniform b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})", flush=True)
            results.append({"cfg": f"uniform_{bits}bit", "ppl": ppl, "bpw": bpw, "degr": degr})

        # Dense-sparse at 4-bit with varying outlier percentages
        for pct in [0.1, 0.5, 1.0, 2.0, 5.0]:
            restore(model, snap)
            _, bpw = quantize_model_dense_sparse(model, 4, pct, 128)
            ppl = ppl_wt2(model, tok, device=dev)
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  dense_sparse b4 pct={pct}%: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})", flush=True)
            results.append({"cfg": f"dense_sparse4_{pct}pct", "ppl": ppl, "bpw": bpw, "degr": degr})

        # Dense-sparse at 3-bit (push lower)
        for pct in [0.5, 1.0, 2.0, 5.0]:
            restore(model, snap)
            _, bpw = quantize_model_dense_sparse(model, 3, pct, 128)
            ppl = ppl_wt2(model, tok, device=dev)
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  dense_sparse b3 pct={pct}%: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})", flush=True)
            results.append({"cfg": f"dense_sparse3_{pct}pct", "ppl": ppl, "bpw": bpw, "degr": degr})

        all_results.append({"model": label, "fp16_ppl": base_ppl, "results": results})
        del model, snap
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}", flush=True)
    for r in all_results:
        print(f"\n{r['model']} (fp16={r['fp16_ppl']:.4f}):", flush=True)
        print(f"  {'cfg':30s} {'bpw':>8s} {'ppl':>10s} {'degr':>8s} {'<5%':>5s}", flush=True)
        for res in r["results"]:
            ok = "YES" if res["degr"] < 0.05 else "no"
            print(f"  {res['cfg']:30s} {res['bpw']:>8.3f} {res['ppl']:>10.4f} {res['degr']:>+8.4f} {ok:>5s}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/dense_sparse" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "dense_sparse")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "Dense-sparse + Hadamard", "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
