"""Reproduce the data-free quant headline numbers through the new quant/ package.

Verifies workstream 1: the package + canonical eval reproduce the June 21
fullrot.json result (full-dim WH + Lloyd-Max, 3-bit ~40.6 ppl on Qwen2.5-1.5B,
FP16 baseline 22.5).  Runs on Kaggle T4 in a few minutes.

Usage on Kaggle: set MODEL_ID / env as needed; run as a notebook or script.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sys
import time
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant.quantize import quantize_model_inplace, snapshot, restore
from quant.eval import ppl_wt2


def main():
    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    GROUP = int(os.environ.get("GROUP", "128"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=dev).eval()
    print(f"loaded {MODEL_ID} in {time.time() - t0:.1f}s, "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)

    snap = snapshot(model)
    base = ppl_wt2(model, tok, device=dev)
    print(f"[fp16] ppl={base:.4f}", flush=True)

    configs = [("rtn", 3), ("fullrot_whlm", 3), ("rtn", 2), ("fullrot_whlm", 2)]
    results = [{"cfg": "fp16", "method": "none", "bits": 16, "ppl": base,
                "ppl_delta": 0.0, "avg_bpw": 16.0, "avg_recon": 0.0}]

    for method, bits in configs:
        label = f"{method}_b{bits}"
        restore(model, snap)
        tq = time.time()
        qlog = quantize_model_inplace(model, bits, GROUP, method, verbose=True)
        avg_bpw = sum(r["bpw"] for r in qlog) / len(qlog)
        avg_err = sum(r["recon_rel_err"] for r in qlog) / len(qlog)
        max_err = max(r["recon_rel_err"] for r in qlog)
        ppl = ppl_wt2(model, tok, device=dev)
        print(f"[{label}] ppl={ppl:.4f}  avg_bpw={avg_bpw:.3f}  "
              f"recon avg={avg_err:.4f} max={max_err:.4f}  ({time.time() - tq:.1f}s)",
              flush=True)
        results.append({"cfg": label, "method": method, "bits": bits,
                        "ppl": ppl, "ppl_delta": ppl - base,
                        "avg_bpw": avg_bpw, "avg_recon": avg_err,
                        "max_recon": max_err})

    print("\n" + "=" * 70)
    print(f"{'cfg':18s} {'avg_bpw':>8s} {'ppl':>10s} {'d_ppl':>9s} {'recon_avg':>9s}")
    print("-" * 70)
    for r in results:
        print(f"{r['cfg']:18s} {r['avg_bpw']:>8.3f} {r['ppl']:>10.4f} "
              f"{r['ppl_delta']:>+9.4f} {r.get('avg_recon', 0):>9.4f}", flush=True)
    print(f"\nfp16 baseline ppl={base:.4f}  total={time.time() - t0:.1f}s")

    # Save to a writable dir: /kaggle/working on Kaggle, else local runs/pkg_repro
    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working/runs/pkg_repro"
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "pkg_repro")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "quant/ package reproduction of fullrot headline",
                   "model": MODEL_ID, "group": GROUP, "hardware": dev,
                   "eval": "wikitext-2 test, 40k tokens, ctx 2048 (canonical)",
                   "results": results}, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
