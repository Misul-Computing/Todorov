"""E8 RVQ across model sizes + model-size-adaptive framework.

Tests E8 lattice RVQ at 2, 4, 6 bit on Qwen2.5 (0.5B, 1.5B, 3B) and
Qwen3 (4B, 8B) models. Compares to scalar Lloyd-Max baseline.

The model-size-adaptive insight: bigger model at lower precision beats
smaller model at full precision at the same memory budget.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sys
import time
import json
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant.quantize import quantize_model_inplace, snapshot, restore
from quant.eval import ppl_wt2
from quant.e8lattice import quantize_model_e8rvq


# Model sizes from small to large, including newer Qwen3
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B", "qwen2.5"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B", "qwen2.5"),
    ("Qwen/Qwen2.5-3B-Instruct",   "3B",   "qwen2.5"),
    ("Qwen/Qwen3-4B",              "4B",   "qwen3"),
    ("Qwen/Qwen3-8B",              "8B",   "qwen3"),
]


def estimate_mem_mb(n_params, bpw):
    """Estimate model memory in MB."""
    return n_params * bpw / 8 / 1e6


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    all_results = []

    for model_id, label, family in MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {label} ({family})", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map=dev).eval()
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            continue

        n_params = sum(p.numel() for p in model.parameters())
        snap = snapshot(model)
        base_ppl = ppl_wt2(model, tok, device=dev)
        fp16_mb = estimate_mem_mb(n_params, 16)
        print(f"  params={n_params/1e6:.1f}M  fp16_ppl={base_ppl:.4f}  fp16_mem={fp16_mb:.0f}MB", flush=True)

        results = []

        # Scalar baselines at 4 and 5 bit
        for bits in [4, 5]:
            try:
                restore(model, snap)
                tq = time.time()
                log = quantize_model_inplace(model, bits, 128, "fullrot_whlm", verbose=False)
                ppl = ppl_wt2(model, tok, device=dev)
                bpw = log[0]["bpw"]
                degr = (ppl - base_ppl) / base_ppl
                mem = estimate_mem_mb(n_params, bpw)
                ok = "OK" if degr < 0.05 else "FAIL"
                print(f"  scalar b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f}  mem={mem:.0f}MB ({ok})  ({time.time()-tq:.1f}s)", flush=True)
                results.append({"cfg": f"scalar_{bits}bit", "ppl": ppl, "bpw": bpw,
                                "degr": degr, "mem_mb": mem})
            except Exception as e:
                print(f"  scalar b{bits}: FAILED ({e})", flush=True)

            # Clear cache between runs
            if dev == "cuda":
                torch.cuda.empty_cache()

        # E8 RVQ at 2, 4, 6 bit
        for bits in [2, 4, 6]:
            try:
                restore(model, snap)
                tq = time.time()
                log = quantize_model_e8rvq(model, total_bits=bits, group_size=128, verbose=False)
                ppl = ppl_wt2(model, tok, device=dev)
                bpw = log[0]["bpw"]
                degr = (ppl - base_ppl) / base_ppl
                mem = estimate_mem_mb(n_params, bpw)
                ok = "OK" if degr < 0.05 else "FAIL"
                print(f"  E8 RVQ b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f}  mem={mem:.0f}MB ({ok})  ({time.time()-tq:.1f}s)", flush=True)
                results.append({"cfg": f"e8rvq_{bits}bit", "ppl": ppl, "bpw": bpw,
                                "degr": degr, "mem_mb": mem})
            except Exception as e:
                print(f"  E8 RVQ b{bits}: FAILED ({e})", flush=True)

            if dev == "cuda":
                torch.cuda.empty_cache()

        all_results.append({
            "model": label, "family": family, "n_params": n_params,
            "fp16_ppl": base_ppl, "fp16_mb": fp16_mb, "results": results
        })

        del model, snap
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'model':>6s} {'family':>8s} {'cfg':>16s} {'bpw':>8s} {'ppl':>10s} {'degr':>8s} {'mem_mb':>8s} {'<5%':>5s}", flush=True)
    print(f"{'-'*100}", flush=True)
    for r in all_results:
        for res in r["results"]:
            ok = "YES" if res["degr"] < 0.05 else "no"
            print(f"{r['model']:>6s} {r['family']:>8s} {res['cfg']:>16s} {res['bpw']:>8.3f} "
                  f"{res['ppl']:>10.4f} {res['degr']:>+8.4f} {res['mem_mb']:>8.0f} {ok:>5s}", flush=True)

    # Model-size-adaptive comparison
    print(f"\n{'='*100}", flush=True)
    print("MODEL-SIZE-ADAPTIVE: quantized big vs FP16 small at similar memory", flush=True)
    print(f"{'='*100}", flush=True)

    # Find pairs with similar memory
    all_configs = []
    for r in all_results:
        all_configs.append({
            "model": r["model"], "family": r["family"], "cfg": "fp16",
            "ppl": r["fp16_ppl"], "bpw": 16, "mem_mb": r["fp16_mb"],
            "degr": 0.0, "n_params": r["n_params"]
        })
        for res in r["results"]:
            all_configs.append({
                "model": r["model"], "family": r["family"], "cfg": res["cfg"],
                "ppl": res["ppl"], "bpw": res["bpw"], "mem_mb": res["mem_mb"],
                "degr": res["degr"], "n_params": r["n_params"]
            })

    # Sort by memory
    all_configs.sort(key=lambda x: x["mem_mb"])

    # For each FP16 config, find the best quantized config with <= same memory
    fp16_configs = [c for c in all_configs if c["cfg"] == "fp16"]
    quant_configs = [c for c in all_configs if c["cfg"] != "fp16"]

    print(f"{'FP16 model':>12s} {'mem':>8s} {'ppl':>8s} | {'best quant':>20s} {'mem':>8s} {'ppl':>8s} {'winner':>8s}", flush=True)
    print(f"{'-'*100}", flush=True)
    for fp16 in fp16_configs:
        # Find quantized configs with <= 1.2x memory of FP16
        candidates = [q for q in quant_configs if q["mem_mb"] <= fp16["mem_mb"] * 1.1]
        if not candidates:
            continue
        # Pick the one with lowest PPL
        best_q = min(candidates, key=lambda x: x["ppl"])
        winner = "QUANT" if best_q["ppl"] < fp16["ppl"] else "FP16"
        savings = (1 - best_q["mem_mb"] / fp16["mem_mb"]) * 100
        print(f"{fp16['model']:>12s} {fp16['mem_mb']:>7.0f}M {fp16['ppl']:>8.2f} | "
              f"{best_q['model']+' '+best_q['cfg']:>20s} {best_q['mem_mb']:>7.0f}M {best_q['ppl']:>8.2f} "
              f"{winner:>8s} ({savings:+.0f}% mem)", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/e8_adaptive" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "e8_adaptive")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "E8 RVQ + model-size-adaptive", "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
