"""E8 RVQ on larger models (4B Qwen3, 8B Qwen3) with careful memory management.

The previous run OOM'd on 4B E8 RVQ and couldn't load 8B at all.
This run uses smaller chunks, CPU offloading for snapshots, and
loads models in bfloat16 to fit on T4.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
import json
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant.quantize import snapshot, restore
from quant.eval import ppl_wt2
from quant.e8lattice import quantize_model_e8rvq
from quant.quantize import quantize_model_inplace


MODELS = [
    ("Qwen/Qwen3-4B",  "4B",  "qwen3"),
    ("Qwen/Qwen3-8B",  "8B",  "qwen3"),
]


def estimate_mem_mb(n_params, bpw):
    return n_params * bpw / 8 / 1e6


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    all_results = []

    for model_id, label, family in MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {label} ({family})", flush=True)
        print(f"{'='*60}", flush=True)

        # Load model, try bfloat16 for 8B to save memory
        dtype = torch.bfloat16 if label == "8B" else torch.float16
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map=dev,
                low_cpu_mem_usage=True).eval()
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            continue

        n_params = sum(p.numel() for p in model.parameters())
        snap = snapshot(model)
        base_ppl = ppl_wt2(model, tok, device=dev)
        fp16_mb = estimate_mem_mb(n_params, 16)
        print(f"  params={n_params/1e6:.1f}M  fp16_ppl={base_ppl:.4f}  fp16_mem={fp16_mb:.0f}MB", flush=True)

        results = []

        # Scalar 4-bit baseline
        try:
            restore(model, snap)
            tq = time.time()
            log = quantize_model_inplace(model, 4, 128, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            mem = estimate_mem_mb(n_params, bpw)
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  scalar b4: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f}  mem={mem:.0f}MB ({ok})  ({time.time()-tq:.1f}s)", flush=True)
            results.append({"cfg": "scalar_4bit", "ppl": ppl, "bpw": bpw,
                            "degr": degr, "mem_mb": mem})
        except Exception as e:
            print(f"  scalar b4: FAILED ({e})", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

        # Scalar 5-bit
        try:
            restore(model, snap)
            tq = time.time()
            log = quantize_model_inplace(model, 5, 128, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            mem = estimate_mem_mb(n_params, bpw)
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  scalar b5: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f}  mem={mem:.0f}MB ({ok})  ({time.time()-tq:.1f}s)", flush=True)
            results.append({"cfg": "scalar_5bit", "ppl": ppl, "bpw": bpw,
                            "degr": degr, "mem_mb": mem})
        except Exception as e:
            print(f"  scalar b5: FAILED ({e})", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

        # E8 RVQ 4-bit (with small chunk size to avoid OOM)
        for bits in [4, 6]:
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

    # Summary
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'model':>6s} {'cfg':>16s} {'bpw':>8s} {'ppl':>10s} {'degr':>8s} {'mem_mb':>8s} {'<5%':>5s}", flush=True)
    print(f"{'-'*100}", flush=True)
    for r in all_results:
        for res in r["results"]:
            ok = "YES" if res["degr"] < 0.05 else "no"
            print(f"{r['model']:>6s} {res['cfg']:>16s} {res['bpw']:>8.3f} "
                  f"{res['ppl']:>10.4f} {res['degr']:>+8.4f} {res['mem_mb']:>8.0f} {ok:>5s}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/e8_large" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "e8_large")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "E8 RVQ large models", "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
