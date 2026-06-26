"""Sweep bit-widths across model sizes to find <5% PPL degradation point.

Tests Qwen2.5-0.5B, 1.5B, 3B at 3, 4, 5, 6 bit with fullrot_whlm.
Finds the minimum BPW where PPL degradation < 5% for each model size.
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


MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct",   "3B"),
]
BIT_WIDTHS = [3, 4, 5, 6]
DEGRADATION_TARGET = 0.05  # 5%


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    all_results = []

    for model_id, model_label in MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_label} ({model_id})", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map=dev).eval()
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            all_results.append({"model": model_label, "model_id": model_id,
                                "error": str(e), "sweep": []})
            continue

        snap = snapshot(model)
        base_ppl = ppl_wt2(model, tok, device=dev)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params/1e6:.1f}M  fp16_ppl={base_ppl:.4f}", flush=True)

        sweep = [{"bits": 16, "bpw": 16.0, "ppl": base_ppl,
                  "degradation": 0.0, "within_target": True}]

        for bits in BIT_WIDTHS:
            restore(model, snap)
            tq = time.time()
            log = quantize_model_inplace(model, bits, 128, "fullrot_whlm",
                                         verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"] if log else bits + 16.0 / 128
            degradation = (ppl - base_ppl) / base_ppl
            within = degradation < DEGRADATION_TARGET
            print(f"  b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  "
                  f"degradation={degradation:+.4f} ({'OK' if within else 'FAIL'})  "
                  f"({time.time() - tq:.1f}s)", flush=True)
            sweep.append({"bits": bits, "bpw": bpw, "ppl": ppl,
                          "degradation": degradation, "within_target": within})

        # Find minimum BPW within target
        ok = [s for s in sweep if s["within_target"] and s["bits"] < 16]
        min_bpw = min(s["bpw"] for s in ok) if ok else None
        print(f"  -> min BPW within 5% target: {min_bpw:.3f}" if min_bpw
              else "  -> no config within 5% target", flush=True)

        all_results.append({"model": model_label, "model_id": model_id,
                            "n_params": n_params,
                            "fp16_ppl": base_ppl,
                            "min_bpw_within_target": min_bpw,
                            "sweep": sweep})
        del model, snap
        torch.cuda.empty_cache() if dev == "cuda" else None

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print(f"{'Model':>8s} {'Params':>10s} {'FP16 PPL':>10s} {'Min BPW':>10s} {'@PPL':>10s} {'Compress':>10s}", flush=True)
    print(f"{'-'*80}", flush=True)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:>8s} {'FAILED':>10s}", flush=True)
            continue
        min_bpw = r.get("min_bpw_within_target")
        if min_bpw:
            entry = [s for s in r["sweep"] if s["bpw"] == min_bpw][0]
            compress = 16.0 / min_bpw
            print(f"{r['model']:>8s} {r['n_params']/1e6:>9.1f}M {r['fp16_ppl']:>10.4f} "
                  f"{min_bpw:>10.3f} {entry['ppl']:>10.4f} {compress:>9.1f}x", flush=True)
        else:
            print(f"{r['model']:>8s} {r['n_params']/1e6:>9.1f}M {r['fp16_ppl']:>10.4f} "
                  f"{'N/A':>10s} {'N/A':>10s} {'N/A':>10s}", flush=True)
    print(f"\ntotal time: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/sweep" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "sweep")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "Model-size-adaptive BPW sweep",
                   "target_degradation": DEGRADATION_TARGET,
                   "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
