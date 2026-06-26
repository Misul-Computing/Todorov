"""Model-size-adaptive: compare quantized big models vs FP16 small models.

The key insight: a 3B at 4-bit might beat a 1.5B at FP16 on PPL, while
using similar memory. This is the "extreme savings" path, use a bigger
model at lower precision instead of a smaller model at full precision.

Test matrix:
- 0.5B FP16 vs 1.5B@4bit vs 3B@3bit (similar memory budget)
- 1.5B FP16 vs 3B@4bit vs 7B@3bit (similar memory budget)
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


def model_size_mb(n_params, bits=16):
    """Model size in MB at given precision."""
    return n_params * bits / 8 / 1e6


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    results = []

    configs = [
        # (model_id, label, bits, expected_size_mb)
        ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B_fp16", 16, None),
        ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B_fp16", 16, None),
        ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B_5bit", 5, None),
        ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B_4bit", 4, None),
        ("Qwen/Qwen2.5-3B-Instruct",   "3B_fp16",   16, None),
        ("Qwen/Qwen2.5-3B-Instruct",   "3B_5bit",   5, None),
        ("Qwen/Qwen2.5-3B-Instruct",   "3B_4bit",   4, None),
        ("Qwen/Qwen2.5-3B-Instruct",   "3B_3bit",   3, None),
    ]

    for model_id, label, bits, _ in configs:
        print(f"\n[{label}]", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map=dev).eval()
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            results.append({"label": label, "error": str(e)})
            continue

        n_params = sum(p.numel() for p in model.parameters())
        size_mb = model_size_mb(n_params, bits if bits < 16 else 16)

        if bits < 16:
            snap = snapshot(model)
            quantize_model_inplace(model, bits, 128, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            restore(model, snap)
        else:
            ppl = ppl_wt2(model, tok, device=dev)

        print(f"  params={n_params/1e6:.1f}M  bits={bits}  "
              f"size={size_mb:.0f}MB  ppl={ppl:.4f}", flush=True)
        results.append({"label": label, "model_id": model_id, "bits": bits,
                        "n_params": n_params, "size_mb": size_mb, "ppl": ppl})

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Summary: group by similar memory budget
    print(f"\n{'='*90}", flush=True)
    print(f"{'Config':20s} {'Params':>10s} {'Bits':>6s} {'Size(MB)':>10s} {'PPL':>10s}", flush=True)
    print(f"{'-'*90}", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['label']:20s} FAILED", flush=True)
            continue
        print(f"{r['label']:20s} {r['n_params']/1e6:>9.1f}M {r['bits']:>6d} "
              f"{r['size_mb']:>10.0f} {r['ppl']:>10.4f}", flush=True)

    # Key comparisons
    print(f"\n{'='*90}", flush=True)
    print("Key comparisons (similar memory budget):", flush=True)
    print(f"{'-'*90}", flush=True)
    by_label = {r["label"]: r for r in results if "error" not in r}

    comparisons = [
        ("0.5B_fp16", "1.5B_4bit", "0.5B FP16 vs 1.5B@4bit"),
        ("0.5B_fp16", "3B_3bit", "0.5B FP16 vs 3B@3bit"),
        ("1.5B_fp16", "3B_4bit", "1.5B FP16 vs 3B@4bit"),
        ("1.5B_5bit", "3B_4bit", "1.5B@5bit vs 3B@4bit"),
    ]
    for a, b, desc in comparisons:
        if a in by_label and b in by_label:
            ra, rb = by_label[a], by_label[b]
            winner = "A" if ra["ppl"] < rb["ppl"] else "B"
            print(f"  {desc}: {ra['label']}={ra['ppl']:.4f} ({ra['size_mb']:.0f}MB) "
                  f"vs {rb['label']}={rb['ppl']:.4f} ({rb['size_mb']:.0f}MB) "
                  f"-> {winner} wins", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/size_adaptive" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "size_adaptive")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "Model-size-adaptive comparison",
                   "results": results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
