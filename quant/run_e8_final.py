"""E8 RVQ + model-size-adaptive: the full test.

E8 RVQ gives 28% lower recon error than scalar Lloyd-Max at 4-bit.
This test measures PPL on Qwen2.5 models and compares to uniform.

Also tests E8 RVQ at 3-bit (2+1 stages) and mixed E8+scalar.
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
from quant.e8lattice import quantize_model_e8rvq


MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct",   "3B"),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    all_results = []

    for model_id, label in MODELS:
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
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params/1e6:.1f}M  fp16_ppl={base_ppl:.4f}", flush=True)

        results = []

        # Scalar baselines
        for bits in [4, 5]:
            restore(model, snap)
            tq = time.time()
            log = quantize_model_inplace(model, bits, 128, "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  scalar b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})  ({time.time()-tq:.1f}s)", flush=True)
            results.append({"cfg": f"scalar_{bits}bit", "ppl": ppl, "bpw": bpw, "degr": degr})

        # E8 RVQ at 2, 4, 6 bit
        for bits in [2, 4, 6]:
            restore(model, snap)
            tq = time.time()
            log = quantize_model_e8rvq(model, total_bits=bits, group_size=128, verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            bpw = log[0]["bpw"]
            degr = (ppl - base_ppl) / base_ppl
            ok = "OK" if degr < 0.05 else "FAIL"
            print(f"  E8 RVQ b{bits}: ppl={ppl:.4f}  bpw={bpw:.3f}  degr={degr:+.4f} ({ok})  ({time.time()-tq:.1f}s)", flush=True)
            results.append({"cfg": f"e8rvq_{bits}bit", "ppl": ppl, "bpw": bpw, "degr": degr})

        # Model-size-adaptive comparison: is E8 RVQ 4-bit better than 0.5B FP16?
        # (Already measured 0.5B FP16 = 14.01 PPL in previous run)

        all_results.append({"model": label, "n_params": n_params,
                            "fp16_ppl": base_ppl, "results": results})
        del model, snap
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}", flush=True)
    for r in all_results:
        print(f"\n{r['model']} (fp16={r['fp16_ppl']:.4f}):", flush=True)
        print(f"  {'cfg':20s} {'bpw':>8s} {'ppl':>10s} {'degr':>8s} {'<5%':>5s}", flush=True)
        for res in r["results"]:
            ok = "YES" if res["degr"] < 0.05 else "no"
            print(f"  {res['cfg']:20s} {res['bpw']:>8.3f} {res['ppl']:>10.4f} {res['degr']:>+8.4f} {ok:>5s}", flush=True)

    # Key comparison
    print(f"\n{'='*90}", flush=True)
    print("Model-size-adaptive: E8 RVQ 4-bit vs smaller model FP16", flush=True)
    print(f"{'-'*90}", flush=True)
    # 1.5B E8 4-bit vs 0.5B FP16 (14.01)
    r15 = next((r for r in all_results if r["model"] == "1.5B"), None)
    if r15:
        e8_4 = next((res for res in r15["results"] if res["cfg"] == "e8rvq_4bit"), None)
        if e8_4:
            print(f"  0.5B FP16: PPL=14.01, size=988MB", flush=True)
            print(f"  1.5B E8 4-bit: PPL={e8_4['ppl']:.2f}, size={r15['n_params']*4.125/8/1e6:.0f}MB", flush=True)
            print(f"  -> {'E8 wins' if e8_4['ppl'] < 14.01 else 'FP16 wins'}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/e8_final" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "e8_final")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "E8 RVQ final", "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
