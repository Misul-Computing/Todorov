"""Test E8 lattice RVQ vs scalar Lloyd-Max on Qwen2.5 models.

E8 lattice RVQ gives ~28% lower reconstruction error than scalar Lloyd-Max
at 4-bit. This test measures the actual PPL improvement.
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
        print(f"  fp16_ppl={base_ppl:.4f}", flush=True)

        results = []

        # Test at 2, 3, 4, 5, 6 bit
        for bits in [2, 3, 4, 5, 6]:
            # Scalar Lloyd-Max baseline
            restore(model, snap)
            tq = time.time()
            log = quantize_model_inplace(model, bits, 128, "fullrot_whlm", verbose=False)
            ppl_scalar = ppl_wt2(model, tok, device=dev)
            bpw_s = log[0]["bpw"]
            degr_s = (ppl_scalar - base_ppl) / base_ppl
            print(f"  scalar b{bits}: ppl={ppl_scalar:.4f}  bpw={bpw_s:.3f}  "
                  f"degr={degr_s:+.4f}  ({time.time()-tq:.1f}s)", flush=True)

            # E8 RVQ (only even bits: 2, 4, 6)
            if bits % 2 == 0:
                restore(model, snap)
                tq = time.time()
                log = quantize_model_e8rvq(model, total_bits=bits, group_size=128,
                                           verbose=False)
                ppl_e8 = ppl_wt2(model, tok, device=dev)
                bpw_e = log[0]["bpw"]
                degr_e = (ppl_e8 - base_ppl) / base_ppl
                ok = "OK" if degr_e < 0.05 else "FAIL"
                print(f"  E8 RVQ b{bits}: ppl={ppl_e8:.4f}  bpw={bpw_e:.3f}  "
                      f"degr={degr_e:+.4f} ({ok})  ({time.time()-tq:.1f}s)", flush=True)
                results.append({"bits": bits, "method": "e8_rvq", "ppl": ppl_e8,
                                "bpw": bpw_e, "degr": degr_e})
            results.append({"bits": bits, "method": "scalar_lm", "ppl": ppl_scalar,
                            "bpw": bpw_s, "degr": degr_s})

        all_results.append({"model": label, "fp16_ppl": base_ppl, "results": results})
        del model, snap
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}", flush=True)
    for r in all_results:
        print(f"\n{r['model']} (fp16={r['fp16_ppl']:.4f}):", flush=True)
        print(f"  {'method':15s} {'bits':>6s} {'bpw':>8s} {'ppl':>10s} {'degr':>8s} {'<5%':>5s}", flush=True)
        for res in r["results"]:
            ok = "YES" if res["degr"] < 0.05 else "no"
            print(f"  {res['method']:15s} {res['bits']:>6d} {res['bpw']:>8.3f} "
                  f"{res['ppl']:>10.4f} {res['degr']:>+8.4f} {ok:>5s}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/e8rvq" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "e8rvq")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "E8 lattice RVQ vs scalar", "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
