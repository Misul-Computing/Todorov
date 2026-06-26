"""Quick AQLM k-means-only test at 1 bpw. No Adam, no fine-tuning, no activations."""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sys
import time
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant.quantize import snapshot, restore
from quant.eval import ppl_wt2
from quant.aqlm import aqlm_quantize_model


def main():
    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=dev).eval()
    print(f"loaded {MODEL_ID} in {time.time() - t0:.1f}s", flush=True)

    snap = snapshot(model)
    base = ppl_wt2(model, tok, device=dev)
    print(f"[fp16] ppl={base:.4f}", flush=True)

    results = [{"cfg": "fp16", "ppl": base, "ppl_delta": 0.0, "avg_bpw": 16.0}]

    # k-means only, no Adam, no activations
    for label, d, nbits in [("aqlm_d8_kmeans", 8, 8), ("aqlm_d4_kmeans", 4, 4)]:
        restore(model, snap)
        tq = time.time()
        try:
            log = aqlm_quantize_model(model, None, d=d, nbits=nbits,
                                      n_iters=0, lr=1e-4,
                                      device=dev, verbose=True)
            avg_recon = sum(e["recon_rel_err"] for e in log if not e.get("skipped")) / \
                        max(1, sum(1 for e in log if not e.get("skipped")))
            bpw = log[0]["bpw"] if log else 0
            ppl = ppl_wt2(model, tok, device=dev)
            print(f"[{label:25s}] ppl={ppl:.4f}  d={ppl - base:+.4f}  "
                  f"bpw={bpw:.3f}  recon={avg_recon:.4f}  ({time.time() - tq:.1f}s)",
                  flush=True)
            results.append({"cfg": label, "ppl": ppl, "ppl_delta": ppl - base,
                            "avg_bpw": bpw, "avg_recon": avg_recon})
        except Exception as e:
            import traceback
            print(f"[{label:25s}] FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"cfg": label, "ppl": None, "error": str(e)})

    print("\n" + "=" * 70)
    print(f"{'cfg':27s} {'bpw':>8s} {'ppl':>10s} {'d_ppl':>9s} {'recon':>7s}")
    print("-" * 70)
    for r in results:
        recon = r.get("avg_recon", 0.0) or 0.0
        ppl = r.get("ppl")
        ppl_s = f"{ppl:.4f}" if ppl is not None else "FAILED"
        bpw = r.get("avg_bpw")
        bpw_s = f"{bpw:.3f}" if bpw is not None else "N/A"
        delta = r.get("ppl_delta")
        delta_s = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{r['cfg']:27s} {bpw_s:>8s} {ppl_s:>10s} {delta_s:>9s} {recon:>7.4f}",
              flush=True)
    print(f"\nfp16 baseline ppl={base:.4f}  total={time.time() - t0:.1f}s")

    out_dir = "/kaggle/working/runs/aqlm1bit" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "aqlm1bit")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "AQLM k-means only 1-bit", "model": MODEL_ID,
                   "results": results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
