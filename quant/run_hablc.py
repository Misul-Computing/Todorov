"""Test HAB-LC: Hadamard + sign quantize + low-rank residual compensation.

Sweep ranks to find the bpw-vs-quality tradeoff at 1-2 bpw.
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
from quant.quantize import snapshot, restore
from quant.hablc import sbr_quant_model
from quant.eval import ppl_wt2


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

    ranks = [0, 8, 16, 24, 32, 48, 64, 96]
    results = [{"cfg": "fp16", "rank": -1, "ppl": base,
                "ppl_delta": 0.0, "avg_bpw": 16.0}]

    for rank in ranks:
        restore(model, snap)
        tq = time.time()
        label = f"hablc:r{rank}" if rank > 0 else "hablc:binary_only"
        try:
            log = sbr_quant_model(model, rank, verbose=False)
            avg_recon = sum(e["recon_rel_err"] for e in log if not e.get("skipped")) / \
                        max(1, sum(1 for e in log if not e.get("skipped")))
            bpw = log[0]["bpw"] if log else 0
            ppl = ppl_wt2(model, tok, device=dev)
            print(f"[{label:25s}] ppl={ppl:.4f}  d={ppl - base:+.4f}  "
                  f"bpw={bpw:.3f}  recon={avg_recon:.4f}  ({time.time() - tq:.1f}s)",
                  flush=True)
            results.append({"cfg": label, "rank": rank, "ppl": ppl,
                            "ppl_delta": ppl - base, "avg_bpw": bpw,
                            "avg_recon": avg_recon})
        except Exception as e:
            print(f"[{label:25s}] FAILED: {e}", flush=True)
            results.append({"cfg": label, "rank": rank, "ppl": None,
                            "ppl_delta": None, "avg_bpw": None,
                            "avg_recon": None, "error": str(e)})

    print("\n" + "=" * 80)
    print(f"{'cfg':27s} {'avg_bpw':>8s} {'ppl':>10s} {'d_ppl':>9s} {'recon':>7s}")
    print("-" * 80)
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

    out_dir = "/kaggle/working/runs/hablc" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "hablc")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "SBR: SVD + binary residual",
                   "model": MODEL_ID, "hardware": dev,
                   "eval": "wikitext-2 test, 40k tokens, ctx 2048",
                   "results": results}, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
