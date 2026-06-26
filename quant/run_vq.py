"""Compare scalar fullrot_whlm vs VQ (D4, E8, Gaussian) at 2-bit and 3-bit.

The key test: does vector quantization in the Hadamard-rotated domain beat
scalar Lloyd-Max at equal BPW? The Gaussian shaping gain says it should.

Configs:
  fp16                    , baseline
  fullrot_whlm_b2         , scalar Lloyd-Max, 2-bit (current best scalar)
  fullrot_vq:gaussian:2:1 , Gaussian k-means VQ, d=4, 2-bit single-stage
  fullrot_vq:d4:2:1       , D4 lattice VQ, d=4, 2-bit single-stage
  fullrot_vq:d4:1:2       , D4 lattice RVQ, d=4, 1-bit x 2 stages (vertical stacking)
  fullrot_vq:e8:1:2       , E8 lattice RVQ, d=8, 1-bit x 2 stages
  fullrot_whlm_b3         , scalar Lloyd-Max, 3-bit
  fullrot_vq:gaussian:3:1 , Gaussian k-means VQ, d=4, 3-bit single-stage
  fullrot_vq:d4:3:1       , D4 lattice VQ, d=4, 3-bit single-stage
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

    configs = [
        ("fullrot_whlm_b3",         "fullrot_whlm",        3,  128),
        ("fullrot_whlm_b4",         "fullrot_whlm",        4,  128),
        ("fullrot_vq:gaussian:2:1", "fullrot_vq:gaussian:2:1", 2, 128),
        ("fullrot_vq:d4:2:1",       "fullrot_vq:d4:2:1",   2, 128),
    ]

    results = [{"cfg": "fp16", "method": "none", "bits": 16, "ppl": base,
                "ppl_delta": 0.0, "avg_bpw": 16.0}]

    for label, method, bits, group in configs:
        restore(model, snap)
        tq = time.time()
        log = quantize_model_inplace(model, bits, group, method, verbose=False)
        avg_recon = sum(e["recon_rel_err"] for e in log if not e.get("skipped")) / \
                    max(1, sum(1 for e in log if not e.get("skipped")))
        max_recon = max(e["recon_rel_err"] for e in log if not e.get("skipped"))
        bpw = log[0]["bpw"] if log else 0
        ppl = ppl_wt2(model, tok, device=dev)
        print(f"[{label:30s}] ppl={ppl:.4f}  d={ppl - base:+.4f}  "
              f"bpw={bpw:.3f}  recon={avg_recon:.4f}  ({time.time() - tq:.1f}s)",
              flush=True)
        results.append({"cfg": label, "method": method, "bits": bits,
                        "ppl": ppl, "ppl_delta": ppl - base, "avg_bpw": bpw,
                        "avg_recon": avg_recon, "max_recon": max_recon})

    print("\n" + "=" * 80)
    print(f"{'cfg':32s} {'avg_bpw':>8s} {'ppl':>10s} {'d_ppl':>9s} {'recon':>7s}")
    print("-" * 80)
    for r in results:
        recon = r.get("avg_recon", 0.0)
        print(f"{r['cfg']:32s} {r['avg_bpw']:>8.3f} {r['ppl']:>10.4f} "
              f"{r['ppl_delta']:>+9.4f} {recon:>7.4f}", flush=True)
    print(f"\nfp16 baseline ppl={base:.4f}  total={time.time() - t0:.1f}s")

    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working/runs/vq"
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "vq")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "VQ vs scalar in Hadamard domain",
                   "model": MODEL_ID, "group": GROUP, "hardware": dev,
                   "eval": "wikitext-2 test, 40k tokens, ctx 2048",
                   "results": results}, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
