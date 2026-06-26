"""Test block-wise fine-tuning on top of VQ quantization.

1. Load model, collect original block outputs (targets)
2. Quantize at 1-bit and 2-bit
3. Fine-tune LayerNorm/bias to match original outputs
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
from quant.finetune import collect_block_targets, finetune_model


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

    # Collect original block outputs BEFORE quantization
    print("collecting original block targets...", flush=True)
    tq = time.time()
    orig_targets, calib_seqs = collect_block_targets(model, tok, n_samples=64,
                                                     seq_len=512, device=dev)
    print(f"collected targets for {len(orig_targets)} blocks in {time.time() - tq:.1f}s",
          flush=True)

    results = [{"cfg": "fp16", "ppl": base, "ppl_delta": 0.0, "avg_bpw": 16.0}]

    # 2-bit: quantize + fine-tune with more epochs
    restore(model, snap)
    tq = time.time()
    log = quantize_model_inplace(model, 2, 128, "fullrot_vq:datafit:2:1", verbose=True)
    avg_recon2 = sum(e["recon_rel_err"] for e in log if not e.get("skipped")) / \
                 max(1, sum(1 for e in log if not e.get("skipped")))
    bpw2 = log[0]["bpw"] if log else 0
    ppl_pre2 = ppl_wt2(model, tok, device=dev)
    print(f"[2bit_vq_pre_ft              ] ppl={ppl_pre2:.4f}  d={ppl_pre2 - base:+.4f}  "
          f"bpw={bpw2:.3f}  recon={avg_recon2:.4f}  ({time.time() - tq:.1f}s)", flush=True)
    results.append({"cfg": "2bit_vq_pre_ft", "ppl": ppl_pre2, "ppl_delta": ppl_pre2 - base,
                    "avg_bpw": bpw2, "avg_recon": avg_recon2})

    try:
        finetune_model(model, tok, orig_targets, calib_seqs,
                       n_epochs=10, lr=3e-4, device=dev, verbose=True)
        ppl_post2 = ppl_wt2(model, tok, device=dev)
        print(f"[2bit_vq_post_ft_10ep        ] ppl={ppl_post2:.4f}  d={ppl_post2 - base:+.4f}  "
              f"({time.time() - tq:.1f}s)", flush=True)
        results.append({"cfg": "2bit_vq_post_ft_10ep", "ppl": ppl_post2,
                        "ppl_delta": ppl_post2 - base, "avg_bpw": bpw2,
                        "avg_recon": avg_recon2})
    except Exception as e:
        print(f"2-bit fine-tuning FAILED: {e}", flush=True)
        results.append({"cfg": "2bit_vq_post_ft_10ep", "ppl": None, "error": str(e)})

    # 1-bit: quantize + fine-tune with more epochs
    restore(model, snap)
    for p in model.parameters():
        p.requires_grad_(True)
    tq = time.time()
    log = quantize_model_inplace(model, 1, 128, "fullrot_vq:datafit8:1:1", verbose=True)
    avg_recon = sum(e["recon_rel_err"] for e in log if not e.get("skipped")) / \
                max(1, sum(1 for e in log if not e.get("skipped")))
    bpw = log[0]["bpw"] if log else 0
    ppl_pre = ppl_wt2(model, tok, device=dev)
    print(f"[1bit_vq_pre_ft              ] ppl={ppl_pre:.4f}  d={ppl_pre - base:+.4f}  "
          f"bpw={bpw:.3f}  recon={avg_recon:.4f}  ({time.time() - tq:.1f}s)", flush=True)
    results.append({"cfg": "1bit_vq_pre_ft", "ppl": ppl_pre, "ppl_delta": ppl_pre - base,
                    "avg_bpw": bpw, "avg_recon": avg_recon})

    try:
        finetune_model(model, tok, orig_targets, calib_seqs,
                       n_epochs=10, lr=3e-4, device=dev, verbose=True)
        ppl_post = ppl_wt2(model, tok, device=dev)
        print(f"[1bit_vq_post_ft_10ep        ] ppl={ppl_post:.4f}  d={ppl_post - base:+.4f}  "
              f"({time.time() - tq:.1f}s)", flush=True)
        results.append({"cfg": "1bit_vq_post_ft_10ep", "ppl": ppl_post,
                        "ppl_delta": ppl_post - base, "avg_bpw": bpw,
                        "avg_recon": avg_recon})
    except Exception as e:
        import traceback
        print(f"1-bit fine-tuning FAILED: {e}", flush=True)
        traceback.print_exc()
        results.append({"cfg": "1bit_vq_post_ft_10ep", "ppl": None, "error": str(e)})

    print("\n" + "=" * 80)
    print(f"{'cfg':30s} {'bpw':>8s} {'ppl':>12s} {'d_ppl':>11s} {'recon':>7s}")
    print("-" * 80)
    for r in results:
        recon = r.get("avg_recon", 0.0) or 0.0
        ppl = r.get("ppl")
        ppl_s = f"{ppl:.4f}" if ppl is not None else "FAILED"
        bpw = r.get("avg_bpw")
        bpw_s = f"{bpw:.3f}" if bpw is not None else "N/A"
        delta = r.get("ppl_delta")
        delta_s = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{r['cfg']:30s} {bpw_s:>8s} {ppl_s:>12s} {delta_s:>11s} {recon:>7.4f}",
              flush=True)
    print(f"\nfp16 baseline ppl={base:.4f}  total={time.time() - t0:.1f}s")

    out_dir = "/kaggle/working/runs/finetune" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "finetune")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "VQ + block-wise fine-tuning", "model": MODEL_ID,
                   "results": results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
