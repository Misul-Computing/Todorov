"""2-bit attack #1: block-wise fine-tuning on top of 2-bit E8 RVQ.

Workstream 7 proved FT recovers 2-bit VQ from 134 -> 48 PPL (2.8x).
2-bit E8 starts at 27.7 (already 5x better than VQ's 134), so FT should
land ~10-15 PPL. Uses the existing finetune.py (LayerNorm/bias recovery,
model weights frozen).
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import snapshot, restore
from eval import ppl_wt2
from e8lattice import quantize_model_e8rvq
from finetune import collect_block_targets, finetune_model


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16,
        device_map=dev, low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)

    snap = snapshot(model)
    base_ppl = ppl_wt2(model, tok, device=dev)
    print(f"baseline PPL: {base_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # collect targets from ORIGINAL model before quantizing
    print("collecting block targets from original model...", flush=True)
    targets, calib = collect_block_targets(model, tok, n_samples=24, seq_len=256, device=dev)
    print(f"  {len(targets)} blocks, {len(calib)} calib seqs ({time.time()-t0:.0f}s)", flush=True)

    # quantize 2-bit E8
    restore(model, snap)
    quantize_model_e8rvq(model, total_bits=2, group_size=128, verbose=False)
    pre_ppl = ppl_wt2(model, tok, device=dev)
    print(f"2-bit E8 pre-FT PPL: {pre_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # fine-tune
    for epochs in [3, 7]:
        print(f"\n=== FT {epochs} epochs ===", flush=True)
        restore(model, snap)
        quantize_model_e8rvq(model, total_bits=2, group_size=128, verbose=False)
        finetune_model(model, tok, targets, calib, n_epochs=epochs, lr=3e-4, device=dev)
        post_ppl = ppl_wt2(model, tok, device=dev)
        degr = (post_ppl - base_ppl) / base_ppl
        print(f"  2-bit E8 + FT({epochs}) PPL: {post_ppl:.4f} degr={degr:+.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out_dir = "/kaggle/working/runs/2bit_ft" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_ft")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "pre_ft_ppl": pre_ppl,
                   "post_ft_ppl": post_ppl}, f, indent=2)
    print(f"saved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
