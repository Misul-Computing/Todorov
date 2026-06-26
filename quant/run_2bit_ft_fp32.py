"""2-bit FT with the PROVEN fp32 approach (workstream 7's finetune.py).

The fast fp16 version diverged (27.7 -> 1099 PPL). The original finetune.py
casts the whole model to fp32 and works (134 -> 48 on VQ). The RTX PRO 6000
has 96GB VRAM, so a 1.5B model in fp32 (6GB) is trivial. No need for the
fp16 shortcut that broke training.

Also runs FT on 3-bit E8 (12.66 PPL), if FT recovers 2.8x there too,
3-bit could hit ~5 PPL (under the 5% degradation target at 3.125 bpw).
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
    print(f"device: {dev}", flush=True)
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

    # collect targets from ORIGINAL model (in fp16, before any quantization)
    print("collecting block targets...", flush=True)
    targets, calib = collect_block_targets(model, tok, n_samples=24, seq_len=256, device=dev)
    print(f"  {len(targets)} blocks, {len(calib)} seqs ({time.time()-t0:.0f}s)", flush=True)

    results = []
    for bits, epoch_list in [(2, [3, 7, 15]), (3, [3, 7])]:
        print(f"\n{'='*60}", flush=True)
        print(f"=== {bits}-bit E8 + fp32 FT ===", flush=True)
        print(f"{'='*60}", flush=True)
        for epochs in epoch_list:
            restore(model, snap)
            quantize_model_e8rvq(model, total_bits=bits, group_size=128, verbose=False)
            if epochs == epoch_list[0]:
                pre_ppl = ppl_wt2(model, tok, device=dev)
                print(f"  {bits}-bit E8 pre-FT PPL: {pre_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)
            tq = time.time()
            # use the original finetune_model which casts to fp32
            finetune_model(model, tok, targets, calib, n_epochs=epochs, lr=3e-4, device=dev)
            post_ppl = ppl_wt2(model, tok, device=dev)
            degr = (post_ppl - base_ppl) / base_ppl
            results.append({"cfg": f"e8_{bits}bit_ft{epochs}", "bpw": bits + 16.0/128,
                            "pre_ft_ppl": pre_ppl if epochs == epoch_list[0] else None,
                            "ppl": post_ppl, "degr": degr})
            print(f"  {bits}-bit E8 + FT({epochs}): ppl={post_ppl:.4f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
            torch.cuda.empty_cache() if dev == "cuda" else None

    print(f"\n{'='*70}", flush=True)
    print(f"{'cfg':>22s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s}", flush=True)
    print(f"{'-'*70}", flush=True)
    for r in results:
        print(f"{r['cfg']:>22s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f}", flush=True)

    out_dir = "/root/novelquant/runs/2bit_ft_fp32" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_ft_fp32")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
