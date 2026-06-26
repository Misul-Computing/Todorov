"""Run the per-tensor bit-width picker vs uniform quantization (workstream 2).

Pipeline:
  1. Load Qwen2.5-1.5B, measure FP16 baseline (canonical eval).
  2. Capture calibration activations (32 examples, one forward pass).
  3. Compute per-tensor sensitivity at {2,3,4,8,16} bit-widths.
  4. Assign bit-widths under a target avg BPW budget (greedy knapsack).
  5. Apply the picker assignment, measure PPL.
  6. Compare to uniform quantization at the same avg BPW.

Verification gate: picker PPL < uniform PPL at equal avg BPW on WikiText-2.
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
from quant.picker import (capture_activations, sensitivity_table,
                          assign_budget, random_assignment, apply_assignment,
                          assignment_stats)


def main():
    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    GROUP = int(os.environ.get("GROUP", "128"))
    N_CALIB = int(os.environ.get("N_CALIB", "32"))
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

    # --- capture activations ---
    tc = time.time()
    acts = capture_activations(model, tok, n_examples=N_CALIB, device=dev)
    print(f"[calib] captured activations for {len(acts)} Linears in {time.time() - tc:.1f}s "
          f"({N_CALIB} examples)", flush=True)

    # --- sensitivity table ---
    ts = time.time()
    table, numel_map = sensitivity_table(model, acts, bits_grid=(2, 3, 4, 8, 16),
                                          group_size=GROUP, method="fullrot_whlm")
    print(f"[sens] computed sensitivity table for {len(table)} tensors x 5 bit-widths "
          f"in {time.time() - ts:.1f}s", flush=True)

    # --- picker assignments at several target BPW ---
    # 3.2: affords 3-bit floor for all (3.125), picker redistributes above 3
    # 2.5, 2.2: sub-3 regime where picker should beat uniform-2 and random-mixed
    targets = [3.2, 2.5, 2.2]
    results = [{"cfg": "fp16", "method": "none", "bits": 16, "ppl": base,
                "ppl_delta": 0.0, "avg_bpw": 16.0}]

    for target in targets:
        # --- picker (sensitivity-based, with floor to avoid the 2-bit cliff) ---
        restore(model, snap)
        floor = 3 if target >= 3.1 else 2
        assignment = assign_budget(table, numel_map, target,
                                   bits_grid=(2, 3, 4, 8, 16), group_size=GROUP,
                                   floor=floor)
        avg_bpw, hist = assignment_stats(assignment, numel_map, GROUP)
        print(f"\n[picker @ {target} bpw floor={floor}] avg={avg_bpw:.3f}  hist={hist}", flush=True)
        tq = time.time()
        apply_assignment(model, assignment, GROUP, "fullrot_whlm", verbose=False)
        ppl = ppl_wt2(model, tok, device=dev)
        print(f"[picker @ {target} bpw] ppl={ppl:.4f}  d={ppl - base:+.4f}  "
              f"({time.time() - tq:.1f}s)", flush=True)
        results.append({"cfg": f"picker_{target}", "method": "picker", "target_bpw": target,
                        "floor": floor, "ppl": ppl, "ppl_delta": ppl - base,
                        "avg_bpw": avg_bpw,
                        "hist": {str(k): v for k, v in sorted(hist.items())}})

        # --- random-mixed control (same budget, random assignment) ---
        restore(model, snap)
        rand_assign = random_assignment(table, numel_map, target,
                                        bits_grid=(2, 3, 4, 8, 16),
                                        group_size=GROUP, seed=0)
        r_avg, r_hist = assignment_stats(rand_assign, numel_map, GROUP)
        tq = time.time()
        apply_assignment(model, rand_assign, GROUP, "fullrot_whlm", verbose=False)
        r_ppl = ppl_wt2(model, tok, device=dev)
        print(f"[random @ {target} bpw] ppl={r_ppl:.4f}  d={r_ppl - base:+.4f}  "
              f"avg={r_avg:.3f}  hist={r_hist}  ({time.time() - tq:.1f}s)", flush=True)
        results.append({"cfg": f"random_{target}", "method": "random", "target_bpw": target,
                        "ppl": r_ppl, "ppl_delta": r_ppl - base, "avg_bpw": r_avg,
                        "hist": {str(k): v for k, v in sorted(r_hist.items())}})

        # --- uniform at the closest available bit-width ---
        uniform_bits = 3 if target >= 2.6 else 2
        restore(model, snap)
        tq = time.time()
        quantize_model_inplace(model, uniform_bits, GROUP, "fullrot_whlm", verbose=False)
        u_ppl = ppl_wt2(model, tok, device=dev)
        u_bpw = uniform_bits + 16.0 / GROUP
        print(f"[uniform {uniform_bits}bit] ppl={u_ppl:.4f}  d={u_ppl - base:+.4f}  "
              f"bpw={u_bpw:.3f}  ({time.time() - tq:.1f}s)", flush=True)
        results.append({"cfg": f"uniform_{uniform_bits}bit", "method": "fullrot_whlm",
                        "bits": uniform_bits, "ppl": u_ppl, "ppl_delta": u_ppl - base,
                        "avg_bpw": u_bpw})

    print("\n" + "=" * 78)
    print(f"{'cfg':22s} {'avg_bpw':>8s} {'ppl':>10s} {'d_ppl':>9s}")
    print("-" * 78)
    for r in results:
        print(f"{r['cfg']:22s} {r['avg_bpw']:>8.3f} {r['ppl']:>10.4f} "
              f"{r['ppl_delta']:>+9.4f}", flush=True)
    print(f"\nfp16 baseline ppl={base:.4f}  total={time.time() - t0:.1f}s")

    # save
    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working/runs/picker"
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "picker")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "per-tensor bit-width picker vs uniform",
                   "model": MODEL_ID, "group": GROUP, "n_calib": N_CALIB,
                   "hardware": dev, "eval": "wikitext-2 test, 40k tokens, ctx 2048",
                   "results": results}, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
