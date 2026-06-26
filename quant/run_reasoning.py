"""Run reasoning eval (HellaSwag, ARC-Challenge, PIQA) on FP16, picker, uniform.

Workstream 3: characterize where sub-1-bit breaks on reasoning (the BitNet
finding) and whether the picker delays the collapse relative to uniform.

Configs:
  fp16          , baseline
  picker_3.2    , picker at 3.2 avg bpw (beats uniform 3-bit on PPL)
  uniform_3bit  , uniform 3-bit (3.125 bpw)
  picker_2.5    , picker at 2.5 avg bpw (sub-3 regime)
  uniform_2bit  , uniform 2-bit (2.125 bpw, broken on PPL)
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
from quant.picker import (capture_activations, sensitivity_table,
                          assign_budget, apply_assignment, assignment_stats)
from quant.reasoning import eval_all

LIMIT = int(os.environ.get("LIMIT", "200"))


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
    print(f"loaded {MODEL_ID} in {time.time() - t0:.1f}s", flush=True)

    snap = snapshot(model)

    # capture activations + sensitivity table once (reused for all picker configs)
    acts = capture_activations(model, tok, n_examples=N_CALIB, device=dev)
    table, numel_map = sensitivity_table(model, acts, bits_grid=(2, 3, 4, 8, 16),
                                          group_size=GROUP, method="fullrot_whlm")
    print(f"[sens] table ready ({len(table)} tensors) in {time.time() - t0:.1f}s", flush=True)

    configs = [
        ("fp16", None, None),
        ("picker_3.2", "picker", {"target": 3.2, "floor": 3}),
        ("uniform_3bit", "uniform", {"bits": 3}),
        ("picker_2.5", "picker", {"target": 2.5, "floor": 2}),
        ("uniform_2bit", "uniform", {"bits": 2}),
    ]
    all_results = []

    for label, kind, params in configs:
        restore(model, snap)
        if kind is None:
            pass  # fp16
        elif kind == "uniform":
            quantize_model_inplace(model, params["bits"], GROUP, "fullrot_whlm",
                                   verbose=False)
        elif kind == "picker":
            assignment = assign_budget(table, numel_map, params["target"],
                                       bits_grid=(2, 3, 4, 8, 16), group_size=GROUP,
                                       floor=params["floor"])
            avg_bpw, hist = assignment_stats(assignment, numel_map, GROUP)
            apply_assignment(model, assignment, GROUP, "fullrot_whlm", verbose=False)
            print(f"  [{label}] avg_bpw={avg_bpw:.3f} hist={hist}", flush=True)

        tq = time.time()
        results = eval_all(model, tok, dev, limit=LIMIT)
        avg_acc = sum(r["acc"] for r in results) / len(results)
        print(f"[{label:16s}] avg_acc={avg_acc:.4f}  ({time.time() - tq:.1f}s)\n", flush=True)
        all_results.append({"cfg": label, "kind": kind, "avg_acc": avg_acc,
                            "tasks": results})

    print("\n" + "=" * 56)
    print(f"{'cfg':18s} {'avg_acc':>8s} {'hellaswag':>10s} {'arc':>8s}")
    print("-" * 56)
    for r in all_results:
        t = {x["task"]: x["acc"] for x in r["tasks"]}
        print(f"{r['cfg']:18s} {r['avg_acc']:>8.4f} "
              f"{t.get('hellaswag', 0):>10.4f} {t.get('arc_challenge', 0):>8.4f}",
              flush=True)
    print(f"\ntotal={time.time() - t0:.1f}s  limit={LIMIT}/task")

    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working/runs/reasoning"
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "reasoning")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": "reasoning eval: picker vs uniform vs fp16",
                   "model": MODEL_ID, "group": GROUP, "limit": LIMIT,
                   "tasks": ["hellaswag", "arc_challenge", "piqa"],
                   "results": all_results}, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
