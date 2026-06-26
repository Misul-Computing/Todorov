"""E8 picker with {2,3,4,5,6} bit granularity.

The original picker used scalar quantization and {2,4,6} E8 granularity was
too coarse to beat uniform 4-bit. Now with odd-bit stages (3=2+1, 5=4+1),
the picker has 5 bit levels to smooth allocations.

Sensitivity metric: Hessian-diagonal-weighted reconstruction error, computed
using E8 RVQ at each candidate bit-width. The picker greedily assigns bits
per tensor under a total-bit budget.

Sweep: target bpw from 2.5 to 4.5, comparing:
- Picker (sensitivity-based allocation)
- Uniform E8 at the nearest achievable bpw
- Random assignment (control)
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import snapshot, restore
from eval import ppl_wt2
from e8lattice import quant_dequant_e8rvq
from picker import capture_activations, assign_budget, random_assignment, assignment_stats

GROUP = 128
BITS_GRID = (2, 3, 4, 5, 6)


def sensitivity_table_e8(model, activations, bits_grid=BITS_GRID, group_size=GROUP):
    """Compute S_l(b) for every Linear at every candidate bit-width using E8 RVQ.
    Returns {name: {bits: sensitivity}} and {name: numel}.
    """
    device = next(model.parameters()).device
    table = {}
    numel_map = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data
        numel_map[name] = W.numel()
        h = activations.get(name)
        if h is None:
            continue
        h = h.to(device).float()
        table[name] = {}
        Wf = W.float()
        for b in bits_grid:
            with torch.no_grad():
                Wq, _, _, _ = quant_dequant_e8rvq(W, b, group_size)
            diff = (Wf - Wq.float())
            col_sq = (diff * diff).sum(dim=0)
            table[name][b] = float((col_sq * h).sum().item())
    return table, numel_map


def apply_assignment_e8(model, assignment, group_size=GROUP, verbose=False):
    """Quantize each Linear at its assigned E8 bit-width."""
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        b = assignment.get(name, 6)
        with torch.no_grad():
            Wq, err, bpw, _ = quant_dequant_e8rvq(mod.weight.data, b, group_size)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "bits": b, "bpw": bpw, "recon_rel_err": err})
        if verbose:
            print(f"    {name}: {b}bit err={err:.4f}", flush=True)
    return log


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

    # Capture activations for sensitivity
    print("capturing activations...", flush=True)
    activations = capture_activations(model, tok, n_examples=32, max_len=512, device=dev)
    print(f"  {len(activations)} tensors ({time.time()-t0:.0f}s)", flush=True)

    # Build sensitivity table with E8
    print("building E8 sensitivity table...", flush=True)
    table, numel_map = sensitivity_table_e8(model, activations, bits_grid=BITS_GRID)
    print(f"  {len(table)} entries ({time.time()-t0:.0f}s)", flush=True)

    results = []

    # Uniform baselines at each bit-width
    print(f"\n=== uniform E8 baselines ===", flush=True)
    for bits in BITS_GRID:
        restore(model, snap)
        with torch.no_grad():
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear):
                    Wq, _, _, _ = quant_dequant_e8rvq(mod.weight.data, bits, GROUP)
                    mod.weight.data.copy_(Wq)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        bpw = bits + 16.0 / GROUP
        results.append({"cfg": f"uniform_e8_{bits}bit", "bpw": bpw,
                        "ppl": ppl, "degr": degr, "type": "uniform"})
        print(f"  uniform E8 {bits}bit: bpw={bpw:.3f} ppl={ppl:.3f} degr={degr:+.4f}", flush=True)

    # Picker sweep
    print(f"\n=== picker sweep (bits_grid={BITS_GRID}) ===", flush=True)
    for target_bpw in [2.5, 3.0, 3.125, 3.5, 4.0, 4.125, 4.5]:
        # Picker
        restore(model, snap)
        assignment = assign_budget(table, numel_map, target_bpw,
                                   bits_grid=BITS_GRID, group_size=GROUP, floor=2)
        avg_bpw, hist = assignment_stats(assignment, numel_map, GROUP)
        apply_assignment_e8(model, assignment, GROUP, verbose=False)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"picker_{target_bpw:.3f}", "target_bpw": target_bpw,
                        "actual_bpw": avg_bpw, "ppl": ppl, "degr": degr,
                        "hist": {str(k): v for k, v in hist.items()}, "type": "picker"})
        print(f"  picker @ {target_bpw:.3f}: actual_bpw={avg_bpw:.3f} ppl={ppl:.3f} degr={degr:+.4f} hist={hist}", flush=True)

        # Random control (same budget)
        restore(model, snap)
        rand_assign = random_assignment(table, numel_map, target_bpw,
                                        bits_grid=BITS_GRID, group_size=GROUP, seed=42)
        rand_bpw, rand_hist = assignment_stats(rand_assign, numel_map, GROUP)
        apply_assignment_e8(model, rand_assign, GROUP, verbose=False)
        rand_ppl = ppl_wt2(model, tok, device=dev)
        rand_degr = (rand_ppl - base_ppl) / base_ppl
        results.append({"cfg": f"random_{target_bpw:.3f}", "target_bpw": target_bpw,
                        "actual_bpw": rand_bpw, "ppl": rand_ppl, "degr": rand_degr,
                        "hist": {str(k): v for k, v in rand_hist.items()}, "type": "random"})
        print(f"  random @ {target_bpw:.3f}: actual_bpw={rand_bpw:.3f} ppl={rand_ppl:.3f} degr={rand_degr:+.4f}", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print(f"{'cfg':>24s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s} {'type':>8s}", flush=True)
    print(f"{'-'*80}", flush=True)
    for r in results:
        bpw = r.get("actual_bpw", r.get("bpw", 0))
        print(f"{r['cfg']:>24s} {bpw:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f} {r['type']:>8s}", flush=True)

    # Picker vs uniform at matched bpw
    print(f"\n=== picker vs uniform at matched bpw ===", flush=True)
    uniform_results = {r["bpw"]: r for r in results if r["type"] == "uniform"}
    picker_results = [r for r in results if r["type"] == "picker"]
    for pr in picker_results:
        # find closest uniform
        closest = min(uniform_results.values(),
                      key=lambda u: abs(u["bpw"] - pr["actual_bpw"]))
        delta = pr["ppl"] - closest["ppl"]
        ratio = pr["ppl"] / closest["ppl"]
        print(f"  picker {pr['actual_bpw']:.3f}bpw={pr['ppl']:.3f} vs "
              f"uniform {closest['bpw']:.3f}bpw={closest['ppl']:.3f} "
              f"delta={delta:+.3f} ratio={ratio:.3f}", flush=True)

    out_dir = "/root/novelquant/runs/picker_e8" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "picker_e8")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
