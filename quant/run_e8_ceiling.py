"""Max out the PTQ ceiling: E8 RVQ Pareto sweep + E8 RVQ combined with the
per-tensor bit-width picker.

Two untested levers:
  1. E8 RVQ at 2-bit was never PPL-tested (only recon error). Scalar 2-bit is
     417 PPL; E8's shaping gain may do far better.
  2. The picker (workstream 2) was never combined with E8 RVQ. Per-tensor
     allocation across {2,4,6,16}-bit E8 could push avg bpw below the 4.125
     uniform floor while holding <5% degradation.

Outputs the E8 Pareto curve (bpw vs PPL degradation) and the picker-vs-random
comparison at matched avg bpw. The knee of the Pareto curve is the honest
PTQ ceiling on this model.
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
from picker import capture_activations

BITS_GRID = (2, 3, 4, 5, 6, 16)
GROUP = 128


def bpw_of(b):
    return 16.0 if b >= 16 else b + 16.0 / GROUP


def e8_sensitivity_table(model, activations, bits_grid=BITS_GRID, group=GROUP):
    """S_l(b) for every Linear at every candidate E8 bit-width.
    Hessian-diagonal-weighted reconstruction error (same metric as picker.py).
    """
    device = next(model.parameters()).device
    table, numel = {}, {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data
        numel[name] = W.numel()
        h = activations.get(name)
        if h is None:
            continue
        h = h.to(device).float()
        Wf = W.float()
        table[name] = {}
        for b in bits_grid:
            if b >= 16:
                table[name][b] = 0.0
                continue
            Wq, _, _, _ = quant_dequant_e8rvq(W, total_bits=b, group_size=group)
            diff = Wf - Wq.float()
            col_sq = (diff * diff).sum(dim=0)
            table[name][b] = float((col_sq * h).sum().item())
    return table, numel


def assign_budget(table, numel, target_avg_bpw, bits_grid=BITS_GRID, group=GROUP, floor=2):
    """Greedy multiple-choice knapsack (mirrors picker.assign_budget)."""
    min_b = max(floor, min(b for b in bits_grid if b < 16))
    total_n = sum(numel.values())
    budget_bits = target_avg_bpw * total_n
    assignment = {n: min_b for n in table}
    floor_bits = sum(numel[n] * bpw_of(min_b) for n in assignment)
    if floor_bits > budget_bits:
        assignment = {n: min(b for b in bits_grid if b < 16) for n in table}

    def total_bits():
        return sum(numel[n] * bpw_of(assignment[n]) for n in assignment)

    def reduction(n, b):
        return table[n][assignment[n]] - table[n][b]

    while total_bits() < budget_bits:
        best, best_ratio = None, 0.0
        for n in assignment:
            cur = assignment[n]
            if cur >= 16:
                continue
            cands = [b for b in bits_grid if b > cur]
            for b in cands:
                red = reduction(n, b)
                extra = numel[n] * (bpw_of(b) - bpw_of(cur))
                if extra <= 0:
                    continue
                ratio = red / extra
                if ratio > best_ratio and total_bits() + extra <= budget_bits * 1.02:
                    best_ratio, best = ratio, (n, b)
        if best is None:
            break
        assignment[best[0]] = best[1]
    return assignment


def random_assignment(table, numel, target_avg_bpw, bits_grid=BITS_GRID, group=GROUP, seed=0):
    import random
    rng = random.Random(seed)
    min_b = min(b for b in bits_grid if b < 16)
    total_n = sum(numel.values())
    budget_bits = target_avg_bpw * total_n
    assignment = {n: min_b for n in table}

    def total_bits():
        return sum(numel[n] * bpw_of(assignment[n]) for n in assignment)

    names = list(table.keys())
    while total_bits() < budget_bits:
        rng.shuffle(names)
        up = False
        for n in names:
            cur = assignment[n]
            if cur >= 16:
                continue
            cands = [b for b in bits_grid if b > cur]
            if not cands:
                continue
            b = rng.choice(cands)
            extra = numel[n] * (bpw_of(b) - bpw_of(cur))
            if total_bits() + extra <= budget_bits * 1.02:
                assignment[n] = b
                up = True
                break
        if not up:
            break
    return assignment


def apply_e8_assignment(model, assignment, group=GROUP, verbose=False):
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        b = assignment.get(name, 16)
        if b >= 16:
            log.append({"name": name, "bits": 16, "bpw": 16.0, "recon_rel_err": 0.0})
            continue
        with torch.no_grad():
            Wq, err, bpw, _ = quant_dequant_e8rvq(mod.weight.data, total_bits=b,
                                                  group_size=group)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "bits": b, "bpw": bpw, "recon_rel_err": err})
    return log


def assignment_avg_bpw(assignment, numel):
    total_n = sum(numel.values())
    return sum(numel[n] * bpw_of(assignment[n]) for n in assignment) / total_n


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
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers loaded ({time.time()-t0:.0f}s)", flush=True)

    snap = snapshot(model)
    base_ppl = ppl_wt2(model, tok, device=dev)
    print(f"  baseline PPL: {base_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = []

    # --- 1. Uniform E8 RVQ sweep (Pareto curve) ---
    print(f"\n=== uniform E8 RVQ sweep ===", flush=True)
    for b in [2, 3, 4, 5, 6]:
        restore(model, snap)
        tq = time.time()
        from e8lattice import quantize_model_e8rvq
        quantize_model_e8rvq(model, total_bits=b, group_size=GROUP, verbose=False)
        ppl = ppl_wt2(model, tok, device=dev)
        bpw = bpw_of(b)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"uniform_e8_{b}bit", "bpw": bpw, "ppl": ppl, "degr": degr})
        print(f"  uniform E8 {b}bit: ppl={ppl:.3f} bpw={bpw:.3f} degr={degr:+.4f} "
              f"({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    # --- 2. Capture activations for picker ---
    print(f"\n=== capturing activations for picker ===", flush=True)
    restore(model, snap)
    activations = capture_activations(model, tok, n_examples=32, max_len=512, device=dev)
    print(f"  captured {len(activations)} tensors ({time.time()-t0:.0f}s)", flush=True)

    print(f"  building E8 sensitivity table...", flush=True)
    table, numel = e8_sensitivity_table(model, activations)
    print(f"  sensitivity table built ({time.time()-t0:.0f}s)", flush=True)

    # --- 3. E8 + picker at several target avg bpw ---
    print(f"\n=== E8 + picker (sensitivity) ===", flush=True)
    for target in [3.0, 3.5, 4.0, 4.5]:
        restore(model, snap)
        asg = assign_budget(table, numel, target_avg_bpw=target, floor=2)
        avg = assignment_avg_bpw(asg, numel)
        apply_e8_assignment(model, asg)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        hist = {}
        for n, b in asg.items():
            hist[b] = hist.get(b, 0) + 1
        results.append({"cfg": f"picker_e8@{target}", "target_bpw": target,
                        "bpw": avg, "ppl": ppl, "degr": degr, "hist": hist})
        print(f"  picker @ {target}: avg_bpw={avg:.3f} ppl={ppl:.3f} degr={degr:+.4f} "
              f"hist={hist} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    # --- 4. Random control at matched avg bpw ---
    print(f"\n=== E8 + random control ===", flush=True)
    # match the picker@3.5 avg bpw
    restore(model, snap)
    asg_rand = random_assignment(table, numel, target_avg_bpw=3.5, seed=0)
    avg_rand = assignment_avg_bpw(asg_rand, numel)
    apply_e8_assignment(model, asg_rand)
    ppl_rand = ppl_wt2(model, tok, device=dev)
    degr_rand = (ppl_rand - base_ppl) / base_ppl
    results.append({"cfg": "random_e8@3.5", "bpw": avg_rand, "ppl": ppl_rand,
                    "degr": degr_rand})
    print(f"  random @ 3.5: avg_bpw={avg_rand:.3f} ppl={ppl_rand:.3f} "
          f"degr={degr_rand:+.4f} ({time.time()-t0:.0f}s)", flush=True)

    restore(model, snap)

    # --- summary ---
    print(f"\n{'='*80}", flush=True)
    print(f"{'cfg':>22s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s} {'<5%':>5s}", flush=True)
    print(f"{'-'*80}", flush=True)
    for r in results:
        ok = "YES" if r["degr"] < 0.05 else "no"
        print(f"{r['cfg']:>22s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} "
              f"{r['degr']:>+8.4f} {ok:>5s}", flush=True)

    out_dir = "/kaggle/working/runs/e8_ceiling" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "e8_ceiling")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"model": "Qwen2.5-1.5B-Instruct", "baseline_ppl": base_ppl,
                   "results": results}, f, indent=2)
    print(f"\nsaved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
