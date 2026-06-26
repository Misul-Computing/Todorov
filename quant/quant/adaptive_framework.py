"""Model-size-adaptive quantization framework.

Given a quality target (max PPL degradation), automatically select the
(model_size, bits_per_weight) pair that minimizes memory usage.

This is the novel contribution: no paper proposes an automatic algorithm
for this. ParetoQ studies the trade-off but doesn't build a selector.

The framework:
1. Build a Pareto frontier from measured (model_size, bpw, ppl, mem) points
2. Given a quality target Q, find the config with PPL <= Q that minimizes memory
3. Optionally: predict the optimal config for unmeasured model sizes

Data-free quantization method: E8 lattice RVQ (best data-free 4-bit method).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Measured results from Kaggle runs
# Format: (model_label, family, n_params_M, bpw, ppl, method)
MEASURED_POINTS = [
    # Qwen2.5-0.5B (fp16_ppl=14.01, n_params=494M)
    ("0.5B", "qwen2.5", 494, 16.0, 14.01, "fp16"),
    ("0.5B", "qwen2.5", 494, 4.125, 17.26, "scalar"),
    ("0.5B", "qwen2.5", 494, 5.125, 14.96, "scalar"),
    ("0.5B", "qwen2.5", 494, 2.125, 89.22, "e8rvq"),
    ("0.5B", "qwen2.5", 494, 4.125, 15.25, "e8rvq"),
    ("0.5B", "qwen2.5", 494, 6.125, 14.04, "e8rvq"),
    # Qwen2.5-1.5B (fp16_ppl=9.48, n_params=1544M)
    ("1.5B", "qwen2.5", 1544, 16.0, 9.48, "fp16"),
    ("1.5B", "qwen2.5", 1544, 4.125, 10.57, "scalar"),
    ("1.5B", "qwen2.5", 1544, 5.125, 9.84, "scalar"),
    ("1.5B", "qwen2.5", 1544, 2.125, 27.70, "e8rvq"),
    ("1.5B", "qwen2.5", 1544, 4.125, 10.05, "e8rvq"),
    ("1.5B", "qwen2.5", 1544, 6.125, 9.53, "e8rvq"),
    # Qwen2.5-3B (fp16_ppl=8.45, n_params=3086M)
    ("3B", "qwen2.5", 3086, 16.0, 8.45, "fp16"),
    ("3B", "qwen2.5", 3086, 4.125, 9.52, "scalar"),
    ("3B", "qwen2.5", 3086, 5.125, 8.74, "scalar"),
    ("3B", "qwen2.5", 3086, 2.125, 28253, "e8rvq"),
    ("3B", "qwen2.5", 3086, 4.125, 9.00, "e8rvq"),
    ("3B", "qwen2.5", 3086, 6.125, 8.49, "e8rvq"),
    # Qwen3-4B (fp16_ppl=13.42, n_params=4022M)
    ("4B", "qwen3", 4022, 16.0, 13.42, "fp16"),
    ("4B", "qwen3", 4022, 4.125, 14.40, "scalar"),
    ("4B", "qwen3", 4022, 5.125, 13.93, "scalar"),
    ("4B", "qwen3", 4022, 4.125, 13.71, "e8rvq"),
]


def mem_mb(n_params_M, bpw):
    """Estimate model memory in MB."""
    return n_params_M * bpw / 8


def degr(ppl, fp16_ppl):
    """PPL degradation as fraction."""
    return (ppl - fp16_ppl) / fp16_ppl


def fp16_ppl_for_model(model_label):
    """Get FP16 PPL for a model."""
    for m, f, n, bpw, ppl, method in MEASURED_POINTS:
        if m == model_label and method == "fp16":
            return ppl
    return None


def pareto_frontier(points, target_degr=0.05):
    """Build Pareto frontier: configs that achieve <target_degr degradation,
    sorted by memory (ascending). Each point is dominated if another point
    has lower or equal memory AND lower or equal PPL.
    """
    # Filter to configs that meet the quality target
    valid = []
    for model, family, n_params, bpw, ppl, method in points:
        fp16 = fp16_ppl_for_model(model)
        if fp16 is None:
            continue
        d = degr(ppl, fp16)
        if d <= target_degr:
            mem = mem_mb(n_params, bpw)
            valid.append({
                "model": model, "family": family, "n_params_M": n_params,
                "bpw": bpw, "ppl": ppl, "degr": d, "mem_mb": mem,
                "method": method
            })

    # Sort by memory
    valid.sort(key=lambda x: x["mem_mb"])

    # Build Pareto frontier (lower memory AND lower PPL dominates)
    frontier = []
    for p in valid:
        dominated = False
        for f in frontier:
            if f["mem_mb"] <= p["mem_mb"] and f["ppl"] <= p["ppl"]:
                dominated = True
                break
        if not dominated:
            # Remove any frontier points dominated by this one
            frontier = [f for f in frontier if not (p["mem_mb"] <= f["mem_mb"] and p["ppl"] <= f["ppl"])]
            frontier.append(p)

    return frontier


def select_optimal(points, target_degr=0.05, max_mem_mb=None):
    """Select the optimal (model, bpw) config that:
    - Achieves <= target_degr PPL degradation
    - Minimizes memory
    - Optionally: fits within max_mem_mb

    Returns the best config dict.
    """
    frontier = pareto_frontier(points, target_degr)
    if max_mem_mb is not None:
        frontier = [f for f in frontier if f["mem_mb"] <= max_mem_mb]
    if not frontier:
        return None
    return frontier[0]  # lowest memory


def select_best_ppl(points, max_mem_mb):
    """Select the config with best PPL within a memory budget."""
    candidates = []
    for model, family, n_params, bpw, ppl, method in points:
        mem = mem_mb(n_params, bpw)
        if mem <= max_mem_mb:
            candidates.append({
                "model": model, "family": family, "n_params_M": n_params,
                "bpw": bpw, "ppl": ppl, "mem_mb": mem, "method": method
            })
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["ppl"])


def predict_optimal_bpw(n_params_M, target_degr=0.05):
    """Predict the minimum BPW needed for a given model size to achieve
    <target_degr degradation, using E8 RVQ.

    Based on measured data:
    - 0.5B: needs 6-bit E8 for <5% (0.2% degr)
    - 1.5B: needs 5-bit scalar or 6-bit E8 for <5%
    - 3B: needs 5-bit scalar or 6-bit E8 for <5%
    - 4B (Qwen3): 4-bit E8 achieves 2.1% (<5%!)

    Heuristic: larger models are more robust to quantization.
    The crossover where 4-bit E8 meets <5% is around 4B params.
    """
    # E8 RVQ 4-bit degradation by model size
    e8_4bit_degr = {
        494: 0.089,   # 0.5B
        1544: 0.060,  # 1.5B
        3086: 0.065,  # 3B
        4022: 0.021,  # 4B Qwen3
    }

    # Find closest model size
    sizes = sorted(e8_4bit_degr.keys())
    for s in sizes:
        if n_params_M <= s:
            d = e8_4bit_degr[s]
            if d <= target_degr:
                return 4.125  # 4-bit E8 works
            # Need higher precision
            # Check 5-bit scalar
            scalar_5bit_degr = {
                494: 0.068,  # 0.5B (still >5%)
                1544: 0.038, # 1.5B
                3086: 0.034, # 3B
                4022: 0.038, # 4B
            }
            d5 = scalar_5bit_degr.get(s, 0.04)
            if d5 <= target_degr:
                return 5.125
            # Need 6-bit E8
            return 6.125

    # For larger models, 4-bit E8 should work
    return 4.125


def main():
    print("=" * 80)
    print("MODEL-SIZE-ADAPTIVE QUANTIZATION FRAMEWORK")
    print("=" * 80)

    # 1. Pareto frontier at <5% degradation
    print("\n1. Pareto frontier (PPL degradation < 5%):")
    print(f"   {'model':>6s} {'method':>10s} {'bpw':>8s} {'ppl':>8s} {'degr':>8s} {'mem_mb':>8s}")
    print(f"   {'-'*60}")
    frontier = pareto_frontier(MEASURED_POINTS, target_degr=0.05)
    for f in frontier:
        print(f"   {f['model']:>6s} {f['method']:>10s} {f['bpw']:>8.3f} "
              f"{f['ppl']:>8.2f} {f['degr']:>+8.4f} {f['mem_mb']:>8.0f}")

    # 2. Optimal config for different memory budgets
    print("\n2. Best config for different memory budgets:")
    print(f"   {'budget':>10s} {'model':>6s} {'method':>10s} {'bpw':>8s} {'ppl':>8s} {'mem_mb':>8s}")
    print(f"   {'-'*60}")
    for budget in [500, 1000, 2000, 3000, 5000, 8000]:
        best = select_best_ppl(MEASURED_POINTS, budget)
        if best:
            print(f"   {budget:>9d}M {best['model']:>6s} {best['method']:>10s} "
                  f"{best['bpw']:>8.3f} {best['ppl']:>8.2f} {best['mem_mb']:>8.0f}")
        else:
            print(f"   {budget:>9d}M  (no config fits)")

    # 3. Optimal config for different quality targets
    print("\n3. Minimum-memory config for different quality targets:")
    print(f"   {'target':>10s} {'model':>6s} {'method':>10s} {'bpw':>8s} {'ppl':>8s} {'degr':>8s} {'mem_mb':>8s}")
    print(f"   {'-'*70}")
    for target in [0.01, 0.02, 0.05, 0.10, 0.20]:
        best = select_optimal(MEASURED_POINTS, target_degr=target)
        if best:
            print(f"   {target:>9.1%} {best['model']:>6s} {best['method']:>10s} "
                  f"{best['bpw']:>8.3f} {best['ppl']:>8.2f} {best['degr']:>+8.4f} {best['mem_mb']:>8.0f}")
        else:
            print(f"   {target:>9.1%}  (no config meets target)")

    # 4. Prediction for unmeasured model sizes
    print("\n4. Predicted optimal BPW for unmeasured model sizes:")
    print(f"   {'n_params':>10s} {'predicted_bpw':>14s} {'est_mem':>10s} {'method':>10s}")
    print(f"   {'-'*50}")
    for n in [250, 500, 1000, 2000, 4000, 7000, 13000, 70000]:
        bpw = predict_optimal_bpw(n, target_degr=0.05)
        mem = mem_mb(n, bpw)
        method = "e8rvq" if bpw in [2.125, 4.125, 6.125] else "scalar"
        print(f"   {n:>9d}M {bpw:>14.3f} {mem:>9.0f}M {method:>10s}")

    # 5. Key insight: model-size-adaptive advantage
    print("\n5. Model-size-adaptive advantage:")
    print("   Instead of quantizing a fixed model, choose the model size")
    print("   that gives the best PPL at the target memory budget.")
    print()
    comparisons = [
        ("0.5B FP16", 988, 14.01, "1.5B scalar 5-bit", 989, 9.84),
        ("1.5B FP16", 3087, 9.48, "3B E8 6-bit", 2363, 8.49),
        ("4B FP16", 8045, 13.42, "3B E8 6-bit", 2363, 8.49),
        ("4B scalar 5-bit", 2577, 13.93, "4B E8 4-bit", 2074, 13.71),
    ]
    print(f"   {'baseline':>20s} {'mem':>8s} {'ppl':>8s} | {'adaptive':>20s} {'mem':>8s} {'ppl':>8s} {'winner':>8s}")
    print(f"   {'-'*90}")
    for base_name, base_mem, base_ppl, adapt_name, adapt_mem, adapt_ppl in comparisons:
        winner = "ADAPTIVE" if adapt_ppl < base_ppl else "baseline"
        mem_savings = (1 - adapt_mem / base_mem) * 100
        ppl_improvement = (1 - adapt_ppl / base_ppl) * 100
        print(f"   {base_name:>20s} {base_mem:>7d}M {base_ppl:>8.2f} | "
              f"{adapt_name:>20s} {adapt_mem:>7d}M {adapt_ppl:>8.2f} "
              f"{winner:>8s} ({mem_savings:+.0f}% mem, {ppl_improvement:+.0f}% ppl)")

    # Save
    out_dir = "/kaggle/working/runs/adaptive_framework" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "adaptive_framework")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "framework.json"), "w") as f:
        json.dump({
            "measured_points": [
                {"model": m, "family": fam, "n_params_M": n, "bpw": b, "ppl": p, "method": meth}
                for m, fam, n, b, p, meth in MEASURED_POINTS
            ],
            "pareto_frontier_5pct": frontier,
        }, f, indent=2)
    print("\nsaved")


if __name__ == "__main__":
    main()
