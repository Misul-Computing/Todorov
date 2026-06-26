"""Model-size-adaptive quantization with improved data-free sensitivity.

The previous data-free picker used recon error * weight norm, which was
worse than uniform. The issue: without activations, we can't know which
tensors matter for output.

New approach: use layer depth as a proxy for sensitivity. Later layers
are more sensitive (closer to output). Also use weight magnitude ratio
between layers as a secondary signal. This is still data-free but uses
structural information about the model.

Also test: does simply using a higher floor (4-bit minimum) help?
The picker with floor=3 gave some tensors 2-3 bit which is catastrophic.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sys
import time
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant.quantize import quant_dequant, snapshot, restore
from quant.eval import ppl_wt2
from quant.picker import assign_budget, apply_assignment, assignment_stats
from quant.codebook import lloyd_max_gaussian


MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct",   "3B"),
]
TARGETS = [3.5, 4.0, 4.5, 5.0]
DEGRADATION_TARGET = 0.05
BITS_GRID = (3, 4, 5, 6, 16)
GROUP_SIZE = 128


def data_free_sensitivity_v2(model, bits_grid=BITS_GRID, method="fullrot_whlm"):
    """Improved data-free sensitivity using layer depth + recon error.

    Sensitivity = recon_error * weight_norm * depth_factor

    depth_factor: later layers get higher weight (closer to output).
    This is a structural prior, no calibration data needed.
    """
    device = next(model.parameters()).device
    centroids_cache = {}
    table = {}
    numel_map = {}

    # Count layers for depth normalization
    n_layers = sum(1 for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and "self_attn" in n)
    n_layers = max(1, n_layers)

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data
        numel_map[name] = W.numel()
        table[name] = {}
        Wf = W.float()
        Wnorm = Wf.norm().clamp(min=1e-12).item()

        # Depth factor: extract layer number from name
        depth = 1.0
        if "layers." in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                total_layers = max(1, n_layers // 7)  # approximate
                depth = 1.0 + 0.5 * (layer_num / max(1, total_layers))
            except (ValueError, IndexError):
                pass

        # lm_head is very sensitive
        if "lm_head" in name or "embed" in name:
            depth = 3.0

        for b in bits_grid:
            if b >= 16:
                table[name][b] = 0.0
                continue
            if b not in centroids_cache:
                centroids_cache[b] = lloyd_max_gaussian(b, device)
            Wq, err, _, _ = quant_dequant(W, b, GROUP_SIZE, method,
                                          centroids=centroids_cache[b])
            table[name][b] = err * Wnorm * depth
    return table, numel_map


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    all_results = []

    for model_id, model_label in MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_label}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map=dev).eval()
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            all_results.append({"model": model_label, "error": str(e), "sweep": []})
            continue

        snap = snapshot(model)
        base_ppl = ppl_wt2(model, tok, device=dev)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params/1e6:.1f}M  fp16_ppl={base_ppl:.4f}", flush=True)

        # Sensitivity table
        print("  computing sensitivity v2...", flush=True)
        tq = time.time()
        table, numel_map = data_free_sensitivity_v2(model)
        print(f"  done in {time.time() - tq:.1f}s", flush=True)

        sweep = []
        for target_bpw in TARGETS:
            restore(model, snap)
            tq = time.time()
            assignment = assign_budget(table, numel_map, target_bpw,
                                        bits_grid=BITS_GRID,
                                        group_size=GROUP_SIZE, floor=3)
            avg_bpw, hist = assignment_stats(assignment, numel_map, GROUP_SIZE)
            apply_assignment(model, assignment, GROUP_SIZE,
                                   "fullrot_whlm", verbose=False)
            ppl = ppl_wt2(model, tok, device=dev)
            degradation = (ppl - base_ppl) / base_ppl
            within = degradation < DEGRADATION_TARGET
            print(f"  picker@{target_bpw}: ppl={ppl:.4f}  bpw={avg_bpw:.3f}  "
                  f"degr={degradation:+.4f} ({'OK' if within else 'FAIL'})  "
                  f"hist={hist}  ({time.time() - tq:.1f}s)", flush=True)
            sweep.append({"target_bpw": target_bpw, "actual_bpw": avg_bpw,
                          "ppl": ppl, "degradation": degradation,
                          "within_target": within, "hist": hist})

        ok = [s for s in sweep if s["within_target"]]
        min_bpw = min(s["actual_bpw"] for s in ok) if ok else None
        print(f"  -> min BPW within 5%: {min_bpw:.3f}" if min_bpw
              else "  -> no config within 5%", flush=True)

        all_results.append({"model": model_label, "n_params": n_params,
                            "fp16_ppl": base_ppl,
                            "picker_min_bpw": min_bpw, "sweep": sweep})
        del model, snap, table
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*90}", flush=True)
    print(f"{'Model':>8s} {'Params':>10s} {'FP16':>8s} {'Picker':>10s} {'@PPL':>10s} {'Compress':>10s}", flush=True)
    print(f"{'-'*90}", flush=True)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:>8s} FAILED", flush=True)
            continue
        min_bpw = r.get("picker_min_bpw")
        if min_bpw:
            entry = min((s for s in r["sweep"] if s["within_target"]),
                        key=lambda s: s["actual_bpw"])
            compress = 16.0 / min_bpw
            print(f"{r['model']:>8s} {r['n_params']/1e6:>9.1f}M {r['fp16_ppl']:>8.4f} "
                  f"{min_bpw:>10.3f} {entry['ppl']:>10.4f} {compress:>9.1f}x", flush=True)
    print(f"\ntotal: {time.time() - t0:.1f}s", flush=True)

    out_dir = "/kaggle/working/runs/adaptive" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "adaptive")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": "Adaptive v2 (depth-weighted sensitivity)",
                   "results": all_results}, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
