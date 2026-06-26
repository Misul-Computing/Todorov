"""Run the fast PTQ LittleBit on Qwen2.5-1.5B at 0.55, 0.3, 0.1 BPW.
Measures: SVD time, reconstruction error, perplexity, peak memory.

Goal: a single run in 5-15 minutes on T4.
"""
import os, sys, time, json, math, statistics
os.chdir("/kaggle/working/research")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Use the local copy
sys.path.insert(0, "novelquant")
from littlebit_ptq import ptq_quantize_model


def measure_perp(model, tokenizer, label, n_examples=30, max_len=512, log=print):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    losses, toks_per = [], []
    n_done = 0
    for ex in ds:
        if n_done >= n_examples: break
        t = ex["text"]
        if 50 < len(t) < 2000:
            ids = tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to("cuda")
            n_tok = ids.shape[1]
            if n_tok < 2: continue
            with torch.no_grad():
                out = model(ids, labels=ids)
            losses.append(out.loss.item())
            toks_per.append(n_tok)
            n_done += 1
    toks = sum(toks_per)
    weighted = sum(l * n for l, n in zip(losses, toks_per))
    perp = math.exp(weighted / toks) if toks else float("nan")
    peak = torch.cuda.max_memory_allocated() / 1e6
    log(f"[{label:35s}]  perp={perp:8.4f}  peak={peak:7.1f}MB  "
        f"n_examples={n_done}  n_tokens={toks}")
    return perp, peak


def run_one_bpw(MODEL_ID, eff_bit, label, n_examples=30):
    print(f"\n=== {label} (eff_bit={eff_bit}) ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    t_load = time.time() - t0
    print(f"  loaded in {t_load:.1f}s, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)
    # Baseline
    base_perp, base_peak = measure_perp(model, tok, f"BASELINE  ({label})",
                                         n_examples=n_examples, log=print)
    # PTQ LittleBit
    t_decomp = time.time()
    log = ptq_quantize_model(model, eff_bit, verbose=True)
    t_decomp = time.time() - t_decomp
    # Per-tensor summary
    bpws = [r["actual_bpw"] for r in log]
    errs = [r["recon_rel_err"] for r in log]
    avg_bpw = statistics.mean(bpws) if bpws else 0
    max_bpw = max(bpws) if bpws else 0
    min_bpw = min(bpws) if bpws else 0
    avg_err = statistics.mean(errs) if errs else 0
    max_err = max(errs) if errs else 0
    print(f"  decompose {len(log)} layers in {t_decomp:.1f}s", flush=True)
    print(f"  per-tensor BPW:    avg={avg_bpw:.3f}  min={min_bpw:.3f}  max={max_bpw:.3f}",
          flush=True)
    print(f"  per-tensor recon:  avg={avg_err:.4f}  max={max_err:.4f}",
          flush=True)
    # Save the per-tensor log
    os.makedirs(f"runs/littlebit_ptq", exist_ok=True)
    with open(f"runs/littlebit_ptq/{label}_tensors.json", "w") as f:
        json.dump({"eff_bit_target": eff_bit, "model": MODEL_ID,
                   "t_decompose_s": t_decomp, "tensors": log}, f, indent=2)
    # Quantized perplexity
    q_perp, q_peak = measure_perp(model, tok, f"LITTLEBIT  ({label})",
                                   n_examples=n_examples, log=print)
    total_t = time.time() - t0
    print(f"  TOTAL: {total_t:.1f}s (load {t_load:.1f} + decomp {t_decomp:.1f} + eval)",
          flush=True)
    return {
        "label": label, "eff_bit_target": eff_bit,
        "avg_bpw": avg_bpw, "max_recon_err": max_err,
        "base_perp": base_perp, "q_perp": q_perp,
        "perp_delta": q_perp - base_perp,
        "base_peak_mb": base_peak, "q_peak_mb": q_peak,
        "t_total_s": total_t, "n_tensors": len(log),
    }


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    N_EXAMPLES = int(os.environ.get("N_EXAMPLES", "20"))  # small for speed
    BPWS = [0.55, 0.3, 0.1]
    results = []
    for bpw in BPWS:
        r = run_one_bpw(MODEL_ID, bpw, f"qwen2.5-1.5b_{bpw}bpw", n_examples=N_EXAMPLES)
        results.append(r)
    # Print final summary
    print("\n" + "=" * 100)
    print(f"{'label':30s}  {'target BPW':>10s}  {'actual BPW':>10s}  "
          f"{'base perp':>10s}  {'q perp':>10s}  {'delta':>10s}  "
          f"{'max recon':>10s}  {'time':>8s}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:30s}  {r['eff_bit_target']:>10.2f}  "
              f"{r['avg_bpw']:>10.3f}  {r['base_perp']:>10.4f}  "
              f"{r['q_perp']:>10.4f}  {r['perp_delta']:>+10.4f}  "
              f"{r['max_recon_err']:>10.4f}  {r['t_total_s']:>7.1f}s")
    # Save
    with open("runs/littlebit_ptq/summary.json", "w") as f:
        json.dump(results, f, indent=2)
