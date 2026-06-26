"""Run WH+Lloyd-Max data-free weight quantization on Qwen2.5-1.5B.

Sweeps RTN (baseline) vs WH+Lloyd-Max at b in {4,3,2}, plus WH+uniform at b=2
to isolate the rotation gain. Measures WikiText-2 weighted perplexity vs FP16.

Goal: a single run in 5-15 minutes on a T4. Pure quantization, no training.
"""
import os
import sys
import time
import json
import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whlm_quant import quantize_model_inplace


def measure_perp(model, tokenizer, label, n_examples=25, max_len=512, log=print):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    losses, toks_per, n_done = [], [], 0
    for ex in ds:
        if n_done >= n_examples:
            break
        t = ex["text"]
        if 50 < len(t) < 2000:
            ids = tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to("cuda")
            if ids.shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(ids, labels=ids)
            losses.append(out.loss.item())
            toks_per.append(ids.shape[1])
            n_done += 1
    toks = sum(toks_per)
    weighted = sum(l * n for l, n in zip(losses, toks_per))
    perp = math.exp(weighted / toks) if toks else float("nan")
    log(f"[{label:28s}]  perp={perp:9.4f}  n_ex={n_done}  n_tok={toks}", flush=True)
    return perp


def snapshot(model):
    return {n: m.weight.data.detach().to("cpu", copy=True)
            for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def restore(model, snap):
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and n in snap:
                m.weight.data.copy_(snap[n].to(m.weight.device))


if __name__ == "__main__":
    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    N_EX = int(os.environ.get("N_EXAMPLES", "25"))
    GROUP = int(os.environ.get("GROUP", "128"))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"loaded {MODEL_ID} in {time.time() - t0:.1f}s, "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)

    snap = snapshot(model)
    base = measure_perp(model, tok, "FP16 baseline", n_examples=N_EX)

    configs = [("rtn", 4), ("whlm", 4),
               ("rtn", 3), ("whlm", 3),
               ("rtn", 2), ("whrtn", 2), ("whlm", 2)]
    results = [{"label": "fp16", "method": "none", "bits": 16, "perp": base,
                "perp_delta": 0.0, "avg_bpw": 16.0, "avg_recon": 0.0}]

    for method, bits in configs:
        label = f"{method}_b{bits}"
        print(f"\n=== {label} (group={GROUP}) ===", flush=True)
        restore(model, snap)
        tq = time.time()
        qlog = quantize_model_inplace(model, bits, group_size=GROUP, method=method)
        avg_bpw = sum(r["bpw"] for r in qlog) / len(qlog)
        avg_err = sum(r["recon_rel_err"] for r in qlog) / len(qlog)
        max_err = max(r["recon_rel_err"] for r in qlog)
        perp = measure_perp(model, tok, label, n_examples=N_EX)
        print(f"  quantize+eval {time.time() - tq:.1f}s  avg_bpw={avg_bpw:.3f}  "
              f"recon avg={avg_err:.4f} max={max_err:.4f}", flush=True)
        results.append({"label": label, "method": method, "bits": bits,
                        "perp": perp, "perp_delta": perp - base,
                        "avg_bpw": avg_bpw, "avg_recon": avg_err,
                        "max_recon": max_err})

    print("\n" + "=" * 78)
    print(f"{'config':14s} {'avg_bpw':>8s} {'perp':>10s} {'d_perp':>9s} "
          f"{'recon_avg':>9s}")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:14s} {r['avg_bpw']:>8.3f} {r['perp']:>10.4f} "
              f"{r['perp_delta']:>+9.4f} {r.get('avg_recon', 0):>9.4f}", flush=True)
    print(f"\nFP16 baseline perp = {base:.4f}   total = {time.time() - t0:.1f}s")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "whlm")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({"model": MODEL_ID, "group": GROUP, "n_examples": N_EX,
                       "results": results}, f, indent=2)
        print(f"saved {out_dir}/summary.json")
    except Exception as e:
        print(f"(could not save json: {e})")
