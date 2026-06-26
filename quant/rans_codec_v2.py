"""Iteration 4: confirm rANS hits entropy with a REPRESENTATIVE sample.

Iterations 2-3 measured 2.81 b/exp on the first-8M exponents (early layers) and
wrongly compared it to the global 2.608 entropy. This:
  - computes global / representative / first-8M exponent entropies,
  - rANS-encodes a uniform random sample across all tensors (representative) AND
    the first-8M block, comparing each coder rate to that sample's OWN entropy.
If rANS rate == sample entropy in both cases, the coder is correct and the
representative rate is the real lossless ratio.
"""
import os, sys, subprocess, time, json
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "constriction"], check=False)
import numpy as np
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM
import constriction

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FRAC = 0.04
rng = np.random.default_rng(0)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval()
print(f"loaded {MODEL_ID} in {time.time()-t0:.1f}s", flush=True)

cnt = np.zeros(256, dtype=np.int64)
total_w, got_first = 0, 0
repr_s, first_s = [], []
for n, m in model.named_modules():
    if not isinstance(m, nn.Linear):
        continue
    u16 = m.weight.data.contiguous().view(torch.int16).numpy().view(np.uint16).astype(np.int32)
    e = ((u16 >> 7) & 0xFF).ravel().astype(np.int32)
    total_w += e.size
    cnt += np.bincount(e, minlength=256)
    repr_s.append(e[rng.random(e.size) < FRAC])          # uniform across all tensors
    if got_first < 8_000_000:
        t = min(8_000_000 - got_first, e.size)
        first_s.append(e[:t]); got_first += t
del model
repr_s = np.concatenate(repr_s)
first_s = np.concatenate(first_s)


def H(arr):
    c = np.bincount(arr, minlength=256).astype(np.float64); p = c / c.sum()
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def rans_rate(sym):
    probs = np.bincount(sym, minlength=256).astype(np.float64) + 1.0
    probs /= probs.sum()
    em = constriction.stream.model.Categorical(probs)
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(sym, em)
    comp = enc.get_compressed()
    dec = constriction.stream.stack.AnsCoder(comp).decode(em, sym.size)
    return comp.size * 32 / sym.size, bool(np.array_equal(dec, sym))


p = cnt / cnt.sum()
Hglob = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
Hrepr, Hfirst = H(repr_s), H(first_s)
r_repr, ok_repr = rans_rate(repr_s)
r_first, ok_first = rans_rate(first_s)
bpw = 1.0 + 7.0 + r_repr
df11 = 10.85

print(f"\nexp entropy: global={Hglob:.4f}  repr-sample={Hrepr:.4f}  first8M={Hfirst:.4f}", flush=True)
print(f"rANS rate vs that sample's entropy:")
print(f"  representative: {r_repr:.4f}  (sample H={Hrepr:.4f}, overhead {r_repr-Hrepr:+.4f}, ok={ok_repr})")
print(f"  first-8M:       {r_first:.4f}  (sample H={Hfirst:.4f}, overhead {r_first-Hfirst:+.4f}, ok={ok_first})")
print(f"\n=== verified rANS lossless coder (representative) ===")
print(f"  exponent {r_repr:.4f} b  ->  total {bpw:.3f} b/w  {16/bpw:.3f}x  (vs DFloat11 {df11/bpw:.3f}x)")
print(f"  weights={total_w/1e6:.0f}M  total={time.time()-t0:.1f}s")
print("RESULTS_JSON " + json.dumps({
    "H_global": Hglob, "H_repr": Hrepr, "H_first8M": Hfirst,
    "rans_repr": r_repr, "rans_first": r_first, "roundtrip_repr": ok_repr,
    "bpw": bpw, "ratio_vs_fp16": 16 / bpw, "ratio_vs_df11": df11 / bpw}))
