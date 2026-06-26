"""Full-dimension incoherence rotation + Lloyd-Max codebook (data-free, uniform).

The lever I skipped. A FIXED randomized Hadamard (random sign flip + full-dim
Walsh-Hadamard over the whole input dimension) is applied identically to every
weight matrix, uniform, data-free, no training, not selective. It Gaussianizes
the weights and spreads outliers across the ENTIRE dimension (vs a 128-wide block),
which is what makes low-bit quant work. Compares, on Qwen2.5-1.5B WikiText perplexity:
  RTN  vs  group-local-128 WH+LM  vs  FULL-DIM rotation + LM,  at 2 and 3 bit.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import time, json, math
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM
from datasets import load_dataset


def fwht(x):  # normalized Walsh-Hadamard along last dim (n = power of 2); self-inverse
    n = x.shape[-1]; shape = x.shape
    x = x.reshape(-1, n).clone(); h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.cat([(a + b).unsqueeze(2), (a - b).unsqueeze(2)], dim=2).reshape(-1, n)
        h *= 2
    return (x / math.sqrt(n)).reshape(shape)


def lloyd_max_gaussian(b, device, n=1_000_000, iters=40, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, generator=g).to(device); k = 2 ** b
    c = torch.distributions.Normal(0., 1.).icdf(torch.linspace(.5 / k, 1 - .5 / k, k, device=device))
    for _ in range(iters):
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        for j in range(k):
            s = x[idx == j]
            if s.numel(): c[j] = s.mean()
        c, _ = torch.sort(c)
    return c


def next_pow2(n):
    p = 1
    while p < n: p *= 2
    return p


_SIGNS = {}
def signs_for(npad, device):
    if npad not in _SIGNS:
        g = torch.Generator(device="cpu").manual_seed(1234 + npad)
        _SIGNS[npad] = (torch.randint(0, 2, (npad,), generator=g).to(device) * 2 - 1).float()
    return _SIGNS[npad]


def q_group(W, bits, cb, G=128):  # group-local-128 WH+LM (our earlier method)
    d_out, d_in = W.shape
    while d_in % G and G > 1: G //= 2
    Wf = W.float().reshape(d_out, d_in // G, G)
    H = fwht(torch.eye(G, device=W.device))  # GxG hadamard via fwht of identity
    Wr = Wf @ H
    sc = Wr.std(-1, keepdim=True).clamp(min=1e-8)
    q = cb[torch.bucketize(Wr / sc, (cb[1:] + cb[:-1]) / 2)] * sc
    return (q @ H).reshape(d_out, d_in).to(W.dtype)


def q_fullrot(W, bits, cb, chunk=8192):  # FULL-dim rotation + LM
    d_out, d_in = W.shape
    npad = next_pow2(d_in); s = signs_for(npad, W.device)
    bounds = (cb[1:] + cb[:-1]) / 2
    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wf = W[i:i + chunk].float()
        Wp = torch.zeros(Wf.shape[0], npad, device=W.device); Wp[:, :d_in] = Wf
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        q = cb[torch.bucketize(Wr / sc, bounds)] * sc
        out[i:i + chunk] = (fwht(q) * s)[:, :d_in].to(W.dtype)
    return out


def q_rtn(W, bits, G=128):  # uniform baseline
    d_out, d_in = W.shape
    while d_in % G and G > 1: G //= 2
    Wf = W.float().reshape(d_out, d_in // G, G); qmax = 2 ** (bits - 1) - 1
    sc = Wf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.clamp(torch.round(Wf / sc), -qmax - 1, qmax) * sc).reshape(d_out, d_in).to(W.dtype)


def measure_perp(model, tok, n_ex=20, max_len=512):
    torch.cuda.empty_cache()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    L, T, k = [], [], 0
    for ex in ds:
        if k >= n_ex: break
        t = ex["text"]
        if 50 < len(t) < 2000:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to("cuda")
            if ids.shape[1] < 2: continue
            with torch.no_grad(): out = model(ids, labels=ids)
            L.append(out.loss.item()); T.append(ids.shape[1]); k += 1
    return math.exp(sum(a * b for a, b in zip(L, T)) / sum(T))


def snap(m): return {n: x.weight.data.detach().to("cpu", copy=True) for n, x in m.named_modules() if isinstance(x, nn.Linear)}
def restore(m, s):
    with torch.no_grad():
        for n, x in m.named_modules():
            if isinstance(x, nn.Linear) and n in s: x.weight.data.copy_(s[n].to(x.weight.device))


if __name__ == "__main__":
    # FWHT self-test
    z = torch.randn(4, 16)
    assert torch.allclose(fwht(fwht(z)), z, atol=1e-4), "FWHT not involutive!"
    print("FWHT self-test PASS", flush=True)

    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    t0 = time.time()
    tok = AutoTokenizer = __import__("transformers").AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    s = snap(model)
    base = measure_perp(model, tok); print(f"[FP16] perp={base:.4f}", flush=True)

    res = [{"cfg": "fp16", "perp": base}]
    for bits in (3, 2):
        cb = lloyd_max_gaussian(bits, "cuda")
        for name, fn in [("rtn", lambda W: q_rtn(W, bits)),
                         ("group128_whlm", lambda W: q_group(W, bits, cb)),
                         ("fullrot_whlm", lambda W: q_fullrot(W, bits, cb))]:
            restore(model, s); tq = time.time()
            with torch.no_grad():
                for _, m in model.named_modules():
                    if isinstance(m, nn.Linear): m.weight.data.copy_(fn(m.weight.data))
            p = measure_perp(model, tok)
            print(f"[{name} b{bits}] perp={p:.4f}  ({time.time()-tq:.1f}s)", flush=True)
            res.append({"cfg": f"{name}_b{bits}", "bits": bits, "perp": p})

    print("\n=== full-dim rotation vs group-local vs RTN (Qwen2.5-1.5B, WikiText ppl) ===")
    for r in res:
        print(f"  {r['cfg']:20s} perp={r['perp']:12.4f}", flush=True)
    print(f"FP16={base:.4f}  total={time.time()-t0:.1f}s")
    print("RESULTS_JSON " + json.dumps(res))
