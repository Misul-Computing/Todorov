"""Best-possible universal (data-free) 4-bit weight quant, lever exploration.

Our earlier 4-bit was bad (27.5 ppl) because of a 128-wide group. The real
levers for data-free 4-bit: codebook (uniform / NF4 / Lloyd-Max), group size,
and incoherence rotation. Sweeps them on Qwen2.5-1.5B with the STANDARD
concatenated WikiText-2 perplexity protocol (not 20 snippets), reporting true
bits/weight (4 + scale_overhead) so comparisons are fair. Target: within ~1-2%
of FP16 = genuinely near-lossless, beating NF4/Q4_K.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import time, json, math
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

NF4 = torch.tensor([-1.0, -0.6961928, -0.5250731, -0.3949175, -0.28444138, -0.18477343,
                    -0.09105004, 0.0, 0.07958030, 0.16093020, 0.24611230, 0.33791524,
                    0.44070983, 0.56261700, 0.72295684, 1.0])


def fwht(x):
    n = x.shape[-1]; shape = x.shape; x = x.reshape(-1, n).clone(); h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.cat([(a + b).unsqueeze(2), (a - b).unsqueeze(2)], dim=2).reshape(-1, n)
        h *= 2
    return (x / math.sqrt(n)).reshape(shape)


def lloyd(b, dev, n=1_000_000, it=40):
    g = torch.Generator(device="cpu").manual_seed(0); x = torch.randn(n, generator=g).to(dev); k = 2 ** b
    c = torch.distributions.Normal(0., 1.).icdf(torch.linspace(.5 / k, 1 - .5 / k, k, device=dev))
    for _ in range(it):
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        for j in range(k):
            s = x[idx == j]
            if s.numel(): c[j] = s.mean()
        c, _ = torch.sort(c)
    return c


def q_group(W, method, g, cbl, nf4):
    d_out, d_in = W.shape
    while d_in % g and g > 1: g //= 2
    Wf = W.float().reshape(d_out, d_in // g, g)
    if method == "uniform_sym":
        sc = Wf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
        q = torch.clamp(torch.round(Wf / sc), -8, 7) * sc
    elif method == "nf4":
        sc = Wf.abs().amax(-1, keepdim=True).clamp(min=1e-8)
        xn = (Wf / sc).clamp(-1, 1)
        q = nf4[torch.bucketize(xn, (nf4[1:] + nf4[:-1]) / 2)] * sc
    else:  # lloyd
        sc = Wf.std(-1, keepdim=True).clamp(min=1e-8)
        q = cbl[torch.bucketize(Wf / sc, (cbl[1:] + cbl[:-1]) / 2)] * sc
    return q.reshape(d_out, d_in).to(W.dtype), 4 + 16.0 / g


def q_rot(W, g, cbl, B=256):  # block-B Hadamard rotation + Lloyd, no padding
    d_out, d_in = W.shape
    while d_in % B and B > 1: B //= 2
    H = fwht(torch.eye(B, device=W.device))
    Wf = W.float().reshape(d_out, d_in // B, B) @ H              # rotate each B-block
    Wr = Wf.reshape(d_out, d_in // g, g)
    sc = Wr.std(-1, keepdim=True).clamp(min=1e-8)
    q = (cbl[torch.bucketize(Wr / sc, (cbl[1:] + cbl[:-1]) / 2)] * sc).reshape(d_out, d_in // B, B)
    return (q @ H).reshape(d_out, d_in).to(W.dtype), 4 + 16.0 / g


def ppl_wt2(model, tok, ctx=512, max_tok=40_000):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0][:max_tok].to("cuda")
    nll, ntok = 0.0, 0
    for i in range(0, ids.shape[0] - 1, ctx):
        w = ids[i:i + ctx + 1]
        if w.shape[0] < 2: break
        with torch.no_grad():
            out = model(w[:-1].unsqueeze(0), labels=w[:-1].unsqueeze(0))
        n = w.shape[0] - 1
        nll += out.loss.item() * n; ntok += n
    return math.exp(nll / ntok)


def snap(m): return {n: x.weight.data.detach().to("cpu", copy=True) for n, x in m.named_modules() if isinstance(x, nn.Linear)}
def restore(m, s):
    with torch.no_grad():
        for n, x in m.named_modules():
            if isinstance(x, nn.Linear) and n in s: x.weight.data.copy_(s[n].to(x.weight.device))


if __name__ == "__main__":
    _z = torch.randn(4, 16); assert torch.allclose(fwht(fwht(_z)), _z, atol=1e-4), "FWHT not involutive"
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    dev = "cuda"; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=dev).eval()
    NF4d = NF4.to(dev); cbl = lloyd(4, dev)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    s = snap(model)
    base = ppl_wt2(model, tok); print(f"[FP16] wt2_ppl={base:.4f}", flush=True)

    configs = [
        ("uniform_sym g32", lambda W: q_group(W, "uniform_sym", 32, cbl, NF4d)),
        ("nf4 g64",         lambda W: q_group(W, "nf4", 64, cbl, NF4d)),
        ("nf4 g32",         lambda W: q_group(W, "nf4", 32, cbl, NF4d)),
        ("lloyd g32",       lambda W: q_group(W, "lloyd", 32, cbl, NF4d)),
        ("lloyd g16",       lambda W: q_group(W, "lloyd", 16, cbl, NF4d)),
        ("lloyd_rot256 g32", lambda W: q_rot(W, 32, cbl)),
    ]
    res = [{"cfg": "fp16", "ppl": base, "bpw": 16.0, "gap%": 0.0}]
    for name, fn in configs:
        restore(model, s); tq = time.time(); bpw = 4.0
        with torch.no_grad():
            for _, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    q, bpw = fn(m.weight.data); m.weight.data.copy_(q)
        p = ppl_wt2(model, tok); gap = 100 * (p - base) / base
        print(f"[{name:18s}] ppl={p:8.4f}  bpw={bpw:.3f}  gap={gap:+.2f}%  ({time.time()-tq:.0f}s)", flush=True)
        res.append({"cfg": name, "ppl": p, "bpw": bpw, "gap%": gap})

    print("\n=== best universal 4-bit (Qwen2.5-1.5B, standard WikiText-2 ppl) ===")
    for r in sorted(res, key=lambda r: r["ppl"]):
        print(f"  {r['cfg']:18s} ppl={r['ppl']:8.4f}  bpw={r['bpw']:.3f}  gap={r['gap%']:+.2f}%", flush=True)
    print(f"FP16={base:.4f}  total={time.time()-t0:.1f}s")
    print("RESULTS_JSON " + json.dumps(res))
