"""Best universal 4-bit, iteration 2: protect tied embedding + optimal codebook.

Findings from v1: NF4 (absmax) >> Lloyd (std) because real weights are heavy-tailed
(absmax preserves outliers, std clips them). Best was NF4 g32 at +15.3%, still not
near-lossless. Hypotheses tested here:
  1. The tied lm_head/embedding (233M = 15% of a 1.5B model) is sensitive; quantizing
     it to 4-bit is a big chunk of the loss. Protect it (keep fp16) and measure, with
     HONEST average bits/weight (protecting 15% of params raises the bitrate).
  2. An optimal data-free codebook (Lloyd on the REAL absmax-normalized weight
     distribution, 'OF4') beats NF4's Gaussian-quantile assumption.
Standard WikiText-2 ppl, fp16 (fast on T4).
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


def lloyd_on(x, k=16, it=40):
    c = torch.quantile(x[:200000].sort().values, torch.linspace(.03, .97, k, device=x.device))
    for _ in range(it):
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        for j in range(k):
            s = x[idx == j]
            if s.numel(): c[j] = s.mean()
        c, _ = torch.sort(c)
    return c


def q_cb(W, cb, g=32):
    d_out, d_in = W.shape; gg = g
    while d_in % gg and gg > 1: gg //= 2
    Wb = W.float().reshape(d_out, d_in // gg, gg)
    sc = Wb.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    xn = (Wb / sc).clamp(cb[0].item(), cb[-1].item())
    q = cb[torch.bucketize(xn, (cb[1:] + cb[:-1]) / 2)] * sc
    return q.reshape(d_out, d_in).to(W.dtype), 4 + 16.0 / gg


def ppl_wt2(model, tok, ctx=512, max_tok=40_000):
    ids = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]),
              return_tensors="pt").input_ids[0][:max_tok].to("cuda")
    nll, nt = 0.0, 0
    for i in range(0, ids.shape[0] - 1, ctx):
        w = ids[i:i + ctx + 1]
        if w.shape[0] < 2: break
        with torch.no_grad():
            loss = model(w[:-1].unsqueeze(0), labels=w[:-1].unsqueeze(0)).loss
        nll += loss.item() * (w.shape[0] - 1); nt += w.shape[0] - 1
    return math.exp(nll / nt)


def snap(m): return {n: x.weight.data.detach().to("cpu", copy=True) for n, x in m.named_modules() if isinstance(x, nn.Linear)}
def restore(m, s):
    with torch.no_grad():
        for n, x in m.named_modules():
            if isinstance(x, nn.Linear) and n in s: x.weight.data.copy_(s[n].to(x.weight.device))


def apply_q(model, cb, g, skip_lmhead):
    tot_bits, tot = 0, 0
    with torch.no_grad():
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear): continue
            if skip_lmhead and "lm_head" in n:
                tot_bits += m.weight.numel() * 16; tot += m.weight.numel(); continue
            q, bpw = q_cb(m.weight.data, cb, g)
            m.weight.data.copy_(q)
            tot_bits += m.weight.numel() * bpw; tot += m.weight.numel()
    return tot_bits / tot


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"; dev = "cuda"; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=dev).eval()
    print(f"loaded in {time.time()-t0:.1f}s  tied_embeddings={model.config.tie_word_embeddings}", flush=True)
    NF4d = NF4.to(dev)

    # Build OF4: Lloyd on the REAL absmax-normalized weight distribution (data-free)
    samp = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data.float(); d_out, d_in = W.shape; gg = 32
            while d_in % gg and gg > 1: gg //= 2
            Wb = W.reshape(d_out, d_in // gg, gg)
            xn = (Wb / Wb.abs().amax(-1, keepdim=True).clamp(min=1e-8)).reshape(-1)
            samp.append(xn[torch.rand(xn.shape[0], device=dev) < 0.01])
    OF4 = lloyd_on(torch.cat(samp)); print(f"OF4 codebook built ({len(OF4)} levels)", flush=True)

    s = snap(model)
    base = ppl_wt2(model, tok); print(f"[FP16] ppl={base:.4f}", flush=True)

    configs = [
        ("nf4 g32 ALL",        NF4d, 32, False),
        ("nf4 g32 keep-emb",   NF4d, 32, True),
        ("nf4 g64 keep-emb",   NF4d, 64, True),
        ("OF4 g32 keep-emb",   OF4, 32, True),
        ("OF4 g32 ALL",        OF4, 32, False),
    ]
    res = [{"cfg": "fp16", "ppl": base, "bpw": 16.0, "gap%": 0.0}]
    for name, cb, g, skip in configs:
        restore(model, s); tq = time.time()
        bpw = apply_q(model, cb, g, skip)
        p = ppl_wt2(model, tok); gap = 100 * (p - base) / base
        print(f"[{name:16s}] ppl={p:7.4f}  avg_bpw={bpw:.3f}  gap={gap:+.2f}%  ({time.time()-tq:.0f}s)", flush=True)
        res.append({"cfg": name, "ppl": p, "avg_bpw": bpw, "gap%": gap})

    print("\n=== best universal 4-bit v2 (Qwen2.5-1.5B, WikiText-2) ===")
    for r in sorted(res, key=lambda r: r["ppl"]):
        print(f"  {r['cfg']:16s} ppl={r['ppl']:7.4f}  avg_bpw={r.get('avg_bpw',16):.3f}  gap={r['gap%']:+.2f}%", flush=True)
    print("RESULTS_JSON " + json.dumps(res))
