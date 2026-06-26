"""Canonical WikiText-2 perplexity.

The repo had two incompatible protocols:
  * `run_whlm.py` / `fullrot_quant.py`: 20 streaming snippets, weighted by length.
  * `quant4_explore.py` / `quant4_v2.py`: concatenated test split, 40k tokens, ctx 512.

The concatenated protocol is the standard one (matches the literature and the
upstream LittleBit eval), so it is the default here.  The streaming protocol is
kept as `ppl_wt2_stream` for back-compat with the old run numbers.
"""
import math
import torch
from datasets import load_dataset


def ppl_wt2(model, tokenizer, ctx=2048, max_tokens=40_000, split="test",
            device="cuda"):
    """Concatenated WikiText-2 perplexity (standard protocol).

    Concatenates the split's text with blank lines, tokenizes, takes the first
    max_tokens, and sums cross-entropy over ctx-length windows.
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    ids = ids[:max_tokens].to(device)
    nll, ntok = 0.0, 0
    for i in range(0, ids.shape[0] - 1, ctx):
        w = ids[i:i + ctx + 1]
        if w.shape[0] < 2:
            break
        with torch.no_grad():
            out = model(w[:-1].unsqueeze(0), labels=w[:-1].unsqueeze(0))
        n = w.shape[0] - 1
        nll += out.loss.item() * n
        ntok += n
    return math.exp(nll / ntok) if ntok else float("nan")


def ppl_wt2_stream(model, tokenizer, n_examples=20, max_len=512, device="cuda"):
    """Streaming WikiText-2 perplexity (legacy protocol used by the June 19 runs)."""
    torch.cuda.empty_cache() if device == "cuda" else None
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    losses, toks = [], []
    k = 0
    for ex in ds:
        if k >= n_examples:
            break
        t = ex["text"]
        if 50 < len(t) < 2000:
            ids = tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(ids, labels=ids)
            losses.append(out.loss.item())
            toks.append(ids.shape[1])
            k += 1
    total = sum(toks)
    weighted = sum(l * n for l, n in zip(losses, toks))
    return math.exp(weighted / total) if total else float("nan")
