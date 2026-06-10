import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SequenceModel, ModelConfig
from data import _structured_strip

BLANK, FLAT, BUMP, COUNT, QUERY = 0, 1, 2, 3, 4


def vocab_of(n):
    return 2 * n + 6


def pos_base():
    return 5


def cnt_base(n):
    return 5 + n


def sweep_part(strip, touch, n):
    b = strip.size(0)
    sw = torch.zeros(b, 2 * n, dtype=torch.long)
    for i in range(n):
        sw[:, 2 * i] = pos_base() + i
        if touch:
            sw[:, 2 * i + 1] = torch.where(strip[:, i].bool(), torch.full((b,), BUMP, dtype=torch.long), torch.full((b,), FLAT, dtype=torch.long))
        else:
            sw[:, 2 * i + 1] = BLANK
    return sw


def make_multitask(b, n, p_stay, touch, g):
    strip = _structured_strip(b, n, p_stay, g)
    ar = torch.arange(b)
    sw = sweep_part(strip, touch, n)
    qpos = torch.randint(0, n, (b,), generator=g)
    rec = torch.where(strip[ar, qpos].bool(), torch.full((b,), BUMP, dtype=torch.long), torch.full((b,), FLAT, dtype=torch.long))
    total = 2 * n + 5
    full = torch.zeros(b, total, dtype=torch.long)
    full[:, :2 * n] = sw
    full[:, 2 * n] = COUNT
    full[:, 2 * n + 1] = cnt_base(n) + strip.sum(dim=1)
    full[:, 2 * n + 2] = QUERY
    full[:, 2 * n + 3] = pos_base() + qpos
    full[:, 2 * n + 4] = rec
    inp, tgt = full[:, :-1].contiguous(), full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    mask[:, 2 * n] = 1.0
    mask[:, 2 * n + 3] = 1.0
    return inp, tgt, mask


def make_count_only(b, n, p_stay, touch, g):
    strip = _structured_strip(b, n, p_stay, g)
    sw = sweep_part(strip, touch, n)
    total = 2 * n + 2
    full = torch.zeros(b, total, dtype=torch.long)
    full[:, :2 * n] = sw
    full[:, 2 * n] = COUNT
    full[:, 2 * n + 1] = cnt_base(n) + strip.sum(dim=1)
    inp, tgt = full[:, :-1].contiguous(), full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    mask[:, 2 * n] = 1.0
    return inp, tgt, mask


def build(args, seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab_of(args.n), d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode=args.mode, mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    return SequenceModel(cfg)


def train(model, mkfn, args, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for _ in range(args.steps):
        inp, tgt, mask = mkfn(args.batch, args.n, args.p_stay, True, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def labels_for(strip, n, kind):
    if kind == "lr":
        left = strip[:, :n // 2].sum(dim=1)
        right = strip[:, n // 2:].sum(dim=1)
        return (left > right).long(), (left != right)
    return strip.sum(dim=1) % 2, torch.ones(strip.size(0), dtype=torch.bool)


@torch.no_grad()
def collect_novel(body, touch, args, n_examples, seed, kind, at_query=False):
    body.eval()
    g = torch.Generator().manual_seed(seed)
    last = 2 * args.n if at_query else 2 * args.n - 1
    feats, labels = [], []
    got = 0
    while got < n_examples:
        strip = _structured_strip(args.batch, args.n, args.p_stay, g)
        label, keep = labels_for(strip, args.n, kind)
        sw = sweep_part(strip, touch, args.n)
        if at_query:
            col = torch.full((sw.size(0), 1), COUNT, dtype=torch.long)
            sw = torch.cat([sw, col], dim=1)
        _, _, hid = body(sw, return_hidden=True)
        feats.append(hid[:, last, :][keep])
        labels.append(label[keep])
        got += int(keep.sum().item())
    return torch.cat(feats)[:n_examples], torch.cat(labels)[:n_examples]


def probe_acc(body, touch, args, seed, kind, at_query=False):
    xtr, ytr = collect_novel(body, touch, args, args.n_probe, seed + 1, kind, at_query)
    xte, yte = collect_novel(body, touch, args, args.n_eval, seed + 2, kind, at_query)
    probe = nn.Linear(xtr.size(1), 2)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2)
    for _ in range(args.probe_steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(xte).argmax(-1) == yte).float().mean().item()
        base = max(yte.float().mean().item(), 1 - yte.float().mean().item())
    return acc, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--p_stay", type=float, default=0.8)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--mode", default="linear", choices=["mlp", "linear"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--probe_steps", type=int, default=300)
    ap.add_argument("--n_probe", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"hard test: answer a NEVER-trained question (left half vs right half: more bumps?) "
          f"from a frozen body's representation of the felt sweep.", flush=True)

    mt = build(args, args.seed)
    train(mt, make_multitask, args, args.seed + 1)
    mt_blind = build(args, args.seed)

    def train_blind(model, mkfn, args, seed):
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        model.train()
        for _ in range(args.steps):
            inp, tgt, mask = mkfn(args.batch, args.n, args.p_stay, False, g)
            opt.zero_grad(set_to_none=True)
            _, loss = model(inp, tgt, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    train_blind(mt_blind, make_multitask, args, args.seed + 1)
    conly = build(args, args.seed)
    train(conly, make_count_only, args, args.seed + 1)
    scratch = build(args, args.seed)

    rows = [
        ("multitask + touch", mt, True),
        ("count-only + touch", conly, True),
        ("multitask + BLIND", mt_blind, False),
        ("untrained (scratch)", scratch, True),
    ]
    for kind, desc in (("lr", "left-vs-right [linear/spatial]"), ("parity", "parity [nonlinear/global]")):
        for at_query in (False, True):
            where = "count-query pos" if at_query else "sweep end"
            print(f"novel question: {desc} probed at {where}", flush=True)
            for label, body, touch in rows:
                acc, base = probe_acc(body, touch, args, args.seed + 50, kind, at_query)
                print(f"  {label:22s} novel_acc={acc:.3f}  (chance {base:.3f})", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
