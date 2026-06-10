import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SequenceModel, ModelConfig
import data as datamod


def gen(batch, touch, args, g):
    return datamod.make_touch_count(batch, args.strip_len, args.bump_prob, touch,
                                    "cpu", generator=g, return_strip=True)


def train_body(touch, args):
    torch.manual_seed(args.seed)
    vocab = 4 + 2 * args.strip_len + 1
    cfg = ModelConfig(vocab_size=vocab, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode=args.mode, mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    model = SequenceModel(cfg)
    g = torch.Generator().manual_seed(args.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for _ in range(args.steps):
        inp, tgt, mask, _ = gen(args.batch, touch, args, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def collect(model, touch, args, n, seed):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    cpos = 2 * args.strip_len
    feats, strips, counts = [], [], []
    got = 0
    while got < n:
        inp, tgt, mask, strip = gen(args.batch, touch, args, g)
        _, _, hid = model(inp, return_hidden=True)
        feats.append(hid[:, cpos, :])
        strips.append(strip)
        counts.append(strip.sum(dim=1))
        got += args.batch
    return torch.cat(feats), torch.cat(strips).float(), torch.cat(counts)


def fit_probe(x_tr, y_tr, steps=400, lr=1e-2):
    probe = nn.Linear(x_tr.size(1), y_tr.size(1))
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(probe(x_tr), y_tr)
        loss.backward()
        opt.step()
    return probe


def cell_acc(probe, x, y):
    with torch.no_grad():
        pred = (probe(x) > 0).float()
    return (pred == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=6)
    ap.add_argument("--bump_prob", type=float, default=0.5)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--mode", default="linear", choices=["mlp", "linear"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--probe_steps", type=int, default=400)
    ap.add_argument("--n_collect", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"transfer probe: can a linear read-out recover all {args.strip_len} felt cells "
          f"from the count-body's state at the moment it reports the total?", flush=True)

    real = train_body(True, args)
    blind = train_body(False, args)

    xr_tr, sr_tr, cr_tr = collect(real, True, args, args.n_collect, args.seed + 10)
    xr_te, sr_te, cr_te = collect(real, True, args, args.n_collect // 2, args.seed + 20)
    xb_tr, sb_tr, cb_tr = collect(blind, False, args, args.n_collect, args.seed + 30)
    xb_te, sb_te, cb_te = collect(blind, False, args, args.n_collect // 2, args.seed + 40)

    nc = args.strip_len + 1
    sum_tr = F.one_hot(cr_tr, nc).float()
    sum_te = F.one_hot(cr_te, nc).float()

    p_real = fit_probe(xr_tr, sr_tr, args.probe_steps)
    p_blind = fit_probe(xb_tr, sb_tr, args.probe_steps)
    p_sum = fit_probe(sum_tr, sr_tr, args.probe_steps)

    print(f"per-cell recovery (chance 0.500):", flush=True)
    print(f"  from real-touch count-body state : {cell_acc(p_real, xr_te, sr_te):.3f}", flush=True)
    print(f"  knowing ONLY the total (ceiling) : {cell_acc(p_sum, sum_te, sr_te):.3f}", flush=True)
    print(f"  from blind count-body state      : {cell_acc(p_blind, xb_te, sb_te):.3f}", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
