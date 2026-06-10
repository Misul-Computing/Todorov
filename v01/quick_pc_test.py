import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SequenceModel, ModelConfig
import data as datamod
import evals


def build(args, seed):
    torch.manual_seed(seed)
    vocab = 4 + 2 * args.strip_len + 1
    cfg = ModelConfig(vocab_size=vocab, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode=args.mode, mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    return SequenceModel(cfg)


def sweep_fn(args, seed):
    g = torch.Generator().manual_seed(seed)

    def mk(b, device):
        return datamod.make_sweep(b, args.strip_len, args.p_stay, device, generator=g)

    return mk


def count_fn(args, seed):
    g = torch.Generator().manual_seed(seed)

    def mk(b, device):
        return datamod.make_touch_count(b, args.strip_len, args.bump_prob, True, device,
                                        generator=g, p_stay=args.p_stay)

    return mk


def recall_fn(args, seed):
    g = torch.Generator().manual_seed(seed)

    def mk(b, device):
        return datamod.make_touch_world(b, args.strip_len, args.n_queries, args.bump_prob,
                                        True, device, generator=g, p_stay=args.p_stay)

    return mk


def train(model, mk, steps, lr, batch):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for _ in range(steps):
        inp, tgt, mask = mk(batch, "cpu")
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


CHECKPOINTS = [25, 50, 100, 200]


def finetune_curve(model, args, seed, task_fn):
    mk = task_fn(args, seed + 5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, betas=(0.9, 0.95), weight_decay=0.01)
    step = 0
    row = []
    for target in CHECKPOINTS:
        model.train()
        while step < target:
            inp, tgt, mask = mk(args.batch, "cpu")
            opt.zero_grad(set_to_none=True)
            _, loss = model(inp, tgt, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        ev = evals.eval_task(model, mk, 200, args.batch, "cpu")
        row.append(ev["token_acc"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=8)
    ap.add_argument("--bump_prob", type=float, default=0.5)
    ap.add_argument("--p_stay", type=float, default=0.8)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--mode", default="linear", choices=["mlp", "linear"])
    ap.add_argument("--pre_steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ft_lr", type=float, default=2e-3)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_queries", type=int, default=2)
    ap.add_argument("--task", default="count", choices=["count", "recall"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    task_fn = count_fn if args.task == "count" else recall_fn
    print(f"predictive-coding pretraining: feel-and-predict a structured world (p_stay={args.p_stay}), "
          f"then learn to {args.task} with few labels (avg {args.n_seeds} seeds)", flush=True)

    seeds = [args.seed + k for k in range(args.n_seeds)]
    pre_accs, pre_rows, scr_rows = [], [], []
    for sd in seeds:
        pre = build(args, sd)
        sweep = sweep_fn(args, sd + 101)
        train(pre, sweep, args.pre_steps, args.lr, args.batch)
        pre_accs.append(evals.eval_task(pre, sweep, 200, args.batch, "cpu")["token_acc"])
        scratch = build(args, sd)
        pre_rows.append(finetune_curve(pre, args, sd, task_fn))
        scr_rows.append(finetune_curve(scratch, args, sd, task_fn))

    def avg(rows):
        return [sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]

    print(f"pretrain next-sensation accuracy: {sum(pre_accs) / len(pre_accs):.3f} (chance 0.500)", flush=True)
    print(f"{args.task}_acc @ finetune steps {CHECKPOINTS}:", flush=True)
    print(f"  pretrained   : " + " ".join(f"{a:.3f}" for a in avg(pre_rows)), flush=True)
    print(f"  from-scratch : " + " ".join(f"{a:.3f}" for a in avg(scr_rows)), flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
