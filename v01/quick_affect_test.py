import sys
import os
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SequenceModel, ModelConfig
import data as datamod
import evals
import sanity


def make_fn(vocab, n_pairs, n_queries, seed):
    g = torch.Generator().manual_seed(seed)

    def mk(b, device):
        return datamod.make_mqar(b, n_pairs, n_queries, vocab, device, generator=g)

    return mk


ARMS = {
    "control": (0.0, "surprise", False),
    "surprise_half": (0.5, "surprise", False),
    "surprise": (1.0, "surprise", False),
    "shuffle": (1.0, "shuffle", False),
    "noise": (1.0, "noise", False),
    "noisetoken": (1.0, "noise_token", False),
    "batchnorm": (1.0, "surprise_batchnorm", False),
    "nowrite": (0.0, "surprise", True),
}


def arm_cfg(affect, mode, args):
    return ModelConfig(vocab_size=args.vocab, d_model=args.d_model, n_layers=args.layers,
                       n_heads=4, mem_mode=args.mode, mem_heads=4, mem_head_dim=args.head_dim,
                       mem_hidden=args.head_dim, forget_bias=-6.0, affect=affect,
                       affect_mode=mode)


def pin_no_write(model):
    with torch.no_grad():
        for blk in model.blocks:
            if hasattr(blk.mixer, "gate_proj"):
                blk.mixer.gate_proj.bias[0:blk.mixer.H].fill_(-12.0)


def run_gates(args):
    torch.manual_seed(args.seed)
    mk = make_fn(args.vocab, args.n_pairs, args.n_queries, args.seed + 1)
    seq = 2 * args.n_pairs + 2 * args.n_queries
    model = SequenceModel(arm_cfg(1.0, "surprise", args))
    g_init = sanity.loss_at_init(model, mk, args.batch, "cpu", args.vocab)
    g_ret = sanity.retention_floor_check(model, mk, 8, "cpu", seq)
    g_fit = sanity.overfit_one_batch(model, mk, 8, "cpu", steps=300,
                                     target=0.5 * math.log(args.vocab))
    print(f"gates(affect=1.0 surprise cfg): loss_at_init={g_init['loss']:.3f}/{g_init['expected']:.3f} ok={g_init['ok']}  "
          f"retention={g_ret['retention_floor']:.3f} ok={g_ret['ok']}  "
          f"overfit_final={g_fit['final_loss']:.3f} ok={g_fit['ok']}", flush=True)
    return g_init["ok"] and g_ret["ok"] and g_fit["ok"]


def run_arm(name, args):
    affect, mode, no_write = ARMS[name]
    torch.manual_seed(args.seed)
    cfg = arm_cfg(affect, mode, args)
    model = SequenceModel(cfg)
    if no_write:
        pin_no_write(model)
    mk = make_fn(args.vocab, args.n_pairs, args.n_queries, args.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        inp, tgt, mask = mk(args.batch, "cpu")
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if no_write:
            pin_no_write(model)
        last_loss = float(loss.item())
        if args.eval_every > 0 and step % args.eval_every == 0:
            ie = evals.eval_task(model, mk, 64, args.batch, "cpu")
            print(f"  {name} step {step} loss {last_loss:.3f} "
                  f"exact {ie['exact_acc']:.3f} token {ie['token_acc']:.3f}", flush=True)
    ev = evals.eval_task(model, mk, args.eval_trials, args.batch, "cpu")
    leak = sanity.causal_no_future_leak(model, args.vocab, 4, args.probe_len, "cpu")
    with torch.no_grad():
        inp, _, _ = mk(8, "cpu")
        model(inp)
    stats = {}
    for blk in model.blocks:
        ls = getattr(blk.mixer, "last_stats", None)
        if ls is not None and "surprise" in ls:
            stats["surprise"] = float(ls["surprise"].item())
            stats["write_gain"] = float(ls["write_gain"].item())
    return ev, last_loss, leak, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=24)
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--mode", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--probe_len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="control,surprise_half,surprise,shuffle,noise,noisetoken,batchnorm,nowrite")
    args = ap.parse_args()

    t0 = time.time()
    chance = 1.0 / (args.vocab - 1)
    print(f"task mqar vocab={args.vocab} pairs={args.n_pairs} queries={args.n_queries} "
          f"seq={2 * args.n_pairs + 2 * args.n_queries} steps={args.steps} batch={args.batch} "
          f"chance_token_acc~{chance:.4f}", flush=True)
    if not run_gates(args):
        print("sanity gates failed; no training number is counted", flush=True)
        return
    for name in args.arms.split(","):
        ta = time.time()
        ev, loss, leak, stats = run_arm(name, args)
        dt = time.time() - ta
        tag = f"{name:10s}" if leak["ok"] else f"INVALID(causal) {name}"
        print(f"{tag}  exact_acc={ev['exact_acc']:.3f} "
              f"token_acc={ev['token_acc']:.3f} ci=({ev['wilson_lo']:.3f},{ev['wilson_hi']:.3f}) "
              f"final_loss={loss:.3f} causal_ok={leak['ok']} (max_diff={leak['max_diff']:.1e}) "
              f"surprise={stats.get('surprise', float('nan')):.3f} "
              f"write_gain={stats.get('write_gain', float('nan')):.3f}  [{dt:.1f}s]", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
