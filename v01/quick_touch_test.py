import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SequenceModel, ModelConfig
import data as datamod
import evals
import sanity


def make_fn(touch, noise, args):
    g = torch.Generator().manual_seed(args.seed + 1)

    def mk(b, device):
        return datamod.make_touch_world(b, args.strip_len, args.n_queries, args.bump_prob,
                                        touch, device, generator=g, noise=noise)

    return mk


def run_arm(touch, noise, args):
    torch.manual_seed(args.seed)
    vocab = 4 + args.strip_len
    cfg = ModelConfig(vocab_size=vocab, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode=args.mode, mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    model = SequenceModel(cfg).to("mps")
    mk = make_fn(touch, noise, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        inp, tgt, mask = mk(args.batch, "mps")
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
    ev = evals.eval_task(model, mk, args.eval_trials, args.batch, "mps")
    leak = sanity.causal_no_future_leak(model, vocab, 4, args.probe_len, "mps")
    return ev, last_loss, leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=6)
    ap.add_argument("--n_queries", type=int, default=3)
    ap.add_argument("--bump_prob", type=float, default=0.5)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--mode", default="linear", choices=["mlp", "linear"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--probe_len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    chance = max(args.bump_prob, 1 - args.bump_prob)
    print(f"task touch_world strip={args.strip_len} queries={args.n_queries} "
          f"seq={2 * args.strip_len + 3 * args.n_queries} steps={args.steps} "
          f"chance_token_acc~{chance:.3f}", flush=True)
    for label, touch, noise in (("blind", False, False), ("real ", True, False), ("fake ", True, True)):
        ta = time.time()
        ev, loss, leak = run_arm(touch, noise, args)
        print(f"{label}  exact_acc={ev['exact_acc']:.3f} token_acc={ev['token_acc']:.3f} "
              f"ci=({ev['wilson_lo']:.3f},{ev['wilson_hi']:.3f}) loss={loss:.3f} "
              f"causal_ok={leak['ok']}  [{time.time() - ta:.1f}s]", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
