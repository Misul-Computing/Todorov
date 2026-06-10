import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SequenceModel, ModelConfig
import data as datamod
import evals


def make_fn(touch, flip_p, args):
    g = torch.Generator().manual_seed(args.seed + 1)

    def mk(b, device):
        return datamod.make_touch_count(b, args.strip_len, args.bump_prob,
                                        touch, device, generator=g, flip_p=flip_p)

    return mk


def run_arm(touch, flip_p, args):
    torch.manual_seed(args.seed)
    vocab = 4 + 2 * args.strip_len + 1
    cfg = ModelConfig(vocab_size=vocab, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode=args.mode, mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    model = SequenceModel(cfg)
    mk = make_fn(touch, flip_p, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for _ in range(args.steps):
        inp, tgt, mask = mk(args.batch, "cpu")
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return evals.eval_task(model, mk, args.eval_trials, args.batch, "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=6)
    ap.add_argument("--bump_prob", type=float, default=0.5)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--mode", default="linear", choices=["mlp", "linear"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"task touch_count robustness strip={args.strip_len} steps={args.steps} "
          f"(touch corrupted by bit-flip prob)", flush=True)
    ev = run_arm(False, 0.0, args)
    print(f"blind            count_acc={ev['token_acc']:.3f}", flush=True)
    for fp in (0.0, 0.1, 0.2, 0.35, 0.5):
        ev = run_arm(True, fp, args)
        print(f"touch flip={fp:.2f}  count_acc={ev['token_acc']:.3f} "
              f"ci=({ev['wilson_lo']:.3f},{ev['wilson_hi']:.3f})", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
