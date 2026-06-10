import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SequenceModel, ModelConfig
import data as datamod
import evals


def make_fn(args, n_pairs, seed):
    g = torch.Generator().manual_seed(seed)

    def mk(b, device):
        return datamod.make_mqar(b, n_pairs, args.n_queries, args.vocab, device, generator=g)

    return mk


def run_one(args, n_pairs):
    torch.manual_seed(args.seed)
    kinds = tuple(args.layers.split(",")) if args.layers else ()
    cfg = ModelConfig(vocab_size=args.vocab, d_model=args.d_model, n_layers=args.layers_n,
                      n_heads=4, mem_mode="linear", mem_heads=args.mem_heads,
                      mem_head_dim=args.head_dim, mem_hidden=args.head_dim,
                      forget_bias=-6.0, layer_kinds=kinds)
    model = SequenceModel(cfg)
    mk = make_fn(args, n_pairs, args.seed + 1)
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
    ap.add_argument("--layers", default="", help="empty=all memory; e.g. attn,attn,attn,attn")
    ap.add_argument("--layers_n", type=int, default=4)
    ap.add_argument("--mem_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--pairs", default="6,10,14,18,24")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    sub = args.layers if args.layers else "memory(linear)"
    cap = args.mem_heads * args.head_dim
    print(f"capacity wall: mqar exact_acc vs n_pairs  [substrate={sub} "
          f"heads={args.mem_heads} head_dim={args.head_dim} rank~{cap}]", flush=True)
    for np_ in [int(x) for x in args.pairs.split(",")]:
        ev = run_one(args, np_)
        print(f"  n_pairs={np_:2d}  exact_acc={ev['exact_acc']:.3f} token_acc={ev['token_acc']:.3f} "
              f"ci=({ev['wilson_lo']:.3f},{ev['wilson_hi']:.3f})", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
