import sys
import os
import math
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SequenceModel, ModelConfig
import data as datamod
import evals
import sanity


ARMS = {
    "control": (0.0, "surprise", False),
    "surprise_half": (0.5, "surprise", False),
    "surprise": (1.0, "surprise", False),
    "shuffle": (1.0, "shuffle", False),
    "noise": (1.0, "noise", False),
    "noisetoken": (1.0, "noise_token", False),
    "batchnorm": (1.0, "surprise_batchnorm", False),
    "nowrite": (0.0, "surprise", True),
    "smooth_noise": (1.0, "smooth_noise", False),
    "smooth_noise_token": (1.0, "smooth_noise_token", False),
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


def make_fn(vocab, n_pairs, n_queries, seed):
    g = torch.Generator().manual_seed(seed)

    def mk(b, device):
        return datamod.make_mqar(b, n_pairs, n_queries, vocab, device, generator=g)

    return mk


def run_one_seed(args_dict):
    args = argparse.Namespace(**args_dict)
    device = "mps"
    torch.manual_seed(args.seed)

    mk = make_fn(args.vocab, args.n_pairs, args.n_queries, args.seed + 1)
    affect, mode, no_write = ARMS[args.arm]

    cfg = arm_cfg(affect, mode, args)
    model = SequenceModel(cfg).to(device)
    if no_write:
        pin_no_write(model)

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    if args.fp16:
        model = model.half()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        inp, tgt, mask = mk(args.batch, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if no_write:
            pin_no_write(model)
        last_loss = float(loss.item())

    model.eval()
    ev = evals.eval_task(model, mk, args.eval_trials, args.batch, device)
    leak = sanity.causal_no_future_leak(model, args.vocab, 4, args.probe_len, device)

    with torch.no_grad():
        inp, _, _ = mk(8, device)
        model(inp)
    stats = {}
    for blk in model.blocks:
        ls = getattr(blk.mixer, "last_stats", None)
        if ls is not None and "surprise" in ls:
            stats["surprise"] = float(ls["surprise"].item())
            stats["write_gain"] = float(ls["write_gain"].item())

    result = {
        "seed": args.seed,
        "arm": args.arm,
        "exact_acc": ev["exact_acc"],
        "token_acc": ev["token_acc"],
        "wilson_lo": ev["wilson_lo"],
        "wilson_hi": ev["wilson_hi"],
        "final_loss": last_loss,
        "causal_ok": leak["ok"],
        "surprise": stats.get("surprise", float("nan")),
        "write_gain": stats.get("write_gain", float("nan")),
    }
    if device == "mps":
        torch.mps.empty_cache()
    return result


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
    ap.add_argument("--probe_len", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--arms", default="shuffle")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true", default=False)
    ap.add_argument("--compile", action="store_true", default=False)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    jobs = []
    for seed in seeds:
        for arm in arms:
            d = vars(args).copy()
            d["seed"] = seed
            d["arm"] = arm
            jobs.append(d)

    chance = 1.0 / (args.vocab - 1)
    print(f"sweep: {len(jobs)} jobs ({len(seeds)} seeds x {len(arms)} arms) "
          f"workers={args.workers} fp16={args.fp16} compile={args.compile} "
          f"chance_token~{chance:.4f}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(run_one_seed, jobs)
    dt = time.time() - t0

    results.sort(key=lambda r: (r["arm"], r["seed"]))
    print(f"\n{'arm':12s} {'seed':>4s} {'exact':>6s} {'token':>6s} {'ci_lo':>6s} {'ci_hi':>6s} "
          f"{'loss':>6s} {'surp':>5s} {'gain':>5s} {'causal':>5s}")
    print("-" * 80)
    for r in results:
        print(f"{r['arm']:12s} {r['seed']:4d} {r['exact_acc']:6.3f} {r['token_acc']:6.3f} "
              f"{r['wilson_lo']:6.3f} {r['wilson_hi']:6.3f} {r['final_loss']:6.3f} "
              f"{r['surprise']:5.3f} {r['write_gain']:5.3f} {str(r['causal_ok']):>5s}")

    ignited = [r for r in results if r["exact_acc"] > 0.0]
    print(f"\nignited: {len(ignited)}/{len(results)} jobs above exact=0")
    if ignited:
        for r in ignited:
            print(f"  {r['arm']} seed={r['seed']} exact={r['exact_acc']:.3f} token={r['token_acc']:.3f}")
    print(f"total {dt:.1f}s ({dt/len(jobs):.1f}s/job)")


if __name__ == "__main__":
    main()
