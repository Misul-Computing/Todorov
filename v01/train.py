import os
import sys
import json
import time
import math
import argparse
import functools
import torch
from model import SequenceModel, ModelConfig
from memory import DescentMemory
import data as datamod
import evals
import sanity


def make_batch(args, batch, device):
    if args.task == "mqar":
        return datamod.make_mqar(batch, args.n_pairs, args.n_queries, args.vocab, device)
    if args.task == "passkey":
        return datamod.make_passkey(batch, args.seq_len, args.vocab, args.key_len, device, gap=args.gap)
    raise ValueError(args.task)


def build_model(args, device):
    kinds = tuple(args.layers.split(",")) if args.layers else ()
    common = dict(vocab_size=args.vocab, mem_mode=args.mode, layer_kinds=kinds,
                  use_spikes=args.spikes, forget_bias=args.forget_bias,
                  out_gate_bias=args.out_gate_bias)
    if args.preset == "toy":
        cfg = ModelConfig(d_model=256, n_layers=4, n_heads=4, mem_heads=4,
                          mem_head_dim=64, mem_hidden=64, **common)
    elif args.preset == "d100m":
        cfg = ModelConfig(d_model=768, n_layers=12, n_heads=12, mem_heads=8,
                          mem_head_dim=96, mem_hidden=96, **common)
    else:
        raise ValueError(args.preset)
    return SequenceModel(cfg).to(device), cfg


@torch.no_grad()
def telemetry(model, inp):
    model(inp)
    keys = ["lam", "beta", "mu", "out_gate", "state_norm"]
    acc = {k: [] for k in keys}
    for blk in model.blocks:
        if isinstance(blk.mixer, DescentMemory) and hasattr(blk.mixer, "last_stats"):
            for k in keys:
                acc[k].append(float(blk.mixer.last_stats[k].item()))
    return {"mem_" + k: (sum(v) / len(v)) for k, v in acc.items() if v}


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def write_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def save_ckpt(run_dir, name, core, opt, step, cfg, extra):
    payload = {"step": step, "model": core.state_dict(), "opt": opt.state_dict(),
               "cfg": cfg.__dict__, "extra": extra}
    torch.save(payload, os.path.join(run_dir, name))


def run_selftest(args, model, cfg, device, mk):
    res = {}
    res["loss_at_init"] = sanity.loss_at_init(model, mk, args.batch, device, args.vocab)
    res["next_token_aligned"] = {"ok": evals.assert_next_token_aligned(mk, 8, device)}
    res["causal"] = sanity.causal_no_future_leak(model, args.vocab, 4, args.probe_len, device)
    res["retention"] = sanity.retention_floor_check(model, mk, 8, device, args.probe_len)
    sk = tuple(args.layers.split(","))[:2] if args.layers else ()
    small = SequenceModel(ModelConfig(vocab_size=args.vocab, d_model=128, n_layers=2,
                                      mem_mode=args.mode, mem_heads=4, mem_head_dim=32,
                                      mem_hidden=32, forget_bias=args.forget_bias,
                                      layer_kinds=sk)).to(device)
    res["overfit"] = sanity.overfit_one_batch(small, mk, 8, device, steps=args.overfit_steps,
                                              target=0.5 * math.log(args.vocab))
    failed = [k for k, v in res.items() if not v.get("ok", False)]
    return res, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mqar", choices=["mqar", "passkey"])
    ap.add_argument("--mode", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--preset", default="toy", choices=["toy", "d100m"])
    ap.add_argument("--layers", default="")
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--n_pairs", type=int, default=32)
    ap.add_argument("--n_queries", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--key_len", type=int, default=5)
    ap.add_argument("--gap", type=int, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--spikes", action="store_true")
    ap.add_argument("--forget_bias", type=float, default=-6.0)
    ap.add_argument("--out_gate_bias", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--probe_len", type=int, default=128)
    ap.add_argument("--overfit_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/v01/runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    core, cfg = build_model(args, device)
    model = torch.compile(core) if args.compile else core
    mk = functools.partial(make_batch, args)
    nparams = core.num_params()
    ts = int(time.time())
    tag = args.tag or f"{args.task}_{args.mode}_{args.preset}"
    run_dir = os.path.join(args.out, f"{ts}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "metrics.jsonl")
    meta = {"event": "start", "args": vars(args), "n_params": nparams,
            "device": device, "run_dir": run_dir, "ts": ts}
    write_jsonl(log_path, meta)
    print(f"run_dir={run_dir} n_params={nparams} device={device}", flush=True)

    st, failed = run_selftest(args, core, cfg, device, mk)
    write_jsonl(log_path, {"event": "selftest", "results": st, "failed": failed})
    print("selftest:", json.dumps(st), flush=True)
    if failed:
        print("SELFTEST FAILED:", failed, flush=True)
        write_jsonl(log_path, {"event": "abort", "reason": "selftest_failed", "failed": failed})
        sys.exit(2)
    if args.selftest:
        return

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    best = -1.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        inp, tgt, mask = mk(args.batch, device)
        opt.zero_grad(set_to_none=True)
        if args.amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(inp, tgt, mask)
        else:
            _, loss = model(inp, tgt, mask)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            write_jsonl(log_path, {"event": "train", "step": step, "loss": float(loss.item()),
                                   "lr": lr, "grad_norm": float(gnorm), "tok_s": args.batch * inp.size(1) * step / (time.time() - t0)})
            print(f"step {step} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evals.eval_task(model, mk, args.eval_trials, args.eval_batch, device)
            tele = telemetry(core, mk(min(16, args.batch), device)[0])
            rec = {"event": "eval", "step": step, **ev, **tele}
            write_jsonl(log_path, rec)
            print(f"EVAL step {step} exact_acc {ev['exact_acc']:.3f} token_acc {ev['token_acc']:.3f} "
                  f"ci=({ev['wilson_lo']:.3f},{ev['wilson_hi']:.3f}) "
                  f"out_gate {tele.get('mem_out_gate', float('nan')):.3f} state_norm {tele.get('mem_state_norm', float('nan')):.3f}", flush=True)
            if ev["exact_acc"] > best:
                best = ev["exact_acc"]
                save_ckpt(run_dir, "best.pt", core, opt, step, cfg, {"eval": ev, "tele": tele})
        if step % args.ckpt_every == 0:
            save_ckpt(run_dir, f"step_{step}.pt", core, opt, step, cfg, {})
            ckpts = sorted([f for f in os.listdir(run_dir) if f.startswith("step_")],
                           key=lambda f: int(f.split("_")[1].split(".")[0]))
            for old in ckpts[:-3]:
                os.remove(os.path.join(run_dir, old))
    write_jsonl(log_path, {"event": "done", "best_exact_acc": best, "elapsed_s": time.time() - t0})
    print(f"DONE best_exact_acc {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
