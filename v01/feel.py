import sys
import os
import time
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as datamod
import evals
import sanity
from model import SequenceModel, ModelConfig


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def task_vocab(task, strip_len):
    if task == "recall":
        return 4 + strip_len
    return 5 + 2 * strip_len


def make_fn(task, touch, noise, args):
    g = torch.Generator().manual_seed(args.seed + 1)
    if task == "recall":
        def mk(b, device):
            return datamod.make_touch_world(b, args.strip_len, args.n_queries, args.bump_prob,
                                            touch, device, generator=g, noise=noise)
        return mk

    def mk(b, device):
        return datamod.make_touch_count(b, args.strip_len, args.bump_prob,
                                        touch, device, generator=g, noise=noise)
    return mk


def build(task, args, device):
    cfg = ModelConfig(vocab_size=task_vocab(task, args.strip_len), d_model=args.d_model,
                      n_layers=args.layers, n_heads=4, mem_mode="linear", mem_heads=4,
                      mem_head_dim=args.head_dim, mem_hidden=args.head_dim,
                      forget_bias=-6.0, affect=0.0)
    return SequenceModel(cfg).to(device), cfg


def run_arm(task, touch, noise, args, device):
    torch.manual_seed(args.seed)
    model, cfg = build(task, args, device)
    mk = make_fn(task, touch, noise, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    last = float("nan")
    for step in range(1, args.steps + 1):
        inp, tgt, mask = mk(args.batch, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
    ev = evals.eval_task(model, mk, args.eval_trials, args.batch, device)
    leak = sanity.causal_no_future_leak(model, task_vocab(task, args.strip_len), 4, args.probe_len, device)
    return model, ev, last, leak


def run_task(task, args, device):
    print(f"[{task}] strip={args.strip_len} steps={args.steps} device={device}", flush=True)
    arms = {}
    real_model = None
    for label, touch, noise in (("blind", False, False), ("real", True, False), ("fake", True, True)):
        t = time.time()
        model, ev, loss, leak = run_arm(task, touch, noise, args, device)
        arms[label] = {"exact": ev["exact_acc"], "token": ev["token_acc"],
                       "ci": [ev["wilson_lo"], ev["wilson_hi"]], "loss": loss, "causal_ok": leak["ok"]}
        if label == "real":
            real_model = (model, ev)
        print(f"  {label:5s} exact={ev['exact_acc']:.3f} token={ev['token_acc']:.3f} "
              f"ci=({ev['wilson_lo']:.3f},{ev['wilson_hi']:.3f}) loss={loss:.3f} "
              f"causal_ok={leak['ok']} [{time.time() - t:.1f}s]", flush=True)
    r, b, f = arms["real"]["exact"], arms["blind"]["exact"], arms["fake"]["exact"]
    causal_ok = all(a["causal_ok"] for a in arms.values())
    passed = (r >= args.pass_real and b <= args.chance_max and f <= args.chance_max
              and (r - max(b, f)) >= args.margin and causal_ok)
    print(f"  GATE {'PASS' if passed else 'FAIL'} (real {r:.3f} >= {args.pass_real}, "
          f"blind {b:.3f} / fake {f:.3f} <= {args.chance_max}, margin {r - max(b, f):.3f} >= {args.margin}, "
          f"causal {causal_ok})", flush=True)
    if real_model is not None and args.out:
        os.makedirs(args.out, exist_ok=True)
        model, ev = real_model
        path = os.path.join(args.out, f"feel_{task}.pt")
        torch.save({"task": task, "state_dict": model.state_dict(), "cfg": model.cfg.__dict__,
                    "eval": ev, "args": vars(args)}, path)
        print(f"  saved {path}", flush=True)
    return passed, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["recall", "count", "both"])
    ap.add_argument("--strip_len", type=int, default=6)
    ap.add_argument("--n_queries", type=int, default=3)
    ap.add_argument("--bump_prob", type=float, default=0.5)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--probe_len", type=int, default=16)
    ap.add_argument("--pass_real", type=float, default=0.9)
    ap.add_argument("--chance_max", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "feel_runs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = pick_device(args.device)
    tasks = ["recall", "count"] if args.task == "both" else [args.task]
    results = {}
    t0 = time.time()
    for task in tasks:
        passed, arms = run_task(task, args, device)
        results[task] = {"passed": passed, "arms": arms}
    allpass = all(v["passed"] for v in results.values())
    print(f"ALL {'PASS' if allpass else 'FAIL'} total {time.time() - t0:.1f}s", flush=True)
    print("RESULT " + json.dumps(results))
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
