import sys
import os
import time
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evals
import sanity
from data import _structured_strip
from model import SequenceModel, ModelConfig

BLANK, FLAT, BUMP, QUERY = 0, 1, 2, 3
POSBASE = 4


def make_occlusion(batch, strip_len, n_occluded, n_queries, p_stay, touch, device="cpu",
                   generator=None, fake=False, query="occluded"):
    assert n_queries <= n_occluded
    assert n_queries <= strip_len - n_occluded
    total = 2 * strip_len + 3 * n_queries
    full = torch.zeros(batch, total, dtype=torch.long)
    if p_stay is None:
        strip = (torch.rand(batch, strip_len, generator=generator) < 0.5).long()
    else:
        strip = _structured_strip(batch, strip_len, p_stay, generator)
    felt = (torch.rand(batch, strip_len, generator=generator) < 0.5).long() if fake else strip.clone()
    rnd = torch.rand(batch, strip_len, generator=generator)
    occ = rnd.argsort(dim=1)[:, :n_occluded]
    occ_mask = torch.zeros(batch, strip_len, dtype=torch.bool)
    occ_mask.scatter_(1, occ, True)
    rnd2 = torch.rand(batch, strip_len, generator=generator).masked_fill(occ_mask, float("inf"))
    obs = rnd2.argsort(dim=1)[:, :n_queries]
    ar = torch.arange(batch)
    for i in range(strip_len):
        full[:, 2 * i] = POSBASE + i
        if touch:
            ft = torch.where(felt[:, i].bool(), torch.full((batch,), BUMP, dtype=torch.long),
                             torch.full((batch,), FLAT, dtype=torch.long))
            full[:, 2 * i + 1] = torch.where(occ_mask[:, i], torch.full((batch,), BLANK, dtype=torch.long), ft)
        else:
            full[:, 2 * i + 1] = BLANK
    qpos_all = occ if query == "occluded" else obs
    for j in range(n_queries):
        base = 2 * strip_len + 3 * j
        qpos = qpos_all[:, j]
        full[:, base] = QUERY
        full[:, base + 1] = POSBASE + qpos
        ans = strip[ar, qpos]
        full[:, base + 2] = torch.where(ans.bool(), torch.full((batch,), BUMP, dtype=torch.long),
                                        torch.full((batch,), FLAT, dtype=torch.long))
    inp = full[:, :-1].contiguous()
    tgt = full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    for j in range(n_queries):
        mask[:, 2 * strip_len + 3 * j + 1] = 1.0
    return inp.to(device), tgt.to(device), mask.to(device)


def make_fn(args, p_stay, touch, fake, query):
    g = torch.Generator().manual_seed(args.seed + 1)

    def mk(b, device):
        return make_occlusion(b, args.strip_len, args.n_occluded, args.n_queries, p_stay, touch,
                              device, generator=g, fake=fake, query=query)
    return mk


def build(args, device):
    cfg = ModelConfig(vocab_size=POSBASE + args.strip_len, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode="linear", mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    return SequenceModel(cfg).to(device)


def run_arm(mk, args, device):
    torch.manual_seed(args.seed)
    model = build(args, device)
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
    leak = sanity.causal_no_future_leak(model, POSBASE + args.strip_len, 4, args.probe_len, device)
    return ev, last, leak["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=8)
    ap.add_argument("--n_occluded", type=int, default=3)
    ap.add_argument("--n_queries", type=int, default=3)
    ap.add_argument("--p_stay", type=float, default=0.85)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=100)
    ap.add_argument("--probe_len", type=int, default=16)
    ap.add_argument("--imag_margin", type=float, default=0.15)
    ap.add_argument("--imag_min", type=float, default=0.65)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = args.device
    arms = (
        ("blind", dict(p_stay=args.p_stay, touch=False, fake=False, query="occluded")),
        ("observed", dict(p_stay=args.p_stay, touch=True, fake=False, query="observed")),
        ("occ_structured", dict(p_stay=args.p_stay, touch=True, fake=False, query="occluded")),
        ("occ_random", dict(p_stay=None, touch=True, fake=False, query="occluded")),
        ("fake", dict(p_stay=args.p_stay, touch=True, fake=True, query="occluded")),
    )
    print(f"occlusion strip={args.strip_len} occ={args.n_occluded} q={args.n_queries} "
          f"p_stay={args.p_stay} steps={args.steps} device={device}", flush=True)
    res = {}
    t0 = time.time()
    for name, kw in arms:
        t = time.time()
        ev, loss, ok = run_arm(make_fn(args, **kw), args, device)
        res[name] = {"exact": ev["exact_acc"], "token": ev["token_acc"], "loss": loss, "causal_ok": ok}
        print(f"  {name:14s} token={ev['token_acc']:.3f} exact={ev['exact_acc']:.3f} "
              f"loss={loss:.3f} causal_ok={ok} [{time.time() - t:.1f}s]", flush=True)
    s = res["occ_structured"]["token"]
    r = res["occ_random"]["token"]
    o = res["observed"]["token"]
    imagines = (s - r >= args.imag_margin and s >= args.imag_min and o >= 0.8
                and res["blind"]["token"] < 0.65 and res["fake"]["token"] < 0.65)
    print(f"  IMAGINATION {'YES' if imagines else 'NO'} "
          f"(occ_structured {s:.3f} - occ_random {r:.3f} = {s - r:.3f} >= {args.imag_margin}, "
          f"structured >= {args.imag_min}, observed {o:.3f} >= 0.8, "
          f"blind {res['blind']['token']:.3f} / fake {res['fake']['token']:.3f} < 0.65)", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
