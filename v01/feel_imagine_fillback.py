import sys
import os
import time
import json
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import _structured_strip
from model import SequenceModel, ModelConfig

BLANK, FLAT, BUMP, COUNT = 0, 1, 2, 3
POSBASE = 4


def make_fillback(batch, strip_len, n_occluded, p_stay, touch, device, generator, fake):
    countbase = POSBASE + strip_len
    total = 2 * strip_len + 2
    full = torch.zeros(batch, total, dtype=torch.long)
    if p_stay is None:
        strip = (torch.rand(batch, strip_len, generator=generator) < 0.5).long()
    else:
        strip = _structured_strip(batch, strip_len, p_stay, generator)
    felt = (torch.rand(batch, strip_len, generator=generator) < 0.5).long() if fake else strip.clone()
    felttok = torch.where(felt.bool(), torch.full_like(felt, BUMP), torch.full_like(felt, FLAT))
    truetok = torch.where(strip.bool(), torch.full_like(strip, BUMP), torch.full_like(strip, FLAT))
    occ = torch.rand(batch, strip_len, generator=generator).argsort(dim=1)[:, :n_occluded].sort(dim=1).values
    occ_mask = torch.zeros(batch, strip_len, dtype=torch.bool).scatter(1, occ, True)
    for i in range(strip_len):
        full[:, 2 * i] = POSBASE + i
        if touch:
            full[:, 2 * i + 1] = torch.where(occ_mask[:, i], torch.full((batch,), BLANK, dtype=torch.long), felttok[:, i])
        else:
            full[:, 2 * i + 1] = BLANK
    full[:, 2 * strip_len] = COUNT
    full[:, 2 * strip_len + 1] = countbase + strip.sum(dim=1)
    inp = full[:, :-1].contiguous().to(device)
    tgt = full[:, 1:].contiguous().to(device)
    true_occ = torch.gather(truetok, 1, occ).to(device)
    return inp, tgt, occ.to(device), true_occ


def count_acc(logits, tgt, strip_len):
    pred = logits[:, 2 * strip_len].argmax(-1)
    return (pred == tgt[:, 2 * strip_len]).float().mean().item()


def build(args, device):
    cfg = ModelConfig(vocab_size=5 + 2 * args.strip_len, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode="linear", mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    return SequenceModel(cfg).to(device)


def train_model(args, device, p_stay, touch, fake):
    torch.manual_seed(args.seed)
    model = build(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed + 1)
    ar = torch.arange(args.batch, device=device)[:, None]
    sl = args.strip_len
    model.train()
    for step in range(args.steps):
        inp, tgt, occ, true_occ = make_fillback(args.batch, sl, args.n_occluded, p_stay, touch, device, g, fake)
        pred_idx = 2 * occ
        fill_idx = 2 * occ + 1
        inp_full = inp.clone()
        inp_full[ar, fill_idx] = true_occ
        logits_occ, _ = model(inp)
        logits_full, _ = model(inp_full)
        imag_logits = logits_occ[ar, pred_idx]
        v = imag_logits.size(-1)
        imag_loss = F.cross_entropy(imag_logits.reshape(-1, v), true_occ.reshape(-1))
        count_loss = F.cross_entropy(logits_full[:, 2 * sl], tgt[:, 2 * sl])
        loss = count_loss + args.imag_weight * imag_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def eval_fill(model, args, device, p_stay, touch, fake, mode):
    model.eval()
    sl = args.strip_len
    ar = torch.arange(args.batch, device=device)[:, None]
    g = torch.Generator().manual_seed(args.seed + 7)
    accs = []
    done = 0
    while done < args.eval_trials:
        inp, tgt, occ, true_occ = make_fillback(args.batch, sl, args.n_occluded, p_stay, touch, device, g, fake)
        pred_idx = 2 * occ
        fill_idx = 2 * occ + 1
        if mode == "nofill":
            logits, _ = model(inp)
        else:
            logits_occ, _ = model(inp)
            if mode == "oracle":
                fillv = true_occ
            elif mode == "random":
                fillv = torch.randint(FLAT, BUMP + 1, true_occ.shape, generator=g).to(device)
            else:
                fillv = logits_occ[ar, pred_idx].argmax(-1)
            inp2 = inp.clone()
            inp2[ar, fill_idx] = fillv
            logits, _ = model(inp2)
        accs.append(count_acc(logits, tgt, sl))
        done += args.batch
    model.train()
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=8)
    ap.add_argument("--n_occluded", type=int, default=1)
    ap.add_argument("--p_stay", type=float, default=0.85)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head_dim", type=int, default=24)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--imag_weight", type=float, default=1.0)
    ap.add_argument("--eval_trials", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = args.device
    print(f"fillback strip={args.strip_len} occ={args.n_occluded} p_stay={args.p_stay} "
          f"steps={args.steps} imag_weight={args.imag_weight} device={device}", flush=True)
    res = {}
    t0 = time.time()
    main_model = train_model(args, device, args.p_stay, True, False)
    for mode in ("oracle", "fillback", "nofill", "random"):
        res[mode] = eval_fill(main_model, args, device, args.p_stay, True, False, mode)
        print(f"  {mode:10s} count_acc={res[mode]:.3f} [{time.time() - t0:.1f}s]", flush=True)
    for name, p_stay, touch, fake in (("occ_random", None, True, False), ("fake", args.p_stay, True, True), ("blind", args.p_stay, False, False)):
        m = train_model(args, device, p_stay, touch, fake)
        res[name] = eval_fill(m, args, device, p_stay, touch, fake, "fillback")
        print(f"  {name:10s} count_acc={res[name]:.3f} [{time.time() - t0:.1f}s]", flush=True)
    passed = (res["fillback"] >= 0.80 and res["fillback"] - res["nofill"] >= 0.08 and res["oracle"] >= 0.90
              and res["random"] <= res["nofill"] + 0.05 and res["occ_random"] <= 0.60
              and res["fake"] <= 0.40 and res["blind"] <= 0.40)
    print(f"  GOAL {'PASS' if passed else 'FAIL'} (fillback {res['fillback']:.3f} >= 0.80 and "
          f"+{res['fillback'] - res['nofill']:.3f} over nofill {res['nofill']:.3f} >= 0.08, "
          f"oracle {res['oracle']:.3f}, random {res['random']:.3f}, occ_random {res['occ_random']:.3f}, "
          f"fake {res['fake']:.3f}, blind {res['blind']:.3f})", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
