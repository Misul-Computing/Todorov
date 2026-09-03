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

BLANK, FLAT, BUMP, QUERY, COUNT = 0, 1, 2, 3, 4
POSBASE = 5


def make_world(batch, strip_len, p_stay, touch, gen, fake):
    if p_stay is None:
        strip = (torch.rand(batch, strip_len, generator=gen) < 0.5).long()
    else:
        strip = _structured_strip(batch, strip_len, p_stay, gen)
    felt = (torch.rand(batch, strip_len, generator=gen) < 0.5).long() if fake else strip.clone()
    felttok = torch.where(felt.bool(), torch.full_like(felt, BUMP), torch.full_like(felt, FLAT))
    truetok = torch.where(strip.bool(), torch.full_like(strip, BUMP), torch.full_like(strip, FLAT))
    hidden = torch.randint(0, strip_len, (batch,), generator=gen)
    ar = torch.arange(batch)
    shown = felttok.clone()
    if not touch:
        shown[:] = BLANK
    else:
        shown[ar, hidden] = BLANK
    return shown, hidden, truetok[ar, hidden], strip.sum(dim=1)


def interleave(shown, device):
    b, sl = shown.shape
    pos = (POSBASE + torch.arange(sl)).expand(b, sl)
    return torch.stack([pos, shown], dim=2).reshape(b, 2 * sl).to(device)


def build_recall(shown, hidden, answer, device, with_answer):
    b, sl = shown.shape
    sweep = interleave(shown, device)
    q = torch.full((b, 1), QUERY, dtype=torch.long, device=device)
    hp = (POSBASE + hidden).unsqueeze(1).to(device)
    parts = [sweep, q, hp]
    if with_answer:
        parts.append(answer.unsqueeze(1).to(device))
    return torch.cat(parts, dim=1)


def build_count(shown, count, strip_len, device):
    b, sl = shown.shape
    sweep = interleave(shown, device)
    c = torch.full((b, 1), COUNT, dtype=torch.long, device=device)
    cv = (POSBASE + strip_len + count).unsqueeze(1).to(device)
    return torch.cat([sweep, c, cv], dim=1)


def masked_ce(logits, tgt, mask):
    v = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, v), tgt.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def build(args, device):
    cfg = ModelConfig(vocab_size=6 + 2 * args.strip_len, d_model=args.d_model, n_layers=args.layers,
                      n_heads=4, mem_mode="linear", mem_heads=4, mem_head_dim=args.head_dim,
                      mem_hidden=args.head_dim, forget_bias=-6.0, affect=0.0)
    return SequenceModel(cfg).to(device)


def train_model(args, device, p_stay, touch, fake):
    torch.manual_seed(args.seed)
    model = build(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed + 1)
    ar = torch.arange(args.batch)
    sl = args.strip_len
    model.train()
    for step in range(args.steps):
        shown, hidden, answer, count = make_world(args.batch, sl, p_stay, touch, g, fake)
        rseq = build_recall(shown, hidden, answer, device, True)
        rl, _ = model(rseq[:, :-1])
        rmask = torch.zeros_like(rseq[:, :-1], dtype=torch.float32)
        rmask[:, 2 * sl + 1] = 1.0
        recall_loss = masked_ce(rl, rseq[:, 1:], rmask)
        shown_full = shown.clone()
        shown_full[ar, hidden] = answer
        cseq = build_count(shown_full, count, sl, device)
        cl, _ = model(cseq[:, :-1])
        cmask = torch.zeros_like(cseq[:, :-1], dtype=torch.float32)
        cmask[:, 2 * sl] = 1.0
        count_loss = masked_ce(cl, cseq[:, 1:], cmask)
        loss = count_loss + args.imag_weight * recall_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def eval_mode(model, args, device, p_stay, touch, fake, mode):
    model.eval()
    g = torch.Generator().manual_seed(args.seed + 7)
    ar = torch.arange(args.batch)
    sl = args.strip_len
    accs = []
    done = 0
    while done < args.eval_trials:
        shown, hidden, answer, count = make_world(args.batch, sl, p_stay, touch, g, fake)
        use = shown.clone()
        if mode != "nofill":
            if mode == "oracle":
                fillv = answer
            elif mode == "random":
                fillv = torch.randint(FLAT, BUMP + 1, (args.batch,), generator=g)
            else:
                rseq = build_recall(shown, hidden, answer, device, False)
                rl, _ = model(rseq)
                fillv = rl[:, -1].argmax(-1).cpu()
            use[ar, hidden] = fillv
        cseq = build_count(use, count, sl, device)
        cl, _ = model(cseq[:, :-1])
        pred = cl[:, 2 * sl].argmax(-1)
        accs.append((pred == cseq[:, 1:][:, 2 * sl]).float().mean().item())
        done += args.batch
    model.train()
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip_len", type=int, default=8)
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
    print(f"fillback2 (bidirectional) strip={args.strip_len} p_stay={args.p_stay} "
          f"steps={args.steps} imag_weight={args.imag_weight} seed={args.seed} device={device}", flush=True)
    res = {}
    t0 = time.time()
    m = train_model(args, device, args.p_stay, True, False)
    for mode in ("oracle", "fillback", "nofill", "random"):
        res[mode] = eval_mode(m, args, device, args.p_stay, True, False, mode)
        print(f"  {mode:10s} count_acc={res[mode]:.3f} [{time.time() - t0:.1f}s]", flush=True)
    for name, p_stay, touch, fake in (("occ_random", None, True, False), ("fake", args.p_stay, True, True), ("blind", args.p_stay, False, False)):
        mc = train_model(args, device, p_stay, touch, fake)
        res[name] = eval_mode(mc, args, device, p_stay, touch, fake, "fillback")
        print(f"  {name:10s} count_acc={res[name]:.3f} [{time.time() - t0:.1f}s]", flush=True)
    passed = (res["fillback"] >= 0.80 and res["fillback"] - res["nofill"] >= 0.08 and res["oracle"] >= 0.90
              and res["random"] <= res["nofill"] + 0.05 and res["occ_random"] <= 0.60
              and res["fake"] <= 0.40 and res["blind"] <= 0.40)
    print(f"  GOAL {'PASS' if passed else 'FAIL'} (fillback {res['fillback']:.3f} >= 0.80, "
          f"+{res['fillback'] - res['nofill']:.3f} over nofill, oracle {res['oracle']:.3f}, "
          f"random {res['random']:.3f}, occ_random {res['occ_random']:.3f}, fake {res['fake']:.3f}, blind {res['blind']:.3f})", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
