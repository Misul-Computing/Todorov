import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import LM
import data as datamod


def masked_ce(logits, tgt, mask):
    ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def eval_feel(model, mk, trials, batch, device):
    model.eval()
    cor = tot = cex = tex = 0
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        inp, tgt, mask = mk(b, device)
        pred = model(inp).argmax(-1)
        mb = mask.bool()
        cor += int(((pred == tgt) & mb).sum())
        tot += int(mb.sum())
        for i in range(b):
            mi = mb[i]
            if bool(mi.any()):
                cex += int(bool(((pred[i] == tgt[i]) | ~mi).all()))
                tex += 1
        done += b
    model.train()
    return cex / max(1, tex), cor / max(1, tot)


def run_arm(touch, noise, args, device):
    torch.manual_seed(0)
    vocab = 4 + args.strip_len
    model = LM(vocab, args.pattern, args.d_model, args.layers, args.n_heads, args.d_head, conv_kernel=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(1)

    def mk(b, dev):
        return datamod.make_touch_world(b, args.strip_len, args.n_queries, 0.5, touch, dev, generator=g, noise=noise)

    model.train()
    for _ in range(args.steps):
        inp, tgt, mask = mk(args.batch, device)
        opt.zero_grad(set_to_none=True)
        loss = masked_ce(model(inp), tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return eval_feel(model, mk, 100, args.batch, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="attn,delta,delta,delta")
    ap.add_argument("--strip_len", type=int, default=6)
    ap.add_argument("--n_queries", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    device = args.device
    print(f"feel grounding on Transformerov hybrid pattern={args.pattern} layers={args.layers} "
          f"steps={args.steps} device={device}", flush=True)
    t0 = time.time()
    res = {}
    for label, touch, noise in (("blind", False, False), ("real", True, False), ("fake", True, True)):
        ex, tok = run_arm(touch, noise, args, device)
        res[label] = ex
        print(f"  {label:5s} exact={ex:.3f} token={tok:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    grounded = res["real"] >= 0.9 and res["blind"] <= 0.5 and res["fake"] <= 0.5
    print(f"  GROUNDED {'YES' if grounded else 'NO'} (real {res['real']:.3f} >> blind {res['blind']:.3f} / fake {res['fake']:.3f})", flush=True)


if __name__ == "__main__":
    main()
