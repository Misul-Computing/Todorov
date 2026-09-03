import sys
import os
import time
import json
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
def eval_mqar(model, mk, trials, batch, device):
    model.eval()
    cor_tok = tot_tok = cor_ex = tot_ex = 0
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        inp, tgt, mask = mk(b, device)
        pred = model(inp).argmax(-1)
        mb = mask.bool()
        cor_tok += int(((pred == tgt) & mb).sum())
        tot_tok += int(mb.sum())
        for i in range(b):
            mi = mb[i]
            if bool(mi.any()):
                cor_ex += int(bool(((pred[i] == tgt[i]) | ~mi).all()))
                tot_ex += 1
        done += b
    model.train()
    return cor_ex / max(1, tot_ex), cor_tok / max(1, tot_tok)


def run(mixer, n_pairs, args, device):
    torch.manual_seed(0)
    model = LM(args.vocab, mixer, args.d_model, args.layers, args.n_heads, args.d_head, conv_kernel=args.conv_kernel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(1)

    def mk(b, dev):
        return datamod.make_mqar(b, n_pairs, args.n_queries, args.vocab, dev, generator=g)

    model.train()
    for _ in range(args.steps):
        inp, tgt, mask = mk(args.batch, device)
        opt.zero_grad(set_to_none=True)
        loss = masked_ce(model(inp), tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return eval_mqar(model, mk, args.eval_trials, args.batch, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--conv_kernel", type=int, default=4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_trials", type=int, default=200)
    ap.add_argument("--pairs", default="4,8,16,32,64,128")
    ap.add_argument("--mixers", default="attn;delta;delta,delta,delta,attn")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = args.device
    pairs = [int(p) for p in args.pairs.split(",")]
    mixers = args.mixers.split(";")
    print(f"MQAR sweep d_model={args.d_model} layers={args.layers} steps={args.steps} "
          f"vocab={args.vocab} token_acc chance~{1.0 / (args.vocab - 1):.3f}", flush=True)
    print("patterns: " + "  ".join(f"[{i}]={m}" for i, m in enumerate(mixers)), flush=True)
    print(f"{'pairs':>6} " + " ".join(f"{'[' + str(i) + ']':>7}" for i in range(len(mixers))), flush=True)
    res = {}
    t0 = time.time()
    for p in pairs:
        row = {}
        for pat in mixers:
            ex, tok = run(pat, p, args, device)
            row[pat] = tok
        res[p] = row
        print(f"{p:>6} " + " ".join(f"{row[m]:>7.3f}" for m in mixers) + f"  [{time.time() - t0:.0f}s]", flush=True)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
