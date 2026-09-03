import sys
import os
import time
import math
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdn import GDNModel
from model import SequenceModel, ModelConfig


def load_bytes(path):
    with open(path, "rb") as f:
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def get_batch(data, seq_len, batch, device, gen):
    ix = torch.randint(0, data.numel() - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def val_bpb(model, data, seq_len, batch, device, n_batches=20):
    g = torch.Generator().manual_seed(123)
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch, device, g)
        logits, _ = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return (tot / n_batches) / math.log(2)


def train(model, tr, va, args, device, tag):
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(1)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(tr, args.seq_len, args.batch, device, g)
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            print(f"  [{tag}] step {step} train_bpb {loss.item() / math.log(2):.3f} [{time.time() - t0:.1f}s]", flush=True)
    return val_bpb(model, va, args.seq_len, args.batch, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/private/tmp/claude-501/-Users-dttdrv-Projects-Transformerov/1f740f01-a998-4e55-83a2-87facb6845a7/scratchpad/corpus_raw.txt")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = args.device

    data = load_bytes(args.corpus)
    n = data.numel()
    split = int(n * 0.9)
    tr, va = data[:split], data[split:]
    print(f"corpus {args.corpus} bytes={n} train={tr.numel()} val={va.numel()} "
          f"seq_len={args.seq_len} steps={args.steps} device={device}", flush=True)

    print("transformer baseline:", flush=True)
    cfg = ModelConfig(vocab_size=256, d_model=args.d_model, n_layers=args.layers, n_heads=4,
                      layer_kinds=tuple(["attn"] * args.layers))
    tf = SequenceModel(cfg).to(device)
    tf_bpb = train(tf, tr, va, args, device, "tf")
    tf_params = tf.num_params()

    print("gated-delta memory:", flush=True)
    gdn = GDNModel(256, d_model=args.d_model, n_layers=args.layers, n_heads=4, d_head=args.d_model // 4).to(device)
    gdn_params = sum(p.numel() for p in gdn.parameters())
    gdn_bpb = train(gdn, tr, va, args, device, "gdn")

    ratio = gdn_bpb / tf_bpb
    print(f"RESULT transformer val_bpb={tf_bpb:.4f} ({tf_params} params) | "
          f"gdn val_bpb={gdn_bpb:.4f} ({gdn_params} params) | ratio={ratio:.3f}x "
          f"({'gdn better' if ratio < 1.0 else 'transformer better'})", flush=True)


if __name__ == "__main__":
    main()
