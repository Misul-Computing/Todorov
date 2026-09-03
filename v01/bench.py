import sys
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_delta import delta_chunkwise_gated
from model import RMSNorm, SwiGLU, CausalAttention


class ChunkDeltaMemory(nn.Module):
    def __init__(self, d_model, n_heads, d_head, chunk=64):
        super().__init__()
        self.h, self.dk, self.dv, self.chunk = n_heads, d_head, d_head, chunk
        self.q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.bp = nn.Linear(d_model, n_heads, bias=True)
        self.ag = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.ag.bias, 0.4)
        self.onorm = RMSNorm(d_head)
        self.g = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        pad = (self.chunk - t % self.chunk) % self.chunk
        q = F.normalize(self.q(x).view(b, t, self.h, self.dk).transpose(1, 2).float(), dim=-1)
        k = F.normalize(self.k(x).view(b, t, self.h, self.dk).transpose(1, 2).float(), dim=-1)
        v = self.v(x).view(b, t, self.h, self.dv).transpose(1, 2).float()
        beta = torch.sigmoid(self.bp(x)).transpose(1, 2).float()
        alpha = (0.975 + 0.025 * torch.sigmoid(self.ag(x))).transpose(1, 2).float()
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            beta = F.pad(beta, (0, pad))
            alpha = F.pad(alpha, (0, pad), value=1.0)
        o = delta_chunkwise_gated(q, k, v, beta, alpha, self.chunk)[:, :, :t]
        o = self.onorm(o)
        o = o.transpose(1, 2).reshape(b, t, self.h * self.dv).to(x.dtype)
        o = o * torch.sigmoid(self.g(x))
        return self.o(o)


class LM(nn.Module):
    def __init__(self, vocab, mixer, d_model, n_layers, n_heads, d_head, conv_kernel=0):
        super().__init__()
        self.conv_kernel = conv_kernel
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList()
        kinds = mixer.split(",")
        for i in range(n_layers):
            kind = kinds[i % len(kinds)]
            mix = ChunkDeltaMemory(d_model, n_heads, d_head) if kind == "delta" else CausalAttention(d_model, n_heads)
            blk = {"n1": RMSNorm(d_model), "mix": mix, "n2": RMSNorm(d_model), "mlp": SwiGLU(d_model, 4.0)}
            if conv_kernel > 0:
                blk["conv"] = nn.Conv1d(d_model, d_model, conv_kernel, groups=d_model, bias=True)
            self.blocks.append(nn.ModuleDict(blk))
        self.nf = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)
        self.head.weight = self.embed.weight

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            h = blk["n1"](x)
            if "conv" in blk:
                h = blk["conv"](F.pad(h.transpose(1, 2), (self.conv_kernel - 1, 0))).transpose(1, 2)
            x = x + blk["mix"](h)
            x = x + blk["mlp"](blk["n2"](x))
        return self.head(self.nf(x))


def load_bytes(path):
    with open(path, "rb") as f:
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()


def get_batch(data, seq_len, batch, device, gen):
    ix = torch.randint(0, data.numel() - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def val_bpb(model, data, seq_len, batch, device, n_batches=40):
    g = torch.Generator().manual_seed(123)
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch, device, g)
        logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return (tot / n_batches) / math.log(2)


def lr_at(step, total, peak, warmup=100):
    if step < warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))


def run(mixer, vocab, tr, va, args, device):
    torch.manual_seed(args.seed)
    model = LM(vocab, mixer, args.d_model, args.layers, args.n_heads, args.d_head).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed + 1)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, args.steps, args.lr)
        x, y = get_batch(tr, args.seq_len, args.batch, device, g)
        opt.zero_grad(set_to_none=True)
        if args.amp == "off":
            logits = model(x)
        else:
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16):
                logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            print(f"  [{mixer}] step {step} train_bpb {loss.item() / math.log(2):.3f} [{time.time() - t0:.1f}s]", flush=True)
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"  saved {mixer} -> {args.save}", flush=True)
    return val_bpb(model, va, args.seq_len, args.batch, device), params, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/private/tmp/claude-501/-Users-dttdrv-Projects-Transformerov/1f740f01-a998-4e55-83a2-87facb6845a7/scratchpad/corpus_raw.txt")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixers", default="attn,delta,hybrid")
    ap.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    device = args.device
    data = load_bytes(args.corpus)
    n = data.numel()
    tr, va = data[:int(n * 0.9)], data[int(n * 0.9):]
    print(f"corpus bytes={n} seq_len={args.seq_len} d_model={args.d_model} layers={args.layers} "
          f"steps={args.steps} device={device}", flush=True)
    named = {"attn": "attn", "delta": "delta", "hybrid": "attn,delta,delta,delta"}
    out = {}
    for name in args.mixers.split(","):
        pat = named.get(name, name)
        print(f"{name}:", flush=True)
        bpb, params, secs = run(pat, 256, tr, va, args, device)
        out[name] = (bpb, params, secs)
        print(f"  -> val_bpb {bpb:.4f} | params {params} | {secs:.1f}s", flush=True)
    print("RESULT seed=" + str(args.seed) + " " + " | ".join(f"{n} {out[n][0]:.4f} ({out[n][2]:.0f}s)" for n in out), flush=True)


if __name__ == "__main__":
    main()
