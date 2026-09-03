import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import RMSNorm, SwiGLU
import data as datamod
import evals


class GatedDeltaMemory(nn.Module):
    def __init__(self, d_model, n_heads=4, d_head=32, conv_kernel=4):
        super().__init__()
        self.h, self.dk, self.dv = n_heads, d_head, d_head
        self.q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.a = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.b = nn.Linear(d_model, n_heads, bias=True)
        self.g = nn.Linear(d_model, n_heads * d_head, bias=True)
        self.o = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.ck = conv_kernel
        self.conv = nn.Conv1d(d_model, d_model, conv_kernel, groups=d_model, bias=True) if conv_kernel > 0 else None
        with torch.no_grad():
            self.a.bias.fill_(4.0)

    def forward(self, x):
        b, t, d = x.shape
        if self.conv is not None:
            xc = F.pad(x.transpose(1, 2), (self.ck - 1, 0))
            xc = self.conv(xc).transpose(1, 2)
        else:
            xc = x
        q = F.normalize(self.q(xc).view(b, t, self.h, self.dk).float(), dim=-1)
        k = F.normalize(self.k(xc).view(b, t, self.h, self.dk).float(), dim=-1)
        v = self.v(xc).view(b, t, self.h, self.dv).float()
        alpha = torch.sigmoid(self.a(xc).view(b, t, self.h, self.dk).float())
        beta = torch.sigmoid(self.b(xc).view(b, t, self.h, 1).float())
        s = torch.zeros(b, self.h, self.dv, self.dk, device=x.device, dtype=torch.float32)
        outs = []
        for i in range(t):
            kt, vt, qt = k[:, i], v[:, i], q[:, i]
            at, bt = alpha[:, i], beta[:, i]
            s = s * at[:, :, None, :]
            pred = torch.einsum("bhvk,bhk->bhv", s, kt)
            s = s + (bt * (vt - pred))[..., None] * kt[:, :, None, :]
            outs.append(torch.einsum("bhvk,bhk->bhv", s, qt))
        out = torch.stack(outs, 1).reshape(b, t, self.h * self.dv).to(x.dtype)
        out = out * torch.sigmoid(self.g(xc))
        return self.o(out)


class GDNModel(nn.Module):
    def __init__(self, vocab, d_model=96, n_layers=2, n_heads=4, d_head=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"n1": RMSNorm(d_model), "mix": GatedDeltaMemory(d_model, n_heads, d_head),
                           "n2": RMSNorm(d_model), "mlp": SwiGLU(d_model, 4.0)})
            for _ in range(n_layers)])
        self.nf = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)
        self.head.weight = self.embed.weight

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = x + blk["mix"](blk["n1"](x))
            x = x + blk["mlp"](blk["n2"](x))
        logits = self.head(self.nf(x))
        loss = None
        if targets is not None:
            lf = logits.float()
            ll = F.cross_entropy(lf.view(-1, lf.size(-1)), targets.reshape(-1), reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1)
                loss = (ll * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss = ll.mean()
        return logits, loss


def train_eval(mk, vocab, steps, device, batch=32, lr=3e-3):
    torch.manual_seed(0)
    model = GDNModel(vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for _ in range(steps):
        inp, tgt, mask = mk(batch, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return evals.eval_task(model, mk, 100, batch, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    print(f"gated-delta memory self-test device={dev}", flush=True)

    strip, vocab = 6, 10
    print("feel recall (regression, must pass real >> blind/fake):", flush=True)
    for label, touch, noise in (("blind", False, False), ("real", True, False), ("fake", True, True)):
        g = torch.Generator().manual_seed(1)
        mk = (lambda tc, ns: (lambda b, d: datamod.make_touch_world(b, strip, 3, 0.5, tc, d, generator=g, noise=ns)))(touch, noise)
        ev = train_eval(mk, 4 + strip, args.steps, dev)
        print(f"  {label:5s} exact={ev['exact_acc']:.3f} token={ev['token_acc']:.3f} [{time.time() - t0:.1f}s]", flush=True)

    print("mqar capacity (associative recall, where the simple linear memory was weak):", flush=True)
    for n_pairs in (6, 12, 18):
        g = torch.Generator().manual_seed(2)
        mk = (lambda npr: (lambda b, d: datamod.make_mqar(b, npr, 4, 32, d, generator=g)))(n_pairs)
        ev = train_eval(mk, 32, args.steps, dev)
        print(f"  pairs={n_pairs:2d} exact={ev['exact_acc']:.3f} token={ev['token_acc']:.3f} [{time.time() - t0:.1f}s]", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
