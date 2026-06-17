import math
from dataclasses import dataclass, field
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import DescentMemory, MemoryConfig


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return n * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4.0):
        super().__init__()
        hidden = int(d * mult)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _rope(x, base=10000.0):
    B, T, H, D = x.shape
    half = D // 2
    freqs = torch.exp(-math.log(base) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freqs[None, :]
    cos = ang.cos()[None, :, None, :]
    sin = ang.sin()[None, :, None, :]
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


class CausalAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.unbind(2)
        q = _rope(q)
        k = _rope(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.o(o)


class TernarySpike(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        theta = self.alpha * x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        s = torch.zeros_like(x)
        s = torch.where(x > theta, torch.ones_like(x), s)
        s = torch.where(x < -theta, -torch.ones_like(x), s)
        return x + (s - x).detach()


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mem_mode: str = "mlp"
    mem_heads: int = 4
    mem_head_dim: int = 64
    mem_hidden: int = 64
    mlp_mult: float = 4.0
    layer_kinds: Tuple[str, ...] = ()
    use_spikes: bool = False
    forget_bias: float = -6.0
    out_gate_bias: float = 0.0
    affect: float = 0.0
    affect_mode: str = "surprise"
    tie_embeddings: bool = True

    def resolved_kinds(self):
        if self.layer_kinds:
            assert len(self.layer_kinds) == self.n_layers
            return list(self.layer_kinds)
        return ["mem"] * self.n_layers


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model)
        if kind == "mem":
            self.mixer = DescentMemory(MemoryConfig(
                d_model=cfg.d_model, n_heads=cfg.mem_heads, d_head=cfg.mem_head_dim,
                mode=cfg.mem_mode, d_hidden=cfg.mem_hidden,
                forget_bias=cfg.forget_bias, out_gate_bias=cfg.out_gate_bias,
                affect=cfg.affect, affect_mode=cfg.affect_mode))
        elif kind == "attn":
            self.mixer = CausalAttention(cfg.d_model, cfg.n_heads)
        else:
            raise ValueError(kind)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_mult)
        self.spike = TernarySpike() if cfg.use_spikes else None

    def forward(self, x):
        h = self.norm1(x)
        if self.spike is not None:
            h = self.spike(h)
        x = x + self.mixer(h)
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        kinds = cfg.resolved_kinds()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, k) for k in kinds])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        for blk in self.blocks:
            if isinstance(blk.mixer, DescentMemory):
                blk.mixer._reset_gate_bias()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None, return_hidden=False):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            lf = logits.float()
            ll = F.cross_entropy(lf.view(-1, lf.size(-1)), targets.reshape(-1), reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (ll * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss = ll.mean()
        if return_hidden:
            return logits, loss, x
        return logits, loss

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.embed.weight.numel()
        return n
