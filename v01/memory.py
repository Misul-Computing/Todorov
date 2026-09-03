import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def _silu(x):
    return F.silu(x)


def _dsilu(a):
    s = torch.sigmoid(a)
    return s * (1.0 + a * (1.0 - s))


@dataclass
class MemoryConfig:
    d_model: int
    n_heads: int = 4
    d_head: int = 64
    mode: str = "mlp"
    d_hidden: int = 64
    forget_bias: float = -6.0
    lr_bias: float = 0.0
    momentum_bias: float = 0.0
    out_gate_bias: float = 0.0
    affect: float = 0.0
    affect_mode: str = "surprise"
    l2_norm_keys: bool = True
    conv_kernel: int = 4
    state_clamp: float = 100.0


class DescentMemory(nn.Module):
    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        H, dk, dv = cfg.n_heads, cfg.d_head, cfg.d_head
        self.H, self.dk, self.dv, self.dh = H, dk, dv, cfg.d_hidden
        self.q_proj = nn.Linear(cfg.d_model, H * dk, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, H * dk, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, H * dv, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, H * 3, bias=True)
        self.out_gate = nn.Linear(cfg.d_model, H, bias=True)
        self.out_proj = nn.Linear(H * dv, cfg.d_model, bias=False)
        self.conv = None
        if cfg.conv_kernel > 0:
            self.conv = nn.Conv1d(cfg.d_model, cfg.d_model, cfg.conv_kernel,
                                  groups=cfg.d_model, bias=True)
        self.W1_init = None
        if cfg.mode == "mlp":
            self.W1_init = nn.Parameter(torch.randn(H, cfg.d_hidden, dk) / math.sqrt(dk))
        self._reset_gate_bias()

    def _reset_gate_bias(self):
        H = self.H
        with torch.no_grad():
            self.gate_proj.bias[0:H].fill_(self.cfg.lr_bias)
            self.gate_proj.bias[H:2 * H].fill_(self.cfg.forget_bias)
            self.gate_proj.bias[2 * H:3 * H].fill_(self.cfg.momentum_bias)
            self.out_gate.bias.fill_(self.cfg.out_gate_bias)

    def _project(self, x):
        B, T, _ = x.shape
        if self.conv is not None:
            kc = self.cfg.conv_kernel
            xc = F.pad(x.transpose(1, 2), (kc - 1, 0))
            xc = self.conv(xc).transpose(1, 2)
        else:
            xc = x
        q = self.q_proj(xc).view(B, T, self.H, self.dk)
        k = self.k_proj(xc).view(B, T, self.H, self.dk)
        v = self.v_proj(xc).view(B, T, self.H, self.dv)
        if self.cfg.l2_norm_keys:
            k = F.normalize(k, dim=-1)
            q = F.normalize(q, dim=-1)
        g = self.gate_proj(xc).view(B, T, 3, self.H).float()
        beta = torch.sigmoid(g[:, :, 0])
        lam = torch.sigmoid(g[:, :, 1])
        mu = torch.sigmoid(g[:, :, 2])
        return q.float(), k.float(), v.float(), beta, lam, mu

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v, beta, lam, mu = self._project(x)
        dev = x.device
        H, dk, dv, dh = self.H, self.dk, self.dv, self.dh
        affect = self.cfg.affect
        mode = self.cfg.affect_mode
        gperm = torch.Generator().manual_seed(0) if mode in ("shuffle", "noise", "noise_token", "smooth_noise", "smooth_noise_token") else None
        s_sum = torch.zeros((), device=dev)
        g_sum = torch.zeros((), device=dev)
        s_run = torch.zeros(B, H, device=dev)
        s_smooth = torch.zeros(B, H, device=dev)
        s_smooth_tok = torch.zeros(B, 1, device=dev)
        s_cnt = 0
        if self.cfg.mode == "linear":
            W = torch.zeros(B, H, dv, dk, device=dev, dtype=torch.float32)
            Mm = torch.zeros_like(W)
            outs = []
            for t in range(T):
                kt, vt, qt = k[:, t], v[:, t], q[:, t]
                bt = beta[:, t][..., None, None]
                lt = lam[:, t][..., None, None]
                mt = mu[:, t][..., None, None]
                pred = torch.einsum("bhvk,bhk->bhv", W, kt)
                e = pred - vt
                if affect > 0.0:
                    enorm = e.norm(dim=-1)
                    s = (enorm / (enorm + vt.norm(dim=-1) + 1e-6)).detach()
                    if mode == "shuffle":
                        s = s[torch.randperm(B, generator=gperm).to(dev)]
                    elif mode == "noise":
                        s = torch.rand(B, H, generator=gperm).to(dev)
                    elif mode == "noise_token":
                        s = torch.rand(B, 1, generator=gperm).to(dev).expand(B, H).contiguous()
                    elif mode == "smooth_noise":
                        s_smooth = 0.7 * s_smooth + 0.3 * torch.rand(B, H, generator=gperm).to(dev)
                        s = s_smooth
                    elif mode == "smooth_noise_token":
                        s_smooth_tok = 0.7 * s_smooth_tok + 0.3 * torch.rand(B, 1, generator=gperm).to(dev)
                        s = s_smooth_tok.expand(B, H).contiguous()
                    if mode == "surprise_batchnorm":
                        rel = s / (s.mean(dim=0, keepdim=True) + 1e-6)
                    else:
                        s_run = s_run + s
                        s_cnt += 1
                        rel = s / (s_run / s_cnt + 1e-6)
                    gain = (1.0 - affect + affect * rel)[..., None, None]
                    bt = bt * gain
                    s_sum = s_sum + s.mean()
                    g_sum = g_sum + gain.mean()
                gW = torch.einsum("bhv,bhk->bhvk", e, kt)
                Mm = mt * Mm + gW
                W = (1.0 - lt) * W - bt * Mm
                W = W * (self.cfg.state_clamp / W.detach().norm(dim=(-2, -1), keepdim=True).clamp_min(self.cfg.state_clamp))
                outs.append(torch.einsum("bhvk,bhk->bhv", W, qt))
            o = torch.stack(outs, dim=1)
            state_norm = W.detach().float().norm(dim=(-2, -1)).mean()
        else:
            W1 = self.W1_init.to(torch.float32)[None].expand(B, H, dh, dk).contiguous()
            W2 = torch.zeros(B, H, dv, dh, device=dev, dtype=torch.float32)
            M1 = torch.zeros_like(W1)
            M2 = torch.zeros_like(W2)
            outs = []
            for t in range(T):
                kt, vt, qt = k[:, t], v[:, t], q[:, t]
                bt = beta[:, t][..., None, None]
                lt = lam[:, t][..., None, None]
                mt = mu[:, t][..., None, None]
                a = torch.einsum("bhpk,bhk->bhp", W1, kt)
                h = _silu(a)
                pred = torch.einsum("bhvp,bhp->bhv", W2, h)
                e = pred - vt
                if affect > 0.0:
                    enorm = e.norm(dim=-1)
                    s = (enorm / (enorm + vt.norm(dim=-1) + 1e-6)).detach()
                    if mode == "shuffle":
                        s = s[torch.randperm(B, generator=gperm).to(dev)]
                    elif mode == "noise":
                        s = torch.rand(B, H, generator=gperm).to(dev)
                    elif mode == "noise_token":
                        s = torch.rand(B, 1, generator=gperm).to(dev).expand(B, H).contiguous()
                    elif mode == "smooth_noise":
                        s_smooth = 0.7 * s_smooth + 0.3 * torch.rand(B, H, generator=gperm).to(dev)
                        s = s_smooth
                    elif mode == "smooth_noise_token":
                        s_smooth_tok = 0.7 * s_smooth_tok + 0.3 * torch.rand(B, 1, generator=gperm).to(dev)
                        s = s_smooth_tok.expand(B, H).contiguous()
                    if mode == "surprise_batchnorm":
                        rel = s / (s.mean(dim=0, keepdim=True) + 1e-6)
                    else:
                        s_run = s_run + s
                        s_cnt += 1
                        rel = s / (s_run / s_cnt + 1e-6)
                    gain = (1.0 - affect + affect * rel)[..., None, None]
                    bt = bt * gain
                    s_sum = s_sum + s.mean()
                    g_sum = g_sum + gain.mean()
                gW2 = torch.einsum("bhv,bhp->bhvp", e, h) / ((h * h).sum(-1)[..., None, None] + 1.0)
                back = torch.einsum("bhvp,bhv->bhp", W2, e)
                da = back * _dsilu(a)
                gW1 = torch.einsum("bhp,bhk->bhpk", da, kt) / ((kt * kt).sum(-1)[..., None, None] + 1.0)
                M1 = mt * M1 + gW1
                M2 = mt * M2 + gW2
                W1 = (1.0 - lt) * W1 - bt * M1
                W2 = (1.0 - lt) * W2 - bt * M2
                sc = self.cfg.state_clamp
                W1 = W1 * (sc / W1.detach().norm(dim=(-2, -1), keepdim=True).clamp_min(sc))
                W2 = W2 * (sc / W2.detach().norm(dim=(-2, -1), keepdim=True).clamp_min(sc))
                aq = torch.einsum("bhpk,bhk->bhp", W1, qt)
                outs.append(torch.einsum("bhvp,bhp->bhv", W2, _silu(aq)))
            o = torch.stack(outs, dim=1)
            state_norm = (W1.detach().float().norm(dim=(-2, -1)) + W2.detach().float().norm(dim=(-2, -1))).mean()
        o = o.to(x.dtype)
        gate = torch.sigmoid(self.out_gate(x))[..., None]
        self.last_stats = {
            "lam": lam.detach().mean(),
            "beta": beta.detach().mean(),
            "mu": mu.detach().mean(),
            "out_gate": gate.detach().mean(),
            "state_norm": state_norm,
            "surprise": (s_sum / T).detach() if affect > 0.0 else torch.full((), float("nan"), device=dev),
            "write_gain": (g_sum / T).detach() if affect > 0.0 else torch.full((), float("nan"), device=dev),
        }
        o = (o * gate).reshape(B, T, H * dv)
        return self.out_proj(o)

    @torch.no_grad()
    def retention_floor(self, x, seq_len):
        _, _, _, _, lam, _ = self._project(x)
        keep = (1.0 - lam).mean().item()
        return keep ** seq_len
