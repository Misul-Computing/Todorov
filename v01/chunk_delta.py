import sys
import os
import time
import argparse
import torch


def delta_recurrent(q, k, v, beta):
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    s = torch.zeros(b, h, dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    for i in range(t):
        kt, vt, qt, bt = k[:, :, i], v[:, :, i], q[:, :, i], beta[:, :, i]
        kts = torch.einsum("bhk,bhkv->bhv", kt, s)
        s = s + bt[..., None, None] * kt[..., :, None] * (vt - kts)[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, s))
    return torch.stack(outs, 2)


def tril_inv(a):
    c = a.shape[-1]
    eye = torch.eye(c, dtype=a.dtype, device=a.device)
    inv = eye + a
    apow = a
    j = 1
    while (1 << j) < c:
        apow = apow @ apow
        inv = inv @ (eye + apow)
        j += 1
    return inv


def delta_chunkwise(q, k, v, beta, chunk):
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    assert t % chunk == 0
    nc = t // chunk
    qc = q.view(b, h, nc, chunk, dk)
    kc = k.view(b, h, nc, chunk, dk)
    vc = v.view(b, h, nc, chunk, dv)
    bc = beta.view(b, h, nc, chunk)
    bk = bc[..., None] * kc
    bv = bc[..., None] * vc
    a = torch.tril(-bc[..., None] * (kc @ kc.transpose(-1, -2)), -1)
    tm = tril_inv(a)
    w = tm @ bk
    u = tm @ bv
    m = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device))
    s = torch.zeros(b, h, dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    for c in range(nc):
        wc, uc, qcc, kcc = w[:, :, c], u[:, :, c], qc[:, :, c], kc[:, :, c]
        umws = uc - wc @ s
        o1 = qcc @ s
        o2 = ((qcc @ kcc.transpose(-1, -2)) * m) @ umws
        outs.append(o1 + o2)
        s = s + kcc.transpose(-1, -2) @ umws
        s = s * torch.clamp(100.0 / s.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6), max=1.0)
    return torch.stack(outs, 2).reshape(b, h, t, dv)


def delta_recurrent_gated(q, k, v, beta, g):
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    s = torch.zeros(b, h, dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    for i in range(t):
        kt, vt, qt, bt, gt = k[:, :, i], v[:, :, i], q[:, :, i], beta[:, :, i], g[:, :, i]
        kts = torch.einsum("bhk,bhkv->bhv", kt, s)
        erase = bt[..., None, None] * kt[..., :, None] * kts[..., None, :]
        write = bt[..., None, None] * kt[..., :, None] * vt[..., None, :]
        s = gt[..., None, None] * (s - erase) + write
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, s))
    return torch.stack(outs, 2)


def delta_chunkwise_gated(q, k, v, beta, g, chunk):
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    assert t % chunk == 0
    nc = t // chunk
    gc = g.view(b, h, nc, chunk)
    gcum = torch.exp(torch.cumsum(torch.log(gc.clamp_min(1e-9)), dim=-1))
    gtot = gcum[..., -1]
    qc = q.view(b, h, nc, chunk, dk) * gcum[..., None]
    kc = k.view(b, h, nc, chunk, dk)
    vc = v.view(b, h, nc, chunk, dv) / gcum[..., None].clamp_min(1e-9)
    bc = beta.view(b, h, nc, chunk)
    a = torch.tril(-bc[..., None] * (kc @ kc.transpose(-1, -2)), -1)
    tm = tril_inv(a)
    w = tm @ (bc[..., None] * kc)
    u = tm @ (bc[..., None] * vc)
    mm = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device))
    s = torch.zeros(b, h, dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    for c in range(nc):
        wcc, ucc, qcc, kcc = w[:, :, c], u[:, :, c], qc[:, :, c], kc[:, :, c]
        umws = ucc - wcc @ s
        outs.append(qcc @ s + ((qcc @ kcc.transpose(-1, -2)) * mm) @ umws)
        s = (s + kcc.transpose(-1, -2) @ umws) * gtot[:, :, c][..., None, None]
        s = s * torch.clamp(100.0 / s.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6), max=1.0)
    return torch.stack(outs, 2).reshape(b, h, t, dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device

    print("=== correctness: chunkwise vs recurrent ===", flush=True)
    torch.manual_seed(0)
    b, h, t, d, c = 2, 4, 64, 32, 16
    q = torch.nn.functional.normalize(torch.randn(b, h, t, d, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, d, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, t, d, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64))
    o_rec = delta_recurrent(q, k, v, beta)
    for c in (8, 16, 32):
        o_chunk = delta_chunkwise(q, k, v, beta, c)
        diff = (o_rec - o_chunk).abs().max().item()
        print(f"  chunk={c:2d}  max_abs_diff={diff:.2e}  {'MATCH' if diff < 1e-9 else 'MISMATCH'}", flush=True)

    print("=== gated correctness: chunkwise vs recurrent ===", flush=True)
    torch.manual_seed(1)
    bg, hg, tg, dg = 2, 4, 64, 32
    qg = torch.nn.functional.normalize(torch.randn(bg, hg, tg, dg, dtype=torch.float64), dim=-1)
    kg = torch.nn.functional.normalize(torch.randn(bg, hg, tg, dg, dtype=torch.float64), dim=-1)
    vg = torch.randn(bg, hg, tg, dg, dtype=torch.float64)
    betag = torch.sigmoid(torch.randn(bg, hg, tg, dtype=torch.float64))
    gg = torch.sigmoid(torch.randn(bg, hg, tg, dtype=torch.float64) + 2.0)
    o_rec_g = delta_recurrent_gated(qg, kg, vg, betag, gg)
    for cc in (8, 16, 32):
        o_chunk_g = delta_chunkwise_gated(qg, kg, vg, betag, gg, cc)
        diff = (o_rec_g - o_chunk_g).abs().max().item()
        print(f"  chunk={cc:2d}  max_abs_diff={diff:.2e}  {'MATCH' if diff < 1e-9 else 'MISMATCH'}", flush=True)

    print("=== speed: cpu vs mps (mac gpu), chunkwise float32 ===", flush=True)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for t in (256, 1024, 4096, 8192):
        b, h, d, c = 8, 4, 64, 64
        row = {}
        for dv in devices:
            q = torch.nn.functional.normalize(torch.randn(b, h, t, d, device=dv), dim=-1)
            k = torch.nn.functional.normalize(torch.randn(b, h, t, d, device=dv), dim=-1)
            v = torch.randn(b, h, t, d, device=dv)
            beta = torch.sigmoid(torch.randn(b, h, t, device=dv))
            for _ in range(3):
                delta_chunkwise(q, k, v, beta, c)
            if dv == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            for _ in range(8):
                delta_chunkwise(q, k, v, beta, c)
            if dv == "mps":
                torch.mps.synchronize()
            row[dv] = (time.time() - t0) / 8 * 1000
        line = f"  T={t:5d}  " + "  ".join(f"{dv}={row[dv]:7.1f}ms" for dv in devices)
        if "mps" in row:
            line += f"  mps_speedup={row['cpu'] / row['mps']:.1f}x"
        print(line, flush=True)


if __name__ == "__main__":
    main()
