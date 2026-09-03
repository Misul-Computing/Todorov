import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import LM


def step_once(model, x, y, opt, dt, device):
    opt.zero_grad(set_to_none=True)
    if dt is not None:
        with torch.autocast(device_type=device, dtype=dt):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
    else:
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    opt.step()
    return loss


def time_steps(model, x, y, opt, n, dt, device):
    for _ in range(4):
        step_once(model, x, y, opt, dt, device)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    last = None
    for _ in range(n):
        last = step_once(model, x, y, opt, dt, device)
    if device == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) / n * 1000, float(last.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pattern", default="attn,delta,delta,delta")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)
    x = torch.randint(0, 256, (args.batch, args.seq), device=device)
    y = torch.randint(0, 256, (args.batch, args.seq), device=device)
    params = sum(p.numel() for p in LM(256, args.pattern, args.d_model, args.layers, args.n_heads, args.d_head, conv_kernel=4).parameters())
    print(f"opt_bench pattern={args.pattern} d={args.d_model} layers={args.layers} seq={args.seq} "
          f"batch={args.batch} params={params} device={device}", flush=True)
    base = None
    for name, dt in (("float32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        torch.manual_seed(0)
        model = LM(256, args.pattern, args.d_model, args.layers, args.n_heads, args.d_head, conv_kernel=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        try:
            ms, loss = time_steps(model, x, y, opt, args.n, dt, device)
            if base is None:
                base = ms
            est = ms * 3000 / 1000 / 60
            print(f"  {name:8s} {ms:7.1f} ms/step  {base / ms:4.2f}x  loss={loss:.3f}  "
                  f"(3000 steps ~ {est:.1f} min)", flush=True)
        except Exception as e:
            print(f"  {name:8s} FAILED: {repr(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
