import sys
import os
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import LM


def fwd_ms(model, x, n=4, warmup=2):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.time()
        for _ in range(n):
            model(x)
    return (time.time() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lengths", default="256,1024,4096,8192")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = args.device
    lengths = [int(x) for x in args.lengths.split(",")]
    torch.manual_seed(0)
    attn = LM(256, "attn", args.d_model, args.layers, args.n_heads, args.d_head).to(device)
    delta = LM(256, "delta", args.d_model, args.layers, args.n_heads, args.d_head).to(device)
    state_mb = args.layers * args.n_heads * args.d_head * args.d_head * 4 / 1e6
    print(f"d_model={args.d_model} layers={args.layers} heads={args.n_heads} d_head={args.d_head} "
          f"batch={args.batch} device={device}", flush=True)
    print(f"delta recurrent state is FIXED at {state_mb:.3f} MB/seq regardless of context", flush=True)
    print(f"{'ctx':>6} {'attn_fwd_ms':>12} {'delta_fwd_ms':>13} {'attn_KV_MB/seq':>15} {'state/KV':>9}", flush=True)
    for L in lengths:
        x = torch.randint(0, 256, (args.batch, L), device=device)
        try:
            am = fwd_ms(attn, x)
        except Exception as e:
            am = float("nan")
        dm = fwd_ms(delta, x)
        kv_mb = args.layers * 2 * L * args.n_heads * args.d_head * 4 / 1e6
        print(f"{L:>6} {am:>12.1f} {dm:>13.1f} {kv_mb:>15.2f} {state_mb / kv_mb:>9.4f}", flush=True)


if __name__ == "__main__":
    main()
