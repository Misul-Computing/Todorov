import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_delta import delta_chunkwise


def bench(fn, n=6, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000


def mk(t, b=8, h=4, d=64, dev="cpu"):
    q = torch.nn.functional.normalize(torch.randn(b, h, t, d, device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, d, device=dev), dim=-1)
    v = torch.randn(b, h, t, d, device=dev)
    beta = torch.sigmoid(torch.randn(b, h, t, device=dev))
    return q, k, v, beta


def main():
    print(f"default threads={torch.get_num_threads()}", flush=True)

    print("=== uncompiled vs torch.compile ===", flush=True)
    compiled = torch.compile(delta_chunkwise)
    for t in (256, 1024, 4096):
        q, k, v, beta = mk(t)
        un = bench(lambda: delta_chunkwise(q, k, v, beta, 64))
        try:
            co = bench(lambda: compiled(q, k, v, beta, 64))
            print(f"  T={t:5d} uncompiled={un:7.1f}ms compiled={co:7.1f}ms speedup={un / co:.2f}x", flush=True)
        except Exception as e:
            print(f"  T={t:5d} uncompiled={un:7.1f}ms compiled=FAILED ({type(e).__name__})", flush=True)

    print("=== chunk size sweep (T=2048) ===", flush=True)
    q, k, v, beta = mk(2048)
    for c in (16, 32, 64, 128, 256):
        print(f"  chunk={c:3d} {bench(lambda: delta_chunkwise(q, k, v, beta, c)):.1f}ms", flush=True)

    print("=== thread sweep (T=1024, chunk=64) ===", flush=True)
    q, k, v, beta = mk(1024)
    for nt in (1, 2, 4, 8):
        torch.set_num_threads(nt)
        print(f"  threads={nt} {bench(lambda: delta_chunkwise(q, k, v, beta, 64)):.1f}ms", flush=True)


if __name__ == "__main__":
    main()
