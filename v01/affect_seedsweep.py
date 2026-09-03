import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quick_affect_test as qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arm", default="shuffle")
    ap.add_argument("--ignite", type=float, default=0.20)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    base = argparse.Namespace(vocab=24, n_pairs=8, n_queries=4, d_model=128, layers=2,
                              head_dim=32, mode="mlp", steps=a.steps, batch=32, lr=3e-3,
                              eval_trials=100, eval_every=0, probe_len=16, seed=0, device=a.device)
    print(f"seed sweep arm={a.arm} steps={a.steps} seeds=0..{a.seeds - 1} "
          f"ignite_threshold(token)={a.ignite} chance_token~0.0435", flush=True)
    igns = 0
    toks = []
    t0 = time.time()
    for s in range(a.seeds):
        base.seed = s
        ev, loss, leak, stats = qa.run_arm(a.arm, base)
        tok = ev["token_acc"]
        toks.append(tok)
        ig = tok > a.ignite
        igns += int(ig)
        print(f"  seed {s:2d}  token {tok:.3f} exact {ev['exact_acc']:.3f} "
              f"gain {stats.get('write_gain', float('nan')):.3f} {'IGNITE' if ig else ''}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    toks.sort(reverse=True)
    print(f"ignition rate: {igns}/{a.seeds}  | best token {toks[0]:.3f}  median {toks[len(toks) // 2]:.3f}", flush=True)
    print("RESULT " + str({"ignitions": igns, "n": a.seeds, "tokens": toks}), flush=True)


if __name__ == "__main__":
    main()
