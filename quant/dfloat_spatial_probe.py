"""Iteration 1: spatial-structure probe for improving the rANS coder beyond DFloat11.

The order-1 (memory-flattened) probe found no structure. But weights are a 2D
matrix [d_out, d_in]; the structure, if any, is spatial. rANS codes to the
conditional entropy of whatever model we give it, so a model that beats the
2.608-bit exponent marginal IS an improved coder. Tests, per field:

  - marginal                         (baseline; DFloat11's model)
  - per-row mean residual            (per-output-channel magnitude structure)
  - per-column mean residual         (per-input-channel structure)
  - 2D context H(x | left, up)       (image-style spatial prediction)
  - order-2 row context H(x | l1,l2)

All data-free, all lossless. Side-info for residuals (per-row/col FP16 mean) is
~16/d_in b/w (negligible). Whatever model gives the lowest conditional entropy
is the rANS coder's achievable rate.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import time, json
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM

ROWCAP = 4096  # cap rows per tensor for the 2D joint histograms


def Hc(counts):
    c = counts.double(); p = c / c.sum().clamp(min=1); nz = p > 0
    return float(-(p[nz] * torch.log2(p[nz])).sum())


def condH(joint, nctx, k):
    return Hc(joint) - Hc(joint.reshape(nctx, k).sum(dim=1))


def main():
    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    dev = "cuda"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()
    print(f"loaded {MODEL_ID} in {time.time()-t0:.1f}s", flush=True)

    e_marg = torch.zeros(256, dtype=torch.long, device=dev)
    m_marg = torch.zeros(128, dtype=torch.long, device=dev)
    e_row = torch.zeros(512, dtype=torch.long, device=dev)
    e_col = torch.zeros(512, dtype=torch.long, device=dev)
    m_row = torch.zeros(256, dtype=torch.long, device=dev)
    e_2d = torch.zeros(256**3, dtype=torch.long, device=dev)
    e_o2 = torch.zeros(256**3, dtype=torch.long, device=dev)
    m_2d = torch.zeros(128**3, dtype=torch.long, device=dev)
    nw = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        try:
            bits = mod.weight.data.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
            exp = (bits >> 7) & 0xFF
            man = bits & 0x7F
            do, di = exp.shape
            nw += exp.numel()
            e_marg += torch.bincount(exp.reshape(-1), minlength=256)
            m_marg += torch.bincount(man.reshape(-1), minlength=128)
            rb = exp.float().mean(1, keepdim=True).round().long()
            e_row += torch.bincount((exp - rb + 256).clamp(0, 511).reshape(-1), minlength=512)
            cb = exp.float().mean(0, keepdim=True).round().long()
            e_col += torch.bincount((exp - cb + 256).clamp(0, 511).reshape(-1), minlength=512)
            mrb = man.float().mean(1, keepdim=True).round().long()
            m_row += torch.bincount((man - mrb + 128).clamp(0, 255).reshape(-1), minlength=256)
            E, M = exp[:min(do, ROWCAP)], man[:min(do, ROWCAP)]
            if E.shape[0] > 2 and E.shape[1] > 2:
                e_2d += torch.bincount((E[1:, :-1] * 65536 + E[:-1, 1:] * 256 + E[1:, 1:]).reshape(-1), minlength=256**3)
                e_o2 += torch.bincount((E[:, :-2] * 65536 + E[:, 1:-1] * 256 + E[:, 2:]).reshape(-1), minlength=256**3)
                m_2d += torch.bincount((M[1:, :-1] * 16384 + M[:-1, 1:] * 128 + M[1:, 1:]).reshape(-1), minlength=128**3)
        except Exception as e:
            print(f"  skip {name}: {e}", flush=True)

    He = Hc(e_marg); Hm = Hc(m_marg)
    He_row, He_col = Hc(e_row), Hc(e_col)
    Hm_row = Hc(m_row)
    He_2d, He_o2 = condH(e_2d, 65536, 256), condH(e_o2, 65536, 256)
    Hm_2d = condH(m_2d, 16384, 128)

    exp_best = min(He, He_row, He_col, He_2d, He_o2)
    man_best = min(Hm, Hm_row, Hm_2d)
    df11 = 10.85
    ours = 1.0 + man_best + exp_best  # sign raw = 1 bit

    print("\n=== exponent model (bits, lower=better; marginal=2.608 is DFloat11) ===", flush=True)
    print(f"  marginal {He:.4f}  row-resid {He_row:.4f}  col-resid {He_col:.4f}  "
          f"2D(left,up) {He_2d:.4f}  order2 {He_o2:.4f}")
    print("=== mantissa model (marginal=6.972) ===")
    print(f"  marginal {Hm:.4f}  row-resid {Hm_row:.4f}  2D(left,up) {Hm_2d:.4f}")
    print("\n=== coder bits/weight (sign 1 + best mantissa + best exponent) ===")
    print(f"  DFloat11           10.850 b/w   1.475x")
    print(f"  ours (best model)  {ours:6.3f} b/w   {16/ours:.3f}x   (vs DF11 {df11/ours:.3f}x)")
    print(f"\nweights={nw/1e6:.0f}M  total={time.time()-t0:.1f}s")
    print("RESULTS_JSON " + json.dumps({
        "He_marg": He, "He_row": He_row, "He_col": He_col, "He_2d": He_2d, "He_o2": He_o2,
        "Hm_marg": Hm, "Hm_row": Hm_row, "Hm_2d": Hm_2d,
        "exp_best": exp_best, "man_best": man_best, "bpw_ours": ours, "ratio_ours": 16/ours}))


if __name__ == "__main__":
    main()
