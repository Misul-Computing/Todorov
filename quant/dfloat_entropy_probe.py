"""Entropy headroom probe for improving DFloat11 (lossless BF16 weight coding).

DFloat11 = 1 (sign, raw) + 7 (mantissa, raw) + ~2.85 (exponent, Huffman) = 10.85 b/w.
This measures, data-free, the achievable lossless ceiling at each modeling level so
we know how much each improvement step can buy BEFORE building a codec:

  - marginal entropy per field         (reproduces DFloat11's 0.99 / 2.6 / 6.87)
  - order-1 conditional H(field_i | field_{i-1})   -> context-model headroom
  - H(mantissa | exponent)                          -> cross-field headroom
  - group-residual exponent entropy                 -> predictive-coding headroom
  - per-tensor vs global exponent entropy           -> adaptive-table headroom

Everything is bit-exact lossless; nothing here changes the model.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import time, json
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM

CAP = 20_000_000  # subsample cap per tensor for the pair-histograms (entropy est.)


def H(counts):
    c = counts.double()
    p = c / c.sum().clamp(min=1)
    nz = p > 0
    return float(-(p[nz] * torch.log2(p[nz])).sum())


def cond_H(joint, nctx, nsym):
    j = joint.reshape(nctx, nsym)
    return H(joint) - H(j.sum(dim=1))


def main():
    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    dev = "cuda"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()
    print(f"loaded {MODEL_ID} in {time.time()-t0:.1f}s", flush=True)

    cnt_sign = torch.zeros(2, dtype=torch.long, device=dev)
    cnt_exp = torch.zeros(256, dtype=torch.long, device=dev)
    cnt_mant = torch.zeros(128, dtype=torch.long, device=dev)
    j_exp = torch.zeros(256 * 256, dtype=torch.long, device=dev)
    j_mant = torch.zeros(128 * 128, dtype=torch.long, device=dev)
    j_me = torch.zeros(256 * 128, dtype=torch.long, device=dev)
    cnt_resid = torch.zeros(512, dtype=torch.long, device=dev)
    pt_exp_H, nw, G = [], 0, 128

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w = (mod.weight.data.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF).reshape(-1)
        sign, exp, mant = (w >> 15) & 1, (w >> 7) & 0xFF, w & 0x7F
        nw += w.numel()
        cnt_sign += torch.bincount(sign, minlength=2)
        e_c = torch.bincount(exp, minlength=256); cnt_exp += e_c
        cnt_mant += torch.bincount(mant, minlength=128)
        pt_exp_H.append((H(e_c), w.numel()))
        # group-residual exponent (predict from rounded group-mean), full tensor
        n = w.numel() - (w.numel() % G)
        if n:
            eg = exp[:n].reshape(-1, G)
            base = eg.float().mean(dim=1, keepdim=True).round().long()
            resid = (eg - base + 256).clamp(0, 511).reshape(-1)
            cnt_resid += torch.bincount(resid, minlength=512)
        # order-1 / cross-field joints, subsample big tensors
        m = min(w.numel(), CAP)
        e, mn = exp[:m], mant[:m]
        j_exp += torch.bincount(e[:-1] * 256 + e[1:], minlength=256 * 256)
        j_mant += torch.bincount(mn[:-1] * 128 + mn[1:], minlength=128 * 128)
        j_me += torch.bincount(e * 128 + mn, minlength=256 * 128)

    Hs, He, Hm = H(cnt_sign), H(cnt_exp), H(cnt_mant)
    He1, Hm1 = cond_H(j_exp, 256, 256), cond_H(j_mant, 128, 128)
    Hme = cond_H(j_me, 256, 128)
    Hres = H(cnt_resid)
    pt_exp = sum(h * n for h, n in pt_exp_H) / sum(n for _, n in pt_exp_H)

    exp_best = min(He, He1, Hres, pt_exp)
    mant_best = min(Hm, Hm1, Hme)
    df11 = 1 + 7 + 2.85
    floor_marg = Hs + Hm + He
    ours_exp = Hs + Hm + exp_best
    ours_full = Hs + mant_best + exp_best

    print("\n=== per-field entropy (bits) ===", flush=True)
    print(f"  sign:     marginal {Hs:.4f}")
    print(f"  exponent: marginal {He:.4f}  order1 {He1:.4f}  group-resid {Hres:.4f}  per-tensor {pt_exp:.4f}")
    print(f"  mantissa: marginal {Hm:.4f}  order1 {Hm1:.4f}  given-exp {Hme:.4f}")
    print("\n=== bits/weight and compression ratio (lossless, 16-bit BF16) ===")
    for label, bpw in [("DFloat11 (reported)", df11),
                       ("entropy floor (marginal coding)", floor_marg),
                       ("ours: +context/predictive exponent", ours_exp),
                       ("ours: + conditional mantissa too", ours_full)]:
        print(f"  {label:38s} {bpw:6.3f} b/w   {16/bpw:.3f}x   "
              f"(vs DF11 {df11/bpw:.3f}x)", flush=True)
    print(f"\nweights={nw/1e6:.0f}M  total={time.time()-t0:.1f}s")
    print("RESULTS_JSON " + json.dumps({
        "H_sign": Hs, "H_exp": He, "H_exp_o1": He1, "H_exp_resid": Hres,
        "H_exp_pertensor": pt_exp, "H_mant": Hm, "H_mant_o1": Hm1, "H_mant_given_exp": Hme,
        "bpw_df11": df11, "bpw_floor_marg": floor_marg, "bpw_ours_exp": ours_exp,
        "bpw_ours_full": ours_full}))


if __name__ == "__main__":
    main()
