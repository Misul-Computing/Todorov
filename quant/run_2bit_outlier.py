"""2-bit attack #4: outlier extraction at 2-bit in the rotated domain.

STATUS says outlier extraction was negligible at 4-bit. But at 2-bit,
outliers dominate the error, a few high-magnitude rotated rows hold
disproportionate energy. Store those rows at FP16 (or 4-bit), quantize
the rest at 2-bit E8. The cost: outlier rows need full precision, but
if only ~5% of rows are outliers, the bpw overhead is small.

Tests several outlier fractions {1%, 2%, 5%, 10%} and two precision
levels for outliers (FP16, 4-bit E8). Reports bpw including outlier
storage and PPL. The knee is the answer.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import snapshot, restore
from eval import ppl_wt2
from rotate import fwht, signs_for, next_pow2
from e8lattice import e8_lattice_quantize


def quant_dequant_e8_outlier(W, outlier_pct=5.0, outlier_bits=16,
                             group_size=128, chunk=2048):
    """2-bit E8 with outlier rows kept at higher precision.

    Outlier selection: in the rotated domain, rows with highest std
    (per-row scale) are outliers. Store those rows at outlier_bits
    (16=FP16, 4=4-bit E8 RVQ), rest at 2-bit E8.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    # determine outlier rows by per-row rotated std (computed once on full matrix)
    Wp_full = torch.zeros(d_out, npad, device=W.device)
    Wp_full[:, :d_in] = Wf
    Wr_full = fwht(Wp_full * s)
    row_std = Wr_full.std(dim=1)
    k_out = max(1, int(d_out * outlier_pct / 100))
    _, outlier_idx = torch.topk(row_std, k_out)
    outlier_mask = torch.zeros(d_out, dtype=torch.bool, device=W.device)
    outlier_mask[outlier_idx] = True

    out = torch.empty_like(W)
    # outlier rows: keep at FP16 (or quantize at higher bits)
    if outlier_bits >= 16:
        out[outlier_mask] = Wf[outlier_mask].to(W.dtype)
    else:
        # 4-bit E8 RVQ for outlier rows (2 E8 stages)
        for i in range(0, k_out, chunk):
            idxs = outlier_idx[i:i + chunk]
            Wc = Wf[idxs]
            Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
            Wp[:, :d_in] = Wc
            Wr = fwht(Wp * s)
            sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
            Wrn = Wr / sc
            blocks = Wrn.reshape(-1, d)
            residual = blocks.clone()
            total_q = torch.zeros_like(blocks)
            for stage in range(outlier_bits // 2):
                if stage == 0:
                    q = e8_lattice_quantize(residual, scale=1.0)
                else:
                    res_std = residual.std().clamp(min=1e-8)
                    q = e8_lattice_quantize(residual, scale=res_std)
                residual = residual - q
                total_q = total_q + q
            q_full = total_q.reshape(Wc.shape[0], npad) * sc
            out[idxs] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    # normal rows: 2-bit E8
    normal_idx = (~outlier_mask).nonzero(as_tuple=True)[0]
    for i in range(0, normal_idx.shape[0], chunk):
        idxs = normal_idx[i:i + chunk]
        Wc = Wf[idxs]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)
        q = e8_lattice_quantize(blocks, scale=1.0)
        q_full = q.reshape(Wc.shape[0], npad) * sc
        out[idxs] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    # bpw: normal rows 2-bit, outlier rows outlier_bits, + scale overhead
    n_out = int(outlier_mask.sum().item())
    n_norm = d_out - n_out
    # per-row scale is FP16 for both (16 bits / d_in per row)
    total_bits = (n_norm * 2 + n_out * outlier_bits) * d_in + d_out * 16
    bpw = total_bits / (d_out * d_in)
    return out, err, bpw, group_size


def apply_outlier_quant(model, outlier_pct, outlier_bits, group_size=128):
    log = []
    total_bpw_num = 0.0
    total_n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _ = quant_dequant_e8_outlier(
                mod.weight.data, outlier_pct, outlier_bits, group_size)
            mod.weight.data.copy_(Wq)
        n = mod.weight.data.numel()
        total_bpw_num += bpw * n
        total_n += n
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
    return log, total_bpw_num / total_n


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16,
        device_map=dev, low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)
    snap = snapshot(model)
    base_ppl = ppl_wt2(model, tok, device=dev)
    print(f"baseline PPL: {base_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = []
    # outlier rows at FP16
    for pct in [1.0, 2.0, 5.0, 10.0]:
        restore(model, snap)
        tq = time.time()
        _, avg_bpw = apply_outlier_quant(model, pct, 16)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"outlier_fp16_{pct}pct", "outlier_pct": pct,
                        "outlier_bits": 16, "bpw": avg_bpw, "ppl": ppl, "degr": degr})
        print(f"  outlier {pct}% FP16: bpw={avg_bpw:.3f} ppl={ppl:.3f} degr={degr:+.4f} "
              f"({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    # outlier rows at 4-bit
    for pct in [5.0, 10.0]:
        restore(model, snap)
        tq = time.time()
        _, avg_bpw = apply_outlier_quant(model, pct, 4)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"outlier_4bit_{pct}pct", "outlier_pct": pct,
                        "outlier_bits": 4, "bpw": avg_bpw, "ppl": ppl, "degr": degr})
        print(f"  outlier {pct}% 4bit: bpw={avg_bpw:.3f} ppl={ppl:.3f} degr={degr:+.4f} "
              f"({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    out_dir = "/kaggle/working/runs/2bit_outlier" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_outlier")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
