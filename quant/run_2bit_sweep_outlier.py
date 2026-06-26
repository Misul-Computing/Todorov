"""2-bit kernel 1: uniform E8 sweep (with odd-bit stages) + outlier extraction.
Fast, ~600s. Run alone to avoid queue contention.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize import snapshot, restore
from eval import ppl_wt2
from rotate import fwht, signs_for, next_pow2
from e8lattice import e8_lattice_quantize, quantize_model_e8rvq

GROUP = 128

def quant_e8_outlier(W, outlier_pct, outlier_bits, group_size=GROUP, chunk=2048):
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    Wp_full = torch.zeros(d_out, npad, device=W.device); Wp_full[:, :d_in] = Wf
    Wr_full = fwht(Wp_full * s)
    row_std = Wr_full.std(dim=1)
    k_out = max(1, int(d_out * outlier_pct / 100))
    _, outlier_idx = torch.topk(row_std, k_out)
    mask = torch.zeros(d_out, dtype=torch.bool, device=W.device)
    mask[outlier_idx] = True
    out = torch.empty_like(W)
    if outlier_bits >= 16:
        out[mask] = Wf[mask].to(W.dtype)
    else:
        for i in range(0, k_out, chunk):
            idxs = outlier_idx[i:i + chunk]
            Wc = Wf[idxs]; Wp = torch.zeros(Wc.shape[0], npad, device=W.device); Wp[:, :d_in] = Wc
            Wr = fwht(Wp * s); sc = Wr.std(1, keepdim=True).clamp(min=1e-8); Wrn = Wr / sc
            blocks = Wrn.reshape(-1, d); residual = blocks.clone(); total_q = torch.zeros_like(blocks)
            for stage in range(outlier_bits // 2):
                q = e8_lattice_quantize(residual, scale=1.0) if stage == 0 else \
                    e8_lattice_quantize(residual, scale=residual.std().clamp(min=1e-8))
                residual = residual - q; total_q = total_q + q
            out[idxs] = (fwht(total_q.reshape(Wc.shape[0], npad) * sc) * s)[:, :d_in].to(W.dtype)
    normal_idx = (~mask).nonzero(as_tuple=True)[0]
    for i in range(0, normal_idx.shape[0], chunk):
        idxs = normal_idx[i:i + chunk]
        Wc = Wf[idxs]; Wp = torch.zeros(Wc.shape[0], npad, device=W.device); Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s); sc = Wr.std(1, keepdim=True).clamp(min=1e-8); Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)
        q = e8_lattice_quantize(blocks, scale=1.0)
        out[idxs] = (fwht(q.reshape(Wc.shape[0], npad) * sc) * s)[:, :d_in].to(W.dtype)
    n_out = int(mask.sum().item()); n_norm = d_out - n_out
    total_bits = (n_norm * 2 + n_out * outlier_bits) * d_in + d_out * 16
    return out, total_bits / (d_out * d_in)

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16,
        device_map=dev, low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)
    snap = snapshot(model)
    base_ppl = ppl_wt2(model, tok, device=dev)
    print(f"baseline PPL: {base_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results = []

    print(f"\n=== uniform E8 sweep (odd-bit sign stages) ===", flush=True)
    for b in [2, 3, 4, 5, 6]:
        restore(model, snap); tq = time.time()
        quantize_model_e8rvq(model, total_bits=b, group_size=GROUP, verbose=False)
        ppl = ppl_wt2(model, tok, device=dev)
        bpw = b + 16.0 / GROUP; degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"uniform_e8_{b}bit", "bpw": bpw, "ppl": ppl, "degr": degr})
        print(f"  E8 {b}bit: ppl={ppl:.3f} bpw={bpw:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    print(f"\n=== 2-bit outlier extraction ===", flush=True)
    for pct, obits in [(1.0, 16), (2.0, 16), (5.0, 16), (10.0, 16), (5.0, 4), (10.0, 4)]:
        restore(model, snap); tq = time.time()
        tbn, tn = 0.0, 0
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear): continue
            with torch.no_grad():
                Wq, bpw = quant_e8_outlier(mod.weight.data, pct, obits)
                mod.weight.data.copy_(Wq)
            n = mod.weight.data.numel(); tbn += bpw * n; tn += n
        avg_bpw = tbn / tn; ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"outlier_{obits}bit_{pct}pct", "bpw": avg_bpw, "ppl": ppl, "degr": degr})
        print(f"  outlier {pct}% @{obits}bit: bpw={avg_bpw:.3f} ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    print(f"\n{'='*70}\n{'cfg':>24s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s}\n{'-'*70}", flush=True)
    for r in results:
        print(f"{r['cfg']:>24s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f}", flush=True)
    out_dir = "/kaggle/working/runs/2bit_sweep_outlier" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_sweep_outlier")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
