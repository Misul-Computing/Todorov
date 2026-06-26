"""2-bit attack #2: 8D Gaussian-fit / data-fit learned RVQ codebook.

E8 lattice is optimal for UNIFORM sources; rotated weights are GAUSSIAN.
The project proved Gaussian k-means beats D4 lattice at 2-bit (d=4).
E8 d=8 already beats d=4 Gaussian (27.7 vs 125 PPL) because 8D captures
more correlation, but a Gaussian-fit 8D codebook should beat E8 further.

2-bit/8D = 16 bits/block. Single-stage = 65536 centroids (OOM). Use
2-stage learned RVQ: 8 bits + 8 bits = 256+256 centroids (fits easily).
Each stage quantizes the residual of the previous. Codebooks from:
  - Gaussian k-means (data-free, N(0,I) samples)
  - Data-fit k-means (actual rotated weights, should be best)
Compare to E8's 27.7 PPL.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import snapshot, restore
from eval import ppl_wt2
from rotate import fwht, signs_for, next_pow2
from vq import gaussian_kmeans_codebook, kmeans, quant_rvq


def quant_dequant_learned_rvq(W, codebooks, group_size=128, chunk=2048):
    """2-stage learned RVQ in Hadamard domain. codebooks: list of [k, 8]."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0
    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        blocks = Wrn.reshape(-1, d)
        # normalize residual per stage to unit variance for codebook fit
        residual = blocks.clone()
        total_q = torch.zeros_like(blocks)
        for si, cb in enumerate(codebooks):
            res_std = residual.std().clamp(min=1e-8)
            res_norm = residual / res_std
            q, _ = quant_rvq(res_norm, [cb])
            q = q * res_std
            total_q = total_q + q
            residual = blocks - total_q
        q_full = total_q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)
    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = 2 + 16.0 / group_size
    return out, err, bpw, group_size


def build_learned_codebooks(model, dev, cb_type="gaussian", group_size=128):
    """Build 2-stage 8-bit RVQ codebooks. cb_type: 'gaussian' or 'datafit'."""
    d = 8
    k = 256  # 8 bits per stage
    if cb_type == "gaussian":
        cb0 = gaussian_kmeans_codebook(d, k, dev, n_samples=200000, iters=25, seed=0)
        # stage 1 codebook: residual of unit-Gaussian after stage 0, approximate
        # by sampling: quantize gaussian samples with cb0, take residual
        g = torch.Generator(device="cpu").manual_seed(1)
        x = torch.randn(200000, d, generator=g).to(dev)
        q0, _ = quant_rvq(x, [cb0])
        res1 = x - q0
        res1_std = res1.std().clamp(min=1e-8)
        cb1 = kmeans(res1 / res1_std, k, iters=25, seed=1)
        return [cb0, cb1]

    # datafit: collect actual rotated blocks from all Linears
    print("  collecting rotated blocks for datafit codebook...", flush=True)
    all_blocks = []
    npad_common = None
    for name, mod in model.named_modules():
        import torch.nn as nn
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data.float()
        npad = next_pow2(W.shape[1])
        npad_common = npad if npad_common is None else npad_common
        if npad != npad_common:
            continue  # skip mismatched (k/v proj)
        s = signs_for(npad, W.device)
        # subsample rows for speed
        nrows = min(512, W.shape[0])
        Wp = torch.zeros(nrows, npad, device=W.device)
        Wp[:, :W.shape[1]] = W[:nrows]
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = (Wr / sc).reshape(-1, d)
        all_blocks.append(Wrn[:4096])  # cap per tensor
    all_blocks = torch.cat(all_blocks, dim=0)
    print(f"  {all_blocks.shape[0]} blocks collected", flush=True)
    cb0 = kmeans(all_blocks, k, iters=30, seed=0, max_samples=80000)
    # stage 1 on residual
    q0, _ = quant_rvq(all_blocks[:40000], [cb0])
    res1 = all_blocks[:40000] - q0
    cb1 = kmeans(res1, k, iters=30, seed=1, max_samples=40000)
    return [cb0, cb1]


def apply_learned_rvq(model, codebooks, group_size=128, verbose=False):
    import torch.nn as nn
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        with torch.no_grad():
            Wq, err, bpw, _ = quant_dequant_learned_rvq(mod.weight.data, codebooks, group_size)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
    return log


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
    for cb_type in ["gaussian", "datafit"]:
        print(f"\n=== learned RVQ 2-bit ({cb_type}) ===", flush=True)
        restore(model, snap)
        tq = time.time()
        cbs = build_learned_codebooks(model, dev, cb_type=cb_type)
        print(f"  codebooks built ({time.time()-tq:.0f}s)", flush=True)
        tq = time.time()
        apply_learned_rvq(model, cbs)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"learned_rvq_{cb_type}", "bpw": 2.125,
                        "ppl": ppl, "degr": degr})
        print(f"  {cb_type} 2-bit: ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    out_dir = "/kaggle/working/runs/2bit_gaussian" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_gaussian")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
