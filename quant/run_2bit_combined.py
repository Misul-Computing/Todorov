"""Combined 2-bit attack kernel (runs in one Kaggle session to avoid queue contention).

Four attacks, all on Qwen2.5-1.5B-Instruct, baseline ~9.48 PPL:
  1. Uniform E8 RVQ sweep {2,3,4,5,6}, Pareto curve with the new odd-bit
     sign-residual stages (3=2+1, 5=4+1).
  2. 2-bit outlier extraction, high-std rotated rows at FP16/4-bit, rest 2-bit E8.
  3. 2-bit learned RVQ, 8D Gaussian-fit and data-fit codebooks (2-stage 8-bit
     RVQ = 256+256 centroids). Attacks E8's uniform-optimality for Gaussian sources.
  4. 2-bit LDLQ with REAL activation Hessians, QuIP#'s adaptive rounding, fed
     with actual X^T X (not the W^T W proxy that failed before). Applied to
     attn projections (d_in<=2048, Cholesky feasible); MLP uses plain E8.

Block-wise FT is a separate kernel (it's the slow one).
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
from datasets import load_dataset

from quantize import snapshot, restore
from eval import ppl_wt2
from rotate import fwht, signs_for, next_pow2
from e8lattice import e8_lattice_quantize, quantize_model_e8rvq
from vq import gaussian_kmeans_codebook, kmeans, quant_rvq

GROUP = 128


# ============ 1. Uniform E8 sweep ============
def run_uniform_sweep(model, tok, snap, base_ppl, dev, results, t0):
    print(f"\n=== uniform E8 RVQ sweep (with odd-bit sign stages) ===", flush=True)
    for b in [2, 3, 4, 5, 6]:
        restore(model, snap)
        tq = time.time()
        quantize_model_e8rvq(model, total_bits=b, group_size=GROUP, verbose=False)
        ppl = ppl_wt2(model, tok, device=dev)
        bpw = b + 16.0 / GROUP
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"uniform_e8_{b}bit", "bpw": bpw, "ppl": ppl, "degr": degr})
        print(f"  E8 {b}bit: ppl={ppl:.3f} bpw={bpw:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None


# ============ 2. Outlier extraction ============
def quant_e8_outlier(W, outlier_pct, outlier_bits, group_size=GROUP, chunk=2048):
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0
    Wp_full = torch.zeros(d_out, npad, device=W.device)
    Wp_full[:, :d_in] = Wf
    Wr_full = fwht(Wp_full * s)
    row_std = Wr_full.std(dim=1)
    k_out = max(1, int(d_out * outlier_pct / 100))
    _, outlier_idx = torch.topk(row_std, k_out)
    outlier_mask = torch.zeros(d_out, dtype=torch.bool, device=W.device)
    outlier_mask[outlier_idx] = True

    out = torch.empty_like(W)
    if outlier_bits >= 16:
        out[outlier_mask] = Wf[outlier_mask].to(W.dtype)
    else:
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

    n_out = int(outlier_mask.sum().item())
    n_norm = d_out - n_out
    total_bits = (n_norm * 2 + n_out * outlier_bits) * d_in + d_out * 16
    bpw = total_bits / (d_out * d_in)
    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    return out, err, bpw


def run_outlier(model, tok, snap, base_ppl, dev, results, t0):
    print(f"\n=== 2-bit outlier extraction ===", flush=True)
    for pct, obits in [(1.0, 16), (2.0, 16), (5.0, 16), (10.0, 16), (5.0, 4), (10.0, 4)]:
        restore(model, snap)
        tq = time.time()
        total_bpw_num, total_n = 0.0, 0
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            with torch.no_grad():
                Wq, err, bpw = quant_e8_outlier(mod.weight.data, pct, obits)
                mod.weight.data.copy_(Wq)
            n = mod.weight.data.numel()
            total_bpw_num += bpw * n
            total_n += n
        avg_bpw = total_bpw_num / total_n
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"outlier_{obits}bit_{pct}pct", "bpw": avg_bpw,
                        "ppl": ppl, "degr": degr})
        print(f"  outlier {pct}% @{obits}bit: bpw={avg_bpw:.3f} ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None


# ============ 3. Learned RVQ ============
def quant_learned_rvq(W, codebooks, group_size=GROUP, chunk=2048):
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
        residual = blocks.clone()
        total_q = torch.zeros_like(blocks)
        for cb in codebooks:
            res_std = residual.std().clamp(min=1e-8)
            res_norm = residual / res_std
            q, _ = quant_rvq(res_norm, [cb])
            q = q * res_std
            total_q = total_q + q
            residual = blocks - total_q
        q_full = total_q.reshape(Wc.shape[0], npad) * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)
    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    return out, err, 2 + 16.0 / group_size


def run_learned_rvq(model, tok, snap, base_ppl, dev, results, t0):
    print(f"\n=== 2-bit learned RVQ (Gaussian + datafit) ===", flush=True)
    d, k = 8, 256
    for cb_type in ["gaussian", "datafit"]:
        restore(model, snap)
        tq = time.time()
        if cb_type == "gaussian":
            cb0 = gaussian_kmeans_codebook(d, k, dev, n_samples=200000, iters=25, seed=0)
            g = torch.Generator(device="cpu").manual_seed(1)
            x = torch.randn(200000, d, generator=g).to(dev)
            q0, _ = quant_rvq(x, [cb0])
            res1 = x - q0
            res1_std = res1.std().clamp(min=1e-8)
            cb1 = kmeans(res1 / res1_std, k, iters=25, seed=1)
        else:
            all_blocks = []
            for name, mod in model.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                W = mod.weight.data.float()
                npad = next_pow2(W.shape[1])
                s = signs_for(npad, W.device)
                nrows = min(384, W.shape[0])
                Wp = torch.zeros(nrows, npad, device=W.device)
                Wp[:, :W.shape[1]] = W[:nrows]
                Wr = fwht(Wp * s)
                sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
                Wrn = (Wr / sc).reshape(-1, d)
                all_blocks.append(Wrn[:3072])
            all_blocks = torch.cat(all_blocks, dim=0)
            cb0 = kmeans(all_blocks, k, iters=30, seed=0, max_samples=60000)
            q0, _ = quant_rvq(all_blocks[:40000], [cb0])
            cb1 = kmeans(all_blocks[:40000] - q0, k, iters=30, seed=1, max_samples=40000)
        cbs = [cb0, cb1]
        print(f"  {cb_type} codebooks built ({time.time()-tq:.0f}s)", flush=True)
        tq = time.time()
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            with torch.no_grad():
                Wq, err, bpw = quant_learned_rvq(mod.weight.data, cbs)
                mod.weight.data.copy_(Wq)
        ppl = ppl_wt2(model, tok, device=dev)
        degr = (ppl - base_ppl) / base_ppl
        results.append({"cfg": f"learned_rvq_{cb_type}", "bpw": 2.125,
                        "ppl": ppl, "degr": degr})
        print(f"  {cb_type} 2-bit: ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None


# ============ 4. LDLQ with real activations ============
def capture_hessians(model, tok, n_examples=48, max_len=512, dev="cuda"):
    hessians = {}
    handles = []
    def make_hook(name):
        def hook(mod, inp, _out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            H = x.T @ x
            if name not in hessians:
                hessians[name] = H.to(dev)
            else:
                hessians[name] += H.to(dev)
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    k = 0
    model.eval()
    with torch.no_grad():
        for ex in ds:
            if k >= n_examples:
                break
            t = ex["text"]
            if 50 < len(t) < 2000:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
                if ids.shape[1] < 2:
                    continue
                model(ids)
                k += 1
    for h in handles:
        h.remove()
    return hessians


def quant_e8_ldlq_real(W, H, group_size=GROUP, chunk=2048):
    """2-bit E8 + LDLQ adaptive rounding with real activation Hessian.
    Only feasible for d_in <= 2048 (Cholesky cost)."""
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    Hpad = torch.zeros(npad, npad, device=W.device)
    Hpad[:d_in, :d_in] = H.to(W.device).float()
    Hs = Hpad * s.unsqueeze(0) * s.unsqueeze(1)
    H_rot = fwht(fwht(Hs).T).T
    H_rot = H_rot / H_rot.diagonal().mean().clamp(min=1e-8)
    H_rot += 0.01 * torch.eye(npad, device=W.device)
    try:
        L = torch.linalg.cholesky(H_rot)
        U = L.T - torch.eye(npad, device=W.device)
    except Exception:
        U = torch.zeros(npad, npad, device=W.device)

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc
        Wrn.shape[0]
        nb = npad // d
        What = torch.zeros_like(Wrn)
        for k in range(nb):
            if k > 0:
                err = Wrn[:, :k * d] - What[:, :k * d]
                A_k = U[:k * d, k * d:(k + 1) * d]
                adj = err @ A_k
            else:
                adj = 0.0
            W_block = Wrn[:, k * d:(k + 1) * d] + adj
            q = e8_lattice_quantize(W_block.reshape(-1, d), scale=1.0)
            What[:, k * d:(k + 1) * d] = q.reshape(W_block.shape)
        q_full = What * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)
    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    return out, err, 2 + 16.0 / group_size


def run_ldlq(model, tok, snap, base_ppl, dev, results, t0):
    print(f"\n=== 2-bit LDLQ with real activation Hessians ===", flush=True)
    print("  capturing activations...", flush=True)
    hessians = capture_hessians(model, tok, n_examples=48, max_len=512, dev=dev)
    print(f"  {len(hessians)} Hessians captured ({time.time()-t0:.0f}s)", flush=True)
    restore(model, snap)
    tq = time.time()
    n_ldlq, n_plain = 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        H = hessians.get(name)
        d_in = mod.weight.data.shape[1]
        with torch.no_grad():
            if H is not None and d_in <= 2048:
                Wq, err, bpw = quant_e8_ldlq_real(mod.weight.data, H)
                n_ldlq += 1
            else:
                # MLP or large d_in: plain 2-bit E8
                Wp = torch.zeros(mod.weight.data.shape[0], next_pow2(d_in), device=dev)
                Wp[:, :d_in] = mod.weight.data.float()
                s = signs_for(next_pow2(d_in), dev)
                Wr = fwht(Wp * s)
                sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
                Wrn = Wr / sc
                blocks = Wrn.reshape(-1, 8)
                q = e8_lattice_quantize(blocks, scale=1.0)
                q_full = q.reshape(Wr.shape) * sc
                Wq = (fwht(q_full) * s)[:, :d_in].to(mod.weight.data.dtype)
                n_plain += 1
            mod.weight.data.copy_(Wq)
    print(f"  LDLQ on {n_ldlq} attn tensors, plain E8 on {n_plain} MLP ({time.time()-tq:.0f}s)", flush=True)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    results.append({"cfg": "e8_ldlq_real_2bit", "bpw": 2.125, "ppl": ppl, "degr": degr,
                    "n_ldlq": n_ldlq, "n_plain": n_plain})
    print(f"  E8+LDLQ 2-bit: ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
    torch.cuda.empty_cache() if dev == "cuda" else None


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)
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
    run_uniform_sweep(model, tok, snap, base_ppl, dev, results, t0)
    run_outlier(model, tok, snap, base_ppl, dev, results, t0)
    run_learned_rvq(model, tok, snap, base_ppl, dev, results, t0)
    run_ldlq(model, tok, snap, base_ppl, dev, results, t0)

    print(f"\n{'='*80}", flush=True)
    print(f"{'cfg':>26s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s}", flush=True)
    print(f"{'-'*80}", flush=True)
    for r in results:
        print(f"{r['cfg']:>26s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f}", flush=True)

    out_dir = "/kaggle/working/runs/2bit_combined" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_combined")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "results": results}, f, indent=2)
    print(f"\nsaved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
