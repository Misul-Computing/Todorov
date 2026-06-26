"""2-bit attack #3: LDLQ adaptive rounding with REAL activation Hessians.

The project tried LDLQ with a W^T W weight-covariance proxy and it failed
(no improvement). But the picker already captures REAL activation energy.
QuIP# uses the real activation Hessian X^T X for LDLQ adaptive rounding, that's its 2-bit secret sauce. This feeds real activations into LDLQ.

For each Linear: capture input activations X, compute H = X^T X (the loss
Hessian w.r.t. weights), Cholesky H = L L^T, and do block-wise adaptive
rounding: when quantizing 8-dim block k, adjust by the cross-block error
weighted by U = L^T - I. This is the QuIP# incoherence-processing step
that plain E8 RVQ skips.
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
from e8lattice import e8_lattice_quantize


def capture_input_hessians(model, tok, n_examples=64, max_len=512, dev="cuda"):
    """Capture X^T X per Linear (the loss Hessian w.r.t. weights)."""
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

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
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


def quant_dequant_e8_ldlq_real(W, H, total_bits=2, group_size=128, chunk=2048):
    """E8 RVQ with LDLQ adaptive rounding using the REAL activation Hessian H=X^T X.

    For 2-bit: single E8 stage, but with LDLQ adjustment, each 8-dim block
    is shifted by the cross-block rounding error weighted by U=L^T-I before
    quantization, so the joint reconstruction error (weighted by H) is minimized.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    # Build the Hessian in the rotated domain. The weights act on activations:
    # loss = ||X (W - Wq)^T||^2 = trace((W-Wq) H (W-Wq)^T), H = X^T X.
    # After rotating columns by R (FWHT*s), W_rot = W R, and the Hessian
    # transforms as H_rot = R^T H R. We compute this once per tensor.
    Hpad = torch.zeros(npad, npad, device=W.device)
    Hpad[:d_in, :d_in] = H.to(W.device).float()
    # R = diag(s) @ FWHT (applied as Wp*s then fwht). H_rot = R^T H R.
    # Since R is orthonormal and symmetric (FWHT self-inverse, diag(s) symmetric):
    Hs = Hpad * s.unsqueeze(0) * s.unsqueeze(1)  # diag(s) H diag(s)
    # apply FWHT on both sides: H_rot = FWHT @ Hs @ FWHT
    H_rot = fwht(fwht(Hs).T).T  # FWHT(Hs FWHT^T), FWHT is orthonormal & symmetric
    # normalize and regularize
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

        # LDLQ block-wise adaptive rounding
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
            flat = W_block.reshape(-1, d)
            q = e8_lattice_quantize(flat, scale=1.0)
            What[:, k * d:(k + 1) * d] = q.reshape(W_block.shape)

        q_full = What * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = total_bits + 16.0 / group_size
    return out, err, bpw, group_size


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

    # capture real activation Hessians
    print("capturing real activation Hessians...", flush=True)
    hessians = capture_input_hessians(model, tok, n_examples=64, max_len=512, dev=dev)
    print(f"  {len(hessians)} Hessians captured ({time.time()-t0:.0f}s)", flush=True)

    # LDLQ 2-bit
    print("\n=== E8 + LDLQ (real Hessian) 2-bit ===", flush=True)
    restore(model, snap)
    tq = time.time()
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        H = hessians.get(name)
        if H is None:
            continue
        with torch.no_grad():
            Wq, err, bpw, _ = quant_dequant_e8_ldlq_real(mod.weight.data, H, total_bits=2)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err})
    avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
    print(f"  quantized {len(log)} Linears, avg_err={avg_err:.4f} ({time.time()-tq:.0f}s)", flush=True)
    ppl = ppl_wt2(model, tok, device=dev)
    degr = (ppl - base_ppl) / base_ppl
    print(f"  E8+LDLQ 2-bit: ppl={ppl:.3f} degr={degr:+.4f} ({time.time()-t0:.0f}s)", flush=True)

    out_dir = "/kaggle/working/runs/2bit_ldlq" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_ldlq")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "ppl": ppl, "degr": degr,
                   "avg_recon_err": avg_err}, f, indent=2)
    print(f"\nsaved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
