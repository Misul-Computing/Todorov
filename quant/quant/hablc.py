"""SVD + Binary Residual (SBR): low-rank FP16 + sign-quantized residual.

1. SVD:              W = U S V^T
2. Truncate:         W_lr = U_k S_k V_k^T   (rank k, FP16)
3. Residual:         R = W - W_lr
4. Sign quantize:    Q = sign(R / std(R)) * std(R)
5. Reconstruct:      W ~ W_lr + Q

BPW = 1 + 16*k*(d_out + d_in) / (d_out * d_in)
For square d×d: BPW = 1 + 32*k/d

The residual after SVD has reduced variance (only the "noise" directions).
Sign quantization adds ~36% of the residual's remaining energy as error.
With 1/k singular value decay, rank-32 removes ~98% of energy,
leaving a small residual that sign quantization handles well.
"""
import torch


def sbr_quant(W, rank):
    """Quantize W with SVD + Binary Residual. Returns (dequantized_W, recon_err, bpw, rank)."""
    """Quantize W with SVD + Binary Residual. Returns (dequantized_W, recon_err, bpw, rank)."""
    d_out, d_in = W.shape
    Wf = W.float()
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    k = min(rank, S.shape[0]) if rank > 0 else 0

    if k > 0:
        W_lr = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    else:
        W_lr = torch.zeros_like(Wf)

    R = Wf - W_lr
    R_std = R.std().clamp(min=1e-8)
    Q = torch.sign(R / R_std) * R_std
    rec = (W_lr + Q).to(W.dtype)

    binary_bits = d_out * d_in
    lr_bits = k * (d_out + d_in) * 16 + k * 16 if k > 0 else 0
    bpw = (binary_bits + lr_bits) / (d_out * d_in)
    err = ((rec.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    return rec, err, bpw, k


def sbr_quant_model(model, rank, skip_names=(), log_every=40, verbose=True):
    """Apply SBR to every nn.Linear in model. Returns per-tensor log."""
    import torch.nn as nn
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        if any(s in name for s in skip_names):
            log.append({"name": name, "shape": list(mod.weight.shape),
                        "skipped": True, "recon_rel_err": 0.0, "bpw": 16.0, "rank": 0})
            continue
        W = mod.weight.data
        Wq, err, bpw, r = sbr_quant(W, rank)
        mod.weight.data = Wq
        entry = {"name": name, "shape": list(W.shape), "skipped": False,
                 "recon_rel_err": err, "bpw": bpw, "rank": r}
        log.append(entry)
        if verbose and (i % log_every == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name:40s} err={err:.4f} bpw={bpw:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return log
