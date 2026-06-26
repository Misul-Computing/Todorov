"""Simplified LDLQ with data-free Hessian proxy.

Instead of full block LDL decomposition, use standard Cholesky H = L L^T
and apply the adaptive rounding adjustment block-wise.

The key insight: LDLQ adjusts each block based on previous rounding errors,
weighted by the Hessian. After Hadamard rotation, the Hessian (W^T W) is
approximately identity, so the adjustment is small. But the residual
structure may still help.
"""
import torch
from quant.rotate import fwht, signs_for, next_pow2
from quant.e8lattice import e8_lattice_quantize


def quant_dequant_e8_ldlq(W, total_bits=4, group_size=128, chunk=4096,
                          use_ldlq=True):
    """E8 RVQ with optional LDLQ adaptive rounding.

    total_bits: 4 (2-stage E8 RVQ)
    use_ldlq: if True, use LDLQ with W^T W Hessian proxy
    """
    d_out, d_in = W.shape
    Wf = W.float()
    npad = next_pow2(d_in)
    s = signs_for(npad, W.device)
    d = 8
    assert npad % d == 0

    n_stages = total_bits // 2
    assert n_stages * 2 == total_bits

    out = torch.empty_like(W)
    for i in range(0, d_out, chunk):
        Wc = Wf[i:i + chunk]
        Wp = torch.zeros(Wc.shape[0], npad, device=W.device)
        Wp[:, :d_in] = Wc
        Wr = fwht(Wp * s)
        sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
        Wrn = Wr / sc

        if use_ldlq:
            # Hessian proxy: W_rot^T W_rot (normalized)
            # Use a chunk of rows for efficiency
            H = Wrn[:min(128, Wrn.shape[0])].T @ Wrn[:min(128, Wrn.shape[0])]
            H /= H.shape[0]
            # Regularize heavily (after rotation, H ≈ I)
            H += 1.0 * torch.eye(npad, device=W.device)

            # Cholesky: H = L L^T, U = L^T - I (upper triangular)
            try:
                L = torch.linalg.cholesky(H)
                U = L.T - torch.eye(npad, device=W.device)
            except Exception:
                # Fallback to identity (no adjustment)
                U = torch.zeros(npad, npad, device=W.device)

            # Stage 1: LDLQ-style adaptive rounding
            Wrn.shape[0]
            What = torch.zeros_like(Wrn)
            nb = npad // d

            for k in range(nb):
                # Adjustment from previous blocks
                if k > 0:
                    err = Wrn[:, :k*d] - What[:, :k*d]  # [m, k*d]
                    A_k = U[:k*d, k*d:(k+1)*d]  # [k*d, d]
                    adj = err @ A_k  # [m, d]
                else:
                    adj = 0.0

                W_block = Wrn[:, k*d:(k+1)*d] + adj  # [m, d]
                # Quantize
                flat = W_block.reshape(-1, d)
                q = e8_lattice_quantize(flat, scale=1.0)
                What[:, k*d:(k+1)*d] = q.reshape(W_block.shape)

            # Stage 2: RVQ on residual
            residual = Wrn - What
            res_std = residual.std().clamp(min=1e-8)
            blocks_res = residual.reshape(-1, d)
            q2 = e8_lattice_quantize(blocks_res, scale=res_std)
            total_q = What + q2.reshape(Wrn.shape)
        else:
            # Plain RVQ
            blocks = Wrn.reshape(-1, d)
            residual = blocks.clone()
            for stage in range(n_stages):
                if stage == 0:
                    q = e8_lattice_quantize(residual, scale=1.0)
                else:
                    res_std = residual.std().clamp(min=1e-8)
                    q = e8_lattice_quantize(residual, scale=res_std)
                residual = residual - q
                if stage == 0:
                    total_q = q
                else:
                    total_q = total_q + q
            total_q = total_q.reshape(Wrn.shape)

        q_full = total_q * sc
        out[i:i + chunk] = (fwht(q_full) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = total_bits + 16.0 / group_size
    return out, err, bpw, group_size


def quantize_model_e8_ldlq(model, total_bits=4, group_size=128,
                           use_ldlq=True, verbose=True):
    """Apply E8 RVQ + LDLQ quantization to every nn.Linear."""
    import torch.nn as nn
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant_e8_ldlq(
                mod.weight.data, total_bits, group_size, use_ldlq=use_ldlq)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "recon_rel_err": err, "bpw": bpw})
        if verbose and (i % 40 == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name} err={err:.4f} bpw={bpw:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    if verbose:
        avg_err = sum(e["recon_rel_err"] for e in log) / len(log)
        method = "E8+LDLQ" if use_ldlq else "E8 RVQ"
        print(f"  quantized {len(targets)} Linears in {time.time() - t0:.1f}s "
              f"({method} {total_bits}bit, avg_err={avg_err:.4f})", flush=True)
    return log
