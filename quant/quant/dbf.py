"""Double Binary Factorization (DBF).

W ≈ (a * sign(A) * m^T) @ (sign(B) * b^T)

where:
  A: [d_out, k] binary (sign)
  B: [k, d_in] binary (sign)
  a: [d_out] FP16 row scales
  m: [k] FP16 intermediate scales
  b: [d_in] FP16 column scales

BPW = (d_out*k + k*d_in) / (d_out * d_in) + overhead
For square d×d: BPW ≈ 2k/d + small

Optimization: alternating least squares + sign/scale extraction.
"""
import torch


def dbf_factorize(W, k, n_iters=30, device="cuda", seed=42, verbose=False):
    """Factorize W ≈ (a * sign(A) * m^T) @ (sign(B) * b^T).

    Returns (A_sign, B_sign, a, m, b, recon_rel_err).
    """
    torch.manual_seed(seed)
    d_out, d_in = W.shape
    Wf = W.float().to(device)

    # SVD-based initialization: use top-k singular vectors for B
    U_s, S_s, Vh_s = torch.linalg.svd(Wf, full_matrices=False)
    k_actual = min(k, S_s.shape[0])
    B = torch.sign(Vh_s[:k_actual])  # [k, d_in], top-k right singular vectors
    if k_actual < k:
        B = torch.cat([B, torch.sign(torch.randn(k - k_actual, d_in, device=device))], 0)
    m = S_s[:k_actual].clone()
    if k_actual < k:
        m = torch.cat([m, torch.ones(k - k_actual, device=device)])
    a = torch.ones(d_out, device=device)
    b = torch.ones(d_in, device=device)

    for it in range(n_iters):
        # Step 1: Fix B, b, m. Solve for A.
        # W ≈ (a * sign(A) * m^T) @ (sign(B) * b^T)
        # Let W_B = (m * sign(B) * b^T)  [k, d_in]  (include m in the right factor)
        # W ≈ a * sign(A) @ W_B
        # Least squares: a * sign(A) = W @ pinv(W_B)
        W_B = m.unsqueeze(1) * B * b.unsqueeze(0)  # [k, d_in]
        A_ls = Wf @ torch.linalg.pinv(W_B)  # [d_out, k]
        A = torch.sign(A_ls)
        # Optimal a: W ≈ diag(a) @ sign(A) @ W_B
        # a[j] = <W[j], (sign(A)@W_B)[j]> / ||(sign(A)@W_B)[j]||^2
        AWB = A @ W_B  # [d_out, d_in]
        denom_a = (AWB ** 2).sum(1).clamp(min=1e-12)
        numer_a = (Wf * AWB).sum(1)
        a = numer_a / denom_a

        # Step 2: Fix A, a, m. Solve for B.
        # W ≈ (a * sign(A) * m^T) @ (sign(B) * b^T)
        # Let W_A = (a * sign(A) * m^T)  [d_out, k]
        # W ≈ W_A @ (sign(B) * b^T)
        # Least squares: sign(B) * b^T = pinv(W_A) @ W
        W_A = a.unsqueeze(1) * A * m.unsqueeze(0)  # [d_out, k]
        B_ls = torch.linalg.pinv(W_A) @ Wf  # [k, d_in]
        B = torch.sign(B_ls)
        # b = optimal column scale
        WA_B = W_A @ B  # [d_out, d_in]
        denom_b = (WA_B ** 2).sum(0).clamp(min=1e-12)  # [d_in]
        numer_b = (Wf * WA_B).sum(0)  # [d_in]
        b = numer_b / denom_b
        # m: update from residual
        # optimal m given A, a, B, b: m = argmin ||a*sign(A)*m^T * sign(B)*b^T - W||^2
        # This is a per-k scalar optimization
        # W ≈ sum_k a * sign(A[:,k]) * m[k] * sign(B[k,:]) * b^T
        # = sum_k m[k] * (a * sign(A[:,k])) @ (sign(B[k,:]) * b^T)
        # Let P_k = (a * sign(A[:,k])).unsqueeze(1) @ (sign(B[k,:]) * b).unsqueeze(0)  [d_out, d_in]
        # W ≈ sum_k m[k] * P_k
        # optimal m[k] = <W, P_k> / <P_k, P_k>
        for kk in range(k):
            P_k = (a * A[:, kk]).unsqueeze(1) @ (B[kk, :] * b).unsqueeze(0)  # [d_out, d_in]
            denom_m = (P_k ** 2).sum().clamp(min=1e-12)
            numer_m = (Wf * P_k).sum()
            m[kk] = numer_m / denom_m

        if verbose and (it % 10 == 0 or it == n_iters - 1):
            recon = (a.unsqueeze(1) * A * m.unsqueeze(0)) @ (B * b.unsqueeze(0))
            err = ((recon - Wf).norm() / Wf.norm()).item()
            print(f"  DBF iter {it+1}/{n_iters} err={err:.4f}", flush=True)

    recon = (a.unsqueeze(1) * A * m.unsqueeze(0)) @ (B * b.unsqueeze(0))
    err = ((recon - Wf).norm() / Wf.norm()).item()
    return A, B, a, m, b, err


def dbf_quantize_weight(W, target_bpw, n_iters=30, device="cuda", seed=42):
    """Quantize W with DBF at target_bpw. Returns (recon_W, err, bpw, k)."""
    d_out, d_in = W.shape
    # k = target_bpw * d_out * d_in / (d_out + d_in)  (binary bits only)
    # Plus scale overhead: (d_out + k + d_in) * 16 bits
    # Solve: d_out*k + k*d_in + (d_out + k + d_in)*16 = target_bpw * d_out * d_in
    # k * (d_out + d_in) + 16*k = target_bpw * d_out * d_in - 16*(d_out + d_in)
    # k = (target_bpw * d_out * d_in - 16*(d_out + d_in)) / (d_out + d_in + 16)
    k = max(1, int((target_bpw * d_out * d_in - 16 * (d_out + d_in)) /
                   (d_out + d_in + 16)))

    A_sign, B_sign, a, m, b, err = dbf_factorize(W, k, n_iters, device=device,
                                                  seed=seed)

    recon = (a.unsqueeze(1) * A_sign * m.unsqueeze(0)) @ (B_sign * b.unsqueeze(0))
    recon = recon.to(W.dtype)

    bits = d_out * k + k * d_in + (d_out + k + d_in) * 16
    bpw = bits / (d_out * d_in)
    return recon, err, bpw, k


def dbf_quantize_model(model, target_bpw, n_iters=30,
                       skip_names=(), device="cuda", verbose=True):
    """Apply DBF to every nn.Linear. Returns per-tensor log."""
    import torch.nn as nn
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        if any(s in name for s in skip_names):
            log.append({"name": name, "shape": list(mod.weight.shape),
                        "skipped": True, "recon_rel_err": 0.0, "bpw": 16.0, "k": 0})
            continue
        W = mod.weight.data
        Wq, err, bpw, k = dbf_quantize_weight(W, target_bpw, n_iters,
                                               device=device)
        mod.weight.data = Wq
        log.append({"name": name, "shape": list(W.shape), "skipped": False,
                    "recon_rel_err": err, "bpw": bpw, "k": k})
        if verbose and (i % 20 == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name:40s} err={err:.4f} bpw={bpw:.3f} "
                  f"k={k} ({time.time() - t0:.1f}s)", flush=True)
    return log
