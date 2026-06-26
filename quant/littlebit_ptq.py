"""Fast PTQ LittleBit, same decomposition, no QAT.

For each Linear weight W: SVD truncate to rank r, binarize the factors,
learn per-row/col/latent scales. Reconstruction: W = diag(h) U_sign diag(l) V_sign^T diag(g).

Bit cost per W (shape d_out x d_in) at rank r:
  U_sign:  d_out * r   bits
  V_sign:  d_in  * r   bits
  h:       d_out * 16  bits (FP16)
  g:       d_in  * 16  bits
  l:       r    * 16  bits
  Total:   r*(d_out + d_in) + 16*(d_out + d_in + r)
  BPW =   Total / (d_out * d_in)

For 4096x4096: r=200 -> 0.10 BPW. r=600 -> 0.30. r=1100 -> 0.55.
"""
import torch
import torch.nn as nn
import time


def decompose_weight(W: torch.Tensor, eff_bit: float, verbose: bool = False):
    """PTQ LittleBit decomposition of a single weight matrix.

    Reconstruction: W ≈ diag(h) @ U_q @ V_q^T @ diag(g)
    where U_q = U_sign * U_scale (per-latent scale), V_q = V_sign * V_scale.
    Bit cost: r*(d_out + d_in) binary + 16*(d_out + d_in + r) for scales.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    # Pick rank r to hit the eff_bit target.
    target_bits = eff_bit * d_out * d_in
    r = max(8, int((target_bits - 16 * (d_out + d_in)) / (d_out + d_in + 16)))
    r = min(r, min(d_out, d_in))
    # SVD truncate (svd_lowrank is fast and within numerical noise for our purpose).
    U, S, V = torch.svd_lowrank(Wf, q=r, niter=2)
    # Move singular values into U for compactness.
    U = U * S.unsqueeze(0)  # [d_out, r]
    # Binarize U and V (sign), learn per-latent scale.
    U_sign = torch.sign(U)
    U_scale = U.abs().mean(dim=0).clamp(min=1e-8)  # [r] per-latent mean
    U_sign[U_sign == 0] = 1
    U_q = U_sign * U_scale.unsqueeze(0)  # [d_out, r]
    V_sign = torch.sign(V)
    V_scale = V.abs().mean(dim=0).clamp(min=1e-8)  # [r]
    V_sign[V_sign == 0] = 1
    V_q = V_sign * V_scale.unsqueeze(0)  # [d_in, r]
    # Fit per-row h and per-col g via alternating least squares to
    # minimize ||W - diag(h) U_q V_q^T diag(g)||_F^2.
    h = torch.ones(d_out)
    g = torch.ones(d_in)
    for _ in range(8):
        # Fix g, fit h: h_i = <W[i,:] .* (U_q V_q^T g)[i,:]> / ||(U_q V_q^T g)[i,:]||^2
        Ug = U_q @ (V_q.T * g)  # [d_out, d_in]
        h = (Wf * Ug).sum(dim=1) / (Ug * Ug).sum(dim=1).clamp(min=1e-8)
        # Fix h, fit g: g_j = <W[:,j] .* (h U_q V_q^T)[:,j]> / ||(h U_q V_q^T)[:,j]||^2
        hU = h.unsqueeze(1) * U_q  # [d_out, r]
        hUV = hU @ V_q.T  # [d_out, d_in]
        g = (Wf * hUV).sum(dim=0) / (hUV * hUV).sum(dim=0).clamp(min=1e-8)
    # Final reconstruction error.
    W_rec = h.unsqueeze(1) * (U_q @ V_q.T) * g.unsqueeze(0)
    recon_err = ((W_rec - Wf).norm() / Wf.norm()).item()
    # Bit accounting.
    bits = (U_sign.numel() + V_sign.numel() +
           (h.numel() + g.numel() + U_scale.numel() + V_scale.numel()) * 16)
    actual_bpw = bits / W.numel()
    if verbose:
        print(f"      decompose [{d_out}x{d_in}] r={r}: BPW={actual_bpw:.3f}, "
              f"recon_rel_err={recon_err:.4f}", flush=True)
    return (U_sign.to(torch.int8), V_sign.to(torch.int8),
            h.to(torch.float16), g.to(torch.float16),
            U_scale.to(torch.float16), V_scale.to(torch.float16),
            r, actual_bpw, recon_err)


class LittleBitLinearPTQ(nn.Module):
    """A Linear layer using the PTQ LittleBit decomposition.

    Forward: y = h * (U_sign * U_scale) @ diag(l) @ (V_sign * V_scale)^T @ g * x
    Storage: U_sign, V_sign as int8; h, g, U_scale, V_scale as FP16.
    """
    def __init__(self, U_sign, V_sign, h, g, U_scale, V_scale, bias=None):
        super().__init__()
        # store as plain attributes (no grad)
        self.U_sign = U_sign  # [d_out, r] int8
        self.V_sign = V_sign  # [d_in, r] int8
        self.h = h  # [d_out]
        self.g = g  # [d_in]
        self.U_scale = U_scale  # [r]
        self.V_scale = V_scale  # [r]
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        # x: [..., d_in]
        # Cast everything to x's dtype to keep matmul happy.
        V_sign = self.V_sign.to(x.dtype)
        U_sign = self.U_sign.to(x.dtype)
        V_scale = self.V_scale.to(x.dtype)
        U_scale = self.U_scale.to(x.dtype)
        g = self.g.to(x.dtype)
        h = self.h.to(x.dtype)
        # step 1: x_scaled = x * g
        x = x * g
        # step 2: hidden = x_scaled @ V_sign @ diag(V_scale)  (..., r)
        hidden = (x @ V_sign) * V_scale
        # step 3: hidden = hidden * U_scale  (latent scale, but we fold it in)
        hidden = hidden * U_scale
        # step 4: y = hidden @ U_sign.T  (..., d_out)
        y = hidden @ U_sign.t()
        # step 5: y = y * h
        y = y * h
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y


def ptq_quantize_model(model, eff_bit: float, target_module_names=None,
                       log_every: int = 20, verbose: bool = True):
    """Replace all nn.Linear with LittleBitLinearPTQ at the given eff_bit.

    Returns a per-tensor log of (name, original_bytes, compressed_bits, actual_bpw, recon_err).
    """
    log = []
    replaced = 0
    t0 = time.time()
    # Collect targets first so we can iterate in a deterministic order.
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if target_module_names is not None and not any(t in name for t in target_module_names):
            continue
        targets.append((name, mod))
    for i, (name, mod) in enumerate(targets):
        if verbose and (i % log_every == 0 or i == len(targets) - 1):
            print(f"  [decompose] {i+1}/{len(targets)}  {name}  shape={tuple(mod.weight.shape)}",
                  flush=True)
        with torch.no_grad():
            try:
                (U_sign, V_sign, h, g, U_scale, V_scale, r, actual_bpw, recon_err
                 ) = decompose_weight(mod.weight.data, eff_bit, verbose=False)
            except Exception as e:
                print(f"  [decompose] FAILED on {name}: {e}", flush=True)
                continue
            bias = mod.bias.data.clone() if mod.bias is not None else None
        # Replace module in parent.
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new_mod = LittleBitLinearPTQ(U_sign, V_sign, h, g, U_scale, V_scale, bias=bias)
        new_mod = new_mod.to(mod.weight.device)
        setattr(parent, child_name, new_mod)
        log.append({
            "name": name,
            "shape": list(mod.weight.shape),
            "rank": r,
            "actual_bpw": actual_bpw,
            "recon_rel_err": recon_err,
        })
        replaced += 1
    if verbose:
        print(f"  [decompose] replaced {replaced}/{len(targets)} Linear layers in "
              f"{time.time()-t0:.1f}s", flush=True)
    return log
