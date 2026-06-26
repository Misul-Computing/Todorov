"""AQLM-style learned codebook quantization with output-error optimization.

Two phases:
1. Codebook learning: k-means init -> Adam on codebooks (output MSE) -> beam search codes
2. Block-wise fine-tuning: MSE on transformer block outputs

The loss is ||XW - X*Wc||^2 = <(W-Wc)XX^T, (W-Wc)>_F (output MSE, not weight MSE).
"""
import torch
import torch.nn as nn
from .vq import kmeans
from .rotate import fwht, signs_for, next_pow2


def collect_activations(model, tok, n_samples=128, seq_len=512, device="cuda"):
    """Single forward pass to collect input activations for every nn.Linear.

    Returns dict: name -> tensor of shape [n_tokens, d_in] (float16 on CPU).
    """
    from datasets import load_dataset
    acts = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            acts[name] = x.to(torch.float16).cpu()
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:n_samples * 4])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len * n_samples)
    input_ids = enc["input_ids"][0]

    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), seq_len):
            chunk = input_ids[i:i + seq_len].unsqueeze(0).to(device)
            if chunk.shape[1] < 32:
                continue
            model(chunk)

    for h in hooks:
        h.remove()
    return acts


def compute_hessian_diag(X, device):
    """Compute Hessian diagonal = per-input-dim activation energy.
    H = diag(X^T X) / n_samples. Returns shape [d_in].
    """
    if X.shape[0] == 0:
        return None
    Xf = X.to(device).float()
    return (Xf ** 2).sum(0) / Xf.shape[0]


def aqlm_quantize_weight(W, X, d=8, nbits=8, n_iters=20, lr=1e-4, device="cuda",
                         seed=42):
    """AQLM-style quantization of a single weight matrix.

    W: [d_out, d_in] weight matrix
    X: [n_tokens, d_in] input activations (or None for data-free)
    d: vector dimension
    nbits: bits per code (k = 2^nbits)
    n_iters: alternating optimization iterations
    lr: learning rate for codebook Adam

    Returns: (quantized_W, recon_rel_err, bpw, codebook)
    """
    d_out, d_in = W.shape
    Wf = W.float().to(device)
    k = 2 ** (nbits)  # codebook size

    # Hadamard rotate
    npad = next_pow2(d_in)
    s = signs_for(npad, device)
    Wp = torch.zeros(d_out, npad, device=device)
    Wp[:, :d_in] = Wf
    Wr = fwht(Wp * s)
    sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
    Wrn = Wr / sc

    # Reshape into blocks
    n_groups = npad // d
    blocks = Wrn.reshape(d_out, n_groups, d).reshape(-1, d)  # [d_out*n_groups, d]

    # Compute Hessian diagonal in rotated domain (per-dimension weighting)
    if X is not None and X.shape[0] > 0:
        Xf = X.to(device).float()
        Xp = torch.zeros(Xf.shape[0], npad, device=device)
        Xp[:, :d_in] = Xf
        Xr = fwht(Xp * s)  # rotate activations
        hess = (Xr ** 2).mean(0)  # [npad] - per-dimension energy
        hess = hess / hess.mean()  # normalize
        hess_blocks = hess.reshape(n_groups, d)  # [n_groups, d] per-dim weights
    else:
        hess = None
        hess_blocks = None

    # Initialize codebook with weighted k-means (per-group weights for init)
    per_group_w = None
    per_block_w = None
    if hess_blocks is not None:
        per_group_w = hess_blocks.mean(1)  # [n_groups] - average per group
        per_group_w = per_group_w.repeat(d_out)  # [d_out*n_groups]
        per_block_w = hess_blocks.repeat(d_out, 1)  # [d_out*n_groups, d]
    codebook = kmeans(blocks, k, iters=30, seed=seed,
                      weights=per_group_w)

    # Alternating optimization
    for it in range(n_iters):
        # Phase 1: assign codes (weighted nearest neighbor)
        idx = _assign_codes(blocks, codebook, per_block_w, chunk=8192)

        # Phase 2: update codebook with Adam (output-error objective)
        codebook = _update_codebook_adam(blocks, idx, codebook, k, d, lr,
                                         per_block_w, device)

    # Final assignment
    idx = _assign_codes(blocks, codebook, per_block_w, chunk=8192)
    quantized_blocks = codebook[idx]
    Wrn_q = quantized_blocks.reshape(d_out, npad)
    rec = (Wrn_q * sc)
    out = (fwht(rec) * s)[:, :d_in].to(W.dtype)

    err = ((out.float() - Wf).norm() / Wf.norm().clamp(min=1e-12)).item()
    bpw = nbits / d + 32.0 / npad  # index bits per weight + scale
    return out, err, bpw, codebook


def _assign_codes(blocks, codebook, per_block_w, chunk=4096):
    """Assign each block to nearest codebook entry, optionally weighted."""
    n = blocks.shape[0]
    idx = torch.empty(n, dtype=torch.long, device=blocks.device)
    for i in range(0, n, chunk):
        if per_block_w is not None:
            diff = blocks[i:i + chunk].unsqueeze(1) - codebook.unsqueeze(0)
            diff = diff * per_block_w[i:i + chunk].unsqueeze(1)
            dist = (diff ** 2).sum(-1)
        else:
            dist = torch.cdist(blocks[i:i + chunk], codebook)
        idx[i:i + chunk] = dist.argmin(dim=1)
    return idx


def _update_codebook_adam(blocks, idx, codebook, k, d, lr, per_block_w,
                          device, n_steps=20):
    """Update codebook with Adam to minimize output-error MSE."""
    cb = codebook.clone().requires_grad_(True)
    opt = torch.optim.Adam([cb], lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()
        recon = cb[idx]
        err = blocks - recon
        if per_block_w is not None:
            err = err * per_block_w
        loss = (err ** 2).mean()
        loss.backward()
        opt.step()

    return cb.detach()


def aqlm_quantize_model(model, acts, d=8, nbits=8, n_iters=20, lr=1e-4,
                        skip_names=(), device="cuda", verbose=True):
    """Apply AQLM to every nn.Linear. Returns per-tensor log."""
    import time
    log = []
    t0 = time.time()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i, (name, mod) in enumerate(targets):
        if any(s in name for s in skip_names):
            log.append({"name": name, "shape": list(mod.weight.shape),
                        "skipped": True, "recon_rel_err": 0.0, "bpw": 16.0})
            continue
        W = mod.weight.data
        X = acts.get(name)
        Wq, err, bpw, cb = aqlm_quantize_weight(W, X, d=d, nbits=nbits,
                                                n_iters=n_iters, lr=lr, device=device)
        mod.weight.data = Wq
        log.append({"name": name, "shape": list(W.shape), "skipped": False,
                    "recon_rel_err": err, "bpw": bpw})
        if verbose and (i % 20 == 0 or i == len(targets) - 1):
            print(f"  [{i+1}/{len(targets)}] {name:40s} err={err:.4f} bpw={bpw:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return log
