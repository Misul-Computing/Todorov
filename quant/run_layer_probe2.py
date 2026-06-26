"""Layer redundancy probe v2, outside-the-box variants after v1 killed naive recycling.

v1 found: raw weight-space layers are orthogonal (cos 0.000), residual energy
1.418 (>1 = worse than zero), diff full-rank. Naive "store prototype + low-rank
residual" is dead in the raw basis.

v2 tests four variants that could still rescue cross-layer redundancy:

  A. Hadamard-domain similarity, same FWHT+signs the project uses for VQ.
     Rotation mixes coordinates; layers orthogonal raw may cluster rotated.
  B. Per-head sub-block similarity, full-matrix cos is 0, but individual
     attention heads (128-wide slices of q_proj) may share structure.
  C. Residual sparsity, v1 only checked low-rank. Top-k magnitude entries
     may hold most energy (sparse residual is cheap to store).
  D. Functional substitution (ground truth), replace layer weights with a
     prototype's, run the model, measure PPL. Weight metrics are proxies;
     this is the real test. Progressive: layers 1..k -> layer 0, all proj.
     Plus per-projection full substitution (all layers -> layer 0).
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time
import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# rotate is inlined by kaggle_push
from rotate import fwht, next_pow2, signs_for

PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
PROJ_PATH = {
    "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def get_weight(model, li, p):
    mods = dict(model.named_modules())
    return mods[f"model.layers.{li}.{PROJ_PATH[p]}"].weight


def collect(model, dev, p):
    n = model.config.num_hidden_layers
    return [get_weight(model, li, p).detach().to(torch.float32).to(dev) for li in range(n)]


def rel_resid(Wp, Wt):
    return ((Wp - Wt).norm() / Wt.norm().clamp(min=1e-12)).item()


def cos(a, b):
    a = a.flatten(); b = b.flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def hadamard_rotate_stack(weights, dev):
    """Apply the SAME sign+FWHT to every layer's weight (rows). Returns rotated stack."""
    d_out, d_in = weights[0].shape
    npad = next_pow2(d_in)
    s = signs_for(npad, dev)
    out = []
    for W in weights:
        Wp = torch.zeros(W.shape[0], npad, device=dev)
        Wp[:, :d_in] = W
        Wr = fwht(Wp * s)
        out.append(Wr)
    return out, npad


def sparsity_profile(diff):
    """What fraction of entries (sorted by |.|) holds 50/90/99% of energy?"""
    a = diff.flatten().abs()
    energy = a ** 2
    total = energy.sum().item()
    a_sorted, _ = torch.sort(a, descending=True)
    e_sorted = a_sorted ** 2
    cum = torch.cumsum(e_sorted, dim=0) / total
    n = a.numel()
    out = {}
    for t in (0.5, 0.9, 0.99, 0.999):
        idx = (cum >= t).nonzero(as_tuple=True)[0]
        out[t] = float((idx[0].item() + 1) / n) if len(idx) > 0 else 1.0
    out["top1pct_energy"] = float(e_sorted[:max(1, n // 100)].sum().item() / total)
    return out


def ppl_wt2(model, tok, dev, ctx=2048, max_tokens=20000):
    """Shortened WT2 PPL (20k tokens) for speed in the probe."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    ids = ids[:max_tokens].to(dev)
    nll, ntok = 0.0, 0
    for i in range(0, ids.shape[0] - 1, ctx):
        w = ids[i:i + ctx + 1]
        if w.shape[0] < 2:
            break
        with torch.no_grad():
            out = model(w[:-1].unsqueeze(0), labels=w[:-1].unsqueeze(0))
        n = w.shape[0] - 1
        nll += out.loss.item() * n
        ntok += n
    return math.exp(nll / ntok) if ntok else float("nan")


def restore_all(model, snap):
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and n in snap:
                m.weight.data.copy_(snap[n].to(m.weight.device))


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
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers loaded ({time.time()-t0:.0f}s)", flush=True)

    snap = {n: m.weight.data.detach().to("cpu", copy=True)
            for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    # baseline PPL
    base_ppl = ppl_wt2(model, tok, dev)
    print(f"  baseline PPL (20k tok): {base_ppl:.3f} ({time.time()-t0:.0f}s)", flush=True)

    report = {"model": "Qwen2.5-1.5B-Instruct", "n_layers": n_layers,
              "baseline_ppl": base_ppl, "projections": {}}

    # ---- A/B/C: weight-space variants (no forward passes) ----
    for p in PROJ_TYPES:
        Ws = collect(model, dev, p)
        L = len(Ws)

        # A. Hadamard-domain pairwise residual
        Wrot, npad = hadamard_rotate_stack(Ws, dev)
        rot_resid = torch.zeros(L, L)
        for i in range(L):
            for j in range(i + 1, L):
                e = rel_resid(Wrot[i], Wrot[j])
                rot_resid[i, j] = e
                rot_resid[j, i] = rel_resid(Wrot[j], Wrot[i])
        rot_offdiag = rot_resid[~torch.eye(L, dtype=bool)]

        # B. per-head sub-block similarity (q/o_proj: 12 heads x 128; k/v: 2 heads x 128)
        head_dim = 128
        n_heads = Ws[0].shape[0] // head_dim
        head_cos_mean = 0.0
        head_cos_max = 0.0
        if n_heads > 1 and Ws[0].shape[0] % head_dim == 0:
            head_sims = []
            for h in range(n_heads):
                # compare head h across layers
                blocks = [W[h * head_dim:(h + 1) * head_dim].flatten() for W in Ws]
                for i in range(L):
                    for j in range(i + 1, L):
                        head_sims.append(cos(blocks[i], blocks[j]))
            head_cos_mean = sum(head_sims) / len(head_sims)
            head_cos_max = max(head_sims)

        # C. residual sparsity (best prototype per layer)
        raw_resid = torch.zeros(L, L)
        for i in range(L):
            for j in range(i + 1, L):
                e = rel_resid(Ws[i], Ws[j])
                raw_resid[i, j] = e
                raw_resid[j, i] = rel_resid(Ws[j], Ws[i])
        raw_offdiag = raw_resid[~torch.eye(L, dtype=bool)]
        # sparsity for layer 1's best prototype diff (representative)
        j = 1
        row = raw_resid[:, j].clone(); row[j] = float("inf")
        best_i = int(row.argmin().item())
        sparsity = sparsity_profile(Ws[best_i] - Ws[j])

        report["projections"][p] = {
            "shape": list(Ws[0].shape),
            "raw_resid_mean": float(raw_offdiag.mean().item()),
            "raw_resid_min": float(raw_offdiag.min().item()),
            "hadamard_resid_mean": float(rot_offdiag.mean().item()),
            "hadamard_resid_min": float(rot_offdiag.min().item()),
            "n_heads": n_heads,
            "head_cos_mean": float(head_cos_mean),
            "head_cos_max": float(head_cos_max),
            "residual_sparsity": sparsity,
        }
        for W in Ws:
            del W
        for W in Wrot:
            del W
        torch.cuda.empty_cache() if dev == "cuda" else None
        print(f"  [{p:10s}] raw_resid={raw_offdiag.mean().item():.3f} "
              f"had_resid={rot_offdiag.mean().item():.3f} "
              f"head_cos_max={head_cos_max:.3f} "
              f"sparsity(99% in top {sparsity[0.99]*100:.1f}%) "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- D: functional substitution (ground truth) ----
    print(f"\n=== D: functional substitution PPL ===", flush=True)
    func = {}

    # D1. per-projection: ALL layers -> layer 0, for that projection only
    func["per_projection_all_to_layer0"] = {}
    for p in PROJ_TYPES:
        restore_all(model, snap)
        w0 = snap[f"model.layers.0.{PROJ_PATH[p]}"].to(dev)
        with torch.no_grad():
            for li in range(1, n_layers):
                get_weight(model, li, p).copy_(w0.to(get_weight(model, li, p).dtype))
        ppl = ppl_wt2(model, tok, dev)
        func["per_projection_all_to_layer0"][p] = ppl
        print(f"  D1 {p:10s} all->L0: PPL={ppl:.2f} (base {base_ppl:.2f}) "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    # D2. ALL projections ALL layers -> layer 0 (extreme)
    restore_all(model, snap)
    with torch.no_grad():
        for p in PROJ_TYPES:
            w0 = snap[f"model.layers.0.{PROJ_PATH[p]}"].to(dev)
            for li in range(1, n_layers):
                get_weight(model, li, p).copy_(w0.to(get_weight(model, li, p).dtype))
    ppl = ppl_wt2(model, tok, dev)
    func["all_proj_all_to_layer0"] = ppl
    print(f"  D2 all proj all->L0: PPL={ppl:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # D3. progressive: layers 1..k -> layer 0 (all projections), for k in {1,7,14,21,27}
    func["progressive_all_to_layer0"] = {}
    for k in [1, 7, 14, 21, 27]:
        restore_all(model, snap)
        with torch.no_grad():
            for p in PROJ_TYPES:
                w0 = snap[f"model.layers.0.{PROJ_PATH[p]}"].to(dev)
                for li in range(1, min(k + 1, n_layers)):
                    get_weight(model, li, p).copy_(w0.to(get_weight(model, li, p).dtype))
        ppl = ppl_wt2(model, tok, dev)
        func["progressive_all_to_layer0"][k] = ppl
        print(f"  D3 layers 1..{k:2d} -> L0: PPL={ppl:.2f} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    report["functional"] = func

    restore_all(model, snap)
    out_dir = "/kaggle/working/runs/layer_probe2" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "layer_probe2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== HEADLINE ===", flush=True)
    print(f"baseline PPL: {base_ppl:.3f}", flush=True)
    print(f"D2 (everything -> layer 0): PPL {func['all_proj_all_to_layer0']:.2f}", flush=True)
    print(f"D3 progressive cliff: " +
          " ".join(f"k={k}:{func['progressive_all_to_layer0'][k]:.1f}"
                   for k in [1, 7, 14, 21, 27]), flush=True)
    print(f"saved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
