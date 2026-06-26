"""Cross-layer redundancy probe (Kaggle T4, no forward passes).

Question: are transformer layers redundant enough that we can store a few
"prototype" layers and reconstruct the rest as prototype + low-rank residual?

Per projection type (q/k/v/o/gate/up/down), measures on Qwen2.5-1.5B:
  1. Pairwise residual energy ||W_i - W_j||_F / ||W_j||_F (28x28).
  2. For each layer j, the best single prototype i!=j.
  3. SVD spectrum of (W_i - W_j): rank r capturing 50/90/99/99.9% of residual.
     Low rank => residual is cheap; high rank => no redundancy.

No forward passes. Pure weight-space linear algebra on GPU.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
from transformers import AutoModelForCausalLM

PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
PROJ_PATH = {
    "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def collect_layer_weights(model, dev):
    n_layers = model.config.num_hidden_layers
    mods = dict(model.named_modules())
    by_proj = {p: [] for p in PROJ_TYPES}
    for li in range(n_layers):
        for p in PROJ_TYPES:
            full = f"model.layers.{li}.{PROJ_PATH[p]}"
            W = mods[full].weight.data.detach().to(torch.float32).to(dev)
            by_proj[p].append((li, W))
    return by_proj


def rel_resid_energy(W_proto, W_target):
    return ((W_proto - W_target).norm() / W_target.norm().clamp(min=1e-12)).item()


def cosine(a, b):
    a = a.flatten(); b = b.flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def svd_rank_for_energy(diff, thresholds=(0.5, 0.9, 0.99, 0.999)):
    s = torch.linalg.svdvals(diff)
    energy = s ** 2
    total = energy.sum().item()
    cum = torch.cumsum(energy, dim=0) / total
    out = {}
    for t in thresholds:
        idx = (cum >= t).nonzero(as_tuple=True)[0]
        out[t] = int(idx[0].item()) + 1 if len(idx) > 0 else int(s.numel())
    out["n_sv"] = int(s.numel())
    out["top1_frac"] = float(energy[0].item() / total)
    out["top10_frac"] = float(energy[:10].sum().item() / total)
    return out


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)
    print("loading Qwen2.5-1.5B-Instruct (fp16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16,
        device_map=dev, low_cpu_mem_usage=True).eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, loaded in {time.time()-t0:.1f}s", flush=True)

    by_proj = collect_layer_weights(model, dev)
    report = {"model": "Qwen2.5-1.5B-Instruct", "n_layers": n_layers,
              "projections": {}}

    for p in PROJ_TYPES:
        layers = by_proj[p]
        L = len(layers)
        cos_mat = torch.zeros(L, L)
        resid_mat = torch.zeros(L, L)
        for i in range(L):
            for j in range(i + 1, L):
                e = rel_resid_energy(layers[i][1], layers[j][1])
                c = cosine(layers[i][1], layers[j][1])
                resid_mat[i, j] = e
                resid_mat[j, i] = rel_resid_energy(layers[j][1], layers[i][1])
                cos_mat[i, j] = c
                cos_mat[j, i] = c

        per_layer = []
        for j in range(L):
            row = resid_mat[:, j].clone()
            row[j] = float("inf")
            best_i = int(row.argmin().item())
            best_e = float(row[best_i].item())
            entry = {"layer": j, "best_proto": best_i,
                     "resid_energy": best_e,
                     "cos_to_proto": float(cos_mat[best_i, j].item())}
            diff = layers[best_i][1] - layers[j][1]
            entry["svd"] = svd_rank_for_energy(diff)
            per_layer.append(entry)

        offdiag = resid_mat[~torch.eye(L, dtype=bool)]
        report["projections"][p] = {
            "shape": list(layers[0][1].shape),
            "n_params_per_layer": int(layers[0][1].numel()),
            "resid_energy_mean": float(offdiag.mean().item()),
            "resid_energy_min": float(offdiag.min().item()),
            "resid_energy_median": float(offdiag.median().item()),
            "cos_mean": float((cos_mat[~torch.eye(L, dtype=bool)]).mean().item()),
            "cos_max_offdiag": float((cos_mat[~torch.eye(L, dtype=bool)]).max().item()),
            "per_layer": per_layer,
        }
        # free GPU
        for _, W in layers:
            del W
        torch.cuda.empty_cache() if dev == "cuda" else None
        print(f"  [{p:10s}] shape={tuple(layers[0][1].shape)} "
              f"resid mean={offdiag.mean().item():.3f} "
              f"min={offdiag.min().item():.3f} "
              f"cos_mean={(cos_mat[~torch.eye(L,dtype=bool)]).mean().item():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    total_params = 0.0
    weighted_resid = 0.0
    for p in PROJ_TYPES:
        d = report["projections"][p]
        w = d["n_params_per_layer"] * n_layers
        weighted_resid += d["resid_energy_mean"] * w
        total_params += w
    report["weighted_resid_energy_mean"] = weighted_resid / total_params

    # rank histogram at 99% energy across all projections
    print(f"\n=== rank for 99% residual energy (best-prototype diff) ===", flush=True)
    for p in PROJ_TYPES:
        ranks = [pl["svd"][0.99] for pl in report["projections"][p]["per_layer"]]
        n_sv = report["projections"][p]["per_layer"][0]["svd"]["n_sv"]
        print(f"  {p:10s} rank99%: mean={sum(ranks)/len(ranks):.0f} "
              f"max={max(ranks)} of {n_sv} sv "
              f"(top10_frac={report['projections'][p]['per_layer'][0]['svd']['top10_frac']:.3f})",
              flush=True)

    out_dir = "/kaggle/working/runs/layer_probe" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "layer_probe")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== HEADLINE ===", flush=True)
    print(f"weighted mean residual energy (subst best non-identical layer): "
          f"{report['weighted_resid_energy_mean']:.3f}", flush=True)
    print(f"(1.0 = no redundancy; 0.0 = perfect duplicate)", flush=True)
    print(f"saved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
