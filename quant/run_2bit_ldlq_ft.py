"""2-bit attacks part 2: LDLQ with real activations + block-wise FT.

LDLQ: QuIP#-style adaptive rounding with real activation Hessians (X^T X),
applied to attn projections (d_in<=2048). MLP uses plain E8.

FT: lean block-wise fine-tuning (fp16 model, only norm/bias trained).
Workstream 7 proved 2.8x recovery on 2-bit VQ (134->48). E8 starts at
27.7, so FT should land ~10-15 PPL.
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

GROUP = 128


# ============ LDLQ with real activations ============
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


# ============ Block-wise FT ============
def get_blocks(model):
    blocks = []
    for name, mod in model.named_modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            blocks.append((name, mod))
    return blocks


def collect_targets(model, tok, n_samples=16, seq_len=192, dev="cuda"):
    blocks = get_blocks(model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:n_samples * 4])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len * n_samples)
    ids = enc["input_ids"][0]
    calib = [ids[i:i + seq_len].unsqueeze(0).to(dev)
             for i in range(0, len(ids) - 32, seq_len)][:n_samples]
    targets = {}
    captured = {}
    hooks = []
    def make_hook(name):
        def hook(mod, inp, out):
            captured[name] = out.detach()
        return hook
    for blk_name, blk_mod in blocks:
        hooks.append(blk_mod.register_forward_hook(make_hook(blk_name)))
    model.eval()
    with torch.no_grad():
        for inp in calib:
            model(inp)
            for blk_name, _ in blocks:
                if blk_name in captured:
                    targets.setdefault(blk_name, []).append(captured[blk_name].clone())
    for h in hooks:
        h.remove()
    return targets, calib


def fast_ft(model, tok, targets, calib, n_epochs=3, lr=3e-4, dev="cuda"):
    blocks = get_blocks(model)
    trainable = []
    for blk_name, blk_mod in blocks:
        for pname, param in blk_mod.named_parameters():
            if "norm" in pname or "bias" in pname:
                param.requires_grad_(True)
                trainable.append(param)
            else:
                param.requires_grad_(False)
    for pname, param in model.named_parameters():
        if "model.norm" in pname:
            param.requires_grad_(True)
            trainable.append(param)
    for p in trainable:
        p.data = p.data.float()
    opt = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.95))
    train_captured = {}
    hooks = []
    def make_hook(name):
        def hook(mod, inp, out):
            train_captured[name] = out
        return hook
    for blk_name, blk_mod in blocks:
        hooks.append(blk_mod.register_forward_hook(make_hook(blk_name)))
    initial_loss = None
    for epoch in range(n_epochs):
        total = 0.0
        for i, inp in enumerate(calib):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev=="cuda")):
                model(inp)
            loss = torch.tensor(0.0, device=dev)
            for blk_name, _ in blocks:
                if blk_name in train_captured and blk_name in targets and i < len(targets[blk_name]):
                    pred = train_captured[blk_name].float()
                    tgt = targets[blk_name][i].to(pred.device).float()
                    loss = loss + ((pred - tgt) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(calib)
        if initial_loss is None:
            initial_loss = avg
        print(f"  epoch {epoch+1}/{n_epochs} loss={avg:.6f}", flush=True)
    for h in hooks:
        h.remove()
    for p in trainable:
        p.data = p.data.half()
        p.requires_grad_(False)
    print(f"  FT done: {initial_loss:.6f} -> {avg:.6f}", flush=True)


def run_ft(model, tok, snap, base_ppl, dev, results, t0):
    print(f"\n=== collecting block targets for FT ===", flush=True)
    targets, calib = collect_targets(model, tok, n_samples=16, seq_len=192, dev=dev)
    print(f"  {len(targets)} blocks, {len(calib)} seqs ({time.time()-t0:.0f}s)", flush=True)
    for epochs in [3, 7, 15]:
        restore(model, snap)
        quantize_model_e8rvq(model, total_bits=2, group_size=128, verbose=False)
        if epochs == 3:
            pre_ppl = ppl_wt2(model, tok, device=dev)
            print(f"  2-bit E8 pre-FT PPL: {pre_ppl:.4f}", flush=True)
        print(f"\n=== FT {epochs} epochs ===", flush=True)
        tq = time.time()
        fast_ft(model, tok, targets, calib, n_epochs=epochs, lr=3e-4, dev=dev)
        post_ppl = ppl_wt2(model, tok, device=dev)
        degr = (post_ppl - base_ppl) / base_ppl
        results.append({"cfg": f"e8_2bit_ft{epochs}", "bpw": 2.125,
                        "ppl": post_ppl, "degr": degr})
        print(f"  2-bit E8 + FT({epochs}): ppl={post_ppl:.4f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None
    return pre_ppl


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
    run_ldlq(model, tok, snap, base_ppl, dev, results, t0)
    pre_ppl = run_ft(model, tok, snap, base_ppl, dev, results, t0)

    print(f"\n{'='*70}", flush=True)
    print(f"{'cfg':>22s} {'bpw':>7s} {'ppl':>10s} {'degr':>8s}", flush=True)
    print(f"{'-'*70}", flush=True)
    for r in results:
        print(f"{r['cfg']:>22s} {r['bpw']:>7.3f} {r['ppl']:>10.3f} {r['degr']:>+8.4f}", flush=True)

    out_dir = "/root/novelquant/runs/2bit_ldlq_ft" if os.path.isdir("/root/novelquant") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_ldlq_ft")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "pre_ft_ppl": pre_ppl,
                   "results": results}, f, indent=2)
    print(f"\nsaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
