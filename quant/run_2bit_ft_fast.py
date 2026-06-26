"""2-bit attack #1 (fast): block-wise fine-tuning on 2-bit E8 RVQ.

The stock finetune.py casts the whole 1.5B model to fp32 (6GB + slow
forward/backward) which OOM'd/timeout'd on T4. This lean version keeps
the model in fp16 and only trains the small norm/bias params (a few MB),
using autocast for the forward pass. ~5x faster, fits easily.

Workstream 7 proved FT recovers 2-bit VQ 134 -> 48 PPL (2.8x). 2-bit E8
starts at 27.7, so FT should land ~10-15 PPL.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from quantize import snapshot, restore
from eval import ppl_wt2
from e8lattice import quantize_model_e8rvq


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
    # trainable: norm/bias params (small). Keep model in fp16; these params
    # get fp32 copies for stable optimization, written back to fp16 after.
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

    # keep trainable params in fp32 for stable gradients
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
    # write trainable back to fp16, freeze
    for p in trainable:
        p.data = p.data.half()
        p.requires_grad_(False)
    print(f"  FT done: {initial_loss:.6f} -> {avg:.6f}", flush=True)


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

    print("collecting block targets...", flush=True)
    targets, calib = collect_targets(model, tok, n_samples=16, seq_len=192, dev=dev)
    print(f"  {len(targets)} blocks, {len(calib)} seqs ({time.time()-t0:.0f}s)", flush=True)

    results = []
    for epochs in [2, 5]:
        restore(model, snap)
        quantize_model_e8rvq(model, total_bits=2, group_size=128, verbose=False)
        if epochs == 2:
            pre_ppl = ppl_wt2(model, tok, device=dev)
            print(f"2-bit E8 pre-FT PPL: {pre_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"\n=== FT {epochs} epochs ===", flush=True)
        tq = time.time()
        fast_ft(model, tok, targets, calib, n_epochs=epochs, lr=3e-4, dev=dev)
        post_ppl = ppl_wt2(model, tok, device=dev)
        degr = (post_ppl - base_ppl) / base_ppl
        results.append({"cfg": f"e8_2bit_ft{epochs}", "bpw": 2.125,
                        "ppl": post_ppl, "degr": degr})
        print(f"  2-bit E8 + FT({epochs}): ppl={post_ppl:.4f} degr={degr:+.4f} ({time.time()-tq:.0f}s)", flush=True)
        torch.cuda.empty_cache() if dev == "cuda" else None

    out_dir = "/kaggle/working/runs/2bit_ft" if os.path.isdir("/kaggle/working") else \
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "2bit_ft")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"baseline_ppl": base_ppl, "pre_ft_ppl": pre_ppl,
                   "results": results}, f, indent=2)
    print(f"\nsaved {out_dir}/summary.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
