"""Block-wise fine-tuning for quantized models.

Collects target block outputs from the ORIGINAL model before quantization,
then fine-tunes LayerNorm/bias params to minimize MSE between quantized
block outputs and original targets.
"""
import torch


def get_transformer_blocks(model):
    blocks = []
    for name, mod in model.named_modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            blocks.append((name, mod))
    return blocks


def collect_block_targets(model, tok, n_samples=32, seq_len=256, device="cuda"):
    """Run the original model and collect each block's output as targets."""
    from datasets import load_dataset

    blocks = get_transformer_blocks(model)
    if not blocks:
        return {}, []

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:n_samples * 4])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len * n_samples)
    input_ids = enc["input_ids"][0]
    calib = [input_ids[i:i + seq_len].unsqueeze(0).to(device)
             for i in range(0, len(input_ids) - 32, seq_len)]
    if len(calib) > n_samples:
        calib = calib[:n_samples]

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


def finetune_model(model, tok, original_targets, calib_seqs,
                   n_epochs=3, lr=3e-4, device="cuda", verbose=True):
    """Fine-tune LayerNorm/bias params to match original block outputs."""
    blocks = get_transformer_blocks(model)
    if not blocks or not original_targets:
        return

    if verbose:
        print(f"Fine-tuning {len(blocks)} blocks, {len(calib_seqs)} seqs, "
              f"{n_epochs} epochs", flush=True)

    # Collect trainable params: all norm/bias params in all blocks + final norm
    trainable = []
    for blk_name, blk_mod in blocks:
        for pname, param in blk_mod.named_parameters():
            if "norm" in pname or "bias" in pname:
                trainable.append(param)
    for pname, param in model.named_parameters():
        if "model.norm" in pname:
            trainable.append(param)

    if not trainable:
        if verbose:
            print("No trainable params found", flush=True)
        return

    # Freeze everything else, unfreeze trainable
    trainable_set = set(id(p) for p in trainable)
    for param in model.parameters():
        param.requires_grad_(id(param) in trainable_set)

    # Cast to float32 for stable gradients
    model.float()

    opt = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.95))

    # Training hooks
    train_captured = {}
    hooks = []
    def make_train_hook(name):
        def hook(mod, inp, out):
            train_captured[name] = out
        return hook

    for blk_name, blk_mod in blocks:
        hooks.append(blk_mod.register_forward_hook(make_train_hook(blk_name)))

    initial_loss = None
    for epoch in range(n_epochs):
        total_loss = 0.0
        for i, inp in enumerate(calib_seqs):
            opt.zero_grad()
            model(inp)
            loss = torch.tensor(0.0, device=device)
            for blk_name, _ in blocks:
                if blk_name in train_captured and blk_name in original_targets:
                    pred = train_captured[blk_name].float()
                    if i < len(original_targets[blk_name]):
                        target = original_targets[blk_name][i].to(pred.device).float()
                        loss = loss + ((pred - target) ** 2).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(calib_seqs)
        if initial_loss is None:
            initial_loss = avg
        if verbose:
            print(f"  Epoch {epoch+1}/{n_epochs} loss={avg:.6f}", flush=True)

    for h in hooks:
        h.remove()

    # Freeze all, cast back to half
    for param in model.parameters():
        param.requires_grad_(False)
    model.half()

    if verbose:
        print(f"Fine-tuning done: {initial_loss:.6f} -> {avg:.6f}", flush=True)
