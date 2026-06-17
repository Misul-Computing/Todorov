import math
import torch
from torch import optim


def loss_at_init(model, make_fn, batch, device, vocab_size, tol=0.7):
    was = model.training
    model.eval()
    with torch.no_grad():
        inp, tgt, mask = make_fn(batch, device)
        _, loss = model(inp, tgt, mask)
    if was:
        model.train()
    expected = math.log(vocab_size)
    return {"loss": float(loss.item()), "expected": expected, "ok": abs(loss.item() - expected) < tol}


def overfit_one_batch(model, make_fn, batch, device, steps=300, lr=3e-3, target=0.1):
    model.train()
    inp, tgt, mask = make_fn(batch, device)
    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    return {"final_loss": last, "ok": last < target}


def causal_no_future_leak(model, vocab_size, batch, seq_len, device, p=None, tol=1e-3):
    was = model.training
    model.eval()
    if p is None:
        p = seq_len // 2
    with torch.no_grad():
        a = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        b = a.clone()
        b[:, p + 1:] = torch.randint(0, vocab_size, (batch, seq_len - p - 1), device=device)
        la, _ = model(a)
        lb, _ = model(b)
        diff = float((la[:, :p + 1] - lb[:, :p + 1]).abs().max().item())
    if was:
        model.train()
    return {"max_diff": diff, "ok": diff < tol}


def retention_floor_check(model, make_fn, batch, device, seq_len, min_keep=0.01):
    from memory import DescentMemory
    inp, _, _ = make_fn(batch, device)
    x = model.embed(inp)
    rf = None
    for blk in model.blocks:
        if isinstance(blk.mixer, DescentMemory):
            xin = blk.norm1(x)
            rf = blk.mixer.retention_floor(xin, seq_len)
            break
    if rf is None:
        return {"retention_floor": None, "ok": True}
    return {"retention_floor": rf, "ok": rf > min_keep}
