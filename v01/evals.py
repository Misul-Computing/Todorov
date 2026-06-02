import math
import torch


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


@torch.no_grad()
def eval_task(model, make_fn, n_trials, batch, device):
    was_training = model.training
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    correct_examples = 0
    total_examples = 0
    done = 0
    while done < n_trials:
        b = min(batch, n_trials - done)
        inp, tgt, mask = make_fn(b, device)
        logits, _ = model(inp)
        pred = logits.argmax(-1)
        m = mask.bool()
        hits = (pred == tgt) & m
        correct_tokens += int(hits.sum().item())
        total_tokens += int(m.sum().item())
        for i in range(b):
            mi = m[i]
            if bool(mi.any().item()):
                ok = bool(((pred[i] == tgt[i]) | (~mi)).all().item())
                correct_examples += int(ok)
                total_examples += 1
        done += b
    if was_training:
        model.train()
    p, lo, hi = wilson_ci(correct_examples, total_examples)
    return {
        "token_acc": correct_tokens / max(1, total_tokens),
        "exact_acc": correct_examples / max(1, total_examples),
        "exact_correct": correct_examples,
        "exact_total": total_examples,
        "wilson_lo": lo,
        "wilson_hi": hi,
    }


def assert_next_token_aligned(make_fn, batch, device):
    inp, tgt, mask = make_fn(batch, device)
    ok = bool((tgt[:, :-1] == inp[:, 1:]).all().item())
    assert ok, "targets are not next-token aligned with inputs"
    return True
