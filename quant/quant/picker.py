"""Per-tensor bit-width picker (workstream 2, the novel contribution).

Instead of giving every tensor the same bit-width, assign bits per tensor under a
total-bit budget so sensitive tensors get more and tolerant ones get fewer,
driving average BPW below the uniform floor while holding PPL.

Sensitivity metric: Hessian-diagonal-weighted reconstruction error (GPTQ/OBQ
theory).  For a Linear with weight W [d_out, d_in] and input activations X, the
loss Hessian w.r.t. W is X^T X; its diagonal is h_j = sum ||x_j||^2 (squared
activation energy per input channel).  The sensitivity of quantizing W to b bits:

    S_l(b) = sum_j  h_j * ||W[:,j] - Q^b[:,j]||^2

This weights reconstruction error by how much each input channel is actually
used.  A tensor with high recon error but dead input channels is less sensitive
than one with moderate recon error but live channels.  One forward pass to
capture activations, no backward pass.

Assignment: multiple-choice knapsack via greedy.  Candidate bit-widths
{2,3,4,8,16} (16 = keep FP16).  Start all at minimum, greedily upgrade the
tensor giving the most sensitivity reduction per additional bit until the budget
is exhausted.
"""
import torch
import torch.nn as nn

from .quantize import quant_dequant
from .codebook import lloyd_max_gaussian, NF4


def capture_activations(model, tokenizer, n_examples=32, max_len=512, device="cuda"):
    """One forward pass over n calibration examples; capture per-Linear input
    activation energy per input channel.

    Returns {name: h} where h is a [d_in] tensor of summed squared activations.
    """
    from datasets import load_dataset
    activations = {}
    handles = []

    def make_hook(name):
        def hook(mod, inp, _out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            sq = (x * x).sum(dim=0)
            if name not in activations:
                activations[name] = sq.to(device)
            else:
                activations[name] += sq.to(device)
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
                ids = tokenizer(t, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
                model(ids)
                k += 1

    for h in handles:
        h.remove()
    return activations


def sensitivity_table(model, activations, bits_grid=(2, 3, 4, 8, 16),
                      group_size=128, method="fullrot_whlm"):
    """Compute S_l(b) for every Linear at every candidate bit-width.

    Returns {name: {bits: sensitivity}} and {name: numel}.
    """
    device = next(model.parameters()).device
    centroids_cache = {}
    nf4 = NF4.to(device)
    table = {}
    numel_map = {}

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        W = mod.weight.data
        numel_map[name] = W.numel()
        h = activations.get(name)
        if h is None:
            continue
        h = h.to(device).float()
        table[name] = {}
        Wf = W.float()
        for b in bits_grid:
            if b >= 16:
                table[name][b] = 0.0
                continue
            if b not in centroids_cache:
                centroids_cache[b] = lloyd_max_gaussian(b, device)
            Wq, _, _, _ = quant_dequant(W, b, group_size, method,
                                        centroids=centroids_cache[b], nf4=nf4)
            diff = (Wf - Wq.float())          # [d_out, d_in]
            # per-column weighted squared error: sum_j h_j * ||diff[:,j]||^2
            col_sq = (diff * diff).sum(dim=0)  # [d_in]
            table[name][b] = float((col_sq * h).sum().item())
    return table, numel_map


def assign_budget(table, numel_map, target_avg_bpw, bits_grid=(2, 3, 4, 8, 16),
                  group_size=128, floor=None):
    """Greedy multiple-choice knapsack: assign bit-widths to minimize total
    sensitivity under a total-bit budget.

    target_avg_bpw is the target average bits-per-weight (index bits + scale
    overhead).  Returns {name: bits}.

    floor: if set, no tensor is assigned below `floor` unless the budget truly
    cannot afford `floor` for all tensors.  This avoids the 2-bit cliff where
    quantization becomes catastrophic regardless of sensitivity.
    """
    min_b = floor if floor is not None else min(bits_grid)
    if floor is not None:
        min_b = max(min_b, min(bits_grid))
    # bpw per candidate = bits + 16/group (scale overhead)
    bpw_of = {b: b + 16.0 / group_size for b in bits_grid}
    bpw_of[16] = 16.0

    total_n = sum(numel_map.values())
    budget_bits = target_avg_bpw * total_n

    # start every tensor at the floor bit-width
    assignment = {name: min_b for name in table}
    # if even the floor doesn't fit, drop to the true minimum
    floor_bits = sum(numel_map[n] * bpw_of[min_b] for n in assignment)
    if floor_bits > budget_bits and floor is not None:
        true_min = min(bits_grid)
        assignment = {name: true_min for name in table}

    def total_bits():
        return sum(numel_map[n] * bpw_of[assignment[n]] for n in assignment)

    def reduction(n, b):
        return table[n][assignment[n]] - table[n][b]

    while total_bits() < budget_bits:
        best = None
        best_ratio = 0.0
        for n in assignment:
            cur = assignment[n]
            if cur >= 16:
                continue
            candidates = [b for b in bits_grid if b > cur]
            if not candidates:
                continue
            for b in candidates:
                red = reduction(n, b)
                extra_bits = numel_map[n] * (bpw_of[b] - bpw_of[cur])
                if extra_bits <= 0:
                    continue
                ratio = red / extra_bits
                if ratio > best_ratio:
                    new_total = total_bits() + extra_bits
                    if new_total <= budget_bits * 1.02:
                        best_ratio = ratio
                        best = (n, b, extra_bits)
        if best is None:
            break
        n, b, _ = best
        assignment[n] = b

    return assignment


def random_assignment(table, numel_map, target_avg_bpw,
                      bits_grid=(2, 3, 4, 8, 16), group_size=128, seed=0):
    """Random bit-width assignment at the same avg BPW as the picker.

    The control: same bit-width budget, but assigned randomly (not by
    sensitivity).  If the picker beats this, sensitivity-based assignment is
    validated.  Uses the same greedy budget-filling but with random priorities.
    """
    import random
    rng = random.Random(seed)
    min_b = min(bits_grid)
    bpw_of = {b: b + 16.0 / group_size for b in bits_grid}
    bpw_of[16] = 16.0
    total_n = sum(numel_map.values())
    budget_bits = target_avg_bpw * total_n
    assignment = {name: min_b for name in table}

    def total_bits():
        return sum(numel_map[n] * bpw_of[assignment[n]] for n in assignment)

    names = list(table.keys())
    while total_bits() < budget_bits:
        rng.shuffle(names)
        upgraded = False
        for n in names:
            cur = assignment[n]
            if cur >= 16:
                continue
            candidates = [b for b in bits_grid if b > cur]
            if not candidates:
                continue
            b = rng.choice(candidates)
            extra_bits = numel_map[n] * (bpw_of[b] - bpw_of[cur])
            if total_bits() + extra_bits <= budget_bits * 1.02:
                assignment[n] = b
                upgraded = True
                break
        if not upgraded:
            break
    return assignment


def apply_assignment(model, assignment, group_size=128, method="fullrot_whlm",
                     verbose=True):
    """Quantize each Linear at its assigned bit-width.  Returns a per-tensor log."""
    device = next(model.parameters()).device
    centroids_cache = {}
    nf4 = NF4.to(device)
    log = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        b = assignment.get(name, 16)
        if b >= 16:
            log.append({"name": name, "bits": 16, "bpw": 16.0,
                        "recon_rel_err": 0.0, "skipped": True})
            continue
        if b not in centroids_cache:
            centroids_cache[b] = lloyd_max_gaussian(b, device)
        with torch.no_grad():
            Wq, err, bpw, G = quant_dequant(mod.weight.data, b, group_size,
                                            method, centroids=centroids_cache[b],
                                            nf4=nf4)
            mod.weight.data.copy_(Wq)
        log.append({"name": name, "bits": b, "bpw": bpw,
                    "recon_rel_err": err})
        if verbose:
            print(f"    {name}: {b}bit err={err:.4f}", flush=True)
    return log


def assignment_stats(assignment, numel_map, group_size=128):
    """Return (avg_bpw, bits_histogram) for an assignment."""
    bpw_of = {b: b + 16.0 / group_size for b in assignment.values()}
    bpw_of[16] = 16.0
    total_n = sum(numel_map.values())
    total_bits = sum(numel_map[n] * bpw_of[assignment[n]] for n in assignment)
    avg_bpw = total_bits / total_n
    hist = {}
    for n, b in assignment.items():
        hist[b] = hist.get(b, 0) + 1
    return avg_bpw, hist
