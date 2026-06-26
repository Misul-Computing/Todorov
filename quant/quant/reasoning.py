"""Zero-shot reasoning eval via loglikelihood multiple-choice scoring.

Same protocol as lm-eval-harness: for each question with N choices, compute the
summed log-likelihood of each choice's continuation tokens given the context,
pick the argmax.  No heavy dependency; works with any HF causal LM.

Tasks (all standard zero-shot reasoning benchmarks):
  hellaswag , 4 choices, sentence completion (common sense)
  arc_challenge, 4 choices, grade-school science
  piqa, 2 choices, physical commonsense

Use limit=N to cap examples per task for speed.
"""
import torch
from datasets import load_dataset


def _loglikelihood(model, tokenizer, context, continuation, device, max_len=1024):
    """Summed log-prob of continuation tokens given context."""
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids[0]
    cont_ids = tokenizer(continuation, return_tensors="pt").input_ids[0]
    if ctx_ids.shape[0] + cont_ids.shape[0] > max_len:
        ctx_ids = ctx_ids[-(max_len - cont_ids.shape[0]):]
    full = torch.cat([ctx_ids, cont_ids]).to(device)
    with torch.no_grad():
        logits = model(full.unsqueeze(0)).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(len(cont_ids)):
        pos = len(ctx_ids) + i - 1
        total += log_probs[pos, cont_ids[i]].item()
    return total


def _score_choices(model, tokenizer, context, choices, device):
    """Return index of highest-loglikelihood choice."""
    scores = [_loglikelihood(model, tokenizer, context, c, device) for c in choices]
    return max(range(len(scores)), key=lambda i: scores[i])


def eval_hellaswag(model, tokenizer, device, limit=200):
    ds = load_dataset("hellaswag", split="validation")
    correct, total = 0, 0
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        ctx = ex["ctx"]
        choices = [ex["endings"][j] for j in range(4)]
        pred = _score_choices(model, tokenizer, ctx, choices, device)
        if pred == int(ex["label"]):
            correct += 1
        total += 1
    return {"task": "hellaswag", "acc": correct / total, "n": total}


def eval_arc_challenge(model, tokenizer, device, limit=200):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    correct, total = 0, 0
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        q = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        ans = ex["answerKey"]
        if ans not in labels:
            continue
        choices = [f" {t}" for t in texts]
        pred = _score_choices(model, tokenizer, q, choices, device)
        if labels[pred] == ans:
            correct += 1
        total += 1
    return {"task": "arc_challenge", "acc": correct / total, "n": total}


def eval_piqa(model, tokenizer, device, limit=200):
    ds = load_dataset("piqa", split="validation")
    correct, total = 0, 0
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        goal = ex["goal"]
        choices = [f" {ex['sol1']}", f" {ex['sol2']}"]
        pred = _score_choices(model, tokenizer, goal, choices, device)
        if pred == int(ex["label"]):
            correct += 1
        total += 1
    return {"task": "piqa", "acc": correct / total, "n": total}


def eval_all(model, tokenizer, device, limit=200):
    """Run HellaSwag + ARC-Challenge. Returns list of {task, acc, n}.

    PIQA dropped: its HF dataset script is no longer supported in datasets>=3.
    The two remaining tasks cover commonsense (HellaSwag) and science reasoning
    (ARC-Challenge), sufficient for the collapse characterization.
    """
    results = []
    for fn in (eval_hellaswag, eval_arc_challenge):
        r = fn(model, tokenizer, device, limit=limit)
        print(f"  [{r['task']:16s}] acc={r['acc']:.4f}  n={r['n']}", flush=True)
        results.append(r)
    return results
