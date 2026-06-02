import torch


def make_mqar(batch, n_pairs, n_queries, vocab_size, device="cpu", generator=None):
    S = vocab_size - 1
    assert n_pairs <= S, "not enough symbols for distinct keys"
    total = 2 * n_pairs + 2 * n_queries
    full = torch.zeros(batch, total, dtype=torch.long)
    qpos = []
    for b in range(batch):
        perm = torch.randperm(S, generator=generator)[:n_pairs] + 1
        vals = torch.randint(1, vocab_size, (n_pairs,), generator=generator)
        for i in range(n_pairs):
            full[b, 2 * i] = perm[i]
            full[b, 2 * i + 1] = vals[i]
        qsel = torch.randint(0, n_pairs, (n_queries,), generator=generator)
        for j in range(n_queries):
            base = 2 * n_pairs + 2 * j
            full[b, base] = perm[qsel[j]]
            full[b, base + 1] = vals[qsel[j]]
    inp = full[:, :-1].contiguous()
    tgt = full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    for j in range(n_queries):
        base = 2 * n_pairs + 2 * j
        mask[:, base] = 1.0
    return inp.to(device), tgt.to(device), mask.to(device)


def make_passkey(batch, seq_len, vocab_size, key_len, device="cpu", generator=None, gap=None):
    MARK, QUERY = 1, 2
    lo = 3
    assert vocab_size > lo + 1
    assert seq_len > 2 * key_len + 4
    pre = 2
    if gap is None:
        gap = seq_len - (pre + 1 + key_len) - (1 + key_len)
    assert gap >= 0
    q_start = pre + 1 + key_len + gap + 1
    assert q_start + key_len == seq_len
    full = torch.randint(lo, vocab_size, (batch, seq_len), generator=generator)
    pk = torch.randint(lo, vocab_size, (batch, key_len), generator=generator)
    full[:, pre] = MARK
    full[:, pre + 1:pre + 1 + key_len] = pk
    full[:, q_start - 1] = QUERY
    full[:, q_start:q_start + key_len] = pk
    inp = full[:, :-1].contiguous()
    tgt = full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    mask[:, q_start - 1:q_start - 1 + key_len] = 1.0
    return inp.to(device), tgt.to(device), mask.to(device)


def decode_to_text(ids, names=None):
    names = names or {0: ".", 1: "MARK", 2: "QUERY"}
    out = []
    for t in ids.tolist():
        out.append(names.get(t, str(t)))
    return " ".join(out)
