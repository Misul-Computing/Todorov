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


def _structured_strip(batch, strip_len, p_stay, generator):
    strip = torch.zeros(batch, strip_len, dtype=torch.long)
    strip[:, 0] = (torch.rand(batch, generator=generator) < 0.5).long()
    for i in range(1, strip_len):
        stay = torch.rand(batch, generator=generator) < p_stay
        strip[:, i] = torch.where(stay, strip[:, i - 1], 1 - strip[:, i - 1])
    return strip


def make_sweep(batch, strip_len, p_stay, device="cpu", generator=None):
    BLANK, FLAT, BUMP = 0, 1, 2
    posbase = 4
    total = 2 * strip_len
    full = torch.zeros(batch, total, dtype=torch.long)
    strip = _structured_strip(batch, strip_len, p_stay, generator)
    for i in range(strip_len):
        full[:, 2 * i] = posbase + i
        full[:, 2 * i + 1] = torch.where(strip[:, i].bool(), torch.full((batch,), BUMP, dtype=torch.long), torch.full((batch,), FLAT, dtype=torch.long))
    inp = full[:, :-1].contiguous()
    tgt = full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    for i in range(strip_len):
        mask[:, 2 * i] = 1.0
    return inp.to(device), tgt.to(device), mask.to(device)


def make_touch_world(batch, strip_len, n_queries, bump_prob, touch, device="cpu", generator=None, noise=False, p_stay=None):
    BLANK, FLAT, BUMP, QUERY = 0, 1, 2, 3
    posbase = 4
    assert n_queries <= strip_len
    total = 2 * strip_len + 3 * n_queries
    full = torch.zeros(batch, total, dtype=torch.long)
    if p_stay is None:
        strip = (torch.rand(batch, strip_len, generator=generator) < bump_prob).long()
    else:
        strip = _structured_strip(batch, strip_len, p_stay, generator)
    felt = (torch.rand(batch, strip_len, generator=generator) < bump_prob).long() if noise else strip
    ar = torch.arange(batch)
    for i in range(strip_len):
        full[:, 2 * i] = posbase + i
        if touch:
            full[:, 2 * i + 1] = torch.where(felt[:, i].bool(), torch.full((batch,), BUMP, dtype=torch.long), torch.full((batch,), FLAT, dtype=torch.long))
        else:
            full[:, 2 * i + 1] = BLANK
    qpos_all = torch.rand(batch, strip_len, generator=generator).argsort(dim=1)[:, :n_queries]
    for j in range(n_queries):
        base = 2 * strip_len + 3 * j
        full[:, base] = QUERY
        qpos = qpos_all[:, j]
        full[:, base + 1] = posbase + qpos
        ans = strip[ar, qpos]
        full[:, base + 2] = torch.where(ans.bool(), torch.full((batch,), BUMP, dtype=torch.long), torch.full((batch,), FLAT, dtype=torch.long))
    inp = full[:, :-1].contiguous()
    tgt = full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    for j in range(n_queries):
        mask[:, 2 * strip_len + 3 * j + 1] = 1.0
    return inp.to(device), tgt.to(device), mask.to(device)


def make_touch_count(batch, strip_len, bump_prob, touch, device="cpu", generator=None, noise=False, flip_p=0.0, return_strip=False, p_stay=None):
    BLANK, FLAT, BUMP, COUNT = 0, 1, 2, 3
    posbase = 4
    countbase = posbase + strip_len
    total = 2 * strip_len + 2
    full = torch.zeros(batch, total, dtype=torch.long)
    if p_stay is None:
        strip = (torch.rand(batch, strip_len, generator=generator) < bump_prob).long()
    else:
        strip = _structured_strip(batch, strip_len, p_stay, generator)
    felt = (torch.rand(batch, strip_len, generator=generator) < bump_prob).long() if noise else strip.clone()
    if flip_p > 0.0:
        flip = torch.rand(batch, strip_len, generator=generator) < flip_p
        felt = torch.where(flip, 1 - felt, felt)
    for i in range(strip_len):
        full[:, 2 * i] = posbase + i
        if touch:
            full[:, 2 * i + 1] = torch.where(felt[:, i].bool(), torch.full((batch,), BUMP, dtype=torch.long), torch.full((batch,), FLAT, dtype=torch.long))
        else:
            full[:, 2 * i + 1] = BLANK
    full[:, 2 * strip_len] = COUNT
    full[:, 2 * strip_len + 1] = countbase + strip.sum(dim=1)
    inp = full[:, :-1].contiguous()
    tgt = full[:, 1:].contiguous()
    mask = torch.zeros_like(inp, dtype=torch.float32)
    mask[:, 2 * strip_len] = 1.0
    if return_strip:
        return inp.to(device), tgt.to(device), mask.to(device), strip.to(device)
    return inp.to(device), tgt.to(device), mask.to(device)


def decode_to_text(ids, names=None):
    names = names or {0: ".", 1: "MARK", 2: "QUERY"}
    out = []
    for t in ids.tolist():
        out.append(names.get(t, str(t)))
    return " ".join(out)
