import os
import math
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn.functional as F


ENWIK8_URLS = [
    "http://mattmahoney.net/dc/enwik8.zip",
    "https://data.deepai.org/enwik8.zip",
]


def ensure_enwik8(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    raw = os.path.join(cache_dir, "enwik8")
    if os.path.exists(raw) and os.path.getsize(raw) > 90_000_000:
        return raw
    zpath = os.path.join(cache_dir, "enwik8.zip")
    last = None
    for url in ENWIK8_URLS:
        try:
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extract("enwik8", cache_dir)
            return raw
        except Exception as e:
            last = e
    raise RuntimeError(f"failed to download enwik8: {last}")


def load_splits(cache_dir, n_train=90_000_000, n_val=5_000_000, n_test=5_000_000):
    raw = ensure_enwik8(cache_dir)
    data = np.fromfile(raw, dtype=np.uint8)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


class ByteData:
    def __init__(self, cache_dir, device):
        tr, va, te = load_splits(cache_dir)
        self.train = torch.from_numpy(tr.astype(np.int64))
        self.val = torch.from_numpy(va.astype(np.int64))
        self.test = torch.from_numpy(te.astype(np.int64))
        self.device = device

    def batch(self, split, batch, seq_len):
        d = getattr(self, split)
        ix = torch.randint(0, d.numel() - seq_len - 1, (batch,))
        x = torch.stack([d[i:i + seq_len] for i in ix]).to(self.device)
        y = torch.stack([d[i + 1:i + 1 + seq_len] for i in ix]).to(self.device)
        return x, y


@torch.no_grad()
def eval_bpb(model, data, split, batch, seq_len, iters, device):
    was = model.training
    model.eval()
    tot_nats = 0.0
    tot_tok = 0
    for _ in range(iters):
        x, y = data.batch(split, batch, seq_len)
        logits, _ = model(x)
        nats = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        tot_nats += float(nats.item())
        tot_tok += y.numel()
    if was:
        model.train()
    return (tot_nats / tot_tok) / math.log(2)
