import torch
import data as datamod
from model import SequenceModel, ModelConfig

torch.manual_seed(0)
dev = "cpu"
m = SequenceModel(ModelConfig(vocab_size=64, d_model=128, n_layers=2, mem_mode="mlp",
    mem_heads=4, mem_head_dim=32, mem_hidden=32, forget_bias=-6.0)).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
maxsn = 0.0
first = None
last = None
bad = False
for s in range(1, 121):
    inp, tgt, mask = datamod.make_mqar(16, 16, 4, 64, dev)
    opt.zero_grad(set_to_none=True)
    _, loss = m(inp, tgt, mask)
    if not torch.isfinite(loss).item():
        print("NAN_at", s)
        bad = True
        break
    if first is None:
        first = float(loss.item())
    last = float(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    sn = max(float(b.mixer.last_stats["state_norm"].item()) for b in m.blocks if hasattr(b.mixer, "last_stats"))
    maxsn = max(maxsn, sn)
    if s % 40 == 0:
        print(f"step {s} loss {last:.4f} max_state_norm {maxsn:.2f}")
ok = (not bad) and maxsn < 1000.0
print("RESULT", "PASS_BOUNDED" if ok else "FAIL", "max_state_norm", round(maxsn, 2), "first", first, "last", last)
