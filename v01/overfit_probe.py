import torch
import data as datamod
from model import SequenceModel, ModelConfig

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
checkpoints = {100, 300, 600, 1000, 1500, 2500}
for mode in ["mlp", "linear"]:
    m = SequenceModel(ModelConfig(
        vocab_size=64, d_model=128, n_layers=2, mem_mode=mode,
        mem_heads=4, mem_head_dim=32, mem_hidden=32, forget_bias=-6.0)).to(dev)
    inp, tgt, mask = datamod.make_mqar(8, 32, 8, 64, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
    for s in range(1, 2501):
        opt.zero_grad(set_to_none=True)
        _, loss = m(inp, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s in checkpoints:
            print(f"{mode} step {s} loss {loss.item():.4f}", flush=True)
print("PROBE_DONE", flush=True)
