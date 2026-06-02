import math
import torch
import pytest
from model import SequenceModel, ModelConfig
from memory import DescentMemory, MemoryConfig
import data as datamod
import evals
import sanity


def tiny(mode="mlp", kinds=()):
    return SequenceModel(ModelConfig(
        vocab_size=32, d_model=64, n_layers=2, n_heads=4,
        mem_mode=mode, mem_heads=2, mem_head_dim=16, mem_hidden=16,
        layer_kinds=kinds))


@pytest.mark.parametrize("mode", ["linear", "mlp"])
def test_forward_backward(mode):
    m = tiny(mode)
    inp, tgt, mask = datamod.make_mqar(4, 8, 2, 32)
    logits, loss = m(inp, tgt, mask)
    assert logits.shape == (4, inp.size(1), 32)
    assert torch.isfinite(loss).item()
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert g > 0


def test_mqar_alignment():
    mk = lambda b, d: datamod.make_mqar(b, 8, 2, 32, d)
    assert evals.assert_next_token_aligned(mk, 8, "cpu")


def test_passkey_alignment():
    mk = lambda b, d: datamod.make_passkey(b, 64, 32, 4, d)
    assert evals.assert_next_token_aligned(mk, 8, "cpu")


def test_passkey_shapes_and_mask():
    inp, tgt, mask = datamod.make_passkey(4, 64, 32, 4)
    assert inp.shape == (4, 63)
    assert int(mask.sum(dim=1)[0].item()) == 4


def test_causal_no_future_leak():
    m = tiny("mlp", kinds=("mem", "attn"))
    r = sanity.causal_no_future_leak(m, 32, 2, 32, "cpu")
    assert r["ok"], r


def test_retention_floor_safe():
    m = tiny("linear")
    mk = lambda b, d: datamod.make_mqar(b, 8, 2, 32, d)
    r = sanity.retention_floor_check(m, mk, 4, "cpu", 64)
    assert r["ok"], r


def test_overfit_reduces_loss():
    m = tiny("mlp")
    mk = lambda b, d: datamod.make_mqar(b, 8, 2, 32, d)
    r = sanity.overfit_one_batch(m, mk, 8, "cpu", steps=150, target=1.0)
    assert r["final_loss"] < math.log(32) * 0.8, r


def test_eval_untrained_near_chance():
    m = tiny("mlp")
    mk = lambda b, d: datamod.make_mqar(b, 8, 2, 32, d)
    ev = evals.eval_task(m, mk, 64, 32, "cpu")
    assert ev["exact_total"] == 64
    assert ev["exact_acc"] < 0.3
