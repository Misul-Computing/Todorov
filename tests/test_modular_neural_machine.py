import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from dataclasses import fields, is_dataclass
from types import MethodType

import pytest
import torch
from torch import nn

import src.model.modular_sources as modular_sources
from src.model.modular_neural_machine import (
    RECURRENT_BLOCK_INDICES,
    ROUTED_BLOCK_INDICES,
    RUNG_ONE_RESET_POSITIONS,
    SEQUENCE_SCHEDULE,
    ModularNeuralMachine,
    copy_compatible_state,
    is_router_parameter,
    named_parameter_categories,
    parameter_category,
    rung_one_config,
    rung_two_config,
)
from src.model.modular_sources import (
    DenseCausalMixer,
    PublicRoutedMixer,
    ResetAwareRecurrentMixer,
    SourceFeatureMixer,
    verify_frozen_source_hashes,
)


def test_frozen_source_loader_uses_one_snapshot_and_replaces_poisoned_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "governed.py"
    raw = b"value = 7\n"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    name = "_todorov_test_governed_snapshot"
    poisoned = object()
    sys.modules[name] = poisoned
    real_compile = compile

    def mutate_after_snapshot(source, filename, mode, **kwargs):
        path.write_bytes(b"value = 99\n")
        return real_compile(source, filename, mode, **kwargs)

    monkeypatch.setattr(modular_sources, "compile", mutate_after_snapshot, raising=False)
    try:
        loaded = modular_sources._load_module(name, path, digest)
        assert loaded is not poisoned
        assert loaded.value == 7
        assert sys.modules[name] is loaded
    finally:
        sys.modules.pop(name, None)


def test_frozen_source_loader_rejects_hash_drift_with_typed_surface(tmp_path: Path) -> None:
    path = tmp_path / "governed.py"
    path.write_bytes(b"value = 7\n")
    with pytest.raises(modular_sources.FrozenSourceMismatchError) as caught:
        modular_sources._load_module("_todorov_test_governed_mismatch", path, "0" * 64)
    assert caught.value.surface == str(path)
    assert "_todorov_test_governed_mismatch" not in sys.modules


def _construct(role="selected", seed=11):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ModularNeuralMachine(rung_one_config(role))


def _construct_rung_two(seed=83):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ModularNeuralMachine(rung_two_config())


def _tensor_bytes_equal(left, right):
    return _tensor_identity(left) == _tensor_identity(right)


def _tensor_identity(value):
    if value.device.type != "cpu":
        return str(value.device), str(value.dtype), tuple(value.shape)
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    body_tensor = value.detach().contiguous().reshape(-1).clone()
    body = bytes(body_tensor.untyped_storage())
    return header, body, hashlib.sha256(header + b"\n" + body).hexdigest()


def _state_snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _mutate_every_state_tensor(module):
    with torch.no_grad():
        for index, value in enumerate(module.state_dict().values()):
            value.add_((index + 1) / 1000)


def _changed_state_names(before, after):
    return {
        name
        for name in before
        if not _tensor_bytes_equal(before[name], after[name])
    }


def _transient_graph_objects(root):
    registered = {id(value) for value in root.parameters()}
    registered.update(id(value) for value in root.buffers())
    packed_types = (
        modular_sources._PUBLIC_ROUTED.PHIPackedIndex,
        modular_sources._PUBLIC_ROUTED.PHISearchOutput,
    )
    found = []
    seen = set()

    def visit(value, path):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            if identity not in registered:
                found.append((path, type(value).__name__))
            return
        if isinstance(value, packed_types):
            found.append((path, type(value).__name__))
        if isinstance(value, nn.Module):
            for name, nested in vars(value).items():
                visit(nested, f"{path}.{name}")
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return
        if isinstance(value, dict):
            for name, nested in value.items():
                visit(nested, f"{path}[{name!r}]")
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(root, "model")
    return found


def _independent_selected_attention(query, key, value, selected_blocks, block_size=8):
    batch, heads, tokens, width = query.shape
    positions = torch.arange(tokens, device=query.device)
    token_blocks = positions.div(block_size, rounding_mode="floor")
    allowed = (
        selected_blocks[..., None] == token_blocks.view(1, 1, 1, 1, tokens)
    ).any(dim=-2)[:, :, 0]
    allowed &= positions.view(1, 1, tokens) <= positions.view(1, tokens, 1)
    scores = torch.einsum("bhtd,bhsd->bhts", query.float(), key.float())
    scores = scores / math.sqrt(width)
    weights = torch.softmax(scores.masked_fill(~allowed[:, None], float("-inf")), dim=-1)
    return torch.einsum("bhts,bhsd->bhtd", weights.to(value.dtype), value)


def _independent_route_histograms(mixer, result):
    telemetry = result.telemetry
    addresses = telemetry["block_addresses"]
    postings = telemetry["postings"]
    batch = addresses.size(0)
    address_space = postings.size(1)
    loads = []
    for row in range(batch):
        loads.extend(torch.bincount(addresses[row], minlength=address_space).tolist())
    load_counts = torch.bincount(torch.tensor(loads, dtype=torch.long))
    block_histogram = torch.tensor(
        [[load, int(count)] for load, count in enumerate(load_counts.tolist()) if count],
        dtype=torch.long,
    )
    width = mixer.selected_remote_blocks
    if width == 0:
        return block_histogram, torch.empty((0, 2), dtype=torch.long)
    config = mixer.source_mixer.config
    query_addresses = modular_sources._PUBLIC_ROUTED.probe_addresses(
        result.query_route.detach(),
        mixer.source_mixer.attention.router.normalized_codebooks().detach(),
        config.phi_probes,
    )
    batch_ids = torch.arange(batch).view(batch, 1, 1, 1)
    candidates = postings[
        batch_ids.expand_as(query_addresses), query_addresses
    ]
    positions = torch.arange(result.query_route.size(1))
    remote_limits = (
        positions.div(config.block_size, rounding_mode="floor")
        - config.local_blocks
        + 1
    ).clamp_min(0)
    valid_counts = (
        (candidates >= 0)
        & (candidates < remote_limits.view(1, -1, 1, 1, 1))
    ).sum(dim=(-1, -2))
    searched = valid_counts[:, remote_limits > width].reshape(-1)
    frequencies = torch.bincount(searched)
    valid_histogram = torch.tensor(
        [[count, int(frequency)] for count, frequency in enumerate(frequencies.tolist()) if frequency],
        dtype=torch.long,
    ).reshape(-1, 2)
    return block_histogram, valid_histogram


def _independent_reset_result(
    mixer,
    x,
    reset_positions,
    carry_intervention="none",
    carry_position=None,
):
    batch, tokens, _ = x.shape
    query = mixer.q(x).reshape(batch, tokens, mixer.h, mixer.dh).transpose(1, 2)
    key = mixer.k(x).reshape(batch, tokens, mixer.h, mixer.dh).transpose(1, 2)
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    value = mixer.v(x).reshape(batch, tokens, mixer.h, mixer.dh).transpose(1, 2)
    write_gate = torch.sigmoid(mixer.bp(x)).transpose(1, 2)
    primary_gate = torch.sigmoid(mixer.ag(x)).transpose(1, 2)
    output_gate = torch.sigmoid(mixer.og(x))
    state = torch.zeros(batch, mixer.h, mixer.dh, mixer.dh, dtype=x.dtype)
    outputs = []
    boundaries = []
    resets = set(reset_positions)
    for position in range(tokens):
        if position in resets:
            boundaries.append(
                (
                    "firewall_before_reset",
                    position,
                    state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
                )
            )
            state = torch.zeros_like(state)
            boundaries.append(
                (
                    "firewall_after_reset",
                    position,
                    state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
                )
            )
        if carry_position == position and carry_intervention != "none":
            boundaries.append(
                (
                    f"carry_before_{carry_intervention}",
                    position,
                    state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
                )
            )
            if carry_intervention == "reset":
                state = torch.zeros_like(state)
            else:
                state = state.roll(1, dims=0)
            boundaries.append(
                (
                    f"carry_after_{carry_intervention}",
                    position,
                    state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
                )
            )
        current_key = key[:, :, position]
        current_value = value[:, :, position]
        projection = (current_key[..., :, None] * state).sum(-2)
        beta = write_gate[:, :, position]
        primary = primary_gate[:, :, position]
        state = primary[..., None, None] * (
            state - beta[..., None, None] * current_key[..., :, None] * projection[..., None, :]
        ) + beta[..., None, None] * current_key[..., :, None] * current_value[..., None, :]
        outputs.append((query[:, :, position, :, None] * state).sum(-2))
        if (position + 1) % mixer.chunk == 0:
            boundaries.append(
                (
                    "chunk_end_before_clamp",
                    position,
                    state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
                )
            )
            norm = torch.sqrt(state.square().sum((-2, -1), keepdim=True)) + 1e-6
            state = state * torch.clamp(100.0 / norm, max=1.0)
            boundaries.append(
                (
                    "chunk_end_after_clamp",
                    position,
                    state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
                )
            )
    outputs = torch.stack(outputs, dim=2)
    outputs = mixer.onorm(outputs)
    outputs = outputs.transpose(1, 2).reshape(batch, tokens, mixer.h * mixer.dh)
    return mixer.o(outputs * output_gate), tuple(boundaries)


def _independent_reset_delta(mixer, x, reset_positions):
    return _independent_reset_result(mixer, x, reset_positions)[0]


def _independent_chunkwise_result(query, key, value, primary_gate, write_gate, chunk):
    batch, heads, tokens, key_width = query.shape
    value_width = value.shape[-1]
    chunk_count = tokens // chunk
    mask = torch.tril(torch.ones(chunk, chunk, dtype=query.dtype))
    strict_mask = mask - torch.eye(chunk, dtype=query.dtype)
    chunk_query = query.reshape(batch, heads, chunk_count, chunk, key_width)
    chunk_key = key.reshape(batch, heads, chunk_count, chunk, key_width)
    chunk_value = value.reshape(batch, heads, chunk_count, chunk, value_width)
    chunk_write = write_gate.reshape(batch, heads, chunk_count, chunk)
    cumulative = torch.cumsum(
        torch.log(primary_gate).reshape(batch, heads, chunk_count, chunk), dim=-1
    )
    gamma = torch.exp(cumulative)
    difference = cumulative[..., :, None] - cumulative[..., None, :]
    lower = mask * torch.exp(mask * difference)
    interaction = (
        chunk_write[..., :, None]
        * (chunk_key @ chunk_key.transpose(-1, -2))
        * lower
    )
    inverse = modular_sources._TRANSFORMER_RECURRENT.tril_inv(
        -(interaction * strict_mask), chunk
    )
    update = inverse @ (chunk_write[..., None] * chunk_value)
    weighted_key = inverse @ ((chunk_write * gamma)[..., None] * chunk_key)
    state = torch.zeros(batch, heads, key_width, value_width, dtype=query.dtype)
    final_cumulative = cumulative[..., -1]
    final_gate = torch.exp(final_cumulative)
    carry = torch.exp(final_cumulative[..., None] - cumulative)
    outputs = []
    boundaries = []
    for index in range(chunk_count):
        corrected = update[:, :, index] - weighted_key[:, :, index] @ state
        local = (
            chunk_query[:, :, index]
            @ chunk_key[:, :, index].transpose(-1, -2)
        ) * lower[:, :, index]
        outputs.append(
            (gamma[:, :, index][..., None] * chunk_query[:, :, index]) @ state
            + local @ corrected
        )
        carried_key = carry[:, :, index][..., None] * chunk_key[:, :, index]
        state = (
            final_gate[:, :, index][..., None, None] * state
            + carried_key.transpose(-1, -2) @ corrected
        )
        position = (index + 1) * chunk - 1
        boundaries.append(
            (
                "chunk_end_before_clamp",
                position,
                state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
            )
        )
        norm = torch.sqrt(state.square().sum((-2, -1), keepdim=True)) + 1e-6
        state = state * torch.clamp(100.0 / norm, max=1.0)
        boundaries.append(
            (
                "chunk_end_after_clamp",
                position,
                state.detach().to(torch.float64).square().sum((-2, -1)).sqrt(),
            )
        )
    return (
        torch.stack(outputs, dim=2).reshape(batch, heads, tokens, value_width),
        tuple(boundaries),
    )


def _independent_tokenwise_boundary_norms(key, value, write_gate, primary_gate, chunk):
    batch, heads, tokens, key_width = key.shape
    state = torch.zeros(batch, heads, key_width, value.shape[-1], dtype=key.dtype)
    boundaries = []
    for position in range(tokens):
        current_key = key[:, :, position]
        current_value = value[:, :, position]
        projection = (current_key[..., :, None] * state).sum(-2)
        beta = write_gate[:, :, position]
        primary = primary_gate[:, :, position]
        state = primary[..., None, None] * (
            state
            - beta[..., None, None]
            * current_key[..., :, None]
            * projection[..., None, :]
        ) + beta[..., None, None] * current_key[..., :, None] * current_value[..., None, :]
        if (position + 1) % chunk == 0:
            boundaries.append(
                state.detach().to(torch.float64).square().sum((-2, -1)).sqrt()
            )
            norm = torch.sqrt(state.square().sum((-2, -1), keepdim=True)) + 1e-6
            state = state * torch.clamp(100.0 / norm, max=1.0)
            boundaries.append(
                state.detach().to(torch.float64).square().sum((-2, -1)).sqrt()
            )
    return tuple(boundaries)


class _ZeroMixer(nn.Module):
    def forward(self, x, **kwargs):
        return torch.zeros_like(x)


class _IdentityMixer(nn.Module):
    def forward(self, x, **kwargs):
        return x


class _ZeroFeature(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _IdentityFeature(nn.Module):
    def forward(self, x):
        return x


class _StateModule(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value)


class _TwoStateModule(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.register_buffer("first", first)
        self.register_buffer("second", second)


def test_frozen_source_hashes_match_preregistered_values():
    observed = verify_frozen_source_hashes()
    assert observed[str(modular_sources.TRANSFORMER_MODEL_PATH)] == modular_sources.TRANSFORMER_MODEL_SHA256
    assert observed[str(modular_sources.TRANSFORMER_RECURRENT_PATH)] == modular_sources.TRANSFORMER_RECURRENT_SHA256
    assert observed[str(modular_sources.PUBLIC_ROUTED_PATH)] == modular_sources.PUBLIC_ROUTED_SHA256


@pytest.mark.parametrize(
    ("role", "kind", "remote", "query"),
    (
        ("selected", "routed", 2, 126),
        ("all_eligible", "routed", 15, 126),
        ("local_only", "routed", 0, 126),
        ("dense", "dense", 0, None),
    ),
)
def test_rung_one_configuration_roles_are_exact(role, kind, remote, query):
    config = rung_one_config(role)
    assert config.vocab_size == 128
    assert config.sequence_length == 128
    assert config.block4_kind == kind
    assert config.block4_remote_blocks == remote
    assert config.query_only_position == query
    assert config.reset_positions == RUNG_ONE_RESET_POSITIONS
    assert config.carry_position == 96


def test_rung_two_configuration_is_exact():
    config = rung_two_config()
    assert config.vocab_size == 256
    assert config.sequence_length == 512
    assert config.block4_remote_blocks == 0
    assert config.query_only_position is None
    assert config.reset_positions == ()
    assert config.carry_position is None


def test_invalid_role_and_geometry_fail_closed():
    with pytest.raises(ValueError):
        rung_one_config("unknown")
    config = rung_one_config()
    with pytest.raises(ValueError):
        type(config)(**{**config.__dict__, "width": 32})


def test_exact_eight_block_architecture_and_source_surfaces():
    model = _construct()
    assert len(model.blocks) == 8
    assert tuple(block.kind for block in model.blocks) == SEQUENCE_SCHEDULE
    assert ROUTED_BLOCK_INDICES == (0, 4)
    assert RECURRENT_BLOCK_INDICES == (1, 2, 3, 5, 6, 7)
    assert isinstance(model.blocks[0].mix, PublicRoutedMixer)
    assert isinstance(model.blocks[4].mix, PublicRoutedMixer)
    assert all(isinstance(model.blocks[index].mix, ResetAwareRecurrentMixer) for index in RECURRENT_BLOCK_INDICES)
    assert all(isinstance(block.mlp, SourceFeatureMixer) for block in model.blocks)
    assert all(isinstance(block.n1, nn.RMSNorm) for block in model.blocks)
    assert all(isinstance(block.n2, nn.RMSNorm) for block in model.blocks)
    assert isinstance(model.nf, nn.RMSNorm)
    assert model.embed.weight.data_ptr() != model.head.weight.data_ptr()
    assert not any("nextlat" in name.lower() for name, _ in model.named_modules())
    assert not any("reciprocal" in name.lower() for name, _ in model.named_modules())
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
    assert not any(
        isinstance(module, modular_sources._PUBLIC_ROUTED.Block)
        for module in model.modules()
    )


def test_projection_biases_feature_shape_and_normalization_epsilon_are_exact():
    model = _construct()
    for block in model.blocks:
        assert block.n1.eps is None
        assert block.n2.eps is None
        assert block.mlp.w1.in_features == 64
        assert block.mlp.w1.out_features == 256
        assert block.mlp.w2.out_features == 256
        assert block.mlp.w3.in_features == 256
        assert block.mlp.w3.out_features == 64
        assert block.mlp.w1.bias is None
        assert block.mlp.w2.bias is None
        assert block.mlp.w3.bias is None
    for index in RECURRENT_BLOCK_INDICES:
        mixer = model.blocks[index].mix
        assert mixer.q.bias is None
        assert mixer.k.bias is None
        assert mixer.v.bias is None
        assert mixer.bp.bias is not None
        assert mixer.ag.bias is not None
        assert mixer.og.bias is not None
        assert mixer.o.bias is None
        assert mixer.onorm.eps is None
    for index in ROUTED_BLOCK_INDICES:
        attention = model.blocks[index].mix.source_mixer.attention
        assert attention.qkv.bias is None
        assert attention.out.bias is None
    assert model.head.bias is None
    assert model.nf.eps is None


def test_ordinary_forward_returns_only_finite_float32_logits():
    model = _construct()
    input_ids = torch.randint(0, 128, (2, 128), dtype=torch.long)
    logits = model(input_ids)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 128, 128)
    assert logits.dtype == torch.float32
    assert logits.device.type == "cpu"
    assert bool(torch.isfinite(logits).all())


def test_host_adds_sequence_delta_once_after_sequence_normalization():
    model = _construct()
    for block in model.blocks:
        block.mix = _ZeroMixer()
        block.mlp = _ZeroFeature()
    model.blocks[0].mix = _IdentityMixer()
    input_ids = torch.randint(0, 128, (1, 128), dtype=torch.long)
    embedded = model.embed(input_ids)
    expected_hidden = embedded + model.blocks[0].n1(embedded)
    expected = model.head(model.nf(expected_hidden))
    assert torch.equal(model(input_ids), expected)


def test_host_adds_feature_delta_once_after_feature_normalization():
    model = _construct()
    for block in model.blocks:
        block.mix = _ZeroMixer()
        block.mlp = _ZeroFeature()
    model.blocks[0].mlp = _IdentityFeature()
    input_ids = torch.randint(0, 128, (1, 128), dtype=torch.long)
    embedded = model.embed(input_ids)
    expected_hidden = embedded + model.blocks[0].n2(embedded)
    expected = model.head(model.nf(expected_hidden))
    assert torch.equal(model(input_ids), expected)


def test_public_adapter_uses_each_registered_lower_level_boundary_once(monkeypatch):
    mixer = PublicRoutedMixer(2)
    attention = mixer.source_mixer.attention
    counts = {
        "qkv": 0,
        "rope": 0,
        "query_features": 0,
        "key_features": 0,
        "build": 0,
        "search": 0,
        "block_ids": 0,
        "selected_attention": 0,
        "out": 0,
        "router_loss": 0,
    }
    attention.qkv.register_forward_hook(lambda *args: counts.__setitem__("qkv", counts["qkv"] + 1))
    attention.out.register_forward_hook(lambda *args: counts.__setitem__("out", counts["out"] + 1))
    original_rope = attention._rope
    original_query_features = attention.router.query_features
    original_key_features = attention.router.key_features
    original_loss = attention._router_loss
    original_build = modular_sources._PUBLIC_ROUTED.build_packed_index
    original_search = modular_sources._PUBLIC_ROUTED.search_packed_index
    original_block_ids = modular_sources._PUBLIC_ROUTED._block_ids_for_attention
    original_selected = modular_sources._sparse_selected_attention
    monkeypatch.setattr(mixer.source_mixer, "forward", MethodType(lambda self, *a, **k: pytest.fail("source mixer forward called"), mixer.source_mixer))
    monkeypatch.setattr(attention, "forward", MethodType(lambda self, *a, **k: pytest.fail("source attention forward called"), attention))

    def rope(value):
        counts["rope"] += 1
        return original_rope(value)

    def query_features(value):
        counts["query_features"] += 1
        return original_query_features(value)

    def key_features(value):
        counts["key_features"] += 1
        return original_key_features(value)

    def router_loss(*args):
        counts["router_loss"] += 1
        return original_loss(*args)

    def build(*args, **kwargs):
        counts["build"] += 1
        return original_build(*args, **kwargs)

    def search(*args, **kwargs):
        counts["search"] += 1
        return original_search(*args, **kwargs)

    def block_ids(*args, **kwargs):
        counts["block_ids"] += 1
        return original_block_ids(*args, **kwargs)

    def selected(*args, **kwargs):
        counts["selected_attention"] += 1
        return original_selected(*args, **kwargs)

    monkeypatch.setattr(attention, "_rope", rope)
    monkeypatch.setattr(attention.router, "query_features", query_features)
    monkeypatch.setattr(attention.router, "key_features", key_features)
    monkeypatch.setattr(attention, "_router_loss", router_loss)
    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "build_packed_index", build)
    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "search_packed_index", search)
    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "_block_ids_for_attention", block_ids)
    monkeypatch.setattr(modular_sources, "_sparse_selected_attention", selected)
    result = mixer(
        torch.randn(1, 128, 64),
        return_aux=True,
        request_router_loss=True,
        query_only_position=126,
    )
    assert result.router_loss is not None
    assert counts == {
        "qkv": 1,
        "rope": 2,
        "query_features": 1,
        "key_features": 1,
        "build": 1,
        "search": 1,
        "block_ids": 1,
        "selected_attention": 1,
        "out": 1,
        "router_loss": 1,
    }
    assert result.events == (
        "qkv",
        "rope_query",
        "rope_key",
        "query_features",
        "key_features",
        "build_index",
        "search_index",
        "clone_raw_remote",
        "query_only_mask",
        "block_ids",
        "selected_attention",
        "output_projection",
        "router_loss",
    )


def test_public_adapter_orders_force_mask_override_and_skips_unrequested_loss(monkeypatch):
    mixer = PublicRoutedMixer(2)
    attention = mixer.source_mixer.attention
    x = torch.randn(1, 128, 64)
    forced = torch.full((1, 128), -1, dtype=torch.long)
    forced[:, 126] = 1
    override = torch.full((1, 128, 1, 2), -1, dtype=torch.long)
    override[:, 126, 0] = torch.tensor([0, 14])
    calls = {"router_loss": 0}

    def unexpected_loss(*args):
        calls["router_loss"] += 1
        pytest.fail("unrequested router loss was computed")

    monkeypatch.setattr(attention, "_router_loss", unexpected_loss)
    result = mixer(
        x,
        return_aux=True,
        forced_blocks=forced,
        route_override=override,
        query_only_position=126,
    )
    assert calls == {"router_loss": 0}
    assert result.router_loss is None
    assert result.events == (
        "qkv",
        "rope_query",
        "rope_key",
        "query_features",
        "key_features",
        "build_index",
        "search_index",
        "clone_raw_remote",
        "force_remote",
        "query_only_mask",
        "route_override",
        "block_ids",
        "selected_attention",
        "output_projection",
    )
    assert torch.equal(result.telemetry["effective_remote"], override)


def test_explicit_remote_underfill_is_preserved_without_fallback(monkeypatch):
    mixer = PublicRoutedMixer(2)
    original_search = modular_sources._PUBLIC_ROUTED.search_packed_index

    def underfilled_search(*args, **kwargs):
        result = original_search(*args, **kwargs)
        selected = result.selected_blocks.clone()
        selected[:, 126, 0] = torch.tensor([3, -1], device=selected.device)
        return modular_sources._PUBLIC_ROUTED.PHISearchOutput(
            selected_blocks=selected,
            addresses_probed=result.addresses_probed,
            postings_read=result.postings_read,
            posting_slots_materialized=result.posting_slots_materialized,
            candidate_blocks=result.candidate_blocks,
            search_rows=result.search_rows,
            bypass_rows=result.bypass_rows,
            workspace_bytes=result.workspace_bytes,
        )

    monkeypatch.setattr(
        modular_sources._PUBLIC_ROUTED,
        "search_packed_index",
        underfilled_search,
    )
    result = mixer(
        torch.randn(1, 128, 64),
        return_aux=True,
        query_only_position=126,
    )
    expected_remote = torch.tensor([3, -1])
    expected_selected = torch.tensor([3, -1, 15])
    assert torch.equal(result.telemetry["raw_remote"][0, 126, 0], expected_remote)
    assert torch.equal(result.telemetry["effective_remote"][0, 126, 0], expected_remote)
    assert torch.equal(result.telemetry["selected_blocks"][0, 126, 0], expected_selected)


def test_local_only_adapter_still_builds_and_searches(monkeypatch):
    mixer = PublicRoutedMixer(0)
    calls = {"build": 0, "search": 0}
    original_build = modular_sources._PUBLIC_ROUTED.build_packed_index
    original_search = modular_sources._PUBLIC_ROUTED.search_packed_index

    def build(*args, **kwargs):
        calls["build"] += 1
        return original_build(*args, **kwargs)

    def search(*args, **kwargs):
        calls["search"] += 1
        return original_search(*args, **kwargs)

    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "build_packed_index", build)
    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "search_packed_index", search)
    result = mixer(torch.randn(1, 128, 64), return_aux=True)
    assert calls == {"build": 1, "search": 1}
    assert result.telemetry["raw_remote"].shape == (1, 128, 1, 0)


def test_detailed_route_telemetry_has_exact_detached_histograms():
    torch.manual_seed(1701)
    mixer = PublicRoutedMixer(2)
    result = mixer(
        torch.randn(2, 128, 64),
        return_aux=True,
        return_detail=True,
        query_only_position=126,
    )
    expected_loads, expected_valid = _independent_route_histograms(mixer, result)
    observed_loads = result.telemetry["block_load_histogram"]
    observed_valid = result.telemetry["valid_posting_histogram"]
    assert torch.equal(observed_loads, expected_loads)
    assert torch.equal(observed_valid, expected_valid)
    assert observed_loads.dtype == torch.long and observed_loads.shape[1:] == (2,)
    assert observed_valid.dtype == torch.long and observed_valid.shape[1:] == (2,)
    assert observed_loads.requires_grad is False
    assert observed_valid.requires_grad is False
    assert int(observed_loads[:, 1].sum()) == 32
    assert int((observed_loads[:, 0] * observed_loads[:, 1]).sum()) == 32
    assert int(observed_loads[:, 0].max()) == result.telemetry["max_bucket_load"]
    assert int(((observed_loads[:, 0] - 64).clamp_min(0) * observed_loads[:, 1]).sum()) == result.telemetry["overflow_count"]
    assert int(observed_valid[:, 1].sum()) == result.telemetry["search_rows"]
    assert int((observed_valid[:, 0] * observed_valid[:, 1]).sum()) == result.telemetry["postings_read"]
    assert result.telemetry["postings_read"] == result.telemetry["candidate_blocks"]


def test_detailed_route_audit_adds_one_detached_probe_without_changing_output_or_gradient(monkeypatch):
    torch.manual_seed(1702)
    mixer = PublicRoutedMixer(2)
    original_probe = modular_sources._PUBLIC_ROUTED.probe_addresses
    calls = 0
    probe_inputs = []

    def probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        probe_inputs.append(
            (
                args[0].requires_grad,
                args[1].requires_grad,
                args[0].dtype,
                bool(torch.isfinite(args[0]).all()),
                kwargs.get("probes", args[2] if len(args) > 2 else None),
            )
        )
        return original_probe(*args, **kwargs)

    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "probe_addresses", probe)
    compact_input = torch.randn(2, 128, 64, requires_grad=True)
    detailed_input = compact_input.detach().clone().requires_grad_(True)
    compact = mixer(
        compact_input,
        return_aux=True,
        request_router_loss=True,
        query_only_position=126,
    )
    compact_calls = calls
    detailed = mixer(
        detailed_input,
        return_aux=True,
        return_detail=True,
        request_router_loss=True,
        query_only_position=126,
    )
    compact_gradient = torch.autograd.grad(
        compact.delta.square().mean() + compact.router_loss,
        compact_input,
    )[0]
    detailed_gradient = torch.autograd.grad(
        detailed.delta.square().mean() + detailed.router_loss,
        detailed_input,
    )[0]
    assert compact_calls == 1
    assert calls == compact_calls + 2
    assert probe_inputs[-1] == (False, False, torch.float32, True, 4)
    assert torch.equal(compact.delta, detailed.delta)
    assert torch.equal(compact.telemetry["raw_remote"], detailed.telemetry["raw_remote"])
    assert torch.equal(compact.telemetry["effective_remote"], detailed.telemetry["effective_remote"])
    assert torch.allclose(compact_gradient, detailed_gradient, atol=2e-11, rtol=0)
    assert "block_load_histogram" not in compact.telemetry
    assert "valid_posting_histogram" not in compact.telemetry
    assert detailed.events[-1] == "evidence_probe"


def test_width_zero_detailed_route_has_empty_valid_posting_histogram():
    torch.manual_seed(1703)
    mixer = PublicRoutedMixer(0)
    result = mixer(torch.randn(2, 128, 64), return_aux=True, return_detail=True)
    expected_loads, expected_valid = _independent_route_histograms(mixer, result)
    assert torch.equal(result.telemetry["block_load_histogram"], expected_loads)
    assert torch.equal(result.telemetry["valid_posting_histogram"], expected_valid)
    assert result.telemetry["valid_posting_histogram"].shape == (0, 2)
    assert result.telemetry["search_rows"] == 0
    assert result.telemetry["addresses_probed"] == 0
    assert result.telemetry["postings_read"] == 0


def test_width_fifteen_detailed_route_has_empty_all_bypass_histogram():
    torch.manual_seed(1705)
    mixer = PublicRoutedMixer(15)
    result = mixer(
        torch.randn(2, 128, 64),
        return_aux=True,
        return_detail=True,
        query_only_position=126,
    )
    expected_loads, expected_valid = _independent_route_histograms(mixer, result)
    assert torch.equal(result.telemetry["block_load_histogram"], expected_loads)
    assert torch.equal(result.telemetry["valid_posting_histogram"], expected_valid)
    assert result.telemetry["valid_posting_histogram"].shape == (0, 2)
    assert result.telemetry["search_rows"] == 0
    assert result.telemetry["bypass_rows"] == 256
    assert result.telemetry["addresses_probed"] == 0
    assert result.telemetry["postings_read"] == 0
    assert result.telemetry["candidate_blocks"] == 0
    assert result.telemetry["raw_remote"][:, 126, 0].tolist() == [list(range(15)), list(range(15))]
    assert result.telemetry["effective_remote"][:, 125, 0].tolist() == [[-1] * 15, [-1] * 15]
    assert result.telemetry["effective_remote"][:, 126, 0].tolist() == [list(range(15)), list(range(15))]
    assert result.telemetry["effective_remote"][:, 127, 0].tolist() == [[-1] * 15, [-1] * 15]


def test_query_firewall_preserves_distinct_early_and_query_raw_routes(monkeypatch):
    torch.manual_seed(1704)
    mixer = PublicRoutedMixer(2)
    original_search = modular_sources._PUBLIC_ROUTED.search_packed_index

    def distinct_search(*args, **kwargs):
        result = original_search(*args, **kwargs)
        selected = result.selected_blocks.clone()
        selected[:, 16, 0] = torch.tensor([0, 1], device=selected.device)
        selected[:, 126, 0] = torch.tensor([7, 8], device=selected.device)
        return modular_sources._PUBLIC_ROUTED.PHISearchOutput(
            selected_blocks=selected,
            addresses_probed=result.addresses_probed,
            postings_read=result.postings_read,
            posting_slots_materialized=result.posting_slots_materialized,
            candidate_blocks=result.candidate_blocks,
            search_rows=result.search_rows,
            bypass_rows=result.bypass_rows,
            workspace_bytes=result.workspace_bytes,
        )

    monkeypatch.setattr(modular_sources._PUBLIC_ROUTED, "search_packed_index", distinct_search)
    result = mixer(
        torch.randn(2, 128, 64),
        return_aux=True,
        return_detail=True,
        query_only_position=126,
    )
    assert result.telemetry["raw_remote"][:, 16, 0].tolist() == [[0, 1], [0, 1]]
    assert result.telemetry["raw_remote"][:, 126, 0].tolist() == [[7, 8], [7, 8]]
    assert result.telemetry["effective_remote"][:, 16, 0].tolist() == [[-1, -1], [-1, -1]]
    assert result.telemetry["effective_remote"][:, 126, 0].tolist() == [[7, 8], [7, 8]]


def test_selected_attention_matches_independent_dense_mask_oracle():
    mixer = PublicRoutedMixer(2)
    result = mixer(
        torch.randn(2, 128, 64),
        return_aux=True,
        return_detail=True,
        query_only_position=126,
    )
    query = result.telemetry["query"]
    key = result.telemetry["key"]
    value = result.telemetry["value"]
    selected = result.telemetry["selected_blocks"]
    attended = _independent_selected_attention(query, key, value, selected)
    expected = mixer.source_mixer.attention.out(
        attended.transpose(1, 2).reshape(2, 128, 64)
    )
    assert torch.allclose(result.delta, expected, atol=1e-5, rtol=0)


def _selected_attention_fixture(batch, tokens):
    query = torch.randn(batch, 4, tokens, 16, requires_grad=True)
    key = torch.randn(batch, 4, tokens, 16, requires_grad=True)
    value = torch.randn(batch, 4, tokens, 16, requires_grad=True)
    positions = torch.arange(tokens)
    local = positions.div(8, rounding_mode="floor")
    selected = torch.full((batch, tokens, 1, 3), -1, dtype=torch.long)
    selected[:, :, 0, 2] = local
    selected[:, :, 0, 0] = torch.where(local >= 2, local - 2, -1)
    return query, key, value, selected


def test_sparse_selected_attention_matches_public_forward_and_gradients_with_underfill():
    torch.manual_seed(90210)
    query, key, value, selected = _selected_attention_fixture(2, 37)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value))
    sparse_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value))
    reference = modular_sources._PUBLIC_ROUTED.selected_attention(
        *reference_inputs,
        selected,
        block_size=8,
    )
    sparse = modular_sources._sparse_selected_attention(
        *sparse_inputs,
        selected,
        block_size=8,
    )
    upstream = torch.randn_like(reference)
    reference_gradients = torch.autograd.grad((reference * upstream).sum(), reference_inputs)
    sparse_gradients = torch.autograd.grad((sparse * upstream).sum(), sparse_inputs)
    assert torch.allclose(sparse, reference, atol=1e-6, rtol=1e-6)
    for observed, expected in zip(sparse_gradients, reference_gradients):
        assert torch.allclose(observed, expected, atol=2e-6, rtol=1e-6)
    assert selected[:, :16, 0, :2].eq(-1).all()


@pytest.mark.parametrize(("batch", "tokens"), ((16, 128), (8, 512)))
def test_sparse_selected_attention_cpu_benchmark_is_faster_than_public_loop(batch, tokens):
    torch.manual_seed(90211 + tokens)
    query, key, value, selected = _selected_attention_fixture(batch, tokens)
    detached = tuple(tensor.detach() for tensor in (query, key, value))
    modular_sources._sparse_selected_attention(*detached, selected, block_size=8)
    start = time.perf_counter()
    modular_sources._sparse_selected_attention(*detached, selected, block_size=8)
    sparse_seconds = time.perf_counter() - start
    start = time.perf_counter()
    modular_sources._PUBLIC_ROUTED.selected_attention(*detached, selected, block_size=8)
    public_seconds = time.perf_counter() - start
    assert sparse_seconds < public_seconds * 0.75


def test_router_loss_and_supervised_route_features_remain_graph_attached():
    mixer = PublicRoutedMixer(2)
    x = torch.randn(2, 128, 64, requires_grad=True)
    result = mixer(
        x,
        return_aux=True,
        request_router_loss=True,
        query_only_position=126,
    )
    assert result.query_route.requires_grad
    assert result.key_route.requires_grad
    block_keys = result.key_route[:, :120].reshape(2, 15, 8, 16).mean(dim=2)
    logits = torch.einsum("bd,bnd->bn", result.query_route[:, 126, 0], block_keys)
    loss = result.router_loss + logits.square().mean()
    loss.backward()
    router = mixer.source_mixer.attention.router
    assert router.query_projection.weight.grad is not None
    assert bool((router.query_projection.weight.grad != 0).any())
    assert router.key_projection.weight.grad is not None
    assert bool((router.key_projection.weight.grad != 0).any())


def test_query_only_route_preserves_raw_search_and_local_path():
    mixer = PublicRoutedMixer(2)
    result = mixer(
        torch.randn(2, 128, 64),
        return_aux=True,
        query_only_position=126,
    )
    raw = result.telemetry["raw_remote"]
    effective = result.telemetry["effective_remote"]
    selected = result.telemetry["selected_blocks"]
    positions = torch.arange(128)
    assert bool((effective[:, positions != 126] == -1).all())
    valid = effective[:, 126][effective[:, 126] >= 0]
    assert bool((valid <= 14).all())
    assert selected.shape[-1] == 3
    assert bool((selected[:, positions != 126, :, -1] >= 0).all())
    assert bool((raw[:,:126] != effective[:,:126]).any())


def test_forced_route_clones_raw_before_force_and_preserves_width():
    mixer = PublicRoutedMixer(2)
    x = torch.randn(1, 128, 64)
    intact = mixer(x, return_aux=True, query_only_position=126)
    raw_query = intact.telemetry["raw_remote"][0, 126, 0]
    missing = next(index for index in range(15) if not bool((raw_query == index).any()))
    forced = torch.full((1, 128), -1, dtype=torch.long)
    forced[:, 126] = missing
    forced_result = mixer(
        x,
        return_aux=True,
        forced_blocks=forced,
        query_only_position=126,
    )
    assert torch.equal(forced_result.telemetry["raw_remote"], intact.telemetry["raw_remote"])
    assert not bool((forced_result.telemetry["raw_remote"][0, 126, 0] == missing).any())
    assert bool((forced_result.telemetry["effective_remote"][0, 126, 0] == missing).any())
    assert forced_result.telemetry["effective_remote"].shape[-1] == 2


def test_full_route_override_is_exact_and_fails_closed_on_duplicates():
    mixer = PublicRoutedMixer(2)
    x = torch.randn(1, 128, 64)
    override = torch.full((1, 128, 1, 2), -1, dtype=torch.long)
    override[:, 126, 0] = torch.tensor([0, 14])
    result = mixer(
        x,
        return_aux=True,
        route_override=override,
        query_only_position=126,
    )
    assert torch.equal(result.telemetry["effective_remote"], override)
    duplicate = override.clone()
    duplicate[:, 126, 0] = 3
    with pytest.raises(ValueError):
        mixer(x, route_override=duplicate, query_only_position=126)
    outside_query = override.clone()
    outside_query[:, 120, 0] = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        mixer(x, route_override=outside_query, query_only_position=126)


def test_a_b_a_calls_are_exact_and_leave_no_transient_tensor_attribute():
    model = _construct()
    model.eval()
    first_input = torch.randint(0, 128, (1, 128), dtype=torch.long)
    second_input = torch.randint(0, 128, (1, 128), dtype=torch.long)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.inference_mode():
        first = model(first_input, return_aux=True)
        model(second_input, return_aux=True)
        repeated = model(first_input, return_aux=True)
    assert torch.equal(first.logits, repeated.logits)
    for index in ROUTED_BLOCK_INDICES:
        assert torch.equal(
            first.blocks[index].mixer_output.telemetry["raw_remote"],
            repeated.blocks[index].mixer_output.telemetry["raw_remote"],
        )
        assert torch.equal(
            first.blocks[index].mixer_output.telemetry["effective_remote"],
            repeated.blocks[index].mixer_output.telemetry["effective_remote"],
        )
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert _transient_graph_objects(model) == []


def test_expensive_route_and_recurrent_detail_are_explicit_opt_in():
    model = _construct()
    input_ids = torch.randint(0, 128, (1, 128), dtype=torch.long)
    compact = model(input_ids, return_aux=True)
    assert compact.blocks[1].mixer_output is None
    assert "query" not in compact.blocks[4].mixer_output.telemetry
    detailed = model(
        input_ids,
        return_aux=True,
        recurrent_telemetry=True,
        route_detail=True,
    )
    assert detailed.blocks[1].mixer_output is not None
    assert "query" in detailed.blocks[4].mixer_output.telemetry


@pytest.mark.parametrize("tokens", (128, 512))
def test_empty_reset_forward_and_gradient_parity_with_frozen_chunkwise(tokens):
    torch.manual_seed(7)
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    chunk_input = torch.randn(1, tokens, 64, requires_grad=True)
    reset_input = chunk_input.detach().clone().requires_grad_(True)
    upstream = torch.randn(1, tokens, 64) / tokens
    chunk_output = mixer(chunk_input)
    chunk_gradients = torch.autograd.grad(
        (chunk_output * upstream).sum(),
        (chunk_input, *tuple(mixer.parameters())),
    )
    reset_output = mixer(reset_input, force_reset_aware=True)
    reset_gradients = torch.autograd.grad(
        (reset_output * upstream).sum(),
        (reset_input, *tuple(mixer.parameters())),
    )
    assert torch.allclose(chunk_output, reset_output, atol=1e-5, rtol=0)
    for chunk_gradient, reset_gradient in zip(chunk_gradients, reset_gradients):
        assert torch.allclose(chunk_gradient, reset_gradient, atol=1e-4, rtol=0)


def test_reset_aware_path_matches_independent_token_reference_and_clamp_positions():
    torch.manual_seed(13)
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    x = torch.randn(2, 128, 64, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_(True)
    result = mixer(x, reset_positions=RUNG_ONE_RESET_POSITIONS, return_aux=True)
    reference = _independent_reset_delta(mixer, reference_input, RUNG_ONE_RESET_POSITIONS)
    assert torch.allclose(result.delta, reference, atol=1e-5, rtol=0)
    upstream = torch.randn_like(reference) / reference.numel()
    observed_gradients = torch.autograd.grad(
        (result.delta * upstream).sum(),
        (x, *tuple(mixer.parameters())),
    )
    reference_gradients = torch.autograd.grad(
        (reference * upstream).sum(),
        (reference_input, *tuple(mixer.parameters())),
    )
    for observed, expected in zip(observed_gradients, reference_gradients):
        assert torch.allclose(observed, expected, atol=1e-4, rtol=0)
    reset_before = [boundary.position for boundary in result.boundaries if boundary.kind == "firewall_before_reset"]
    reset_after = [boundary.position for boundary in result.boundaries if boundary.kind == "firewall_after_reset"]
    clamp_before = [boundary.position for boundary in result.boundaries if boundary.kind == "chunk_end_before_clamp"]
    clamp_after = [boundary.position for boundary in result.boundaries if boundary.kind == "chunk_end_after_clamp"]
    assert tuple(reset_before) == RUNG_ONE_RESET_POSITIONS
    assert tuple(reset_after) == RUNG_ONE_RESET_POSITIONS
    assert clamp_before == [31, 63, 95, 127]
    assert clamp_after == [31, 63, 95, 127]


def test_active_chunk_clamp_emits_output_before_exact_state_clamp():
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    query = torch.zeros(2, 4, 32, 16)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    amplitudes = torch.arange(8, dtype=torch.float32).reshape(2, 4) * 10 + 200
    query[..., 0] = 1
    key[..., 0] = 1
    value[..., 0] = amplitudes.unsqueeze(-1)
    write_gate = torch.ones(2, 4, 32)
    primary_gate = torch.ones_like(write_gate)
    outputs, boundaries = mixer._reset_aware_outputs(
        query,
        key,
        value,
        write_gate,
        primary_gate,
        (),
        "none",
        None,
        True,
    )
    before = next(
        boundary
        for boundary in boundaries
        if boundary.kind == "chunk_end_before_clamp" and boundary.position == 31
    )
    after = next(
        boundary
        for boundary in boundaries
        if boundary.kind == "chunk_end_after_clamp" and boundary.position == 31
    )
    expected_before = torch.zeros(2, 4, 16, 16)
    expected_before[..., 0, 0] = amplitudes
    expected_norm = torch.sqrt(
        expected_before.square().sum((-2, -1), keepdim=True)
    ) + 1e-6
    expected_after = expected_before * torch.clamp(100.0 / expected_norm, max=1.0)
    assert bool((before.norms > 100).all())
    assert torch.equal(outputs[:, :, 31, 0], amplitudes)
    assert torch.equal(before.norms, modular_sources._state_norms(expected_before))
    assert torch.equal(after.norms, modular_sources._state_norms(expected_after))


def test_carry_reset_and_shuffle_are_call_local_at_position_96():
    torch.manual_seed(19)
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    x = torch.randn(3, 128, 64)
    reset = mixer(
        x,
        reset_positions=RUNG_ONE_RESET_POSITIONS,
        carry_intervention="reset",
        carry_position=96,
        return_aux=True,
    )
    shuffle = mixer(
        x,
        reset_positions=RUNG_ONE_RESET_POSITIONS,
        carry_intervention="shuffle",
        carry_position=96,
        return_aux=True,
    )
    reset_reference, reset_boundaries = _independent_reset_result(
        mixer,
        x,
        RUNG_ONE_RESET_POSITIONS,
        "reset",
        96,
    )
    shuffle_reference, shuffle_boundaries = _independent_reset_result(
        mixer,
        x,
        RUNG_ONE_RESET_POSITIONS,
        "shuffle",
        96,
    )
    assert torch.allclose(reset.delta, reset_reference, atol=1e-5, rtol=0)
    assert torch.allclose(shuffle.delta, shuffle_reference, atol=1e-5, rtol=0)
    for observed, expected in zip(reset.boundaries, reset_boundaries):
        assert (observed.kind, observed.position) == expected[:2]
        assert torch.equal(observed.norms, expected[2])
    for observed, expected in zip(shuffle.boundaries, shuffle_boundaries):
        assert (observed.kind, observed.position) == expected[:2]
        assert torch.equal(observed.norms, expected[2])
    reset_before = next(boundary for boundary in reset.boundaries if boundary.kind == "carry_before_reset")
    reset_after = next(boundary for boundary in reset.boundaries if boundary.kind == "carry_after_reset")
    shuffle_before = next(boundary for boundary in shuffle.boundaries if boundary.kind == "carry_before_shuffle")
    shuffle_after = next(boundary for boundary in shuffle.boundaries if boundary.kind == "carry_after_shuffle")
    assert reset_before.position == 96
    assert reset_after.position == 96
    assert bool((reset_before.norms > 0).all())
    assert torch.equal(reset_after.norms, torch.zeros_like(reset_before.norms))
    assert not torch.equal(shuffle_before.norms[0], shuffle_before.norms[1])
    assert torch.equal(shuffle_after.norms, shuffle_before.norms.roll(1, dims=0))
    assert 96 not in RUNG_ONE_RESET_POSITIONS


def test_rung_two_ordinary_path_dispatches_frozen_chunkwise(monkeypatch):
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    calls = {"chunkwise": 0}
    original = modular_sources._TRANSFORMER_RECURRENT.chunkwise_gated

    def chunkwise(*args, **kwargs):
        calls["chunkwise"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(modular_sources._TRANSFORMER_RECURRENT, "chunkwise_gated", chunkwise)
    monkeypatch.setattr(mixer, "_reset_aware_outputs", lambda *args, **kwargs: pytest.fail("reset-aware output path entered"))
    result = mixer(torch.randn(1, 512, 64))
    assert calls["chunkwise"] == 1
    assert result.shape == (1, 512, 64)


def test_rung_two_telemetry_keeps_pinned_output_and_uses_parity_checked_shadow(
    monkeypatch,
):
    torch.manual_seed(29)
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    pinned = modular_sources._TRANSFORMER_RECURRENT.chunkwise_gated
    shadow = mixer._chunkwise_outputs_and_boundaries
    calls = {"pinned": 0, "shadow": 0}

    def counted_pinned(*args, **kwargs):
        calls["pinned"] += 1
        return pinned(*args, **kwargs)

    def shifted_shadow(*args, **kwargs):
        calls["shadow"] += 1
        outputs, boundaries = shadow(*args, **kwargs)
        return outputs + 5e-6, boundaries

    monkeypatch.setattr(
        modular_sources._TRANSFORMER_RECURRENT,
        "chunkwise_gated",
        counted_pinned,
    )
    monkeypatch.setattr(mixer, "_chunkwise_outputs_and_boundaries", shifted_shadow)
    x = torch.randn(1, 512, 64, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_(True)
    result = mixer(x, return_aux=True)
    assert calls == {"pinned": 1, "shadow": 1}
    query, key, value, write_gate, primary_gate, output_gate = mixer._project(
        reference_input
    )
    raw_reference, expected_boundaries = _independent_chunkwise_result(
        query.float(),
        key.float(),
        value.float(),
        primary_gate.float(),
        write_gate.float(),
        mixer.chunk,
    )
    frozen_reference = pinned(
        query.float(),
        key.float(),
        value.float(),
        primary_gate.float(),
        write_gate.float(),
        mixer.chunk,
    )
    assert torch.equal(raw_reference, frozen_reference)
    normalized = mixer.onorm(raw_reference)
    normalized = normalized.transpose(1, 2).reshape(1, 512, 64)
    delta_reference = mixer.o(normalized * output_gate)
    assert torch.equal(result.delta, delta_reference)
    assert len(result.boundaries) == len(expected_boundaries) == 32
    for observed, expected in zip(result.boundaries, expected_boundaries):
        assert (observed.kind, observed.position) == expected[:2]
        assert torch.equal(observed.norms, expected[2])
    tokenwise_boundaries = _independent_tokenwise_boundary_norms(
        key.detach(),
        value.detach(),
        write_gate.detach(),
        primary_gate.detach(),
        mixer.chunk,
    )
    assert any(
        not torch.equal(observed.norms, replayed)
        for observed, replayed in zip(result.boundaries, tokenwise_boundaries)
    )
    upstream = torch.randn_like(result.delta) / result.delta.numel()
    observed_gradients = torch.autograd.grad(
        (result.delta * upstream).sum(),
        (x, *tuple(mixer.parameters())),
    )
    reference_gradients = torch.autograd.grad(
        (delta_reference * upstream).sum(),
        (reference_input, *tuple(mixer.parameters())),
    )
    for observed, expected in zip(observed_gradients, reference_gradients):
        assert torch.equal(observed, expected)
    clamp_positions = [
        boundary.position
        for boundary in result.boundaries
        if boundary.kind == "chunk_end_after_clamp"
    ]
    assert clamp_positions == list(range(31, 512, 32))


def test_rung_two_telemetry_rejects_shadow_divergence(monkeypatch):
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    shadow = mixer._chunkwise_outputs_and_boundaries

    def divergent_shadow(*args, **kwargs):
        outputs, boundaries = shadow(*args, **kwargs)
        return outputs + 1e-3, boundaries

    monkeypatch.setattr(mixer, "_chunkwise_outputs_and_boundaries", divergent_shadow)
    with pytest.raises(FloatingPointError, match="telemetry shadow diverged"):
        mixer(torch.randn(1, 512, 64), return_aux=True)


def test_recurrent_auxiliary_abi_exposes_no_state_tensor_escape():
    mixer = ResetAwareRecurrentMixer(64, 4, 16, 32)
    signature = inspect.signature(mixer.forward)
    assert "return_state_tensors" not in signature.parameters
    result = mixer(
        torch.randn(2, 128, 64),
        reset_positions=RUNG_ONE_RESET_POSITIONS,
        return_aux=True,
    )
    assert [field.name for field in fields(result.boundaries[0])] == [
        "kind",
        "position",
        "norms",
    ]
    assert all(boundary.norms.shape == (2, 4) for boundary in result.boundaries)
    assert not any(hasattr(boundary, "state") for boundary in result.boundaries)


def test_recurrent_knockout_runs_complete_mixers_and_exposes_only_zero_delta():
    model = _construct()
    input_ids = torch.randint(0, 128, (2, 128), dtype=torch.long)
    normalized_inputs = {}
    handles = [
        model.blocks[index].n1.register_forward_hook(
            lambda module, arguments, output, index=index: normalized_inputs.__setitem__(
                index, output.detach().clone()
            )
        )
        for index in RECURRENT_BLOCK_INDICES
    ]
    try:
        result = model(
            input_ids,
            return_aux=True,
            recurrent_telemetry=True,
            recurrent_knockout=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    for index in RECURRENT_BLOCK_INDICES:
        execution = result.blocks[index]
        intact = model.blocks[index].mix(
            normalized_inputs[index],
            reset_positions=RUNG_ONE_RESET_POSITIONS,
            return_aux=True,
        )
        assert torch.equal(execution.computed_sequence_delta, intact.delta)
        assert torch.equal(execution.mixer_output.primary_gate, intact.primary_gate)
        assert torch.equal(execution.mixer_output.write_gate, intact.write_gate)
        assert torch.equal(execution.mixer_output.output_gate, intact.output_gate)
        assert len(execution.mixer_output.boundaries) == len(intact.boundaries)
        for observed, expected in zip(
            execution.mixer_output.boundaries,
            intact.boundaries,
        ):
            assert observed.kind == expected.kind
            assert observed.position == expected.position
            assert torch.equal(observed.norms, expected.norms)
        assert bool(torch.isfinite(execution.computed_sequence_delta).all())
        assert bool((execution.computed_sequence_delta != 0).any())
        assert torch.equal(execution.exposed_sequence_delta, torch.zeros_like(execution.exposed_sequence_delta))
        assert bool(torch.isfinite(execution.feature_delta).all())
        assert execution.mixer_output.primary_gate.numel() == 2 * 4 * 128
        assert not hasattr(model.blocks[index].mix, "g_last")


def test_block4_knockout_changes_only_exposed_sequence_delta_and_keeps_feature_active():
    model = _construct()
    input_ids = torch.randint(0, 128, (1, 128), dtype=torch.long)
    intact = model(input_ids, return_aux=True)
    knockout = model(input_ids, return_aux=True, block4_routed_knockout=True)
    assert torch.equal(intact.blocks[0].computed_sequence_delta, knockout.blocks[0].computed_sequence_delta)
    assert torch.equal(intact.blocks[4].computed_sequence_delta, knockout.blocks[4].computed_sequence_delta)
    assert torch.equal(knockout.blocks[4].exposed_sequence_delta, torch.zeros_like(knockout.blocks[4].exposed_sequence_delta))
    assert bool((knockout.blocks[4].feature_delta != 0).any())


def test_compatible_copy_changes_all_and_only_allowed_state_tensors():
    selected = _construct("selected")
    _mutate_every_state_tensor(selected)
    selected_state = _state_snapshot(selected)
    all_eligible = _construct("all_eligible", seed=37)
    before = _state_snapshot(all_eligible)
    assert all(
        not _tensor_bytes_equal(selected_state[name], before[name])
        for name in selected_state
    )
    report = copy_compatible_state(selected, all_eligible)
    after = _state_snapshot(all_eligible)
    assert set(report.compatible) == set(selected_state)
    assert report.incompatible_source == ()
    assert report.incompatible_destination == ()
    assert _changed_state_names(before, after) == set(report.compatible)
    assert all(_tensor_bytes_equal(after[name], selected_state[name]) for name in after)


def test_router_exclusion_preserves_every_router_tensor_and_copies_every_other_tensor():
    selected = _construct("selected")
    _mutate_every_state_tensor(selected)
    selected_state = _state_snapshot(selected)
    destination = _construct("selected")
    before = _state_snapshot(destination)
    report = copy_compatible_state(selected, destination, include_router=False)
    after = _state_snapshot(destination)
    allowed = {name for name in selected_state if not is_router_parameter(name)}
    routers = {name for name in selected_state if is_router_parameter(name)}
    assert set(report.compatible) == allowed
    assert set(report.incompatible_source) == routers
    assert set(report.incompatible_destination) == routers
    assert _changed_state_names(before, after) == allowed
    assert all(_tensor_bytes_equal(after[name], selected_state[name]) for name in allowed)
    assert all(_tensor_bytes_equal(after[name], before[name]) for name in routers)
    assert all(not _tensor_bytes_equal(after[name], selected_state[name]) for name in routers)


def test_dense_copy_changes_only_compatible_tensors_and_reports_block4_router():
    selected = _construct("selected")
    _mutate_every_state_tensor(selected)
    selected_state = _state_snapshot(selected)
    dense = _construct("dense")
    before = _state_snapshot(dense)
    report = copy_compatible_state(selected, dense)
    after = _state_snapshot(dense)
    assert report.incompatible_source == (
        "blocks.4.mix.source_mixer.attention.router.codebooks",
        "blocks.4.mix.source_mixer.attention.router.query_projection.weight",
        "blocks.4.mix.source_mixer.attention.router.key_projection.weight",
    )
    assert report.incompatible_destination == ()
    assert _changed_state_names(before, after) == set(report.compatible)
    assert all(
        _tensor_bytes_equal(after[name], selected_state[name])
        for name in report.compatible
    )


def test_compatible_copy_refuses_dtype_mismatch_without_mutation():
    source = _StateModule(torch.tensor([1.0, -0.0], dtype=torch.float32))
    destination = _StateModule(torch.tensor([7.0, 9.0], dtype=torch.float64))
    before = _tensor_identity(destination.value)
    with pytest.raises(ValueError, match="dtype mismatch"):
        copy_compatible_state(source, destination)
    assert _tensor_identity(destination.value) == before


def test_compatible_copy_prevalidates_every_tensor_before_any_mutation():
    source = _TwoStateModule(
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.float32),
    )
    destination = _TwoStateModule(
        torch.tensor([7.0], dtype=torch.float32),
        torch.tensor([9.0], dtype=torch.float64),
    )
    before = _state_snapshot(destination)
    with pytest.raises(ValueError, match="dtype mismatch"):
        copy_compatible_state(source, destination)
    after = _state_snapshot(destination)
    assert all(_tensor_bytes_equal(before[name], after[name]) for name in before)


def test_compatible_copy_refuses_device_mismatch_without_mutation():
    source = _StateModule(torch.ones(2, dtype=torch.float32))
    destination = _StateModule(torch.empty(2, dtype=torch.float32, device="meta"))
    with pytest.raises(ValueError, match="device mismatch"):
        copy_compatible_state(source, destination)
    assert destination.value.device.type == "meta"


def test_compatible_copy_preserves_header_raw_bytes_and_hash_identity():
    source_bits = torch.tensor(
        [0, -2147483648, 2143294004, -4194300], dtype=torch.int32
    )
    source = _StateModule(source_bits.view(torch.float32))
    destination = _StateModule(torch.ones(4, dtype=torch.float32))
    report = copy_compatible_state(source, destination)
    assert report.compatible == ("value",)
    assert _tensor_identity(source.value) == _tensor_identity(destination.value)
    assert torch.equal(
        source.value.view(torch.int32), destination.value.view(torch.int32)
    )
    assert not torch.equal(source.value, destination.value)


def test_compatible_copy_rejects_equal_byte_count_with_different_header_shape():
    source = _StateModule(torch.arange(4, dtype=torch.float32))
    destination = _StateModule(torch.full((2, 2), -1.0, dtype=torch.float32))
    before = _tensor_identity(destination.value)
    report = copy_compatible_state(source, destination)
    assert report.compatible == ()
    assert report.incompatible_source == ("value",)
    assert report.incompatible_destination == ("value",)
    assert _tensor_identity(destination.value) == before


def test_selected_checkpoint_loads_strictly_into_local_and_all_eligible_roles():
    selected = _construct("selected")
    _mutate_every_state_tensor(selected)
    checkpoint = _state_snapshot(selected)
    for role in ("local_only", "all_eligible"):
        destination = _construct(role, seed=99)
        destination.load_state_dict(checkpoint, strict=True)
        loaded = _state_snapshot(destination)
        assert loaded.keys() == checkpoint.keys()
        assert all(_tensor_bytes_equal(loaded[name], checkpoint[name]) for name in loaded)
    assert not any(
        fragment in name
        for name in checkpoint
        for fragment in ("state", "postings", "raw_remote", "effective_remote")
    )


def test_parameter_categories_cover_every_parameter_without_hiding_biases_or_codebooks():
    model = _construct()
    parameters = dict(model.named_parameters())
    categories = named_parameter_categories(model)
    assert categories.keys() == parameters.keys()
    assert set(categories.values()) == {
        "matrix",
        "normalization_scale",
        "recurrent_bias",
        "codebook",
    }
    assert parameter_category("blocks.1.mix.bp.bias", parameters["blocks.1.mix.bp.bias"]) == "recurrent_bias"
    assert parameter_category("blocks.1.mix.ag.bias", parameters["blocks.1.mix.ag.bias"]) == "recurrent_bias"
    assert parameter_category("blocks.1.mix.og.bias", parameters["blocks.1.mix.og.bias"]) == "recurrent_bias"
    assert parameter_category("blocks.1.mix.onorm.weight", parameters["blocks.1.mix.onorm.weight"]) == "normalization_scale"
    assert parameter_category(
        "blocks.4.mix.source_mixer.attention.router.codebooks",
        parameters["blocks.4.mix.source_mixer.attention.router.codebooks"],
    ) == "codebook"
    assert is_router_parameter("blocks.4.mix.source_mixer.attention.router.query_projection.weight", 4)
    assert not is_router_parameter("blocks.4.mix.source_mixer.attention.qkv.weight", 4)
    assert named_parameter_categories(_construct("dense")).keys() == dict(
        _construct("dense").named_parameters()
    ).keys()


@pytest.mark.parametrize(
    ("name", "shape"),
    (
        ("blocks.1.mix.mystery.bias", (4,)),
        ("blocks.1.mix.mystery.scalar", ()),
        ("blocks.1.mix.mystery.weight", (64, 64)),
        ("blocks.1.mix.bp.bias", (5,)),
        ("blocks.0.mix.q.weight", (64, 64)),
    ),
)
def test_parameter_category_fails_closed_on_unknown_names_and_wrong_shapes(name, shape):
    with pytest.raises(ValueError):
        parameter_category(name, nn.Parameter(torch.zeros(shape)))
    with pytest.raises(TypeError):
        parameter_category("blocks.1.mix.bp.bias", torch.zeros(4))


def test_all_eligible_selected_read_matches_dense_causal_read_at_query():
    selected = _construct("all_eligible")
    dense = _construct("dense")
    copy_compatible_state(selected, dense)
    x = torch.randn(2, 128, 64)
    selected_result = selected.blocks[4].mix(
        x,
        return_aux=True,
        query_only_position=126,
    )
    dense_result = dense.blocks[4].mix(x, return_aux=True)
    assert torch.allclose(
        selected_result.delta[:, 126],
        dense_result.delta[:, 126],
        atol=1e-5,
        rtol=0,
    )
    assert not hasattr(dense.blocks[4].mix.source_mixer.attention, "router")


def test_firewall_factorization_separates_candidate_keys_query_and_cue():
    model = _construct()
    model.eval()
    base = torch.randint(0, 128, (1, 128), dtype=torch.long)
    candidate_perturbation = base.clone()
    candidate_perturbation[:, :8] = (candidate_perturbation[:, :8] + 1).remainder(128)
    all_candidate_perturbation = base.clone()
    all_candidate_perturbation[:, :80] = (all_candidate_perturbation[:, :80] + 1).remainder(128)
    cue_perturbation = base.clone()
    cue_perturbation[:, 80] = (cue_perturbation[:, 80] + 1).remainder(128)
    with torch.inference_mode():
        original = model(base, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
        candidate = model(candidate_perturbation, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
        all_candidate = model(all_candidate_perturbation, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
        cue = model(cue_perturbation, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
    assert torch.equal(original["key"][:, :, 8:16], candidate["key"][:, :, 8:16])
    assert torch.equal(original["value"][:, :, 8:16], candidate["value"][:, :, 8:16])
    assert torch.equal(original["query"][:, :, 126], all_candidate["query"][:, :, 126])
    assert not torch.equal(original["query"][:, :, 126], cue["query"][:, :, 126])
    assert torch.equal(original["key"][:, :, :80], cue["key"][:, :, :80])
    assert torch.equal(original["value"][:, :, :80], cue["value"][:, :, :80])


@pytest.mark.parametrize("intervention", ("none", "reset", "shuffle"))
@pytest.mark.parametrize("boundary", (8, 80, 96, 127))
def test_future_perturbations_preserve_every_strict_prefix(boundary, intervention):
    model = _construct()
    model.eval()
    base = torch.randint(0, 128, (2, 128), dtype=torch.long)
    changed = base.clone()
    changed[:, boundary:] = (changed[:, boundary:] + 1).remainder(128)
    with torch.inference_mode():
        original = model(base, recurrent_intervention=intervention)
        perturbed = model(changed, recurrent_intervention=intervention)
    assert torch.equal(original[:, :boundary], perturbed[:, :boundary])


def test_input_and_intervention_validation_fail_closed():
    model = _construct()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 128, dtype=torch.int32))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 127, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.full((1, 128), 128, dtype=torch.long))
    with pytest.raises(ValueError):
        _construct("dense")(
            torch.zeros(1, 128, dtype=torch.long),
            route_override=torch.full((1, 128, 1, 2), -1, dtype=torch.long),
        )
    with pytest.raises(ValueError):
        _construct_rung_two()(
            torch.zeros(1, 512, dtype=torch.long),
            recurrent_intervention="reset",
        )
    double_model = _construct().double()
    with pytest.raises(ValueError):
        double_model(torch.zeros(1, 128, dtype=torch.long))
    with pytest.raises(ValueError):
        _construct("all_eligible")(
            torch.zeros(1, 128, dtype=torch.long),
            return_aux=True,
            request_block4_router_loss=True,
        )
