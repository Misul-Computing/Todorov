from __future__ import annotations

import gc
import hashlib
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_unflatten


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.modular_mlx_backend import IPC_SCHEMA_VERSION, MLX_VERSION, RUNG_ONE_SEEDS, OrderedBackgroundWriter, backend_contract, dependency_hashes, durable_gzip_prefix, mapped_mlx_parameter_name, optimizer_parameter_policy, validate_closed_training_gzip, validate_parameter_mapping, validate_stage_request
from src.model.modular_neural_machine import BlockExecution, ModularModelOutput, ModularNeuralMachine, copy_compatible_state, rung_one_config, rung_two_config
from src.model.modular_sources import DenseMixerOutput, RecurrentMixerOutput, RoutedMixerOutput, StateBoundary


WIDTH = 64
HEADS = 4
HEAD_WIDTH = 16
BLOCK_SIZE = 8
RESET_POSITIONS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80)
SCHEDULE = ("routed", "recurrent", "recurrent", "recurrent", "routed", "recurrent", "recurrent", "recurrent")
AUDIT_MEMBERSHIP_CONTRACT = {
    "donor": {"total": 119, "trainable": 113, "groups": {"all_trainable_decay": 72, "all_trainable_zero_decay": 41}},
    "router_only": {"total": 119, "trainable": 3, "groups": {"block_4_router_decay": 2, "block_4_router_zero_decay": 1}},
    "joint": {"total": 119, "trainable": 116, "groups": {"block_4_router_decay": 2, "block_4_router_zero_decay": 1, "other_trainable_decay": 72, "other_trainable_zero_decay": 41}},
    "dense_base": {"total": 116, "trainable": 113, "groups": {"all_trainable_decay": 72, "all_trainable_zero_decay": 41}},
    "dense_continuation": {"total": 116, "trainable": 113, "groups": {"all_trainable_decay": 72, "all_trainable_zero_decay": 41}},
    "rung_two": {"total": 119, "trainable": 113, "groups": {"all_trainable_decay": 72, "all_trainable_zero_decay": 41}},
}
ENDPOINT_STAGE_ORDER = ("donor", "router_only", "joint", "dense_base", "dense_continuation", "rung_two")
ENDPOINT_PARITY_DATA_SEED_BASE = {
    "donor": 610000,
    "router_only": 620000,
    "joint": 630000,
    "dense_base": 640000,
    "dense_continuation": 650000,
    "rung_two": 660000,
}
ROLE_MAP = {
    "selected": "selected",
    "all_eligible": "all_eligible",
    "local_only": "local_only",
    "dense": "dense",
    "rung_two": "rung_two",
}
PILOT_SEED_BASE = 9_999_983
PILOT_SEED_STRIDE = 100
PILOT_DATA_SEED_OFFSET = 1
PILOT_ROUTE_SEED_OFFSET = 2
PILOT_WORKLOADS = (
    ("donor", "donor", "all_eligible", "one_MLX_lane", 1, 16, 128),
    ("selected_vmap5", "joint", "selected", "compiled_MLX_vmap_width_5", 5, 16, 128),
    ("dense_vmap5", "dense_base", "dense", "compiled_MLX_vmap_width_5", 5, 16, 128),
    ("rung_two", "rung_two", "rung_two", "one_MLX_lane", 1, 8, 512),
)
PILOT_WARMUP_UPDATES = (1, 2, 3)
PILOT_TIMED_UPDATES = (4, 5, 6, 7, 8, 9, 10, 11)
PILOT_FINAL_ATTEMPTED_UPDATES = 132
PILOT_FINAL_TOKEN_POSITIONS = 292_864


class MlxEngineError(RuntimeError):
    pass


def recurrent_normalize(value: mx.array) -> mx.array:
    return value * mx.rsqrt(mx.sum(value * value, axis=-1, keepdims=True) + 1e-6)


def routing_normalize(value: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))
    return value / mx.maximum(norm, 1e-12)


def rope(value: mx.array) -> mx.array:
    tokens = value.shape[2]
    half = value.shape[-1] // 2
    frequency = mx.arange(half, dtype=mx.float32) / half
    inverse = 1.0 / (10000.0**frequency)
    angles = mx.arange(tokens, dtype=mx.float32)[:, None] * inverse[None]
    cosine = mx.cos(angles)[None, None]
    sine = mx.sin(angles)[None, None]
    first, second = mx.split(value, 2, axis=-1)
    return mx.concatenate((first * cosine - second * sine, second * cosine + first * sine), axis=-1)


class ExactRMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = mx.ones((width,))
        self.eps = float(np.finfo(np.float32).eps)

    def __call__(self, value: mx.array) -> mx.array:
        return value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps) * self.weight


class FeatureMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(WIDTH, 4 * WIDTH, bias=False)
        self.w2 = nn.Linear(WIDTH, 4 * WIDTH, bias=False)
        self.w3 = nn.Linear(4 * WIDTH, WIDTH, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.w3(nn.silu(self.w1(value)) * self.w2(value))


def triangular_inverse(matrix: mx.array, width: int) -> mx.array:
    identity = mx.eye(width)
    inverse = identity + matrix
    power = matrix
    exponent = 1
    while (1 << exponent) < width:
        power = power @ power
        inverse = inverse @ (identity + power)
        exponent += 1
    return inverse


def independent_chunks(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    primary: mx.array,
    write: mx.array,
    initial: mx.array,
) -> tuple[mx.array, mx.array]:
    chunk = query.shape[-2]
    mask = mx.tril(mx.ones((chunk, chunk)))
    strict = mask - mx.eye(chunk)
    cumulative = mx.cumsum(mx.log(primary), axis=-1)
    gamma = mx.exp(cumulative)
    difference = cumulative[..., :, None] - cumulative[..., None, :]
    lower = mask * mx.exp(mask * difference)
    interaction = write[..., :, None] * (key @ mx.swapaxes(key, -1, -2)) * lower
    inverse = triangular_inverse(-(interaction * strict), chunk)
    update = inverse @ (write[..., None] * value)
    weighted_key = inverse @ ((write * gamma)[..., None] * key)
    corrected = update - weighted_key @ initial
    local = (query @ mx.swapaxes(key, -1, -2)) * lower
    outputs = (gamma[..., None] * query) @ initial + local @ corrected
    carry = mx.exp(cumulative[..., -1, None] - cumulative)[..., None] * key
    final = mx.exp(cumulative[..., -1, None, None]) * initial + mx.swapaxes(carry, -1, -2) @ corrected
    return outputs, final


def recurrent_segmented(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    primary: mx.array,
    write: mx.array,
    intervention: str = "none",
    telemetry: bool = False,
) -> Any:
    if intervention not in {"none", "reset", "shuffle"}:
        raise MlxEngineError("recurrent intervention differs")
    batch, heads, _, width = query.shape
    first_query = query[:, :, :80].reshape(batch, heads, 10, 8, width)
    first_key = key[:, :, :80].reshape(batch, heads, 10, 8, width)
    first_value = value[:, :, :80].reshape(batch, heads, 10, 8, width)
    first_primary = primary[:, :, :80].reshape(batch, heads, 10, 8)
    first_write = write[:, :, :80].reshape(batch, heads, 10, 8)
    first_state = mx.zeros((batch, heads, 10, width, width))
    first_outputs, first_final = independent_chunks(first_query, first_key, first_value, first_primary, first_write, first_state)
    middle_state = mx.zeros((batch, heads, width, width))
    middle_outputs, middle_state = independent_chunks(
        query[:, :, None, 80:96],
        key[:, :, None, 80:96],
        value[:, :, None, 80:96],
        primary[:, :, None, 80:96],
        write[:, :, None, 80:96],
        middle_state[:, :, None],
    )
    middle_state = middle_state[:, :, 0]
    middle_norm = mx.sqrt(mx.sum(middle_state * middle_state, axis=(-2, -1), keepdims=True)) + 1e-6
    middle_state = middle_state * mx.minimum(1.0, 100.0 / middle_norm)
    carry_before = middle_state
    if intervention == "reset":
        middle_state = mx.zeros_like(middle_state)
    elif intervention == "shuffle":
        middle_state = mx.roll(middle_state, 1, axis=0)
    carry_after = middle_state
    last_outputs, last_final = independent_chunks(
        query[:, :, None, 96:],
        key[:, :, None, 96:],
        value[:, :, None, 96:],
        primary[:, :, None, 96:],
        write[:, :, None, 96:],
        middle_state[:, :, None],
    )
    outputs = mx.concatenate((first_outputs.reshape(batch, heads, 80, width), middle_outputs[:, :, 0], last_outputs[:, :, 0]), axis=2)
    if not telemetry:
        return outputs
    boundaries = []
    for index, position in enumerate(RESET_POSITIONS):
        state = first_final[:, :, index]
        if position in {32, 64}:
            norm = mx.sqrt(mx.sum(state * state, axis=(-2, -1), keepdims=True)) + 1e-6
            state = state * mx.minimum(1.0, 100.0 / norm)
        boundaries.append(("firewall_before_reset", position, mx.sqrt(mx.sum(state * state, axis=(-2, -1)))))
        boundaries.append(("firewall_after_reset", position, mx.zeros((batch, heads))))
    for position, state in ((31, first_final[:, :, 3]), (63, first_final[:, :, 7]), (95, carry_before), (127, last_final[:, :, 0])):
        norm = mx.sqrt(mx.sum(state * state, axis=(-2, -1), keepdims=True)) + 1e-6
        clamped = state * mx.minimum(1.0, 100.0 / norm)
        boundaries.append(("chunk_end_after_clamp", position, mx.sqrt(mx.sum(clamped * clamped, axis=(-2, -1)))))
    if intervention != "none":
        boundaries.append((f"carry_before_{intervention}", 96, mx.sqrt(mx.sum(carry_before * carry_before, axis=(-2, -1)))))
        boundaries.append((f"carry_after_{intervention}", 96, mx.sqrt(mx.sum(carry_after * carry_after, axis=(-2, -1)))))
    return outputs, tuple(boundaries)


def recurrent_chunked(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    primary: mx.array,
    write: mx.array,
    telemetry: bool = False,
) -> Any:
    batch, heads, tokens, width = query.shape
    chunks = tokens // 32
    mask = mx.tril(mx.ones((32, 32)))
    strict = mask - mx.eye(32)
    chunk_query = query.reshape(batch, heads, chunks, 32, width)
    chunk_key = key.reshape(batch, heads, chunks, 32, width)
    chunk_value = value.reshape(batch, heads, chunks, 32, width)
    chunk_write = write.reshape(batch, heads, chunks, 32)
    cumulative = mx.cumsum(mx.log(primary).reshape(batch, heads, chunks, 32), axis=-1)
    gamma = mx.exp(cumulative)
    difference = cumulative[..., :, None] - cumulative[..., None, :]
    lower = mask * mx.exp(mask * difference)
    interaction = chunk_write[..., :, None] * (chunk_key @ mx.swapaxes(chunk_key, -1, -2)) * lower
    inverse = triangular_inverse(-(interaction * strict), 32)
    update = inverse @ (chunk_write[..., None] * chunk_value)
    weighted_key = inverse @ ((chunk_write * gamma)[..., None] * chunk_key)
    state = mx.zeros((batch, heads, width, width))
    final_cumulative = cumulative[..., -1]
    final_gate = mx.exp(final_cumulative)
    carry = mx.exp(final_cumulative[..., None] - cumulative)
    outputs = []
    states = []
    for index in range(chunks):
        corrected = update[:, :, index] - weighted_key[:, :, index] @ state
        local = (chunk_query[:, :, index] @ mx.swapaxes(chunk_key[:, :, index], -1, -2)) * lower[:, :, index]
        outputs.append((gamma[:, :, index][..., None] * chunk_query[:, :, index]) @ state + local @ corrected)
        carried_key = carry[:, :, index][..., None] * chunk_key[:, :, index]
        state = final_gate[:, :, index][..., None, None] * state + mx.swapaxes(carried_key, -1, -2) @ corrected
        state_norm = mx.sqrt(mx.sum(state * state, axis=(-2, -1), keepdims=True)) + 1e-6
        state = state * mx.minimum(1.0, 100.0 / state_norm)
        states.append(state)
    result = mx.stack(outputs, axis=2).reshape(batch, heads, tokens, width)
    if not telemetry:
        return result
    boundaries = tuple(("chunk_end_after_clamp", (index + 1) * 32 - 1, mx.sqrt(mx.sum(state * state, axis=(-2, -1)))) for index, state in enumerate(states))
    return result, boundaries


class RecurrentMixer(nn.Module):
    def __init__(self, resets: tuple[int, ...]):
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH, bias=False)
        self.k = nn.Linear(WIDTH, WIDTH, bias=False)
        self.v = nn.Linear(WIDTH, WIDTH, bias=False)
        self.bp = nn.Linear(WIDTH, HEADS, bias=True)
        self.ag = nn.Linear(WIDTH, HEADS, bias=True)
        self.og = nn.Linear(WIDTH, WIDTH, bias=True)
        self.onorm = ExactRMSNorm(HEAD_WIDTH)
        self.o = nn.Linear(WIDTH, WIDTH, bias=False)
        self.resets = resets

    def __call__(self, value: mx.array, intervention: str = "none", telemetry: bool = False) -> Any:
        batch, tokens, _ = value.shape
        query = recurrent_normalize(self.q(value).reshape(batch, tokens, HEADS, HEAD_WIDTH).transpose(0, 2, 1, 3))
        key = recurrent_normalize(self.k(value).reshape(batch, tokens, HEADS, HEAD_WIDTH).transpose(0, 2, 1, 3))
        projected_value = self.v(value).reshape(batch, tokens, HEADS, HEAD_WIDTH).transpose(0, 2, 1, 3)
        write = mx.sigmoid(self.bp(value)).transpose(0, 2, 1)
        primary = mx.sigmoid(self.ag(value)).transpose(0, 2, 1)
        output_gate = mx.sigmoid(self.og(value))
        if self.resets:
            recurrent = recurrent_segmented(query, key, projected_value, primary, write, intervention, telemetry)
        else:
            if intervention != "none":
                raise MlxEngineError("rung-two recurrent intervention differs")
            recurrent = recurrent_chunked(query, key, projected_value, primary, write, telemetry)
        if telemetry:
            outputs, boundaries = recurrent
        else:
            outputs = recurrent
        outputs = self.onorm(outputs).transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH)
        delta = self.o(outputs * output_gate)
        if not telemetry:
            return delta
        return delta, primary, write, output_gate, boundaries


def assign_addresses(features: mx.array, codebooks: mx.array) -> mx.array:
    normalized_books = routing_normalize(codebooks)
    pieces = routing_normalize(features.reshape(*features.shape[:-1], 2, 8))
    assignments = mx.argmax(mx.einsum("...md,mcd->...mc", pieces, normalized_books), axis=-1)
    return assignments[..., 0] * 4 + assignments[..., 1]


def probe_addresses(features: mx.array, codebooks: mx.array) -> mx.array:
    normalized_books = routing_normalize(codebooks)
    prefix = features.shape[:-1]
    pieces = routing_normalize(features.reshape(-1, 2, 8))
    scores = mx.einsum("nmd,mcd->nmc", pieces, normalized_books)
    beam_scores = mx.zeros((scores.shape[0], 1))
    beam_addresses = mx.zeros((scores.shape[0], 1), dtype=mx.int32)
    for subspace in range(2):
        expanded = beam_scores[..., None] + scores[:, subspace, None]
        flattened = expanded.reshape(expanded.shape[0], -1)
        width = min(4, flattened.shape[-1])
        indexes = mx.argsort(flattened, axis=-1)[:, -width:][:, ::-1]
        beam_scores = mx.take_along_axis(flattened, indexes, axis=-1)
        parent = indexes // 4
        code = indexes % 4
        beam_addresses = mx.take_along_axis(beam_addresses, parent, axis=-1) * 4 + code
    return beam_addresses.reshape(*prefix, 4)


def postings_for_addresses(addresses: mx.array) -> mx.array:
    batch, blocks = addresses.shape
    ids = mx.arange(blocks, dtype=mx.int32)[None]
    rows = []
    for address in range(16):
        matched = mx.where(addresses == address, ids, blocks)
        ordered = mx.sort(matched, axis=-1)
        padded = mx.concatenate((ordered, mx.full((batch, 64 - blocks), blocks, dtype=ordered.dtype)), axis=-1)
        rows.append(mx.where(padded < blocks, padded, -1))
    return mx.stack(rows, axis=1)


def searched_remote(
    query_route: mx.array,
    codebooks: mx.array,
    block_features: mx.array,
    addresses: mx.array,
    selected_width: int,
    return_probes: bool = False,
) -> Any:
    batch, tokens, groups, _ = query_route.shape
    if selected_width == 0:
        selected = mx.zeros((batch, tokens, groups, 0), dtype=mx.int32)
        probes = mx.zeros((batch, tokens, groups, 4), dtype=mx.int32)
        return (selected, probes) if return_probes else selected
    postings = postings_for_addresses(addresses)
    probes = probe_addresses(query_route, codebooks)
    batch_ids = mx.arange(batch)[:, None, None, None]
    candidates = postings[batch_ids, probes].reshape(batch, tokens, groups, 256)
    positions = mx.arange(tokens, dtype=mx.int32)
    remote_limits = positions // BLOCK_SIZE
    valid = (candidates >= 0) & (candidates < remote_limits[None, :, None, None])
    safe = mx.maximum(candidates, 0)
    candidate_features = block_features[mx.arange(batch)[:, None, None, None], safe]
    scores = mx.einsum("bqgd,bqgcd->bqgc", query_route, candidate_features)
    scores = mx.where(valid, scores, -mx.inf)
    slots = mx.argsort(scores, axis=-1)[..., -selected_width:][..., ::-1]
    ranked = mx.take_along_axis(candidates, slots, axis=-1)
    ranked_scores = mx.take_along_axis(scores, slots, axis=-1)
    ranked = mx.where(mx.isfinite(ranked_scores), ranked, -1)
    canonical = mx.arange(selected_width, dtype=mx.int32)[None, None, None]
    canonical = mx.broadcast_to(canonical, ranked.shape)
    canonical = mx.where(canonical < remote_limits[None, :, None, None], canonical, -1)
    selected = mx.where((remote_limits <= selected_width)[None, :, None, None], canonical, ranked)
    return (selected, probes) if return_probes else selected


def selected_attention(query: mx.array, key: mx.array, value: mx.array, selected_blocks: mx.array) -> mx.array:
    batch, _, tokens, dimension = query.shape
    block_ids = selected_blocks[..., 0, :]
    candidate_ids = (mx.maximum(block_ids, 0)[..., None] * BLOCK_SIZE + mx.arange(BLOCK_SIZE)[None, None, None]).reshape(batch, tokens, -1)
    valid = mx.broadcast_to((block_ids >= 0)[..., None], (*block_ids.shape, BLOCK_SIZE)).reshape(batch, tokens, -1)
    valid = valid & (candidate_ids <= mx.arange(tokens)[None, :, None]) & (candidate_ids < tokens)
    candidate_ids = mx.sort(mx.where(valid, candidate_ids, tokens), axis=-1)
    valid = candidate_ids < tokens
    unique = mx.concatenate((mx.ones((*candidate_ids.shape[:-1], 1), dtype=mx.bool_), candidate_ids[..., 1:] != candidate_ids[..., :-1]), axis=-1)
    valid = valid & unique
    candidate_ids = mx.where(valid, candidate_ids, 0)
    batch_ids = mx.arange(batch)[:, None, None]
    key_time = key.transpose(0, 2, 1, 3)
    value_time = value.transpose(0, 2, 1, 3)
    candidate_key = key_time[batch_ids, candidate_ids]
    candidate_value = value_time[batch_ids, candidate_ids]
    query_time = query.transpose(0, 2, 1, 3)
    logits = mx.einsum("bthd,btchd->bthc", query_time, candidate_key) / math.sqrt(dimension)
    weights = mx.softmax(mx.where(valid[:, :, None], logits, -mx.inf), axis=-1)
    return mx.einsum("bthc,btchd->bthd", weights, candidate_value).transpose(0, 2, 1, 3)


class RoutedMixer(nn.Module):
    def __init__(self, remote_width: int, query_only: bool):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.out = nn.Linear(WIDTH, WIDTH, bias=False)
        self.query_projection = nn.Linear(WIDTH, 16, bias=False)
        self.key_projection = nn.Linear(WIDTH, 16, bias=False)
        self.codebooks = mx.random.normal((2, 4, 8))
        self.remote_width = remote_width
        self.query_only = query_only

    def __call__(self, value: mx.array, route_override: mx.array | None = None, forced_blocks: mx.array | None = None, internal_loss: bool = False) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        batch, tokens, _ = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, HEADS, HEAD_WIDTH)
        query = rope(qkv[:, :, 0].transpose(0, 2, 1, 3))
        key = rope(qkv[:, :, 1].transpose(0, 2, 1, 3))
        projected_value = qkv[:, :, 2].transpose(0, 2, 1, 3)
        query_input = query.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH)
        key_input = key.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH)
        query_route = routing_normalize(self.query_projection(query_input))[:, :, None]
        key_route = routing_normalize(self.key_projection(key_input))
        blocks = tokens // BLOCK_SIZE
        block_features = routing_normalize(key_route.reshape(batch, blocks, BLOCK_SIZE, 16).mean(axis=2))
        addresses = assign_addresses(block_features, self.codebooks)
        raw_remote = searched_remote(query_route, self.codebooks, block_features, addresses, self.remote_width)
        effective_remote = raw_remote
        if forced_blocks is not None:
            forced = forced_blocks[:, :, None, None]
            already = mx.any(effective_remote == forced, axis=-1, keepdims=True)
            replacement = mx.concatenate((effective_remote[..., :-1], forced), axis=-1)
            effective_remote = mx.where((forced >= 0) & (~already), replacement, effective_remote)
        if self.query_only:
            effective_remote = mx.where((mx.arange(tokens) == 126)[None, :, None, None], effective_remote, -1)
        if route_override is not None:
            effective_remote = route_override
        local = (mx.arange(tokens, dtype=mx.int32) // BLOCK_SIZE)[None, :, None, None]
        local = mx.broadcast_to(local, (batch, tokens, 1, 1))
        selected = mx.concatenate((effective_remote, local), axis=-1)
        attended = selected_attention(query, key, projected_value, selected)
        delta = self.out(attended.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH))
        router_loss = self.router_loss(query, key, query_route, key_route) if internal_loss else mx.array(0.0)
        return delta, query_route, key_route, router_loss, raw_remote, effective_remote, addresses

    def router_loss(self, query: mx.array, key: mx.array, query_route: mx.array, key_route: mx.array) -> mx.array:
        batch, _, tokens, _ = query.shape
        block_count = tokens // BLOCK_SIZE
        teacher_logits = mx.einsum("bhtd,bhsd->bhts", mx.stop_gradient(query), mx.stop_gradient(key)).mean(axis=1)
        positions = mx.arange(tokens)
        remote_limits = positions // BLOCK_SIZE
        valid = positions[None, None, :] < (remote_limits * BLOCK_SIZE)[None, :, None]
        active = remote_limits > 0
        safe_valid = valid | ((~active)[None, :, None] & (positions[None, None, :] == 0))
        teacher = mx.softmax(mx.where(safe_valid, teacher_logits, -mx.inf), axis=-1)
        teacher = mx.where(remote_limits[None, :, None] > 0, teacher, 0.0)
        teacher_blocks = teacher.reshape(batch, tokens, block_count, BLOCK_SIZE).sum(axis=-1)
        blocks = key_route.reshape(batch, block_count, BLOCK_SIZE, 16).mean(axis=2)
        logits = mx.einsum("btgd,bnd->btgn", query_route, blocks)[:, :, 0]
        block_ids = mx.arange(block_count)
        valid_blocks = block_ids[None, :] < remote_limits[:, None]
        safe_blocks = valid_blocks | ((~active)[:, None] & (block_ids[None, :] == 0))
        masked_logits = mx.where(safe_blocks[None], logits, -mx.inf)
        log_probs = masked_logits - mx.logsumexp(masked_logits, axis=-1, keepdims=True)
        log_probs = mx.where(valid_blocks[None], log_probs, 0.0)
        losses = -mx.sum(teacher_blocks * log_probs, axis=-1)
        return mx.sum(mx.where(active[None], losses, 0.0)) / (batch * mx.sum(active))


class DenseMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.out = nn.Linear(WIDTH, WIDTH, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        batch, tokens, _ = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, HEADS, HEAD_WIDTH)
        query = rope(qkv[:, :, 0].transpose(0, 2, 1, 3))
        key = rope(qkv[:, :, 1].transpose(0, 2, 1, 3))
        projected_value = qkv[:, :, 2].transpose(0, 2, 1, 3)
        scores = mx.einsum("bhtd,bhsd->bhts", query, key) / math.sqrt(HEAD_WIDTH)
        causal = mx.arange(tokens)[None, :] <= mx.arange(tokens)[:, None]
        weights = mx.softmax(mx.where(causal[None, None], scores, -mx.inf), axis=-1)
        attended = mx.einsum("bhts,bhsd->bhtd", weights, projected_value)
        return self.out(attended.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH))


class MlxModularBlock(nn.Module):
    def __init__(self, index: int, role: str):
        super().__init__()
        self.n1 = ExactRMSNorm(WIDTH)
        resets = () if role == "rung_two" else RESET_POSITIONS
        if SCHEDULE[index] == "recurrent":
            self.mix = RecurrentMixer(resets)
        elif index == 0:
            self.mix = RoutedMixer(0, False)
        elif role == "dense":
            self.mix = DenseMixer()
        else:
            remote = {"selected": 2, "all_eligible": 15, "local_only": 0, "rung_two": 0}[role]
            self.mix = RoutedMixer(remote, role in {"selected", "all_eligible", "local_only"})
        self.n2 = ExactRMSNorm(WIDTH)
        self.mlp = FeatureMixer()
        self.kind = SCHEDULE[index]
        self.index = index
        self.dense = role == "dense" and index == 4


class MlxModularModel(nn.Module):
    def __init__(self, role: str):
        super().__init__()
        if role not in ROLE_MAP:
            raise MlxEngineError("model role differs")
        vocabulary = 256 if role == "rung_two" else 128
        self.embed = nn.Embedding(vocabulary, WIDTH)
        self.blocks = [MlxModularBlock(index, role) for index in range(8)]
        self.nf = ExactRMSNorm(WIDTH)
        self.head = nn.Linear(WIDTH, vocabulary, bias=False)
        self.role = role

    def __call__(
        self,
        tokens: mx.array,
        route_override: mx.array | None = None,
        internal_loss: bool = False,
        forced_blocks: mx.array | None = None,
        recurrent_intervention: str = "none",
        recurrent_knockout: bool = False,
        block4_routed_knockout: bool = False,
        evaluation_telemetry: bool = False,
    ) -> tuple[Any, ...]:
        hidden = self.embed(tokens)
        query_route = None
        key_route = None
        route_loss = mx.array(0.0)
        raw_routes = []
        effective_routes = []
        address_routes = []
        recurrent_records = []
        sequence_deltas = []
        feature_deltas = []
        for block in self.blocks:
            normalized = block.n1(hidden)
            if block.kind == "routed" and not block.dense:
                mixed, current_query, current_key, current_route_loss, raw_remote, effective_remote, addresses = block.mix(
                    normalized,
                    route_override if block.index == 4 else None,
                    forced_blocks if block.index == 4 else None,
                    internal_loss and block.index == 4,
                )
                raw_routes.append(raw_remote)
                effective_routes.append(effective_remote)
                address_routes.append(addresses)
                if block.index == 4:
                    query_route = current_query
                    key_route = current_key
                    route_loss = current_route_loss
            elif block.kind == "recurrent":
                recurrent = block.mix(normalized, recurrent_intervention, evaluation_telemetry)
                if evaluation_telemetry:
                    mixed, primary, write, output_gate, boundaries = recurrent
                    recurrent_records.append((block.index, primary, write, output_gate, boundaries))
                else:
                    mixed = recurrent
            else:
                mixed = block.mix(normalized)
            sequence_deltas.append(mixed)
            expose_zero = (recurrent_knockout and block.kind == "recurrent") or (block4_routed_knockout and block.index == 4 and not block.dense)
            hidden = hidden + (mx.zeros_like(mixed) if expose_zero else mixed)
            feature_delta = block.mlp(block.n2(hidden))
            feature_deltas.append(feature_delta)
            hidden = hidden + feature_delta
        final_hidden = self.nf(hidden)
        result = self.head(final_hidden), final_hidden, query_route, key_route, route_loss, tuple(raw_routes), tuple(effective_routes), tuple(sequence_deltas), tuple(feature_deltas)
        if not evaluation_telemetry:
            return result
        return (*result, tuple(address_routes), tuple(recurrent_records))


class MlxTorchEvaluationAdapter:
    def __init__(self, role: str, construction_seed: int, checkpoint: Mapping[str, Any]):
        self.role = role
        self.reference = torch_model(role, construction_seed)
        self.reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.config = self.reference.config
        self.model = MlxModularModel(role)
        load_torch_state(self.model, checkpoint["model_state_dict"])

    def eval(self) -> Any:
        return self

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.reference.state_dict()

    def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> ModularModelOutput:
        allowed = {"return_aux", "recurrent_telemetry", "route_detail", "request_block4_router_loss", "forced_blocks", "route_override", "recurrent_intervention", "recurrent_knockout", "block4_routed_knockout"}
        if set(kwargs) - allowed or kwargs.get("return_aux") is not True:
            raise MlxEngineError("evaluation adapter arguments differ")
        forced = kwargs.get("forced_blocks")
        override = kwargs.get("route_override")
        output = self.model(
            mx.array(tokens.numpy()),
            None if override is None else mx.array(override.numpy()),
            bool(kwargs.get("request_block4_router_loss", False)),
            None if forced is None else mx.array(forced.numpy()),
            str(kwargs.get("recurrent_intervention", "none")),
            bool(kwargs.get("recurrent_knockout", False)),
            bool(kwargs.get("block4_routed_knockout", False)),
            True,
        )
        mx.eval(output)
        recurrent_by_block = {record[0]: record[1:] for record in output[10]}
        route_index = 0
        blocks = []
        for block_index, kind in enumerate(SCHEDULE):
            computed = torch.from_numpy(np.array(output[7][block_index]).copy())
            exposed_zero = (kwargs.get("recurrent_knockout") is True and kind == "recurrent") or (kwargs.get("block4_routed_knockout") is True and block_index == 4 and self.role != "dense")
            exposed = torch.zeros_like(computed) if exposed_zero else computed
            feature = torch.from_numpy(np.array(output[8][block_index]).copy())
            if kind == "recurrent":
                primary, write, output_gate, raw_boundaries = recurrent_by_block[block_index]
                boundaries = tuple(StateBoundary(boundary_kind, position, torch.from_numpy(np.array(norms).copy()).to(torch.float64)) for boundary_kind, position, norms in raw_boundaries)
                mixer_output = RecurrentMixerOutput(computed, torch.from_numpy(np.array(primary).copy()), torch.from_numpy(np.array(write).copy()), torch.from_numpy(np.array(output_gate).copy()), boundaries)
                block_kind = "recurrent"
            elif block_index == 4 and self.role == "dense":
                empty = torch.empty(0)
                mixer_output = DenseMixerOutput(computed, empty, empty, empty)
                block_kind = "dense"
            else:
                selected_width = int(output[5][route_index].shape[-1])
                if selected_width:
                    query_route = output[2]
                    probes = probe_addresses(query_route, self.model.blocks[block_index].mix.codebooks)
                else:
                    query_route = mx.zeros((tokens.shape[0], tokens.shape[1], 1, 16))
                    probes = mx.zeros((tokens.shape[0], tokens.shape[1], 1, 4), dtype=mx.int32)
                telemetry = routing_telemetry(output[5][route_index], output[6][route_index], output[9][route_index], probes)
                telemetry["selected_blocks"] = torch.empty(0, dtype=torch.long)
                route_tensor = torch.from_numpy(np.array(query_route).copy())
                mixer_output = RoutedMixerOutput(computed, None, route_tensor, torch.empty(0), telemetry, ())
                route_index += 1
                block_kind = "routed"
            blocks.append(BlockExecution(block_index, block_kind, computed, exposed, feature, mixer_output))
        logits = torch.from_numpy(np.array(output[0]).copy())
        hidden = torch.from_numpy(np.array(output[1]).copy())
        return ModularModelOutput(logits, hidden, tuple(blocks))


class QualificationEvaluationConnection:
    def __init__(self, resource_sample_ids: list[int]):
        self.message = None
        self.resource_sample_ids = list(resource_sample_ids)

    def send(self, message: Mapping[str, Any]) -> None:
        self.message = message

    def recv(self) -> dict[str, Any]:
        return {"ack": True, "sample_ids": self.resource_sample_ids} if self.message["kind"] == "resource_refs" else {"ack": True}


def evaluate_rung_one_qualification(run_root: Path, seed: int, routing_stream: Any, forward_sequence: int, resource_sample_ids: list[int]) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    runtime = cpu._import_runtime()
    selected_canonical = torch_model("selected", seed)
    state_tensors, state_sha256 = cpu._state_manifest(selected_canonical)
    state_manifest = {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "construction_seed": seed, "role": "selected_canonical", "state_tensors": state_tensors, "state_sha256": state_sha256}
    state_manifest_path = run_root / "rung1" / str(seed) / "selected_canonical_state_manifest.json"
    cpu.write_canonical_json(state_manifest_path, state_manifest)
    sentinel_path = run_root / "run" / "sentinels" / "selected_attention_oracle_payload.json"
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    oracle_error = cpu._selected_attention_oracle_for_model(selected_canonical, runtime, sentinel)
    oracle_detail = {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "construction_seed": seed, "constructor_state_manifest_sha256": cpu.sha256_file(state_manifest_path), "sentinel_payload_sha256": cpu.sha256_file(sentinel_path), "max_error": oracle_error, "tolerance": 1e-5, "pass": math.isfinite(oracle_error) and oracle_error <= 1e-5}
    oracle_detail_path = run_root / "rung1" / str(seed) / "selected_attention_oracle_detail.json"
    cpu.write_canonical_json(oracle_detail_path, oracle_detail)
    if oracle_detail["pass"] is not True:
        raise MlxEngineError("selected attention oracle failed")
    oracle_provenance = [cpu.sha256_file(state_manifest_path), cpu.sha256_file(sentinel_path), cpu.sha256_file(oracle_detail_path)]
    checkpoint_root = run_root / "rung1" / str(seed) / "checkpoints"
    checkpoint_paths = {"selected": checkpoint_root / "final_last.pt", "donor": checkpoint_root / "donor_last.pt", "dense": checkpoint_root / "dense_last.pt"}
    checkpoints = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in checkpoint_paths.items()}
    models = {
        "selected": MlxTorchEvaluationAdapter("selected", seed, checkpoints["selected"]),
        "local": MlxTorchEvaluationAdapter("local_only", seed, checkpoints["selected"]),
        "donor": MlxTorchEvaluationAdapter("all_eligible", seed, checkpoints["donor"]),
        "clone": MlxTorchEvaluationAdapter("all_eligible", seed, checkpoints["selected"]),
        "dense": MlxTorchEvaluationAdapter("dense", seed, checkpoints["dense"]),
    }
    hashes = {
        "selected": cpu.sha256_file(checkpoint_paths["selected"]),
        "local": cpu.sha256_file(checkpoint_paths["selected"]),
        "donor": cpu.sha256_file(checkpoint_paths["donor"]),
        "clone": cpu.sha256_file(checkpoint_paths["selected"]),
        "dense": cpu.sha256_file(checkpoint_paths["dense"]),
    }
    started = time.perf_counter_ns()
    evaluation, predictions, state, interventions, next_sequence, usage = cpu._evaluate_rung_one(QualificationEvaluationConnection(resource_sample_ids), "Q", run_root, seed, models, hashes, oracle_error, oracle_provenance, runtime, routing_stream, forward_sequence, 0, 0)
    checkpoint_by_condition = {condition: (cpu.RUNG_ONE_MODEL_BY_CONDITION[condition], hashes[cpu.RUNG_ONE_MODEL_BY_CONDITION[condition]]) for condition in cpu.RUNG_ONE_CONDITIONS}
    cpu.validate_state_records(state, 1, checkpoint_by_condition)
    cpu.validate_intervention_records(interventions, 1, checkpoint_by_condition)
    seed_root = run_root / "rung1" / str(seed)
    cpu._write_canonical_jsonl(seed_root / "evaluation.jsonl", evaluation)
    cpu._write_canonical_gzip(seed_root / "predictions.jsonl.gz", predictions)
    cpu.write_canonical_json(seed_root / "state_stats.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": state})
    cpu.write_canonical_json(seed_root / "intervention_deltas.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": interventions})
    result = {
        "seed": seed,
        "wall_seconds": (time.perf_counter_ns() - started) / 1_000_000_000,
        "evaluation_rows": len(evaluation),
        "prediction_rows": len(predictions),
        "state_rows": len(state),
        "intervention_rows": len(interventions),
        "forward_sequence": next_sequence,
        "usage": usage,
        "oracle_error": oracle_error,
    }
    if (result["evaluation_rows"], result["prediction_rows"], result["state_rows"], result["intervention_rows"], result["forward_sequence"]) != (65, 6144, 1980, 108, 4048):
        raise MlxEngineError("rung-one qualification cardinality differs")
    return result


def evaluate_rung_two_qualification(run_root: Path, resource_sample_ids: list[int]) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    seed_root = run_root / "rung2" / "83"
    checkpoint_path = seed_root / "checkpoints" / "final_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hash = cpu.sha256_file(checkpoint_path)
    model = MlxTorchEvaluationAdapter("rung_two", 83, checkpoint)
    evaluation_path = run_root / "data" / "r2_eval_1000083.pt"
    artifact = torch.load(evaluation_path, map_location="cpu", weights_only=False)
    evaluation = cpu.payload_to_tensors(artifact["payload"], torch)
    evaluation_hash = cpu.sha256_file(evaluation_path)
    gate_accumulators = {}
    state_accumulators = {}
    intervention_sums = {}
    checkpoint_by_condition = {condition: ("rung_two", checkpoint_hash) for condition in cpu.RUNG_TWO_CONDITIONS}
    identities = cpu._intervention_identity_registry(2, checkpoint_by_condition)
    baseline_squares = {}
    predictions = []
    results = []
    parity_error = 0.0
    started_all = time.perf_counter_ns()
    for condition in cpu.RUNG_TWO_CONDITIONS:
        correct = 0
        started = time.perf_counter_ns()
        for batch_index in range(16):
            start = batch_index * 32
            stop = start + 32
            tokens = evaluation["tokens"][start:stop]
            knockout = condition == "recurrent_knockout"
            source = model(tokens, return_aux=True, route_detail=True, recurrent_knockout=knockout)
            output = model(tokens, return_aux=True, route_detail=True, recurrent_telemetry=True, recurrent_knockout=knockout)
            target = evaluation["targets"][start:stop, 510]
            predicted, matches, error = cpu._rung_two_source_prediction(torch, source, output, target, {"seed": 83, "stage": condition, "logical_update": batch_index})
            parity_error = max(parity_error, error)
            correct += int(matches.sum())
            for offset in range(32):
                predictions.append({"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "condition": condition, "example_index": start + offset, "original_condition": None, "foreign_condition": None, "original_source": None, "foreign_source": None, "target": int(target[offset]), "prediction": int(predicted[offset]), "correct": bool(matches[offset]), "original_source_hit": None, "foreign_source_hit": None, "condition_stratum": "not_applicable", "checkpoint_sha256": checkpoint_hash})
            for block in output.blocks:
                if block.kind != "recurrent":
                    continue
                recurrent = block.mixer_output
                for head in range(4):
                    cpu._stat_accumulate(gate_accumulators.setdefault((condition, block.block_index, head), cpu._new_stat_accumulator()), recurrent.primary_gate[:, head], torch)
                for statistic, tensor in (("primary_gate", recurrent.primary_gate), ("beta_gate", recurrent.write_gate), ("output_gate", recurrent.output_gate)):
                    key = ("rung_two", checkpoint_hash, block.block_index, condition, "not_applicable", None, statistic)
                    cpu._stat_accumulate(state_accumulators.setdefault(key, cpu._new_stat_accumulator()), tensor, torch)
                for boundary in recurrent.boundaries:
                    key = ("rung_two", checkpoint_hash, block.block_index, condition, "global_chunk_end", boundary.position, "state_l2")
                    cpu._stat_accumulate(state_accumulators.setdefault(key, cpu._new_stat_accumulator()), boundary.norms, torch)
                post_square = float(block.computed_sequence_delta.to(torch.float64).square().sum())
                exposed_square = float(block.exposed_sequence_delta.to(torch.float64).square().sum())
                identity = identities[condition]
                baseline_key = (identity["baseline_model"], identity["baseline_checkpoint_sha256"], identity["baseline_condition"], batch_index, block.block_index)
                if condition == identity["baseline_condition"]:
                    baseline_squares[baseline_key] = post_square
                values = intervention_sums.setdefault((block.block_index, condition), cpu._new_l2_accumulator())
                values[0].append(baseline_squares[baseline_key])
                values[1].append(post_square)
                values[2].append(exposed_square)
        results.append({"condition": condition, "successes": correct, "elapsed_seconds": (time.perf_counter_ns() - started) / 1_000_000_000})
    evaluation_rows = [cpu._rung_two_evaluation_row(run_root.name, result["condition"], result["successes"], checkpoint_hash, evaluation_hash, result["elapsed_seconds"], resource_sample_ids) for result in results]
    gate_conditions = []
    for condition in cpu.RUNG_TWO_CONDITIONS:
        records = []
        parts = []
        for block in cpu.RECURRENT_BLOCKS:
            for head in range(4):
                accumulator = gate_accumulators[(condition, block, head)]
                records.append({"block": block, "head": head, **cpu._stat_values(accumulator)})
                parts.append(accumulator)
        aggregate = {"block": None, "head": None, **cpu._stat_values(cpu._merge_stat_accumulators(parts))}
        gate_conditions.append({"condition": condition, "gate_id": f"r2.{condition}.primary_gate_nonfinite_count.not_applicable", "records": records, "aggregate": aggregate, "gate_operator": "==", "gate_threshold": 0, "gate_threshold_count": 0, "gate_threshold_unit": "count", "gate_pass": aggregate["nonfinite_count"] == 0})
        primary = [state_accumulators[("rung_two", checkpoint_hash, block, condition, "not_applicable", None, "primary_gate")] for block in cpu.RECURRENT_BLOCKS]
        state_accumulators[("rung_two", checkpoint_hash, None, condition, "not_applicable", None, "primary_gate")] = cpu._merge_stat_accumulators(primary)
    state_records = []
    for key in sorted(state_accumulators, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        model_name, checkpoint_identity, block, condition, boundary, position, statistic = key
        state_records.append({"model": model_name, "checkpoint_sha256": checkpoint_identity, "block": block, "condition": condition, "boundary": boundary, "position": position, "statistic": statistic, **cpu._stat_values(state_accumulators[key])})
    intervention_records = []
    for (block, condition), accumulator in intervention_sums.items():
        values = cpu._l2_values(accumulator)
        intervention_records.append({**identities[condition], "block": block, "condition": condition, "pre_delta_l2": values[0], "post_delta_l2": values[1], "exposed_delta_l2": values[2]})
    for condition in cpu.RUNG_TWO_CONDITIONS:
        values = cpu._l2_values(cpu._merge_l2_accumulators([intervention_sums[(block, condition)] for block in cpu.RECURRENT_BLOCKS]))
        intervention_records.append({**identities[condition], "block": None, "condition": condition, "pre_delta_l2": values[0], "post_delta_l2": values[1], "exposed_delta_l2": values[2]})
    intervention_records.sort(key=lambda record: (record["condition"], -1 if record["block"] is None else record["block"]))
    cpu.validate_state_records(state_records, 2, checkpoint_by_condition)
    cpu.validate_intervention_records(intervention_records, 2, checkpoint_by_condition)
    cpu._write_canonical_jsonl(seed_root / "evaluation.jsonl", evaluation_rows)
    cpu._write_canonical_gzip(seed_root / "predictions.jsonl.gz", predictions)
    cpu.write_canonical_json(seed_root / "gate_stats.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "checkpoint_sha256": checkpoint_hash, "conditions": gate_conditions})
    cpu.write_canonical_json(seed_root / "state_stats.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "records": state_records})
    cpu.write_canonical_json(seed_root / "intervention_deltas.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "records": intervention_records})
    result = {"wall_seconds": (time.perf_counter_ns() - started_all) / 1_000_000_000, "evaluation_rows": len(evaluation_rows), "prediction_rows": len(predictions), "state_rows": len(state_records), "intervention_rows": len(intervention_records), "gate_conditions": len(gate_conditions), "source_telemetry_max_error": parity_error}
    if tuple(result[key] for key in ("evaluation_rows", "prediction_rows", "state_rows", "intervention_rows", "gate_conditions")) != (2, 1024, 230, 14, 2) or parity_error != 0.0:
        raise MlxEngineError("rung-two qualification cardinality or parity differs")
    return result


def checkpoint_model(role: str, seed: int, checkpoint: Mapping[str, Any]) -> ModularNeuralMachine:
    model = torch_model(role, seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def qualification_runtime_accounting(model: ModularNeuralMachine, checkpoint: Mapping[str, Any], usage: Mapping[str, int], batch_size: int) -> dict[str, int]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    record = cpu._evaluation_runtime_accounting(model, usage, batch_size)
    optimizer_count, optimizer_bytes = cpu._tensor_tree_storage(checkpoint["optimizer_state_dict"], torch)
    record["optimizer_state_count"] = optimizer_count
    record["optimizer_state_bytes"] = optimizer_bytes
    return record


def write_qualification_accounting(run_root: Path, stage_audits: Mapping[str, Mapping[int, list[dict[str, Any]]]], evaluation: Mapping[str, Any], resource_sample_ids_by_seed: Mapping[int, list[int]]) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    rung_one_results = {record["seed"]: record for record in evaluation["rung_one"]}
    for seed in RUNG_ONE_SEEDS:
        resource_sample_ids = resource_sample_ids_by_seed[seed]
        seed_root = run_root / "rung1" / str(seed)
        checkpoints = {
            "all_eligible_donor": torch.load(seed_root / "checkpoints" / "donor_last.pt", map_location="cpu", weights_only=False),
            "selected": torch.load(seed_root / "checkpoints" / "final_last.pt", map_location="cpu", weights_only=False),
            "dense_causal": torch.load(seed_root / "checkpoints" / "dense_last.pt", map_location="cpu", weights_only=False),
        }
        models = {
            "all_eligible_donor": checkpoint_model("all_eligible", seed, checkpoints["all_eligible_donor"]),
            "selected": checkpoint_model("selected", seed, checkpoints["selected"]),
            "dense_causal": checkpoint_model("dense", seed, checkpoints["dense_causal"]),
        }
        audit_records = [record for stage in ("donor", "router_only", "joint", "dense_base", "dense_continuation") for record in stage_audits[stage][seed]]
        usage_keys = {"all_eligible_donor": "donor", "selected": "selected", "dense_causal": "dense"}
        train_rows = cpu._canonical_jsonl_records(seed_root / "train.jsonl")
        accounting_models = []
        for model_name in ("all_eligible_donor", "selected", "dense_causal"):
            model = models[model_name]
            audits = [record for record in audit_records if record["model"] == model_name]
            runtime_record = qualification_runtime_accounting(model, checkpoints[model_name], rung_one_results[seed]["usage"][usage_keys[model_name]], 32)
            work = cpu._model_work_from_train_rows(train_rows, model_name)
            accounting_models.append({"model": model_name, "entries": cpu._accounting_entries(model, audits, runtime_record), **work, "resource_sample_ids": resource_sample_ids})
        cpu.validate_model_accounting(accounting_models, train_rows)
        cpu.write_canonical_json(seed_root / "accounting.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "models": accounting_models})
        cpu.write_canonical_json(seed_root / "resource_refs.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "sample_ids": resource_sample_ids})
    seed_root = run_root / "rung2" / "83"
    resource_sample_ids = resource_sample_ids_by_seed[83]
    checkpoint = torch.load(seed_root / "checkpoints" / "final_last.pt", map_location="cpu", weights_only=False)
    model = checkpoint_model("rung_two", 83, checkpoint)
    route_count, route_bytes = cpu._route_index_storage(32, 512, 2)
    usage = {"route_index_storage_count": route_count, "route_index_storage_bytes": route_bytes, "routing_workspace_count": 0, "routing_workspace_bytes": 0}
    runtime_record = qualification_runtime_accounting(model, checkpoint, usage, 32)
    train_rows = cpu._canonical_jsonl_records(seed_root / "train.jsonl")
    audits = stage_audits["rung_two"][83]
    work = cpu._model_work_from_train_rows(train_rows, "rung_two")
    accounting_model = {"model": "rung_two", "entries": cpu._accounting_entries(model, audits, runtime_record), **work, "resource_sample_ids": resource_sample_ids}
    cpu.validate_model_accounting([accounting_model], train_rows)
    cpu.write_canonical_json(seed_root / "accounting.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "models": [accounting_model]})
    cpu.write_canonical_json(seed_root / "resource_refs.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "sample_ids": resource_sample_ids})


def write_qualification_parity(run_root: Path, endpoint_parity_records: list[Mapping[str, Any]]) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    runtime = cpu._import_runtime()
    for seed in (*RUNG_ONE_SEEDS, 83):
        rung = 1 if seed != 83 else 2
        seed_root = run_root / (f"rung1/{seed}" if rung == 1 else "rung2/83")
        endpoints = cpu._load_claim_endpoints(run_root, rung, seed, torch)
        train_rows = cpu._canonical_jsonl_records(seed_root / "train.jsonl")
        evaluation_payload, evaluation_sha256 = cpu._load_evaluation_evidence(run_root, rung, seed, torch)
        checkpoint_by_condition = cpu._checkpoint_by_condition(rung, endpoints)
        prediction_evidence = cpu._validate_prediction_artifact(run_root, cpu.load_prereg_payload(), seed_root, rung, seed, checkpoint_by_condition, evaluation_payload, evaluation_sha256)
        evaluation_rows = cpu._canonical_jsonl_records(seed_root / "evaluation.jsonl")
        routing_evidence = cpu._validate_routing_artifact(run_root, cpu.load_prereg_payload(), seed_root, seed, evaluation_rows, prediction_evidence, checkpoint_by_condition) if rung == 1 else None
        data_evidence = cpu._validate_rung_one_data_artifacts(run_root, seed, evaluation_payload, routing_evidence, torch) if rung == 1 else None
        oracle_evidence = cpu._validate_selected_oracle_evidence(run_root, cpu.load_prereg_payload(), seed, evaluation_rows, runtime) if rung == 1 else None
        intervention_records = cpu._canonical_json_artifact(seed_root / "intervention_deltas.json")["records"]
        trained_records = [record for record in endpoint_parity_records if record["construction_seed"] == seed]
        expected_stages = ENDPOINT_STAGE_ORDER[:-1] if rung == 1 else ("rung_two",)
        trained_records.sort(key=lambda record: expected_stages.index(record["execution_stage"]))
        if tuple(record["execution_stage"] for record in trained_records) != expected_stages:
            raise MlxEngineError("trained endpoint parity per-seed closure differs")
        facts = cpu.reconstruct_semantic_parity_facts(run_root, rung, seed, endpoints, train_rows, evaluation_rows, evaluation_payload, intervention_records, runtime, routing_evidence, data_evidence, oracle_evidence, trained_records)
        checks = cpu.build_ordered_parity_checks(run_root, seed, facts)
        checkpoint_sha256 = endpoints["joint" if rung == 1 else "rung_two"]["sha256"]
        cpu.write_canonical_json(seed_root / "parity.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": rung, "claim_seed": seed, "construction_seed": seed, "checkpoint_sha256": checkpoint_sha256, "checks": checks})


def evaluate_qualification(run_root: Path, routing_streams: Mapping[int, Any], forward_sequences: Mapping[int, int], resource_sample_ids_by_seed: Mapping[int, list[int]], stage_audits: Mapping[str, Mapping[int, list[dict[str, Any]]]], endpoint_parity_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rung_one = []
    for seed in RUNG_ONE_SEEDS:
        rung_one.append(evaluate_rung_one_qualification(run_root, seed, routing_streams[seed], forward_sequences[seed], resource_sample_ids_by_seed[seed]))
    rung_two = evaluate_rung_two_qualification(run_root, resource_sample_ids_by_seed[83])
    result = {"rung_one": rung_one, "rung_two": rung_two}
    write_qualification_accounting(run_root, stage_audits, result, resource_sample_ids_by_seed)
    validate_trained_endpoint_parity_records(endpoint_parity_records, 26)
    write_qualification_parity(run_root, endpoint_parity_records)
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    if len(cpu.validate_gate_input_package(run_root, cpu.load_prereg_payload())) != 124:
        raise MlxEngineError("qualification gate package cardinality differs")
    return result


def validate_resource_sample_ids_by_seed(value: Any) -> dict[int, list[int]]:
    expected = {str(seed) for seed in (*RUNG_ONE_SEEDS, 83)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MlxEngineError("evaluation resource seed identities differ")
    result = {}
    for raw_seed in sorted(value, key=int):
        sample_ids = value[raw_seed]
        if not isinstance(sample_ids, list) or not sample_ids or any(type(sample_id) is not int or sample_id < 0 for sample_id in sample_ids) or sample_ids != sorted(set(sample_ids)):
            raise MlxEngineError("evaluation resource sample identities differ")
        result[int(raw_seed)] = list(sample_ids)
    return result


def functional_linear(value: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    output = value @ mx.swapaxes(weight, -1, -2)
    return output if bias is None else output + bias


def functional_norm(value: mx.array, weight: mx.array) -> mx.array:
    return value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + float(np.finfo(np.float32).eps)) * weight


def functional_feature(value: mx.array, prefix: str, parameters: Mapping[str, mx.array]) -> mx.array:
    first = functional_linear(value, parameters[f"{prefix}.w1.weight"])
    second = functional_linear(value, parameters[f"{prefix}.w2.weight"])
    return functional_linear(nn.silu(first) * second, parameters[f"{prefix}.w3.weight"])


def functional_recurrent(value: mx.array, prefix: str, parameters: Mapping[str, mx.array], resets: tuple[int, ...]) -> mx.array:
    batch, tokens, _ = value.shape
    query = functional_linear(value, parameters[f"{prefix}.q.weight"])
    key = functional_linear(value, parameters[f"{prefix}.k.weight"])
    projected_value = functional_linear(value, parameters[f"{prefix}.v.weight"])
    query = recurrent_normalize(query.reshape(batch, tokens, HEADS, HEAD_WIDTH).transpose(0, 2, 1, 3))
    key = recurrent_normalize(key.reshape(batch, tokens, HEADS, HEAD_WIDTH).transpose(0, 2, 1, 3))
    projected_value = projected_value.reshape(batch, tokens, HEADS, HEAD_WIDTH).transpose(0, 2, 1, 3)
    write = mx.sigmoid(functional_linear(value, parameters[f"{prefix}.bp.weight"], parameters[f"{prefix}.bp.bias"])).transpose(0, 2, 1)
    primary = mx.sigmoid(functional_linear(value, parameters[f"{prefix}.ag.weight"], parameters[f"{prefix}.ag.bias"])).transpose(0, 2, 1)
    output_gate = mx.sigmoid(functional_linear(value, parameters[f"{prefix}.og.weight"], parameters[f"{prefix}.og.bias"]))
    outputs = recurrent_segmented(query, key, projected_value, primary, write) if resets else recurrent_chunked(query, key, projected_value, primary, write)
    outputs = functional_norm(outputs, parameters[f"{prefix}.onorm.weight"])
    outputs = outputs.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH)
    return functional_linear(outputs * output_gate, parameters[f"{prefix}.o.weight"])


def functional_routed(
    value: mx.array,
    prefix: str,
    parameters: Mapping[str, mx.array],
    remote_width: int,
    query_only: bool,
    internal_loss: bool,
    route_override: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    batch, tokens, _ = value.shape
    qkv = functional_linear(value, parameters[f"{prefix}.qkv.weight"]).reshape(batch, tokens, 3, HEADS, HEAD_WIDTH)
    query = rope(qkv[:, :, 0].transpose(0, 2, 1, 3))
    key = rope(qkv[:, :, 1].transpose(0, 2, 1, 3))
    projected_value = qkv[:, :, 2].transpose(0, 2, 1, 3)
    query_input = query.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH)
    key_input = key.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH)
    query_route = routing_normalize(functional_linear(query_input, parameters[f"{prefix}.query_projection.weight"]))[:, :, None]
    key_route = routing_normalize(functional_linear(key_input, parameters[f"{prefix}.key_projection.weight"]))
    blocks = tokens // BLOCK_SIZE
    block_features = routing_normalize(key_route.reshape(batch, blocks, BLOCK_SIZE, 16).mean(axis=2))
    codebooks = parameters[f"{prefix}.codebooks"]
    addresses = assign_addresses(block_features, codebooks)
    raw_remote, probes = searched_remote(query_route, codebooks, block_features, addresses, remote_width, True)
    effective_remote = raw_remote
    if query_only:
        effective_remote = mx.where((mx.arange(tokens) == 126)[None, :, None, None], effective_remote, -1)
    if route_override is not None:
        effective_remote = route_override
    local = (mx.arange(tokens, dtype=mx.int32) // BLOCK_SIZE)[None, :, None, None]
    local = mx.broadcast_to(local, (batch, tokens, 1, 1))
    selected = mx.concatenate((effective_remote, local), axis=-1)
    attended = selected_attention(query, key, projected_value, selected)
    delta = functional_linear(attended.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH), parameters[f"{prefix}.out.weight"])
    if internal_loss:
        block_count = tokens // BLOCK_SIZE
        teacher_logits = mx.einsum("bhtd,bhsd->bhts", mx.stop_gradient(query), mx.stop_gradient(key)).mean(axis=1)
        positions = mx.arange(tokens)
        remote_limits = positions // BLOCK_SIZE
        valid = positions[None, None, :] < (remote_limits * BLOCK_SIZE)[None, :, None]
        active = remote_limits > 0
        safe_valid = valid | ((~active)[None, :, None] & (positions[None, None, :] == 0))
        teacher = mx.softmax(mx.where(safe_valid, teacher_logits, -mx.inf), axis=-1)
        teacher = mx.where(remote_limits[None, :, None] > 0, teacher, 0.0)
        teacher_blocks = teacher.reshape(batch, tokens, block_count, BLOCK_SIZE).sum(axis=-1)
        route_blocks = key_route.reshape(batch, block_count, BLOCK_SIZE, 16).mean(axis=2)
        logits = mx.einsum("btgd,bnd->btgn", query_route, route_blocks)[:, :, 0]
        block_ids = mx.arange(block_count)
        valid_blocks = block_ids[None, :] < remote_limits[:, None]
        safe_blocks = valid_blocks | ((~active)[:, None] & (block_ids[None, :] == 0))
        masked_logits = mx.where(safe_blocks[None], logits, -mx.inf)
        log_probs = masked_logits - mx.logsumexp(masked_logits, axis=-1, keepdims=True)
        log_probs = mx.where(valid_blocks[None], log_probs, 0.0)
        losses = -mx.sum(teacher_blocks * log_probs, axis=-1)
        router_loss = mx.sum(mx.where(active[None], losses, 0.0)) / (batch * mx.sum(active))
    else:
        router_loss = mx.array(0.0)
    return delta, query_route, key_route, router_loss, raw_remote, effective_remote, addresses, probes


def functional_dense(value: mx.array, prefix: str, parameters: Mapping[str, mx.array]) -> mx.array:
    batch, tokens, _ = value.shape
    qkv = functional_linear(value, parameters[f"{prefix}.qkv.weight"]).reshape(batch, tokens, 3, HEADS, HEAD_WIDTH)
    query = rope(qkv[:, :, 0].transpose(0, 2, 1, 3))
    key = rope(qkv[:, :, 1].transpose(0, 2, 1, 3))
    projected_value = qkv[:, :, 2].transpose(0, 2, 1, 3)
    scores = mx.einsum("bhtd,bhsd->bhts", query, key) / math.sqrt(HEAD_WIDTH)
    causal = mx.arange(tokens)[None, :] <= mx.arange(tokens)[:, None]
    weights = mx.softmax(mx.where(causal[None, None], scores, -mx.inf), axis=-1)
    attended = mx.einsum("bhts,bhsd->bhtd", weights, projected_value)
    return functional_linear(attended.transpose(0, 2, 1, 3).reshape(batch, tokens, WIDTH), parameters[f"{prefix}.out.weight"])


def functional_forward(parameters: Mapping[str, mx.array], role: str, tokens: mx.array, internal_loss: bool, route_override: mx.array | None = None) -> tuple[Any, ...]:
    resets = () if role == "rung_two" else RESET_POSITIONS
    hidden = parameters["embed.weight"][tokens]
    query_route = None
    key_route = None
    route_loss = mx.array(0.0)
    raw_routes = []
    effective_routes = []
    address_routes = []
    probe_routes = []
    for index, kind in enumerate(SCHEDULE):
        block_prefix = f"blocks.{index}"
        normalized = functional_norm(hidden, parameters[f"{block_prefix}.n1.weight"])
        mix_prefix = f"{block_prefix}.mix"
        if kind == "recurrent":
            mixed = functional_recurrent(normalized, mix_prefix, parameters, resets)
        elif index == 4 and role == "dense":
            mixed = functional_dense(normalized, mix_prefix, parameters)
        else:
            remote = 0 if index == 0 else {"selected": 2, "all_eligible": 15, "local_only": 0, "rung_two": 0}[role]
            query_only = index == 4 and role != "rung_two"
            mixed, current_query, current_key, current_loss, raw_remote, effective_remote, addresses, probes = functional_routed(
                normalized,
                mix_prefix,
                parameters,
                remote,
                query_only,
                internal_loss and index == 4,
                route_override if index == 4 else None,
            )
            raw_routes.append(raw_remote)
            effective_routes.append(effective_remote)
            address_routes.append(addresses)
            probe_routes.append(probes)
            if index == 4:
                query_route = current_query
                key_route = current_key
                route_loss = current_loss
            elif query_route is None:
                query_route = current_query
                key_route = current_key
        hidden = hidden + mixed
        feature_input = functional_norm(hidden, parameters[f"{block_prefix}.n2.weight"])
        hidden = hidden + functional_feature(feature_input, f"{block_prefix}.mlp", parameters)
    final_hidden = functional_norm(hidden, parameters["nf.weight"])
    logits = functional_linear(final_hidden, parameters["head.weight"])
    return logits, query_route, key_route, route_loss, tuple(raw_routes), tuple(effective_routes), tuple(address_routes), tuple(probe_routes)


def stage_loss(
    train_values: tuple[mx.array, ...],
    frozen_values: tuple[mx.array, ...],
    train_names: tuple[str, ...],
    frozen_names: tuple[str, ...],
    role: str,
    stage: str,
    tokens: mx.array,
    targets: mx.array,
    required_source: mx.array,
    route_override: mx.array | None = None,
) -> tuple[Any, ...]:
    parameters = {name: value for name, value in zip(train_names, train_values)}
    parameters.update({name: value for name, value in zip(frozen_names, frozen_values)})
    logits, query_route, key_route, internal, raw_routes, effective_routes, address_routes, probe_routes = functional_forward(parameters, role, tokens, stage == "joint", route_override)
    if stage == "rung_two":
        task = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean")
        supervised = mx.array(0.0)
        total = task
    else:
        task = nn.losses.cross_entropy(logits[:, 126], targets, reduction="mean")
        if stage in {"router_only", "joint"}:
            route_blocks = key_route[:, :120].reshape(key_route.shape[0], 15, 8, 16).mean(axis=2)
            supervised_logits = mx.einsum("bd,bnd->bn", query_route[:, 126, 0], route_blocks)
            supervised = nn.losses.cross_entropy(supervised_logits, required_source, reduction="mean")
        else:
            supervised = mx.array(0.0)
        total = supervised if stage == "router_only" else task + 0.1 * (internal + supervised) if stage == "joint" else task
    return total, task, internal, supervised, raw_routes, effective_routes, address_routes, query_route, probe_routes


def lane_reduction_axes(value: mx.array) -> tuple[int, ...]:
    return tuple(range(1, value.ndim))


def lane_all_finite(value: mx.array) -> mx.array:
    axes = lane_reduction_axes(value)
    return mx.all(mx.isfinite(value), axis=axes) if axes else mx.isfinite(value)


def lane_any_nonzero(value: mx.array) -> mx.array:
    axes = lane_reduction_axes(value)
    return mx.any(value != 0, axis=axes) if axes else value != 0


def lane_broadcast(value: mx.array, target: mx.array) -> mx.array:
    return value.reshape((value.shape[0],) + (1,) * (target.ndim - 1))


def batched_optimizer_step(
    train_values: tuple[mx.array, ...],
    first_moments: tuple[mx.array, ...],
    second_moments: tuple[mx.array, ...],
    gradients: tuple[mx.array, ...],
    output: tuple[Any, ...],
    learning_rates: tuple[mx.array, ...],
    decay: tuple[float, ...],
    step: mx.array,
    include_clipped_gradients: bool = False,
) -> tuple[Any, ...]:
    norm_squares = tuple(mx.sum(gradient * gradient, axis=lane_reduction_axes(gradient)) for gradient in gradients)
    norm_square = sum(norm_squares[1:], norm_squares[0])
    gradient_norm = mx.sqrt(norm_square)
    scale = mx.minimum(1.0, 1.0 / (gradient_norm + 1e-6))
    clipped_gradients = tuple(gradient * lane_broadcast(scale, gradient) for gradient in gradients)
    next_first = tuple(0.9 * moment + 0.1 * gradient for moment, gradient in zip(first_moments, clipped_gradients))
    next_second = tuple(0.95 * moment + 0.05 * gradient * gradient for moment, gradient in zip(second_moments, clipped_gradients))
    first_correction = 1.0 - 0.9**step
    second_correction = 1.0 - 0.95**step
    updated = tuple(
        (1.0 - rate * weight_decay) * parameter - rate * (first / first_correction) / (mx.sqrt(second / second_correction) + 1e-8)
        for parameter, first, second, rate, weight_decay in zip(train_values, next_first, next_second, learning_rates, decay)
    )
    finite_rows = tuple(lane_all_finite(value) for value in (*updated, *next_first, *next_second, *output[:4], gradient_norm))
    global_finite = mx.all(mx.stack(finite_rows, axis=1), axis=1)
    audit_status = mx.stack(
        tuple(
            mx.stack(
                (
                    lane_all_finite(gradient) & global_finite,
                    lane_any_nonzero(gradient),
                    lane_all_finite(next_value) & lane_all_finite(next_value - prior_value) & global_finite,
                    lane_any_nonzero(next_value - prior_value),
                ),
                axis=1,
            )
            for gradient, next_value, prior_value in zip(gradients, updated, train_values)
        ),
        axis=1,
    )
    result = updated, next_first, next_second, output, gradient_norm, audit_status
    return (*result, gradients, clipped_gradients) if include_clipped_gradients else result


def compiled_stage_step(
    train_names: tuple[str, ...],
    frozen_names: tuple[str, ...],
    decay: tuple[float, ...],
    role: str,
    stage: str,
    include_clipped_gradients: bool = False,
):
    def loss_function(train_values, frozen_values, tokens, targets, required_source):
        return stage_loss(train_values, frozen_values, train_names, frozen_names, role, stage, tokens, targets, required_source)

    value_and_grad = mx.value_and_grad(loss_function)
    vectorized_value_and_grad = mx.vmap(value_and_grad, in_axes=(0, 0, 0, 0, 0))

    def outer_step(train_values, first_moments, second_moments, frozen_values, tokens, targets, required_source, learning_rates, step):
        output, gradients = vectorized_value_and_grad(train_values, frozen_values, tokens, targets, required_source)
        return batched_optimizer_step(train_values, first_moments, second_moments, gradients, output, learning_rates, decay, step, include_clipped_gradients)

    return mx.compile(outer_step)


def compiled_pilot_step(
    train_names: tuple[str, ...],
    frozen_names: tuple[str, ...],
    decay: tuple[float, ...],
    role: str,
    stage: str,
    use_route_override: bool,
):
    def loss_function(train_values, frozen_values, tokens, targets, required_source, route_override):
        result = stage_loss(train_values, frozen_values, train_names, frozen_names, role, stage, tokens, targets, required_source, route_override if use_route_override else None)
        return result[0], result[1], result[2], result[3], result[6]

    value_and_grad = mx.value_and_grad(loss_function)
    vectorized_value_and_grad = mx.vmap(value_and_grad, in_axes=(0, 0, 0, 0, 0, None))

    def outer_step(train_values, first_moments, second_moments, frozen_values, tokens, targets, required_source, route_override, learning_rates, step):
        output, gradients = vectorized_value_and_grad(train_values, frozen_values, tokens, targets, required_source, route_override)
        return batched_optimizer_step(train_values, first_moments, second_moments, gradients, output, learning_rates, decay, step)

    return mx.compile(outer_step)


def torch_policy_name(mlx_name: str) -> str:
    for block in (0, 4):
        prefix = f"blocks.{block}.mix."
        if not mlx_name.startswith(prefix):
            continue
        suffix = mlx_name[len(prefix):]
        if suffix in {"codebooks", "query_projection.weight", "key_projection.weight"}:
            return f"blocks.{block}.mix.source_mixer.attention.router.{suffix}"
        if suffix in {"qkv.weight", "out.weight"}:
            return f"blocks.{block}.mix.source_mixer.attention.{suffix}"
    return mlx_name


def stage_role(stage: str) -> str:
    return {
        "donor": "all_eligible",
        "router_only": "selected",
        "joint": "selected",
        "dense_base": "dense",
        "dense_continuation": "dense",
        "rung_two": "rung_two",
    }[stage]


def stage_multiplier(update: int, updates: int, warmup: int) -> float:
    if update <= warmup:
        return update / warmup
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (update - warmup) / (updates - warmup)))


def initialized_stage_model(stage: str, seed: int, run_root: Path, checkpoint_input: str | None) -> ModularNeuralMachine:
    role = stage_role(stage)
    if stage == "donor":
        canonical = torch_model("selected", seed)
        model = torch_model("all_eligible", seed)
        copy_compatible_state(canonical, model, include_router=True)
        return model
    if stage == "dense_base":
        canonical = torch_model("selected", seed)
        model = torch_model("dense", seed)
        copy_compatible_state(canonical, model, include_router=True)
        return model
    model = torch_model(role, seed)
    if checkpoint_input is not None:
        checkpoint = torch.load(run_root / checkpoint_input, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def stage_parameter_state(request: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    stage = request["stage"]
    role = stage_role(stage)
    inputs = request["checkpoint_inputs"]
    lane_values = []
    torch_models = []
    names = None
    for index, seed in enumerate(request["construction_seeds"]):
        checkpoint_input = None if not inputs else inputs[index]
        torch_instance = initialized_stage_model(stage, seed, run_root, checkpoint_input)
        mlx_instance = MlxModularModel(role)
        load_torch_state(mlx_instance, torch_instance.state_dict())
        flattened = tuple(tree_flatten(mlx_instance.parameters()))
        current_names = tuple(name for name, _ in flattened)
        if names is not None and current_names != names:
            raise MlxEngineError("stage lane parameter order differs")
        names = current_names
        lane_values.append(tuple(value for _, value in flattened))
        torch_models.append(torch_instance)
    if names is None:
        raise MlxEngineError("stage has no parameter lanes")
    policies = {name: optimizer_parameter_policy(torch_policy_name(name), stage) for name in names}
    train_names = tuple(name for name in names if policies[name]["trainable"])
    frozen_names = tuple(name for name in names if not policies[name]["trainable"])
    indexes = {name: index for index, name in enumerate(names)}
    train = tuple(mx.stack([lane[indexes[name]] for lane in lane_values]) for name in train_names)
    frozen = tuple(mx.stack([lane[indexes[name]] for lane in lane_values]) for name in frozen_names)
    first = tuple(mx.zeros_like(value) for value in train)
    second = tuple(mx.zeros_like(value) for value in train)
    decay = tuple(float(policies[name]["weight_decay"]) for name in train_names)
    peak_rates = tuple(float(policies[name]["peak_lr"]) for name in train_names)
    mx.eval(train, frozen, first, second)
    return {
        "role": role,
        "names": names,
        "train_names": train_names,
        "frozen_names": frozen_names,
        "train": train,
        "frozen": frozen,
        "first": first,
        "second": second,
        "decay": decay,
        "peak_rates": peak_rates,
        "torch_models": torch_models,
    }


def stage_batches(request: Mapping[str, Any]) -> tuple[list[torch.Generator], Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    generators = []
    for seed in request["data_generator_seeds"]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        generators.append(generator)

    def draw() -> tuple[list[dict[str, Any]], list[str]]:
        batches = []
        hashes = []
        for generator in generators:
            if request["stage"] == "rung_two":
                batch = cpu._continuous_rung_two_batch(generator, request["batch_size"], torch)
            else:
                batch = cpu._continuous_rung_one_batch(generator, request["batch_size"], torch)
            batches.append(batch)
            hashes.append(cpu._batch_payload_hash(batch, request["stage"]))
        return batches, hashes

    return generators, draw


def mx_stage_batch(batches: list[Mapping[str, Any]], stage: str) -> tuple[mx.array, mx.array, mx.array]:
    tokens = mx.array(np.stack([batch["tokens"].numpy() for batch in batches]))
    targets = mx.array(np.stack([batch["targets"].numpy() for batch in batches]))
    if stage == "rung_two":
        required = mx.zeros((len(batches), batches[0]["tokens"].shape[0]), dtype=mx.int32)
    else:
        required = mx.array(np.stack([batch["required_source"].numpy() for batch in batches]))
    return tokens, targets, required


def tensor_tuple_finite(values: Any) -> bool:
    for _, value in tree_flatten(values):
        if not bool(np.isfinite(np.array(value)).all()):
            return False
    return True


def full_lane_parameters(state: Mapping[str, Any], lane: int) -> dict[str, mx.array]:
    values = {name: value[lane] for name, value in zip(state["train_names"], state["train"])}
    values.update({name: value[lane] for name, value in zip(state["frozen_names"], state["frozen"])})
    if set(values) != set(state["names"]):
        raise MlxEngineError("full lane parameter closure differs")
    return values


def canonical_stage_name(stage: str) -> str:
    return {
        "donor": "donor",
        "router_only": "router",
        "joint": "joint",
        "dense_base": "dense_base",
        "dense_continuation": "dense",
        "rung_two": "rung2",
    }[stage]


def maximum_array_error(observed: Any, expected: Any) -> float:
    left = np.array(observed)
    right = np.array(expected)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise MlxEngineError("endpoint parity tensor descriptor differs")
    if left.size == 0:
        return 0.0
    error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    if not math.isfinite(error):
        raise MlxEngineError("endpoint parity tensor error is nonfinite")
    return error


def tensor_comparison_statistics(comparisons: list[tuple[str, Any, Any]]) -> dict[str, Any]:
    if not comparisons or len({name for name, _, _ in comparisons}) != len(comparisons):
        raise MlxEngineError("tensor comparison identity differs")
    records = []
    for name, observed_value, expected_value in comparisons:
        observed = np.asarray(observed_value, dtype=np.float64)
        expected = np.asarray(expected_value, dtype=np.float64)
        if not isinstance(name, str) or not name or observed.shape != expected.shape or observed.size == 0 or not np.isfinite(observed).all() or not np.isfinite(expected).all():
            raise MlxEngineError("tensor comparison differs")
        difference = observed - expected
        flat_index = int(np.argmax(np.abs(difference)))
        maximum = float(np.abs(difference).reshape(-1)[flat_index])
        expected_scale = float(np.max(np.abs(expected)))
        expected_l2 = float(np.linalg.norm(expected.reshape(-1)))
        observed_l2 = float(np.linalg.norm(observed.reshape(-1)))
        difference_l2 = float(np.linalg.norm(difference.reshape(-1)))
        relative = maximum / max(expected_scale, 1e-30)
        normalized_l2 = difference_l2 / max(expected_l2, 1e-30)
        denominator = observed_l2 * expected_l2
        if denominator == 0.0:
            cosine = 1.0 if difference_l2 == 0.0 else 0.0
        else:
            cosine = min(1.0, max(-1.0, float(np.dot(observed.reshape(-1), expected.reshape(-1)) / denominator)))
        gradient_floor = 1e-8
        active = (np.abs(observed) >= gradient_floor) | (np.abs(expected) >= gradient_floor)
        mismatch_count = int(np.count_nonzero(observed != expected))
        sign_flip_count = int(np.count_nonzero(active & (np.signbit(observed) != np.signbit(expected))))
        records.append(
            {
                "name": name,
                "max_abs": maximum,
                "relative_max": relative,
                "normalized_l2": normalized_l2,
                "cosine": cosine,
                "expected_max_magnitude": expected_scale,
                "expected_l2": expected_l2,
                "observed_l2": observed_l2,
                "difference_l2": difference_l2,
                "cosine_denominator": denominator,
                "gradient_floor": gradient_floor,
                "mismatch_count": mismatch_count,
                "sign_flip_count": sign_flip_count,
                "worst_index": [int(value) for value in np.unravel_index(flat_index, difference.shape)],
                "worst_observed": float(observed.reshape(-1)[flat_index]),
                "worst_expected": float(expected.reshape(-1)[flat_index]),
            }
        )
    metric_worst_records = {
        "max_abs": max(records, key=lambda value: (value["max_abs"], value["name"])),
        "relative_max": max(records, key=lambda value: (value["relative_max"], value["name"])),
        "normalized_l2": max(records, key=lambda value: (value["normalized_l2"], value["name"])),
        "cosine": min(records, key=lambda value: (value["cosine"], value["name"])),
    }
    worst = metric_worst_records["max_abs"]
    gradient_max_abs = max(value["max_abs"] for value in records)
    gradient_relative_max = max(value["relative_max"] for value in records)
    gradient_normalized_l2_max = max(value["normalized_l2"] for value in records)
    gradient_cosine_min = min(value["cosine"] for value in records)
    return {
        "tensor_count": len(records),
        "max_abs": gradient_max_abs,
        "relative_max": gradient_relative_max,
        "normalized_l2_max": gradient_normalized_l2_max,
        "cosine_min": gradient_cosine_min,
        "worst_tensor": worst["name"],
        "worst_index": worst["worst_index"],
        "worst_observed": worst["worst_observed"],
        "worst_expected": worst["worst_expected"],
        "metric_worst": {
            metric: {
                "tensor": record["name"],
                "worst_index": record["worst_index"],
                "worst_observed": record["worst_observed"],
                "worst_expected": record["worst_expected"],
                "expected_max_magnitude": record["expected_max_magnitude"],
                "expected_l2": record["expected_l2"],
                "observed_l2": record["observed_l2"],
                "difference_l2": record["difference_l2"],
                "cosine_denominator": record["cosine_denominator"],
                "gradient_floor": record["gradient_floor"],
                "mismatch_count": record["mismatch_count"],
                "sign_flip_count": record["sign_flip_count"],
                "value": record[metric],
            }
            for metric, record in metric_worst_records.items()
        },
    }


def gradient_comparison_evidence(
    comparisons: list[tuple[str, Any, Any]],
    scale_aware_absolute_tolerance: float = 3e-5,
    relative_tolerance: float = 1e-4,
    normalized_l2_tolerance: float = 5e-5,
    cosine_tolerance: float = 0.999999999,
) -> dict[str, Any]:
    statistics = tensor_comparison_statistics(comparisons)
    gradient_max_abs = statistics["max_abs"]
    gradient_relative_max = statistics["relative_max"]
    gradient_normalized_l2_max = statistics["normalized_l2_max"]
    gradient_cosine_min = statistics["cosine_min"]
    gradient_absolute_pass = gradient_max_abs <= 1e-5
    gradient_scale_aware_pass = gradient_max_abs <= scale_aware_absolute_tolerance and gradient_relative_max <= relative_tolerance and gradient_normalized_l2_max <= normalized_l2_tolerance and gradient_cosine_min >= cosine_tolerance
    return {
        "gradient_count": statistics["tensor_count"],
        "gradient_max_abs": gradient_max_abs,
        "gradient_relative_max": gradient_relative_max,
        "gradient_normalized_l2_max": gradient_normalized_l2_max,
        "gradient_cosine_min": gradient_cosine_min,
        "gradient_worst_tensor": statistics["worst_tensor"],
        "gradient_worst_index": statistics["worst_index"],
        "gradient_worst_observed": statistics["worst_observed"],
        "gradient_worst_expected": statistics["worst_expected"],
        "gradient_metric_worst": statistics["metric_worst"],
        "gradient_absolute_pass": gradient_absolute_pass,
        "gradient_scale_aware_pass": gradient_scale_aware_pass,
        "gradient_pass": gradient_absolute_pass or gradient_scale_aware_pass,
        "gradient_scale_aware_absolute_tolerance": scale_aware_absolute_tolerance,
        "gradient_relative_tolerance": relative_tolerance,
        "gradient_normalized_l2_tolerance": normalized_l2_tolerance,
        "gradient_cosine_tolerance": cosine_tolerance,
    }


def trained_endpoint_parity(
    request: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    lane: int,
    checkpoint_record: Mapping[str, str],
) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    stage = request["stage"]
    seed = request["construction_seeds"][lane]
    checkpoint_path = run_root / checkpoint_record["path"]
    checkpoint_sha256 = cpu.sha256_file(checkpoint_path)
    if checkpoint_sha256 != checkpoint_record["checkpoint_sha256"]:
        raise MlxEngineError("endpoint parity checkpoint identity differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_optimizer_sha256 = hashlib.sha256(cpu._torch_artifact_bytes(checkpoint["optimizer_state_dict"], torch)).hexdigest()
    if loaded_optimizer_sha256 != checkpoint_record["optimizer_state_sha256"]:
        raise MlxEngineError("endpoint parity optimizer checkpoint identity differs")
    reference = torch_model(stage_role(stage), seed)
    reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
    runtime = cpu._import_runtime()
    optimizer, _, membership = cpu._make_optimizer(reference, stage, runtime)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    data_seed = ENDPOINT_PARITY_DATA_SEED_BASE[stage] + seed
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    batch = cpu._continuous_rung_two_batch(generator, 1, torch) if stage == "rung_two" else cpu._continuous_rung_one_batch(generator, 1, torch)
    input_sha256 = cpu._batch_payload_hash(batch, stage)
    train_values = tuple(value[lane] for value in state["train"])
    frozen_values = tuple(value[lane] for value in state["frozen"])
    lane_parameters = full_lane_parameters(state, lane)
    destination = MlxModularModel(stage_role(stage))
    destination.update(tree_unflatten([(name, lane_parameters[name]) for name in state["names"]]))
    tokens = mx.array(batch["tokens"].numpy())
    targets = mx.array(batch["targets"].numpy())
    required_source = mx.zeros((1,), dtype=mx.int32) if stage == "rung_two" else mx.array(batch["required_source"].numpy())

    def loss_function(current_train: tuple[mx.array, ...]) -> mx.array:
        return stage_loss(current_train, frozen_values, state["train_names"], state["frozen_names"], state["role"], stage, tokens, targets, required_source)[0]

    mlx_losses = stage_loss(train_values, frozen_values, state["train_names"], state["frozen_names"], state["role"], stage, tokens, targets, required_source)
    _, mlx_gradients = mx.value_and_grad(loss_function)(train_values)
    mlx_output = destination(tokens, None, stage == "joint", evaluation_telemetry=True)
    mx.eval(mlx_losses, mlx_gradients, mlx_output)
    reference.train()
    torch_total, _, _, _, torch_output = cpu._training_forward_loss(reference, batch, stage, runtime)
    torch_task = torch.nn.functional.cross_entropy(
        torch_output.logits.reshape(-1, torch_output.logits.shape[-1]) if stage == "rung_two" else torch_output.logits[:, 126],
        batch["targets"].reshape(-1) if stage == "rung_two" else batch["targets"],
        reduction="mean",
        label_smoothing=0.0,
    )
    torch_supervised = cpu._supervised_route_loss(torch_output, batch["required_source"], runtime) if stage in {"router_only", "joint"} else None
    torch_internal = cpu._block_output(torch_output, 4).router_loss if stage == "joint" else None
    torch_total.backward()
    parameter_errors = []
    parameter_exact = True
    for torch_name, tensor in checkpoint["model_state_dict"].items():
        observed = np.array(lane_parameters[mapped_mlx_parameter_name(torch_name)])
        expected = tensor.detach().cpu().numpy()
        parameter_errors.append(maximum_array_error(observed, expected))
        parameter_exact = parameter_exact and np.array_equal(observed, expected)
    comparison_positions = [510] if stage == "rung_two" else [126]
    full_tensor_logits_max_abs = maximum_array_error(mlx_output[0], torch_output.logits.detach().cpu().numpy())
    full_tensor_hidden_max_abs = maximum_array_error(mlx_output[1], torch_output.hidden.detach().cpu().numpy())
    full_tensor_sequence_delta_max_abs = max((maximum_array_error(observed, expected.computed_sequence_delta.detach().cpu().numpy()) for observed, expected in zip(mlx_output[7], torch_output.blocks)), default=0.0)
    if len(mlx_output[7]) != len(torch_output.blocks):
        raise MlxEngineError("endpoint parity sequence delta closure differs")
    logits_max_abs = maximum_array_error(mlx_output[0][:, comparison_positions], torch_output.logits[:, comparison_positions].detach().cpu().numpy())
    hidden_max_abs = maximum_array_error(mlx_output[1][:, comparison_positions], torch_output.hidden[:, comparison_positions].detach().cpu().numpy())
    sequence_delta_max_abs = max((maximum_array_error(observed[:, comparison_positions], expected.computed_sequence_delta[:, comparison_positions].detach().cpu().numpy()) for observed, expected in zip(mlx_output[7], torch_output.blocks)), default=0.0)
    total_loss_max_abs = abs(float(mlx_losses[0].item()) - float(torch_total.detach()))
    component_loss_errors = {"task": abs(float(mlx_losses[1].item()) - float(torch_task.detach()))}
    if stage == "joint":
        component_loss_errors["internal_router"] = abs(float(mlx_losses[2].item()) - float(torch_internal.detach()))
    if stage in {"router_only", "joint"}:
        component_loss_errors["supervised_route"] = abs(float(mlx_losses[3].item()) - float(torch_supervised.detach()))
    if not all(math.isfinite(value) for value in (total_loss_max_abs, *component_loss_errors.values())):
        raise MlxEngineError("endpoint parity loss error is nonfinite")
    torch_routed = [block.mixer_output for block in torch_output.blocks if block.kind == "routed"]
    raw_route_exact = len(torch_routed) == len(mlx_output[5]) and all(np.array_equal(np.array(observed), expected.telemetry["raw_remote"].numpy()) for observed, expected in zip(mlx_output[5], torch_routed))
    effective_route_exact = len(torch_routed) == len(mlx_output[6]) and all(np.array_equal(np.array(observed), expected.telemetry["effective_remote"].numpy()) for observed, expected in zip(mlx_output[6], torch_routed))
    address_route_exact = len(torch_routed) == len(mlx_output[9]) and all(np.array_equal(np.array(observed), expected.telemetry["block_addresses"].numpy()) for observed, expected in zip(mlx_output[9], torch_routed))
    gradient_comparisons = []
    gradient_none_zero_exact = True
    gradient_by_name = {name: np.array(value) for name, value in zip(state["train_names"], mlx_gradients)}
    for name, parameter in reference.named_parameters():
        mlx_name = mapped_mlx_parameter_name(name)
        if membership[name]["requires_grad"]:
            observed = gradient_by_name[mlx_name]
            if parameter.grad is None:
                gradient_none_zero_exact = gradient_none_zero_exact and np.array_equal(observed, np.zeros_like(observed))
                expected = np.zeros_like(observed)
            else:
                expected = parameter.grad.detach().cpu().numpy()
            gradient_comparisons.append((name, observed, expected))
        elif parameter.grad is not None or mlx_name in gradient_by_name:
            gradient_none_zero_exact = False
    gradient_evidence = gradient_comparison_evidence(gradient_comparisons)
    train_indexes = {name: index for index, name in enumerate(state["train_names"])}
    first_errors = []
    second_errors = []
    optimizer_step_exact = True
    optimizer_parameter_identity_exact = True
    optimizer_state_count = 0
    for name, parameter in reference.named_parameters():
        mlx_name = mapped_mlx_parameter_name(name)
        if mlx_name in train_indexes:
            optimizer_state_count += 1
            optimizer_value = optimizer.state.get(parameter)
            if not isinstance(optimizer_value, Mapping) or set(optimizer_value) != {"step", "exp_avg", "exp_avg_sq"}:
                optimizer_parameter_identity_exact = False
                continue
            index = train_indexes[mlx_name]
            first_errors.append(maximum_array_error(state["first"][index][lane], optimizer_value["exp_avg"].detach().cpu().numpy()))
            second_errors.append(maximum_array_error(state["second"][index][lane], optimizer_value["exp_avg_sq"].detach().cpu().numpy()))
            optimizer_step_exact = optimizer_step_exact and float(optimizer_value["step"]) == float(request["updates"])
        elif parameter in optimizer.state:
            optimizer_parameter_identity_exact = False
    optimizer_parameter_identity_exact = optimizer_parameter_identity_exact and optimizer_state_count == len(state["train_names"]) and len(optimizer.state) == len(state["train_names"])
    parameter_max_abs = max(parameter_errors, default=0.0)
    component_loss_max_abs = max(component_loss_errors.values(), default=0.0)
    gradient_max_abs = gradient_evidence["gradient_max_abs"]
    optimizer_first_moment_max_abs = max(first_errors, default=0.0)
    optimizer_second_moment_max_abs = max(second_errors, default=0.0)
    comparison_tolerance = 1e-5
    logit_loss_gradient_tolerance = 1e-5
    optimizer_tolerance = 0.0
    max_error = max(logits_max_abs, hidden_max_abs, sequence_delta_max_abs, total_loss_max_abs, component_loss_max_abs, gradient_max_abs, optimizer_first_moment_max_abs, optimizer_second_moment_max_abs)
    passed = max(logits_max_abs, hidden_max_abs, sequence_delta_max_abs) <= comparison_tolerance
    passed = passed and logits_max_abs <= logit_loss_gradient_tolerance and max(total_loss_max_abs, component_loss_max_abs) <= 1e-6 and gradient_evidence["gradient_pass"] and gradient_none_zero_exact
    passed = passed and optimizer_first_moment_max_abs <= optimizer_tolerance and optimizer_second_moment_max_abs <= optimizer_tolerance
    passed = passed and optimizer_step_exact and optimizer_parameter_identity_exact and parameter_exact and parameter_max_abs == 0.0
    passed = passed and raw_route_exact and effective_route_exact and address_route_exact
    record = {
        "schema_version": cpu.SCHEMA_VERSION,
        "run_id": run_root.name,
        "rung": 2 if stage == "rung_two" else 1,
        "construction_seed": seed,
        "execution_stage": stage,
        "checkpoint_stage": checkpoint["stage"],
        "checkpoint_model": checkpoint["model"],
        "completed_update": checkpoint["completed_update"],
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_state_sha256": checkpoint_record["optimizer_state_sha256"],
        "data_seed": data_seed,
        "input_sha256": input_sha256,
        "comparison_positions": comparison_positions,
        "parameter_count": len(parameter_errors),
        "parameter_max_abs": parameter_max_abs,
        "parameter_exact": bool(parameter_exact),
        "logits_max_abs": logits_max_abs,
        "hidden_max_abs": hidden_max_abs,
        "sequence_delta_max_abs": sequence_delta_max_abs,
        "full_tensor_logits_max_abs": full_tensor_logits_max_abs,
        "full_tensor_hidden_max_abs": full_tensor_hidden_max_abs,
        "full_tensor_sequence_delta_max_abs": full_tensor_sequence_delta_max_abs,
        "total_loss_max_abs": total_loss_max_abs,
        "component_loss_max_abs": component_loss_max_abs,
        "component_loss_errors": component_loss_errors,
        "gradient_count": gradient_evidence["gradient_count"],
        "gradient_max_abs": gradient_evidence["gradient_max_abs"],
        "gradient_relative_max": gradient_evidence["gradient_relative_max"],
        "gradient_normalized_l2_max": gradient_evidence["gradient_normalized_l2_max"],
        "gradient_cosine_min": gradient_evidence["gradient_cosine_min"],
        "gradient_worst_tensor": gradient_evidence["gradient_worst_tensor"],
        "gradient_worst_index": gradient_evidence["gradient_worst_index"],
        "gradient_worst_observed": gradient_evidence["gradient_worst_observed"],
        "gradient_worst_expected": gradient_evidence["gradient_worst_expected"],
        "gradient_absolute_pass": gradient_evidence["gradient_absolute_pass"],
        "gradient_scale_aware_pass": gradient_evidence["gradient_scale_aware_pass"],
        "gradient_pass": gradient_evidence["gradient_pass"],
        "gradient_scale_aware_absolute_tolerance": 3e-5,
        "gradient_relative_tolerance": 1e-4,
        "gradient_normalized_l2_tolerance": 5e-5,
        "gradient_cosine_tolerance": 0.999999999,
        "gradient_none_zero_exact": bool(gradient_none_zero_exact),
        "optimizer_state_count": optimizer_state_count,
        "optimizer_first_moment_max_abs": optimizer_first_moment_max_abs,
        "optimizer_second_moment_max_abs": optimizer_second_moment_max_abs,
        "optimizer_step_exact": bool(optimizer_step_exact),
        "optimizer_parameter_identity_exact": bool(optimizer_parameter_identity_exact),
        "raw_route_exact": bool(raw_route_exact),
        "effective_route_exact": bool(effective_route_exact),
        "address_route_exact": bool(address_route_exact),
        "comparison_tolerance": 1e-5,
        "logit_loss_gradient_tolerance": 1e-5,
        "loss_tolerance": 1e-6,
        "optimizer_tolerance": 0.0,
        "max_error": max_error,
        "pass": bool(passed),
    }
    validate_trained_endpoint_parity_record(record)
    if record["pass"] is not True:
        raise MlxEngineError("trained endpoint MLX and Torch parity differs")
    return record


def endpoint_checkpoint_path(stage: str, seed: int) -> str:
    suffix = {
        "donor": "donor_last.pt",
        "router_only": "router_last.pt",
        "joint": "final_last.pt",
        "dense_base": "dense_base_last.pt",
        "dense_continuation": "dense_last.pt",
        "rung_two": "final_last.pt",
    }[stage]
    return f"rung2/83/checkpoints/{suffix}" if stage == "rung_two" else f"rung1/{seed}/checkpoints/{suffix}"


def validate_trained_endpoint_parity_record(record: Mapping[str, Any]) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    expected_keys = {
        "schema_version", "run_id", "rung", "construction_seed", "execution_stage", "checkpoint_stage", "checkpoint_model", "completed_update", "checkpoint_path", "checkpoint_sha256", "optimizer_state_sha256", "data_seed", "input_sha256", "comparison_positions", "parameter_count", "parameter_max_abs", "parameter_exact", "logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "full_tensor_logits_max_abs", "full_tensor_hidden_max_abs", "full_tensor_sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "component_loss_errors", "gradient_count", "gradient_max_abs", "gradient_relative_max", "gradient_normalized_l2_max", "gradient_cosine_min", "gradient_worst_tensor", "gradient_worst_index", "gradient_worst_observed", "gradient_worst_expected", "gradient_absolute_pass", "gradient_scale_aware_pass", "gradient_pass", "gradient_scale_aware_absolute_tolerance", "gradient_relative_tolerance", "gradient_normalized_l2_tolerance", "gradient_cosine_tolerance", "gradient_none_zero_exact", "optimizer_state_count", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs", "optimizer_step_exact", "optimizer_parameter_identity_exact", "raw_route_exact", "effective_route_exact", "address_route_exact", "comparison_tolerance", "logit_loss_gradient_tolerance", "loss_tolerance", "optimizer_tolerance", "max_error", "pass",
    }
    if set(record) != expected_keys:
        raise MlxEngineError("trained endpoint parity record keys differ")
    stage = record["execution_stage"]
    seed = record["construction_seed"]
    expected_rung = 2 if stage == "rung_two" else 1
    expected_seed = 83 if stage == "rung_two" else seed
    if stage not in ENDPOINT_STAGE_ORDER or type(seed) is not int or seed != expected_seed or record["rung"] != expected_rung:
        raise MlxEngineError("trained endpoint parity stage identity differs")
    expected_checkpoint_stages = {"donor": "donor", "router_only": "router", "joint": "joint", "dense_base": "dense_base", "dense_continuation": "dense", "rung_two": "rung2"}
    expected_models = {"donor": "all_eligible_donor", "router_only": "selected", "joint": "selected", "dense_base": "dense_causal", "dense_continuation": "dense_causal", "rung_two": "rung_two"}
    if record["schema_version"] != cpu.SCHEMA_VERSION or not isinstance(record["run_id"], str) or not record["run_id"] or record["checkpoint_stage"] != expected_checkpoint_stages[stage] or record["checkpoint_model"] != expected_models[stage]:
        raise MlxEngineError("trained endpoint parity checkpoint metadata differs")
    if type(record["completed_update"]) is not int or record["completed_update"] <= 0:
        raise MlxEngineError("trained endpoint parity completed update differs")
    if record["checkpoint_path"] != endpoint_checkpoint_path(stage, seed) or record["data_seed"] != ENDPOINT_PARITY_DATA_SEED_BASE[stage] + seed:
        raise MlxEngineError("trained endpoint parity input identity differs")
    if record["comparison_positions"] != ([510] if stage == "rung_two" else [126]):
        raise MlxEngineError("trained endpoint parity comparison positions differ")
    if any(not isinstance(record[name], str) or len(record[name]) != 64 or any(character not in "0123456789abcdef" for character in record[name]) for name in ("checkpoint_sha256", "optimizer_state_sha256", "input_sha256")):
        raise MlxEngineError("trained endpoint parity digest differs")
    numeric_names = ("parameter_max_abs", "logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "full_tensor_logits_max_abs", "full_tensor_hidden_max_abs", "full_tensor_sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "gradient_max_abs", "gradient_relative_max", "gradient_normalized_l2_max", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs", "max_error")
    if any(isinstance(record[name], bool) or not isinstance(record[name], (int, float)) or not math.isfinite(float(record[name])) or float(record[name]) < 0.0 for name in numeric_names):
        raise MlxEngineError("trained endpoint parity numeric evidence differs")
    signed_gradient_names = ("gradient_cosine_min", "gradient_worst_observed", "gradient_worst_expected")
    if any(isinstance(record[name], bool) or not isinstance(record[name], (int, float)) or not math.isfinite(float(record[name])) for name in signed_gradient_names) or not -1.0 <= record["gradient_cosine_min"] <= 1.0:
        raise MlxEngineError("trained endpoint parity signed gradient evidence differs")
    if not isinstance(record["gradient_worst_tensor"], str) or not record["gradient_worst_tensor"] or not isinstance(record["gradient_worst_index"], list) or not record["gradient_worst_index"] or any(type(value) is not int or value < 0 for value in record["gradient_worst_index"]):
        raise MlxEngineError("trained endpoint parity worst gradient identity differs")
    if record["comparison_tolerance"] != 1e-5 or record["logit_loss_gradient_tolerance"] != 1e-5 or record["loss_tolerance"] != 1e-6 or record["optimizer_tolerance"] != 0.0 or record["gradient_scale_aware_absolute_tolerance"] != 3e-5 or record["gradient_relative_tolerance"] != 1e-4 or record["gradient_normalized_l2_tolerance"] != 5e-5 or record["gradient_cosine_tolerance"] != 0.999999999:
        raise MlxEngineError("trained endpoint parity tolerance differs")
    component_keys = {"task"}
    if stage in {"router_only", "joint"}:
        component_keys.add("supervised_route")
    if stage == "joint":
        component_keys.add("internal_router")
    component_errors = record["component_loss_errors"]
    if not isinstance(component_errors, Mapping) or set(component_errors) != component_keys or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0 for value in component_errors.values()) or record["component_loss_max_abs"] != max(component_errors.values()):
        raise MlxEngineError("trained endpoint parity component loss evidence differs")
    expected_parameters = AUDIT_MEMBERSHIP_CONTRACT[stage]
    if record["parameter_count"] != expected_parameters["total"] or record["gradient_count"] != expected_parameters["trainable"] or record["optimizer_state_count"] != expected_parameters["trainable"]:
        raise MlxEngineError("trained endpoint parity parameter cardinality differs")
    gradient_absolute_pass = record["gradient_max_abs"] <= record["logit_loss_gradient_tolerance"]
    gradient_scale_aware_pass = record["gradient_max_abs"] <= record["gradient_scale_aware_absolute_tolerance"] and record["gradient_relative_max"] <= record["gradient_relative_tolerance"] and record["gradient_normalized_l2_max"] <= record["gradient_normalized_l2_tolerance"] and record["gradient_cosine_min"] >= record["gradient_cosine_tolerance"]
    if type(record["gradient_absolute_pass"]) is not bool or type(record["gradient_scale_aware_pass"]) is not bool or type(record["gradient_pass"]) is not bool or record["gradient_absolute_pass"] is not gradient_absolute_pass or record["gradient_scale_aware_pass"] is not gradient_scale_aware_pass or record["gradient_pass"] is not (gradient_absolute_pass or gradient_scale_aware_pass):
        raise MlxEngineError("trained endpoint parity gradient decision differs")
    exact_names = ("parameter_exact", "gradient_none_zero_exact", "gradient_pass", "optimizer_step_exact", "optimizer_parameter_identity_exact", "raw_route_exact", "effective_route_exact", "address_route_exact", "pass")
    failed_exact = [name for name in exact_names if record[name] is not True]
    if failed_exact:
        numeric = ",".join(f"{name}={record[name]}" for name in ("logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "full_tensor_logits_max_abs", "full_tensor_hidden_max_abs", "full_tensor_sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "gradient_max_abs", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs"))
        raise MlxEngineError(f"trained endpoint parity exact evidence differs: {','.join(failed_exact)}; {numeric}")
    expected_max = max(record[name] for name in ("logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "gradient_max_abs", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs"))
    threshold_pass = max(record["logits_max_abs"], record["hidden_max_abs"], record["sequence_delta_max_abs"]) <= 1e-5 and max(record["total_loss_max_abs"], record["component_loss_max_abs"]) <= record["loss_tolerance"] and record["gradient_pass"]
    threshold_pass = threshold_pass and max(record["optimizer_first_moment_max_abs"], record["optimizer_second_moment_max_abs"]) == 0.0
    if record["max_error"] != expected_max or not threshold_pass or record["parameter_max_abs"] != 0.0:
        raise MlxEngineError("trained endpoint parity bound differs")


def validate_trained_endpoint_parity_records(records: list[Mapping[str, Any]], expected_count: int) -> None:
    if len(records) != expected_count:
        raise MlxEngineError("trained endpoint parity record cardinality differs")
    for record in records:
        validate_trained_endpoint_parity_record(record)
    identities = [(record["execution_stage"], record["construction_seed"]) for record in records]
    expected = [(stage, seed) for stage in ENDPOINT_STAGE_ORDER[:-1] for seed in RUNG_ONE_SEEDS] + [("rung_two", 83)]
    if expected_count == 26:
        if sorted(identities, key=lambda value: (ENDPOINT_STAGE_ORDER.index(value[0]), value[1])) != expected:
            raise MlxEngineError("trained endpoint parity endpoint closure differs")
        expected_updates = {"donor": 1024, "router_only": 768, "joint": 512, "dense_base": 1024, "dense_continuation": 512, "rung_two": 1536}
        if any(record["completed_update"] != expected_updates[record["execution_stage"]] for record in records):
            raise MlxEngineError("trained endpoint parity production update closure differs")


def save_stage_checkpoints(
    request: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    generators: list[torch.Generator],
    final_hashes: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    results = []
    parity_records = []
    for lane, seed in enumerate(request["construction_seeds"]):
        model = state["torch_models"][lane]
        model_state = model.state_dict()
        lane_parameters = full_lane_parameters(state, lane)
        for torch_name, template in model_state.items():
            mlx_name = mapped_mlx_parameter_name(torch_name)
            observed = lane_parameters[mlx_name]
            array = np.array(observed)
            if tuple(array.shape) != tuple(template.shape) or array.dtype != np.float32:
                raise MlxEngineError("checkpoint tensor shape or dtype differs")
            model_state[torch_name] = torch.from_numpy(array.copy())
        model.load_state_dict(model_state, strict=True)
        runtime = cpu._import_runtime()
        optimizer, _, _ = cpu._make_optimizer(model, request["stage"], runtime)
        train_indexes = {name: index for index, name in enumerate(state["train_names"])}
        for name, parameter in model.named_parameters():
            mlx_name = mapped_mlx_parameter_name(name)
            if mlx_name not in train_indexes:
                continue
            index = train_indexes[mlx_name]
            optimizer.state[parameter] = {
                "step": torch.tensor(float(request["updates"]), dtype=torch.float32),
                "exp_avg": torch.from_numpy(np.array(state["first"][index][lane]).copy()),
                "exp_avg_sq": torch.from_numpy(np.array(state["second"][index][lane]).copy()),
            }
        path = run_root / request["checkpoint_outputs"][lane]
        last_id = cpu.attempt_id(run_root.name, 2 if request["stage"] == "rung_two" else 1, seed, "rung_two" if request["stage"] == "rung_two" else "dense_causal" if request["stage"].startswith("dense") else "all_eligible_donor" if request["stage"] == "donor" else "selected", request["stage"], request["updates"])
        checkpoint = {
            "schema_version": cpu.SCHEMA_VERSION,
            "run_id": run_root.name,
            "rung": 2 if request["stage"] == "rung_two" else 1,
            "construction_seed": seed,
            "model": "rung_two" if request["stage"] == "rung_two" else "dense_causal" if request["stage"].startswith("dense") else "all_eligible_donor" if request["stage"] == "donor" else "selected",
            "stage": canonical_stage_name(request["stage"]),
            "completed_update": request["updates"],
            "last_attempt_id": last_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": {"kind": "linear_warmup_then_cosine_to_tenth_peak", "updates": request["updates"], "warmup_updates": request["warmup_updates"], "completed_update": request["updates"]},
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "generator_states": {f"{request['stage']}_data": generators[lane].get_state()},
            "final_batch_sha256": final_hashes[lane],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu._save_torch_artifact(path, checkpoint, torch)
        checkpoint_sha = cpu.sha256_file(path)
        optimizer_sha = hashlib.sha256(cpu._torch_artifact_bytes(checkpoint["optimizer_state_dict"], torch)).hexdigest()
        rng_sha = hashlib.sha256(cpu._torch_artifact_bytes({"python": checkpoint["python_rng_state"], "torch": checkpoint["torch_rng_state"], "generators": checkpoint["generator_states"]}, torch)).hexdigest()
        checkpoint_record = {"path": request["checkpoint_outputs"][lane], "checkpoint_sha256": checkpoint_sha, "optimizer_state_sha256": optimizer_sha, "rng_state_sha256": rng_sha}
        results.append(checkpoint_record)
        parity_records.append(trained_endpoint_parity(request, run_root, state, lane, checkpoint_record))
    return results, parity_records


def initialize_stage_audits(request: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    audits = []
    inverse = {}
    contract = AUDIT_MEMBERSHIP_CONTRACT[request["stage"]]
    for model in state["torch_models"]:
        _, _, membership = cpu._make_optimizer(model, request["stage"], cpu._import_runtime())
        groups = {}
        for name, record in membership.items():
            if record["requires_grad"]:
                groups[record["parameter_group"]] = groups.get(record["parameter_group"], 0) + 1
            expected_policy = {"trainable": record["requires_grad"], "peak_lr": record["peak_lr"], "weight_decay": record["weight_decay"]}
            if optimizer_parameter_policy(name, request["stage"]) != expected_policy:
                raise MlxEngineError("gradient audit optimizer policy differs")
        if len(membership) != contract["total"] or sum(record["requires_grad"] for record in membership.values()) != contract["trainable"] or groups != contract["groups"]:
            raise MlxEngineError("gradient audit membership differs")
        audits.append(cpu._initialize_audit(model, request["stage"], cpu._import_runtime(), membership))
        for name in membership:
            mlx_name = mapped_mlx_parameter_name(name)
            if mlx_name in inverse and inverse[mlx_name] != name:
                raise MlxEngineError("gradient audit parameter mapping differs")
            inverse[mlx_name] = name
    if set(inverse) != set(state["names"]):
        raise MlxEngineError("gradient audit parameter closure differs")
    torch_names = tuple(inverse[name] for name in state["names"])
    train_positions = np.array([state["names"].index(name) for name in state["train_names"]], dtype=np.int64)
    frozen_positions = np.array([state["names"].index(name) for name in state["frozen_names"]], dtype=np.int64)
    if len(train_positions) != contract["trainable"] or len(frozen_positions) != contract["total"] - contract["trainable"]:
        raise MlxEngineError("gradient audit trainable closure differs")
    tracker = {
        "torch_names": torch_names,
        "train_positions": train_positions,
        "frozen_positions": frozen_positions,
        "counters": np.zeros((len(audits), len(torch_names), 7), dtype=np.int64),
        "first_nonzero": np.zeros((len(audits), len(torch_names)), dtype=np.int64),
    }
    return audits, tracker


def update_stage_audits(tracker: Mapping[str, Any], status: np.ndarray, logical_update: int) -> None:
    counters = tracker["counters"]
    train_positions = tracker["train_positions"]
    frozen_positions = tracker["frozen_positions"]
    if status.shape != (counters.shape[0], len(train_positions), 4) or status.dtype != np.bool_:
        raise MlxEngineError("gradient audit device status differs")
    gradient_finite = status[:, :, 0]
    gradient_nonzero = status[:, :, 1]
    update_finite = status[:, :, 2]
    update_nonzero = status[:, :, 3]
    counters[:, frozen_positions, 0] += 1
    counters[:, train_positions, 1] += gradient_finite & ~gradient_nonzero
    counters[:, train_positions, 2] += gradient_finite & gradient_nonzero
    counters[:, train_positions, 3] += ~gradient_finite
    counters[:, train_positions, 4] += update_finite & ~update_nonzero
    counters[:, train_positions, 5] += update_finite & update_nonzero
    counters[:, train_positions, 6] += ~update_finite
    first = tracker["first_nonzero"]
    unseen = first[:, train_positions] == 0
    lane_indexes, train_indexes = np.nonzero(unseen & gradient_finite & gradient_nonzero)
    first[lane_indexes, train_positions[train_indexes]] = logical_update
    if not bool(np.all(gradient_finite)) or not bool(np.all(update_finite)):
        raise MlxEngineError("gradient audit observed a nonfinite value")


def finalize_stage_audits(request: Mapping[str, Any], state: Mapping[str, Any], audits: list[dict[str, Any]], tracker: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    results = {}
    counter_fields = ("grad_none_steps", "grad_zero_steps", "grad_nonzero_steps", "grad_nonfinite_steps", "update_zero_steps", "update_nonzero_steps", "update_nonfinite_steps")
    for lane, seed in enumerate(request["construction_seeds"]):
        for position, name in enumerate(tracker["torch_names"]):
            record = audits[lane][name]
            for field, value in zip(counter_fields, tracker["counters"][lane, position]):
                record[field] = int(value)
            first = int(tracker["first_nonzero"][lane, position])
            record["first_nonzero_step"] = first or None
        records = cpu._finalize_audit(audits[lane], state["torch_models"][lane], "rung_two" if request["stage"] == "rung_two" else "dense_causal" if request["stage"].startswith("dense") else "all_eligible_donor" if request["stage"] == "donor" else "selected", torch)
        cpu.validate_gradient_audit(records, {request["stage"]: request["updates"]})
        results[seed] = records
    return results


def write_qualification_gradient_artifacts(run_root: Path, stage_audits: Mapping[str, Mapping[int, list[dict[str, Any]]]]) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    expected_stages = ("donor", "router_only", "joint", "dense_base", "dense_continuation", "rung_two")
    if tuple(stage_audits) != expected_stages:
        raise MlxEngineError("qualification gradient stage closure differs")
    for seed in RUNG_ONE_SEEDS:
        if any(seed not in stage_audits[stage] for stage in expected_stages[:-1]):
            raise MlxEngineError("qualification gradient seed closure differs")
        selected = sorted([record for stage in expected_stages[:3] for record in stage_audits[stage][seed]], key=lambda record: (record["stage"], record["name"]))
        dense = sorted([record for stage in expected_stages[3:5] for record in stage_audits[stage][seed]], key=lambda record: (record["stage"], record["name"]))
        if len(selected) != 357 or len(dense) != 232:
            raise MlxEngineError("qualification gradient record cardinality differs")
        cpu.validate_gradient_audit(selected, {"donor": 1024, "router_only": 768, "joint": 512})
        cpu.validate_gradient_audit(dense, {"dense_base": 1024, "dense_continuation": 512})
        seed_root = run_root / "rung1" / str(seed)
        cpu.write_canonical_json(seed_root / "grad_audit.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": selected})
        cpu.write_canonical_json(seed_root / "dense_grad_audit.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": dense})
    rung_two = sorted(stage_audits["rung_two"].get(83, []), key=lambda record: (record["stage"], record["name"]))
    if len(rung_two) != 119:
        raise MlxEngineError("qualification rung-two gradient cardinality differs")
    cpu.validate_gradient_audit(rung_two, {"rung_two": 1536})
    seed_root = run_root / "rung2" / "83"
    cpu.write_canonical_json(seed_root / "grad_audit.json", {"schema_version": cpu.SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "records": rung_two})


def require_parent_ack(response: Mapping[str, Any], expected_kind: str, stage: str, logical_update: int | None) -> None:
    expected = {"ack", "kind", "stage", "logical_update"}
    if not isinstance(response, Mapping) or set(response) != expected:
        raise MlxEngineError("parent acknowledgement keys differ")
    if response["ack"] is not True or response["kind"] != expected_kind or response["stage"] != stage or response["logical_update"] != logical_update:
        raise MlxEngineError("parent acknowledgement identity differs")


def routing_reduction(address_routes: tuple[mx.array, ...], lane: int) -> tuple[int, int]:
    overflow = 0
    maximum = 0
    for values in address_routes:
        addresses = np.array(values[lane])
        for row in addresses:
            counts = np.bincount(row, minlength=16)
            maximum = max(maximum, int(counts.max(initial=0)))
            overflow += int(np.maximum(counts - 64, 0).sum())
    return overflow, maximum


def routing_telemetry(raw_remote: mx.array, effective_remote: mx.array, addresses: mx.array, probes: mx.array) -> dict[str, Any]:
    raw = np.array(raw_remote)
    effective = np.array(effective_remote)
    block_addresses = np.array(addresses)
    probe_values = np.array(probes)
    batch, tokens, groups, selected_width = raw.shape
    block_count = block_addresses.shape[1]
    address_ids = np.arange(16, dtype=np.int64)[None, :, None]
    block_ids = np.arange(block_count, dtype=np.int64)[None, None, :]
    matches = block_addresses[:, None, :] == address_ids
    load_array = matches.sum(axis=-1, dtype=np.int64)
    ordered = np.sort(np.where(matches, block_ids, block_count), axis=-1)
    postings = np.full((batch, 16, 64), -1, dtype=np.int64)
    postings[:, :, :block_count] = np.where(ordered < block_count, ordered, -1)
    observed_loads, load_counts = np.unique(load_array.reshape(-1), return_counts=True)
    block_load_histogram = torch.tensor(np.stack((observed_loads, load_counts), axis=1), dtype=torch.long)
    maximum = int(load_array.max())
    overflow = int(np.maximum(load_array - 64, 0).sum())
    if selected_width == 0:
        valid_posting_histogram = torch.empty((0, 2), dtype=torch.long)
        search_rows = 0
        addresses_probed = 0
        posting_reads = 0
        workspace_bytes = 0
        posting_slots = 0
    else:
        batch_ids = np.arange(batch)[:, None, None, None]
        candidates = postings[batch_ids, probe_values]
        remote_limits = np.arange(tokens, dtype=np.int64) // BLOCK_SIZE
        valid_counts = ((candidates >= 0) & (candidates < remote_limits[None, :, None, None, None])).sum(axis=(-1, -2))
        searched = valid_counts[:, remote_limits > selected_width].reshape(-1)
        observed_valid, valid_counts_frequency = np.unique(searched.astype(np.int64), return_counts=True)
        valid_posting_histogram = torch.tensor(np.stack((observed_valid, valid_counts_frequency), axis=1), dtype=torch.long)
        search_rows = int(searched.size)
        addresses_probed = search_rows * 4
        posting_reads = int(searched.sum())
        peak_slots = batch * min(tokens, 128) * groups * 4 * 64
        workspace_bytes = peak_slots * (8 + 4 + 4 * 16)
        posting_slots = batch * tokens * groups * 4 * 64
    return {
        "raw_remote": torch.from_numpy(raw.copy()).to(torch.long),
        "effective_remote": torch.from_numpy(effective.copy()).to(torch.long),
        "block_features": torch.zeros((batch, tokens // BLOCK_SIZE, 16), dtype=torch.float32),
        "block_addresses": torch.from_numpy(block_addresses.copy()).to(torch.long),
        "postings": torch.from_numpy(postings.copy()).to(torch.long),
        "block_load_histogram": block_load_histogram,
        "valid_posting_histogram": valid_posting_histogram,
        "addresses_probed": addresses_probed,
        "postings_read": posting_reads,
        "candidate_blocks": posting_reads,
        "overflow_count": overflow,
        "max_bucket_load": maximum,
        "workspace_bytes": workspace_bytes,
        "posting_slots_materialized": posting_slots,
        "search_rows": search_rows,
        "bypass_rows": batch * tokens * groups - search_rows,
    }


def capture_training_routing_payload(
    run_root: Path,
    stage: str,
    construction_seed: int,
    logical_update: int,
    forward_sequence: int,
    required_source: torch.Tensor,
    output: tuple[Any, ...],
    lane: int,
) -> dict[str, Any]:
    role = stage_role(stage)
    model = "dense_causal" if stage.startswith("dense") else "all_eligible_donor" if stage == "donor" else "selected"
    block_indexes = (0,) if role == "dense" else (0, 4)
    blocks = []
    for route_index, block_index in enumerate(block_indexes):
        try:
            route_shape = tuple(int(value) for value in output[4][route_index].shape[1:])
            raw = np.zeros(route_shape, dtype=np.int32) if route_shape[-1] == 0 else np.array(output[4][route_index][lane]).copy()
            effective = np.zeros(route_shape, dtype=np.int32) if route_shape[-1] == 0 else np.array(output[5][route_index][lane]).copy()
            addresses = np.array(output[6][route_index][lane]).copy()
            probes = np.array(output[8][route_index][lane]).copy()
        except BaseException as error:
            shapes = [list(value.shape) for value in (output[4][route_index], output[5][route_index], output[6][route_index], output[8][route_index])]
            raise MlxEngineError(f"training routing conversion failed at block {block_index}: {shapes}") from error
        blocks.append((block_index, raw, effective, addresses, probes))
    return {
        "run_id": run_root.name,
        "stage": stage,
        "construction_seed": construction_seed,
        "logical_update": logical_update,
        "forward_sequence": forward_sequence,
        "required_source": required_source.detach().cpu().clone(),
        "model": model,
        "blocks": tuple(blocks),
    }


def write_training_routing_payload(value: tuple[Mapping[str, Any], Any]) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    payload, stream = value
    blocks = []
    for block_index, raw, effective, addresses, probes in payload["blocks"]:
        telemetry = routing_telemetry(raw, effective, addresses, probes)
        blocks.append(SimpleNamespace(block_index=block_index, kind="routed", mixer_output=SimpleNamespace(telemetry=telemetry)))
    model_output = SimpleNamespace(blocks=tuple(blocks))
    rows = cpu._routing_rows(
        model_output,
        payload["run_id"],
        payload["construction_seed"],
        "training",
        payload["model"],
        payload["stage"],
        None,
        payload["logical_update"],
        payload["forward_sequence"],
        None,
        0,
        payload["required_source"],
        None,
        None,
        None,
    )
    for row in rows:
        stream.write(row)


def execute_stage(
    request_value: Mapping[str, Any],
    run_root: Path,
    exchange: Any,
    child_sequence: int,
    routing_streams: Mapping[int, Any] | None = None,
    forward_sequences: dict[int, int] | None = None,
    endpoint_parity_records: list[Mapping[str, Any]] | None = None,
) -> tuple[int, list[dict[str, str]], dict[int, list[dict[str, Any]]]]:
    request = validate_stage_request(request_value)
    state = stage_parameter_state(request, run_root)
    generators, draw = stage_batches(request)
    runner = compiled_stage_step(state["train_names"], state["frozen_names"], state["decay"], state["role"], request["stage"])
    audits, audit_tracker = initialize_stage_audits(request, state)
    start_message = {
        "kind": "stage_started",
        "sequence": child_sequence,
        "stage": request["stage"],
        "construction_seeds": request["construction_seeds"],
    }
    response = exchange(start_message)
    require_parent_ack(response, "stage_start_committed", request["stage"], None)
    child_sequence += 1
    final_hashes = [""] * len(request["construction_seeds"])
    routing_writer = OrderedBackgroundWriter(write_training_routing_payload, 16) if routing_streams is not None else None
    try:
        for logical_update in range(1, request["updates"] + 1):
            batches, batch_hashes = draw()
            final_hashes = batch_hashes
            ready = {
                "kind": "update_ready",
                "sequence": child_sequence,
                "stage": request["stage"],
                "construction_seeds": request["construction_seeds"],
                "logical_update": logical_update,
                "batch_sha256s": batch_hashes,
                "token_positions": [int(batch["tokens"].numel()) for batch in batches],
            }
            response = exchange(ready)
            require_parent_ack(response, "update_start_committed", request["stage"], logical_update)
            child_sequence += 1
            tokens, targets, required = mx_stage_batch(batches, request["stage"])
            multiplier = stage_multiplier(logical_update, request["updates"], request["warmup_updates"])
            rates = tuple(mx.array(peak * multiplier, dtype=mx.float32) for peak in state["peak_rates"])
            started = time.perf_counter_ns()
            updated, next_first, next_second, output, gradient_norm, audit_status = runner(
                state["train"],
                state["first"],
                state["second"],
                state["frozen"],
                tokens,
                targets,
                required,
                rates,
                mx.array(float(logical_update), dtype=mx.float32),
            )
            mx.eval(updated, next_first, next_second, output, gradient_norm, audit_status)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
            host_audit_status = np.asarray(audit_status)
            update_stage_audits(audit_tracker, host_audit_status, logical_update)
            state["train"] = updated
            state["first"] = next_first
            state["second"] = next_second
            if routing_writer is not None:
                if forward_sequences is None or request["stage"] == "rung_two" or set(routing_streams) != set(request["construction_seeds"]):
                    raise MlxEngineError("training routing stream identity differs")
                for lane, construction_seed in enumerate(request["construction_seeds"]):
                    sequence = forward_sequences[construction_seed]
                    payload = capture_training_routing_payload(run_root, request["stage"], construction_seed, logical_update, sequence, batches[lane]["required_source"], output, lane)
                    routing_writer.submit((payload, routing_streams[construction_seed]))
                    forward_sequences[construction_seed] = sequence + 1
            metrics = []
            for lane in range(len(request["construction_seeds"])):
                overflow, maximum = routing_reduction(output[6], lane)
                if overflow != 0:
                    raise MlxEngineError("stage update produced route overflow")
                metrics.append(
                    {
                        "total_loss": float(output[0][lane].item()),
                        "task_loss": float(output[1][lane].item()),
                        "internal_router_loss": float(output[2][lane].item()) if request["stage"] == "joint" else None,
                        "supervised_route_loss": float(output[3][lane].item()) if request["stage"] in {"router_only", "joint"} else None,
                        "gradient_norm": float(gradient_norm[lane].item()),
                        "clip_result": "clipped" if float(gradient_norm[lane].item()) > 1.0 else "unchanged",
                        "raw_overflow_count": overflow,
                        "max_bucket_load": maximum,
                        "elapsed_seconds": elapsed,
                        "finite": True,
                    }
                )
            complete = {
                "kind": "update_complete",
                "sequence": child_sequence,
                "stage": request["stage"],
                "construction_seeds": request["construction_seeds"],
                "logical_update": logical_update,
                "batch_sha256s": batch_hashes,
                "metrics": metrics,
                "mx_eval_complete": True,
                "memory": runtime_memory(),
            }
            response = exchange(complete)
            require_parent_ack(response, "update_complete_committed", request["stage"], logical_update)
            child_sequence += 1
        if routing_writer is not None:
            routing_writer.close()
            routing_writer = None
            for construction_seed in request["construction_seeds"]:
                routing_streams[construction_seed].close()
                validate_closed_training_gzip(routing_streams[construction_seed].path, run_root.name, construction_seed, request["stage"])
        checkpoints, records = save_stage_checkpoints(request, run_root, state, generators, final_hashes)
        if endpoint_parity_records is None:
            raise MlxEngineError("trained endpoint parity sink is absent")
        endpoint_parity_records.extend(records)
        finalized_audits = finalize_stage_audits(request, state, audits, audit_tracker)
        finish = {
            "kind": "stage_complete",
            "sequence": child_sequence,
            "stage": request["stage"],
            "construction_seeds": request["construction_seeds"],
            "checkpoint_paths": [record["path"] for record in checkpoints],
            "checkpoint_sha256s": [record["checkpoint_sha256"] for record in checkpoints],
            "optimizer_state_sha256s": [record["optimizer_state_sha256"] for record in checkpoints],
            "rng_state_sha256s": [record["rng_state_sha256"] for record in checkpoints],
        }
        response = exchange(finish)
        require_parent_ack(response, "stage_complete_committed", request["stage"], None)
        return child_sequence + 1, checkpoints, finalized_audits
    finally:
        if routing_writer is not None:
            routing_writer.abort()


def load_torch_state(model: MlxModularModel, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    model_parameters = dict(tree_flatten(model.parameters()))
    torch_descriptors = {}
    mlx_descriptors = {}
    values = []
    for torch_name, tensor in state.items():
        mlx_name = mapped_mlx_parameter_name(torch_name)
        if mlx_name not in model_parameters:
            raise MlxEngineError(f"unmapped torch parameter: {torch_name}")
        if tuple(model_parameters[mlx_name].shape) != tuple(tensor.shape):
            raise MlxEngineError(f"mapped parameter shape differs: {torch_name}")
        if tensor.dtype != torch.float32 or model_parameters[mlx_name].dtype != mx.float32:
            raise MlxEngineError(f"mapped parameter dtype differs: {torch_name}")
        values.append((mlx_name, mx.array(tensor.detach().cpu().numpy())))
        torch_descriptors[torch_name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        mlx_descriptors[mlx_name] = {"shape": list(model_parameters[mlx_name].shape), "dtype": "float32"}
    manifest = validate_parameter_mapping(torch_descriptors, mlx_descriptors)
    model.update(tree_unflatten(values))
    mx.eval(model.parameters())
    return manifest


def parameter_value_mapping_evidence(model: MlxModularModel, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    model_parameters = dict(tree_flatten(model.parameters()))
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    maximum = 0.0
    exact = True
    for torch_name in sorted(state):
        mlx_name = mapped_mlx_parameter_name(torch_name)
        source = state[torch_name].detach().cpu().numpy()
        destination = np.array(model_parameters[mlx_name])
        exact = exact and np.array_equal(source, destination)
        maximum = max(maximum, maximum_array_error(destination, source))
        identity = torch_name.encode("utf-8") + b"\0"
        source_digest.update(identity)
        source_digest.update(source.tobytes())
        destination_digest.update(identity)
        destination_digest.update(destination.tobytes())
    return {
        "parameter_count": len(state),
        "max_abs": maximum,
        "byte_exact": bool(exact),
        "source_sha256": source_digest.hexdigest(),
        "destination_sha256": destination_digest.hexdigest(),
    }


def torch_model(role: str, seed: int) -> ModularNeuralMachine:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = rung_two_config() if role == "rung_two" else rung_one_config(role)
        return ModularNeuralMachine(config)


def full_model_parity(
    role: str = "selected",
    model_seed: int = 3123,
    data_seed: int = 4123,
    batch_size: int = 2,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    source = torch_model(role, model_seed)
    destination = MlxModularModel(role)
    mapping = load_torch_state(destination, source.state_dict())
    value_mapping = parameter_value_mapping_evidence(destination, source.state_dict())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    batch = batch_size
    tokens = sequence_length if sequence_length is not None else 512 if role == "rung_two" else 128
    vocabulary = 256 if role == "rung_two" else 128
    input_ids = torch.randint(0, vocabulary, (batch, tokens), generator=generator)
    source.eval()
    with torch.inference_mode():
        source_output = source(input_ids, return_aux=True, route_detail=True, request_block4_router_loss=role == "selected")
    destination_output = destination(mx.array(input_ids.numpy()), None, role == "selected")
    mx.eval(destination_output)
    destination_logits, destination_hidden, _, _, destination_router_loss, raw_routes, effective_routes, sequence_deltas, feature_deltas = destination_output
    logits_error = float(np.max(np.abs(np.array(destination_logits) - source_output.logits.detach().numpy())))
    hidden_error = float(np.max(np.abs(np.array(destination_hidden) - source_output.hidden.detach().numpy())))
    if len(sequence_deltas) != len(source_output.blocks) or len(feature_deltas) != len(source_output.blocks):
        raise MlxEngineError("initial full-model block parity closure differs")
    sequence_errors = [maximum_array_error(observed, expected.computed_sequence_delta.detach().numpy()) for observed, expected in zip(sequence_deltas, source_output.blocks)]
    feature_errors = [maximum_array_error(observed, expected.feature_delta.detach().numpy()) for observed, expected in zip(feature_deltas, source_output.blocks)]
    forward_statistics = tensor_comparison_statistics(
        [
            ("logits", destination_logits, source_output.logits.detach().numpy()),
            ("final_hidden", destination_hidden, source_output.hidden.detach().numpy()),
            *((f"sequence_delta_{index}", observed, expected.computed_sequence_delta.detach().numpy()) for index, (observed, expected) in enumerate(zip(sequence_deltas, source_output.blocks))),
            *((f"feature_delta_{index}", observed, expected.feature_delta.detach().numpy()) for index, (observed, expected) in enumerate(zip(feature_deltas, source_output.blocks))),
        ]
    )
    routed_outputs = [block.mixer_output for block in source_output.blocks if block.kind == "routed"]
    route_exact = len(raw_routes) == len(routed_outputs) and len(effective_routes) == len(routed_outputs)
    for source_routed, mlx_raw, mlx_effective in zip(routed_outputs, raw_routes, effective_routes):
        route_exact = route_exact and np.array_equal(np.array(mlx_raw), source_routed.telemetry["raw_remote"].numpy())
        route_exact = route_exact and np.array_equal(np.array(mlx_effective), source_routed.telemetry["effective_remote"].numpy())
    source_router_loss = next(block.mixer_output.router_loss for block in source_output.blocks if block.block_index == 4) if role == "selected" else None
    router_loss_error = abs(float(destination_router_loss.item())) if source_router_loss is None else abs(float(destination_router_loss.item()) - float(source_router_loss))
    numeric = (logits_error, hidden_error, router_loss_error, *sequence_errors, *feature_errors)
    forward_maximum = max(logits_error, hidden_error, *sequence_errors, *feature_errors)
    forward_absolute_pass = forward_maximum <= 1e-5
    forward_scale_aware_pass = forward_maximum <= 5e-5 and forward_statistics["relative_max"] <= 5e-5 and forward_statistics["normalized_l2_max"] <= 5e-6 and forward_statistics["cosine_min"] >= 0.99999999999
    forward_pass = forward_absolute_pass or forward_scale_aware_pass
    passed = all(math.isfinite(value) and value >= 0.0 for value in numeric)
    passed = passed and forward_pass
    passed = passed and router_loss_error <= 1e-6 and mapping["bijective"] and value_mapping["byte_exact"] and value_mapping["max_abs"] == 0.0 and route_exact
    return {
        "role": role,
        "stage": "joint" if role == "selected" else "forward_calibration",
        "objective": "task_plus_0.1_times_internal_router_plus_supervised_route" if role == "selected" else "full_forward_output_surface",
        "model_seed": model_seed,
        "data_seed": data_seed,
        "batch_size": batch,
        "sequence_length": tokens,
        "mapped_parameter_count": len(mapping["records"]),
        "mapping_bijective": mapping["bijective"],
        "mapping_sha256": mapping["sha256"],
        "mapping_transpose": mapping["transpose"],
        "mapping_value_count": value_mapping["parameter_count"],
        "mapping_value_max_abs": value_mapping["max_abs"],
        "mapping_value_byte_exact": value_mapping["byte_exact"],
        "mapping_source_value_sha256": value_mapping["source_sha256"],
        "mapping_destination_value_sha256": value_mapping["destination_sha256"],
        "block_count": len(source_output.blocks),
        "raw_route_count": len(raw_routes),
        "effective_route_count": len(effective_routes),
        "logits_max_abs": logits_error,
        "hidden_max_abs": hidden_error,
        "sequence_delta_max_abs_by_block": sequence_errors,
        "feature_delta_max_abs_by_block": feature_errors,
        "forward_relative_max": forward_statistics["relative_max"],
        "forward_normalized_l2_max": forward_statistics["normalized_l2_max"],
        "forward_cosine_min": forward_statistics["cosine_min"],
        "forward_worst_tensor": forward_statistics["worst_tensor"],
        "forward_worst_index": forward_statistics["worst_index"],
        "forward_worst_observed": forward_statistics["worst_observed"],
        "forward_worst_expected": forward_statistics["worst_expected"],
        "forward_absolute_pass": bool(forward_absolute_pass),
        "forward_scale_aware_pass": bool(forward_scale_aware_pass),
        "forward_pass": bool(forward_pass),
        "forward_scale_aware_absolute_tolerance": 5e-5,
        "forward_relative_tolerance": 5e-5,
        "forward_normalized_l2_tolerance": 5e-6,
        "forward_cosine_tolerance": 0.99999999999,
        "route_exact": bool(route_exact),
        "router_loss_max_abs": router_loss_error,
        "pass": bool(passed),
    }


def full_gradient_parity(role: str, model_seed: int, data_seed: int, batch_size: int) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    source = torch_model(role, model_seed)
    destination = MlxModularModel(role)
    load_torch_state(destination, source.state_dict())
    if role != "selected":
        raise MlxEngineError("initial gradient parity role differs")
    runtime = cpu._import_runtime()
    _, _, membership = cpu._make_optimizer(source, "joint", runtime)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    batch = cpu._continuous_rung_one_batch(generator, batch_size, torch)
    source.train()
    source_loss, source_components, _, _, _ = cpu._training_forward_loss(source, batch, "joint", runtime)
    source_loss.backward()
    flattened = tuple(tree_flatten(destination.parameters()))
    names = tuple(name for name, _ in flattened)
    policies = {name: optimizer_parameter_policy(torch_policy_name(name), "joint") for name in names}
    train_names = tuple(name for name in names if policies[name]["trainable"])
    frozen_names = tuple(name for name in names if not policies[name]["trainable"])
    parameter_values = dict(flattened)
    train_values = tuple(parameter_values[name] for name in train_names)
    frozen_values = tuple(parameter_values[name] for name in frozen_names)
    tokens = mx.array(batch["tokens"].numpy())
    targets = mx.array(batch["targets"].numpy())
    required = mx.array(batch["required_source"].numpy())

    def loss_function(current_values):
        return stage_loss(current_values, frozen_values, train_names, frozen_names, role, "joint", tokens, targets, required)

    destination_losses, gradients = mx.value_and_grad(loss_function)(train_values)
    mx.eval(destination_losses, gradients)
    gradient_by_name = {name: np.array(value) for name, value in zip(train_names, gradients)}
    comparisons = []
    none_exact = True
    for name, parameter in source.named_parameters():
        mlx_name = mapped_mlx_parameter_name(name)
        if membership[name]["requires_grad"]:
            if mlx_name not in gradient_by_name:
                none_exact = False
                continue
            observed = gradient_by_name[mlx_name]
            expected = np.zeros_like(observed) if parameter.grad is None else parameter.grad.detach().cpu().numpy()
            if parameter.grad is None:
                none_exact = none_exact and np.array_equal(observed, expected)
            comparisons.append((name, observed, expected))
        elif parameter.grad is not None or mlx_name in gradient_by_name:
            none_exact = False
    evidence = gradient_comparison_evidence(comparisons)
    component_errors = {
        "task_loss": abs(float(destination_losses[1].item()) - float(source_components["task_loss"])),
        "internal_router_loss": abs(float(destination_losses[2].item()) - float(source_components["internal_router_loss"])),
        "supervised_route_loss": abs(float(destination_losses[3].item()) - float(source_components["supervised_route_loss"])),
    }
    loss_error = abs(float(destination_losses[0].item()) - float(source_loss.detach()))
    component_maximum = max(component_errors.values())
    passed = math.isfinite(loss_error) and all(math.isfinite(value) for value in component_errors.values())
    passed = passed and loss_error <= 1e-6 and component_maximum <= 1e-6 and evidence["gradient_pass"] and none_exact
    return {
        "role": role,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "batch_size": batch_size,
        "loss_max_abs": loss_error,
        "component_loss_max_abs": component_maximum,
        "component_loss_errors": component_errors,
        **evidence,
        "grad_none_zero_exact": none_exact,
        "pass": bool(passed),
    }


def all_role_forward_calibration(selected: Mapping[str, Any]) -> dict[str, Any]:
    records = [
        dict(selected),
        full_model_parity("all_eligible", 3124, 4124),
        full_model_parity("dense", 3125, 4125),
        full_model_parity("rung_two", 3126, 4126),
    ]
    return {
        "roles": ["selected", "all_eligible", "dense", "rung_two"],
        "fresh_process_required": True,
        "records": records,
        "pass": all(record["pass"] for record in records),
    }


def held_out_forward_admission() -> dict[str, Any]:
    records = [
        full_model_parity("selected", 8123, 9123, 2, 128),
        full_model_parity("all_eligible", 8124, 9124, 3, 128),
    ]
    return {
        "thresholds_frozen_before_execution": True,
        "records": records,
        "pass": all(record["pass"] for record in records),
    }


def adamw_parity() -> dict[str, Any]:
    torch_parameter = torch.tensor([[0.5, -1.25], [2.0, -0.75]], dtype=torch.float32, requires_grad=True)
    torch_optimizer = torch.optim.AdamW([{"params": [torch_parameter], "lr": 0.003, "weight_decay": 0.01}], betas=(0.9, 0.95), eps=1e-8, foreach=False, fused=False)
    torch_parameter.grad = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    torch_optimizer.step()
    mlx_parameter = mx.array([[0.5, -1.25], [2.0, -0.75]], dtype=mx.float32)
    mlx_gradient = mx.array([[0.25, -0.5], [0.75, -1.0]], dtype=mx.float32)
    optimizer = optim.AdamW(learning_rate=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, bias_correction=True)
    optimizer.init({"weight": mlx_parameter})
    updated = optimizer.apply_gradients({"weight": mlx_gradient}, {"weight": mlx_parameter})
    mx.eval(updated, optimizer.state)
    error = float(np.max(np.abs(np.array(updated["weight"]) - torch_parameter.detach().numpy())))
    return {"max_abs": error, "tolerance": 1e-7, "pass": math.isfinite(error) and error <= 1e-7}


def independent_vmap_probe() -> dict[str, Any]:
    parameters = mx.arange(20, dtype=mx.float32).reshape(5, 4)
    gradients = mx.arange(20, dtype=mx.float32).reshape(5, 4) * 0.01
    moments = mx.zeros_like(parameters)

    def lane(parameter: mx.array, gradient: mx.array, moment: mx.array) -> tuple[mx.array, mx.array]:
        norm = mx.sqrt(mx.sum(gradient * gradient))
        clipped = gradient * mx.minimum(1.0, 1.0 / (norm + 1e-6))
        next_moment = 0.9 * moment + 0.1 * clipped
        return parameter - 0.003 * next_moment, next_moment

    compiled = mx.compile(mx.vmap(lane))
    updated, next_moments = compiled(parameters, gradients, moments)
    mx.eval(updated, next_moments)
    unique = len({hashlib.sha256(np.array(updated[index]).tobytes()).hexdigest() for index in range(5)})
    return {"lanes": 5, "unique_parameter_hashes": unique, "lane_local_clipping": True, "pure_parameter_tree": True, "pass": unique == 5}


def functional_forward_parity() -> dict[str, Any]:
    source = torch_model("selected", 3123)
    model = MlxModularModel("selected")
    load_torch_state(model, source.state_dict())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4123)
    input_ids = torch.randint(0, 128, (2, 128), generator=generator)
    tokens = mx.array(input_ids.numpy())
    module_output = model(tokens, None, True)
    parameters = dict(tree_flatten(model.parameters()))
    function_output = functional_forward(parameters, "selected", tokens, True)
    mx.eval(module_output, function_output)
    logits_error = float(np.max(np.abs(np.array(module_output[0]) - np.array(function_output[0]))))
    query_error = float(np.max(np.abs(np.array(module_output[2]) - np.array(function_output[1]))))
    key_error = float(np.max(np.abs(np.array(module_output[3]) - np.array(function_output[2]))))
    loss_error = abs(float(module_output[4].item()) - float(function_output[3].item()))
    route_exact = all(np.array_equal(np.array(left), np.array(right)) for left, right in zip(module_output[5], function_output[4]))
    route_exact = route_exact and all(np.array_equal(np.array(left), np.array(right)) for left, right in zip(module_output[6], function_output[5]))
    return {
        "logits_max_abs": logits_error,
        "query_route_max_abs": query_error,
        "key_route_max_abs": key_error,
        "router_loss_max_abs": loss_error,
        "route_exact": bool(route_exact),
        "pass": max(logits_error, query_error, key_error, loss_error) == 0.0 and route_exact,
    }


def adamw_state_oracle(
    parameter: Any,
    gradient: Any,
    first_moment: Any,
    second_moment: Any,
    step: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(parameter, dtype=np.float32).astype(np.float64)
    clipped = np.asarray(gradient, dtype=np.float32).astype(np.float64)
    prior_first = np.asarray(first_moment, dtype=np.float32).astype(np.float64)
    prior_second = np.asarray(second_moment, dtype=np.float32).astype(np.float64)
    rate = float(np.float32(learning_rate))
    decay = float(np.float32(weight_decay))
    next_first = 0.9 * prior_first + 0.1 * clipped
    next_second = 0.95 * prior_second + 0.05 * clipped * clipped
    first_correction = 1.0 - 0.9**step
    second_correction = 1.0 - 0.95**step
    updated = (1.0 - rate * decay) * value - rate * (next_first / first_correction) / (np.sqrt(next_second / second_correction) + 1e-8)
    return updated, next_first, next_second


def adamw_state_error_bounds(
    parameter: Any,
    gradient: Any,
    first_moment: Any,
    second_moment: Any,
    step: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(parameter, dtype=np.float32).astype(np.float64)
    clipped = np.asarray(gradient, dtype=np.float32).astype(np.float64)
    prior_first = np.asarray(first_moment, dtype=np.float32).astype(np.float64)
    prior_second = np.asarray(second_moment, dtype=np.float32).astype(np.float64)
    rate = float(np.float32(learning_rate))
    decay = float(np.float32(weight_decay))
    updated, first, second = adamw_state_oracle(value, clipped, prior_first, prior_second, step, rate, decay)
    unit_roundoff = 2.0**-24

    def gamma(operations: int) -> float:
        return operations * unit_roundoff / (1.0 - operations * unit_roundoff)

    decayed = (1.0 - rate * decay) * value
    update = rate * (first / (1.0 - 0.9**step)) / (np.sqrt(second / (1.0 - 0.95**step)) + 1e-8)
    parameter_bound = gamma(32) * (np.abs(decayed) + np.abs(update))
    first_bound = gamma(6) * (np.abs(0.9 * prior_first) + np.abs(0.1 * clipped))
    second_bound = gamma(8) * (np.abs(0.95 * prior_second) + np.abs(0.05 * clipped * clipped))
    return parameter_bound, first_bound, second_bound


def adamw_one_step_oracle(parameter: Any, gradient: Any, learning_rate: float, weight_decay: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(parameter, dtype=np.float32)
    return adamw_state_oracle(value, gradient, np.zeros_like(value), np.zeros_like(value), 1, learning_rate, weight_decay)


def adamw_one_step_error_bounds(parameter: Any, gradient: Any, learning_rate: float, weight_decay: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(parameter, dtype=np.float32)
    return adamw_state_error_bounds(value, gradient, np.zeros_like(value), np.zeros_like(value), 1, learning_rate, weight_decay)


def carried_adamw_parity() -> dict[str, Any]:
    initial = (
        np.linspace(-0.75, 0.65, 20, dtype=np.float32).reshape(5, 4),
        np.linspace(-0.45, 0.55, 30, dtype=np.float32).reshape(5, 2, 3),
    )
    first_gradients = (
        np.linspace(-0.08, 0.09, 20, dtype=np.float32).reshape(5, 4),
        np.linspace(0.07, -0.06, 30, dtype=np.float32).reshape(5, 2, 3),
    )
    second_gradients = (
        np.linspace(0.05, -0.04, 20, dtype=np.float32).reshape(5, 4),
        np.linspace(-0.03, 0.065, 30, dtype=np.float32).reshape(5, 2, 3),
    )
    rates = (mx.array(0.003), mx.array(0.00025))
    decay = (0.01, 0.0)
    output = tuple(mx.zeros((5, 1), dtype=mx.float32) for _ in range(4))

    def transition(values, first, second, gradients, step):
        return batched_optimizer_step(values, first, second, gradients, output, rates, decay, step)

    compiled = mx.compile(transition)
    mlx_initial = tuple(mx.array(value) for value in initial)
    zeros = tuple(mx.zeros_like(value) for value in mlx_initial)
    mlx_first_gradients = tuple(mx.array(value) for value in first_gradients)
    mlx_second_gradients = tuple(mx.array(value) for value in second_gradients)
    first_result = compiled(mlx_initial, zeros, zeros, mlx_first_gradients, mx.array(1.0))
    mx.eval(first_result)
    prior_values = tuple(np.array(value) for value in first_result[0])
    prior_first = tuple(np.array(value) for value in first_result[1])
    prior_second = tuple(np.array(value) for value in first_result[2])
    second_result = compiled(first_result[0], first_result[1], first_result[2], mlx_second_gradients, mx.array(2.0))
    mx.eval(second_result)
    formula_records = []
    cross_runtime_errors = []
    gradient_digest = hashlib.sha256()
    for name, values in (("first", first_gradients), ("second", second_gradients)):
        gradient_digest.update(name.encode("utf-8"))
        for value in values:
            gradient_digest.update(value.tobytes())
    nonzero_moments = all(np.any(value != 0.0) for value in (*prior_first, *prior_second))
    for tensor_index, name in enumerate(("decayed", "nondecayed")):
        for lane in range(5):
            expected = adamw_state_oracle(
                prior_values[tensor_index][lane],
                second_gradients[tensor_index][lane],
                prior_first[tensor_index][lane],
                prior_second[tensor_index][lane],
                2,
                float(rates[tensor_index].item()),
                decay[tensor_index],
            )
            bounds = adamw_state_error_bounds(
                prior_values[tensor_index][lane],
                second_gradients[tensor_index][lane],
                prior_first[tensor_index][lane],
                prior_second[tensor_index][lane],
                2,
                float(rates[tensor_index].item()),
                decay[tensor_index],
            )
            for surface, observed, oracle, bound in zip(("parameter", "first_moment", "second_moment"), (second_result[0][tensor_index][lane], second_result[1][tensor_index][lane], second_result[2][tensor_index][lane]), expected, bounds):
                observed_array = np.array(observed).astype(np.float64)
                difference = np.abs(observed_array - oracle)
                allowed = np.nextafter(bound, np.full_like(bound, np.inf))
                ratio = np.zeros_like(difference)
                np.divide(difference, allowed, out=ratio, where=difference != 0.0)
                flat_index = int(np.argmax(ratio))
                formula_records.append(
                    {
                        "runtime": "MLX",
                        "surface": surface,
                        "tensor": name,
                        "lane": lane,
                        "index": [int(value) for value in np.unravel_index(flat_index, difference.shape)],
                        "worst_abs": float(difference.reshape(-1)[flat_index]),
                        "worst_bound": float(allowed.reshape(-1)[flat_index]),
                        "max_bound_ratio": float(ratio.reshape(-1)[flat_index]),
                    }
                )
    torch_prior = []
    torch_final = []
    for lane in range(5):
        parameters = [torch.tensor(value[lane], dtype=torch.float32, requires_grad=True) for value in initial]
        optimizer = torch.optim.AdamW(
            [
                {"params": [parameters[0]], "lr": 0.003, "weight_decay": 0.01},
                {"params": [parameters[1]], "lr": 0.00025, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False,
        )
        for parameter, gradient in zip(parameters, first_gradients):
            parameter.grad = torch.from_numpy(gradient[lane].copy())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        lane_prior = []
        for parameter in parameters:
            state = optimizer.state[parameter]
            lane_prior.append((parameter.detach().numpy().copy(), state["exp_avg"].detach().numpy().copy(), state["exp_avg_sq"].detach().numpy().copy()))
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, second_gradients):
            parameter.grad = torch.from_numpy(gradient[lane].copy())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        lane_final = []
        for parameter in parameters:
            state = optimizer.state[parameter]
            if float(state["step"]) != 2.0:
                raise MlxEngineError("carried AdamW step identity differs")
            lane_final.append((parameter.detach().numpy().copy(), state["exp_avg"].detach().numpy().copy(), state["exp_avg_sq"].detach().numpy().copy()))
        torch_prior.append(lane_prior)
        torch_final.append(lane_final)
    for lane in range(5):
        for tensor_index, name in enumerate(("decayed", "nondecayed")):
            expected = adamw_state_oracle(
                torch_prior[lane][tensor_index][0],
                second_gradients[tensor_index][lane],
                torch_prior[lane][tensor_index][1],
                torch_prior[lane][tensor_index][2],
                2,
                float(rates[tensor_index].item()),
                decay[tensor_index],
            )
            bounds = adamw_state_error_bounds(
                torch_prior[lane][tensor_index][0],
                second_gradients[tensor_index][lane],
                torch_prior[lane][tensor_index][1],
                torch_prior[lane][tensor_index][2],
                2,
                float(rates[tensor_index].item()),
                decay[tensor_index],
            )
            for surface_index, (surface, oracle, bound) in enumerate(zip(("parameter", "first_moment", "second_moment"), expected, bounds)):
                observed = torch_final[lane][tensor_index][surface_index].astype(np.float64)
                difference = np.abs(observed - oracle)
                allowed = np.nextafter(bound, np.full_like(bound, np.inf))
                ratio = np.zeros_like(difference)
                np.divide(difference, allowed, out=ratio, where=difference != 0.0)
                flat_index = int(np.argmax(ratio))
                formula_records.append(
                    {
                        "runtime": "Torch",
                        "surface": surface,
                        "tensor": name,
                        "lane": lane,
                        "index": [int(value) for value in np.unravel_index(flat_index, difference.shape)],
                        "worst_abs": float(difference.reshape(-1)[flat_index]),
                        "worst_bound": float(allowed.reshape(-1)[flat_index]),
                        "max_bound_ratio": float(ratio.reshape(-1)[flat_index]),
                    }
                )
                cross_runtime_errors.append(maximum_array_error(second_result[surface_index][tensor_index][lane], torch_final[lane][tensor_index][surface_index]))
    worst = max(formula_records, key=lambda value: (value["max_bound_ratio"], value["runtime"], value["surface"], value["tensor"], value["lane"]))
    mlx_preclip = np.sqrt(sum(np.sum(value.astype(np.float64) ** 2, axis=tuple(range(1, value.ndim))) for value in second_gradients))
    passed = nonzero_moments and float(np.max(mlx_preclip)) < 1.0 and worst["max_bound_ratio"] <= 1.0 and all(np.isfinite(value) for value in cross_runtime_errors)
    return {
        "lanes": 5,
        "tensor_count": 2,
        "first_update": 1,
        "tested_update": 2,
        "bias_correction": True,
        "nonzero_carried_first_and_second_moments": bool(nonzero_moments),
        "distinct_second_gradient": True,
        "canonical_gradient_sha256": gradient_digest.hexdigest(),
        "gradient_clip_identity": bool(float(np.max(mlx_preclip)) < 1.0),
        "formula_unit_roundoff": 2.0**-24,
        "formula_parameter_operation_budget": 32,
        "formula_first_moment_operation_budget": 6,
        "formula_second_moment_operation_budget": 8,
        "max_bound_ratio": worst["max_bound_ratio"],
        "worst_runtime": worst["runtime"],
        "worst_surface": worst["surface"],
        "worst_tensor": worst["tensor"],
        "worst_lane": worst["lane"],
        "worst_index": worst["index"],
        "worst_abs": worst["worst_abs"],
        "worst_bound": worst["worst_bound"],
        "cross_runtime_max_abs": max(cross_runtime_errors),
        "pass": bool(passed),
    }


def actual_model_vmap5_probe() -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    lane_parameters = []
    torch_models = []
    value_mappings = []
    names = None
    for seed in RUNG_ONE_SEEDS:
        source = torch_model("selected", seed)
        model = MlxModularModel("selected")
        load_torch_state(model, source.state_dict())
        value_mappings.append(parameter_value_mapping_evidence(model, source.state_dict()))
        flattened = tuple(tree_flatten(model.parameters()))
        current_names = tuple(name for name, _ in flattened)
        if names is not None and current_names != names:
            raise MlxEngineError("vmap lane parameter order differs")
        names = current_names
        lane_parameters.append(tuple(value for _, value in flattened))
        torch_models.append(source)
    if names is None:
        raise MlxEngineError("vmap lanes are absent")
    mapping_source_digest = hashlib.sha256()
    mapping_destination_digest = hashlib.sha256()
    for seed, evidence in zip(RUNG_ONE_SEEDS, value_mappings):
        mapping_source_digest.update(str(seed).encode("utf-8"))
        mapping_source_digest.update(evidence["source_sha256"].encode("ascii"))
        mapping_destination_digest.update(str(seed).encode("utf-8"))
        mapping_destination_digest.update(evidence["destination_sha256"].encode("ascii"))
    mapping_value_byte_exact = all(evidence["byte_exact"] and evidence["max_abs"] == 0.0 for evidence in value_mappings)
    policies = {name: optimizer_parameter_policy(torch_policy_name(name), "joint") for name in names}
    train_names = tuple(name for name in names if policies[name]["trainable"])
    frozen_names = tuple(name for name in names if name not in set(train_names))
    name_indexes = {name: index for index, name in enumerate(names)}
    stacked_train = tuple(mx.stack([lane[name_indexes[name]] for lane in lane_parameters]) for name in train_names)
    stacked_frozen = tuple(mx.stack([lane[name_indexes[name]] for lane in lane_parameters]) for name in frozen_names)
    first = tuple(mx.zeros_like(value) for value in stacked_train)
    second = tuple(mx.zeros_like(value) for value in stacked_train)
    decay = tuple(policies[name]["weight_decay"] for name in train_names)
    rates = tuple(mx.array(policies[name]["peak_lr"]) for name in train_names)
    token_rows = []
    target_rows = []
    required_rows = []
    torch_batches = []
    for seed in RUNG_ONE_SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(300000 + seed)
        batch = {
            "tokens": torch.randint(0, 128, (2, 128), generator=generator),
            "targets": torch.randint(0, 128, (2,), generator=generator),
            "required_source": torch.randint(0, 15, (2,), generator=generator),
        }
        torch_batches.append(batch)
        token_rows.append(batch["tokens"].numpy())
        target_rows.append(batch["targets"].numpy())
        required_rows.append(batch["required_source"].numpy())
    tokens = mx.array(np.stack(token_rows))
    targets = mx.array(np.stack(target_rows))
    required = mx.array(np.stack(required_rows))
    runner = compiled_stage_step(train_names, frozen_names, decay, "selected", "joint", True)
    updated, next_first, next_second, output, gradient_norm, audit_status, raw_gradients, clipped_gradients = runner(stacked_train, first, second, stacked_frozen, tokens, targets, required, rates, mx.array(1.0))
    mx.eval(updated, next_first, next_second, output, gradient_norm, audit_status, raw_gradients, clipped_gradients)
    codebook_indexes = [index for index, name in enumerate(train_names) if name.endswith(".codebooks")]
    codebook_exact = all(np.array_equal(np.array(updated[index]), np.array(stacked_train[index])) for index in codebook_indexes)
    lane_hashes = []
    for lane in range(5):
        digest = hashlib.sha256()
        for value in updated:
            digest.update(np.array(value[lane]).tobytes())
        lane_hashes.append(digest.hexdigest())
    runtime = cpu._import_runtime()
    loss_errors = []
    parameter_errors = []
    first_errors = []
    second_errors = []
    gradient_comparisons = []
    clipped_gradient_comparisons = []
    parameter_update_comparisons = []
    end_to_end_records = []
    parameter_residual_errors = []
    first_residual_errors = []
    second_residual_errors = []
    causal_residual_records = []
    five_lane_grad_none_zero_exact = True
    torch_preclip_norms = []
    torch_postclip_norms = []
    route_exact = True
    train_indexes = {name: index for index, name in enumerate(train_names)}
    optimizer_gradient_sha256 = hashlib.sha256()
    torch_optimizer_gradient_sha256 = hashlib.sha256()
    raw_gradient_sha256 = hashlib.sha256()
    optimizer_parameter_formula_errors = []
    optimizer_first_formula_errors = []
    optimizer_second_formula_errors = []
    torch_optimizer_parameter_formula_errors = []
    torch_optimizer_first_formula_errors = []
    torch_optimizer_second_formula_errors = []
    optimizer_parameter_formula_records = []
    mlx_optimizer_oracles = {}
    mlx_optimizer_bounds = {}
    for index, name in enumerate(train_names):
        parameter = np.array(stacked_train[index])
        gradient = np.array(clipped_gradients[index])
        observed_first = np.array(next_first[index])
        observed_second = np.array(next_second[index])
        observed_updated = np.array(updated[index])
        expected_updated, expected_first, expected_second = adamw_one_step_oracle(parameter, gradient, policies[name]["peak_lr"], decay[index])
        parameter_bound, first_bound, second_bound = adamw_one_step_error_bounds(parameter, gradient, policies[name]["peak_lr"], decay[index])
        mlx_optimizer_oracles[name] = (expected_updated, expected_first, expected_second)
        mlx_optimizer_bounds[name] = (parameter_bound, first_bound, second_bound)
        optimizer_first_formula_errors.append(float(np.max(np.abs(observed_first.astype(np.float64) - expected_first))))
        optimizer_second_formula_errors.append(float(np.max(np.abs(observed_second.astype(np.float64) - expected_second))))
        optimizer_parameter_formula_errors.append(float(np.max(np.abs(observed_updated.astype(np.float64) - expected_updated))))
        for surface, observed_value, expected_value, error_bound in (
            ("parameter", observed_updated, expected_updated, parameter_bound),
            ("first_moment", observed_first, expected_first, first_bound),
            ("second_moment", observed_second, expected_second, second_bound),
        ):
            formula_difference = np.abs(observed_value.astype(np.float64) - expected_value.astype(np.float64))
            allowed = np.nextafter(error_bound, np.full_like(error_bound, np.inf))
            formula_ratio = np.zeros_like(formula_difference)
            np.divide(formula_difference, allowed, out=formula_ratio, where=formula_difference != 0.0)
            formula_flat_index = int(np.argmax(formula_ratio))
            formula_index = np.unravel_index(formula_flat_index, formula_difference.shape)
            optimizer_parameter_formula_records.append(
                {
                    "runtime": "MLX",
                    "surface": surface,
                    "tensor": name,
                    "lane": int(formula_index[0]),
                    "index": [int(value) for value in formula_index[1:]],
                    "max_abs": float(np.max(formula_difference)),
                    "max_bound_ratio": float(formula_ratio.reshape(-1)[formula_flat_index]),
                    "worst_abs": float(formula_difference.reshape(-1)[formula_flat_index]),
                    "worst_bound": float(allowed.reshape(-1)[formula_flat_index]),
                    "observed": float(observed_value.reshape(-1)[formula_flat_index]),
                    "expected": float(expected_value.reshape(-1)[formula_flat_index]),
                }
            )
        optimizer_gradient_sha256.update(name.encode("utf-8"))
        optimizer_gradient_sha256.update(gradient.tobytes())
        raw_gradient_sha256.update(name.encode("utf-8"))
        raw_gradient_sha256.update(np.array(raw_gradients[index]).tobytes())
    for lane, (source, batch) in enumerate(zip(torch_models, torch_batches)):
        optimizer, _, membership = cpu._make_optimizer(source, "joint", runtime)
        cpu._set_optimizer_rates(optimizer, 1.0)
        source.train()
        optimizer.zero_grad(set_to_none=True)
        total, components, _, _, source_output = cpu._training_forward_loss(source, batch, "joint", runtime)
        total.backward()
        torch_raw_gradients = {
            name: np.zeros_like(parameter.detach().cpu().numpy()) if parameter.grad is None else parameter.grad.detach().cpu().numpy().copy()
            for name, parameter in source.named_parameters()
        }
        torch_preclip_norms.append(cpu._clip_gradient_norm_finite(torch, source, optimizer, {"stage": "joint", "lane": lane}))
        torch_clipped_gradients = {
            name: np.zeros_like(parameter.detach().cpu().numpy()) if parameter.grad is None else parameter.grad.detach().cpu().numpy().copy()
            for name, parameter in source.named_parameters()
        }
        torch_postclip_norms.append(math.sqrt(sum(float(np.square(value.astype(np.float64)).sum()) for name, value in torch_clipped_gradients.items() if membership[name]["requires_grad"])))
        for torch_name, parameter in source.named_parameters():
            if membership[torch_name]["requires_grad"] and parameter.grad is None:
                mlx_name = mapped_mlx_parameter_name(torch_name)
                five_lane_grad_none_zero_exact = five_lane_grad_none_zero_exact and np.array_equal(np.array(raw_gradients[train_indexes[mlx_name]])[lane], torch_raw_gradients[torch_name])
                five_lane_grad_none_zero_exact = five_lane_grad_none_zero_exact and np.array_equal(np.array(clipped_gradients[train_indexes[mlx_name]])[lane], torch_clipped_gradients[torch_name])
        optimizer.step()
        expected_losses = (
            float(total.detach()),
            float(components["task_loss"]),
            float(components["internal_router_loss"]),
            float(components["supervised_route_loss"]),
        )
        loss_errors.extend(abs(float(output[index][lane].item()) - expected) for index, expected in enumerate(expected_losses))
        routed_outputs = [block.mixer_output for block in source_output.blocks if block.kind == "routed"]
        route_exact = route_exact and len(output[4]) == len(routed_outputs) and len(output[5]) == len(routed_outputs)
        route_exact = route_exact and all(np.array_equal(np.array(observed)[lane], expected.telemetry["raw_remote"].numpy()) for observed, expected in zip(output[4], routed_outputs))
        route_exact = route_exact and all(np.array_equal(np.array(observed)[lane], expected.telemetry["effective_remote"].numpy()) for observed, expected in zip(output[5], routed_outputs))
        for torch_name, parameter in source.named_parameters():
            mlx_name = mapped_mlx_parameter_name(torch_name)
            if not membership[torch_name]["requires_grad"]:
                continue
            index = train_indexes[mlx_name]
            initial_parameter = np.array(stacked_train[index])[lane]
            observed_parameter = np.array(updated[index])[lane]
            expected_parameter = parameter.detach().cpu().numpy()
            parameter_error = np.abs(observed_parameter.astype(np.float64) - expected_parameter.astype(np.float64))
            parameter_errors.append(float(np.max(parameter_error)))
            flat_index = int(np.argmax(parameter_error))
            end_to_end_records.append(
                {
                    "lane": lane,
                    "tensor": torch_name,
                    "index": [int(value) for value in np.unravel_index(flat_index, parameter_error.shape)],
                    "max_abs": float(parameter_error.reshape(-1)[flat_index]),
                    "observed": float(observed_parameter.reshape(-1)[flat_index]),
                    "expected": float(expected_parameter.reshape(-1)[flat_index]),
                    "mlx_clipped_gradient": float(np.array(clipped_gradients[index])[lane].reshape(-1)[flat_index]),
                    "torch_clipped_gradient": float(torch_clipped_gradients[torch_name].reshape(-1)[flat_index]),
                }
            )
            gradient_comparisons.append((f"lane_{lane}.{torch_name}", np.array(raw_gradients[index])[lane], torch_raw_gradients[torch_name]))
            clipped_gradient_comparisons.append((f"lane_{lane}.{torch_name}", np.array(clipped_gradients[index])[lane], torch_clipped_gradients[torch_name]))
            parameter_update_comparisons.append((f"lane_{lane}.{torch_name}", observed_parameter - initial_parameter, expected_parameter - initial_parameter))
            optimizer_value = optimizer.state.get(parameter)
            if optimizer_value is None:
                expected_first = np.zeros_like(parameter.detach().cpu().numpy())
                expected_second = np.zeros_like(parameter.detach().cpu().numpy())
            else:
                if set(optimizer_value) != {"step", "exp_avg", "exp_avg_sq"} or float(optimizer_value["step"]) != 1.0:
                    raise MlxEngineError("initial five-lane optimizer identity differs")
                expected_first = optimizer_value["exp_avg"].detach().cpu().numpy()
                expected_second = optimizer_value["exp_avg_sq"].detach().cpu().numpy()
            first_errors.append(maximum_array_error(next_first[index][lane], expected_first))
            second_errors.append(maximum_array_error(next_second[index][lane], expected_second))
            torch_oracle_parameter, torch_oracle_first, torch_oracle_second = adamw_one_step_oracle(initial_parameter, torch_clipped_gradients[torch_name], policies[mlx_name]["peak_lr"], policies[mlx_name]["weight_decay"])
            torch_optimizer_gradient_sha256.update(mlx_name.encode("utf-8"))
            torch_optimizer_gradient_sha256.update(torch_clipped_gradients[torch_name].tobytes())
            torch_parameter_bound, torch_first_bound, torch_second_bound = adamw_one_step_error_bounds(initial_parameter, torch_clipped_gradients[torch_name], policies[mlx_name]["peak_lr"], policies[mlx_name]["weight_decay"])
            torch_optimizer_parameter_formula_errors.append(float(np.max(np.abs(expected_parameter.astype(np.float64) - torch_oracle_parameter))))
            torch_optimizer_first_formula_errors.append(float(np.max(np.abs(expected_first.astype(np.float64) - torch_oracle_first))))
            torch_optimizer_second_formula_errors.append(float(np.max(np.abs(expected_second.astype(np.float64) - torch_oracle_second))))
            for surface, observed_value, expected_value, error_bound in (
                ("parameter", expected_parameter, torch_oracle_parameter, torch_parameter_bound),
                ("first_moment", expected_first, torch_oracle_first, torch_first_bound),
                ("second_moment", expected_second, torch_oracle_second, torch_second_bound),
            ):
                formula_difference = np.abs(observed_value.astype(np.float64) - expected_value.astype(np.float64))
                allowed = np.nextafter(error_bound, np.full_like(error_bound, np.inf))
                formula_ratio = np.zeros_like(formula_difference)
                np.divide(formula_difference, allowed, out=formula_ratio, where=formula_difference != 0.0)
                formula_flat_index = int(np.argmax(formula_ratio))
                optimizer_parameter_formula_records.append(
                    {
                        "runtime": "Torch",
                        "surface": surface,
                        "tensor": torch_name,
                        "lane": lane,
                        "index": [int(value) for value in np.unravel_index(formula_flat_index, formula_difference.shape)],
                        "max_abs": float(np.max(formula_difference)),
                        "max_bound_ratio": float(formula_ratio.reshape(-1)[formula_flat_index]),
                        "worst_abs": float(formula_difference.reshape(-1)[formula_flat_index]),
                        "worst_bound": float(allowed.reshape(-1)[formula_flat_index]),
                        "observed": float(observed_value.reshape(-1)[formula_flat_index]),
                        "expected": float(expected_value.reshape(-1)[formula_flat_index]),
                    }
                )
            mlx_oracle_parameter, mlx_oracle_first, mlx_oracle_second = (value[lane] for value in mlx_optimizer_oracles[mlx_name])
            mlx_parameter_bound, mlx_first_bound, mlx_second_bound = (value[lane] for value in mlx_optimizer_bounds[mlx_name])
            parameter_residual_errors.append(maximum_array_error(observed_parameter.astype(np.float64) - expected_parameter.astype(np.float64), mlx_oracle_parameter.astype(np.float64) - torch_oracle_parameter.astype(np.float64)))
            first_residual_errors.append(maximum_array_error(np.array(next_first[index])[lane].astype(np.float64) - expected_first.astype(np.float64), mlx_oracle_first.astype(np.float64) - torch_oracle_first.astype(np.float64)))
            second_residual_errors.append(maximum_array_error(np.array(next_second[index])[lane].astype(np.float64) - expected_second.astype(np.float64), mlx_oracle_second.astype(np.float64) - torch_oracle_second.astype(np.float64)))
            for surface, mlx_actual, torch_actual, mlx_oracle, torch_oracle, mlx_bound, torch_bound in (
                ("parameter", observed_parameter, expected_parameter, mlx_oracle_parameter, torch_oracle_parameter, mlx_parameter_bound, torch_parameter_bound),
                ("first_moment", np.array(next_first[index])[lane], expected_first, mlx_oracle_first, torch_oracle_first, mlx_first_bound, torch_first_bound),
                ("second_moment", np.array(next_second[index])[lane], expected_second, mlx_oracle_second, torch_oracle_second, mlx_second_bound, torch_second_bound),
            ):
                residual = np.abs((mlx_actual.astype(np.float64) - torch_actual.astype(np.float64)) - (mlx_oracle.astype(np.float64) - torch_oracle.astype(np.float64)))
                bound = mlx_bound.astype(np.float64) + torch_bound.astype(np.float64)
                allowed = np.nextafter(bound, np.full_like(bound, np.inf))
                excess = residual - allowed
                ratio = np.zeros_like(residual)
                np.divide(residual, allowed, out=ratio, where=residual != 0.0)
                residual_flat_index = int(np.argmax(ratio))
                causal_residual_records.append(
                    {
                        "surface": surface,
                        "tensor": torch_name,
                        "lane": lane,
                        "index": [int(value) for value in np.unravel_index(residual_flat_index, residual.shape)],
                        "max_abs": float(np.max(residual)),
                        "max_bound": float(np.max(allowed)),
                        "max_bound_ratio": float(ratio.reshape(-1)[residual_flat_index]),
                        "max_excess": float(np.max(excess)),
                        "worst_excess": float(excess.reshape(-1)[residual_flat_index]),
                        "worst_residual": float(residual.reshape(-1)[residual_flat_index]),
                        "worst_bound": float(allowed.reshape(-1)[residual_flat_index]),
                    }
                )
    torch_loss_max_abs = max(loss_errors, default=0.0)
    torch_parameter_max_abs = max(parameter_errors, default=0.0)
    torch_first_moment_max_abs = max(first_errors, default=0.0)
    torch_second_moment_max_abs = max(second_errors, default=0.0)
    five_lane_gradient = gradient_comparison_evidence(gradient_comparisons, 1.25e-4, 2.5e-4, 1.25e-4, 0.99999999)
    five_lane_clipped_gradient = gradient_comparison_evidence(clipped_gradient_comparisons, 1.25e-4, 2.5e-4, 1.25e-4, 0.99999999)
    parameter_update_statistics = tensor_comparison_statistics(parameter_update_comparisons)
    worst_end_to_end = max(end_to_end_records, key=lambda value: (value["max_abs"], value["tensor"], value["lane"]))
    mlx_optimizer_parameter_formula_max_abs = max(optimizer_parameter_formula_errors)
    mlx_optimizer_first_formula_max_abs = max(optimizer_first_formula_errors)
    mlx_optimizer_second_formula_max_abs = max(optimizer_second_formula_errors)
    torch_optimizer_parameter_formula_max_abs = max(torch_optimizer_parameter_formula_errors)
    torch_optimizer_first_formula_max_abs = max(torch_optimizer_first_formula_errors)
    torch_optimizer_second_formula_max_abs = max(torch_optimizer_second_formula_errors)
    optimizer_formula_max_bound_ratio = max(record["max_bound_ratio"] for record in optimizer_parameter_formula_records)
    optimizer_formula_pass = optimizer_formula_max_bound_ratio <= 1.0
    worst_optimizer_formula = max(optimizer_parameter_formula_records, key=lambda value: (value["max_bound_ratio"], value["runtime"], value["surface"], value["tensor"], value["lane"]))
    parameter_residual_max_abs = max(parameter_residual_errors)
    first_residual_max_abs = max(first_residual_errors)
    second_residual_max_abs = max(second_residual_errors)
    causal_residual_summary = {}
    for surface in ("parameter", "first_moment", "second_moment"):
        surface_records = [record for record in causal_residual_records if record["surface"] == surface]
        causal_residual_summary[surface] = {
            "max_abs": max(record["max_abs"] for record in surface_records),
            "max_bound": max(record["max_bound"] for record in surface_records),
            "max_bound_ratio": max(record["max_bound_ratio"] for record in surface_records),
            "worst_excess": max(record["max_excess"] for record in surface_records),
        }
    worst_causal_residual = max(causal_residual_records, key=lambda value: (value["max_bound_ratio"], value["surface"], value["tensor"], value["lane"]))
    causal_residual_pass = all(record["worst_excess"] <= 0.0 for record in causal_residual_summary.values())
    mlx_preclip_norms = [float(value) for value in np.array(gradient_norm)]
    mlx_postclip_norms = [
        math.sqrt(sum(float(np.square(np.array(value)[lane].astype(np.float64)).sum()) for value in clipped_gradients))
        for lane in range(5)
    ]
    finite = bool(all(np.isfinite(np.array(value)).all() for value in output[:4]) and np.isfinite(np.array(gradient_norm)).all() and np.asarray(audit_status)[:, :, (0, 2)].all())
    torch_parity = torch_loss_max_abs <= 1e-6 and five_lane_gradient["gradient_pass"] and five_lane_clipped_gradient["gradient_pass"] and five_lane_grad_none_zero_exact and optimizer_formula_pass and causal_residual_pass and route_exact
    return {
        "lanes": 5,
        "stage": "joint",
        "objective": "task_plus_0.1_times_internal_router_plus_supervised_route",
        "construction_seeds": list(RUNG_ONE_SEEDS),
        "data_seeds": [300000 + seed for seed in RUNG_ONE_SEEDS],
        "batch_size_per_lane": 2,
        "sequence_length": 128,
        "logical_update": 1,
        "learning_rates": {"block_4_router": 0.001, "other_trainable": 0.00025},
        "initial_optimizer_step": 0,
        "initial_first_and_second_moments_exact_zero": all(np.array_equal(np.array(value), np.zeros_like(np.array(value))) for value in (*first, *second)),
        "mapping_value_count": sum(evidence["parameter_count"] for evidence in value_mappings),
        "mapping_value_max_abs": max(evidence["max_abs"] for evidence in value_mappings),
        "mapping_value_byte_exact": bool(mapping_value_byte_exact),
        "mapping_source_value_sha256": mapping_source_digest.hexdigest(),
        "mapping_destination_value_sha256": mapping_destination_digest.hexdigest(),
        "unique_parameter_hashes": len(set(lane_hashes)),
        "codebook_grad_none_effect_exact": codebook_exact,
        "finite": finite,
        "torch_loss_max_abs": torch_loss_max_abs,
        "torch_parameter_max_abs": torch_parameter_max_abs,
        "torch_first_moment_max_abs": torch_first_moment_max_abs,
        "torch_second_moment_max_abs": torch_second_moment_max_abs,
        "torch_route_exact": bool(route_exact),
        "five_lane_gradient_count": five_lane_gradient["gradient_count"],
        "five_lane_gradient_max_abs": five_lane_gradient["gradient_max_abs"],
        "five_lane_gradient_relative_max": five_lane_gradient["gradient_relative_max"],
        "five_lane_gradient_normalized_l2_max": five_lane_gradient["gradient_normalized_l2_max"],
        "five_lane_gradient_cosine_min": five_lane_gradient["gradient_cosine_min"],
        "five_lane_gradient_worst_tensor": five_lane_gradient["gradient_worst_tensor"],
        "five_lane_gradient_worst_index": five_lane_gradient["gradient_worst_index"],
        "five_lane_gradient_worst_observed": five_lane_gradient["gradient_worst_observed"],
        "five_lane_gradient_worst_expected": five_lane_gradient["gradient_worst_expected"],
        "five_lane_gradient_metric_worst": five_lane_gradient["gradient_metric_worst"],
        "five_lane_gradient_absolute_pass": five_lane_gradient["gradient_absolute_pass"],
        "five_lane_gradient_scale_aware_pass": five_lane_gradient["gradient_scale_aware_pass"],
        "five_lane_gradient_pass": five_lane_gradient["gradient_pass"],
        "five_lane_gradient_scale_aware_absolute_tolerance": five_lane_gradient["gradient_scale_aware_absolute_tolerance"],
        "five_lane_gradient_relative_tolerance": five_lane_gradient["gradient_relative_tolerance"],
        "five_lane_gradient_normalized_l2_tolerance": five_lane_gradient["gradient_normalized_l2_tolerance"],
        "five_lane_gradient_cosine_tolerance": five_lane_gradient["gradient_cosine_tolerance"],
        "five_lane_grad_none_zero_exact": bool(five_lane_grad_none_zero_exact),
        "five_lane_raw_gradient_sha256": raw_gradient_sha256.hexdigest(),
        "five_lane_clipped_gradient_count": five_lane_clipped_gradient["gradient_count"],
        "five_lane_clipped_gradient_max_abs": five_lane_clipped_gradient["gradient_max_abs"],
        "five_lane_clipped_gradient_relative_max": five_lane_clipped_gradient["gradient_relative_max"],
        "five_lane_clipped_gradient_normalized_l2_max": five_lane_clipped_gradient["gradient_normalized_l2_max"],
        "five_lane_clipped_gradient_cosine_min": five_lane_clipped_gradient["gradient_cosine_min"],
        "five_lane_clipped_gradient_metric_worst": five_lane_clipped_gradient["gradient_metric_worst"],
        "five_lane_clipped_gradient_pass": five_lane_clipped_gradient["gradient_pass"],
        "parameter_update_max_abs": parameter_update_statistics["max_abs"],
        "parameter_update_relative_max": parameter_update_statistics["relative_max"],
        "parameter_update_normalized_l2_max": parameter_update_statistics["normalized_l2_max"],
        "parameter_update_cosine_min": parameter_update_statistics["cosine_min"],
        "parameter_update_metric_worst": parameter_update_statistics["metric_worst"],
        "optimizer_gradient_sha256": optimizer_gradient_sha256.hexdigest(),
        "torch_optimizer_gradient_sha256": torch_optimizer_gradient_sha256.hexdigest(),
        "mlx_optimizer_parameter_formula_max_abs": mlx_optimizer_parameter_formula_max_abs,
        "mlx_optimizer_first_formula_max_abs": mlx_optimizer_first_formula_max_abs,
        "mlx_optimizer_second_formula_max_abs": mlx_optimizer_second_formula_max_abs,
        "torch_optimizer_parameter_formula_max_abs": torch_optimizer_parameter_formula_max_abs,
        "torch_optimizer_first_formula_max_abs": torch_optimizer_first_formula_max_abs,
        "torch_optimizer_second_formula_max_abs": torch_optimizer_second_formula_max_abs,
        "optimizer_formula_max_bound_ratio": optimizer_formula_max_bound_ratio,
        "optimizer_formula_bound_ratio_tolerance": 1.0,
        "optimizer_formula_pass": bool(optimizer_formula_pass),
        "optimizer_formula_worst_runtime": worst_optimizer_formula["runtime"],
        "optimizer_formula_worst_surface": worst_optimizer_formula["surface"],
        "optimizer_formula_worst_tensor": worst_optimizer_formula["tensor"],
        "optimizer_formula_worst_lane": worst_optimizer_formula["lane"],
        "optimizer_formula_worst_index": worst_optimizer_formula["index"],
        "optimizer_formula_worst_abs": worst_optimizer_formula["worst_abs"],
        "optimizer_formula_worst_bound": worst_optimizer_formula["worst_bound"],
        "optimizer_formula_worst_bound_ratio": worst_optimizer_formula["max_bound_ratio"],
        "optimizer_formula_worst_observed": worst_optimizer_formula["observed"],
        "optimizer_formula_worst_expected": worst_optimizer_formula["expected"],
        "causal_parameter_residual_max_abs": parameter_residual_max_abs,
        "causal_first_moment_residual_max_abs": first_residual_max_abs,
        "causal_second_moment_residual_max_abs": second_residual_max_abs,
        "causal_residual_summary": causal_residual_summary,
        "causal_residual_pass": bool(causal_residual_pass),
        "causal_residual_worst_surface": worst_causal_residual["surface"],
        "causal_residual_worst_tensor": worst_causal_residual["tensor"],
        "causal_residual_worst_lane": worst_causal_residual["lane"],
        "causal_residual_worst_index": worst_causal_residual["index"],
        "causal_residual_worst_abs": worst_causal_residual["worst_residual"],
        "causal_residual_worst_bound": worst_causal_residual["worst_bound"],
        "causal_residual_worst_bound_ratio": worst_causal_residual["max_bound_ratio"],
        "causal_residual_worst_excess": worst_causal_residual["worst_excess"],
        "end_to_end_worst_max_abs": worst_end_to_end["max_abs"],
        "end_to_end_worst_lane": worst_end_to_end["lane"],
        "end_to_end_worst_tensor": worst_end_to_end["tensor"],
        "end_to_end_worst_index": worst_end_to_end["index"],
        "end_to_end_worst_observed": worst_end_to_end["observed"],
        "end_to_end_worst_expected": worst_end_to_end["expected"],
        "end_to_end_worst_mlx_clipped_gradient": worst_end_to_end["mlx_clipped_gradient"],
        "end_to_end_worst_torch_clipped_gradient": worst_end_to_end["torch_clipped_gradient"],
        "mlx_preclip_gradient_norms": mlx_preclip_norms,
        "mlx_postclip_gradient_norms": mlx_postclip_norms,
        "torch_preclip_gradient_norms": torch_preclip_norms,
        "torch_postclip_gradient_norms": torch_postclip_norms,
        "torch_parity_pass": bool(torch_parity),
        "pass": bool(finite and mapping_value_byte_exact and codebook_exact and len(set(lane_hashes)) == 5 and torch_parity),
    }


def pilot_emit(message: Mapping[str, Any]) -> None:
    print(json.dumps(dict(message), sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def pilot_exchange(message: Mapping[str, Any]) -> None:
    pilot_emit(message)
    raw = sys.stdin.readline()
    if not raw:
        raise MlxEngineError("pilot parent transport closed")
    response = json.loads(raw)
    expected = {
        "ack": True,
        "kind": "pilot_update_start_committed",
        "workload": message["workload"],
        "logical_update": message["logical_update"],
    }
    if response != expected:
        raise MlxEngineError("pilot update acknowledgement differs")


def pilot_roots() -> tuple[Path, Path]:
    values = []
    for name in ("MODULAR_MLX_RUN_ROOT", "MODULAR_MLX_SCRATCH_ROOT"):
        text = os.environ.get(name)
        if not isinstance(text, str) or not text:
            raise MlxEngineError("pilot child root is absent")
        value = Path(text)
        if not value.is_absolute() or not value.is_dir() or value.is_symlink():
            raise MlxEngineError("pilot child root differs")
        values.append(value)
    run_root, scratch_root = values
    if run_root == scratch_root or Path("/private/tmp") not in scratch_root.parents:
        raise MlxEngineError("pilot scratch root differs")
    return run_root, scratch_root


def pilot_parameter_state(stage: str, role: str, model_seed: int, lanes: int) -> dict[str, Any]:
    source = torch_model(role, model_seed)
    destination = MlxModularModel(role)
    load_torch_state(destination, source.state_dict())
    flattened = tuple(tree_flatten(destination.parameters()))
    names = tuple(name for name, _ in flattened)
    values = {name: value for name, value in flattened}
    policies = {name: optimizer_parameter_policy(torch_policy_name(name), stage) for name in names}
    train_names = tuple(name for name in names if policies[name]["trainable"])
    frozen_names = tuple(name for name in names if not policies[name]["trainable"])
    train = tuple(mx.stack(tuple(values[name] for _ in range(lanes))) for name in train_names)
    frozen = tuple(mx.stack(tuple(values[name] for _ in range(lanes))) for name in frozen_names)
    first = tuple(mx.zeros_like(value) for value in train)
    second = tuple(mx.zeros_like(value) for value in train)
    decay = tuple(float(policies[name]["weight_decay"]) for name in train_names)
    mx.eval(train, frozen, first, second)
    del source, destination, flattened, values
    return {
        "stage": stage,
        "role": role,
        "model_seed": model_seed,
        "lanes": lanes,
        "names": names,
        "train_names": train_names,
        "frozen_names": frozen_names,
        "train": train,
        "frozen": frozen,
        "first": first,
        "second": second,
        "decay": decay,
    }


def pilot_learning_rates(stage: str, train_names: tuple[str, ...]) -> tuple[mx.array, ...]:
    if stage != "joint":
        return tuple(mx.array(0.002, dtype=mx.float32) for _ in train_names)
    rates = []
    for name in train_names:
        torch_name = torch_policy_name(name)
        router = torch_name.startswith("blocks.4.mix.source_mixer.attention.router.")
        rates.append(mx.array(0.001 if router else 0.00025, dtype=mx.float32))
    return tuple(rates)


def pilot_batch(
    workload: str,
    lanes: int,
    batch_size: int,
    sequence_length: int,
    data_generator: torch.Generator,
    route_generator: torch.Generator,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    vocabulary = 256 if workload == "rung_two" else 128
    tokens = torch.randint(0, vocabulary, (batch_size, sequence_length), generator=data_generator, dtype=torch.int64)
    if workload == "rung_two":
        targets = torch.randint(0, vocabulary, (batch_size, sequence_length), generator=data_generator, dtype=torch.int64)
        required_source = torch.zeros((batch_size,), dtype=torch.int64)
    else:
        targets = torch.randint(0, vocabulary, (batch_size,), generator=data_generator, dtype=torch.int64)
        required_source = torch.randint(0, 15, (batch_size,), generator=data_generator, dtype=torch.int64) if workload == "selected_vmap5" else torch.zeros((batch_size,), dtype=torch.int64)
    route_override = torch.full((batch_size, sequence_length, 1, 2), -1, dtype=torch.int64)
    if workload == "selected_vmap5":
        route_override[:, 126, 0] = torch.stack(tuple(torch.sort(torch.randperm(15, generator=route_generator)[:2]).values for _ in range(batch_size)))
    token_lanes = np.repeat(tokens.numpy()[None], lanes, axis=0)
    target_lanes = np.repeat(targets.numpy()[None], lanes, axis=0)
    required_lanes = np.repeat(required_source.numpy()[None], lanes, axis=0)
    return mx.array(token_lanes), mx.array(target_lanes), mx.array(required_lanes), mx.array(route_override.numpy())


def pilot_canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pilot_fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def pilot_owned_scratch(scratch_root: Path, family: str, repetition: int) -> Path:
    path = scratch_root / f"pilot-{family}-{repetition}-{os.getpid()}"
    path.mkdir(mode=0o700, exist_ok=False)
    pilot_fsync_directory(scratch_root)
    return path


def pilot_remove_owned_scratch(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise MlxEngineError("pilot owned scratch differs")
    if any(path.iterdir()):
        raise MlxEngineError("pilot owned scratch is not empty")
    parent = path.parent
    path.rmdir()
    pilot_fsync_directory(parent)


def pilot_torch_model_from_state(state: Mapping[str, Any], lane: int, role: str | None = None) -> ModularNeuralMachine:
    target_role = state["role"] if role is None else role
    model = torch_model(target_role, state["model_seed"])
    lane_parameters = full_lane_parameters(state, lane)
    model_state = model.state_dict()
    for torch_name, template in model_state.items():
        mlx_name = mapped_mlx_parameter_name(torch_name)
        if mlx_name not in lane_parameters:
            raise MlxEngineError("pilot Torch model parameter closure differs")
        array = np.array(lane_parameters[mlx_name])
        if array.shape != tuple(template.shape) or array.dtype != np.float32:
            raise MlxEngineError("pilot Torch model parameter descriptor differs")
        model_state[torch_name] = torch.from_numpy(array.copy())
    model.load_state_dict(model_state, strict=True)
    return model


def pilot_stage_view(state: Mapping[str, Any], stage: str) -> dict[str, Any]:
    policies = {name: optimizer_parameter_policy(torch_policy_name(name), stage) for name in state["names"]}
    train_names = tuple(name for name in state["names"] if policies[name]["trainable"])
    frozen_names = tuple(name for name in state["names"] if not policies[name]["trainable"])
    full = full_lane_parameters(state, 0)
    moment_indexes = {name: index for index, name in enumerate(state["train_names"])}
    if any(name not in moment_indexes for name in train_names):
        raise MlxEngineError("pilot stage view lacks trained optimizer moments")
    return {
        "train_names": train_names,
        "frozen_names": frozen_names,
        "train": tuple(full[name] for name in train_names),
        "frozen": tuple(full[name] for name in frozen_names),
        "first": tuple(state["first"][moment_indexes[name]][0] for name in train_names),
        "second": tuple(state["second"][moment_indexes[name]][0] for name in train_names),
    }


def pilot_evaluation_fixture() -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    rung_one_payload = cpu.generate_rung_one_batch(123456, 2, torch)
    rung_two_payload = cpu.generate_rung_two_batch(123456, 2, torch)
    random_payload = cpu.generate_random_routes(500011, 2, torch)
    rung_one_two = cpu.payload_to_tensors(rung_one_payload, torch)
    rung_two_two = cpu.payload_to_tensors(rung_two_payload, torch)
    rung_one = {name: value.repeat((16,) + (1,) * (value.ndim - 1)) for name, value in rung_one_two.items()}
    rung_two = {name: value.repeat((16,) + (1,) * (value.ndim - 1)) for name, value in rung_two_two.items() if isinstance(value, torch.Tensor)}
    random_routes = torch.full((32, 128, 1, 2), -1, dtype=torch.long)
    random_routes[:, 126, 0] = torch.tensor(random_payload["routes"], dtype=torch.long).repeat(16, 1)
    rung_one_row = {name: value[:1].clone() for name, value in rung_one_two.items()}
    rung_two_row = {name: value[:1].clone() for name, value in rung_two_two.items() if isinstance(value, torch.Tensor)}
    return {
        "rung_one": rung_one,
        "rung_two": rung_two,
        "rung_one_row": rung_one_row,
        "rung_two_row": rung_two_row,
        "random_routes": random_routes,
        "fixture_sha256s": {
            "random_routes_seed_500011": cpu.canonical_json_sha256(random_payload),
            "rung_one_seed_123456": cpu.canonical_json_sha256(rung_one_payload),
            "rung_two_seed_123456": cpu.canonical_json_sha256(rung_two_payload),
        },
    }


def pilot_evaluation_reduce(output: ModularModelOutput, batch: Mapping[str, torch.Tensor], rung: int, condition: str) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    cpu._assert_finite_tree(torch, output, {"stage": condition}, "pilot_evaluation_output")
    position = 510 if rung == 2 else 126
    target = batch["targets"][:, position] if rung == 2 else batch["targets"]
    prediction = output.logits[:, position].argmax(dim=-1)
    correct = prediction.eq(target)
    routed = [block.mixer_output for block in output.blocks if block.kind == "routed"]
    overflow = 0
    maximum = 0
    underfill = 0
    source_hits = 0
    for value in routed:
        telemetry = value.telemetry
        overflow += int(telemetry["overflow_count"])
        maximum = max(maximum, int(telemetry["max_bucket_load"]))
    if rung == 1 and condition != "dense_causal":
        block = cpu._block_output(output, 4)
        effective = block.telemetry["effective_remote"][:, 126, 0]
        underfill = int((effective == -1).sum())
        source_hits = int((effective == batch["required_source"][:, None]).any(dim=-1).sum())
    prediction_records = []
    for index in range(target.shape[0]):
        prediction_records.append(
            {
                "schema_version": cpu.SCHEMA_VERSION,
                "run_id": "pilot-nonclaim",
                "rung": rung,
                "claim_seed": 123456,
                "construction_seed": 123456,
                "condition": condition,
                "example_index": index,
                "original_condition": None if rung == 2 else int(batch["condition"][index]),
                "foreign_condition": None,
                "original_source": None if rung == 2 else int(batch["required_source"][index]),
                "foreign_source": None,
                "target": int(target[index]),
                "prediction": int(prediction[index]),
                "correct": bool(correct[index]),
                "original_source_hit": None if rung == 2 or condition == "dense_causal" else bool((cpu._block_output(output, 4).telemetry["effective_remote"][index, 126, 0] == batch["required_source"][index]).any()),
                "foreign_source_hit": None,
                "condition_stratum": "not_applicable",
                "checkpoint_sha256": "0" * 64,
            }
        )
    evaluation_record = {
        "condition": condition,
        "answer_correct": int(correct.sum()),
        "answer_total": int(correct.numel()),
        "original_source_hits": source_hits if rung == 1 and condition != "dense_causal" else None,
        "original_source_total": int(correct.numel()) if rung == 1 and condition != "dense_causal" else None,
        "query_underfill_count": underfill if rung == 1 and condition != "dense_causal" else None,
        "overflow_count": overflow,
        "max_bucket_load": maximum,
    }
    state_records = []
    intervention_records = []
    for block in output.blocks:
        if block.kind != "recurrent":
            continue
        recurrent = block.mixer_output
        for statistic, tensor in (("primary_gate", recurrent.primary_gate), ("beta_gate", recurrent.write_gate), ("output_gate", recurrent.output_gate)):
            state_records.append(
                {
                    "model": "rung_two" if rung == 2 else "pilot",
                    "checkpoint_sha256": "0" * 64,
                    "block": block.block_index,
                    "condition": condition,
                    "boundary": "not_applicable",
                    "position": None,
                    "statistic": statistic,
                    "count": int(tensor.numel()),
                    "mean": float(tensor.to(torch.float64).mean()),
                    "std": float(tensor.to(torch.float64).std(unbiased=False)),
                    "minimum": float(tensor.min()),
                    "maximum": float(tensor.max()),
                    "nonfinite_count": int((~torch.isfinite(tensor)).sum()),
                }
            )
        computed = block.computed_sequence_delta.to(torch.float64)
        exposed = block.exposed_sequence_delta.to(torch.float64)
        intervention_records.append(
            {
                "baseline_model": "pilot",
                "baseline_checkpoint_sha256": "0" * 64,
                "baseline_condition": "intact",
                "intervention_model": "pilot",
                "intervention_checkpoint_sha256": "0" * 64,
                "block": block.block_index,
                "condition": condition,
                "pre_delta_l2": float(torch.linalg.vector_norm(computed)),
                "post_delta_l2": float(torch.linalg.vector_norm(computed)),
                "exposed_delta_l2": float(torch.linalg.vector_norm(exposed)),
            }
        )
    if not prediction_records or not evaluation_record or not state_records or not intervention_records:
        raise MlxEngineError("pilot evaluation record construction differs")
    return {"prediction_records": prediction_records, "evaluation_record": evaluation_record, "state_records": state_records, "intervention_records": intervention_records}


def pilot_record_duration(records: dict[str, dict[str, list[int]]], family: str, repetition: int, duration_ns: int) -> None:
    if type(duration_ns) is not int or duration_ns <= 0:
        raise MlxEngineError("pilot tail duration differs")
    target = "warmup" if repetition == 0 else "timed"
    records.setdefault(family, {"warmup": [], "timed": []})[target].append(duration_ns)


def pilot_evaluation_once(model: ModularNeuralMachine, role: str, batch: Mapping[str, torch.Tensor], condition: str, random_routes: torch.Tensor | None, exclusion_routes: torch.Tensor | None) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    kwargs: dict[str, Any] = {"return_aux": True, "recurrent_telemetry": True, "route_detail": role != "dense"}
    if condition == "target_forced":
        forced = torch.full((batch["tokens"].shape[0], 128), -1, dtype=torch.long)
        forced[:, 126] = batch["required_source"]
        kwargs["forced_blocks"] = forced
    elif condition == "recurrent_knockout":
        kwargs["recurrent_knockout"] = True
    elif condition == "carry_reset":
        kwargs["recurrent_intervention"] = "reset"
    elif condition == "carry_shuffle":
        kwargs["recurrent_intervention"] = "shuffle"
    elif condition == "matched_random_route":
        if random_routes is None:
            raise MlxEngineError("pilot random route fixture is absent")
        kwargs["route_override"] = random_routes
    elif condition == "block4_routed_knockout":
        kwargs["block4_routed_knockout"] = True
    elif condition == "required_source_excluded":
        if exclusion_routes is None:
            raise MlxEngineError("pilot exclusion route fixture is absent")
        kwargs["route_override"] = exclusion_routes
    with torch.inference_mode():
        if role == "rung_two":
            source_kwargs: dict[str, Any] = {"return_aux": True, "route_detail": True}
            if condition == "recurrent_knockout":
                source_kwargs["recurrent_knockout"] = True
            source = model(batch["tokens"], **source_kwargs)
        output = model(batch["tokens"], **kwargs)
        if role == "rung_two":
            cpu._rung_two_source_prediction(torch, source, output, batch["targets"][:, 510], {"seed": 123456, "stage": condition, "logical_update": 0})
        pilot_evaluation_reduce(output, batch, 2 if role == "rung_two" else 1, condition)


def pilot_evaluation_benchmark(state: Mapping[str, Any], workload: str, fixture: Mapping[str, Any], scratch_root: Path) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    records: dict[str, dict[str, list[int]]] = {}
    exclusion_routes = None
    source_exclusion_fixture = None
    source_exclusion_route_bytes = 0
    cleanup = []
    fixture_sha256s = dict(fixture["fixture_sha256s"])
    if workload == "selected_vmap5":
        preflight_model = pilot_torch_model_from_state(state, 0, "selected")
        preflight_model.eval()
        with torch.inference_mode():
            preflight_output = preflight_model(fixture["rung_one"]["tokens"], return_aux=True, route_detail=True, recurrent_telemetry=True)
        raw = cpu._block_output(preflight_output, 4).telemetry["raw_remote"]
        exclusion_payload = cpu.generate_source_exclusion_routes(510000 + 123456, raw, fixture["rung_one"]["required_source"], torch)
        source_exclusion_fixture = exclusion_payload
        exclusion_routes = torch.full((32, 128, 1, 2), -1, dtype=torch.long)
        exclusion_routes[:, 126, 0] = torch.tensor(exclusion_payload["routes"], dtype=torch.long)
        source_exclusion_route_bytes = int(exclusion_routes.numel() * exclusion_routes.element_size())
        fixture_sha256s["source_exclusion_seed_633456"] = cpu.canonical_json_sha256(exclusion_payload)
        del preflight_model, preflight_output, raw
    families = []
    if workload == "selected_vmap5":
        families.append(("route_acquisition", "selected", "route_acquisition"))
        for condition in (
            "intact",
            "target_forced",
            "recurrent_knockout",
            "carry_reset",
            "carry_shuffle",
            "matched_random_route",
            "block4_routed_knockout",
            "block4_local_only",
            "required_source_excluded",
        ):
            families.append((f"rung_one_routed.{condition}", "local_only" if condition == "block4_local_only" else "selected", condition))
        families.append(("rung_one_routed.all_eligible_clone", "all_eligible", "all_eligible_clone"))
    elif workload == "donor":
        families.append(("rung_one_routed.all_eligible_donor", "all_eligible", "all_eligible_donor"))
    elif workload == "dense_vmap5":
        families.append(("rung_one_dense", "dense", "dense_causal"))
    elif workload == "rung_two":
        families.extend((("rung_two.intact", "rung_two", "intact"), ("rung_two.recurrent_knockout", "rung_two", "recurrent_knockout")))
    else:
        raise MlxEngineError("pilot evaluation workload differs")
    for family, role, condition in families:
        batch = fixture["rung_two"] if role == "rung_two" else fixture["rung_one"]
        for repetition in range(4):
            scratch = pilot_owned_scratch(scratch_root, f"evaluation-{family.replace('.', '-')}", repetition)
            reference = pilot_torch_model_from_state(state, 0, role)
            model = MlxTorchEvaluationAdapter(role, state["model_seed"], {"model_state_dict": reference.state_dict()})
            model.eval()
            del reference
            mx.eval(state["train"], state["frozen"], state["first"], state["second"])
            started_ns = time.perf_counter_ns()
            pilot_evaluation_once(model, role, batch, condition, fixture["random_routes"], exclusion_routes)
            pilot_remove_owned_scratch(scratch)
            duration_ns = time.perf_counter_ns() - started_ns
            pilot_record_duration(records, family, repetition, duration_ns)
            cleanup.append(not scratch.exists())
    for family, values in records.items():
        if len(values["warmup"]) != 1 or len(values["timed"]) != 3:
            raise MlxEngineError("pilot evaluation repetition count differs")
    return {"records": records, "fixture_sha256s": fixture_sha256s, "source_exclusion_fixture": source_exclusion_fixture, "source_exclusion_route_bytes": source_exclusion_route_bytes, "scratch_cleanup_pass": all(cleanup)}


def pilot_optimizer_from_state(state: Mapping[str, Any], lane: int, stage: str, model: ModularNeuralMachine) -> tuple[Any, Mapping[str, Any]]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    optimizer, _, membership = cpu._make_optimizer(model, stage, cpu._import_runtime())
    first_indexes = {name: index for index, name in enumerate(state["train_names"])}
    for torch_name, parameter in model.named_parameters():
        mlx_name = mapped_mlx_parameter_name(torch_name)
        if not membership[torch_name]["requires_grad"]:
            continue
        if mlx_name not in first_indexes:
            raise MlxEngineError("pilot optimizer moment membership differs")
        index = first_indexes[mlx_name]
        optimizer.state[parameter] = {
            "step": torch.tensor(11.0, dtype=torch.float32),
            "exp_avg": torch.from_numpy(np.array(state["first"][index][lane]).copy()),
            "exp_avg_sq": torch.from_numpy(np.array(state["second"][index][lane]).copy()),
        }
    return optimizer, membership


def pilot_endpoint_replay_context(state: Mapping[str, Any], stage: str, role: str, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    view = pilot_stage_view(state, stage)
    tokens = mx.array(batch["tokens"].numpy())
    targets = mx.array(batch["targets"].numpy())
    required = mx.zeros((1,), dtype=mx.int32) if stage == "rung_two" else mx.array(batch["required_source"].numpy())
    reference = pilot_torch_model_from_state(state, 0, role)
    optimizer, membership = pilot_optimizer_from_state(state, 0, stage, reference)
    return {"view": view, "tokens": tokens, "targets": targets, "required": required, "reference": reference, "optimizer": optimizer, "membership": membership}


def pilot_endpoint_replay_once(context: Mapping[str, Any], stage: str, role: str, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    view = context["view"]
    tokens = context["tokens"]
    targets = context["targets"]
    required = context["required"]

    def loss_function(current_train: tuple[mx.array, ...]) -> Any:
        return stage_loss(current_train, view["frozen"], view["train_names"], view["frozen_names"], role, stage, tokens, targets, required)

    mlx_losses, mlx_gradients = mx.value_and_grad(loss_function)(view["train"])
    parameters = {name: value for name, value in zip(view["train_names"], view["train"])}
    parameters.update({name: value for name, value in zip(view["frozen_names"], view["frozen"])})
    mlx_output = functional_forward(parameters, role, tokens, stage == "joint")
    mx.eval(mlx_losses, mlx_gradients, mlx_output, view["first"], view["second"])
    if not tensor_tuple_finite((mlx_losses, mlx_gradients, mlx_output, view["first"], view["second"])):
        raise MlxEngineError("pilot endpoint MLX nonfinite")
    reference = context["reference"]
    optimizer = context["optimizer"]
    membership = context["membership"]
    reference.train()
    torch_total, _, _, _, torch_output = cpu._training_forward_loss(reference, batch, stage, cpu._import_runtime())
    torch_total.backward()
    cpu._assert_finite_tree(torch, torch_output, {"stage": stage}, "pilot_endpoint_output")
    cpu._assert_finite_tree(torch, torch_total, {"stage": stage}, "pilot_endpoint_loss")
    task_loss = torch.nn.functional.cross_entropy(
        torch_output.logits.reshape(-1, torch_output.logits.shape[-1]) if stage == "rung_two" else torch_output.logits[:, 126],
        batch["targets"].reshape(-1) if stage == "rung_two" else batch["targets"],
        reduction="mean",
        label_smoothing=0.0,
    )
    supervised = cpu._supervised_route_loss(torch_output, batch["required_source"], cpu._import_runtime()) if stage in {"router_only", "joint"} else None
    internal = cpu._block_output(torch_output, 4).router_loss if stage == "joint" else None
    component_errors = [abs(float(mlx_losses[1].item()) - float(task_loss.detach()))]
    if stage == "joint":
        component_errors.append(abs(float(mlx_losses[2].item()) - float(internal.detach())))
    if stage in {"router_only", "joint"}:
        component_errors.append(abs(float(mlx_losses[3].item()) - float(supervised.detach())))
    gradient_by_name = {name: np.array(value) for name, value in zip(view["train_names"], mlx_gradients)}
    gradient_errors = []
    moment_indexes = {name: index for index, name in enumerate(view["train_names"])}
    first_errors = []
    second_errors = []
    optimizer_parameter_identity_exact = True
    optimizer_step_exact = True
    for torch_name, parameter in reference.named_parameters():
        mlx_name = mapped_mlx_parameter_name(torch_name)
        if membership[torch_name]["requires_grad"]:
            observed = gradient_by_name[mlx_name]
            expected = np.zeros_like(observed) if parameter.grad is None else parameter.grad.detach().cpu().numpy()
            gradient_errors.append(maximum_array_error(observed, expected))
            optimizer_value = optimizer.state.get(parameter)
            if not isinstance(optimizer_value, Mapping) or set(optimizer_value) != {"step", "exp_avg", "exp_avg_sq"}:
                optimizer_parameter_identity_exact = False
                continue
            index = moment_indexes[mlx_name]
            first_errors.append(maximum_array_error(view["first"][index], optimizer_value["exp_avg"].numpy()))
            second_errors.append(maximum_array_error(view["second"][index], optimizer_value["exp_avg_sq"].numpy()))
            optimizer_step_exact = optimizer_step_exact and float(optimizer_value["step"]) == 11.0
        elif parameter in optimizer.state:
            optimizer_parameter_identity_exact = False
    torch_routed = [block.mixer_output for block in torch_output.blocks if block.kind == "routed"]
    raw_route_exact = len(mlx_losses[4]) == len(torch_routed) and all(np.array_equal(np.array(observed), expected.telemetry["raw_remote"].numpy()) for observed, expected in zip(mlx_losses[4], torch_routed))
    effective_route_exact = len(mlx_losses[5]) == len(torch_routed) and all(np.array_equal(np.array(observed), expected.telemetry["effective_remote"].numpy()) for observed, expected in zip(mlx_losses[5], torch_routed))
    output_error = maximum_array_error(mlx_output[0][:, 510 if stage == "rung_two" else 126], torch_output.logits[:, 510 if stage == "rung_two" else 126].detach().numpy())
    total_error = abs(float(mlx_losses[0].item()) - float(torch_total.detach()))
    gradient_error = max(gradient_errors, default=0.0)
    optimizer_first_moment_max_abs = max(first_errors, default=0.0)
    optimizer_second_moment_max_abs = max(second_errors, default=0.0)
    maximum = max(total_error, output_error, max(component_errors, default=0.0), gradient_error, optimizer_first_moment_max_abs, optimizer_second_moment_max_abs)
    if not math.isfinite(maximum) or not raw_route_exact or not effective_route_exact or not optimizer_parameter_identity_exact or not optimizer_step_exact:
        evidence = {"stage": stage, "maximum": maximum, "total": total_error, "output": output_error, "component": max(component_errors, default=0.0), "gradient": gradient_error, "first": optimizer_first_moment_max_abs, "second": optimizer_second_moment_max_abs, "raw": raw_route_exact, "effective": effective_route_exact, "optimizer_identity": optimizer_parameter_identity_exact, "optimizer_step": optimizer_step_exact}
        raise MlxEngineError(f"pilot endpoint replay parity differs: {json.dumps(evidence, sort_keys=True)}")
    return {
        "total_loss_max_abs": total_error,
        "component_loss_max_abs": max(component_errors, default=0.0),
        "gradient_max_abs": gradient_error,
        "optimizer_first_moment_max_abs": optimizer_first_moment_max_abs,
        "optimizer_second_moment_max_abs": optimizer_second_moment_max_abs,
        "endpoint_output_max_abs": output_error,
    }


def pilot_endpoint_replay_benchmark(state: Mapping[str, Any], workload: str, fixture: Mapping[str, Any]) -> dict[str, Any]:
    roles = {
        "donor": (("donor", "donor", "all_eligible"),),
        "selected_vmap5": (("router_only", "router_only", "selected"), ("joint", "joint", "selected")),
        "dense_vmap5": (("dense_base", "dense_base", "dense"), ("dense_continuation", "dense_continuation", "dense")),
        "rung_two": (("rung_two", "rung_two", "rung_two"),),
    }[workload]
    records: dict[str, dict[str, list[int]]] = {}
    reductions = {}
    for family, stage, role in roles:
        batch = fixture["rung_two_row"] if stage == "rung_two" else fixture["rung_one_row"]
        for repetition in range(4):
            context = pilot_endpoint_replay_context(state, stage, role, batch)
            mx.eval(state["train"], state["frozen"], state["first"], state["second"], context["tokens"], context["targets"], context["required"], context["view"])
            started_ns = time.perf_counter_ns()
            reductions.setdefault(family, []).append(pilot_endpoint_replay_once(context, stage, role, batch))
            duration_ns = time.perf_counter_ns() - started_ns
            pilot_record_duration(records, family, repetition, duration_ns)
    return {"records": records, "reductions": reductions}


def pilot_checkpoint_fixture(state: Mapping[str, Any], lane: int, stage: str, model: ModularNeuralMachine, optimizer: Any) -> dict[str, Any]:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123456 + lane)
    rung = 2 if stage == "rung_two" else 1
    model_name = "rung_two" if stage == "rung_two" else "dense_causal" if stage.startswith("dense") else "all_eligible_donor" if stage == "donor" else "selected"
    last_attempt = cpu.attempt_id("pilot-nonclaim", rung, 123456 + lane, model_name, stage, 11)
    return {
        "schema_version": cpu.SCHEMA_VERSION,
        "run_id": "pilot-nonclaim",
        "rung": rung,
        "construction_seed": 123456 + lane,
        "model": model_name,
        "stage": canonical_stage_name(stage),
        "completed_update": 11,
        "last_attempt_id": last_attempt,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": {"kind": "pilot_fixed_rate", "updates": 11, "warmup_updates": 0, "completed_update": 11},
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "generator_states": {f"{stage}_data": generator.get_state()},
        "final_batch_sha256": "0" * 64,
    }


def pilot_validate_checkpoint_readback(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    if tuple(observed) != tuple(expected):
        raise MlxEngineError("pilot checkpoint key order differs")
    cpu._validate_checkpoint(
        observed,
        {name: expected[name] for name in ("schema_version", "run_id", "rung", "construction_seed", "model", "stage", "completed_update", "last_attempt_id", "final_batch_sha256")},
        torch,
    )
    if tuple(observed["model_state_dict"]) != tuple(expected["model_state_dict"]):
        raise MlxEngineError("pilot checkpoint model keys differ")
    for name, tensor in expected["model_state_dict"].items():
        loaded = observed["model_state_dict"][name]
        if tensor.dtype != loaded.dtype or tuple(tensor.shape) != tuple(loaded.shape) or not torch.equal(tensor, loaded):
            raise MlxEngineError("pilot checkpoint model tensor bytes differ")
    expected_optimizer = expected["optimizer_state_dict"]
    loaded_optimizer = observed["optimizer_state_dict"]
    if hashlib.sha256(cpu._torch_artifact_bytes(expected_optimizer, torch)).hexdigest() != hashlib.sha256(cpu._torch_artifact_bytes(loaded_optimizer, torch)).hexdigest():
        raise MlxEngineError("pilot checkpoint optimizer bytes differ")
    expected_ids = {parameter for group in expected_optimizer["param_groups"] for parameter in group["params"]}
    loaded_ids = {parameter for group in loaded_optimizer["param_groups"] for parameter in group["params"]}
    if expected_ids != set(expected_optimizer["state"]) or loaded_ids != set(loaded_optimizer["state"]) or expected_ids != loaded_ids:
        raise MlxEngineError("pilot checkpoint optimizer parameter identity differs")
    for parameter_id in expected_ids:
        left = expected_optimizer["state"][parameter_id]
        right = loaded_optimizer["state"][parameter_id]
        if set(left) != {"step", "exp_avg", "exp_avg_sq"} or set(right) != set(left):
            raise MlxEngineError("pilot checkpoint optimizer state keys differ")
        for name in ("step", "exp_avg", "exp_avg_sq"):
            if not torch.equal(left[name], right[name]):
                raise MlxEngineError("pilot checkpoint optimizer moment bytes differ")


def pilot_checkpoint_reload_once(state: Mapping[str, Any], stage: str, lanes: int, scratch: Path) -> int:
    from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

    checkpoints = []
    paths = []
    total_bytes = 0
    for lane in range(lanes):
        role = "selected" if stage in {"router_only", "joint"} else "dense" if stage.startswith("dense") else state["role"]
        model = pilot_torch_model_from_state(state, lane, role)
        optimizer, _ = pilot_optimizer_from_state(state, lane, stage, model)
        checkpoint = pilot_checkpoint_fixture(state, lane, stage, model, optimizer)
        path = scratch / f"endpoint-{lane}.pt"
        with path.open("xb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        checkpoints.append(checkpoint)
        paths.append(path)
    pilot_fsync_directory(scratch)
    for checkpoint, path in zip(checkpoints, paths):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(digest) != 64:
            raise MlxEngineError("pilot checkpoint raw hash differs")
        total_bytes += len(raw)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        pilot_validate_checkpoint_readback(checkpoint, loaded)
    for path in paths:
        path.unlink()
    pilot_fsync_directory(scratch)
    if any(scratch.iterdir()) or total_bytes <= 0:
        raise MlxEngineError("pilot checkpoint cleanup or bytes differ")
    return total_bytes


def pilot_checkpoint_reload_benchmark(state: Mapping[str, Any], workload: str, scratch_root: Path) -> dict[str, Any]:
    term_specs = {
        "donor": (("donor_single", "donor", 1),),
        "selected_vmap5": (("router_only_vmap5_all_lanes", "router_only", 5), ("joint_vmap5_all_lanes", "joint", 5)),
        "dense_vmap5": (("dense_vmap5_all_lanes", "dense_base", 5),),
        "rung_two": (("rung_two_single", "rung_two", 1),),
    }[workload]
    records: dict[str, dict[str, list[int]]] = {}
    byte_sizes = {}
    fixture_sha256s = {}
    cleanup = []
    engine_sha256 = dependency_hashes()["engine"]
    for family, stage, lanes in term_specs:
        fixture_sha256s[f"{family}_metadata"] = pilot_canonical_sha256({"family": family, "stage": stage, "lanes": lanes, "completed_update": 11, "claim_data": False, "engine_sha256": engine_sha256})
        for repetition in range(4):
            scratch = pilot_owned_scratch(scratch_root, family, repetition)
            mx.eval(state["train"], state["frozen"], state["first"], state["second"])
            started_ns = time.perf_counter_ns()
            observed_bytes = pilot_checkpoint_reload_once(state, stage, lanes, scratch)
            pilot_remove_owned_scratch(scratch)
            duration_ns = time.perf_counter_ns() - started_ns
            pilot_record_duration(records, family, repetition, duration_ns)
            if family in byte_sizes and byte_sizes[family] != observed_bytes:
                raise MlxEngineError("pilot checkpoint repeated byte size differs")
            byte_sizes[family] = observed_bytes
            cleanup.append(not scratch.exists())
    return {"records": records, "fixture_sha256s": fixture_sha256s, "byte_sizes": byte_sizes, "scratch_cleanup_pass": all(cleanup)}


def pilot_workload(specification: tuple[Any, ...], ordinal: int, sequence: int, scratch_root: Path, fixture: Mapping[str, Any]) -> tuple[int, dict[str, Any], int, dict[str, Any]]:
    workload, stage, role, execution, lanes, batch_size, sequence_length = specification
    model_seed = PILOT_SEED_BASE + PILOT_SEED_STRIDE * ordinal
    data_seed = model_seed + PILOT_DATA_SEED_OFFSET
    route_seed = model_seed + PILOT_ROUTE_SEED_OFFSET
    mx.clear_cache()
    mx.reset_peak_memory()
    state = pilot_parameter_state(stage, role, model_seed, lanes)
    runner = compiled_pilot_step(state["train_names"], state["frozen_names"], state["decay"], role, stage, workload == "selected_vmap5")
    rates = pilot_learning_rates(stage, state["train_names"])
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(data_seed)
    route_generator = torch.Generator(device="cpu")
    route_generator.manual_seed(route_seed)
    pilot_emit(
        {
            "kind": "pilot_workload_started",
            "sequence": sequence,
            "workload": workload,
            "workload_ordinal": ordinal,
            "model_seed": model_seed,
            "data_seed": data_seed,
            "route_seed": route_seed,
            "execution": execution,
            "lanes": lanes,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "memory": runtime_memory(),
        }
    )
    sequence += 1
    warmup_update_ns = []
    timed_update_ns = []
    observed_peak_memory = int(mx.get_peak_memory())
    for logical_update in (*PILOT_WARMUP_UPDATES, *PILOT_TIMED_UPDATES):
        tokens, targets, required_source, route_override = pilot_batch(workload, lanes, batch_size, sequence_length, data_generator, route_generator)
        pilot_exchange(
            {
                "kind": "pilot_update_ready",
                "sequence": sequence,
                "workload": workload,
                "workload_ordinal": ordinal,
                "logical_update": logical_update,
                "lanes": lanes,
                "token_positions_per_lane": batch_size * sequence_length,
            }
        )
        sequence += 1
        step = mx.array(float(logical_update), dtype=mx.float32)
        started_ns = time.perf_counter_ns()
        updated, next_first, next_second, output, gradient_norm, audit_status = runner(
            state["train"],
            state["first"],
            state["second"],
            state["frozen"],
            tokens,
            targets,
            required_source,
            route_override,
            rates,
            step,
        )
        mx.eval(updated, next_first, next_second, output, gradient_norm, audit_status)
        audit = np.asarray(audit_status)
        gradients_finite = bool(audit[:, :, 0].all())
        optimizer_finite = bool(audit[:, :, 2].all())
        gradient_norm_finite = bool(np.isfinite(np.asarray(gradient_norm)).all())
        raw_overflow_count = sum(routing_reduction(output[4], lane)[0] for lane in range(lanes))
        if not gradients_finite or not optimizer_finite or not gradient_norm_finite or raw_overflow_count != 0:
            raise MlxEngineError("pilot update audit failed")
        elapsed_ns = time.perf_counter_ns() - started_ns
        state["train"] = updated
        state["first"] = next_first
        state["second"] = next_second
        if logical_update in PILOT_WARMUP_UPDATES:
            warmup_update_ns.append(elapsed_ns)
        else:
            timed_update_ns.append(elapsed_ns)
        memory = runtime_memory()
        observed_peak_memory = max(observed_peak_memory, memory["peak_memory_bytes"])
        pilot_emit(
            {
                "kind": "pilot_update_complete",
                "sequence": sequence,
                "workload": workload,
                "workload_ordinal": ordinal,
                "logical_update": logical_update,
                "elapsed_ns": elapsed_ns,
                "warmup": logical_update in PILOT_WARMUP_UPDATES,
                "finite": True,
                "raw_overflow_count": raw_overflow_count,
                "gradient_norm_finite": gradient_norm_finite,
                "optimizer_finite": optimizer_finite,
                "memory": memory,
            }
        )
        sequence += 1
    evaluation = pilot_evaluation_benchmark(state, workload, fixture, scratch_root)
    endpoint_replay = pilot_endpoint_replay_benchmark(state, workload, fixture)
    checkpoint_reload = pilot_checkpoint_reload_benchmark(state, workload, scratch_root)
    tail = {"evaluation": evaluation, "endpoint_replay": endpoint_replay, "checkpoint_reload": checkpoint_reload}
    del state, runner, rates, data_generator, route_generator, tokens, targets, required_source, route_override, step, updated, next_first, next_second, output, gradient_norm, audit_status, audit, memory
    gc.collect()
    mx.clear_cache()
    cache_released = int(mx.get_cache_memory()) == 0
    if not cache_released:
        raise MlxEngineError("pilot Metal cache release failed")
    record = {
        "workload": workload,
        "workload_ordinal": ordinal,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "route_seed": route_seed,
        "execution": execution,
        "warmup_update_ns": warmup_update_ns,
        "timed_update_ns": timed_update_ns,
        "model_destroyed": True,
        "optimizer_destroyed": True,
        "metal_cache_released": True,
    }
    pilot_emit(
        {
            "kind": "pilot_workload_complete",
            "sequence": sequence,
            "record": record,
            "cold_compiled_update_ns": warmup_update_ns[0],
            "peak_memory_bytes": observed_peak_memory,
        }
    )
    return sequence + 1, record, observed_peak_memory, tail


def pilot_measured_components(records: list[Mapping[str, Any]]) -> dict[str, float]:
    expected = [specification[0] for specification in PILOT_WORKLOADS]
    if [record.get("workload") for record in records] != expected:
        raise MlxEngineError("pilot measured workload order differs")
    components = {}
    cold_compile_ns = 0
    for record in records:
        warmup = record.get("warmup_update_ns")
        timed = record.get("timed_update_ns")
        if not isinstance(warmup, list) or len(warmup) != 3 or not isinstance(timed, list) or len(timed) != 8 or any(type(value) is not int or value <= 0 for value in (*warmup, *timed)):
            raise MlxEngineError("pilot measured update durations differ")
        components[f"{record['workload']}_step_seconds"] = sum(timed) / 8_000_000_000
        cold_compile_ns += warmup[0]
    components["cold_compile_seconds"] = cold_compile_ns / 1_000_000_000
    return components


def pilot_merge_tail_benchmarks(parts: list[Mapping[str, Any]], fixture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_records = {}
    checkpoint_records = {}
    fixture_sha256s = dict(fixture["fixture_sha256s"])
    checkpoint_fixture_sha256s = {}
    checkpoint_byte_sizes = {}
    endpoint_reductions = {}
    evaluation_cleanup = []
    checkpoint_cleanup = []
    source_exclusion_route_bytes = []
    source_exclusion_fixtures = []
    for part in parts:
        evaluation = part["evaluation"]
        endpoint = part["endpoint_replay"]
        checkpoint = part["checkpoint_reload"]
        evaluation_cleanup.append(evaluation["scratch_cleanup_pass"])
        checkpoint_cleanup.append(checkpoint["scratch_cleanup_pass"])
        if evaluation["source_exclusion_route_bytes"]:
            source_exclusion_route_bytes.append(evaluation["source_exclusion_route_bytes"])
        if evaluation["source_exclusion_fixture"] is not None:
            source_exclusion_fixtures.append(evaluation["source_exclusion_fixture"])
        for name, digest in evaluation["fixture_sha256s"].items():
            if name in fixture_sha256s and fixture_sha256s[name] != digest:
                raise MlxEngineError("pilot evaluation fixture hash differs")
            fixture_sha256s[name] = digest
        for family, values in evaluation["records"].items():
            if family in evaluation_records:
                raise MlxEngineError("pilot evaluation family duplicated")
            evaluation_records[family] = values
        for family, values in endpoint["records"].items():
            name = f"endpoint_replay.{family}"
            if name in evaluation_records:
                raise MlxEngineError("pilot endpoint family duplicated")
            evaluation_records[name] = values
        endpoint_reductions.update(endpoint["reductions"])
        checkpoint_fixture_sha256s.update(checkpoint["fixture_sha256s"])
        checkpoint_byte_sizes.update(checkpoint["byte_sizes"])
        for family, values in checkpoint["records"].items():
            if family in checkpoint_records:
                raise MlxEngineError("pilot checkpoint family duplicated")
            checkpoint_records[family] = values
    if len([name for name in evaluation_records if name.startswith("rung_one_routed.")]) != 11:
        raise MlxEngineError("pilot routed evaluation condition count differs")
    if len([name for name in evaluation_records if name.startswith("endpoint_replay.")]) != 6:
        raise MlxEngineError("pilot endpoint replay role count differs")
    if set(endpoint_reductions) != {"donor", "router_only", "joint", "dense_base", "dense_continuation", "rung_two"} or any(not isinstance(values, list) or len(values) != 4 for values in endpoint_reductions.values()):
        raise MlxEngineError("pilot endpoint replay reduction count differs")
    if evaluation_cleanup != [True, True, True, True] or checkpoint_cleanup != [True, True, True, True] or source_exclusion_route_bytes != [65_536] or len(source_exclusion_fixtures) != 1:
        raise MlxEngineError("pilot evaluation fixture lifecycle differs")
    source_exclusion_fixture = source_exclusion_fixtures[0]
    if fixture_sha256s.get("source_exclusion_seed_633456") != pilot_canonical_sha256(source_exclusion_fixture):
        raise MlxEngineError("pilot source exclusion fixture identity differs")
    evaluation_maxima = {name: max(values["timed"]) for name, values in sorted(evaluation_records.items())}
    selected = {
        "route_acquisition": evaluation_maxima["route_acquisition"],
        "rung_one_routed": max(value for name, value in evaluation_maxima.items() if name.startswith("rung_one_routed.")),
        "rung_one_dense": evaluation_maxima["rung_one_dense"],
        "rung_two": max(value for name, value in evaluation_maxima.items() if name.startswith("rung_two.")),
        "endpoint_replay": max(value for name, value in evaluation_maxima.items() if name.startswith("endpoint_replay.")),
    }
    evaluation_ns = 80 * selected["route_acquisition"]
    evaluation_ns += 880 * selected["rung_one_routed"]
    evaluation_ns += 80 * selected["rung_one_dense"]
    evaluation_ns += 32 * selected["rung_two"]
    evaluation_ns += 26 * selected["endpoint_replay"]
    evaluation_component = evaluation_ns / 1_000_000_000
    fixture_bytes = 0
    for name in ("rung_one", "rung_two", "rung_one_row", "rung_two_row", "random_routes"):
        value = fixture[name]
        trees = value.values() if isinstance(value, Mapping) else (value,)
        fixture_bytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in trees)
    fixture_bytes += source_exclusion_route_bytes[0]
    evaluation_detail = {
        "fixture_sha256s": dict(sorted(fixture_sha256s.items())),
        "warmup_duration_ns": {name: values["warmup"] for name, values in sorted(evaluation_records.items())},
        "timed_duration_ns": {name: values["timed"] for name, values in sorted(evaluation_records.items())},
        "selected_max_duration_ns": dict(sorted(evaluation_maxima.items())),
        "counts": {"route_acquisition_calls": 80, "rung_one_routed_calls": 880, "rung_one_routed_conditions": 11, "rung_one_dense_calls": 80, "rung_two_calls": 32, "rung_two_conditions": 2, "endpoint_replay_calls": 26, "endpoint_replay_roles": 6},
        "byte_sizes": {"nonclaim_fixture_bytes": fixture_bytes},
        "scaling": {"route_acquisition_ns": 80 * selected["route_acquisition"], "rung_one_routed_ns": 880 * selected["rung_one_routed"], "rung_one_dense_ns": 80 * selected["rung_one_dense"], "rung_two_ns": 32 * selected["rung_two"], "endpoint_replay_ns": 26 * selected["endpoint_replay"], "total_ns": evaluation_ns},
        "scratch_cleanup_pass": all(evaluation_cleanup),
        "component_seconds": evaluation_component,
    }
    checkpoint_maxima = {name: max(values["timed"]) for name, values in sorted(checkpoint_records.items())}
    selected = checkpoint_maxima
    checkpoint_ns = 5 * selected["donor_single"]
    checkpoint_ns += selected["router_only_vmap5_all_lanes"]
    checkpoint_ns += selected["joint_vmap5_all_lanes"]
    checkpoint_ns += 2 * selected["dense_vmap5_all_lanes"]
    checkpoint_ns += selected["rung_two_single"]
    projected_checkpoint_bytes = 5 * checkpoint_byte_sizes["donor_single"]
    projected_checkpoint_bytes += checkpoint_byte_sizes["router_only_vmap5_all_lanes"]
    projected_checkpoint_bytes += checkpoint_byte_sizes["joint_vmap5_all_lanes"]
    projected_checkpoint_bytes += 2 * checkpoint_byte_sizes["dense_vmap5_all_lanes"]
    projected_checkpoint_bytes += checkpoint_byte_sizes["rung_two_single"]
    checkpoint_detail = {
        "fixture_sha256s": dict(sorted(checkpoint_fixture_sha256s.items())),
        "warmup_duration_ns": {name: values["warmup"] for name, values in sorted(checkpoint_records.items())},
        "timed_duration_ns": {name: values["timed"] for name, values in sorted(checkpoint_records.items())},
        "selected_max_duration_ns": dict(sorted(checkpoint_maxima.items())),
        "counts": {"donor_single_coefficient": 5, "router_only_vmap5_all_lanes_coefficient": 1, "joint_vmap5_all_lanes_coefficient": 1, "dense_vmap5_all_lanes_coefficient": 2, "rung_two_single_coefficient": 1, "trained_endpoint_files": 26},
        "byte_sizes": {**dict(sorted(checkpoint_byte_sizes.items())), "projected_checkpoint_bytes": projected_checkpoint_bytes},
        "scaling": {"donor_single_ns": 5 * selected["donor_single"], "router_only_vmap5_all_lanes_ns": selected["router_only_vmap5_all_lanes"], "joint_vmap5_all_lanes_ns": selected["joint_vmap5_all_lanes"], "dense_vmap5_all_lanes_ns": 2 * selected["dense_vmap5_all_lanes"], "rung_two_single_ns": selected["rung_two_single"], "total_ns": checkpoint_ns},
        "scratch_cleanup_pass": all(checkpoint_cleanup),
        "component_seconds": checkpoint_ns / 1_000_000_000,
    }
    for detail in (evaluation_detail, checkpoint_detail):
        encoded = json.dumps(detail, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > 1_048_576:
            raise MlxEngineError("pilot tail benchmark detail exceeds one MiB")
    return {"evaluation": evaluation_detail, "checkpoint_reload": checkpoint_detail}, source_exclusion_fixture


def pilot_runtime_observation() -> dict[str, Any]:
    return {
        "python_path": sys.executable,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "mlx_version": MLX_VERSION,
        "device": str(mx.default_device()),
        "training_dtype": "float32",
        "compilation": "mx.compile",
        "vectorization": "mx.vmap",
    }


def pilot() -> int:
    _, scratch_root = pilot_roots()
    preflight = self_check()
    hello = {
        "kind": "pilot_hello",
        "sequence": 0,
        "schema_version": IPC_SCHEMA_VERSION,
        "mlx_version": MLX_VERSION,
        "device": str(mx.default_device()),
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256s": dependency_hashes(),
        "self_check": preflight,
        "runtime": pilot_runtime_observation(),
    }
    pilot_emit(hello)
    if preflight["pass"] is not True or hello["device"] != "Device(gpu, 0)":
        return 1
    sequence = 1
    records = []
    tail_parts = []
    fixture = pilot_evaluation_fixture()
    observed_peak_memory = preflight["memory"]["peak_memory_bytes"]
    attempted_updates = 0
    token_positions = 0
    try:
        for ordinal, specification in enumerate(PILOT_WORKLOADS):
            sequence, record, peak_memory, tail = pilot_workload(specification, ordinal, sequence, scratch_root, fixture)
            records.append(record)
            tail_parts.append(tail)
            lanes = specification[4]
            batch_size = specification[5]
            sequence_length = specification[6]
            attempted_updates += lanes * (len(PILOT_WARMUP_UPDATES) + len(PILOT_TIMED_UPDATES))
            token_positions += lanes * batch_size * sequence_length * (len(PILOT_WARMUP_UPDATES) + len(PILOT_TIMED_UPDATES))
            observed_peak_memory = max(observed_peak_memory, peak_memory)
        if attempted_updates != PILOT_FINAL_ATTEMPTED_UPDATES or token_positions != PILOT_FINAL_TOKEN_POSITIONS:
            raise MlxEngineError("pilot cumulative accounting differs")
        memory = runtime_memory()
        memory["peak_memory_bytes"] = max(memory["peak_memory_bytes"], observed_peak_memory)
        tail_benchmarks, source_exclusion_fixture = pilot_merge_tail_benchmarks(tail_parts, fixture)
        measured_components = pilot_measured_components(records)
        measured_components["evaluation_seconds"] = tail_benchmarks["evaluation"]["component_seconds"]
        measured_components["checkpoint_reload_seconds"] = tail_benchmarks["checkpoint_reload"]["component_seconds"]
        pilot_emit(
            {
                "kind": "pilot_complete",
                "sequence": sequence,
                "status": "clean_complete",
                "workload_order": [record["workload"] for record in records],
                "attempted_updates": attempted_updates,
                "token_positions": token_positions,
                "measured_components": measured_components,
                "tail_benchmarks": tail_benchmarks,
                "source_exclusion_fixture": source_exclusion_fixture,
                "memory": memory,
                "runtime": pilot_runtime_observation(),
            }
        )
        response = json.loads(sys.stdin.readline())
        if response != {"ack": True, "kind": "close_committed"}:
            raise MlxEngineError("pilot close acknowledgement differs")
        return 0
    except BaseException as error:
        pilot_emit(
            {
                "kind": "pilot_hard_abort",
                "sequence": sequence,
                "error_type": type(error).__name__,
                "message": str(error),
                "attempted_updates": attempted_updates,
                "token_positions": token_positions,
                "memory": runtime_memory(),
            }
        )
        return 1


def runtime_memory() -> dict[str, Any]:
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "parent_rss_and_swap_required": True,
    }


def self_check() -> dict[str, Any]:
    parity = full_model_parity()
    forward_calibration = all_role_forward_calibration(parity)
    held_out_forward = held_out_forward_admission()
    gradient = full_gradient_parity("selected", 3123, 4123, 2)
    adam = adamw_parity()
    carried_adam = carried_adamw_parity()
    vectorized = independent_vmap_probe()
    functional = functional_forward_parity()
    actual_vectorized = actual_model_vmap5_probe()
    passed = parity["pass"] and forward_calibration["pass"] and held_out_forward["pass"] and gradient["pass"] and adam["pass"] and carried_adam["pass"] and vectorized["pass"] and functional["pass"] and actual_vectorized["pass"]
    return {
        "schema_version": IPC_SCHEMA_VERSION,
        "mlx_version": MLX_VERSION,
        "device": str(mx.default_device()),
        "contract": backend_contract(),
        "full_model_parity": parity,
        "all_role_forward_calibration": forward_calibration,
        "held_out_forward_admission": held_out_forward,
        "full_gradient_parity": gradient,
        "adamw_parity": adam,
        "carried_adamw_parity": carried_adam,
        "vmap5": vectorized,
        "functional_forward": functional,
        "actual_model_vmap5": actual_vectorized,
        "memory": runtime_memory(),
        "pass": bool(passed),
    }


def serve() -> int:
    run_root_text = os.environ.get("MODULAR_MLX_RUN_ROOT")
    scratch_root_text = os.environ.get("MODULAR_MLX_SCRATCH_ROOT")
    if not isinstance(run_root_text, str) or not run_root_text:
        raise MlxEngineError("MODULAR_MLX_RUN_ROOT is required")
    if not isinstance(scratch_root_text, str) or not scratch_root_text:
        raise MlxEngineError("MODULAR_MLX_SCRATCH_ROOT is required")
    run_root = Path(run_root_text)
    scratch_root = Path(scratch_root_text)
    if not run_root.is_absolute() or not run_root.is_dir() or run_root.is_symlink():
        raise MlxEngineError("MLX run root differs")
    if not scratch_root.is_absolute() or not scratch_root.is_dir() or scratch_root.is_symlink() or scratch_root == run_root or Path("/private/tmp") not in scratch_root.parents:
        raise MlxEngineError("MLX scratch root differs")

    preflight = self_check()
    if preflight["pass"] is not True or preflight["device"] != "Device(gpu, 0)":
        raise MlxEngineError("MLX GPU self-check failed")
    hello = {
        "kind": "hello",
        "sequence": 0,
        "schema_version": IPC_SCHEMA_VERSION,
        "mlx_version": MLX_VERSION,
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256s": dependency_hashes(),
        "self_check": preflight,
        "self_check_sha256": hashlib.sha256(json.dumps(preflight, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        "device": str(mx.default_device()),
    }
    print(json.dumps(hello, sort_keys=True, separators=(",", ":")), flush=True)
    child_sequence = 1
    routing_streams = {}
    routing_parts = {seed: [] for seed in RUNG_ONE_SEEDS}
    stage_audits = {stage: {} for stage in ("donor", "router_only", "joint", "dense_base", "dense_continuation", "rung_two")}
    endpoint_parity_records = []
    active_stage_streams = {}
    forward_sequences = {seed: 0 for seed in RUNG_ONE_SEEDS}

    def exchange(message: Mapping[str, Any]) -> Mapping[str, Any]:
        print(json.dumps(message, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        raw_response = sys.stdin.readline()
        if not raw_response:
            raise MlxEngineError("parent transport closed")
        response = json.loads(raw_response)
        if not isinstance(response, Mapping):
            raise MlxEngineError("parent response differs")
        return response

    try:
        while True:
            raw = sys.stdin.readline()
            if not raw:
                raise MlxEngineError("serve input closed before clean completion")
            request = json.loads(raw)
            if isinstance(request, Mapping) and set(request) == {"kind", "resource_sample_ids_by_seed"} and request["kind"] == "evaluate":
                resource_sample_ids_by_seed = validate_resource_sample_ids_by_seed(request["resource_sample_ids_by_seed"])
                if any(len(routing_parts[seed]) != 5 for seed in RUNG_ONE_SEEDS) or any(forward_sequences[seed] != 3840 for seed in RUNG_ONE_SEEDS):
                    raise MlxEngineError("evaluation routing stream state differs")
                write_qualification_gradient_artifacts(run_root, stage_audits)
                from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

                for seed in RUNG_ONE_SEEDS:
                    path = run_root / "rung1" / str(seed) / "routing.jsonl.gz"
                    stream = cpu.CanonicalGzipStream(path)
                    stream.open()
                    copied = 0
                    for part in routing_parts[seed]:
                        with gzip.open(part, "rt", encoding="utf-8") as handle:
                            for line in handle:
                                row = json.loads(line)
                                if row.get("phase") != "training" or row.get("construction_seed") != seed:
                                    raise MlxEngineError("training routing assembly identity differs")
                                stream.write(row)
                                copied += 1
                    if copied != 104448:
                        raise MlxEngineError("training routing assembly cardinality differs")
                    durable_gzip_prefix(stream)
                    routing_streams[seed] = stream
                validate_trained_endpoint_parity_records(endpoint_parity_records, 26)
                result = evaluate_qualification(run_root, routing_streams, forward_sequences, resource_sample_ids_by_seed, stage_audits, endpoint_parity_records)
                print(json.dumps({"kind": "evaluation_complete", "sequence": child_sequence, "result": result}, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
                child_sequence += 1
                continue
            if request == {"kind": "close"}:
                if set(routing_streams) != set(RUNG_ONE_SEEDS) or any(forward_sequences[seed] != 3840 for seed in RUNG_ONE_SEEDS):
                    raise MlxEngineError("training routing stream completion differs")
                routing_rows = 0
                training_rows = 0
                evaluation_rows = 0
                for seed in RUNG_ONE_SEEDS:
                    stream = routing_streams[seed]
                    if stream.compressed is not None:
                        stream.close()
                    with gzip.open(stream.path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            row = json.loads(line)
                            if row.get("phase") not in {"training", "route_acquisition", "evaluation"} or row.get("construction_seed") != seed:
                                raise MlxEngineError("training routing row identity differs")
                            training_rows += row["phase"] == "training"
                            evaluation_rows += row["phase"] in {"route_acquisition", "evaluation"}
                            routing_rows += 1
                if training_rows != 522240 or evaluation_rows != 66000 or routing_rows != 588240:
                    raise MlxEngineError("training routing row cardinality differs")
                print(json.dumps({"kind": "closed", "sequence": child_sequence, "status": "clean_complete"}, sort_keys=True, separators=(",", ":")), flush=True)
                response = json.loads(sys.stdin.readline())
                if response != {"ack": True, "kind": "close_committed"}:
                    raise MlxEngineError("parent close acknowledgement differs")
                return 0
            validated = validate_stage_request(request)
            selected_streams = None
            selected_sequences = None
            if validated["stage"] != "rung_two":
                from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu

                selected_streams = {}
                selected_sequences = forward_sequences
                active_stage_streams = {}
                for seed in validated["construction_seeds"]:
                    path = scratch_root / "routing_parts" / f"{validated['sequence']:02d}_{validated['stage']}_{seed}.jsonl.gz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = cpu.CanonicalGzipStream(path)
                    stream.open()
                    selected_streams[seed] = stream
                    active_stage_streams[seed] = stream
            child_sequence, _, finalized_audits = execute_stage(validated, run_root, exchange, child_sequence, selected_streams, selected_sequences, endpoint_parity_records)
            for seed, records in finalized_audits.items():
                if seed in stage_audits[validated["stage"]]:
                    raise MlxEngineError("qualification gradient stage duplicate")
                stage_audits[validated["stage"]][seed] = records
            for seed, stream in active_stage_streams.items():
                routing_parts[seed].append(stream.path)
            active_stage_streams = {}
    except BaseException as error:
        for stream in [*routing_streams.values(), *active_stage_streams.values()]:
            if stream.compressed is not None:
                try:
                    stream.close()
                except BaseException:
                    pass
        failure = {
            "kind": "hard_abort",
            "sequence": child_sequence,
            "reason": "nonfinite" if "nonfinite" in str(error) else "route_overflow" if "overflow" in str(error) else "artifact_inconsistency",
            "error_type": type(error).__name__,
            "message": str(error),
            "memory": runtime_memory(),
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["describe"]:
        print(json.dumps(backend_contract(), sort_keys=True, separators=(",", ":")))
        return 0
    if arguments == ["self-check"]:
        result = self_check()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0 if result["pass"] else 1
    if arguments == ["pilot"]:
        return pilot()
    if arguments == ["serve"]:
        return serve()
    raise MlxEngineError("expected exactly one mode: describe, self-check, pilot, or serve")


if __name__ == "__main__":
    raise SystemExit(main())
