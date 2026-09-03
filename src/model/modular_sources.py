from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn


TRANSFORMER_MODEL_PATH = Path("/Users/dttdrv/Projects/Transformerov/scale/model.py")
TRANSFORMER_MODEL_SHA256 = "6de04cac73a5f1d67cf2c9f5c51691658fcd06e4b62e1704528d51338643d904"
TRANSFORMER_RECURRENT_PATH = Path("/Users/dttdrv/Projects/Transformerov/scale/gated_delta.py")
TRANSFORMER_RECURRENT_SHA256 = "e638ba2cb6c9861344befe25d21cce208fc391718f758f5a4338ed4936747bf2"
PUBLIC_ROUTED_PATH = Path("/Users/dttdrv/Projects/Monodratic-public/src/monodratic/core.py")
PUBLIC_ROUTED_SHA256 = "e094ec94580b3a382b5604f96decced513e712cd8b944d4238cd59ef266d24ef"
PUBLIC_ROUTED_REVISION = "0f9bf59ebdd032da46553d985bcf23348e1d5289"


class FrozenSourceMismatchError(ImportError):
    def __init__(self, path: Path, expected: str, observed: str | None):
        self.surface = str(path)
        self.expected = expected
        self.observed = observed
        super().__init__(f"frozen source hash mismatch: {self.surface}")


def _frozen_source_bytes(path: Path, expected: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise FrozenSourceMismatchError(path, expected, None) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FrozenSourceMismatchError(path, expected, None)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise FrozenSourceMismatchError(path, expected, observed)
    return raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_source_hashes() -> Dict[str, str]:
    expected = {
        str(TRANSFORMER_MODEL_PATH): TRANSFORMER_MODEL_SHA256,
        str(TRANSFORMER_RECURRENT_PATH): TRANSFORMER_RECURRENT_SHA256,
        str(PUBLIC_ROUTED_PATH): PUBLIC_ROUTED_SHA256,
    }
    observed = {
        path: hashlib.sha256(_frozen_source_bytes(Path(path), digest)).hexdigest()
        for path, digest in expected.items()
    }
    return observed


def _load_module(name: str, path: Path, expected: str):
    raw = _frozen_source_bytes(path, expected)
    code = compile(raw, str(path), "exec", dont_inherit=True)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    return module


_TRANSFORMER_RECURRENT = _load_module(
    "_todorov_frozen_transformerov_gated_delta",
    TRANSFORMER_RECURRENT_PATH,
    TRANSFORMER_RECURRENT_SHA256,
)
_prior_gated_delta = sys.modules.get("gated_delta")
sys.modules["gated_delta"] = _TRANSFORMER_RECURRENT
try:
    _TRANSFORMER_MODEL = _load_module(
        "_todorov_frozen_transformerov_model",
        TRANSFORMER_MODEL_PATH,
        TRANSFORMER_MODEL_SHA256,
    )
finally:
    if _prior_gated_delta is None:
        sys.modules.pop("gated_delta", None)
    else:
        sys.modules["gated_delta"] = _prior_gated_delta
_PUBLIC_ROUTED = _load_module(
    "_todorov_frozen_monodratic_core",
    PUBLIC_ROUTED_PATH,
    PUBLIC_ROUTED_SHA256,
)


@dataclass(frozen=True)
class StateBoundary:
    kind: str
    position: int
    norms: Tensor


@dataclass(frozen=True)
class RecurrentMixerOutput:
    delta: Tensor
    primary_gate: Tensor
    write_gate: Tensor
    output_gate: Tensor
    boundaries: Tuple[StateBoundary, ...]


@dataclass(frozen=True)
class RoutedMixerOutput:
    delta: Tensor
    router_loss: Optional[Tensor]
    query_route: Tensor
    key_route: Tensor
    telemetry: Dict[str, Union[Tensor, int]]
    events: Tuple[str, ...]


@dataclass(frozen=True)
class DenseMixerOutput:
    delta: Tensor
    query: Tensor
    key: Tensor
    value: Tensor


class SourceFeatureMixer(_TRANSFORMER_MODEL.SwiGLU):
    pass


def _validate_mixer_input(x: Tensor, width: int) -> None:
    if x.ndim != 3 or x.size(-1) != width:
        raise ValueError(f"x must have shape batch, time, {width}")
    if x.dtype != torch.float32:
        raise ValueError("x must use torch.float32")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must be finite")


def _state_norms(state: Tensor) -> Tensor:
    state64 = state.detach().to(torch.float64)
    return state64.square().sum(dim=(-2, -1)).sqrt()


def _state_boundary(
    kind: str,
    position: int,
    state: Tensor,
) -> StateBoundary:
    return StateBoundary(
        kind=kind,
        position=position,
        norms=_state_norms(state),
    )


class ResetAwareRecurrentMixer(_TRANSFORMER_MODEL.GatedDelta):
    def _project(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, tokens, _ = x.shape
        query = _TRANSFORMER_RECURRENT.l2norm(
            self.q(x).reshape(batch, tokens, self.h, self.dh).transpose(1, 2)
        )
        key = _TRANSFORMER_RECURRENT.l2norm(
            self.k(x).reshape(batch, tokens, self.h, self.dh).transpose(1, 2)
        )
        value = self.v(x).reshape(batch, tokens, self.h, self.dh).transpose(1, 2)
        write_gate = torch.sigmoid(self.bp(x)).transpose(1, 2)
        primary_gate = torch.sigmoid(self.ag(x)).transpose(1, 2)
        output_gate = torch.sigmoid(self.og(x))
        return query, key, value, write_gate, primary_gate, output_gate

    def _reset_aware_outputs(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        write_gate: Tensor,
        primary_gate: Tensor,
        reset_positions: Tuple[int, ...],
        carry_intervention: Literal["none", "reset", "shuffle"],
        carry_position: Optional[int],
        collect_boundaries: bool,
    ) -> Tuple[Tensor, Tuple[StateBoundary, ...]]:
        batch, heads, tokens, key_width = query.shape
        value_width = value.size(-1)
        state = torch.zeros(
            batch,
            heads,
            key_width,
            value_width,
            dtype=query.dtype,
            device=query.device,
        )
        resets = set(reset_positions)
        outputs = []
        boundaries = []
        for position in range(tokens):
            if position in resets:
                if collect_boundaries:
                    boundaries.append(
                        _state_boundary(
                            "firewall_before_reset",
                            position,
                            state,
                        )
                    )
                state = torch.zeros_like(state)
                if collect_boundaries:
                    boundaries.append(
                        _state_boundary(
                            "firewall_after_reset",
                            position,
                            state,
                        )
                    )
            if carry_position == position and carry_intervention != "none":
                if collect_boundaries:
                    boundaries.append(
                        _state_boundary(
                            f"carry_before_{carry_intervention}",
                            position,
                            state,
                        )
                    )
                if carry_intervention == "reset":
                    state = torch.zeros_like(state)
                else:
                    state = state.roll(1, dims=0)
                if collect_boundaries:
                    boundaries.append(
                        _state_boundary(
                            f"carry_after_{carry_intervention}",
                            position,
                            state,
                        )
                    )
            current_key = key[:, :, position]
            current_value = value[:, :, position]
            current_query = query[:, :, position]
            current_write = write_gate[:, :, position]
            current_primary = primary_gate[:, :, position]
            projection = (current_key[..., :, None] * state).sum(-2)
            erase = (
                current_write[..., None, None]
                * current_key[..., :, None]
                * projection[..., None, :]
            )
            write = (
                current_write[..., None, None]
                * current_key[..., :, None]
                * current_value[..., None, :]
            )
            state = current_primary[..., None, None] * (state - erase) + write
            outputs.append((current_query[..., :, None] * state).sum(-2))
            if (position + 1) % self.chunk == 0:
                if collect_boundaries:
                    boundaries.append(
                        _state_boundary(
                            "chunk_end_before_clamp",
                            position,
                            state,
                        )
                    )
                norm = torch.sqrt((state * state).sum((-2, -1), keepdim=True)) + 1e-6
                state = state * torch.clamp(100.0 / norm, max=1.0)
                if collect_boundaries:
                    boundaries.append(
                        _state_boundary(
                            "chunk_end_after_clamp",
                            position,
                            state,
                        )
                    )
        return torch.stack(outputs, dim=2), tuple(boundaries)

    def _chunkwise_outputs_and_boundaries(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        primary_gate: Tensor,
        write_gate: Tensor,
    ) -> Tuple[Tensor, Tuple[StateBoundary, ...]]:
        batch, heads, tokens, key_width = query.shape
        value_width = value.shape[-1]
        chunk = self.chunk
        chunk_count = tokens // chunk
        mask = torch.tril(
            torch.ones(chunk, chunk, device=query.device, dtype=query.dtype)
        )
        strict_mask = mask - torch.eye(
            chunk, device=query.device, dtype=query.dtype
        )
        chunk_query = query.reshape(batch, heads, chunk_count, chunk, key_width)
        chunk_key = key.reshape(batch, heads, chunk_count, chunk, key_width)
        chunk_value = value.reshape(batch, heads, chunk_count, chunk, value_width)
        chunk_write = write_gate.reshape(batch, heads, chunk_count, chunk)
        cumulative = torch.cumsum(
            torch.log(primary_gate).reshape(batch, heads, chunk_count, chunk),
            dim=-1,
        )
        gamma = torch.exp(cumulative)
        difference = cumulative[..., :, None] - cumulative[..., None, :]
        lower = mask * torch.exp(mask * difference)
        interaction = (
            chunk_write[..., :, None]
            * (chunk_key @ chunk_key.transpose(-1, -2))
            * lower
        )
        inverse = _TRANSFORMER_RECURRENT.tril_inv(
            -(interaction * strict_mask), chunk
        )
        update = inverse @ (chunk_write[..., None] * chunk_value)
        weighted_key = inverse @ (
            (chunk_write * gamma)[..., None] * chunk_key
        )
        state = torch.zeros(
            batch,
            heads,
            key_width,
            value_width,
            dtype=query.dtype,
            device=query.device,
        )
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
                (gamma[:, :, index][..., None] * chunk_query[:, :, index])
                @ state
                + local @ corrected
            )
            carried_key = carry[:, :, index][..., None] * chunk_key[:, :, index]
            state = (
                final_gate[:, :, index][..., None, None] * state
                + carried_key.transpose(-1, -2) @ corrected
            )
            position = (index + 1) * chunk - 1
            boundaries.append(
                _state_boundary("chunk_end_before_clamp", position, state)
            )
            norm = torch.sqrt(
                (state * state).sum((-2, -1), keepdim=True)
            ) + 1e-6
            state = state * torch.clamp(100.0 / norm, max=1.0)
            boundaries.append(
                _state_boundary("chunk_end_after_clamp", position, state)
            )
        stacked = torch.stack(outputs, dim=2).reshape(
            batch, heads, tokens, value_width
        )
        return stacked, tuple(boundaries)

    def forward(
        self,
        x: Tensor,
        *,
        reset_positions: Tuple[int, ...] = (),
        carry_intervention: Literal["none", "reset", "shuffle"] = "none",
        carry_position: Optional[int] = None,
        return_aux: bool = False,
        force_reset_aware: bool = False,
    ) -> Union[Tensor, RecurrentMixerOutput]:
        _validate_mixer_input(x, self.q.in_features)
        if x.size(1) % self.chunk:
            raise ValueError("sequence length must be divisible by recurrent chunk length")
        if tuple(sorted(set(reset_positions))) != reset_positions:
            raise ValueError("reset positions must be sorted and unique")
        if any(position <= 0 or position >= x.size(1) for position in reset_positions):
            raise ValueError("reset positions must be internal sequence positions")
        if carry_intervention not in ("none", "reset", "shuffle"):
            raise ValueError("invalid carry intervention")
        if carry_intervention == "none" and carry_position is not None:
            raise ValueError("carry position requires an intervention")
        if carry_intervention != "none":
            if carry_position is None or carry_position <= 0 or carry_position >= x.size(1):
                raise ValueError("carry intervention position must be internal")
            if carry_position in reset_positions:
                raise ValueError("carry intervention cannot coincide with a firewall reset")
        query, key, value, write_gate, primary_gate, output_gate = self._project(x)
        use_reset_path = bool(reset_positions) or carry_intervention != "none" or force_reset_aware
        if use_reset_path:
            outputs, boundaries = self._reset_aware_outputs(
                query.float(),
                key.float(),
                value.float(),
                write_gate.float(),
                primary_gate.float(),
                reset_positions,
                carry_intervention,
                carry_position,
                return_aux,
            )
        else:
            outputs = _TRANSFORMER_RECURRENT.chunkwise_gated(
                query.float(),
                key.float(),
                value.float(),
                primary_gate.float(),
                write_gate.float(),
                self.chunk,
            )
            if return_aux:
                with torch.no_grad():
                    shadow, boundaries = self._chunkwise_outputs_and_boundaries(
                        query.detach().float(),
                        key.detach().float(),
                        value.detach().float(),
                        primary_gate.detach().float(),
                        write_gate.detach().float(),
                    )
                    shadow_error = (outputs.detach() - shadow).abs().max()
                    if not bool(torch.isfinite(shadow).all()) or not bool(
                        torch.isfinite(shadow_error)
                    ) or float(shadow_error) > 1e-5:
                        raise FloatingPointError(
                            "chunkwise telemetry shadow diverged from frozen source output"
                        )
            else:
                boundaries = ()
        outputs = self.onorm(outputs.to(x.dtype))
        outputs = outputs.transpose(1, 2).reshape(x.size(0), x.size(1), self.h * self.dh)
        delta = self.o(outputs * output_gate)
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("recurrent mixer produced nonfinite output")
        if not return_aux:
            return delta
        return RecurrentMixerOutput(
            delta=delta,
            primary_gate=primary_gate.detach().clone(),
            write_gate=write_gate.detach().clone(),
            output_gate=output_gate.detach().clone(),
            boundaries=boundaries,
        )


def _public_config(selected_remote_blocks: int):
    return _PUBLIC_ROUTED.MonodraticConfig(
        vocab_size=1,
        d_model=64,
        n_layers=1,
        n_heads=4,
        block_size=8,
        n_selected_blocks=selected_remote_blocks,
        local_blocks=1,
        phi_routing_dim=16,
        phi_subspaces=2,
        phi_codes=4,
        phi_probes=4,
        phi_bucket_capacity=64,
        phi_query_chunk_size=128,
        mlp_hidden_dim=0,
        attention_bias=False,
        tie_embeddings=False,
    )


def _validate_route_tensor(
    route: Tensor,
    expected: Tensor,
    block_size: int,
    local_blocks: int,
    query_only_position: Optional[int],
) -> None:
    if route.shape != expected.shape:
        raise ValueError("route override shape differs from the searched remote route")
    if route.dtype != torch.long:
        raise ValueError("route override must use torch.long")
    if route.device != expected.device:
        raise ValueError("route override must use the mixer device")
    if bool((route < -1).any()):
        raise ValueError("route override contains an invalid negative ID")
    if query_only_position is not None:
        outside_query = torch.arange(route.size(1), device=route.device) != query_only_position
        if bool((route[:, outside_query] != -1).any()):
            raise ValueError("route override violates the query-only remote firewall")
    positions = torch.arange(route.size(1), device=route.device)
    limits = (
        positions.div(block_size, rounding_mode="floor") - local_blocks + 1
    ).clamp_min(0)
    if bool(((route >= limits.view(1, -1, 1, 1)) & (route >= 0)).any()):
        raise ValueError("route override contains a noncausal block ID")
    sorted_route = route.sort(dim=-1).values
    duplicates = (sorted_route[..., 1:] == sorted_route[..., :-1]) & (
        sorted_route[..., 1:] >= 0
    )
    if bool(duplicates.any()):
        raise ValueError("route override contains duplicate valid IDs")


def _sparse_selected_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_blocks: Tensor,
    *,
    block_size: int,
) -> Tensor:
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value shapes differ")
    batch, heads, tokens, dimension = query.shape
    if selected_blocks.ndim != 4 or selected_blocks.shape[:3] != (batch, tokens, 1):
        raise ValueError("selected blocks must have one route group")
    if selected_blocks.dtype != torch.long or selected_blocks.device != query.device:
        raise ValueError("selected blocks identity differs")
    if type(block_size) is not int or block_size < 1 or selected_blocks.size(-1) < 1:
        raise ValueError("selected block geometry differs")
    blocks = selected_blocks[:, :, 0].sort(dim=-1).values
    valid_blocks = blocks >= 0
    unique_blocks = valid_blocks.clone()
    unique_blocks[..., 1:] &= blocks[..., 1:] != blocks[..., :-1]
    safe_blocks = blocks.clamp_min(0)
    offsets = torch.arange(block_size, device=query.device)
    candidate_ids = (safe_blocks[..., None] * block_size + offsets).flatten(start_dim=-2)
    candidate_valid = unique_blocks[..., None].expand_as(
        safe_blocks[..., None] * block_size + offsets
    ).flatten(start_dim=-2).clone()
    positions = torch.arange(tokens, device=query.device)
    candidate_valid &= candidate_ids <= positions.view(1, tokens, 1)
    candidate_valid &= candidate_ids < tokens
    safe_ids = candidate_ids.clamp_max(max(tokens - 1, 0))
    batch_ids = torch.arange(batch, device=query.device).view(batch, 1, 1)
    candidate_key = key.transpose(1, 2)[batch_ids, safe_ids]
    candidate_value = value.transpose(1, 2)[batch_ids, safe_ids]
    logits = (
        query.transpose(1, 2).float().unsqueeze(2) * candidate_key.float()
    ).sum(dim=-1).transpose(2, 3) / math.sqrt(dimension)
    mask = candidate_valid[:, :, None]
    any_valid = candidate_valid.any(dim=-1).view(batch, tokens, 1, 1)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    masked_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
    weights = torch.softmax(masked_logits, dim=-1).masked_fill(~mask, 0).to(value.dtype)
    mixed = (weights.transpose(2, 3).unsqueeze(-1) * candidate_value).sum(dim=2)
    return mixed.transpose(1, 2)


def _routing_evidence_histograms(
    query_route: Tensor,
    codebooks: Tensor,
    probes: int,
    block_addresses: Tensor,
    postings: Tensor,
    remote_limits: Tensor,
    selected_width: int,
) -> Tuple[Tensor, Tensor]:
    if query_route.requires_grad or codebooks.requires_grad:
        raise RuntimeError("routing evidence inputs remain graph attached")
    if not query_route.is_floating_point() or not codebooks.is_floating_point():
        raise RuntimeError("routing evidence feature dtype differs")
    if not bool(torch.isfinite(query_route).all()) or not bool(torch.isfinite(codebooks).all()):
        raise RuntimeError("routing evidence features are nonfinite")
    if probes != 4 or type(selected_width) is not int or selected_width < 0:
        raise RuntimeError("routing evidence search geometry differs")
    if block_addresses.dtype != torch.long or postings.dtype != torch.long or remote_limits.dtype != torch.long:
        raise RuntimeError("routing evidence index dtype differs")
    if query_route.ndim != 4 or block_addresses.ndim != 2 or postings.ndim != 3 or remote_limits.shape != (query_route.size(1),):
        raise RuntimeError("routing evidence input shape differs")
    if query_route.device != codebooks.device or query_route.device != block_addresses.device or query_route.device != postings.device or query_route.device != remote_limits.device:
        raise RuntimeError("routing evidence input device differs")
    with torch.no_grad():
        batch = block_addresses.size(0)
        address_space = postings.size(1)
        offsets = torch.arange(batch, device=block_addresses.device).view(-1, 1) * address_space
        flat_addresses = (block_addresses + offsets).reshape(-1)
        loads = torch.bincount(flat_addresses, minlength=batch * address_space)
        load_frequencies = torch.bincount(loads, minlength=1)
        observed_loads = torch.nonzero(load_frequencies, as_tuple=False).flatten()
        block_load_histogram = torch.stack(
            (observed_loads, load_frequencies[observed_loads]), dim=1
        ).to(dtype=torch.long)
        probe_addresses = _PUBLIC_ROUTED.probe_addresses(
            query_route,
            codebooks,
            probes,
        )
        if selected_width == 0:
            valid_posting_histogram = torch.empty(
                (0, 2), dtype=torch.long, device=query_route.device
            )
        else:
            batch_ids = torch.arange(batch, device=query_route.device).view(batch, 1, 1, 1)
            candidates = postings[
                batch_ids.expand_as(probe_addresses), probe_addresses
            ]
            valid_counts = (
                (candidates >= 0)
                & (candidates < remote_limits.view(1, -1, 1, 1, 1))
            ).sum(dim=(-1, -2))
            searched_counts = valid_counts[:, remote_limits > selected_width].reshape(-1)
            if searched_counts.numel() == 0:
                valid_posting_histogram = torch.empty(
                    (0, 2), dtype=torch.long, device=query_route.device
                )
            else:
                valid_frequencies = torch.bincount(searched_counts, minlength=1)
                observed_valid = torch.nonzero(valid_frequencies, as_tuple=False).flatten()
                valid_posting_histogram = torch.stack(
                    (observed_valid, valid_frequencies[observed_valid]), dim=1
                ).to(dtype=torch.long)
        return block_load_histogram.detach(), valid_posting_histogram.detach()


def _validate_routing_evidence_reductions(
    block_load_histogram: Tensor,
    valid_posting_histogram: Tensor,
    batch: int,
    tokens: int,
    config,
    index,
    search,
) -> None:
    bucket_count = int(block_load_histogram[:, 1].sum())
    indexed_blocks = int((block_load_histogram[:, 0] * block_load_histogram[:, 1]).sum())
    maximum_load = int(block_load_histogram[:, 0].max())
    overflow = int(
        (
            (block_load_histogram[:, 0] - config.phi_bucket_capacity).clamp_min(0)
            * block_load_histogram[:, 1]
        ).sum()
    )
    search_rows = int(valid_posting_histogram[:, 1].sum()) if valid_posting_histogram.numel() else 0
    valid_postings = int(
        (valid_posting_histogram[:, 0] * valid_posting_histogram[:, 1]).sum()
    ) if valid_posting_histogram.numel() else 0
    expected_blocks = batch * (tokens // config.block_size)
    if bucket_count != batch * index.postings.size(1) or indexed_blocks != expected_blocks:
        raise RuntimeError("routing block-load histogram differs")
    if maximum_load != index.max_bucket_load or overflow != index.overflow_count:
        raise RuntimeError("routing block-load reduction differs")
    if search_rows != search.search_rows or search_rows * config.phi_probes != search.addresses_probed:
        raise RuntimeError("routing valid-posting row reduction differs")
    if valid_postings != search.postings_read or valid_postings != search.candidate_blocks:
        raise RuntimeError("routing valid-posting count reduction differs")


class PublicRoutedMixer(nn.Module):
    def __init__(self, selected_remote_blocks: int) -> None:
        super().__init__()
        if selected_remote_blocks < 0:
            raise ValueError("selected remote block count must be non-negative")
        self.source_mixer = _PUBLIC_ROUTED.MonodraticPHIMixer(
            _public_config(selected_remote_blocks)
        )

    @property
    def selected_remote_blocks(self) -> int:
        return self.source_mixer.config.n_selected_blocks

    def forward(
        self,
        x: Tensor,
        *,
        return_aux: bool = False,
        return_detail: bool = False,
        request_router_loss: bool = False,
        forced_blocks: Optional[Tensor] = None,
        route_override: Optional[Tensor] = None,
        query_only_position: Optional[int] = None,
    ) -> Union[Tensor, RoutedMixerOutput]:
        _validate_mixer_input(x, 64)
        if return_detail and not return_aux:
            raise ValueError("route detail requires the auxiliary return path")
        if request_router_loss and not return_aux:
            raise ValueError("router loss requires the auxiliary return path")
        if query_only_position is not None and not 0 <= query_only_position < x.size(1):
            raise ValueError("query-only position is outside the sequence")
        attention = self.source_mixer.attention
        config = self.source_mixer.config
        events = []
        batch, tokens, width = x.shape
        qkv = attention.qkv(x).reshape(
            batch, tokens, 3, config.n_heads, attention.head_dim
        )
        events.append("qkv")
        query = attention._rope(qkv[:, :, 0].transpose(1, 2))
        events.append("rope_query")
        key = attention._rope(qkv[:, :, 1].transpose(1, 2))
        events.append("rope_key")
        value = qkv[:, :, 2].transpose(1, 2)
        query_input = query.transpose(1, 2).reshape(batch, tokens, width)
        key_input = key.transpose(1, 2).reshape(batch, tokens, width)
        query_route = attention.router.query_features(query_input)
        events.append("query_features")
        key_route = attention.router.key_features(key_input)
        events.append("key_features")
        codebooks = attention.router.normalized_codebooks()
        index = _PUBLIC_ROUTED.build_packed_index(
            key_route,
            codebooks,
            block_size=config.block_size,
            bucket_capacity=config.phi_bucket_capacity,
        )
        events.append("build_index")
        search = _PUBLIC_ROUTED.search_packed_index(
            query_route,
            codebooks,
            index,
            probes=config.phi_probes,
            selected_blocks=config.n_selected_blocks,
            local_blocks=config.local_blocks,
            query_chunk_size=config.phi_query_chunk_size,
        )
        events.append("search_index")
        raw_remote = search.selected_blocks.clone()
        events.append("clone_raw_remote")
        effective_remote = search.selected_blocks.clone()
        if forced_blocks is not None:
            if forced_blocks.dtype != torch.long or forced_blocks.device != x.device:
                raise ValueError("forced blocks must use torch.long on the mixer device")
            effective_remote = _PUBLIC_ROUTED._force_remote_blocks(
                effective_remote,
                forced_blocks,
                block_size=config.block_size,
                local_blocks=config.local_blocks,
            )
            events.append("force_remote")
        if query_only_position is not None:
            remote_mask = torch.arange(tokens, device=x.device) != query_only_position
            effective_remote = effective_remote.masked_fill(
                remote_mask.view(1, tokens, 1, 1), -1
            )
            events.append("query_only_mask")
        if route_override is not None:
            _validate_route_tensor(
                route_override,
                effective_remote,
                config.block_size,
                config.local_blocks,
                query_only_position,
            )
            effective_remote = route_override
            events.append("route_override")
        selected = _PUBLIC_ROUTED._block_ids_for_attention(
            effective_remote,
            tokens,
            config.block_size,
            config.local_blocks,
        )
        events.append("block_ids")
        attended = _sparse_selected_attention(
            query,
            key,
            value,
            selected,
            block_size=config.block_size,
        )
        events.append("selected_attention")
        delta = attention.out(attended.transpose(1, 2).reshape(batch, tokens, width))
        events.append("output_projection")
        router_loss = (
            attention._router_loss(query, key, query_route, key_route)
            if request_router_loss
            else None
        )
        if request_router_loss:
            events.append("router_loss")
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("routed mixer produced nonfinite output")
        block_load_histogram = None
        valid_posting_histogram = None
        if return_detail:
            with torch.no_grad():
                audit_query_route = query_route.detach()
                audit_codebooks = codebooks.detach()
                audit_block_addresses = index.block_addresses.detach()
                audit_postings = index.postings.detach()
                audit_positions = torch.arange(tokens, device=x.device)
                audit_remote_limits = (
                    audit_positions.div(config.block_size, rounding_mode="floor")
                    - config.local_blocks
                    + 1
                ).clamp_min(0)
            block_load_histogram, valid_posting_histogram = _routing_evidence_histograms(
                audit_query_route,
                audit_codebooks,
                config.phi_probes,
                audit_block_addresses,
                audit_postings,
                audit_remote_limits,
                config.n_selected_blocks,
            )
            _validate_routing_evidence_reductions(
                block_load_histogram,
                valid_posting_histogram,
                batch,
                tokens,
                config,
                index,
                search,
            )
            events.append("evidence_probe")
        if not return_aux:
            return delta
        telemetry: Dict[str, Union[Tensor, int]] = {
            "raw_remote": raw_remote.detach().clone(),
            "effective_remote": effective_remote.detach().clone(),
            "selected_blocks": selected.detach().clone(),
            "indexed_tokens": index.indexed_tokens,
            "addresses_probed": search.addresses_probed,
            "postings_read": search.postings_read,
            "valid_posting_entries": search.postings_read,
            "posting_slots_materialized": search.posting_slots_materialized,
            "candidate_blocks": search.candidate_blocks,
            "search_rows": search.search_rows,
            "bypass_rows": search.bypass_rows,
            "overflow_count": index.overflow_count,
            "max_bucket_load": index.max_bucket_load,
            "workspace_bytes": search.workspace_bytes,
        }
        if return_detail:
            telemetry.update(
                {
                    "query": query.detach().clone(),
                    "key": key.detach().clone(),
                    "value": value.detach().clone(),
                    "block_features": index.block_features.detach().clone(),
                    "block_addresses": index.block_addresses.detach().clone(),
                    "postings": index.postings.detach().clone(),
                    "block_load_histogram": block_load_histogram,
                    "valid_posting_histogram": valid_posting_histogram,
                }
            )
        return RoutedMixerOutput(
            delta=delta,
            router_loss=router_loss,
            query_route=query_route,
            key_route=key_route,
            telemetry=telemetry,
            events=tuple(events),
        )


class DenseCausalMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source_mixer = _PUBLIC_ROUTED.MonodraticPHIMixer(_public_config(0))
        del self.source_mixer.attention.router

    def forward(
        self,
        x: Tensor,
        *,
        return_aux: bool = False,
    ) -> Union[Tensor, DenseMixerOutput]:
        _validate_mixer_input(x, 64)
        attention = self.source_mixer.attention
        config = self.source_mixer.config
        batch, tokens, width = x.shape
        qkv = attention.qkv(x).reshape(
            batch, tokens, 3, config.n_heads, attention.head_dim
        )
        query = attention._rope(qkv[:, :, 0].transpose(1, 2))
        key = attention._rope(qkv[:, :, 1].transpose(1, 2))
        value = qkv[:, :, 2].transpose(1, 2)
        scores = torch.einsum("bhtd,bhsd->bhts", query.float(), key.float())
        scores = scores / math.sqrt(attention.head_dim)
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
        weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        attended = torch.einsum("bhts,bhsd->bhtd", weights.to(value.dtype), value)
        delta = attention.out(attended.transpose(1, 2).reshape(batch, tokens, width))
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("dense mixer produced nonfinite output")
        if not return_aux:
            return delta
        return DenseMixerOutput(
            delta=delta,
            query=query.detach().clone(),
            key=key.detach().clone(),
            value=value.detach().clone(),
        )


def frozen_chunkwise_gated(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    primary_gate: Tensor,
    write_gate: Tensor,
    chunk: int,
) -> Tensor:
    return _TRANSFORMER_RECURRENT.chunkwise_gated(
        query,
        key,
        value,
        primary_gate,
        write_gate,
        chunk,
    )


def frozen_recurrent_gated(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    primary_gate: Tensor,
    write_gate: Tensor,
) -> Tensor:
    return _TRANSFORMER_RECURRENT.recurrent_gated(
        query,
        key,
        value,
        primary_gate,
        write_gate,
    )
