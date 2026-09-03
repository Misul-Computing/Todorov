from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from src.model.modular_sources import (
    DenseCausalMixer,
    DenseMixerOutput,
    PublicRoutedMixer,
    RecurrentMixerOutput,
    ResetAwareRecurrentMixer,
    RoutedMixerOutput,
    SourceFeatureMixer,
)


RUNG_ONE_RESET_POSITIONS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80)
ROUTED_BLOCK_INDICES = (0, 4)
RECURRENT_BLOCK_INDICES = (1, 2, 3, 5, 6, 7)
SEQUENCE_SCHEDULE = (
    "routed",
    "recurrent",
    "recurrent",
    "recurrent",
    "routed",
    "recurrent",
    "recurrent",
    "recurrent",
)


@dataclass(frozen=True)
class ModularModelConfig:
    role: Literal["selected", "all_eligible", "local_only", "dense", "rung_two"]
    vocab_size: int
    sequence_length: int
    block4_kind: Literal["routed", "dense"]
    block4_remote_blocks: int
    query_only_position: Optional[int]
    reset_positions: Tuple[int, ...]
    carry_position: Optional[int]
    width: int = 64
    block_count: int = 8
    heads: int = 4
    recurrent_head_width: int = 16
    recurrent_chunk_length: int = 32
    route_block_size: int = 8
    local_blocks: int = 1
    routing_width: int = 16
    routing_subspaces: int = 2
    routing_codes: int = 4
    routing_probes: int = 4
    routing_bucket_capacity: int = 64
    routing_query_chunk_length: int = 128

    def __post_init__(self) -> None:
        if self.role not in ("selected", "all_eligible", "local_only", "dense", "rung_two"):
            raise ValueError("invalid model role")
        fixed = (
            self.width == 64,
            self.block_count == 8,
            self.heads == 4,
            self.recurrent_head_width == 16,
            self.recurrent_chunk_length == 32,
            self.route_block_size == 8,
            self.local_blocks == 1,
            self.routing_width == 16,
            self.routing_subspaces == 2,
            self.routing_codes == 4,
            self.routing_probes == 4,
            self.routing_bucket_capacity == 64,
            self.routing_query_chunk_length == 128,
        )
        if not all(fixed):
            raise ValueError("the base witness geometry is fixed")
        if self.vocab_size <= 0:
            raise ValueError("vocabulary size must be positive")
        if self.sequence_length % self.recurrent_chunk_length:
            raise ValueError("sequence length must divide into complete recurrent chunks")
        if self.role == "rung_two":
            expected = (
                self.vocab_size == 256,
                self.sequence_length == 512,
                self.block4_kind == "routed",
                self.block4_remote_blocks == 0,
                self.query_only_position is None,
                self.reset_positions == (),
                self.carry_position is None,
            )
        else:
            expected_remote = {
                "selected": 2,
                "all_eligible": 15,
                "local_only": 0,
                "dense": 0,
            }[self.role]
            expected_kind = "dense" if self.role == "dense" else "routed"
            expected_query = None if self.role == "dense" else 126
            expected = (
                self.vocab_size == 128,
                self.sequence_length == 128,
                self.block4_kind == expected_kind,
                self.block4_remote_blocks == expected_remote,
                self.query_only_position == expected_query,
                self.reset_positions == RUNG_ONE_RESET_POSITIONS,
                self.carry_position == 96,
            )
        if not all(expected):
            raise ValueError("configuration does not match a preregistered model role")


@dataclass(frozen=True)
class BlockExecution:
    block_index: int
    kind: str
    computed_sequence_delta: Tensor
    exposed_sequence_delta: Tensor
    feature_delta: Tensor
    mixer_output: Optional[Union[RoutedMixerOutput, RecurrentMixerOutput, DenseMixerOutput]]


@dataclass(frozen=True)
class ModularModelOutput:
    logits: Tensor
    hidden: Tensor
    blocks: Tuple[BlockExecution, ...]


@dataclass(frozen=True)
class StateCopyReport:
    compatible: Tuple[str, ...]
    incompatible_source: Tuple[str, ...]
    incompatible_destination: Tuple[str, ...]


def _state_tensor_identity(value: Tensor) -> Tuple[bytes, bytes, str]:
    if value.device.type != "cpu":
        raise ValueError("state tensor identity requires CPU storage")
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    body_tensor = value.detach().contiguous().reshape(-1).clone()
    body = bytes(body_tensor.untyped_storage())
    digest = hashlib.sha256(header + b"\n" + body).hexdigest()
    return header, body, digest


def rung_one_config(
    role: Literal["selected", "all_eligible", "local_only", "dense"] = "selected",
) -> ModularModelConfig:
    remote = {
        "selected": 2,
        "all_eligible": 15,
        "local_only": 0,
        "dense": 0,
    }.get(role)
    if remote is None:
        raise ValueError("invalid rung-one role")
    return ModularModelConfig(
        role=role,
        vocab_size=128,
        sequence_length=128,
        block4_kind="dense" if role == "dense" else "routed",
        block4_remote_blocks=remote,
        query_only_position=None if role == "dense" else 126,
        reset_positions=RUNG_ONE_RESET_POSITIONS,
        carry_position=96,
    )


def rung_two_config() -> ModularModelConfig:
    return ModularModelConfig(
        role="rung_two",
        vocab_size=256,
        sequence_length=512,
        block4_kind="routed",
        block4_remote_blocks=0,
        query_only_position=None,
        reset_positions=(),
        carry_position=None,
    )


class ModularBlock(nn.Module):
    def __init__(
        self,
        block_index: int,
        config: ModularModelConfig,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        self.kind = SEQUENCE_SCHEDULE[block_index]
        self.n1 = nn.RMSNorm(config.width)
        if self.kind == "recurrent":
            self.mix = ResetAwareRecurrentMixer(
                config.width,
                config.heads,
                config.recurrent_head_width,
                config.recurrent_chunk_length,
            )
        elif block_index == 0:
            self.mix = PublicRoutedMixer(0)
        elif config.block4_kind == "dense":
            self.mix = DenseCausalMixer()
        else:
            self.mix = PublicRoutedMixer(config.block4_remote_blocks)
        self.n2 = nn.RMSNorm(config.width)
        self.mlp = SourceFeatureMixer(config.width)


class ModularNeuralMachine(nn.Module):
    def __init__(self, config: ModularModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(
            ModularBlock(index, config) for index in range(config.block_count)
        )
        self.nf = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        *,
        return_aux: bool = False,
        recurrent_telemetry: bool = False,
        route_detail: bool = False,
        request_block4_router_loss: bool = False,
        forced_blocks: Optional[Tensor] = None,
        route_override: Optional[Tensor] = None,
        recurrent_intervention: Literal["none", "reset", "shuffle"] = "none",
        recurrent_knockout: bool = False,
        block4_routed_knockout: bool = False,
    ) -> Union[Tensor, ModularModelOutput]:
        if input_ids.ndim != 2:
            raise ValueError("input IDs must have shape batch, time")
        if input_ids.dtype != torch.long:
            raise ValueError("input IDs must use torch.long")
        if input_ids.device.type != "cpu":
            raise ValueError("the base witness is CPU-only")
        if input_ids.size(1) != self.config.sequence_length:
            raise ValueError("input sequence length differs from the fixed role")
        if bool((input_ids < 0).any()) or bool((input_ids >= self.config.vocab_size).any()):
            raise ValueError("input IDs are outside the vocabulary")
        if request_block4_router_loss and (
            not return_aux or self.config.role != "selected"
        ):
            raise ValueError("block-4 router loss requires the selected auxiliary role")
        if (recurrent_telemetry or route_detail) and not return_aux:
            raise ValueError("telemetry detail requires the auxiliary return path")
        if recurrent_intervention != "none" and self.config.carry_position is None:
            raise ValueError("this role has no registered carry intervention")
        if block4_routed_knockout and self.config.block4_kind != "routed":
            raise ValueError("block-4 routed knockout requires a routed block")
        if (forced_blocks is not None or route_override is not None) and self.config.block4_kind != "routed":
            raise ValueError("route interventions require a routed block 4")
        hidden = self.embed(input_ids)
        executions = []
        for block in self.blocks:
            normalized = block.n1(hidden)
            if block.kind == "recurrent":
                mixed = block.mix(
                    normalized,
                    reset_positions=self.config.reset_positions,
                    carry_intervention=recurrent_intervention,
                    carry_position=(
                        self.config.carry_position
                        if recurrent_intervention != "none"
                        else None
                    ),
                    return_aux=return_aux and recurrent_telemetry,
                )
            elif block.block_index == 4 and self.config.block4_kind == "dense":
                mixed = block.mix(normalized, return_aux=return_aux)
            else:
                mixed = block.mix(
                    normalized,
                    return_aux=return_aux,
                    return_detail=route_detail,
                    request_router_loss=(
                        request_block4_router_loss and block.block_index == 4
                    ),
                    forced_blocks=forced_blocks if block.block_index == 4 else None,
                    route_override=route_override if block.block_index == 4 else None,
                    query_only_position=(
                        self.config.query_only_position if block.block_index == 4 else None
                    ),
                )
            computed_delta = mixed.delta if not isinstance(mixed, torch.Tensor) else mixed
            expose_zero = (
                recurrent_knockout and block.kind == "recurrent"
            ) or (
                block4_routed_knockout
                and block.block_index == 4
                and self.config.block4_kind == "routed"
            )
            exposed_delta = torch.zeros_like(computed_delta) if expose_zero else computed_delta
            hidden = hidden + exposed_delta
            feature_delta = block.mlp(block.n2(hidden))
            hidden = hidden + feature_delta
            if return_aux:
                executions.append(
                    BlockExecution(
                        block_index=block.block_index,
                        kind=(
                            "dense"
                            if block.block_index == 4 and self.config.block4_kind == "dense"
                            else block.kind
                        ),
                        computed_sequence_delta=computed_delta,
                        exposed_sequence_delta=exposed_delta,
                        feature_delta=feature_delta,
                        mixer_output=(
                            None if isinstance(mixed, torch.Tensor) else mixed
                        ),
                    )
                )
        final_hidden = self.nf(hidden)
        logits = self.head(final_hidden)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("model produced nonfinite logits")
        if not return_aux:
            return logits
        return ModularModelOutput(logits=logits, hidden=final_hidden, blocks=tuple(executions))


def is_router_parameter(name: str, block_index: Optional[int] = None) -> bool:
    marker = ".mix.source_mixer.attention.router."
    if marker not in name:
        return False
    if block_index is None:
        return True
    return name.startswith(f"blocks.{block_index}{marker}")


def parameter_category(name: str, parameter: nn.Parameter) -> str:
    if not isinstance(parameter, nn.Parameter):
        raise TypeError("parameter classification requires torch.nn.Parameter")
    shape = tuple(parameter.shape)
    if name in ("embed.weight", "head.weight"):
        if shape not in ((128, 64), (256, 64)):
            raise ValueError(f"invalid embedding or head shape for {name}: {shape}")
        return "matrix"
    if name == "nf.weight":
        if shape != (64,):
            raise ValueError(f"invalid final normalization shape: {shape}")
        return "normalization_scale"
    block_match = re.fullmatch(r"blocks\.([0-7])\.(.+)", name)
    if block_match is None:
        raise ValueError(f"unrecognized parameter name: {name}")
    block_index = int(block_match.group(1))
    suffix = block_match.group(2)
    if suffix in ("n1.weight", "n2.weight"):
        if shape != (64,):
            raise ValueError(f"invalid block normalization shape for {name}: {shape}")
        return "normalization_scale"
    feature_shapes = {
        "mlp.w1.weight": (256, 64),
        "mlp.w2.weight": (256, 64),
        "mlp.w3.weight": (64, 256),
    }
    if suffix in feature_shapes:
        if shape != feature_shapes[suffix]:
            raise ValueError(f"invalid feature matrix shape for {name}: {shape}")
        return "matrix"
    if block_index in ROUTED_BLOCK_INDICES:
        routed_shapes = {
            "mix.source_mixer.attention.qkv.weight": (192, 64),
            "mix.source_mixer.attention.out.weight": (64, 64),
            "mix.source_mixer.attention.router.query_projection.weight": (16, 64),
            "mix.source_mixer.attention.router.key_projection.weight": (16, 64),
        }
        if suffix in routed_shapes:
            if shape != routed_shapes[suffix]:
                raise ValueError(f"invalid routed matrix shape for {name}: {shape}")
            return "matrix"
        if suffix == "mix.source_mixer.attention.router.codebooks":
            if shape != (2, 4, 8):
                raise ValueError(f"invalid codebook shape for {name}: {shape}")
            return "codebook"
    if block_index in RECURRENT_BLOCK_INDICES:
        recurrent_matrix_shapes = {
            "mix.q.weight": (64, 64),
            "mix.k.weight": (64, 64),
            "mix.v.weight": (64, 64),
            "mix.bp.weight": (4, 64),
            "mix.ag.weight": (4, 64),
            "mix.og.weight": (64, 64),
            "mix.o.weight": (64, 64),
        }
        recurrent_bias_shapes = {
            "mix.bp.bias": (4,),
            "mix.ag.bias": (4,),
            "mix.og.bias": (64,),
        }
        if suffix in recurrent_matrix_shapes:
            if shape != recurrent_matrix_shapes[suffix]:
                raise ValueError(f"invalid recurrent matrix shape for {name}: {shape}")
            return "matrix"
        if suffix in recurrent_bias_shapes:
            if shape != recurrent_bias_shapes[suffix]:
                raise ValueError(f"invalid recurrent bias shape for {name}: {shape}")
            return "recurrent_bias"
        if suffix == "mix.onorm.weight":
            if shape != (16,):
                raise ValueError(f"invalid recurrent normalization shape for {name}: {shape}")
            return "normalization_scale"
    raise ValueError(f"unrecognized parameter name or block role: {name}")


def copy_compatible_state(
    source: nn.Module,
    destination: nn.Module,
    *,
    include_router: bool = True,
) -> StateCopyReport:
    source_state = source.state_dict()
    destination_state = destination.state_dict()
    compatible = []
    incompatible_source = []
    staged = {}
    for name, value in source_state.items():
        target = destination_state.get(name)
        router_allowed = include_router or not is_router_parameter(name)
        if target is not None and target.shape == value.shape and router_allowed:
            if target.dtype != value.dtype:
                raise ValueError(f"state tensor dtype mismatch for {name}")
            if target.device != value.device:
                raise ValueError(f"state tensor device mismatch for {name}")
            source_identity = _state_tensor_identity(value)
            with torch.no_grad():
                candidate = torch.empty_like(target)
                candidate.copy_(value)
            if _state_tensor_identity(candidate) != source_identity:
                raise RuntimeError(f"state tensor staging identity mismatch for {name}")
            compatible.append(name)
            staged[name] = (candidate, source_identity)
        else:
            incompatible_source.append(name)
    compatible_set = set(compatible)
    incompatible_destination = [
        name for name in destination_state if name not in compatible_set
    ]
    with torch.no_grad():
        for name in compatible:
            destination_state[name].copy_(staged[name][0])
    for name in compatible:
        if _state_tensor_identity(destination_state[name]) != staged[name][1]:
            raise RuntimeError(f"state tensor destination identity mismatch for {name}")
    return StateCopyReport(
        compatible=tuple(compatible),
        incompatible_source=tuple(incompatible_source),
        incompatible_destination=tuple(incompatible_destination),
    )


def named_parameter_categories(model: nn.Module) -> Dict[str, str]:
    return {
        name: parameter_category(name, parameter)
        for name, parameter in model.named_parameters()
    }
