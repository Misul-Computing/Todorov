from __future__ import annotations

import hashlib
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import best_codec, decompress_best, decode_codec_name, encode_codec_name, learn_indent_unit, random_block, read_joined, restore_structure, train_paths, transform_structure
from neuroloc.simulations.memory.local_100k_source_subtoken_shared_dictionary_corpus_codec import ZSTD_CHARGED_PUBLIC_BITS, ZSTD_UNDERCHARGED_PUBLIC_BITS, standard_payload_bits, token_candidates
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import FROZEN_BLOCKS, corpus_blocks, read_block
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import decode_unsigned_varint, delta_decode, delta_encode, dictionary_stream, encode_unsigned_varint, shuffle_bytes, tokens_from_stream

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_subtoken_global_stream_corpus_codec"
SEED = env_int("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_SEED", 12829)
MAX_BLOCK_BYTES = env_int("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_MAX_BLOCK_BYTES", 250000)
SHARED_TOKEN_COUNT = env_int("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_SHARED_TOKEN_COUNT", 256)
ONE_BYTE_TOKEN_COUNT = env_int("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_ONE_BYTE_TOKEN_COUNT", 120)
GLOBAL_HEADER_BITS = env_int("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_GLOBAL_HEADER_BITS", 2048)
GLOBAL_RAW_STANDARD_HEADER_BITS = env_int("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_GLOBAL_RAW_STANDARD_HEADER_BITS", 64)
PRIOR_SHARED_DICTIONARY_PAYLOAD_BITS = float(os.environ.get("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_PRIOR_SHARED_PAYLOAD_BITS", "803400"))
PRIOR_SUBTOKEN_CORPUS_PAYLOAD_BITS = float(os.environ.get("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_PRIOR_SUBTOKEN_PAYLOAD_BITS", "812688"))

require_positive("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_MAX_BLOCK_BYTES", MAX_BLOCK_BYTES)
require_positive("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_SHARED_TOKEN_COUNT", SHARED_TOKEN_COUNT)
require_positive("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_ONE_BYTE_TOKEN_COUNT", ONE_BYTE_TOKEN_COUNT)
require_positive("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_GLOBAL_HEADER_BITS", GLOBAL_HEADER_BITS)
require_positive("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_GLOBAL_RAW_STANDARD_HEADER_BITS", GLOBAL_RAW_STANDARD_HEADER_BITS)
if int(ONE_BYTE_TOKEN_COUNT) > 127:
    raise ValueError("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_ONE_BYTE_TOKEN_COUNT must be <= 127")
if int(SHARED_TOKEN_COUNT) < int(ONE_BYTE_TOKEN_COUNT):
    raise ValueError("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_SHARED_TOKEN_COUNT must be >= one byte token count")

PROFILES = {
    "smoke": {"block_count": 3, "min_improvement": 0.16, "min_prior_margin_bits": 50000.0, "min_global_raw_margin_bits": 15000.0},
    "hard": {"block_count": 5, "min_improvement": 0.17, "min_prior_margin_bits": 90000.0, "min_global_raw_margin_bits": 30000.0},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_SUBTOKEN_GLOBAL_STREAM_CORPUS_CODEC_PROFILE must be smoke or hard")
    return value


def target_rows(profile: str) -> list[dict[str, Any]]:
    return FROZEN_BLOCKS[: int(PROFILES[profile]["block_count"])]


def read_limited_block(row: dict[str, Any]) -> bytes:
    return bytes(read_block(row)[: int(MAX_BLOCK_BYTES)])


def length_stream(rows: list[dict[str, Any]]) -> bytes:
    out = bytearray()
    for row in rows:
        out.extend(encode_unsigned_varint(int(row["count_stream_len"])))
        out.extend(encode_unsigned_varint(int(row["body_stream_len"])))
        out.extend(encode_unsigned_varint(int(row["line_count"])))
    return bytes(out)


def length_rows(stream: bytes, count: int) -> list[dict[str, int]]:
    rows = []
    index = 0
    for _item in range(int(count)):
        count_stream_len, index = decode_unsigned_varint(stream, index)
        body_stream_len, index = decode_unsigned_varint(stream, index)
        line_count, index = decode_unsigned_varint(stream, index)
        rows.append({"count_stream_len": int(count_stream_len), "body_stream_len": int(body_stream_len), "line_count": int(line_count)})
    if index != len(stream):
        raise ValueError("unused length stream bytes")
    return rows


def substitute_global_tokens(body: bytes, tokens: list[bytes], one_byte_count: int) -> bytes:
    if not tokens:
        return body
    one_byte_count = int(min(max(int(one_byte_count), 0), len(tokens)))
    one_byte_ids = {token: 128 + index for index, token in enumerate(tokens[:one_byte_count])}
    extended_ids = {token: index for index, token in enumerate(tokens[one_byte_count:])}
    ordered = sorted(tokens, key=lambda token: (-len(token), token))
    out = bytearray()
    index = 0
    while index < len(body):
        chosen = b""
        for token in ordered:
            if body.startswith(token, index):
                chosen = token
                break
        if not chosen:
            out.append(int(body[index]))
            index += 1
        elif chosen in one_byte_ids:
            out.append(int(one_byte_ids[chosen]))
            index += len(chosen)
        else:
            out.append(255)
            out.extend(encode_unsigned_varint(int(extended_ids[chosen])))
            index += len(chosen)
    return bytes(out)


def restore_global_tokens(substituted: bytes, tokens: list[bytes], one_byte_count: int) -> bytes:
    if not tokens:
        return substituted
    one_byte_count = int(min(max(int(one_byte_count), 0), len(tokens)))
    out = bytearray()
    index = 0
    while index < len(substituted):
        value = int(substituted[index])
        index += 1
        if 128 <= value < 128 + one_byte_count:
            out.extend(tokens[value - 128])
        elif value == 255:
            token_index, index = decode_unsigned_varint(substituted, index)
            target_index = one_byte_count + int(token_index)
            if target_index < one_byte_count or target_index >= len(tokens):
                raise ValueError("extended token out of range")
            out.extend(tokens[target_index])
        else:
            out.append(value)
    return bytes(out)


def global_codec(blocks: list[bytes]) -> dict[str, Any]:
    train_block = read_joined(train_paths())
    indent_unit = learn_indent_unit(train_block)
    transformed = [transform_structure(block, indent_unit) for block in blocks]
    body_streams = [bytes(row["body_stream"]) for row in transformed]
    shared_tokens = token_candidates(body_streams, int(SHARED_TOKEN_COUNT))
    shared_dictionary = dictionary_stream(shared_tokens)
    if shared_dictionary:
        shared_dictionary_codec_name, shared_dictionary_payload = best_codec(shared_dictionary)
    else:
        shared_dictionary_codec_name, shared_dictionary_payload = "none0", b""
    rows = []
    count_stream = bytearray()
    body_stream = bytearray()
    for row in transformed:
        raw_count = bytes(row["count_stream"])
        substituted_body = substitute_global_tokens(bytes(row["body_stream"]), shared_tokens, int(ONE_BYTE_TOKEN_COUNT))
        count_stream.extend(raw_count)
        body_stream.extend(substituted_body)
        rows.append(
            {
                "line_count": int(row["line_count"]),
                "count_stream_len": int(len(raw_count)),
                "body_stream_len": int(len(substituted_body)),
                "original_body_stream_len": int(len(bytes(row["body_stream"]))),
            }
        )
    count_delta = delta_encode(bytes(count_stream))
    lengths = length_stream(rows)
    count_codec_name, count_payload = best_codec(count_delta)
    body_codec_name, body_payload = best_codec(bytes(body_stream))
    length_codec_name, length_payload = best_codec(lengths)
    return {
        "indent_unit": int(indent_unit),
        "block_count": int(len(blocks)),
        "shared_token_count": int(len(shared_tokens)),
        "one_byte_token_count": int(min(int(ONE_BYTE_TOKEN_COUNT), len(shared_tokens))),
        "shared_dictionary_codec_name": shared_dictionary_codec_name,
        "count_codec_name": count_codec_name,
        "body_codec_name": body_codec_name,
        "length_codec_name": length_codec_name,
        "shared_dictionary_payload": shared_dictionary_payload,
        "count_payload": count_payload,
        "body_payload": body_payload,
        "length_payload": length_payload,
        "rows": rows,
        "raw_length_stream_len": int(len(lengths)),
        "count_stream_len": int(len(count_stream)),
        "body_stream_len": int(len(body_stream)),
    }


def shared_tokens_from_codec(codec: dict[str, Any]) -> list[bytes]:
    count = int(codec["shared_token_count"])
    if count <= 0:
        return []
    dictionary = decompress_best(str(codec["shared_dictionary_codec_name"]), bytes(codec["shared_dictionary_payload"]))
    return tokens_from_stream(dictionary, count)


def restore_all(codec: dict[str, Any]) -> list[bytes]:
    shared_tokens = shared_tokens_from_codec(codec)
    count_stream = delta_decode(decompress_best(str(codec["count_codec_name"]), bytes(codec["count_payload"])))
    body_stream = decompress_best(str(codec["body_codec_name"]), bytes(codec["body_payload"]))
    length_payload = decompress_best(str(codec["length_codec_name"]), bytes(codec["length_payload"]))
    rows = length_rows(length_payload, int(codec["block_count"]))
    count_index = 0
    body_index = 0
    out = []
    for row in rows:
        count_end = count_index + int(row["count_stream_len"])
        body_end = body_index + int(row["body_stream_len"])
        count_slice = count_stream[count_index:count_end]
        body_slice = body_stream[body_index:body_end]
        if len(count_slice) != int(row["count_stream_len"]) or len(body_slice) != int(row["body_stream_len"]):
            raise ValueError("truncated global stream")
        restored_body = restore_global_tokens(body_slice, shared_tokens, int(codec["one_byte_token_count"]))
        out.append(restore_structure(count_slice, restored_body, int(row["line_count"]), int(codec["indent_unit"])))
        count_index = count_end
        body_index = body_end
    if count_index != len(count_stream) or body_index != len(body_stream):
        raise ValueError("unused global stream bytes")
    return out


def codec_payload_bits(codec: dict[str, Any]) -> int:
    return int(
        len(bytes(codec["shared_dictionary_payload"])) * 8
        + len(bytes(codec["count_payload"])) * 8
        + len(bytes(codec["body_payload"])) * 8
        + len(bytes(codec["length_payload"])) * 8
        + int(GLOBAL_HEADER_BITS)
    )


def encode_optional_codec_name(name: str) -> int:
    if str(name) == "none0":
        return 0
    return int(encode_codec_name(str(name)))


def decode_optional_codec_name(code: int) -> str:
    if int(code) == 0:
        return "none0"
    return decode_codec_name(int(code))


class SourceSubtokenGlobalStreamCorpusModule(nn.Module):
    def __init__(self, codec: dict[str, Any] | None = None, state_shapes: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        if codec is None and state_shapes is None:
            raise ValueError("codec or state shapes required")
        if codec is not None:
            header = torch.tensor(
                [
                    int(codec["indent_unit"]),
                    int(codec["block_count"]),
                    int(codec["shared_token_count"]),
                    int(codec["one_byte_token_count"]),
                    encode_optional_codec_name(str(codec["shared_dictionary_codec_name"])),
                    encode_optional_codec_name(str(codec["count_codec_name"])),
                    encode_optional_codec_name(str(codec["body_codec_name"])),
                    encode_optional_codec_name(str(codec["length_codec_name"])),
                ],
                dtype=torch.int64,
            )
            self.register_buffer("global_header", header, persistent=True)
            self.register_buffer("shared_dictionary_payload", torch.tensor(list(bytes(codec["shared_dictionary_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("count_payload", torch.tensor(list(bytes(codec["count_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("body_payload", torch.tensor(list(bytes(codec["body_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("length_payload", torch.tensor(list(bytes(codec["length_payload"])), dtype=torch.uint8), persistent=True)
        else:
            for name in ("global_header", "shared_dictionary_payload", "count_payload", "body_payload", "length_payload"):
                self.register_buffer(name, torch.empty_like(state_shapes[name]), persistent=True)

    @classmethod
    def empty_from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "SourceSubtokenGlobalStreamCorpusModule":
        return cls(state_shapes=state_dict)

    def codec(self) -> dict[str, Any]:
        header = [int(item) for item in self.global_header.tolist()]
        return {
            "indent_unit": header[0],
            "block_count": header[1],
            "shared_token_count": header[2],
            "one_byte_token_count": header[3],
            "shared_dictionary_codec_name": decode_optional_codec_name(header[4]),
            "count_codec_name": decode_optional_codec_name(header[5]),
            "body_codec_name": decode_optional_codec_name(header[6]),
            "length_codec_name": decode_optional_codec_name(header[7]),
            "shared_dictionary_payload": bytes(int(item) for item in self.shared_dictionary_payload.tolist()),
            "count_payload": bytes(int(item) for item in self.count_payload.tolist()),
            "body_payload": bytes(int(item) for item in self.body_payload.tolist()),
            "length_payload": bytes(int(item) for item in self.length_payload.tolist()),
        }

    def reconstruct(self) -> list[bytes]:
        return restore_all(self.codec())


def model_state_probe(codec: dict[str, Any], blocks: list[bytes]) -> dict[str, float]:
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    preload_success = float(module.reconstruct() == blocks)
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    reload_success = float(reload_module.reconstruct() == blocks)
    state_payload = b"".join(bytes(int(item) for item in state[name].tolist()) for name in ("shared_dictionary_payload", "count_payload", "body_payload", "length_payload"))
    raw_retained = float(any(block[: min(128, len(block))] in state_payload for block in blocks))
    required_keys = {"global_header", "shared_dictionary_payload", "count_payload", "body_payload", "length_payload"}
    return {
        "model_state_codec_payload_used": 1.0,
        "state_dict_buffer_payload_used": float(int(required_keys.issubset(set(state.keys())))),
        "model_state_exact_reconstruction_success": float(preload_success),
        "state_dict_reload_reconstruction_success": float(reload_success),
        "state_dict_raw_source_block_retained": float(raw_retained),
    }


def global_raw_standard_payload_bits(blocks: list[bytes]) -> int:
    raw_lengths = bytearray()
    for block in blocks:
        raw_lengths.extend(encode_unsigned_varint(len(block)))
    _, raw_payload = best_codec(b"".join(blocks))
    _, length_payload = best_codec(bytes(raw_lengths))
    return int(len(raw_payload) * 8 + len(length_payload) * 8 + int(GLOBAL_RAW_STANDARD_HEADER_BITS))


def random_blocks(seed: int, blocks: list[bytes]) -> list[bytes]:
    return [random_block(int(seed) + index * 19, len(block)) for index, block in enumerate(blocks)]


def block_hashes(rows: list[dict[str, Any]], blocks: list[bytes]) -> dict[str, float]:
    values = []
    for row, block in zip(rows, blocks):
        values.append(float(hashlib.sha256(block).hexdigest() == str(row["sha256"])))
    return {"frozen_manifest_hash_success_min": float(min(values))}


def control_success(codec: dict[str, Any], blocks: list[bytes], seed: int) -> dict[str, float]:
    wrong_indent = dict(codec)
    wrong_indent["indent_unit"] = int(wrong_indent["indent_unit"]) + 1
    try:
        wrong_indent_success = float(restore_all(wrong_indent) == blocks)
    except Exception:
        wrong_indent_success = 0.0
    shared_disabled = dict(codec)
    shared_disabled["shared_token_count"] = 0
    shared_disabled["shared_dictionary_payload"] = b""
    shared_disabled["shared_dictionary_codec_name"] = "none0"
    try:
        shared_disabled_success = float(restore_all(shared_disabled) == blocks)
    except Exception:
        shared_disabled_success = 0.0
    shuffled_shared = dict(codec)
    shuffled_shared["shared_dictionary_payload"] = shuffle_bytes(bytes(codec["shared_dictionary_payload"]), int(seed))
    try:
        shuffled_shared_success = float(restore_all(shuffled_shared) == blocks)
    except Exception:
        shuffled_shared_success = 0.0
    shuffled_body = dict(codec)
    shuffled_body["body_payload"] = shuffle_bytes(bytes(codec["body_payload"]), int(seed) + 31)
    try:
        shuffled_body_success = float(restore_all(shuffled_body) == blocks)
    except Exception:
        shuffled_body_success = 0.0
    shuffled_count = dict(codec)
    shuffled_count["count_payload"] = shuffle_bytes(bytes(codec["count_payload"]), int(seed) + 47)
    try:
        shuffled_count_success = float(restore_all(shuffled_count) == blocks)
    except Exception:
        shuffled_count_success = 0.0
    shuffled_length = dict(codec)
    shuffled_length["length_payload"] = shuffle_bytes(bytes(codec["length_payload"]), int(seed) + 59)
    try:
        shuffled_length_success = float(restore_all(shuffled_length) == blocks)
    except Exception:
        shuffled_length_success = 0.0
    return {
        "wrong_indent_unit_exact_reconstruction_success": wrong_indent_success,
        "shared_dictionary_disabled_exact_reconstruction_success": shared_disabled_success,
        "shuffled_shared_dictionary_exact_reconstruction_success": shuffled_shared_success,
        "shuffled_body_payload_exact_reconstruction_success": shuffled_body_success,
        "shuffled_count_payload_exact_reconstruction_success": shuffled_count_success,
        "shuffled_length_payload_exact_reconstruction_success": shuffled_length_success,
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    rows = target_rows(profile)
    blocks = [read_limited_block(row) for row in rows]
    codec = global_codec(blocks)
    restored = restore_all(codec)
    standard_bits = standard_payload_bits(blocks)
    global_standard_bits = global_raw_standard_payload_bits(blocks)
    selected_bits = codec_payload_bits(codec)
    random_payloads = random_blocks(int(seed), blocks)
    random_codec = global_codec(random_payloads)
    random_standard_bits = standard_payload_bits(random_payloads)
    random_global_standard_bits = global_raw_standard_payload_bits(random_payloads)
    random_selected_bits = codec_payload_bits(random_codec)
    aggregate_improvement = float(standard_bits - selected_bits) / max(float(standard_bits), 1.0)
    global_standard_improvement = float(global_standard_bits - selected_bits) / max(float(global_standard_bits), 1.0)
    global_standard_margin = float(global_standard_bits - selected_bits)
    prior_shared_margin = float(PRIOR_SHARED_DICTIONARY_PAYLOAD_BITS - selected_bits)
    prior_subtoken_margin = float(PRIOR_SUBTOKEN_CORPUS_PAYLOAD_BITS - selected_bits)
    random_improvement = float(random_standard_bits - random_selected_bits) / max(float(random_standard_bits), 1.0)
    random_global_improvement = float(random_global_standard_bits - random_selected_bits) / max(float(random_global_standard_bits), 1.0)
    hash_success = block_hashes(rows, blocks)["frozen_manifest_hash_success_min"]
    controls = control_success(codec, blocks, int(seed))
    state_probe = model_state_probe(codec, blocks)
    controls_collapse = float(
        int(
            controls["wrong_indent_unit_exact_reconstruction_success"] == 0.0
            and controls["shared_dictionary_disabled_exact_reconstruction_success"] == 0.0
            and controls["shuffled_shared_dictionary_exact_reconstruction_success"] == 0.0
            and controls["shuffled_body_payload_exact_reconstruction_success"] == 0.0
            and controls["shuffled_count_payload_exact_reconstruction_success"] == 0.0
            and controls["shuffled_length_payload_exact_reconstruction_success"] == 0.0
        )
    )
    engineering_pass = float(
        int(
            restored == blocks
            and hash_success == 1.0
            and aggregate_improvement >= float(PROFILES[profile]["min_improvement"])
            and global_standard_margin >= float(PROFILES[profile]["min_global_raw_margin_bits"])
            and prior_shared_margin >= float(PROFILES[profile]["min_prior_margin_bits"])
            and random_improvement <= 0.0
            and random_global_improvement <= 0.0
            and controls_collapse == 1.0
            and state_probe["model_state_codec_payload_used"] == 1.0
            and state_probe["state_dict_buffer_payload_used"] == 1.0
            and state_probe["model_state_exact_reconstruction_success"] == 1.0
            and state_probe["state_dict_reload_reconstruction_success"] == 1.0
            and state_probe["state_dict_raw_source_block_retained"] == 0.0
        )
    )
    return {
        "profile": profile,
        "block_count": float(len(blocks)),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "source_subtoken_global_stream_corpus_codec_candidate": engineering_pass,
        "source_code_corpus_codec_product_authorized": engineering_pass,
        "source_code_corpus_codec_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "general_unknown_structure_breakthrough_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "exact_reconstruction_success_min": float(int(restored == blocks)),
        "frozen_manifest_hash_success_min": float(hash_success),
        "useful_retrievable_bits": float(sum(len(block) * 8 for block in blocks)),
        "aggregate_standard_payload_bits": float(standard_bits),
        "global_raw_standard_payload_bits": float(global_standard_bits),
        "prior_shared_dictionary_corpus_payload_bits": float(PRIOR_SHARED_DICTIONARY_PAYLOAD_BITS),
        "prior_subtoken_corpus_payload_bits": float(PRIOR_SUBTOKEN_CORPUS_PAYLOAD_BITS),
        "aggregate_selected_payload_bits": float(selected_bits),
        "aggregate_payload_improvement": float(aggregate_improvement),
        "global_raw_standard_payload_improvement": float(global_standard_improvement),
        "margin_over_global_raw_standard_bits": float(global_standard_margin),
        "aggregate_payload_margin_over_prior_shared_bits": float(prior_shared_margin),
        "aggregate_payload_margin_over_prior_subtoken_bits": float(prior_subtoken_margin),
        "aggregate_payload_improvement_delta_over_prior_shared": float(prior_shared_margin / max(float(PRIOR_SHARED_DICTIONARY_PAYLOAD_BITS), 1.0)),
        "shared_token_count": float(codec["shared_token_count"]),
        "one_byte_token_count": float(codec["one_byte_token_count"]),
        "local_token_count_per_block": 0.0,
        "shared_dictionary_payload_bits": float(len(bytes(codec["shared_dictionary_payload"])) * 8),
        "global_count_payload_bits": float(len(bytes(codec["count_payload"])) * 8),
        "global_body_payload_bits": float(len(bytes(codec["body_payload"])) * 8),
        "global_length_payload_bits": float(len(bytes(codec["length_payload"])) * 8),
        "global_header_bits": float(GLOBAL_HEADER_BITS),
        "global_raw_standard_header_bits": float(GLOBAL_RAW_STANDARD_HEADER_BITS),
        "raw_length_stream_bytes": float(codec["raw_length_stream_len"]),
        "global_count_stream_bytes": float(codec["count_stream_len"]),
        "global_body_stream_bytes": float(codec["body_stream_len"]),
        "zstd_charged_public_baseline_bits": float(ZSTD_CHARGED_PUBLIC_BITS),
        "zstd_undercharged_public_baseline_bits": float(ZSTD_UNDERCHARGED_PUBLIC_BITS),
        "margin_over_zstd_charged_public_bits": float(float(ZSTD_CHARGED_PUBLIC_BITS) - selected_bits),
        "margin_over_zstd_undercharged_public_bits": float(float(ZSTD_UNDERCHARGED_PUBLIC_BITS) - selected_bits),
        "random_label_payload_incompressible": float(int(random_selected_bits >= random_standard_bits)),
        "random_label_payload_improvement_over_best_standard": float(random_improvement),
        "random_label_global_raw_payload_incompressible": float(int(random_selected_bits >= random_global_standard_bits)),
        "random_label_global_raw_payload_improvement": float(random_global_improvement),
        "random_label_selected_payload_bits": float(random_selected_bits),
        "random_label_best_standard_payload_bits": float(random_standard_bits),
        "random_label_global_raw_standard_payload_bits": float(random_global_standard_bits),
        "controls_collapse": controls_collapse,
        "shared_dictionary_payload_retained": 1.0,
        "global_count_payload_retained": 1.0,
        "global_body_payload_retained": 1.0,
        "global_length_payload_retained": 1.0,
        "per_block_local_dictionary_payload_retained": 0.0,
        "raw_source_block_retained": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "engineering_pass": engineering_pass,
        **controls,
        **state_probe,
    }


@lru_cache(maxsize=8)
def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["engineering_pass"]),
    }
    for key, value in row.items():
        if key == "profile":
            continue
        if isinstance(value, str):
            continue
        summary[f"{SIMULATION_ID}_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_source_subtoken_global_stream_corpus_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "shared_token_count": int(SHARED_TOKEN_COUNT), "global_header_bits": int(GLOBAL_HEADER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_block_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_subtoken_global_stream_corpus_codec_metrics.json", "path": metrics_path}],
        warnings=["source-code corpus global-stream codec only; no nm, chat, knowledge, paid-compute, or broad breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} aggregate_improvement={summary[f'{SIMULATION_ID}_aggregate_payload_improvement']:.6f} prior_shared_margin_bits={summary[f'{SIMULATION_ID}_aggregate_payload_margin_over_prior_shared_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
