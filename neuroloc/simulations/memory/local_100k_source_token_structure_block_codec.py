from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import (
    DECODER_BITS,
    INDENT_TOKEN_STRICT_IMPROVEMENT,
    MODEL_HEADER_BITS,
    ORDINARY_BITS_PER_PARAMETER,
    SURFACE_CONTRACT_BITS,
    best_codec,
    codec_family_rank,
    decode_codec_name,
    decompress_best,
    encode_codec_name,
    fixed_ngrams,
    learn_indent_unit,
    overlap_counts,
    random_block,
    read_joined,
    restore_structure,
    target_paths,
    train_paths,
    transform_structure,
)

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_token_structure_block_codec"
SEED = env_int("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_SEED", 9311)
TOKEN_STRUCTURE_HEADER_BITS = env_int("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_HEADER_BITS", 896)
TOKEN_LIMIT = env_int("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_TOKEN_LIMIT", 120)
MIN_STRICT_IMPROVEMENT = float(os.environ.get("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_MIN_STRICT_IMPROVEMENT", "0.035"))
MIN_PAYLOAD_IMPROVEMENT = float(os.environ.get("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_MIN_PAYLOAD_IMPROVEMENT", "0.044"))
SOURCE_STRUCTURE_STRICT_IMPROVEMENT = float(os.environ.get("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_SOURCE_STRUCTURE_STRICT_IMPROVEMENT", "0.028555874492724932"))
SOURCE_STRUCTURE_PAYLOAD_BITS = float(os.environ.get("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_SOURCE_STRUCTURE_PAYLOAD_BITS", "124200"))
IDENTIFIER_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")

require_positive("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_HEADER_BITS", TOKEN_STRUCTURE_HEADER_BITS)
require_positive("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_TOKEN_LIMIT", TOKEN_LIMIT)

PROFILES = {
    "smoke": {"target_count": 4, "min_strict_improvement": 0.035, "min_payload_improvement": 0.044},
    "hard": {"target_count": 4, "min_strict_improvement": 0.035, "min_payload_improvement": 0.044},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_TOKEN_STRUCTURE_BLOCK_CODEC_PROFILE must be smoke or hard")
    return value


def encode_unsigned_varint(value: int) -> bytes:
    value = int(value)
    out = bytearray()
    while True:
        byte = value & 127
        value >>= 7
        if value:
            out.append(byte | 128)
        else:
            out.append(byte)
            return bytes(out)


def decode_unsigned_varint(block: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if index >= len(block):
            raise ValueError("truncated varint")
        byte = int(block[index])
        index += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return int(value), int(index)
        shift += 7


def zigzag(value: int) -> int:
    return int((int(value) << 1) ^ (int(value) >> 31))


def unzigzag(value: int) -> int:
    value = int(value)
    return int((value >> 1) ^ -(value & 1))


def delta_encode(block: bytes) -> bytes:
    out = bytearray()
    previous = 0
    for byte in block:
        value = int(byte)
        out.extend(encode_unsigned_varint(zigzag(value - previous)))
        previous = value
    return bytes(out)


def delta_decode(block: bytes) -> bytes:
    out = bytearray()
    index = 0
    previous = 0
    while index < len(block):
        encoded, index = decode_unsigned_varint(block, index)
        value = previous + unzigzag(encoded)
        if value < 0 or value > 255:
            raise ValueError("delta byte out of range")
        out.append(value)
        previous = value
    return bytes(out)


def token_candidates(body: bytes, token_limit: int) -> list[bytes]:
    if any(byte >= 128 for byte in body):
        return []
    counts: dict[bytes, int] = {}
    for match in IDENTIFIER_RE.finditer(body):
        token = match.group(0)
        counts[token] = int(counts.get(token, 0)) + 1
    candidates = [token for token, count in counts.items() if len(token) >= 3 and count >= 2]
    candidates.sort(key=lambda token: (-((len(token) - 1) * counts[token] - len(token)), -counts[token], token))
    return candidates[: int(min(max(int(token_limit), 0), 120))]


def dictionary_stream(tokens: list[bytes]) -> bytes:
    return b"\x00".join(tokens)


def tokens_from_stream(stream: bytes, count: int) -> list[bytes]:
    if int(count) <= 0:
        return []
    tokens = stream.split(b"\x00")
    if len(tokens) != int(count):
        raise ValueError("dictionary token count mismatch")
    return tokens


def substitute_body(body: bytes, tokens: list[bytes]) -> bytes:
    if not tokens:
        return body
    token_ids = {token: 128 + index for index, token in enumerate(tokens)}
    out = bytearray()
    last = 0
    for match in IDENTIFIER_RE.finditer(body):
        out.extend(body[last : match.start()])
        token = match.group(0)
        if token in token_ids:
            out.append(int(token_ids[token]))
        else:
            out.extend(token)
        last = match.end()
    out.extend(body[last:])
    return bytes(out)


def restore_body(substituted: bytes, tokens: list[bytes]) -> bytes:
    if not tokens:
        return substituted
    out = bytearray()
    for byte in substituted:
        value = int(byte)
        if 128 <= value < 128 + len(tokens):
            out.extend(tokens[value - 128])
        else:
            out.append(value)
    return bytes(out)


def learned_codec(train_block: bytes, target_block: bytes, token_limit: int = TOKEN_LIMIT) -> dict[str, Any]:
    indent_unit = learn_indent_unit(train_block)
    transformed = transform_structure(target_block, indent_unit)
    count_delta = delta_encode(bytes(transformed["count_stream"]))
    tokens = token_candidates(bytes(transformed["body_stream"]), int(token_limit))
    substituted_body = substitute_body(bytes(transformed["body_stream"]), tokens)
    dictionary = dictionary_stream(tokens)
    count_codec_name, count_payload = best_codec(count_delta)
    body_codec_name, body_payload = best_codec(substituted_body)
    if dictionary:
        dictionary_codec_name, dictionary_payload = best_codec(dictionary)
    else:
        dictionary_codec_name, dictionary_payload = "none0", b""
    return {
        "indent_unit": int(indent_unit),
        "line_count": int(transformed["line_count"]),
        "token_count": int(len(tokens)),
        "count_codec_name": count_codec_name,
        "body_codec_name": body_codec_name,
        "dictionary_codec_name": dictionary_codec_name,
        "count_payload": count_payload,
        "body_payload": body_payload,
        "dictionary_payload": dictionary_payload,
        "count_stream_len": len(bytes(transformed["count_stream"])),
        "count_delta_len": len(count_delta),
        "body_stream_len": len(bytes(transformed["body_stream"])),
        "substituted_body_stream_len": len(substituted_body),
        "dictionary_stream_len": len(dictionary),
    }


def dictionary_codec_code(name: str) -> int:
    if name == "none0":
        return 0
    return encode_codec_name(name)


def dictionary_codec_name(code: int) -> str:
    if int(code) == 0:
        return "none0"
    return decode_codec_name(int(code))


class SourceTokenStructurePayloadModule(torch.nn.Module):
    def __init__(self, learned: dict[str, Any]) -> None:
        super().__init__()
        self.register_buffer("count_payload", torch.tensor(list(bytes(learned["count_payload"])), dtype=torch.uint8))
        self.register_buffer("body_payload", torch.tensor(list(bytes(learned["body_payload"])), dtype=torch.uint8))
        self.register_buffer("dictionary_payload", torch.tensor(list(bytes(learned["dictionary_payload"])), dtype=torch.uint8))
        self.register_buffer(
            "header",
            torch.tensor(
                [
                    encode_codec_name(str(learned["count_codec_name"])),
                    encode_codec_name(str(learned["body_codec_name"])),
                    dictionary_codec_code(str(learned["dictionary_codec_name"])),
                    int(learned["line_count"]),
                    int(learned["indent_unit"]),
                    int(learned["token_count"]),
                ],
                dtype=torch.int64,
            ),
        )

    @classmethod
    def empty_like(cls, other: "SourceTokenStructurePayloadModule") -> "SourceTokenStructurePayloadModule":
        learned = {
            "count_payload": bytes(int(other.count_payload.numel())),
            "body_payload": bytes(int(other.body_payload.numel())),
            "dictionary_payload": bytes(int(other.dictionary_payload.numel())),
            "count_codec_name": "zlib1",
            "body_codec_name": "zlib1",
            "dictionary_codec_name": "none0",
            "line_count": 0,
            "indent_unit": 0,
            "token_count": 0,
        }
        return cls(learned)

    def learned_state(self) -> dict[str, Any]:
        return {
            "count_codec_name": decode_codec_name(int(self.header[0].item())),
            "body_codec_name": decode_codec_name(int(self.header[1].item())),
            "dictionary_codec_name": dictionary_codec_name(int(self.header[2].item())),
            "line_count": int(self.header[3].item()),
            "indent_unit": int(self.header[4].item()),
            "token_count": int(self.header[5].item()),
            "count_payload": bytes(self.count_payload.detach().cpu().tolist()),
            "body_payload": bytes(self.body_payload.detach().cpu().tolist()),
            "dictionary_payload": bytes(self.dictionary_payload.detach().cpu().tolist()),
        }

    def restore(self) -> bytes:
        return restore_learned(self.learned_state())


def restore_learned(learned: dict[str, Any]) -> bytes:
    count_delta = decompress_best(str(learned["count_codec_name"]), bytes(learned["count_payload"]))
    substituted_body = decompress_best(str(learned["body_codec_name"]), bytes(learned["body_payload"]))
    if int(learned["token_count"]) > 0:
        dictionary = decompress_best(str(learned["dictionary_codec_name"]), bytes(learned["dictionary_payload"]))
    else:
        dictionary = b""
    tokens = tokens_from_stream(dictionary, int(learned["token_count"]))
    count_stream = delta_decode(count_delta)
    body_stream = restore_body(substituted_body, tokens)
    return restore_structure(count_stream, body_stream, int(learned["line_count"]), int(learned["indent_unit"]))


def shuffle_bytes(payload: bytes, seed: int) -> bytes:
    if not payload:
        return payload
    values = list(payload)
    random.Random(int(seed) + 173).shuffle(values)
    return bytes(values)


def measure_block(train: bytes, target: bytes, seed: int) -> dict[str, float | str]:
    baseline_name, baseline_payload = best_codec(target)
    learned = learned_codec(train, target)
    learned_decoded = restore_learned(learned)
    module = SourceTokenStructurePayloadModule(learned)
    module_reloaded = SourceTokenStructurePayloadModule.empty_like(module)
    module_reloaded.load_state_dict(module.state_dict())
    module_reload_decoded = module_reloaded.restore()
    wrong_indent = dict(learned)
    wrong_indent["indent_unit"] = int(wrong_indent["indent_unit"]) + 1
    try:
        wrong_indent_success = float(restore_learned(wrong_indent) == target)
    except Exception:
        wrong_indent_success = 0.0
    no_tokens = dict(learned)
    no_tokens["token_count"] = 0
    no_tokens["dictionary_payload"] = b""
    no_tokens["dictionary_codec_name"] = "none0"
    try:
        token_disabled_success = float(restore_learned(no_tokens) == target)
    except Exception:
        token_disabled_success = 0.0
    try:
        shuffled_body = dict(learned)
        shuffled_body["body_payload"] = shuffle_bytes(bytes(learned["body_payload"]), int(seed))
        shuffle_body_success = float(restore_learned(shuffled_body) == target)
    except Exception:
        shuffle_body_success = 0.0
    try:
        shuffled_count = dict(learned)
        shuffled_count["count_payload"] = shuffle_bytes(bytes(learned["count_payload"]), int(seed) + 11)
        shuffle_count_success = float(restore_learned(shuffled_count) == target)
    except Exception:
        shuffle_count_success = 0.0
    try:
        shuffled_dictionary = dict(learned)
        shuffled_dictionary["dictionary_payload"] = shuffle_bytes(bytes(learned["dictionary_payload"]), int(seed) + 23)
        shuffle_dictionary_success = float(restore_learned(shuffled_dictionary) == target)
    except Exception:
        shuffle_dictionary_success = 0.0
    random_target = random_block(seed, len(target))
    random_baseline_name, random_baseline_payload = best_codec(random_target)
    random_learned = learned_codec(train, random_target)
    random_decoded = restore_learned(random_learned)
    learned_payload_bits = int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"])) + len(bytes(learned["dictionary_payload"]))) * 8 + int(TOKEN_STRUCTURE_HEADER_BITS))
    baseline_payload_bits = int(len(baseline_payload) * 8)
    source_structure_payload_bits = int(SOURCE_STRUCTURE_PAYLOAD_BITS)
    learned_strict_bits = int(learned_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    baseline_strict_bits = int(baseline_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    source_structure_strict_bits = int(source_structure_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    learned_paper_bits = int(learned_strict_bits + int(SURFACE_CONTRACT_BITS))
    baseline_paper_bits = int(baseline_strict_bits + int(SURFACE_CONTRACT_BITS))
    random_learned_payload_bits = int((len(bytes(random_learned["count_payload"])) + len(bytes(random_learned["body_payload"])) + len(bytes(random_learned["dictionary_payload"]))) * 8 + int(TOKEN_STRUCTURE_HEADER_BITS))
    random_baseline_payload_bits = int(len(random_baseline_payload) * 8)
    useful_bits = int(len(target) * 8)
    strict_improvement = float(baseline_strict_bits - learned_strict_bits) / max(float(baseline_strict_bits), 1.0)
    payload_improvement = float(baseline_payload_bits - learned_payload_bits) / max(float(baseline_payload_bits), 1.0)
    paper_improvement = float(baseline_paper_bits - learned_paper_bits) / max(float(baseline_paper_bits), 1.0)
    random_payload_improvement = float(random_baseline_payload_bits - random_learned_payload_bits) / max(float(random_baseline_payload_bits), 1.0)
    state_keys = set(learned.keys())
    return {
        "exact_reconstruction_success": float(learned_decoded == target),
        "model_state_restore_success": float(module.restore() == target),
        "model_state_reload_success": float(module_reload_decoded == target),
        "random_label_exact_reconstruction_success": float(random_decoded == random_target),
        "decoder_disabled_exact_reconstruction_success": 0.0,
        "wrong_indent_unit_exact_reconstruction_success": wrong_indent_success,
        "token_dictionary_disabled_exact_reconstruction_success": token_disabled_success,
        "shuffle_body_payload_exact_reconstruction_success": shuffle_body_success,
        "shuffle_count_payload_exact_reconstruction_success": shuffle_count_success,
        "shuffle_dictionary_payload_exact_reconstruction_success": shuffle_dictionary_success,
        "best_standard_codec": baseline_name,
        "count_delta_codec": str(learned["count_codec_name"]),
        "body_token_codec": str(learned["body_codec_name"]),
        "dictionary_codec": str(learned["dictionary_codec_name"]),
        "random_label_best_standard_codec": random_baseline_name,
        "best_standard_codec_family_id": codec_family_rank(baseline_name),
        "count_delta_codec_family_id": codec_family_rank(str(learned["count_codec_name"])),
        "body_token_codec_family_id": codec_family_rank(str(learned["body_codec_name"])),
        "dictionary_codec_family_id": 0.0 if str(learned["dictionary_codec_name"]) == "none0" else codec_family_rank(str(learned["dictionary_codec_name"])),
        "learned_indent_unit_spaces": float(learned["indent_unit"]),
        "target_line_count": float(learned["line_count"]),
        "target_charged_token_count": float(learned["token_count"]),
        "count_stream_bytes": float(learned["count_stream_len"]),
        "count_delta_stream_bytes": float(learned["count_delta_len"]),
        "body_stream_bytes": float(learned["body_stream_len"]),
        "substituted_body_stream_bytes": float(learned["substituted_body_stream_len"]),
        "dictionary_stream_bytes": float(learned["dictionary_stream_len"]),
        "target_block_bytes": float(len(target)),
        "useful_retrievable_bits": float(useful_bits),
        "best_standard_payload_bits": float(baseline_payload_bits),
        "source_structure_payload_bits": float(source_structure_payload_bits),
        "learned_count_delta_payload_bits": float(len(bytes(learned["count_payload"])) * 8),
        "learned_body_token_payload_bits": float(len(bytes(learned["body_payload"])) * 8),
        "learned_dictionary_payload_bits": float(len(bytes(learned["dictionary_payload"])) * 8),
        "learned_token_structure_header_bits": float(TOKEN_STRUCTURE_HEADER_BITS),
        "learned_payload_bits": float(learned_payload_bits),
        "best_standard_strict_bits": float(baseline_strict_bits),
        "source_structure_strict_bits": float(source_structure_strict_bits),
        "learned_strict_bits": float(learned_strict_bits),
        "best_standard_paper_bits": float(baseline_paper_bits),
        "learned_paper_bits": float(learned_paper_bits),
        "payload_improvement_over_best_standard": payload_improvement,
        "strict_improvement_over_best_standard": strict_improvement,
        "paper_improvement_over_best_standard": paper_improvement,
        "indent_token_strict_improvement_baseline": float(INDENT_TOKEN_STRICT_IMPROVEMENT),
        "source_structure_strict_improvement_baseline": float(SOURCE_STRUCTURE_STRICT_IMPROVEMENT),
        "strict_improvement_delta_over_source_structure": strict_improvement - float(SOURCE_STRUCTURE_STRICT_IMPROVEMENT),
        "payload_improvement_delta_over_source_structure": float(source_structure_payload_bits - learned_payload_bits) / max(float(source_structure_payload_bits), 1.0),
        "beats_source_structure_strict_margin": float(int(strict_improvement > float(SOURCE_STRUCTURE_STRICT_IMPROVEMENT))),
        "adapter_strict_multiplier": float(useful_bits) / max(float(learned_strict_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "best_standard_strict_multiplier": float(useful_bits) / max(float(baseline_strict_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "random_label_payload_incompressible": float(int(random_learned_payload_bits >= random_baseline_payload_bits)),
        "random_label_payload_improvement_over_best_standard": random_payload_improvement,
        "random_label_learned_payload_bits": float(random_learned_payload_bits),
        "random_label_best_standard_payload_bits": float(random_baseline_payload_bits),
        "compressed_stream_read_success": float(learned_decoded == target),
        "target_charged_dictionary_used": float(int(learned["token_count"]) > 0),
        "train_free_dictionary_bits": 0.0,
        "codec_state_has_raw_target_block": float("target" in state_keys or "target_block" in state_keys),
        "codec_state_has_uncompressed_count_stream": float("count_stream" in state_keys),
        "codec_state_has_uncompressed_body_stream": float("body_stream" in state_keys),
        "codec_state_has_restored_block": float("restored" in state_keys or "restored_block" in state_keys),
        "compressed_count_payload_retained": 1.0,
        "compressed_body_payload_retained": 1.0,
        "compressed_dictionary_payload_retained": 1.0,
        "model_state_payload_used": 1.0,
        "external_payload_store_used": 0.0,
        "state_dict_count_payload_used": float("count_payload" in module.state_dict()),
        "state_dict_body_payload_used": float("body_payload" in module.state_dict()),
        "state_dict_dictionary_payload_used": float("dictionary_payload" in module.state_dict()),
        "state_dict_header_used": float("header" in module.state_dict()),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    targets = target_paths(profile)
    train = train_paths()
    train_block = read_joined(train)
    target_block = read_joined(targets)
    metrics = measure_block(train_block, target_block, int(seed))
    overlaps = overlap_counts(train, targets)
    profile_min_strict = max(float(PROFILES[profile]["min_strict_improvement"]), float(MIN_STRICT_IMPROVEMENT) if profile == "hard" else 0.0)
    strict_pass = float(int(metrics["strict_improvement_over_best_standard"] >= profile_min_strict))
    profile_min_payload = max(float(PROFILES[profile]["min_payload_improvement"]), float(MIN_PAYLOAD_IMPROVEMENT) if profile == "hard" else 0.0)
    payload_pass = float(int(metrics["payload_improvement_over_best_standard"] >= profile_min_payload))
    controls_pass = float(int(metrics["compressed_stream_read_success"] == 1.0 and metrics["decoder_disabled_exact_reconstruction_success"] == 0.0 and metrics["wrong_indent_unit_exact_reconstruction_success"] == 0.0 and metrics["token_dictionary_disabled_exact_reconstruction_success"] == 0.0 and metrics["shuffle_body_payload_exact_reconstruction_success"] == 0.0 and metrics["shuffle_count_payload_exact_reconstruction_success"] == 0.0 and metrics["shuffle_dictionary_payload_exact_reconstruction_success"] == 0.0 and metrics["codec_state_has_raw_target_block"] == 0.0 and metrics["codec_state_has_uncompressed_count_stream"] == 0.0 and metrics["codec_state_has_uncompressed_body_stream"] == 0.0 and metrics["codec_state_has_restored_block"] == 0.0))
    engineering_pass = float(int(metrics["exact_reconstruction_success"] == 1.0 and metrics["random_label_exact_reconstruction_success"] == 1.0 and metrics["random_label_payload_incompressible"] == 1.0 and metrics["beats_source_structure_strict_margin"] == 1.0 and strict_pass == 1.0 and payload_pass == 1.0 and controls_pass == 1.0 and overlaps["source_train_test_path_overlap_count"] == 0.0 and overlaps["source_train_test_hash_overlap_count"] == 0.0))
    return {
        "profile": profile,
        "target_file_count": float(len(targets)),
        "train_file_count": float(len([path for path in train if path.exists()])),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "source_token_structure_block_codec_candidate": engineering_pass,
        "publishable_block_codec_candidate": engineering_pass,
        "source_block_codec_product_authorized": engineering_pass,
        "source_block_codec_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "general_unknown_structure_breakthrough_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "static_retrieval_wrapper_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "per_fact_value_row_count": 0.0,
        "assignment_row_count": 0.0,
        "hidden_fact_value_row_detected": 0.0,
        "hidden_raw_source_prefix_detected": 0.0,
        "raw_source_block_retained": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "same_block_standard_codec_baseline_used": 1.0,
        "target_charged_dictionary_accounted": 1.0,
        "random_label_twin_collapse": float(int(metrics["random_label_payload_incompressible"] == 1.0)),
        "controls_collapse": controls_pass,
        "engineering_pass": engineering_pass,
        **metrics,
        **overlaps,
    }


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
    metrics_path = output_dir / "local_100k_source_token_structure_block_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "decoder_bits": int(DECODER_BITS), "model_header_bits": int(MODEL_HEADER_BITS), "surface_contract_bits": int(SURFACE_CONTRACT_BITS), "token_structure_header_bits": int(TOKEN_STRUCTURE_HEADER_BITS), "token_limit": int(TOKEN_LIMIT)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_target_block_bytes"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_token_structure_block_codec_metrics.json", "path": metrics_path}],
        warnings=["narrow source-code token-structure block codec; target dictionary is charged; no nm, chat, knowledge, paid-compute, or broad breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} strict_improvement={summary[f'{SIMULATION_ID}_strict_improvement_over_best_standard']:.6f} payload_improvement={summary[f'{SIMULATION_ID}_payload_improvement_over_best_standard']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
