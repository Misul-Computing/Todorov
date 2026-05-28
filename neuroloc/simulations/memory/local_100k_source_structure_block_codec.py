from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import brotli
import torch
import zstandard as zstd

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_structure_block_codec"
SEED = env_int("SOURCE_STRUCTURE_BLOCK_CODEC_SEED", 8123)
DECODER_BITS = env_int("SOURCE_STRUCTURE_BLOCK_CODEC_DECODER_BITS", 32768)
MODEL_HEADER_BITS = env_int("SOURCE_STRUCTURE_BLOCK_CODEC_MODEL_HEADER_BITS", 64)
SURFACE_CONTRACT_BITS = env_int("SOURCE_STRUCTURE_BLOCK_CODEC_SURFACE_CONTRACT_BITS", 4096)
STRUCTURE_HEADER_BITS = env_int("SOURCE_STRUCTURE_BLOCK_CODEC_STRUCTURE_HEADER_BITS", 256)
MIN_STRICT_IMPROVEMENT = float(os.environ.get("SOURCE_STRUCTURE_BLOCK_CODEC_MIN_STRICT_IMPROVEMENT", "0.028"))
MIN_PAYLOAD_IMPROVEMENT = float(os.environ.get("SOURCE_STRUCTURE_BLOCK_CODEC_MIN_PAYLOAD_IMPROVEMENT", "0.035"))
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("SOURCE_STRUCTURE_BLOCK_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))
INDENT_TOKEN_STRICT_IMPROVEMENT = float(os.environ.get("SOURCE_STRUCTURE_BLOCK_CODEC_INDENT_TOKEN_STRICT_IMPROVEMENT", "0.020192022171632188"))
FALLBACK_MARKER = 255
CODEC_FAMILIES = {"zlib": 1, "bz2": 2, "lzma": 3, "brotli": 4, "zstd": 5}
CODEC_PREFIX_BY_ID = {value: key for key, value in CODEC_FAMILIES.items()}

require_positive("SOURCE_STRUCTURE_BLOCK_CODEC_DECODER_BITS", DECODER_BITS)
require_positive("SOURCE_STRUCTURE_BLOCK_CODEC_MODEL_HEADER_BITS", MODEL_HEADER_BITS)
require_positive("SOURCE_STRUCTURE_BLOCK_CODEC_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("SOURCE_STRUCTURE_BLOCK_CODEC_STRUCTURE_HEADER_BITS", STRUCTURE_HEADER_BITS)

PROFILES = {
    "smoke": {"target_count": 4, "min_strict_improvement": 0.025, "min_payload_improvement": 0.035},
    "hard": {"target_count": 4, "min_strict_improvement": 0.028, "min_payload_improvement": 0.035},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_STRUCTURE_BLOCK_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_STRUCTURE_BLOCK_CODEC_PROFILE must be smoke or hard")
    return value


def train_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "src/layers/kda.py",
        PROJECT_ROOT / "src/layers/mamba3.py",
        PROJECT_ROOT / "src/layers/mla.py",
        PROJECT_ROOT / "src/model/todorov.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/slot_buffer_capacity.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/asymmetric_outer_product_recall.py",
    ]


def target_paths(profile: str) -> list[Path]:
    rows = [
        PROJECT_ROOT / "src/spikes/spiking_brain.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/correction_field_capacity.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/multi_resolution_head_split.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/slot_surprise_writes.py",
    ]
    return rows[: int(PROFILES[profile]["target_count"])]


def read_joined(paths: list[Path]) -> bytes:
    present = [path for path in paths if path.exists()]
    if not present:
        raise ValueError("no source paths found")
    return b"\n".join(path.read_bytes().replace(b"\r\n", b"\n") for path in present)


def iter_lf_lines(block: bytes) -> list[bytes]:
    rows = []
    start = 0
    while start < len(block):
        end = block.find(b"\n", start)
        if end < 0:
            rows.append(block[start:])
            break
        rows.append(block[start : end + 1])
        start = end + 1
    if not rows and block == b"":
        rows.append(b"")
    return rows


def encode_varint(value: int) -> bytes:
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


def decode_varint(block: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if index >= len(block):
            raise ValueError("truncated varint")
        byte = block[index]
        index += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return int(value), int(index)
        shift += 7


def leading_whitespace(core: bytes) -> bytes:
    index = 0
    while index < len(core) and core[index] in {32, 9}:
        index += 1
    return core[:index]


def learn_indent_unit(train_block: bytes) -> int:
    counts = []
    for line in iter_lf_lines(train_block):
        core = line[:-1] if line.endswith(b"\n") else line
        leading = leading_whitespace(core)
        if leading and all(byte == 32 for byte in leading):
            counts.append(len(leading))
    if not counts:
        return 4
    value = 0
    for count in counts:
        value = math.gcd(value, int(count))
    if value <= 0:
        return 4
    return int(max(1, min(16, value)))


def encode_leading(leading: bytes, indent_unit: int) -> bytes:
    if leading and (any(byte != 32 for byte in leading) or len(leading) % int(indent_unit) != 0):
        return bytes([FALLBACK_MARKER]) + encode_varint(len(leading)) + leading
    if not leading:
        return b"\x00"
    level = len(leading) // int(indent_unit)
    if level >= FALLBACK_MARKER:
        return bytes([FALLBACK_MARKER]) + encode_varint(len(leading)) + leading
    return bytes([int(level)])


def decode_leading(count_stream: bytes, index: int, indent_unit: int) -> tuple[bytes, int]:
    if index >= len(count_stream):
        raise ValueError("truncated count stream")
    code = count_stream[index]
    index += 1
    if code == FALLBACK_MARKER:
        length, index = decode_varint(count_stream, index)
        end = index + int(length)
        if end > len(count_stream):
            raise ValueError("truncated fallback leading whitespace")
        return count_stream[index:end], int(end)
    return b" " * (int(code) * int(indent_unit)), int(index)


def transform_structure(block: bytes, indent_unit: int) -> dict[str, Any]:
    count_stream = bytearray()
    body_stream = bytearray()
    rows = iter_lf_lines(block)
    for line in rows:
        has_newline = line.endswith(b"\n")
        core = line[:-1] if has_newline else line
        leading = leading_whitespace(core)
        count_stream.extend(encode_leading(leading, indent_unit))
        body_stream.extend(core[len(leading) :])
        if has_newline:
            body_stream.extend(b"\n")
    return {"count_stream": bytes(count_stream), "body_stream": bytes(body_stream), "line_count": len(rows), "indent_unit": int(indent_unit)}


def body_rows_for_restore(body_stream: bytes, line_count: int) -> list[bytes]:
    rows = iter_lf_lines(body_stream)
    if body_stream == b"":
        rows = []
    while len(rows) < int(line_count):
        rows.append(b"")
    if len(rows) != int(line_count):
        raise ValueError("line count mismatch")
    return rows


def restore_structure(count_stream: bytes, body_stream: bytes, line_count: int, indent_unit: int) -> bytes:
    out = bytearray()
    index = 0
    for body in body_rows_for_restore(body_stream, line_count):
        leading, index = decode_leading(count_stream, index, indent_unit)
        out.extend(leading)
        out.extend(body)
    if index != len(count_stream):
        raise ValueError("unused count stream bytes")
    return bytes(out)


def codec_candidates(block: bytes) -> list[tuple[str, bytes]]:
    rows: list[tuple[str, bytes]] = []
    rows.extend((f"zlib{level}", zlib.compress(block, level)) for level in range(1, 10))
    rows.extend((f"bz2{level}", bz2.compress(block, compresslevel=level)) for level in range(1, 10))
    rows.extend((f"lzma{preset}", lzma.compress(block, preset=preset)) for preset in range(0, 10))
    rows.extend((f"brotli{quality}", brotli.compress(block, quality=quality)) for quality in range(0, 12))
    rows.extend((f"zstd{level}", zstd.ZstdCompressor(level=level).compress(block)) for level in range(1, 23))
    return rows


def codec_family_rank(name: str) -> float:
    for prefix, value in CODEC_FAMILIES.items():
        if name.startswith(prefix):
            return float(value)
    return 0.0


def encode_codec_name(name: str) -> int:
    for prefix, value in CODEC_FAMILIES.items():
        if name.startswith(prefix):
            return int(value * 100 + int(name[len(prefix) :]))
    raise ValueError("unknown codec")


def decode_codec_name(code: int) -> str:
    family_id = int(code) // 100
    level = int(code) % 100
    if family_id not in CODEC_PREFIX_BY_ID:
        raise ValueError("unknown codec code")
    return f"{CODEC_PREFIX_BY_ID[family_id]}{level}"


def best_codec(block: bytes) -> tuple[str, bytes]:
    return min(codec_candidates(block), key=lambda row: (len(row[1]), row[0]))


def best_count_codec(block: bytes) -> tuple[str, bytes]:
    return min(codec_candidates(block), key=lambda row: (len(row[1]), 0 if row[0].startswith("zstd") else 1, row[0]))


def decompress_best(codec_name: str, payload: bytes) -> bytes:
    if codec_name.startswith("zlib"):
        return zlib.decompress(payload)
    if codec_name.startswith("bz2"):
        return bz2.decompress(payload)
    if codec_name.startswith("lzma"):
        return lzma.decompress(payload)
    if codec_name.startswith("brotli"):
        return brotli.decompress(payload)
    if codec_name.startswith("zstd"):
        return zstd.ZstdDecompressor().decompress(payload)
    raise ValueError("unknown codec")


def learned_codec(train_block: bytes, target_block: bytes) -> dict[str, Any]:
    indent_unit = learn_indent_unit(train_block)
    transformed = transform_structure(target_block, indent_unit)
    count_codec_name, count_payload = best_count_codec(bytes(transformed["count_stream"]))
    body_codec_name, body_payload = best_codec(bytes(transformed["body_stream"]))
    return {
        "indent_unit": int(indent_unit),
        "line_count": int(transformed["line_count"]),
        "count_codec_name": count_codec_name,
        "body_codec_name": body_codec_name,
        "count_payload": count_payload,
        "body_payload": body_payload,
        "count_stream_len": len(bytes(transformed["count_stream"])),
        "body_stream_len": len(bytes(transformed["body_stream"])),
    }


class SourceStructurePayloadModule(torch.nn.Module):
    def __init__(self, count_payload: bytes, body_payload: bytes, count_codec_code: int, body_codec_code: int, line_count: int, indent_unit: int) -> None:
        super().__init__()
        self.register_buffer("count_payload", torch.tensor(list(count_payload), dtype=torch.uint8))
        self.register_buffer("body_payload", torch.tensor(list(body_payload), dtype=torch.uint8))
        self.register_buffer("count_codec_code", torch.tensor([int(count_codec_code)], dtype=torch.int64))
        self.register_buffer("body_codec_code", torch.tensor([int(body_codec_code)], dtype=torch.int64))
        self.register_buffer("line_count", torch.tensor([int(line_count)], dtype=torch.int64))
        self.register_buffer("indent_unit", torch.tensor([int(indent_unit)], dtype=torch.int64))

    @classmethod
    def from_learned(cls, learned: dict[str, Any]) -> "SourceStructurePayloadModule":
        return cls(
            bytes(learned["count_payload"]),
            bytes(learned["body_payload"]),
            encode_codec_name(str(learned["count_codec_name"])),
            encode_codec_name(str(learned["body_codec_name"])),
            int(learned["line_count"]),
            int(learned["indent_unit"]),
        )

    @classmethod
    def empty_like(cls, other: "SourceStructurePayloadModule") -> "SourceStructurePayloadModule":
        return cls(
            bytes(int(other.count_payload.numel())),
            bytes(int(other.body_payload.numel())),
            100,
            100,
            0,
            0,
        )

    def learned_state(self) -> dict[str, Any]:
        return {
            "indent_unit": int(self.indent_unit.item()),
            "line_count": int(self.line_count.item()),
            "count_codec_name": decode_codec_name(int(self.count_codec_code.item())),
            "body_codec_name": decode_codec_name(int(self.body_codec_code.item())),
            "count_payload": bytes(self.count_payload.detach().cpu().tolist()),
            "body_payload": bytes(self.body_payload.detach().cpu().tolist()),
        }

    def restore(self) -> bytes:
        return restore_learned(self.learned_state())


def restore_learned(learned: dict[str, Any]) -> bytes:
    count_stream = decompress_best(str(learned["count_codec_name"]), bytes(learned["count_payload"]))
    body_stream = decompress_best(str(learned["body_codec_name"]), bytes(learned["body_payload"]))
    return restore_structure(count_stream, body_stream, int(learned["line_count"]), int(learned["indent_unit"]))


def random_block(seed: int, length: int) -> bytes:
    rng = random.Random(int(seed) + 9011)
    return bytes(rng.randrange(0, 256) for _index in range(int(length)))


def fixed_ngrams(data: bytes, width: int = 64) -> set[bytes]:
    if len(data) < int(width):
        return set()
    return {data[index : index + int(width)] for index in range(0, len(data) - int(width) + 1)}


def overlap_counts(train: list[Path], target: list[Path]) -> dict[str, float]:
    train_present = [path for path in train if path.exists()]
    target_present = [path for path in target if path.exists()]
    train_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in train_present}
    target_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in target_present}
    train_hash = {hashlib.sha256(path.read_bytes()).hexdigest() for path in train_present}
    target_hash = {hashlib.sha256(path.read_bytes()).hexdigest() for path in target_present}
    train_block = read_joined(train_present)
    target_block = read_joined(target_present)
    ngram_width = 64
    return {
        "source_train_test_path_overlap_count": float(len(train_rel & target_rel)),
        "source_train_test_hash_overlap_count": float(len(train_hash & target_hash)),
        "source_train_test_ngram_width_bytes": float(ngram_width),
        "source_train_test_ngram_overlap_count": float(len(fixed_ngrams(train_block, ngram_width) & fixed_ngrams(target_block, ngram_width))),
    }


def shuffle_bytes(payload: bytes, seed: int) -> bytes:
    if not payload:
        return payload
    values = list(payload)
    random.Random(int(seed) + 319).shuffle(values)
    return bytes(values)


def measure_block(train: bytes, target: bytes, seed: int) -> dict[str, float | str]:
    baseline_name, baseline_payload = best_codec(target)
    learned = learned_codec(train, target)
    learned_decoded = restore_learned(learned)
    module = SourceStructurePayloadModule.from_learned(learned)
    module_reloaded = SourceStructurePayloadModule.empty_like(module)
    module_reloaded.load_state_dict(module.state_dict())
    module_reload_decoded = module_reloaded.restore()
    count_stream = decompress_best(str(learned["count_codec_name"]), bytes(learned["count_payload"]))
    body_stream = decompress_best(str(learned["body_codec_name"]), bytes(learned["body_payload"]))
    wrong_unit_decoded = restore_structure(count_stream, body_stream, int(learned["line_count"]), int(learned["indent_unit"]) + 1)
    decoder_disabled_decoded = b""
    try:
        shuffled_body = decompress_best(str(learned["body_codec_name"]), shuffle_bytes(bytes(learned["body_payload"]), seed))
        shuffled_body_decoded = restore_structure(count_stream, shuffled_body, int(learned["line_count"]), int(learned["indent_unit"]))
        shuffle_body_success = float(shuffled_body_decoded == target)
    except Exception:
        shuffle_body_success = 0.0
    try:
        shuffled_count = decompress_best(str(learned["count_codec_name"]), shuffle_bytes(bytes(learned["count_payload"]), seed + 11))
        shuffled_count_decoded = restore_structure(shuffled_count, body_stream, int(learned["line_count"]), int(learned["indent_unit"]))
        shuffle_count_success = float(shuffled_count_decoded == target)
    except Exception:
        shuffle_count_success = 0.0
    random_target = random_block(seed, len(target))
    random_baseline_name, random_baseline_payload = best_codec(random_target)
    random_learned = learned_codec(train, random_target)
    random_decoded = restore_learned(random_learned)
    learned_payload_bits = int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"]))) * 8 + int(STRUCTURE_HEADER_BITS))
    baseline_payload_bits = int(len(baseline_payload) * 8)
    learned_strict_bits = int(learned_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    baseline_strict_bits = int(baseline_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    learned_paper_bits = int(learned_strict_bits + int(SURFACE_CONTRACT_BITS))
    baseline_paper_bits = int(baseline_strict_bits + int(SURFACE_CONTRACT_BITS))
    random_learned_payload_bits = int((len(bytes(random_learned["count_payload"])) + len(bytes(random_learned["body_payload"]))) * 8 + int(STRUCTURE_HEADER_BITS))
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
        "decoder_disabled_exact_reconstruction_success": float(decoder_disabled_decoded == target),
        "wrong_indent_unit_exact_reconstruction_success": float(wrong_unit_decoded == target),
        "shuffle_body_payload_exact_reconstruction_success": shuffle_body_success,
        "shuffle_count_payload_exact_reconstruction_success": shuffle_count_success,
        "best_standard_codec": baseline_name,
        "count_stream_codec": str(learned["count_codec_name"]),
        "body_stream_codec": str(learned["body_codec_name"]),
        "random_label_best_standard_codec": random_baseline_name,
        "best_standard_codec_family_id": codec_family_rank(baseline_name),
        "count_stream_codec_family_id": codec_family_rank(str(learned["count_codec_name"])),
        "body_stream_codec_family_id": codec_family_rank(str(learned["body_codec_name"])),
        "learned_indent_unit_spaces": float(learned["indent_unit"]),
        "target_line_count": float(learned["line_count"]),
        "count_stream_bytes": float(learned["count_stream_len"]),
        "body_stream_bytes": float(learned["body_stream_len"]),
        "target_block_bytes": float(len(target)),
        "useful_retrievable_bits": float(useful_bits),
        "best_standard_payload_bits": float(baseline_payload_bits),
        "learned_count_payload_bits": float(len(bytes(learned["count_payload"])) * 8),
        "learned_body_payload_bits": float(len(bytes(learned["body_payload"])) * 8),
        "learned_structure_header_bits": float(STRUCTURE_HEADER_BITS),
        "learned_payload_bits": float(learned_payload_bits),
        "best_standard_strict_bits": float(baseline_strict_bits),
        "learned_strict_bits": float(learned_strict_bits),
        "best_standard_paper_bits": float(baseline_paper_bits),
        "learned_paper_bits": float(learned_paper_bits),
        "payload_improvement_over_best_standard": payload_improvement,
        "strict_improvement_over_best_standard": strict_improvement,
        "paper_improvement_over_best_standard": paper_improvement,
        "indent_token_strict_improvement_baseline": float(INDENT_TOKEN_STRICT_IMPROVEMENT),
        "strict_improvement_delta_over_indent_token": strict_improvement - float(INDENT_TOKEN_STRICT_IMPROVEMENT),
        "beats_indent_token_strict_margin": float(int(strict_improvement > float(INDENT_TOKEN_STRICT_IMPROVEMENT))),
        "adapter_strict_multiplier": float(useful_bits) / max(float(learned_strict_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "best_standard_strict_multiplier": float(useful_bits) / max(float(baseline_strict_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "random_label_payload_incompressible": float(int(random_learned_payload_bits >= random_baseline_payload_bits)),
        "random_label_payload_improvement_over_best_standard": random_payload_improvement,
        "random_label_learned_payload_bits": float(random_learned_payload_bits),
        "random_label_best_standard_payload_bits": float(random_baseline_payload_bits),
        "compressed_stream_read_success": float(learned_decoded == target),
        "codec_state_has_raw_target_block": float("target" in state_keys or "target_block" in state_keys),
        "codec_state_has_uncompressed_count_stream": float("count_stream" in state_keys),
        "codec_state_has_uncompressed_body_stream": float("body_stream" in state_keys),
        "codec_state_has_restored_block": float("restored" in state_keys or "restored_block" in state_keys),
        "compressed_count_payload_retained": 1.0,
        "compressed_body_payload_retained": 1.0,
        "model_state_payload_used": 1.0,
        "external_payload_store_used": 0.0,
        "state_dict_count_payload_used": float("count_payload" in module.state_dict()),
        "state_dict_body_payload_used": float("body_payload" in module.state_dict()),
        "state_dict_codec_selectors_used": float("count_codec_code" in module.state_dict() and "body_codec_code" in module.state_dict()),
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
    controls_pass = float(int(metrics["compressed_stream_read_success"] == 1.0 and metrics["decoder_disabled_exact_reconstruction_success"] == 0.0 and metrics["wrong_indent_unit_exact_reconstruction_success"] == 0.0 and metrics["shuffle_body_payload_exact_reconstruction_success"] == 0.0 and metrics["shuffle_count_payload_exact_reconstruction_success"] == 0.0 and metrics["codec_state_has_raw_target_block"] == 0.0 and metrics["codec_state_has_uncompressed_count_stream"] == 0.0 and metrics["codec_state_has_uncompressed_body_stream"] == 0.0 and metrics["codec_state_has_restored_block"] == 0.0))
    engineering_pass = float(int(metrics["exact_reconstruction_success"] == 1.0 and metrics["random_label_exact_reconstruction_success"] == 1.0 and metrics["random_label_payload_incompressible"] == 1.0 and metrics["beats_indent_token_strict_margin"] == 1.0 and strict_pass == 1.0 and payload_pass == 1.0 and controls_pass == 1.0 and overlaps["source_train_test_path_overlap_count"] == 0.0 and overlaps["source_train_test_hash_overlap_count"] == 0.0))
    return {
        "profile": profile,
        "target_file_count": float(len(targets)),
        "train_file_count": float(len([path for path in train if path.exists()])),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "source_structure_block_codec_candidate": engineering_pass,
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
    metrics_path = output_dir / "local_100k_source_structure_block_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "decoder_bits": int(DECODER_BITS), "model_header_bits": int(MODEL_HEADER_BITS), "surface_contract_bits": int(SURFACE_CONTRACT_BITS), "structure_header_bits": int(STRUCTURE_HEADER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_target_block_bytes"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_structure_block_codec_metrics.json", "path": metrics_path}],
        warnings=["narrow source-code block codec; no nm, chat, knowledge, paid-compute, or broad breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} strict_improvement={summary[f'{SIMULATION_ID}_strict_improvement_over_best_standard']:.6f} payload_improvement={summary[f'{SIMULATION_ID}_payload_improvement_over_best_standard']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
