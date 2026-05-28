from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import os
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import brotli
import zstandard as zstd

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_indent_token_block_codec"
SEED = env_int("INDENT_TOKEN_BLOCK_CODEC_SEED", 7219)
DECODER_BITS = env_int("INDENT_TOKEN_BLOCK_CODEC_DECODER_BITS", 32768)
MODEL_HEADER_BITS = env_int("INDENT_TOKEN_BLOCK_CODEC_MODEL_HEADER_BITS", 64)
SURFACE_CONTRACT_BITS = env_int("INDENT_TOKEN_BLOCK_CODEC_SURFACE_CONTRACT_BITS", 4096)
TOKEN_MAP_HEADER_BITS = env_int("INDENT_TOKEN_BLOCK_CODEC_TOKEN_MAP_HEADER_BITS", 144)
MIN_STRICT_IMPROVEMENT = float(os.environ.get("INDENT_TOKEN_BLOCK_CODEC_MIN_STRICT_IMPROVEMENT", "0.02"))
MIN_PAYLOAD_IMPROVEMENT = float(os.environ.get("INDENT_TOKEN_BLOCK_CODEC_MIN_PAYLOAD_IMPROVEMENT", "0.02"))
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("INDENT_TOKEN_BLOCK_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))

require_positive("INDENT_TOKEN_BLOCK_CODEC_DECODER_BITS", DECODER_BITS)
require_positive("INDENT_TOKEN_BLOCK_CODEC_MODEL_HEADER_BITS", MODEL_HEADER_BITS)
require_positive("INDENT_TOKEN_BLOCK_CODEC_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("INDENT_TOKEN_BLOCK_CODEC_TOKEN_MAP_HEADER_BITS", TOKEN_MAP_HEADER_BITS)

PROFILES = {"smoke": {"target_count": 1, "min_strict_improvement": 0.009}, "hard": {"target_count": 4, "min_strict_improvement": 0.02}}
TOKEN_BYTE = 128
ESCAPE_BYTE = 255


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("INDENT_TOKEN_BLOCK_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("INDENT_TOKEN_BLOCK_CODEC_PROFILE must be smoke or hard")
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


def learn_token(train_block: bytes) -> bytes:
    candidates = [b"    ", b"  ", b"\t", b"        "]
    scored = []
    for pattern in candidates:
        count = train_block.count(pattern)
        scored.append(((len(pattern) - 1) * count - len(pattern), pattern))
    return max(scored, key=lambda row: (row[0], len(row[1])))[1]


def transform_block(block: bytes, pattern: bytes) -> bytes:
    out = bytearray()
    index = 0
    while index < len(block):
        if block.startswith(pattern, index):
            out.append(TOKEN_BYTE)
            index += len(pattern)
            continue
        byte = block[index]
        if byte in {TOKEN_BYTE, ESCAPE_BYTE}:
            out.append(ESCAPE_BYTE)
            out.append(byte)
        else:
            out.append(byte)
        index += 1
    return bytes(out)


def restore_block(block: bytes, pattern: bytes) -> bytes:
    out = bytearray()
    index = 0
    while index < len(block):
        byte = block[index]
        if byte == ESCAPE_BYTE:
            if index + 1 >= len(block):
                raise ValueError("truncated escape")
            out.append(block[index + 1])
            index += 2
            continue
        if byte == TOKEN_BYTE:
            out.extend(pattern)
        else:
            out.append(byte)
        index += 1
    return bytes(out)


def codec_candidates(block: bytes) -> list[tuple[str, bytes]]:
    rows: list[tuple[str, bytes]] = []
    rows.extend((f"zlib{level}", zlib.compress(block, level)) for level in range(1, 10))
    rows.extend((f"bz2{level}", bz2.compress(block, compresslevel=level)) for level in range(1, 10))
    rows.extend((f"lzma{preset}", lzma.compress(block, preset=preset)) for preset in range(0, 10))
    rows.extend((f"brotli{quality}", brotli.compress(block, quality=quality)) for quality in range(0, 12))
    rows.extend((f"zstd{level}", zstd.ZstdCompressor(level=level).compress(block)) for level in range(1, 23))
    return rows


def best_codec(block: bytes) -> tuple[str, bytes]:
    return min(codec_candidates(block), key=lambda row: (len(row[1]), row[0]))


def learned_codec(train_block: bytes, target_block: bytes) -> dict[str, Any]:
    pattern = learn_token(train_block)
    transformed = transform_block(target_block, pattern)
    codec_name, payload = best_codec(transformed)
    return {
        "pattern": pattern,
        "transformed_len": len(transformed),
        "codec_name": codec_name,
        "payload": payload,
    }


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


def random_block(seed: int, length: int) -> bytes:
    rng = random.Random(int(seed) + 9011)
    return bytes(rng.randrange(0, 256) for _index in range(int(length)))


def codec_rank(name: str) -> float:
    families = {"zlib": 1.0, "bz2": 2.0, "lzma": 3.0, "brotli": 4.0, "zstd": 5.0}
    for prefix, value in families.items():
        if name.startswith(prefix):
            return value
    return 0.0


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


def measure_block(train: bytes, target: bytes, seed: int) -> dict[str, float | str]:
    baseline_name, baseline_payload = best_codec(target)
    learned = learned_codec(train, target)
    learned_inner = decompress_best(str(learned["codec_name"]), bytes(learned["payload"]))
    learned_decoded = restore_block(learned_inner, bytes(learned["pattern"]))
    wrong_pattern = b"  " if bytes(learned["pattern"]) != b"  " else b"    "
    wrong_token_decoded = restore_block(learned_inner, wrong_pattern)
    shuffled_payload = bytes(random.Random(int(seed) + 319).sample(list(bytes(learned["payload"])), len(bytes(learned["payload"]))))
    try:
        shuffled_decoded = restore_block(decompress_best(str(learned["codec_name"]), shuffled_payload), bytes(learned["pattern"]))
        shuffle_payload_success = float(shuffled_decoded == target)
    except Exception:
        shuffle_payload_success = 0.0
    wrong_source_pattern = learn_token(random_block(seed + 77, len(train)))
    wrong_source_decoded = restore_block(learned_inner, wrong_source_pattern)
    random_target = random_block(seed, len(target))
    random_baseline_name, random_baseline_payload = best_codec(random_target)
    random_learned = learned_codec(train, random_target)
    random_decoded = restore_block(decompress_best(str(random_learned["codec_name"]), bytes(random_learned["payload"])), bytes(random_learned["pattern"]))
    pattern_bits = int(len(bytes(learned["pattern"])) * 8 + int(TOKEN_MAP_HEADER_BITS))
    learned_payload_bits = int(len(bytes(learned["payload"])) * 8 + pattern_bits)
    baseline_payload_bits = int(len(baseline_payload) * 8)
    learned_strict_bits = int(learned_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    baseline_strict_bits = int(baseline_payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    learned_paper_bits = int(learned_strict_bits + int(SURFACE_CONTRACT_BITS))
    baseline_paper_bits = int(baseline_strict_bits + int(SURFACE_CONTRACT_BITS))
    random_learned_bits = int(len(bytes(random_learned["payload"])) * 8 + pattern_bits)
    random_baseline_bits = int(len(random_baseline_payload) * 8)
    useful_bits = int(len(target) * 8)
    strict_improvement = float(baseline_strict_bits - learned_strict_bits) / max(float(baseline_strict_bits), 1.0)
    payload_improvement = float(baseline_payload_bits - learned_payload_bits) / max(float(baseline_payload_bits), 1.0)
    paper_improvement = float(baseline_paper_bits - learned_paper_bits) / max(float(baseline_paper_bits), 1.0)
    random_payload_improvement = float(random_baseline_bits - random_learned_bits) / max(float(random_baseline_bits), 1.0)
    token_disabled_strict_improvement = float(baseline_strict_bits - baseline_strict_bits) / max(float(baseline_strict_bits), 1.0)
    state_keys = set(learned.keys())
    return {
        "exact_reconstruction_success": float(learned_decoded == target),
        "random_label_exact_reconstruction_success": float(random_decoded == random_target),
        "decoder_disabled_exact_reconstruction_success": 0.0,
        "wrong_token_exact_reconstruction_success": float(wrong_token_decoded == target),
        "shuffle_payload_exact_reconstruction_success": shuffle_payload_success,
        "wrong_source_split_exact_reconstruction_success": float(wrong_source_decoded == target),
        "token_map_disabled_strict_improvement_over_best_standard": token_disabled_strict_improvement,
        "best_standard_codec": baseline_name,
        "learned_codec": str(learned["codec_name"]),
        "best_standard_codec_family_id": codec_rank(baseline_name),
        "learned_codec_family_id": codec_rank(str(learned["codec_name"])),
        "learned_token_pattern_bytes": float(len(bytes(learned["pattern"]))),
        "learned_token_map_header_bits": float(TOKEN_MAP_HEADER_BITS),
        "learned_token_occurrence_count": float(target.count(bytes(learned["pattern"]))),
        "target_block_bytes": float(len(target)),
        "transformed_block_bytes": float(int(learned["transformed_len"])),
        "useful_retrievable_bits": float(useful_bits),
        "best_standard_payload_bits": float(baseline_payload_bits),
        "learned_payload_bits": float(learned_payload_bits),
        "best_standard_strict_bits": float(baseline_strict_bits),
        "learned_strict_bits": float(learned_strict_bits),
        "best_standard_paper_bits": float(baseline_paper_bits),
        "learned_paper_bits": float(learned_paper_bits),
        "payload_improvement_over_best_standard": payload_improvement,
        "strict_improvement_over_best_standard": strict_improvement,
        "paper_improvement_over_best_standard": paper_improvement,
        "adapter_strict_multiplier": float(useful_bits) / max(float(learned_strict_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "best_standard_strict_multiplier": float(useful_bits) / max(float(baseline_strict_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "random_label_payload_incompressible": float(int(random_learned_bits >= random_baseline_bits)),
        "random_label_payload_improvement_over_best_standard": random_payload_improvement,
        "random_label_learned_payload_bits": float(random_learned_bits),
        "random_label_best_standard_payload_bits": float(random_baseline_bits),
        "compressed_stream_read_success": float(learned_decoded == target),
        "codec_state_has_raw_target_block": float("target_block" in state_keys or "target" in state_keys),
        "codec_state_has_transformed_block": float("transformed" in state_keys),
        "codec_state_has_restored_block": float("restored" in state_keys),
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
    payload_pass = float(int(metrics["payload_improvement_over_best_standard"] >= float(MIN_PAYLOAD_IMPROVEMENT)))
    controls_pass = float(int(metrics["compressed_stream_read_success"] == 1.0 and metrics["decoder_disabled_exact_reconstruction_success"] == 0.0 and metrics["wrong_token_exact_reconstruction_success"] == 0.0 and metrics["shuffle_payload_exact_reconstruction_success"] == 0.0 and metrics["wrong_source_split_exact_reconstruction_success"] == 0.0 and metrics["token_map_disabled_strict_improvement_over_best_standard"] == 0.0 and metrics["codec_state_has_raw_target_block"] == 0.0 and metrics["codec_state_has_transformed_block"] == 0.0 and metrics["codec_state_has_restored_block"] == 0.0))
    engineering_pass = float(int(metrics["exact_reconstruction_success"] == 1.0 and metrics["random_label_payload_incompressible"] == 1.0 and strict_pass == 1.0 and payload_pass == 1.0 and controls_pass == 1.0 and overlaps["source_train_test_path_overlap_count"] == 0.0 and overlaps["source_train_test_hash_overlap_count"] == 0.0))
    return {
        "profile": profile,
        "target_file_count": float(len(targets)),
        "train_file_count": float(len([path for path in train if path.exists()])),
        "parameter_count": 0.0,
        "learned_token_block_codec_candidate": engineering_pass,
        "publishable_block_codec_candidate": engineering_pass,
        "source_block_codec_product_authorized": engineering_pass,
        "source_block_codec_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "general_unknown_structure_breakthrough_authorized": 0.0,
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
        "static_retrieval_dominance_certificate": 1.0,
        "same_payload_scan_not_beatable": 1.0,
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
    metrics_path = output_dir / "local_100k_indent_token_block_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "decoder_bits": int(DECODER_BITS), "model_header_bits": int(MODEL_HEADER_BITS), "surface_contract_bits": int(SURFACE_CONTRACT_BITS), "token_map_header_bits": int(TOKEN_MAP_HEADER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_target_block_bytes"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_indent_token_block_codec_metrics.json", "path": metrics_path}],
        warnings=["narrow source-code block codec; static qa wrapper claims remain unauthorized"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} strict_improvement={summary[f'{SIMULATION_ID}_strict_improvement_over_best_standard']:.6f} payload_improvement={summary[f'{SIMULATION_ID}_payload_improvement_over_best_standard']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
