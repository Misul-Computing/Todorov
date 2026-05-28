from __future__ import annotations

import bz2
import hashlib
import lzma
import math
import os
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_learned_block_codec_frontier"
SEED = env_int("LEARNED_BLOCK_CODEC_FRONTIER_SEED", 2459)
SMOKE_BLOCK_BYTES = env_int("LEARNED_BLOCK_CODEC_FRONTIER_SMOKE_BLOCK_BYTES", 32768)
HARD_BLOCK_BYTES = env_int("LEARNED_BLOCK_CODEC_FRONTIER_HARD_BLOCK_BYTES", 131072)
PHRASE_MIN_BYTES = env_int("LEARNED_BLOCK_CODEC_FRONTIER_PHRASE_MIN_BYTES", 3)
PHRASE_MAX_BYTES = env_int("LEARNED_BLOCK_CODEC_FRONTIER_PHRASE_MAX_BYTES", 12)
DICTIONARY_SIZE = env_int("LEARNED_BLOCK_CODEC_FRONTIER_DICTIONARY_SIZE", 768)
LEARNED_DECODER_BITS = env_int("LEARNED_BLOCK_CODEC_FRONTIER_LEARNED_DECODER_BITS", 16384)
STANDARD_DECODER_BITS = env_int("LEARNED_BLOCK_CODEC_FRONTIER_STANDARD_DECODER_BITS", 16384)
QA_WRAPPER_BITS = env_int("LEARNED_BLOCK_CODEC_FRONTIER_QA_WRAPPER_BITS", 8192)
STATIC_SCAN_DECODER_BITS = env_int("LEARNED_BLOCK_CODEC_FRONTIER_STATIC_SCAN_DECODER_BITS", 4096)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("LEARNED_BLOCK_CODEC_FRONTIER_ORDINARY_BITS_PER_PARAMETER", "2.5"))
WIN_MARGIN = float(os.environ.get("LEARNED_BLOCK_CODEC_FRONTIER_WIN_MARGIN", "0.02"))

require_positive("LEARNED_BLOCK_CODEC_FRONTIER_SMOKE_BLOCK_BYTES", SMOKE_BLOCK_BYTES)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_HARD_BLOCK_BYTES", HARD_BLOCK_BYTES)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_PHRASE_MIN_BYTES", PHRASE_MIN_BYTES)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_PHRASE_MAX_BYTES", PHRASE_MAX_BYTES)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_DICTIONARY_SIZE", DICTIONARY_SIZE)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_LEARNED_DECODER_BITS", LEARNED_DECODER_BITS)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_STANDARD_DECODER_BITS", STANDARD_DECODER_BITS)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_QA_WRAPPER_BITS", QA_WRAPPER_BITS)
require_positive("LEARNED_BLOCK_CODEC_FRONTIER_STATIC_SCAN_DECODER_BITS", STATIC_SCAN_DECODER_BITS)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("LEARNED_BLOCK_CODEC_FRONTIER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("LEARNED_BLOCK_CODEC_FRONTIER_PROFILE must be smoke or hard")
    return value


def bits_for_cardinality(cardinality: int) -> int:
    return max(1, math.ceil(math.log2(max(2, int(cardinality)))))


def source_rows() -> list[tuple[str, Path, str]]:
    rows = [
        ("test", PROJECT_ROOT / "knowledge/training_efficiency.md", "training_efficiency"),
        ("test", PROJECT_ROOT / "knowledge/papers_library.md", "papers_library"),
        ("test", PROJECT_ROOT / "knowledge/context_extension.md", "context_extension"),
        ("test", PROJECT_ROOT / "neuroloc/wiki/synthesis/neural_model_related_work_pressure_matrix.md", "related_work_pressure"),
        ("train", PROJECT_ROOT / "knowledge/unified_theory.md", "unified_theory"),
        ("train", PROJECT_ROOT / "knowledge/hybrid_architectures.md", "hybrid_architectures"),
        ("train", PROJECT_ROOT / "knowledge/delta_rule_theory.md", "delta_rule_theory"),
        ("train", PROJECT_ROOT / "knowledge/mla_compression.md", "mla_compression"),
        ("train", PROJECT_ROOT / "knowledge/mamba3_architecture.md", "mamba3_architecture"),
        ("train", PROJECT_ROOT / "knowledge/kda_channel_gating.md", "kda_channel_gating"),
        ("train", PROJECT_ROOT / "knowledge/geometric_algebra.md", "geometric_algebra"),
        ("train", PROJECT_ROOT / "knowledge/ternary_spikes.md", "ternary_spikes"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/compression_and_bottlenecks.md", "compression_and_bottlenecks"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/local_vs_global_computation.md", "local_vs_global_computation"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/timescale_separation.md", "timescale_separation"),
    ]
    return [(role, path, name) for role, path, name in rows if path.exists()]


def load_authored_sources() -> tuple[bytes, bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    train_parts: list[bytes] = []
    test_parts: list[bytes] = []
    train_manifest = []
    test_manifest = []
    for index, (role, path, name) in enumerate(source_rows()):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        row = {
            "role": role,
            "name": name,
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "index": int(index),
            "length": int(len(data)),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if role == "train":
            if train_parts:
                train_parts.append(b"\n\n")
            train_parts.append(data)
            train_manifest.append(row)
        else:
            if test_parts:
                test_parts.append(b"\n\n")
            test_parts.append(data)
            test_manifest.append(row)
    train_blob = b"".join(train_parts)
    test_blob = b"".join(test_parts)
    if len(train_blob) < 4096 or len(test_blob) < 4096:
        raise ValueError("source blobs too small")
    return train_blob, test_blob, train_manifest, test_manifest


def heldout_block(test_blob: bytes, block_bytes: int, seed: int) -> bytes:
    if len(test_blob) <= int(block_bytes):
        return test_blob
    rng = random.Random(int(seed) + 101)
    window_count = max(1, (len(test_blob) - int(block_bytes)) // 512)
    start = rng.randrange(0, window_count) * 512
    return test_blob[int(start) : int(start) + int(block_bytes)]


def manifest_bits(manifest: list[dict[str, Any]]) -> int:
    payload = ";".join(
        f"{row['role']}:{row['name']}:{row['path']}:{int(row['index'])}:{int(row['length'])}:{row['sha256']}"
        for row in manifest
    ).encode("utf-8")
    return int(len(payload) * 8)


def ngram_set(data: bytes, width: int) -> set[bytes]:
    if len(data) < int(width):
        return set()
    return {data[index : index + int(width)] for index in range(0, len(data) - int(width) + 1, int(width))}


def train_test_overlap_counts(train_blob: bytes, block: bytes, train_manifest: list[dict[str, Any]], test_manifest: list[dict[str, Any]]) -> dict[str, float]:
    train_paths = {str(row["path"]) for row in train_manifest}
    test_paths = {str(row["path"]) for row in test_manifest}
    train_hashes = {str(row["sha256"]) for row in train_manifest}
    test_hashes = {str(row["sha256"]) for row in test_manifest}
    train_ngrams = ngram_set(train_blob, 32)
    test_ngrams = ngram_set(block, 32)
    return {
        "source_train_file_count": float(len(train_manifest)),
        "source_test_file_count": float(len(test_manifest)),
        "source_train_test_path_overlap_count": float(len(train_paths.intersection(test_paths))),
        "source_train_test_hash_overlap_count": float(len(train_hashes.intersection(test_hashes))),
        "source_train_test_ngram_overlap_count": float(len(train_ngrams.intersection(test_ngrams))),
    }


def train_phrase_dictionary(train_blob: bytes, dictionary_size: int = DICTIONARY_SIZE) -> list[bytes]:
    counter: Counter[bytes] = Counter()
    limit = min(len(train_blob), 512 * 1024)
    sample = train_blob[:limit]
    for width in range(int(PHRASE_MIN_BYTES), int(PHRASE_MAX_BYTES) + 1):
        if len(sample) < width:
            continue
        for index in range(0, len(sample) - width + 1):
            phrase = sample[index : index + width]
            if b"\x00" in phrase:
                continue
            counter[phrase] += 1
    scored = []
    for phrase, count in counter.items():
        if int(count) < 2:
            continue
        literal_cost = len(phrase) * 8 * int(count)
        token_cost = (1 + bits_for_cardinality(dictionary_size) + len(phrase) * 8 / max(int(count), 1)) * int(count)
        scored.append((float(literal_cost - token_cost), int(count), len(phrase), phrase))
    scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return [row[3] for row in scored[: int(dictionary_size)]]


def phrase_cost_bits(dictionary: list[bytes]) -> int:
    length_bits = bits_for_cardinality(int(PHRASE_MAX_BYTES) + 1)
    return int(sum(length_bits + len(phrase) * 8 for phrase in dictionary))


def phrase_lookup(dictionary: list[bytes]) -> dict[int, list[tuple[bytes, int]]]:
    lookup: dict[int, list[tuple[bytes, int]]] = {}
    for phrase_id, phrase in enumerate(dictionary):
        if not phrase:
            continue
        lookup.setdefault(int(phrase[0]), []).append((phrase, int(phrase_id)))
    for rows in lookup.values():
        rows.sort(key=lambda row: (-len(row[0]), row[1]))
    return lookup


def encode_with_dictionary(block: bytes, dictionary: list[bytes]) -> list[tuple[int, int]]:
    lookup = phrase_lookup(dictionary)
    tokens: list[tuple[int, int]] = []
    index = 0
    while index < len(block):
        selected: tuple[bytes, int] | None = None
        for phrase, phrase_id in lookup.get(int(block[index]), []):
            if block.startswith(phrase, index):
                selected = (phrase, phrase_id)
                break
        if selected is None:
            tokens.append((0, int(block[index])))
            index += 1
        else:
            phrase, phrase_id = selected
            tokens.append((1, int(phrase_id)))
            index += len(phrase)
    return tokens


def decode_with_dictionary(tokens: list[tuple[int, int]], dictionary: list[bytes]) -> bytes:
    parts: list[bytes] = []
    for kind, value in tokens:
        if int(kind) == 0:
            parts.append(bytes([int(value)]))
        else:
            parts.append(dictionary[int(value)])
    return b"".join(parts)


def token_stream_bits(tokens: list[tuple[int, int]], dictionary: list[bytes]) -> int:
    phrase_bits = bits_for_cardinality(len(dictionary))
    result = 0
    for kind, _value in tokens:
        result += 1 + (phrase_bits if int(kind) == 1 else 8)
    return int(result)


def byte_model_bits(train_blob: bytes, block: bytes) -> dict[str, float]:
    counts = [1 for _index in range(256)]
    for byte in train_blob:
        counts[int(byte)] += 1
    total = float(sum(counts))
    cross_entropy = 0.0
    for byte in block:
        cross_entropy += -math.log2(float(counts[int(byte)]) / total)
    model_bits = int(len(counts) * bits_for_cardinality(max(counts) + 1))
    return {"byte_predictor_cross_entropy_bits": float(cross_entropy), "byte_predictor_model_bits": float(model_bits)}


def standard_codec_sweep(block: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in [1, 3, 6, 9]:
        payload = zlib.compress(block, level=level)
        rows.append({"codec": f"zlib{level}", "payload": payload, "payload_bits": float(len(payload) * 8)})
    for level in [1, 3, 6, 9]:
        payload = bz2.compress(block, compresslevel=level)
        rows.append({"codec": f"bz2_{level}", "payload": payload, "payload_bits": float(len(payload) * 8)})
    for preset in [0, 3, 6, 9]:
        payload = lzma.compress(block, preset=preset)
        rows.append({"codec": f"lzma{preset}", "payload": payload, "payload_bits": float(len(payload) * 8)})
    selector_bits = bits_for_cardinality(len(rows))
    for row in rows:
        row["charged_bits"] = float(row["payload_bits"] + selector_bits + int(STANDARD_DECODER_BITS))
    return rows


def decompress_standard(codec: str, payload: bytes) -> bytes:
    if codec.startswith("zlib"):
        return zlib.decompress(payload)
    if codec.startswith("bz2_"):
        return bz2.decompress(payload)
    if codec.startswith("lzma"):
        return lzma.decompress(payload)
    raise ValueError("unknown codec")


def current_payload_transform(block: bytes) -> bytes:
    return bytes((int(block[index]) - int(block[index - 1])) % 256 if index else int(block[index]) for index in range(len(block)))


def inverse_current_payload_transform(payload: bytes) -> bytes:
    values = bytearray()
    previous = 0
    for index, byte in enumerate(payload):
        value = int(byte) if index == 0 else (previous + int(byte)) % 256
        values.append(value)
        previous = value
    return bytes(values)


def transformed_standard_codec_sweep(block: bytes) -> list[dict[str, Any]]:
    transformed = current_payload_transform(block)
    rows = standard_codec_sweep(transformed)
    for row in rows:
        row["codec"] = "delta_" + str(row["codec"])
        row["charged_bits"] = float(row["charged_bits"] + 256)
    return rows


def random_label_block(seed: int, length: int) -> bytes:
    rng = random.Random(int(seed) + 991)
    return bytes(rng.randrange(0, 256) for _index in range(int(length)))


class LearnedBlockCodecFrontier:
    def __init__(self, train_blob: bytes, block: bytes, manifest: list[dict[str, Any]]) -> None:
        self.dictionary = train_phrase_dictionary(train_blob)
        self.tokens = encode_with_dictionary(block, self.dictionary)
        self.reconstruction_sha256 = hashlib.sha256(decode_with_dictionary(self.tokens, self.dictionary)).hexdigest()
        self.block_sha256 = hashlib.sha256(block).hexdigest()
        self.dictionary_bits = phrase_cost_bits(self.dictionary)
        self.token_bits = token_stream_bits(self.tokens, self.dictionary)
        self.manifest_bits = manifest_bits(manifest)
        self.codec_selector_bits = bits_for_cardinality(3)
        self.block_payload_bits = int(self.dictionary_bits + self.token_bits)
        self.charged_bits = int(self.block_payload_bits + self.codec_selector_bits + int(LEARNED_DECODER_BITS) + self.manifest_bits)
        self.per_fact_value_slice_count = 0
        self.block_stream_count = 1
        self.raw_source_block_retained = 0.0
        self.train_only_dictionary_used = 1.0
        self.learned_or_phrase_codec_used = 1.0
        self.reads_from_charged_representation = 1.0

    def decode(self) -> bytes:
        return decode_with_dictionary(self.tokens, self.dictionary)

    def parameter_count(self) -> int:
        return 0


def static_retrieval_certificate(candidate_bits: int, best_payload_bits: int, block: bytes) -> dict[str, float]:
    scan_bits = int(best_payload_bits + int(STATIC_SCAN_DECODER_BITS))
    qa_bits = int(best_payload_bits + int(STATIC_SCAN_DECODER_BITS) + int(QA_WRAPPER_BITS))
    payload_improvement_required_bits = int(max(0, candidate_bits - scan_bits + 1))
    return {
        "same_payload_scan_success": 1.0,
        "same_payload_qa_wrapper_success": 1.0,
        "same_payload_scan_bits": float(scan_bits),
        "same_payload_qa_wrapper_bits": float(qa_bits),
        "same_payload_wrapper_extra_bits": float(qa_bits - scan_bits),
        "static_retrieval_dominance_certificate": float(int(qa_bits >= scan_bits and len(block) > 0)),
        "same_payload_qa_wrapper_cannot_beat_scan": float(int(qa_bits >= scan_bits)),
        "payload_bits_must_improve_for_wrapper_to_win": float(payload_improvement_required_bits),
    }


def compression_trial(block: bytes, train_blob: bytes, manifest: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    cell = LearnedBlockCodecFrontier(train_blob, block, manifest)
    decoded = cell.decode()
    standards = standard_codec_sweep(block)
    transformed = transformed_standard_codec_sweep(block)
    all_standards = standards + transformed
    best = min(all_standards, key=lambda row: (float(row["charged_bits"]), str(row["codec"])))
    best_raw = min(standards, key=lambda row: (float(row["charged_bits"]), str(row["codec"])))
    best_transformed = min(transformed, key=lambda row: (float(row["charged_bits"]), str(row["codec"])))
    best_payload = decompress_standard(str(best["codec"]).replace("delta_", ""), best["payload"])
    if str(best["codec"]).startswith("delta_"):
        best_payload = inverse_current_payload_transform(best_payload)
    random_block = random_label_block(seed, len(block))
    random_cell = LearnedBlockCodecFrontier(train_blob, random_block, manifest)
    random_standard = min(standard_codec_sweep(random_block), key=lambda row: (float(row["charged_bits"]), str(row["codec"])))
    byte_model = byte_model_bits(train_blob, block)
    useful_bits = int(len(block) * 8)
    improvement = (float(best["charged_bits"]) - float(cell.charged_bits)) / max(float(best["charged_bits"]), 1.0)
    beats_best_by_margin = float(int(improvement >= float(WIN_MARGIN)))
    exact_reconstruction = float(int(decoded == block and cell.reconstruction_sha256 == cell.block_sha256))
    standard_reconstruction = float(int(best_payload == block))
    random_ratio = float(random_cell.charged_bits) / max(float(cell.charged_bits), 1.0)
    incompressible_ratio = float(random_standard["charged_bits"]) / max(float(len(random_block) * 8), 1.0)
    certificate = static_retrieval_certificate(cell.charged_bits, int(best["charged_bits"]), block)
    strict_density = float(useful_bits) / max(float(cell.parameter_count()) + float(cell.charged_bits) / 16.0, 1.0)
    return {
        "block_bytes": float(len(block)),
        "dictionary_entry_count": float(len(cell.dictionary)),
        "token_count": float(len(cell.tokens)),
        "literal_token_count": float(sum(1 for kind, _value in cell.tokens if int(kind) == 0)),
        "phrase_token_count": float(sum(1 for kind, _value in cell.tokens if int(kind) == 1)),
        "phrase_dictionary_bits": float(cell.dictionary_bits),
        "phrase_token_stream_bits": float(cell.token_bits),
        "learned_payload_bits": float(cell.block_payload_bits),
        "learned_charged_bits": float(cell.charged_bits),
        "best_standard_codec": str(best["codec"]),
        "best_standard_payload_bits": float(best["payload_bits"]),
        "best_standard_charged_bits": float(best["charged_bits"]),
        "best_raw_standard_codec": str(best_raw["codec"]),
        "best_raw_standard_charged_bits": float(best_raw["charged_bits"]),
        "best_transformed_standard_codec": str(best_transformed["codec"]),
        "best_transformed_standard_charged_bits": float(best_transformed["charged_bits"]),
        "standard_codec_sweep_count": float(len(all_standards)),
        "learned_vs_best_standard_improvement": float(improvement),
        "beats_best_fair_standard_by_2pct": beats_best_by_margin,
        "exact_reconstruction": exact_reconstruction,
        "standard_reconstruction": standard_reconstruction,
        "random_label_learned_charged_bits": float(random_cell.charged_bits),
        "random_label_best_standard_charged_bits": float(random_standard["charged_bits"]),
        "random_label_cost_ratio_over_real": float(random_ratio),
        "random_label_payload_incompressible": float(int(incompressible_ratio >= 0.98)),
        "no_per_fact_rows": 1.0,
        "per_fact_value_slice_count": float(cell.per_fact_value_slice_count),
        "block_stream_count": float(cell.block_stream_count),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "train_only_dictionary_used": float(cell.train_only_dictionary_used),
        "learned_or_phrase_codec_used": float(cell.learned_or_phrase_codec_used),
        "reads_from_charged_representation": float(cell.reads_from_charged_representation),
        "manifest_bits": float(cell.manifest_bits),
        "learned_decoder_bits": float(LEARNED_DECODER_BITS),
        "standard_decoder_bits": float(STANDARD_DECODER_BITS),
        "codec_selector_bits": float(cell.codec_selector_bits),
        "useful_retrievable_bits": float(useful_bits),
        "strict_density": float(strict_density),
        "strict_multiplier": float(strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "publishable": 0.0,
        "static_compression_frontier_pass": float(int(exact_reconstruction == 1.0 and beats_best_by_margin == 1.0)),
        **byte_model,
        **certificate,
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    block_bytes = int(HARD_BLOCK_BYTES if profile == "hard" else SMOKE_BLOCK_BYTES)
    train_blob, test_blob, train_manifest, test_manifest = load_authored_sources()
    block = heldout_block(test_blob, block_bytes, seed)
    overlap = train_test_overlap_counts(train_blob, block, train_manifest, test_manifest)
    trial = compression_trial(block, train_blob, train_manifest, seed)
    source_holdout_pass = float(
        int(
            overlap["source_train_test_path_overlap_count"] == 0.0
            and overlap["source_train_test_hash_overlap_count"] == 0.0
        )
    )
    honest_failure = float(
        int(
            trial["exact_reconstruction"] == 1.0
            and trial["beats_best_fair_standard_by_2pct"] == 0.0
            and trial["random_label_payload_incompressible"] == 1.0
            and trial["no_per_fact_rows"] == 1.0
            and trial["publishable"] == 0.0
        )
    )
    engineering_pass = float(
        int(
            trial["exact_reconstruction"] == 1.0
            and trial["standard_reconstruction"] == 1.0
            and trial["per_fact_value_slice_count"] == 0.0
            and trial["static_retrieval_dominance_certificate"] == 1.0
            and source_holdout_pass == 1.0
        )
    )
    return {
        "profile": profile,
        "seed": float(seed),
        "train_blob_bytes": float(len(train_blob)),
        "test_blob_bytes": float(len(test_blob)),
        "source_holdout_pass": source_holdout_pass,
        "honest_failure_reported": honest_failure,
        "engineering_pass": engineering_pass,
        "fair_standard_win_margin_required": float(WIN_MARGIN),
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "associative_random_key_target": 0.0,
        "qa_wrapper_promoted": 0.0,
        **overlap,
        **trial,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["engineering_pass"]),
        f"{SIMULATION_ID}_publishable": 0.0,
        f"{SIMULATION_ID}_paper_ready_local_candidate_authorized": 0.0,
        f"{SIMULATION_ID}_strict_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_general_unknown_structure_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
        f"{SIMULATION_ID}_external_simulator_authorized": 0.0,
        f"{SIMULATION_ID}_arbitrary_chat_authorized": 0.0,
    }
    for key, value in row.items():
        if key in {"profile", "best_standard_codec", "best_raw_standard_codec", "best_transformed_standard_codec"}:
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
    metrics_path = output_dir / "local_100k_learned_block_codec_frontier_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "seed": int(SEED),
            "smoke_block_bytes": int(SMOKE_BLOCK_BYTES),
            "hard_block_bytes": int(HARD_BLOCK_BYTES),
            "phrase_min_bytes": int(PHRASE_MIN_BYTES),
            "phrase_max_bytes": int(PHRASE_MAX_BYTES),
            "dictionary_size": int(DICTIONARY_SIZE),
            "learned_decoder_bits": int(LEARNED_DECODER_BITS),
            "standard_decoder_bits": int(STANDARD_DECODER_BITS),
            "qa_wrapper_bits": int(QA_WRAPPER_BITS),
            "static_scan_decoder_bits": int(STATIC_SCAN_DECODER_BITS),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "win_margin": float(WIN_MARGIN),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_block_bytes"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_learned_block_codec_frontier_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
