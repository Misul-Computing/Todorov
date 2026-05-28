from __future__ import annotations

import hashlib
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import train_paths
from neuroloc.simulations.memory.local_100k_source_subtoken_global_stream_corpus_codec import SourceSubtokenGlobalStreamCorpusModule, codec_payload_bits, control_success, global_codec, global_raw_standard_payload_bits, random_blocks, read_limited_block, restore_all, standard_payload_bits
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import FROZEN_BLOCKS

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_subtoken_disjoint_retrieval_codec"
SEED = env_int("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_SEED", 12829)
CHUNK_BYTES = env_int("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_CHUNK_BYTES", 32)
RETRIEVAL_DECODER_BITS = env_int("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_DECODER_BITS", 2048)
UNDERCHARGED_MPH_BITS = env_int("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_UNDERCHARGED_MPH_BITS", 16)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))

require_positive("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_CHUNK_BYTES", CHUNK_BYTES)
require_positive("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_DECODER_BITS", RETRIEVAL_DECODER_BITS)
require_positive("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_UNDERCHARGED_MPH_BITS", UNDERCHARGED_MPH_BITS)

PROFILES = {
    "smoke": {"indices": (0, 3), "min_raw_margin_bits": 10000.0, "min_retrieval_count": 1000.0},
    "hard": {"indices": (0, 3, 4), "min_raw_margin_bits": 19000.0, "min_retrieval_count": 10000.0},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_SUBTOKEN_DISJOINT_RETRIEVAL_CODEC_PROFILE must be smoke or hard")
    return value


def target_rows(profile: str) -> list[dict[str, Any]]:
    return [FROZEN_BLOCKS[int(index)] for index in PROFILES[profile]["indices"]]


def digest_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_overlap_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    train = [path for path in train_paths() if path.exists()]
    train_resolved = {str(path.resolve()) for path in train}
    train_hashes = {digest_path(path) for path in train}
    target_paths = [PROJECT_ROOT / str(item) for row in rows for item in row["paths"]]
    target_resolved = {str(path.resolve()) for path in target_paths if path.exists()}
    target_hashes = {digest_path(path) for path in target_paths if path.exists()}
    return {
        "source_train_file_count": float(len(train_resolved)),
        "source_target_file_count": float(len(target_resolved)),
        "source_train_test_path_overlap_count": float(len(train_resolved & target_resolved)),
        "source_train_test_hash_overlap_count": float(len(train_hashes & target_hashes)),
    }


def retrieval_rows(blocks: list[bytes]) -> list[tuple[int, int, bytes]]:
    rows = []
    for block_index, block in enumerate(blocks):
        limit = max(0, len(block) - int(CHUNK_BYTES) + 1)
        for offset in range(0, limit, int(CHUNK_BYTES)):
            rows.append((int(block_index), int(offset), bytes(block[offset : offset + int(CHUNK_BYTES)])))
    return rows


def retrieval_success(restored: list[bytes], rows: list[tuple[int, int, bytes]]) -> float:
    if not rows:
        return 0.0
    values = []
    for block_index, offset, expected in rows:
        actual = restored[int(block_index)][int(offset) : int(offset) + int(CHUNK_BYTES)]
        values.append(float(actual == expected))
    return float(min(values))


def state_probe(codec: dict[str, Any], blocks: list[bytes], rows: list[tuple[int, int, bytes]]) -> dict[str, float]:
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    restored = module.reconstruct()
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    reload_restored = reload_module.reconstruct()
    state_payload = b"".join(bytes(int(item) for item in state[name].tolist()) for name in ("shared_dictionary_payload", "count_payload", "body_payload", "length_payload"))
    raw_retained = float(any(block[: min(128, len(block))] in state_payload for block in blocks))
    return {
        "model_state_codec_payload_used": 1.0,
        "state_dict_buffer_payload_used": float(int({"global_header", "shared_dictionary_payload", "count_payload", "body_payload", "length_payload"}.issubset(set(state.keys())))),
        "model_state_exact_reconstruction_success": float(restored == blocks),
        "state_dict_reload_reconstruction_success": float(reload_restored == blocks),
        "model_state_chunk_retrieval_success": retrieval_success(restored, rows),
        "state_dict_reload_chunk_retrieval_success": retrieval_success(reload_restored, rows),
        "state_dict_raw_source_block_retained": raw_retained,
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    rows = target_rows(profile)
    blocks = [read_limited_block(row) for row in rows]
    codec = global_codec(blocks)
    restored = restore_all(codec)
    retrieval = retrieval_rows(blocks)
    selected_payload_bits = codec_payload_bits(codec)
    selected_retrieval_bits = int(selected_payload_bits + int(RETRIEVAL_DECODER_BITS))
    standard_retrieval_bits = int(standard_payload_bits(blocks) + int(RETRIEVAL_DECODER_BITS))
    raw_content_scan_bits = int(global_raw_standard_payload_bits(blocks) + int(RETRIEVAL_DECODER_BITS))
    undercharged_mph_bits = int(raw_content_scan_bits + int(UNDERCHARGED_MPH_BITS))
    useful_bits = int(len(retrieval) * int(CHUNK_BYTES) * 8)
    random_payloads = random_blocks(int(seed), blocks)
    random_codec = global_codec(random_payloads)
    random_selected_bits = int(codec_payload_bits(random_codec) + int(RETRIEVAL_DECODER_BITS))
    random_standard_bits = int(standard_payload_bits(random_payloads) + int(RETRIEVAL_DECODER_BITS))
    random_raw_scan_bits = int(global_raw_standard_payload_bits(random_payloads) + int(RETRIEVAL_DECODER_BITS))
    controls = control_success(codec, blocks, int(seed))
    overlaps = source_overlap_metrics(rows)
    state = state_probe(codec, blocks, retrieval)
    exact_retrieval_success = retrieval_success(restored, retrieval)
    raw_margin = float(raw_content_scan_bits - selected_retrieval_bits)
    mph_margin = float(undercharged_mph_bits - selected_retrieval_bits)
    standard_margin = float(standard_retrieval_bits - selected_retrieval_bits)
    controls_collapse = float(int(controls["wrong_indent_unit_exact_reconstruction_success"] == 0.0 and controls["shared_dictionary_disabled_exact_reconstruction_success"] == 0.0 and controls["shuffled_shared_dictionary_exact_reconstruction_success"] == 0.0 and controls["shuffled_body_payload_exact_reconstruction_success"] == 0.0 and controls["shuffled_count_payload_exact_reconstruction_success"] == 0.0 and controls["shuffled_length_payload_exact_reconstruction_success"] == 0.0))
    random_incompressible = float(int(random_selected_bits >= random_standard_bits and random_selected_bits >= random_raw_scan_bits))
    density = float(useful_bits) / max(float(selected_retrieval_bits) / 16.0, 1.0)
    engineering_pass = float(int(restored == blocks and exact_retrieval_success == 1.0 and state["state_dict_reload_chunk_retrieval_success"] == 1.0 and state["state_dict_raw_source_block_retained"] == 0.0 and overlaps["source_train_test_path_overlap_count"] == 0.0 and overlaps["source_train_test_hash_overlap_count"] == 0.0 and raw_margin >= float(PROFILES[profile]["min_raw_margin_bits"]) and mph_margin > raw_margin and standard_margin > 0.0 and len(retrieval) >= float(PROFILES[profile]["min_retrieval_count"]) and random_incompressible == 1.0 and controls_collapse == 1.0))
    return {
        "profile": profile,
        "block_count": float(len(blocks)),
        "chunk_bytes": float(CHUNK_BYTES),
        "retrieval_fact_count": float(len(retrieval)),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "useful_retrievable_bits": float(useful_bits),
        "selected_payload_bits": float(selected_payload_bits),
        "selected_retrieval_accounted_bits": float(selected_retrieval_bits),
        "standard_retrieval_accounted_bits": float(standard_retrieval_bits),
        "raw_content_scan_accounted_bits": float(raw_content_scan_bits),
        "undercharged_mph_accounted_bits": float(undercharged_mph_bits),
        "retrieval_decoder_bits": float(RETRIEVAL_DECODER_BITS),
        "undercharged_mph_bits": float(UNDERCHARGED_MPH_BITS),
        "margin_over_standard_retrieval_bits": float(standard_margin),
        "margin_over_raw_content_scan_bits": float(raw_margin),
        "margin_over_undercharged_mph_bits": float(mph_margin),
        "strict_density": float(density),
        "strict_multiplier": float(density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "exact_reconstruction_success": float(restored == blocks),
        "heldout_chunk_retrieval_success": float(exact_retrieval_success),
        "raw_content_scan_beaten": float(int(raw_margin > 0.0)),
        "undercharged_mph_beaten": float(int(mph_margin > 0.0)),
        "standard_retrieval_beaten": float(int(standard_margin > 0.0)),
        "random_label_payload_incompressible": float(random_incompressible),
        "random_label_selected_retrieval_bits": float(random_selected_bits),
        "random_label_standard_retrieval_bits": float(random_standard_bits),
        "random_label_raw_content_scan_bits": float(random_raw_scan_bits),
        "controls_collapse": float(controls_collapse),
        "source_code_retrieval_codec_product_authorized": float(engineering_pass),
        "source_code_retrieval_breakthrough_authorized": 0.0,
        "static_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "engineering_pass": float(engineering_pass),
        **controls,
        **overlaps,
        **state,
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
    metrics_path = output_dir / "local_100k_source_subtoken_disjoint_retrieval_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "chunk_bytes": int(CHUNK_BYTES), "retrieval_decoder_bits": int(RETRIEVAL_DECODER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_retrieval_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_subtoken_disjoint_retrieval_codec_metrics.json", "path": metrics_path}],
        warnings=["source-code disjoint retrieval codec only; no broad knowledge, chat, full nm, paid-compute, strict breakthrough, or 600x authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} retrieval_facts={summary[f'{SIMULATION_ID}_retrieval_fact_count']:.0f} raw_scan_margin_bits={summary[f'{SIMULATION_ID}_margin_over_raw_content_scan_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
