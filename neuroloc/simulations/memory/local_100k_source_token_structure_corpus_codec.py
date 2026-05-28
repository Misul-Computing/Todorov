from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import best_codec, decompress_best, read_joined, target_paths, train_paths
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import learned_codec, restore_learned

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_token_structure_corpus_codec"
SEED = env_int("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_SEED", 10433)
MAX_BLOCK_BYTES = env_int("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_MAX_BLOCK_BYTES", 250000)
SELECTOR_BITS_PER_BLOCK = env_int("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_SELECTOR_BITS_PER_BLOCK", 16)
MIN_AGGREGATE_PAYLOAD_IMPROVEMENT = float(os.environ.get("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_MIN_AGGREGATE_PAYLOAD_IMPROVEMENT", "0.025"))

require_positive("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_MAX_BLOCK_BYTES", MAX_BLOCK_BYTES)
require_positive("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_SELECTOR_BITS_PER_BLOCK", SELECTOR_BITS_PER_BLOCK)

PROFILES = {
    "smoke": {"block_count": 3},
    "hard": {"block_count": 5},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_TOKEN_STRUCTURE_CORPUS_CODEC_PROFILE must be smoke or hard")
    return value


def safe_py_files() -> list[Path]:
    excluded = {".git", "codex_local_output", ".tmp_pytest", ".codex_pytest_tmp", ".tmp_pytest_local", ".tmp_pytest_run", "codex_tmp_a", "temp_ok", "__pycache__"}
    rows = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in excluded for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        rows.append(path)
    return sorted(rows)


def corpus_blocks(profile: str) -> list[tuple[str, list[Path]]]:
    all_py = safe_py_files()
    rel = lambda path: str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    blocks = [
        ("heldout_hard_source", target_paths("hard")),
        ("src_library", [path for path in all_py if rel(path).startswith("src/")]),
        ("memory_core", [path for path in all_py if rel(path).startswith("neuroloc/simulations/memory/") and not path.name.startswith("local_100k")][:24]),
        ("local100k_simulations", [path for path in all_py if path.name.startswith("local_100k")][:20]),
        ("local100k_tests", [path for path in all_py if path.name.startswith("test_local_100k")][:20]),
    ]
    return blocks[: int(PROFILES[profile]["block_count"])]


def read_block(paths: list[Path]) -> bytes:
    block = read_joined([path for path in paths if path.exists()])
    return bytes(block[: int(MAX_BLOCK_BYTES)])


def restore_standard(codec_name: str, payload: bytes) -> bytes:
    return decompress_best(codec_name, payload)


def measure_one(train_block: bytes, name: str, block: bytes) -> dict[str, Any]:
    standard_name, standard_payload = best_codec(block)
    learned = learned_codec(train_block, block)
    learned_payload_bits = int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"])) + len(bytes(learned["dictionary_payload"]))) * 8 + 896)
    standard_payload_bits = int(len(standard_payload) * 8)
    if learned_payload_bits < standard_payload_bits:
        selected = "token_structure"
        selected_bits = learned_payload_bits + int(SELECTOR_BITS_PER_BLOCK)
        restored = restore_learned(learned)
    else:
        selected = "standard"
        selected_bits = standard_payload_bits + int(SELECTOR_BITS_PER_BLOCK)
        restored = restore_standard(standard_name, standard_payload)
    return {
        "name": name,
        "block_bytes": float(len(block)),
        "standard_payload_bits": float(standard_payload_bits),
        "token_structure_payload_bits": float(learned_payload_bits),
        "selected_payload_bits": float(selected_bits),
        "selected_token_structure": float(int(selected == "token_structure")),
        "selected_standard": float(int(selected == "standard")),
        "exact_reconstruction_success": float(restored == block),
        "standard_codec_family_id": 0.0,
        "selector_bits": float(SELECTOR_BITS_PER_BLOCK),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    train_block = read_joined(train_paths())
    rows = [measure_one(train_block, name, read_block(paths)) for name, paths in corpus_blocks(profile)]
    standard_bits = sum(float(row["standard_payload_bits"]) for row in rows)
    selected_bits = sum(float(row["selected_payload_bits"]) for row in rows)
    useful_bits = sum(float(row["block_bytes"]) * 8.0 for row in rows)
    aggregate_improvement = float(standard_bits - selected_bits) / max(float(standard_bits), 1.0)
    exact_success = min(float(row["exact_reconstruction_success"]) for row in rows)
    token_selected_count = sum(float(row["selected_token_structure"]) for row in rows)
    standard_selected_count = sum(float(row["selected_standard"]) for row in rows)
    fallback_required = 1.0 if profile == "hard" else 0.0
    engineering_pass = float(int(exact_success == 1.0 and aggregate_improvement >= float(MIN_AGGREGATE_PAYLOAD_IMPROVEMENT) and token_selected_count >= 1.0 and standard_selected_count >= fallback_required))
    return {
        "profile": profile,
        "block_count": float(len(rows)),
        "train_file_count": float(len([path for path in train_paths() if path.exists()])),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "source_token_structure_corpus_codec_candidate": engineering_pass,
        "source_code_corpus_codec_product_authorized": 0.0,
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
        "useful_retrievable_bits": float(useful_bits),
        "aggregate_standard_payload_bits": float(standard_bits),
        "aggregate_selected_payload_bits": float(selected_bits),
        "aggregate_payload_improvement": float(aggregate_improvement),
        "exact_reconstruction_success_min": float(exact_success),
        "token_structure_selected_block_count": float(token_selected_count),
        "standard_fallback_selected_block_count": float(standard_selected_count),
        "selector_bits_per_block": float(SELECTOR_BITS_PER_BLOCK),
        "random_label_payload_control_required": 0.0,
        "same_block_standard_codec_baseline_used": 1.0,
        "controls_collapse": 0.0,
        "engineering_pass": engineering_pass,
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
    metrics_path = output_dir / "local_100k_source_token_structure_corpus_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "max_block_bytes": int(MAX_BLOCK_BYTES), "selector_bits_per_block": int(SELECTOR_BITS_PER_BLOCK)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_block_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_token_structure_corpus_codec_metrics.json", "path": metrics_path}],
        warnings=["broad source-code corpus codec benchmark with standard fallback; no nm, chat, knowledge, paid-compute, or broad breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} aggregate_improvement={summary[f'{SIMULATION_ID}_aggregate_payload_improvement']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
