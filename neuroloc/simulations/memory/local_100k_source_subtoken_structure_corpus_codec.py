from __future__ import annotations

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import best_codec, decode_codec_name, encode_codec_name, decompress_best, read_joined, train_paths
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_block_codec import learned_codec, measure_block, restore_learned

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_subtoken_structure_corpus_codec"
SEED = env_int("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_SEED", 10547)
MAX_BLOCK_BYTES = env_int("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_MAX_BLOCK_BYTES", 250000)
SELECTOR_BITS_PER_BLOCK = env_int("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_SELECTOR_BITS_PER_BLOCK", 16)
STANDARD_CODEC_HEADER_BITS_PER_BLOCK = env_int("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_STANDARD_HEADER_BITS_PER_BLOCK", 16)
MIN_AGGREGATE_PAYLOAD_IMPROVEMENT = float(os.environ.get("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_MIN_AGGREGATE_PAYLOAD_IMPROVEMENT", "0.043"))

require_positive("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_MAX_BLOCK_BYTES", MAX_BLOCK_BYTES)
require_positive("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_SELECTOR_BITS_PER_BLOCK", SELECTOR_BITS_PER_BLOCK)
require_positive("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_STANDARD_HEADER_BITS_PER_BLOCK", STANDARD_CODEC_HEADER_BITS_PER_BLOCK)

PROFILES = {
    "smoke": {"block_count": 3},
    "hard": {"block_count": 5},
}

FROZEN_BLOCKS = [
    {
        "name": "heldout_hard_source",
        "sha256": "55de69a989511489075b7aacd8c2814117957d09fead16dc17d1ed5096ebd75b",
        "paths": [
            "src/spikes/spiking_brain.py",
            "neuroloc/simulations/memory/correction_field_capacity.py",
            "neuroloc/simulations/memory/multi_resolution_head_split.py",
            "neuroloc/simulations/memory/slot_surprise_writes.py",
        ],
    },
    {
        "name": "src_library",
        "sha256": "7973cc631a36cc5221f4f52d45d9a88e5b6126f1cbaf2683f4b5c897d75f0a6a",
        "paths": [
            "src/algebra/equivariant_linear.py",
            "src/algebra/geometric_product.py",
            "src/algebra/multivector.py",
            "src/layers/kda.py",
            "src/layers/mamba3.py",
            "src/layers/mla.py",
            "src/layers/swiglu.py",
            "src/model/decode_head.py",
            "src/model/embedding.py",
            "src/model/todorov.py",
            "src/spikes/atmn_spike.py",
            "src/spikes/spiking_brain.py",
            "src/spikes/ternary_spike.py",
            "src/training/evaluator.py",
            "src/training/loss.py",
            "src/training/optimizer.py",
            "src/utils/convergence.py",
            "src/utils/erf.py",
            "src/utils/memory.py",
        ],
    },
    {
        "name": "memory_core",
        "sha256": "0def9569e3de52a72cd60942e44b9024f3eccd99a102a463fc51a8272f9579f1",
        "paths": [
            "neuroloc/simulations/memory/asymmetric_outer_product_recall.py",
            "neuroloc/simulations/memory/capacity_scaling.py",
            "neuroloc/simulations/memory/compression_under_bit_budget_mirror.py",
            "neuroloc/simulations/memory/contextual_gate_routing.py",
            "neuroloc/simulations/memory/contextual_recall_world.py",
            "neuroloc/simulations/memory/correction_field_capacity.py",
            "neuroloc/simulations/memory/correction_field_trained_prediction.py",
            "neuroloc/simulations/memory/delayed_cue_world.py",
            "neuroloc/simulations/memory/eligibility_gated_local_commit.py",
            "neuroloc/simulations/memory/episodic_replay_reuse.py",
            "neuroloc/simulations/memory/episodic_separation_completion.py",
            "neuroloc/simulations/memory/imagination_recombination.py",
            "neuroloc/simulations/memory/multi_association_recall.py",
            "neuroloc/simulations/memory/multi_resolution_head_split.py",
            "neuroloc/simulations/memory/nm_hard_symbolic_test_material.py",
            "neuroloc/simulations/memory/oracle_compression_analysis.py",
            "neuroloc/simulations/memory/pattern_completion.py",
            "neuroloc/simulations/memory/slot_buffer_capacity.py",
            "neuroloc/simulations/memory/slot_integration.py",
            "neuroloc/simulations/memory/slot_key_interference_sweep.py",
            "neuroloc/simulations/memory/slot_surprise_writes.py",
        ],
    },
    {
        "name": "local100k_simulations",
        "sha256": "fa379f0f1f86289cace0c45ddf26176d436297cd50cf61f97c913afbad9f5eca",
        "paths": [
            "neuroloc/simulations/memory/local_100k_3d_nm_mirror.py",
            "neuroloc/simulations/memory/local_100k_content_addressed_source_codec.py",
            "neuroloc/simulations/memory/local_100k_full_nm.py",
            "neuroloc/simulations/memory/local_100k_high_density_cell.py",
            "neuroloc/simulations/memory/local_100k_indent_token_block_codec.py",
            "neuroloc/simulations/memory/local_100k_learned_unknown_structure_density_cell.py",
            "neuroloc/simulations/memory/local_100k_llm_semantic_qa_codec.py",
            "neuroloc/simulations/memory/local_100k_margin_recompression_adapter.py",
            "neuroloc/simulations/memory/local_100k_paper_ready_adapter_benchmark.py",
            "neuroloc/simulations/memory/local_100k_replay_answer_mirror.py",
            "neuroloc/simulations/memory/local_100k_source_block_codec.py",
            "neuroloc/simulations/memory/local_100k_source_structure_block_codec.py",
            "neuroloc/simulations/memory/local_100k_source_token_structure_block_codec.py",
            "neuroloc/simulations/memory/local_100k_unknown_structure_density_probe.py",
            "neuroloc/simulations/memory/local_100k_weight_carried_qa_codec.py",
        ],
    },
    {
        "name": "tests_selected",
        "sha256": "fcd123100aab3cc23878de735d867c1f40d36d5acc34f8c7f4a514ad71a47d6d",
        "paths": [
            "tests/test_local_100k_3d_nm_mirror.py",
            "tests/test_local_100k_content_addressed_source_codec.py",
            "tests/test_local_100k_full_nm.py",
            "tests/test_local_100k_high_density_cell.py",
            "tests/test_local_100k_indent_token_block_codec.py",
            "tests/test_local_100k_llm_semantic_qa_codec.py",
            "tests/test_local_100k_margin_recompression_adapter.py",
            "tests/test_local_100k_paper_ready_adapter_benchmark.py",
            "tests/test_local_100k_source_block_codec.py",
            "tests/test_local_100k_source_structure_block_codec.py",
            "tests/test_local_100k_source_token_structure_block_codec.py",
            "tests/test_local_100k_unknown_structure_density_probe.py",
            "tests/test_local_100k_weight_carried_qa_codec.py",
            "tests/test_local_state_write_read_mirror.py",
            "tests/test_simulation_suite.py",
        ],
    },
]


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_SUBTOKEN_STRUCTURE_CORPUS_CODEC_PROFILE must be smoke or hard")
    return value


def block_paths(row: dict[str, Any]) -> list[Path]:
    return [PROJECT_ROOT / str(path) for path in row["paths"]]


def read_block(row: dict[str, Any]) -> bytes:
    present = [path for path in block_paths(row) if path.exists()]
    if len(present) != len(row["paths"]):
        raise ValueError(f"missing frozen corpus path for {row['name']}")
    block = read_joined(present)
    return bytes(block[: int(MAX_BLOCK_BYTES)])


def corpus_blocks(profile: str) -> list[dict[str, Any]]:
    return FROZEN_BLOCKS[: int(PROFILES[profile]["block_count"])]


def restore_standard(codec_code: int, payload: bytes) -> bytes:
    return decompress_best(decode_codec_name(int(codec_code)), payload)


def measure_one(train_block: bytes, row: dict[str, Any], block: bytes, seed: int, index: int) -> dict[str, Any]:
    block_hash = hashlib.sha256(block).hexdigest()
    standard_name, standard_payload = best_codec(block)
    learned = learned_codec(train_block, block)
    learned_payload_bits = int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"])) + len(bytes(learned["dictionary_payload"]))) * 8 + 896)
    standard_payload_bits = int(len(standard_payload) * 8)
    if learned_payload_bits < standard_payload_bits:
        selected = "subtoken_structure"
        selected_bits = learned_payload_bits + int(SELECTOR_BITS_PER_BLOCK)
        restored = restore_learned(learned)
        control = measure_block(train_block, block, int(seed) + int(index))
        controls_collapse = float(int(control["compressed_stream_read_success"] == 1.0 and control["decoder_disabled_exact_reconstruction_success"] == 0.0 and control["wrong_indent_unit_exact_reconstruction_success"] == 0.0 and control["token_dictionary_disabled_exact_reconstruction_success"] == 0.0 and control["shuffle_body_payload_exact_reconstruction_success"] == 0.0 and control["shuffle_count_payload_exact_reconstruction_success"] == 0.0 and control["shuffle_dictionary_payload_exact_reconstruction_success"] == 0.0 and control["codec_state_has_raw_target_block"] == 0.0 and control["codec_state_has_uncompressed_count_stream"] == 0.0 and control["codec_state_has_uncompressed_body_stream"] == 0.0 and control["codec_state_has_restored_block"] == 0.0))
        random_label_payload_incompressible = float(control["random_label_payload_incompressible"])
        random_label_payload_improvement = float(control["random_label_payload_improvement_over_best_standard"])
        selected_codec_header_bits = 0
    else:
        selected = "standard"
        selected_bits = standard_payload_bits + int(SELECTOR_BITS_PER_BLOCK) + int(STANDARD_CODEC_HEADER_BITS_PER_BLOCK)
        codec_code = encode_codec_name(standard_name)
        restored = restore_standard(codec_code, standard_payload)
        controls_collapse = 1.0
        random_label_payload_incompressible = 1.0
        random_label_payload_improvement = 0.0
        selected_codec_header_bits = int(STANDARD_CODEC_HEADER_BITS_PER_BLOCK)
    return {
        "name": str(row["name"]),
        "block_bytes": float(len(block)),
        "block_hash_success": float(block_hash == str(row["sha256"])),
        "standard_payload_bits": float(standard_payload_bits),
        "subtoken_structure_payload_bits": float(learned_payload_bits),
        "selected_payload_bits": float(selected_bits),
        "selected_codec_header_bits": float(selected_codec_header_bits),
        "selected_subtoken_structure": float(int(selected == "subtoken_structure")),
        "selected_standard": float(int(selected == "standard")),
        "exact_reconstruction_success": float(restored == block),
        "standard_codec_family_id": float(encode_codec_name(standard_name)),
        "selector_bits": float(SELECTOR_BITS_PER_BLOCK),
        "controls_collapse": float(controls_collapse),
        "random_label_payload_incompressible": float(random_label_payload_incompressible),
        "random_label_payload_improvement_over_best_standard": float(random_label_payload_improvement),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    train_block = read_joined(train_paths())
    rows = [measure_one(train_block, row, read_block(row), int(seed), index) for index, row in enumerate(corpus_blocks(profile))]
    standard_bits = sum(float(row["standard_payload_bits"]) for row in rows)
    selected_bits = sum(float(row["selected_payload_bits"]) for row in rows)
    useful_bits = sum(float(row["block_bytes"]) * 8.0 for row in rows)
    aggregate_improvement = float(standard_bits - selected_bits) / max(float(standard_bits), 1.0)
    exact_success = min(float(row["exact_reconstruction_success"]) for row in rows)
    hash_success = min(float(row["block_hash_success"]) for row in rows)
    control_success = min(float(row["controls_collapse"]) for row in rows)
    random_label_incompressible = min(float(row["random_label_payload_incompressible"]) for row in rows)
    random_label_payload_improvement_max = max(float(row["random_label_payload_improvement_over_best_standard"]) for row in rows)
    subtoken_selected_count = sum(float(row["selected_subtoken_structure"]) for row in rows)
    standard_selected_count = sum(float(row["selected_standard"]) for row in rows)
    engineering_pass = float(int(exact_success == 1.0 and hash_success == 1.0 and control_success == 1.0 and random_label_incompressible == 1.0 and random_label_payload_improvement_max <= 0.0 and aggregate_improvement >= float(MIN_AGGREGATE_PAYLOAD_IMPROVEMENT) and subtoken_selected_count >= 1.0))
    return {
        "profile": profile,
        "block_count": float(len(rows)),
        "train_file_count": float(len([path for path in train_paths() if path.exists()])),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "source_subtoken_structure_corpus_codec_candidate": engineering_pass,
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
        "useful_retrievable_bits": float(useful_bits),
        "aggregate_standard_payload_bits": float(standard_bits),
        "aggregate_selected_payload_bits": float(selected_bits),
        "aggregate_payload_improvement": float(aggregate_improvement),
        "exact_reconstruction_success_min": float(exact_success),
        "frozen_manifest_hash_success_min": float(hash_success),
        "subtoken_structure_selected_block_count": float(subtoken_selected_count),
        "standard_fallback_selected_block_count": float(standard_selected_count),
        "selector_bits_per_block": float(SELECTOR_BITS_PER_BLOCK),
        "standard_codec_header_bits_per_block": float(STANDARD_CODEC_HEADER_BITS_PER_BLOCK),
        "random_label_payload_incompressible_min": float(random_label_incompressible),
        "random_label_payload_improvement_over_best_standard_max": float(random_label_payload_improvement_max),
        "random_label_payload_control_required": 1.0,
        "same_block_standard_codec_baseline_used": 1.0,
        "controls_collapse": float(control_success),
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
    metrics_path = output_dir / "local_100k_source_subtoken_structure_corpus_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "max_block_bytes": int(MAX_BLOCK_BYTES), "selector_bits_per_block": int(SELECTOR_BITS_PER_BLOCK), "standard_codec_header_bits_per_block": int(STANDARD_CODEC_HEADER_BITS_PER_BLOCK)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_block_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_subtoken_structure_corpus_codec_metrics.json", "path": metrics_path}],
        warnings=["broad source-code corpus codec benchmark over a frozen manifest; no nm, chat, knowledge, paid-compute, or broad breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} aggregate_improvement={summary[f'{SIMULATION_ID}_aggregate_payload_improvement']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
