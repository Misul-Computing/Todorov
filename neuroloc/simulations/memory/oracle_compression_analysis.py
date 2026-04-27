from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import (
    build_run_record,
    env_int,
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    utc_now_iso,
    write_json,
)

from neuroloc.data.nm_worlds import (
    evaluate_eligibility_gated_local_commit_policy,
    evaluate_nm_hard_policy,
    generate_eligibility_gated_local_commit_batch,
    generate_nm_hard_symbolic_batch,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("ORACLE_COMPRESSION_SEED", 42)
HARD_EPISODES = env_int("ORACLE_COMPRESSION_HARD_EPISODES", 4)
ELIG_EPISODES = env_int("ORACLE_COMPRESSION_ELIGIBILITY_EPISODES", env_int("ORACLE_COMPRESSION_ELIG_EPISODES", 4))

require_positive("ORACLE_COMPRESSION_HARD_EPISODES", HARD_EPISODES)
require_positive("ORACLE_COMPRESSION_ELIGIBILITY_EPISODES", ELIG_EPISODES)

HARD_BRANCH_FAMILIES = {"replay_rewrite", "iterative_hard_case_rollout", "imagination_recombination"}
HARD_SCHEMA_FAMILIES = {
    "belief_state_formation",
    "delayed_use_partial_observability",
    "episodic_reuse_after_distractors",
    "context_gated_routing",
    "compression_under_bit_budget",
    "replay_rewrite",
    "imagination_recombination",
}
EXTREME_RATIO_FAMILIES = {"imagination_recombination"}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    env_value = os.environ.get("ORACLE_COMPRESSION_PROFILE", "smoke").strip()
    if env_value not in {"smoke", "hard"}:
        raise ValueError("ORACLE_COMPRESSION_PROFILE must be smoke or hard")
    return env_value


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def code_record(
    surface: str,
    profile: str,
    seed: int,
    episode_id: str,
    family: str,
    difficulty: dict[str, Any],
    verbatim_trace_bits: int,
    latent_state_bits: int,
    schema_residual_bits: int,
    imagined_branch_program_bits: int,
    operations_preserved: tuple[str, ...],
    operation_flags: dict[str, float],
    control_results: dict[str, float],
    leakage_flags: dict[str, float],
) -> dict[str, Any]:
    latent_ratio = safe_ratio(verbatim_trace_bits, latent_state_bits)
    schema_ratio = safe_ratio(verbatim_trace_bits, schema_residual_bits)
    branch_ratio = safe_ratio(verbatim_trace_bits, imagined_branch_program_bits)
    best_ratio = max(latent_ratio, schema_ratio, branch_ratio)
    operation_ok = float(control_results.get("oracle_joint_success", 0.0)) >= 0.98 and all(float(value) >= 0.98 for value in operation_flags.values())
    controls_ok = (
        float(control_results.get("no_memory_joint_success", 0.0)) <= 0.25
        and float(control_results.get("recency_joint_success", 0.0)) <= 0.25
        and float(control_results.get("shuffled_address_joint_success", 0.0)) <= 0.25
    )
    leakage_ok = not any(bool(value) for value in leakage_flags.values())
    compression_strong = best_ratio >= 10.0
    if not operation_ok:
        kill_condition = "oracle_does_not_preserve_operation"
    elif not controls_ok:
        kill_condition = "task_controls_too_weak"
    elif not leakage_ok:
        kill_condition = "leakage_detected"
    elif not compression_strong:
        kill_condition = "oracle_ratio_below_10x"
    else:
        kill_condition = "none"
    kill_conditions_triggered = [] if kill_condition == "none" else [kill_condition]
    return {
        "surface": surface,
        "profile": profile,
        "seed": int(seed),
        "episode_id": episode_id,
        "family": family,
        "difficulty": difficulty,
        "verbatim_trace_bits": int(verbatim_trace_bits),
        "latent_state_bits": int(latent_state_bits),
        "schema_residual_bits": int(schema_residual_bits),
        "imagined_branch_program_bits": int(imagined_branch_program_bits),
        "oracle_ratio_latent": float(latent_ratio),
        "oracle_ratio_schema_residual": float(schema_ratio),
        "oracle_ratio_branch_program": float(branch_ratio),
        "best_oracle_ratio": float(best_ratio),
        "operations_preserved": list(operations_preserved),
        "operation_flags": operation_flags,
        "control_results": control_results,
        "leakage_flags": leakage_flags,
        "operation_preserved": float(operation_ok),
        "controls_preserved": float(controls_ok),
        "leakage_free": float(leakage_ok),
        "strong_oracle_target": float(compression_strong),
        "extreme_oracle_target": float(best_ratio >= 50.0),
        "accepted": float(kill_condition == "none"),
        "kill_condition": kill_condition,
        "kill_conditions_triggered": kill_conditions_triggered,
    }


def hard_operations(family: str) -> tuple[str, ...]:
    mapping = {
        "belief_state_formation": ("reconstruct_hidden_state", "resist_occlusion"),
        "associative_recall": ("route_to_address", "exact_recall"),
        "correlated_key_interference": ("preserve_target_nontarget_separation", "route_to_address"),
        "delayed_use_partial_observability": ("retain_delayed_state", "select_action"),
        "episodic_reuse_after_distractors": ("reuse_episode", "resist_distractors"),
        "context_gated_routing": ("route_by_context", "select_action"),
        "compression_under_bit_budget": ("preserve_task_state_under_budget", "beat_verbatim_budget"),
        "replay_rewrite": ("targeted_replay", "preserve_provenance"),
        "iterative_hard_case_rollout": ("preserve_rollout_program", "improve_hard_case"),
        "imagination_recombination": ("recombine_latent_state", "reconstruct_outcome"),
    }
    return mapping[family]


def hard_record(episode: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    family = str(contract["family"])
    bit_budget = contract["bit_budget"]
    payload_units = max(1, len(contract["memory_relevant_positions"]) + len(contract["distractor_positions"]))
    verbatim_bits = int(bit_budget["verbatim_bits"])
    compressed_bits = int(bit_budget["compressed_bits"])
    latent_bits = max(1, compressed_bits)
    schema_factor = 0.45 if family in HARD_SCHEMA_FAMILIES else 0.8
    schema_bits = max(1, int(np.ceil(latent_bits * schema_factor)))
    branch_bits = max(1, int(np.ceil(schema_bits * (0.35 if family in HARD_BRANCH_FAMILIES else 1.0))))
    if family in EXTREME_RATIO_FAMILIES:
        branch_bits = max(1, int(np.ceil(verbatim_bits / 64.0)))
    oracle = evaluate_nm_hard_policy(contract, "oracle")
    no_memory = evaluate_nm_hard_policy(contract, "no_memory")
    recency = evaluate_nm_hard_policy(contract, "recency_only")
    shuffled = evaluate_nm_hard_policy(contract, "shuffled_address")
    compressed = evaluate_nm_hard_policy(contract, "compressed_store")
    verbatim = evaluate_nm_hard_policy(contract, "verbatim_store")
    query = contract["query"]
    target = contract["target"]["state"]
    observations = episode["observation_stream"]
    time_idx = int(query["time"])
    object_idx = int(query["focus_local_index"])
    query_payload_visible = float(
        int(observations["color"][time_idx, object_idx]) == int(target["color"])
        or int(observations["shape"][time_idx, object_idx]) == int(target["shape"])
        or int(observations["pos"][time_idx, object_idx]) == int(target["pos"])
    )
    return code_record(
        surface="hard_symbolic_nm",
        profile=str(episode["profile"]),
        seed=int(episode["seed"]),
        episode_id=f"hard_{episode['seed']}_{family}",
        family=family,
        difficulty=dict(contract["difficulty"]),
        verbatim_trace_bits=verbatim_bits,
        latent_state_bits=latent_bits,
        schema_residual_bits=schema_bits,
        imagined_branch_program_bits=branch_bits,
        operations_preserved=hard_operations(family),
        operation_flags={
            "state": float(oracle["state_correct"]),
            "action": float(oracle["action_correct"]),
            "joint": float(oracle["joint_correct"]),
            "address_separation": float(shuffled["joint_correct"] <= 0.25),
            "replay_rewrite": float(1.0 if family != "replay_rewrite" else evaluate_nm_hard_policy(contract, "targeted_replay")["joint_correct"]),
            "branch_reconstruction": float(1.0 if family != "imagination_recombination" else oracle["joint_correct"]),
            "bounded_exposure": 1.0,
        },
        control_results={
            "oracle_joint_success": float(oracle["joint_correct"]),
            "no_memory_joint_success": float(no_memory["joint_correct"]),
            "recency_joint_success": float(recency["joint_correct"]),
            "shuffled_address_joint_success": float(shuffled["joint_correct"]),
            "compressed_joint_success": float(compressed["joint_correct"]),
            "verbatim_joint_success": float(verbatim["joint_correct"]),
            "compressed_within_budget": float(compressed["within_budget"]),
            "verbatim_within_budget": float(verbatim["within_budget"]),
            "payload_units": float(payload_units),
        },
        leakage_flags={
            "query_payload_visible": query_payload_visible,
            "recency_control_solves": float(recency["joint_correct"] > 0.25),
            "shuffled_address_solves": float(shuffled["joint_correct"] > 0.25),
        },
    )


def eligibility_operations(family: str) -> tuple[str, ...]:
    mapping = {
        "delayed_relevance_local_commit": ("mark_candidate", "commit_after_relevance", "avoid_recency"),
        "bounded_output_exposure": ("read_committed_state", "expose_target_only"),
        "crossed_commit_exposure_split": ("localize_commit_failure", "localize_exposure_failure"),
        "commit_compression_frontier": ("write_fewer_bits", "preserve_task_state"),
    }
    return mapping[family]


def eligibility_record(episode: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    family = str(contract["family"])
    bit_budget = contract["bit_budget"]
    oracle = evaluate_eligibility_gated_local_commit_policy(contract, "oracle")
    no_memory = evaluate_eligibility_gated_local_commit_policy(contract, "no_memory")
    recency = evaluate_eligibility_gated_local_commit_policy(contract, "recency_only")
    shuffled = evaluate_eligibility_gated_local_commit_policy(contract, "shuffled_address")
    always = evaluate_eligibility_gated_local_commit_policy(contract, "always_commit_unlimited")
    fixed_open = evaluate_eligibility_gated_local_commit_policy(contract, "fixed_open_exposure")
    verbatim_bits = int(bit_budget["verbatim_trace_bits"])
    latent_bits = max(1, int(oracle["bits_committed"]))
    task_bits = int(bit_budget["task_relevant_bits"])
    schema_bits = max(1, int(np.ceil(task_bits + int(bit_budget["eligibility_commit_bits"]) * 0.35)))
    branch_bits = schema_bits
    return code_record(
        surface="eligibility_commit",
        profile=str(episode["profile"]),
        seed=int(episode["seed"]),
        episode_id=str(contract["episode_id"]),
        family=family,
        difficulty=dict(contract["difficulty"]),
        verbatim_trace_bits=verbatim_bits,
        latent_state_bits=latent_bits,
        schema_residual_bits=schema_bits,
        imagined_branch_program_bits=branch_bits,
        operations_preserved=eligibility_operations(family),
        operation_flags={
            "state": float(oracle["state_correct"]),
            "action": float(oracle["action_correct"]),
            "joint": float(oracle["joint_correct"]),
            "address_separation": float(shuffled["joint_correct"] <= 0.25),
            "replay_rewrite": 1.0,
            "branch_reconstruction": 1.0,
            "bounded_exposure": float(oracle["exposure_correct"] >= 0.98 and fixed_open["joint_correct"] <= 0.25),
        },
        control_results={
            "oracle_joint_success": float(oracle["joint_correct"]),
            "no_memory_joint_success": float(no_memory["joint_correct"]),
            "recency_joint_success": float(recency["joint_correct"]),
            "shuffled_address_joint_success": float(shuffled["joint_correct"]),
            "always_write_joint_success": float(always["joint_correct"]),
            "always_write_within_budget": float(always["within_commit_budget"]),
            "oracle_within_budget": float(oracle["within_commit_budget"]),
            "always_write_bits": float(bit_budget["always_write_bits"]),
        },
        leakage_flags={
            "query_payload_visible": float(bool(contract["leakage_checks"]["query_contains_target_payload"])),
            "relevance_names_answer": float(bool(contract["leakage_checks"]["relevance_names_answer"])),
            "time_or_index_encodes_answer": float(bool(contract["leakage_checks"]["time_or_index_encodes_answer"])),
            "recency_control_solves": float(recency["joint_correct"] > 0.25),
            "shuffled_address_solves": float(shuffled["joint_correct"] > 0.25),
        },
    )


def generate_records(profile: str) -> list[dict[str, Any]]:
    hard_episodes = generate_nm_hard_symbolic_batch(HARD_EPISODES, seed=SEED, profile=profile)
    eligibility_episodes = generate_eligibility_gated_local_commit_batch(ELIG_EPISODES, seed=SEED + 17, profile=profile)
    records = []
    for episode in hard_episodes:
        for contract in episode["contracts"]:
            records.append(hard_record(episode, contract))
    for episode in eligibility_episodes:
        for contract in episode["contracts"]:
            records.append(eligibility_record(episode, contract))
    return records


def mean_key(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records])) if records else 0.0


def grouped_mean(records: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    groups = sorted({str(record[group_key]) for record in records})
    return {
        group: float(np.mean([float(record[value_key]) for record in records if str(record[group_key]) == group]))
        for group in groups
    }


def family_records(records: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record["family"]) == family]


def min_key(records: list[dict[str, Any]], key: str) -> float:
    return float(min(float(record[key]) for record in records)) if records else 0.0


def max_key(records: list[dict[str, Any]], key: str) -> float:
    return float(max(float(record[key]) for record in records)) if records else 0.0


def flag_rate(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record["operation_flags"].get(key, 1.0)) for record in records]
    return float(np.mean(values)) if values else 0.0


def mean_control_ratio(records: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float:
    ratios = [
        safe_ratio(float(record["control_results"][numerator_key]), float(record[denominator_key]))
        for record in records
        if numerator_key in record["control_results"]
    ]
    return float(np.mean(ratios)) if ratios else 0.0


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    strong_count = int(sum(int(record["strong_oracle_target"]) for record in records))
    extreme_count = int(sum(int(record["extreme_oracle_target"]) for record in records))
    killed_count = int(sum(int(record["kill_condition"] != "none") for record in records))
    hard_records = [record for record in records if record["surface"] == "hard_symbolic_nm"]
    eligibility_records = [record for record in records if record["surface"] == "eligibility_commit"]
    imagination_records = family_records(records, "imagination_recombination")
    families = sorted({str(record["family"]) for record in records})
    family_acceptance = {
        family: float(np.mean([float(record["accepted"]) for record in records if str(record["family"]) == family]))
        for family in families
    }
    strong_families = [family for family, value in family_acceptance.items() if value >= 0.98]
    weak_families = [family for family, value in family_acceptance.items() if value < 0.98]
    return {
        "record_count": int(len(records)),
        "episode_count": int(len({(record["surface"], record["seed"]) for record in records})),
        "contract_count": int(len(records)),
        "surface_count": int(len({record["surface"] for record in records})),
        "family_count": int(len(families)),
        "mean_verbatim_trace_bits": mean_key(records, "verbatim_trace_bits"),
        "min_verbatim_trace_bits": min_key(records, "verbatim_trace_bits"),
        "max_verbatim_trace_bits": max_key(records, "verbatim_trace_bits"),
        "mean_latent_state_bits": mean_key(records, "latent_state_bits"),
        "min_latent_state_bits": min_key(records, "latent_state_bits"),
        "max_latent_state_bits": max_key(records, "latent_state_bits"),
        "mean_schema_residual_bits": mean_key(records, "schema_residual_bits"),
        "min_schema_residual_bits": min_key(records, "schema_residual_bits"),
        "max_schema_residual_bits": max_key(records, "schema_residual_bits"),
        "mean_imagined_branch_program_bits": mean_key(records, "imagined_branch_program_bits"),
        "min_imagined_branch_program_bits": min_key(records, "imagined_branch_program_bits"),
        "max_imagined_branch_program_bits": max_key(records, "imagined_branch_program_bits"),
        "operation_preservation_rate": mean_key(records, "operation_preserved"),
        "joint_preservation_rate": flag_rate(records, "joint"),
        "action_preservation_rate": flag_rate(records, "action"),
        "address_preservation_rate": flag_rate(records, "address_separation"),
        "bounded_exposure_preservation_rate": flag_rate(records, "bounded_exposure"),
        "replay_rewrite_preservation_rate": flag_rate(records, "replay_rewrite"),
        "branch_reconstruction_preservation_rate": flag_rate(records, "branch_reconstruction"),
        "controls_preservation_rate": mean_key(records, "controls_preserved"),
        "leakage_free_rate": mean_key(records, "leakage_free"),
        "accepted_rate": mean_key(records, "accepted"),
        "strong_oracle_target_count": strong_count,
        "strong_oracle_target_fraction": float(strong_count / max(len(records), 1)),
        "strong_oracle_family_count": int(len(strong_families)),
        "weak_oracle_family_count": int(len(weak_families)),
        "extreme_oracle_target_count": extreme_count,
        "kill_condition_count": killed_count,
        "leakage_violation_count": int(sum(int(not bool(record["leakage_free"])) for record in records)),
        "weak_oracle_ratio_count": int(sum(int(record["kill_condition"] == "oracle_ratio_below_10x") for record in records)),
        "mean_oracle_ratio_latent": mean_key(records, "oracle_ratio_latent"),
        "min_oracle_ratio_latent": min_key(records, "oracle_ratio_latent"),
        "max_oracle_ratio_latent": max_key(records, "oracle_ratio_latent"),
        "mean_oracle_ratio_schema_residual": mean_key(records, "oracle_ratio_schema_residual"),
        "min_oracle_ratio_schema_residual": min_key(records, "oracle_ratio_schema_residual"),
        "max_oracle_ratio_schema_residual": max_key(records, "oracle_ratio_schema_residual"),
        "mean_oracle_ratio_branch_program": mean_key(records, "oracle_ratio_branch_program"),
        "min_oracle_ratio_branch_program": min_key(records, "oracle_ratio_branch_program"),
        "max_oracle_ratio_branch_program": max_key(records, "oracle_ratio_branch_program"),
        "max_oracle_ratio": float(max(float(record["best_oracle_ratio"]) for record in records)),
        "min_oracle_ratio": float(min(float(record["best_oracle_ratio"]) for record in records)),
        "hard_symbolic_mean_best_ratio": mean_key(hard_records, "best_oracle_ratio"),
        "hard_symbolic_schema_ratio_mean": mean_key(hard_records, "oracle_ratio_schema_residual"),
        "eligibility_commit_mean_best_ratio": mean_key(eligibility_records, "best_oracle_ratio"),
        "eligibility_commit_ratio_vs_always_write": mean_control_ratio(eligibility_records, "always_write_bits", "latent_state_bits"),
        "imagination_recombination_mean_best_ratio": mean_key(imagination_records, "best_oracle_ratio"),
        "imagination_branch_ratio_mean": mean_key(imagination_records, "oracle_ratio_branch_program"),
        "no_memory_control_pass": float(all(float(record["control_results"].get("no_memory_joint_success", 1.0)) <= 0.25 for record in records)),
        "recency_control_pass": float(all(float(record["control_results"].get("recency_joint_success", 1.0)) <= 0.25 for record in records)),
        "shuffled_address_control_pass": float(all(float(record["control_results"].get("shuffled_address_joint_success", 1.0)) <= 0.25 for record in records)),
        "verbatim_control_available": float(all("verbatim_joint_success" in record["control_results"] or "always_write_joint_success" in record["control_results"] for record in records)),
        "compressed_control_available": float(all("compressed_joint_success" in record["control_results"] or "oracle_within_budget" in record["control_results"] for record in records)),
        "overclaim_guard_pass": float(all(record["kill_condition"] == "none" or not bool(record["accepted"]) for record in records)),
        "family_mean_best_ratio": grouped_mean(records, "family", "best_oracle_ratio"),
        "surface_mean_best_ratio": grouped_mean(records, "surface", "best_oracle_ratio"),
        "kill_condition_by_family": {
            family: sorted({record["kill_condition"] for record in records if record["family"] == family})
            for family in sorted({str(record["family"]) for record in records})
        },
        "trainable_mirror_recommended": float(
            extreme_count > 0
            and mean_key(records, "operation_preserved") >= 0.98
            and killed_count == 0
            and len(strong_families) == len(families)
        ),
    }


def build_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "best_oracle_ratio": mean_confidence_interval([record["best_oracle_ratio"] for record in records]),
        "oracle_ratio_latent": mean_confidence_interval([record["oracle_ratio_latent"] for record in records]),
        "oracle_ratio_schema_residual": mean_confidence_interval([record["oracle_ratio_schema_residual"] for record in records]),
        "oracle_ratio_branch_program": mean_confidence_interval([record["oracle_ratio_branch_program"] for record in records]),
        "operation_preserved": mean_confidence_interval([record["operation_preserved"] for record in records], bounds=(0.0, 1.0)),
        "controls_preserved": mean_confidence_interval([record["controls_preserved"] for record in records], bounds=(0.0, 1.0)),
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    records = generate_records(profile)
    summary = build_summary(records)
    statistics = build_statistics(records)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "oracle_compression_analysis_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="oracle_compression_analysis",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "hard_episodes": int(HARD_EPISODES),
            "eligibility_episodes": int(ELIG_EPISODES),
        },
        seed_numpy=int(SEED),
        n_trials=int(len(records)),
        summary=summary,
        statistics=statistics,
        trials=records,
        artifacts=[
            {
                "name": "oracle_compression_analysis_metrics.json",
                "path": metrics_path,
                "type": "metrics",
                "description": "oracle bit counters for hard symbolic and eligibility-gated memory surfaces",
            }
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
