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
    ELIGIBILITY_COMMIT_FAMILIES,
    ELIGIBILITY_COMMIT_POLICIES,
    evaluate_eligibility_gated_local_commit_episode,
    generate_eligibility_gated_local_commit_batch,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("ELIG_COMMIT_SEED", 42)
EPISODES = env_int("ELIG_COMMIT_EPISODES", 6)

require_positive("ELIG_COMMIT_EPISODES", EPISODES)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    env_value = os.environ.get("ELIG_COMMIT_PROFILE", "smoke").strip()
    if env_value not in {"smoke", "hard"}:
        raise ValueError("ELIG_COMMIT_PROFILE must be smoke or hard")
    return env_value


def mean_for(rows: list[dict[str, Any]], policy: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["policy"] == policy]
    return float(np.mean(values)) if values else 0.0


def family_policy_mean(rows: list[dict[str, Any]], family: str, policy: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["family"] == family and row["policy"] == policy]
    return float(np.mean(values)) if values else 0.0


def summarize_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    policy_rows = [row for row in rows if row["policy"] == policy]
    return {
        "mark_correct": mean_confidence_interval([row["mark_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "commit_correct": mean_confidence_interval([row["commit_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "read_correct": mean_confidence_interval([row["read_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "exposure_correct": mean_confidence_interval([row["exposure_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "state_probe_accuracy": mean_confidence_interval([row["state_probe_accuracy"] for row in policy_rows], bounds=(0.0, 1.0)),
        "action_success": mean_confidence_interval([row["action_success"] for row in policy_rows], bounds=(0.0, 1.0)),
        "joint_success": mean_confidence_interval([row["joint_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "delayed_use_success": mean_confidence_interval([row["delayed_use_success"] for row in policy_rows], bounds=(0.0, 1.0)),
    }


def leakage_violation_count(episodes: list[dict[str, Any]]) -> int:
    violations = 0
    for episode in episodes:
        observations = episode["observation_stream"]
        for contract in episode["contracts"]:
            for failed in contract["leakage_checks"].values():
                violations += int(bool(failed))
            query = contract["query"]
            target = contract["target"]["state"]
            time_idx = int(query["time"])
            object_idx = int(query["focus_local_index"])
            violations += int(int(observations["color"][time_idx, object_idx]) == int(target["color"]))
            violations += int(int(observations["shape"][time_idx, object_idx]) == int(target["shape"]))
            violations += int(int(observations["pos"][time_idx, object_idx]) == int(target["pos"]))
            for relevance in contract["relevance_events"]:
                violations += int(bool(relevance["names_answer"]))
                violations += int(bool(relevance["names_target_identity"]))
                violations += int(bool(relevance["names_unique_candidate_index"]))
    return int(violations)


def contract_field_violation_count(episodes: list[dict[str, Any]]) -> int:
    required = {
        "episode_id",
        "seed",
        "family",
        "profile",
        "hidden_state",
        "observation_stream",
        "query",
        "target",
        "candidate_events",
        "relevance_events",
        "commit_targets",
        "read_queries",
        "exposure_targets",
        "memory_relevant_positions",
        "distractor_positions",
        "negative_commit_positions",
        "trace_eligible_positions",
        "commit_positions",
        "exposure_positions",
        "difficulty",
        "bit_budget",
        "output_budget",
        "oracle_codes",
        "expected",
        "telemetry",
        "leakage_checks",
        "kill_conditions",
    }
    violations = 0
    for episode in episodes:
        for contract in episode["contracts"]:
            violations += len(required - set(contract))
            for policy in ELIGIBILITY_COMMIT_POLICIES:
                if policy not in contract["expected"]:
                    violations += 1
    return int(violations)


def build_summary(episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_joint = mean_for(rows, "oracle", "joint_correct")
    no_memory_joint = mean_for(rows, "no_memory", "joint_correct")
    recency_joint = mean_for(rows, "recency_only", "joint_correct")
    shuffled_joint = mean_for(rows, "shuffled_address", "joint_correct")
    random_trace_commit = mean_for(rows, "random_trace", "commit_f1")
    no_trace_joint = mean_for(rows, "no_trace", "joint_correct")
    oracle_commit_oracle_exposure_joint = mean_for(rows, "oracle_commit_oracle_exposure", "joint_correct")
    oracle_mark_no_commit_joint = mean_for(rows, "oracle_mark_no_commit", "joint_correct")
    no_commit_oracle_exposure_joint = mean_for(rows, "no_commit_oracle_exposure", "joint_correct")
    hand_opened_joint = mean_for(rows, "hand_opened_exposure", "joint_correct")
    always_unlimited_joint = mean_for(rows, "always_commit_unlimited", "joint_correct")
    always_unlimited_within = mean_for(rows, "always_commit_unlimited", "within_commit_budget")
    always_matched_joint = mean_for(rows, "always_commit_matched_budget", "joint_correct")
    residual_joint = mean_for(rows, "matched_residual_capacity", "joint_correct")
    compute_joint = mean_for(rows, "matched_compute_budget", "joint_correct")
    fixed_closed_action = mean_for(rows, "fixed_closed_exposure", "action_correct")
    fixed_open_joint = mean_for(rows, "fixed_open_exposure", "joint_correct")
    oracle_bits = mean_for(rows, "oracle", "bits_committed")
    always_bits = mean_for(rows, "always_commit_unlimited", "bits_committed")
    useful_bits = float(np.mean([contract["bit_budget"]["task_relevant_bits"] for episode in episodes for contract in episode["contracts"]]))
    verbatim_bits = float(np.mean([contract["bit_budget"]["verbatim_trace_bits"] for episode in episodes for contract in episode["contracts"]]))
    always_write_bits = float(np.mean([contract["bit_budget"]["always_write_bits"] for episode in episodes for contract in episode["contracts"]]))
    eligibility_bits = float(np.mean([contract["bit_budget"]["eligibility_commit_bits"] for episode in episodes for contract in episode["contracts"]]))
    oracle_writes = float(np.mean([len(contract["commit_positions"]) for episode in episodes for contract in episode["contracts"]]))
    compression_saving = float(1.0 - (oracle_bits / max(always_bits, 1.0)))
    no_memory_control_gap = float(oracle_joint - no_memory_joint)
    recency_control_gap = float(oracle_joint - recency_joint)
    shuffled_control_gap = float(oracle_joint - shuffled_joint)
    random_trace_control_margin = float(0.25 - random_trace_commit)
    fixed_closed_control_gap = float(oracle_joint - fixed_closed_action)
    fixed_open_control_gap = float(oracle_joint - fixed_open_joint)
    residual_control_gap = float(oracle_joint - residual_joint)
    compute_control_gap = float(oracle_joint - compute_joint)
    no_memory_ceiling_pass = float(no_memory_joint <= 0.0)
    recency_ceiling_pass = float(recency_joint <= 0.0)
    shuffled_address_ceiling_pass = float(shuffled_joint <= 0.0)
    random_trace_ceiling_pass = float(random_trace_commit <= 0.25)
    always_commit_unlimited_budget_fail_pass = float(always_unlimited_within <= 0.0)
    always_commit_matched_ceiling_pass = float(always_matched_joint <= 0.0)
    oracle_mark_no_commit_ceiling_pass = float(oracle_mark_no_commit_joint <= 0.0)
    no_commit_oracle_exposure_ceiling_pass = float(no_commit_oracle_exposure_joint <= 0.0)
    hand_opened_exposure_ceiling_pass = float(hand_opened_joint <= 0.0)
    fixed_closed_exposure_ceiling_pass = float(fixed_closed_action <= 0.0)
    fixed_open_exposure_ceiling_pass = float(fixed_open_joint <= 0.0)
    matched_residual_capacity_ceiling_pass = float(residual_joint <= 0.0)
    matched_compute_budget_ceiling_pass = float(compute_joint <= 0.0)
    return {
        "family_count": int(len(ELIGIBILITY_COMMIT_FAMILIES)),
        "policy_count": int(len(ELIGIBILITY_COMMIT_POLICIES)),
        "episode_count": int(len(episodes)),
        "state_probe_accuracy": float(mean_for(rows, "oracle", "state_probe_accuracy")),
        "action_success": float(mean_for(rows, "oracle", "action_success")),
        "joint_success": float(oracle_joint),
        "delayed_use_success": float(mean_for(rows, "oracle", "delayed_use_success")),
        "exact_recall": float(mean_for(rows, "oracle", "exact_recall")),
        "oracle_joint_success": float(oracle_joint),
        "oracle_commit_oracle_exposure_joint_success": float(oracle_commit_oracle_exposure_joint),
        "no_memory_joint_success": float(no_memory_joint),
        "recency_only_joint_success": float(recency_joint),
        "shuffled_address_joint_success": float(shuffled_joint),
        "random_trace_commit_f1": float(random_trace_commit),
        "no_trace_joint_success": float(no_trace_joint),
        "oracle_mark_no_commit_joint_success": float(oracle_mark_no_commit_joint),
        "no_commit_oracle_exposure_joint_success": float(no_commit_oracle_exposure_joint),
        "hand_opened_exposure_joint_success": float(hand_opened_joint),
        "always_commit_unlimited_joint_success": float(always_unlimited_joint),
        "always_commit_unlimited_within_budget": float(always_unlimited_within),
        "always_commit_matched_budget_joint_success": float(always_matched_joint),
        "matched_residual_capacity_joint_success": float(residual_joint),
        "matched_compute_budget_joint_success": float(compute_joint),
        "fixed_closed_exposure_action_success": float(fixed_closed_action),
        "fixed_open_exposure_joint_success": float(fixed_open_joint),
        "no_memory_control_gap": no_memory_control_gap,
        "recency_control_gap": recency_control_gap,
        "shuffled_address_control_gap": shuffled_control_gap,
        "random_trace_control_margin": random_trace_control_margin,
        "fixed_closed_exposure_control_gap": fixed_closed_control_gap,
        "fixed_open_exposure_control_gap": fixed_open_control_gap,
        "matched_residual_capacity_control_gap": residual_control_gap,
        "matched_compute_budget_control_gap": compute_control_gap,
        "no_memory_ceiling_pass": no_memory_ceiling_pass,
        "recency_only_ceiling_pass": recency_ceiling_pass,
        "shuffled_address_ceiling_pass": shuffled_address_ceiling_pass,
        "random_trace_ceiling_pass": random_trace_ceiling_pass,
        "always_commit_unlimited_budget_fail_pass": always_commit_unlimited_budget_fail_pass,
        "always_commit_matched_budget_ceiling_pass": always_commit_matched_ceiling_pass,
        "oracle_mark_no_commit_ceiling_pass": oracle_mark_no_commit_ceiling_pass,
        "no_commit_oracle_exposure_ceiling_pass": no_commit_oracle_exposure_ceiling_pass,
        "hand_opened_exposure_ceiling_pass": hand_opened_exposure_ceiling_pass,
        "fixed_closed_exposure_ceiling_pass": fixed_closed_exposure_ceiling_pass,
        "fixed_open_exposure_ceiling_pass": fixed_open_exposure_ceiling_pass,
        "matched_residual_capacity_ceiling_pass": matched_residual_capacity_ceiling_pass,
        "matched_compute_budget_ceiling_pass": matched_compute_budget_ceiling_pass,
        "mark_success": float(mean_for(rows, "oracle", "mark_correct")),
        "commit_success": float(mean_for(rows, "oracle", "commit_correct")),
        "read_success": float(mean_for(rows, "oracle", "read_correct")),
        "exposure_success": float(mean_for(rows, "oracle", "exposure_correct")),
        "trace_precision": float(mean_for(rows, "oracle", "trace_precision")),
        "trace_recall": float(mean_for(rows, "oracle", "trace_recall")),
        "write_precision": float(mean_for(rows, "oracle", "write_precision")),
        "write_recall": float(mean_for(rows, "oracle", "write_recall")),
        "commit_f1": float(mean_for(rows, "oracle", "commit_f1")),
        "commit_latency": float(mean_for(rows, "oracle", "commit_latency")),
        "false_commit_rate": float(mean_for(rows, "oracle", "false_commit_rate")),
        "negative_commit_rejection_rate": float(mean_for(rows, "oracle", "negative_commit_rejection_rate")),
        "output_capacity_precision": float(mean_for(rows, "oracle", "output_capacity_precision")),
        "output_capacity_recall": float(mean_for(rows, "oracle", "output_capacity_recall")),
        "oracle_trace_learned_commit_gap": float(mean_for(rows, "oracle", "joint_correct") - mean_for(rows, "oracle_trace_learned_commit", "joint_correct")),
        "learned_trace_oracle_commit_gap": float(mean_for(rows, "oracle", "joint_correct") - mean_for(rows, "learned_trace_oracle_commit", "joint_correct")),
        "oracle_commit_learned_exposure_gap": float(mean_for(rows, "oracle", "joint_correct") - mean_for(rows, "oracle_commit_learned_exposure", "joint_correct")),
        "learned_commit_oracle_exposure_gap": float(mean_for(rows, "oracle", "joint_correct") - mean_for(rows, "learned_commit_oracle_exposure", "joint_correct")),
        "bits_committed_per_successful_episode": float(oracle_bits / max(oracle_joint, 1e-9)),
        "writes_per_successful_episode": float(oracle_writes / max(oracle_joint, 1e-9)),
        "verbatim_trace_bits": float(verbatim_bits),
        "always_write_bits": float(always_write_bits),
        "eligibility_commit_bits": float(eligibility_bits),
        "task_relevant_bits": float(useful_bits),
        "useful_bits_fraction": float(useful_bits / max(oracle_bits, 1.0)),
        "commit_compression_saving_vs_always": float(compression_saving),
        "memory_output_vs_residual_norm": float(mean_for(rows, "oracle", "memory_output_norm") / max(mean_for(rows, "oracle", "residual_norm"), 1e-9)),
        "exposure_noise_cost": float(mean_for(rows, "oracle", "exposure_noise_cost")),
        "bounded_exposure_noise_penalty": float(mean_for(rows, "fixed_open_exposure", "exposure_noise_cost") - mean_for(rows, "oracle", "exposure_noise_cost")),
        "leakage_violation_count": int(leakage_violation_count(episodes)),
        "contract_field_violation_count": int(contract_field_violation_count(episodes)),
        "delayed_relevance_oracle_joint": float(family_policy_mean(rows, "delayed_relevance_local_commit", "oracle", "joint_correct")),
        "bounded_output_oracle_joint": float(family_policy_mean(rows, "bounded_output_exposure", "oracle", "joint_correct")),
        "crossed_split_oracle_joint": float(family_policy_mean(rows, "crossed_commit_exposure_split", "oracle", "joint_correct")),
        "compression_frontier_oracle_joint": float(family_policy_mean(rows, "commit_compression_frontier", "oracle", "joint_correct")),
    }


def build_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        policy: summarize_policy(rows, policy)
        for policy in ELIGIBILITY_COMMIT_POLICIES
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    episodes = generate_eligibility_gated_local_commit_batch(EPISODES, seed=SEED, profile=profile)
    rows = []
    for episode in episodes:
        rows.extend(evaluate_eligibility_gated_local_commit_episode(episode))
    summary = build_summary(episodes, rows)
    statistics = build_statistics(rows)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eligibility_gated_local_commit_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="eligibility_gated_local_commit",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "episodes": int(EPISODES),
            "families": list(ELIGIBILITY_COMMIT_FAMILIES),
            "policies": list(ELIGIBILITY_COMMIT_POLICIES),
        },
        seed_numpy=int(SEED),
        n_trials=int(len(rows)),
        summary=summary,
        statistics=statistics,
        trials=rows,
        artifacts=[
            {
                "name": "eligibility_gated_local_commit_metrics.json",
                "path": metrics_path,
                "type": "metrics",
                "description": "symbolic/oracle eligibility-gated local commit contract metrics",
            }
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
