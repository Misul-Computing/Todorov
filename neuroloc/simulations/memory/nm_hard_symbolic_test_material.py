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
    HARD_SYMBOLIC_FAMILIES,
    HARD_SYMBOLIC_POLICIES,
    evaluate_nm_hard_symbolic_episode,
    generate_nm_hard_symbolic_batch,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("NM_HARD_SYMBOLIC_SEED", 42)
EPISODES = env_int("NM_HARD_SYMBOLIC_EPISODES", 8)

require_positive("NM_HARD_SYMBOLIC_EPISODES", EPISODES)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    env_value = os.environ.get("NM_HARD_SYMBOLIC_PROFILE", "smoke").strip()
    if env_value not in {"smoke", "hard"}:
        raise ValueError("NM_HARD_SYMBOLIC_PROFILE must be smoke or hard")
    return env_value


def summarize_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    policy_rows = [row for row in rows if row["policy"] == policy]
    return {
        "state_probe_accuracy": mean_confidence_interval([row["state_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "action_success": mean_confidence_interval([row["action_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "joint_success": mean_confidence_interval([row["joint_correct"] for row in policy_rows], bounds=(0.0, 1.0)),
        "exact_recall": mean_confidence_interval([row["exact_recall"] for row in policy_rows], bounds=(0.0, 1.0)),
    }


def mean_for(rows: list[dict[str, Any]], policy: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["policy"] == policy]
    return float(np.mean(values)) if values else 0.0


def family_policy_mean(rows: list[dict[str, Any]], family: str, policy: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["family"] == family and row["policy"] == policy]
    return float(np.mean(values)) if values else 0.0


def leakage_violation_count(episodes: list[dict[str, Any]]) -> int:
    violations = 0
    for episode in episodes:
        observations = episode["observation_stream"]
        hidden = episode["hidden_state"]
        for contract in episode["contracts"]:
            query = contract["query"]
            target = contract["target"]["state"]
            time_idx = int(query["time"])
            object_idx = int(query["focus_local_index"])
            if int(observations["color"][time_idx, object_idx]) == int(target["color"]):
                violations += 1
            if int(observations["shape"][time_idx, object_idx]) == int(target["shape"]):
                violations += 1
            if int(observations["pos"][time_idx, object_idx]) == int(hidden["positions"][time_idx, object_idx]):
                violations += 1
    return int(violations)


def contract_field_violation_count(episodes: list[dict[str, Any]]) -> int:
    required_contract_keys = {
        "family",
        "query",
        "target",
        "memory_relevant_positions",
        "distractor_positions",
        "difficulty",
        "bit_budget",
        "expected",
        "telemetry",
    }
    violations = 0
    for episode in episodes:
        for contract in episode["contracts"]:
            violations += len(required_contract_keys - set(contract))
            for policy in HARD_SYMBOLIC_POLICIES:
                if policy not in contract["expected"]:
                    violations += 1
    return int(violations)


def rollout_curves(episodes: list[dict[str, Any]]) -> dict[str, float]:
    contracts = [
        contract
        for episode in episodes
        for contract in episode["contracts"]
        if contract["family"] == "iterative_hard_case_rollout"
    ]
    easy_no = float(np.mean([contract["difficulty"]["easy_no_rollout"] for contract in contracts]))
    easy_iterative = float(np.mean([contract["difficulty"]["easy_iterative"] for contract in contracts]))
    hard_no = float(np.mean([contract["difficulty"]["hard_no_rollout"] for contract in contracts]))
    hard_iterative = float(np.mean([contract["difficulty"]["hard_iterative"] for contract in contracts]))
    return {
        "easy_case_pre_rollout_success": easy_no,
        "hard_case_pre_rollout_success": hard_no,
        "hard_case_pre_rollout_gap": float(easy_no - hard_no),
        "easy_case_rollout_gain": float(easy_iterative - easy_no),
        "hard_case_rollout_gain": float(hard_iterative - hard_no),
        "hard_minus_easy_rollout_gain": float((hard_iterative - hard_no) - (easy_iterative - easy_no)),
    }


def compression_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    compressed_success = family_policy_mean(rows, "compression_under_bit_budget", "compressed_store", "joint_correct")
    verbatim_success = family_policy_mean(rows, "compression_under_bit_budget", "verbatim_store", "joint_correct")
    compressed_bits = mean_for(
        [row for row in rows if row["family"] == "compression_under_bit_budget"],
        "compressed_store",
        "bits_written",
    )
    verbatim_bits = mean_for(
        [row for row in rows if row["family"] == "compression_under_bit_budget"],
        "verbatim_store",
        "bits_written",
    )
    return {
        "compression_success_gap_vs_verbatim": float(compressed_success - verbatim_success),
        "compression_bit_saving_fraction": float(1.0 - (compressed_bits / max(verbatim_bits, 1.0))),
        "bits_written_per_successful_episode": float(compressed_bits / max(compressed_success, 1e-9)),
        "verbatim_within_budget": float(mean_for(
            [row for row in rows if row["family"] == "compression_under_bit_budget"],
            "verbatim_store",
            "within_budget",
        )),
        "compressed_within_budget": float(mean_for(
            [row for row in rows if row["family"] == "compression_under_bit_budget"],
            "compressed_store",
            "within_budget",
        )),
    }


def telemetry_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    oracle_rows = [row for row in rows if row["policy"] == "oracle"]
    return {
        "gate_open_fraction": mean_for(rows, "oracle", "gate_open_fraction"),
        "memory_output_vs_residual_norm": float(
            mean_for(rows, "oracle", "memory_output_norm")
            / max(mean_for(rows, "oracle", "residual_norm"), 1e-9)
        ),
        "slot_address_entropy": float(np.mean([row["slot_entropy"] for row in oracle_rows])),
        "address_margin": mean_for(rows, "oracle", "address_margin"),
        "write_frequency": float(np.mean([row["write_frequency"] for row in oracle_rows])),
        "read_concentration": mean_for(rows, "oracle", "read_concentration"),
        "retention_over_delay": float(np.mean([row["retention_over_delay"] for row in oracle_rows])),
        "compression_budget": float(np.mean([row["compression_budget"] for row in oracle_rows])),
        "reconstruction_error": mean_for(rows, "oracle", "reconstruction_error"),
    }


def build_summary(episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_metrics = {policy: summarize_policy(rows, policy) for policy in HARD_SYMBOLIC_POLICIES}
    curves = rollout_curves(episodes)
    compression = compression_metrics(rows)
    telemetry = telemetry_summary(rows)
    return {
        "family_count": int(len(HARD_SYMBOLIC_FAMILIES)),
        "policy_count": int(len(HARD_SYMBOLIC_POLICIES)),
        "episode_count": int(len(episodes)),
        "state_probe_accuracy": float(policy_metrics["oracle"]["state_probe_accuracy"]["mean"]),
        "action_success": float(policy_metrics["oracle"]["action_success"]["mean"]),
        "joint_success": float(policy_metrics["oracle"]["joint_success"]["mean"]),
        "exact_recall": float(policy_metrics["oracle"]["exact_recall"]["mean"]),
        "degraded_cue_recall": float(family_policy_mean(rows, "associative_recall", "oracle", "exact_recall")),
        "oracle_joint_success": float(policy_metrics["oracle"]["joint_success"]["mean"]),
        "no_memory_joint_success": float(policy_metrics["no_memory"]["joint_success"]["mean"]),
        "recency_only_joint_success": float(policy_metrics["recency_only"]["joint_success"]["mean"]),
        "shuffled_address_joint_success": float(policy_metrics["shuffled_address"]["joint_success"]["mean"]),
        "random_replay_joint_success": float(policy_metrics["random_replay"]["joint_success"]["mean"]),
        "targeted_replay_joint_success": float(policy_metrics["targeted_replay"]["joint_success"]["mean"]),
        "targeted_replay_gain": float(
            family_policy_mean(rows, "replay_rewrite", "targeted_replay", "joint_correct")
            - family_policy_mean(rows, "replay_rewrite", "random_replay", "joint_correct")
        ),
        "interference_slope": float(
            family_policy_mean(rows, "correlated_key_interference", "oracle", "joint_correct")
            - family_policy_mean(rows, "correlated_key_interference", "shuffled_address", "joint_correct")
        ),
        "reuse_advantage": float(
            family_policy_mean(rows, "episodic_reuse_after_distractors", "targeted_replay", "joint_correct")
            - family_policy_mean(rows, "episodic_reuse_after_distractors", "random_replay", "joint_correct")
        ),
        "hard_case_rollout_gain": float(curves["hard_case_rollout_gain"]),
        "hard_case_pre_rollout_gap": float(curves["hard_case_pre_rollout_gap"]),
        "hard_minus_easy_rollout_gain": float(curves["hard_minus_easy_rollout_gain"]),
        "leakage_violation_count": int(leakage_violation_count(episodes)),
        "contract_field_violation_count": int(contract_field_violation_count(episodes)),
        **compression,
        **telemetry,
        "policy_metrics": policy_metrics,
    }


def build_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        policy: summarize_policy(rows, policy)
        for policy in HARD_SYMBOLIC_POLICIES
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    episodes = generate_nm_hard_symbolic_batch(EPISODES, seed=SEED, profile=profile)
    rows = []
    for episode in episodes:
        rows.extend(evaluate_nm_hard_symbolic_episode(episode))
    summary = build_summary(episodes, rows)
    statistics = build_statistics(rows)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "nm_hard_symbolic_test_material_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="nm_hard_symbolic_test_material",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "episodes": int(EPISODES),
            "families": list(HARD_SYMBOLIC_FAMILIES),
            "policies": list(HARD_SYMBOLIC_POLICIES),
        },
        seed_numpy=int(SEED),
        n_trials=int(len(rows)),
        summary=summary,
        statistics=statistics,
        trials=rows,
        artifacts=[
            {
                "name": "nm_hard_symbolic_test_material_metrics.json",
                "path": metrics_path,
                "type": "json",
            }
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
