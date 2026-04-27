import sys
from pathlib import Path

from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES, get_suite_specs
from neuroloc.simulations.suite_runner import run_specs
from neuroloc.simulations.shared import validate_metrics_file


def test_eligibility_commit_smoke_suite(tmp_path) -> None:
    results = run_specs(
        specs=get_suite_specs("eligibility_commit"),
        profile="smoke",
        output_root=tmp_path / "eligibility_commit",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [
        (result.simulation_id, result.validation_error, result.stderr_tail)
        for result in results
        if not result.ok
    ]
    assert not failures, failures
    payload = validate_metrics_file(Path(results[0].metrics_path))
    summary = payload["summary"]
    assert summary["episode_count"] == 4
    assert summary["state_probe_accuracy"] >= 0.98
    assert summary["action_success"] >= 0.98
    assert summary["joint_success"] >= 0.98
    assert summary["delayed_use_success"] >= 0.98
    assert summary["exact_recall"] >= 0.98
    assert summary["oracle_joint_success"] >= 0.98
    assert summary["oracle_commit_oracle_exposure_joint_success"] >= 0.98
    assert summary["no_memory_joint_success"] <= 0.0
    assert summary["recency_only_joint_success"] <= 0.0
    assert summary["shuffled_address_joint_success"] <= 0.0
    assert summary["random_trace_commit_f1"] <= 0.25
    assert summary["no_trace_joint_success"] <= 0.0
    assert summary["oracle_mark_no_commit_joint_success"] <= 0.0
    assert summary["no_commit_oracle_exposure_joint_success"] <= 0.0
    assert summary["hand_opened_exposure_joint_success"] <= 0.0
    assert summary["always_commit_unlimited_joint_success"] == 1.0
    assert summary["always_commit_unlimited_within_budget"] == 0.0
    assert summary["always_commit_matched_budget_joint_success"] == 0.0
    assert summary["matched_residual_capacity_joint_success"] == 0.0
    assert summary["matched_compute_budget_joint_success"] == 0.0
    assert summary["fixed_closed_exposure_action_success"] == 0.0
    assert summary["fixed_open_exposure_joint_success"] == 0.0
    assert summary["no_memory_control_gap"] >= 0.98
    assert summary["recency_control_gap"] >= 0.98
    assert summary["shuffled_address_control_gap"] >= 0.98
    assert summary["random_trace_control_margin"] > 0.0
    assert summary["fixed_closed_exposure_control_gap"] >= 0.98
    assert summary["fixed_open_exposure_control_gap"] >= 0.98
    assert summary["matched_residual_capacity_control_gap"] >= 0.98
    assert summary["matched_compute_budget_control_gap"] >= 0.98
    assert summary["no_memory_ceiling_pass"] == 1.0
    assert summary["recency_only_ceiling_pass"] == 1.0
    assert summary["shuffled_address_ceiling_pass"] == 1.0
    assert summary["random_trace_ceiling_pass"] == 1.0
    assert summary["always_commit_unlimited_budget_fail_pass"] == 1.0
    assert summary["always_commit_matched_budget_ceiling_pass"] == 1.0
    assert summary["oracle_mark_no_commit_ceiling_pass"] == 1.0
    assert summary["no_commit_oracle_exposure_ceiling_pass"] == 1.0
    assert summary["hand_opened_exposure_ceiling_pass"] == 1.0
    assert summary["fixed_closed_exposure_ceiling_pass"] == 1.0
    assert summary["fixed_open_exposure_ceiling_pass"] == 1.0
    assert summary["matched_residual_capacity_ceiling_pass"] == 1.0
    assert summary["matched_compute_budget_ceiling_pass"] == 1.0
    assert summary["oracle_trace_learned_commit_gap"] > 0.0
    assert summary["learned_trace_oracle_commit_gap"] > 0.0
    assert summary["oracle_commit_learned_exposure_gap"] > 0.0
    assert summary["learned_commit_oracle_exposure_gap"] > 0.0
    assert summary["commit_compression_saving_vs_always"] > 0.0
    assert summary["writes_per_successful_episode"] > 0.0
    assert summary["commit_latency"] > 0.0
    assert summary["false_commit_rate"] == 0.0
    assert summary["verbatim_trace_bits"] > summary["always_write_bits"]
    assert summary["always_write_bits"] > summary["eligibility_commit_bits"]
    assert summary["task_relevant_bits"] > 0.0
    assert summary["memory_output_vs_residual_norm"] > 0.0
    assert summary["exposure_noise_cost"] == 0.0
    assert summary["bounded_exposure_noise_penalty"] > 0.0
    assert summary["leakage_violation_count"] == 0
    assert summary["contract_field_violation_count"] == 0
    assert summary["delayed_relevance_oracle_joint"] >= 0.98
    assert summary["bounded_output_oracle_joint"] >= 0.98
    assert summary["crossed_split_oracle_joint"] >= 0.98
    assert summary["compression_frontier_oracle_joint"] >= 0.98


def test_eligibility_commit_hard_suite(tmp_path) -> None:
    results = run_specs(
        specs=get_suite_specs("eligibility_commit"),
        profile="hard",
        output_root=tmp_path / "eligibility_commit",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [
        (result.simulation_id, result.validation_error, result.stderr_tail)
        for result in results
        if not result.ok
    ]
    assert not failures, failures
    payload = validate_metrics_file(Path(results[0].metrics_path))
    summary = payload["summary"]
    assert summary["episode_count"] == 32
    assert summary["family_count"] == 4
    assert summary["policy_count"] == 20
    assert summary["leakage_violation_count"] == 0
    assert summary["contract_field_violation_count"] == 0
    assert summary["no_memory_control_gap"] >= 0.98
    assert summary["recency_control_gap"] >= 0.98
    assert summary["shuffled_address_control_gap"] >= 0.98
    assert summary["fixed_closed_exposure_control_gap"] >= 0.98
    assert summary["fixed_open_exposure_control_gap"] >= 0.98
    assert summary["matched_residual_capacity_control_gap"] >= 0.98
    assert summary["matched_compute_budget_control_gap"] >= 0.98
    assert summary["no_memory_ceiling_pass"] == 1.0
    assert summary["recency_only_ceiling_pass"] == 1.0
    assert summary["shuffled_address_ceiling_pass"] == 1.0
    assert summary["random_trace_ceiling_pass"] == 1.0
    assert summary["always_commit_unlimited_budget_fail_pass"] == 1.0
    assert summary["always_commit_matched_budget_ceiling_pass"] == 1.0
    assert summary["oracle_mark_no_commit_ceiling_pass"] == 1.0
    assert summary["no_commit_oracle_exposure_ceiling_pass"] == 1.0
    assert summary["hand_opened_exposure_ceiling_pass"] == 1.0
    assert summary["fixed_closed_exposure_ceiling_pass"] == 1.0
    assert summary["fixed_open_exposure_ceiling_pass"] == 1.0
    assert summary["matched_residual_capacity_ceiling_pass"] == 1.0
    assert summary["matched_compute_budget_ceiling_pass"] == 1.0


def test_eligibility_commit_registry_entries() -> None:
    assert "eligibility_commit" in SUITES
    assert "eligibility_gated_local_commit" in SUITES["eligibility_commit"]
    assert "eligibility_gated_local_commit" in SUITES["precompute"]
    spec = SIMULATION_SPECS["eligibility_gated_local_commit"]
    assert spec.category == "eligibility_commit"
    assert spec.hard_env is not None
    assert spec.hard_env["ELIG_COMMIT_PROFILE"] == "hard"
