import sys
from pathlib import Path

from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES, get_suite_specs
from neuroloc.simulations.suite_runner import run_specs
from neuroloc.simulations.shared import validate_metrics_file


def test_hard_symbolic_nm_smoke_suite(tmp_path) -> None:
    results = run_specs(
        specs=get_suite_specs("hard_symbolic_nm"),
        profile="smoke",
        output_root=tmp_path / "hard_symbolic_nm",
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
    assert summary["oracle_joint_success"] >= 0.98
    assert summary["no_memory_joint_success"] <= 0.0
    assert summary["recency_only_joint_success"] <= 0.0
    assert summary["shuffled_address_joint_success"] <= 0.0
    assert summary["leakage_violation_count"] == 0
    assert summary["contract_field_violation_count"] == 0
    assert summary["compression_bit_saving_fraction"] > 0.0
    assert summary["verbatim_within_budget"] == 0.0
    assert summary["compressed_within_budget"] == 1.0
    assert summary["hard_case_pre_rollout_gap"] > 0.0
    assert summary["hard_minus_easy_rollout_gain"] > 0.0
    assert summary["memory_output_vs_residual_norm"] > 0.0
    assert summary["slot_address_entropy"] > 0.0
    assert summary["write_frequency"] > 0.0
    assert summary["retention_over_delay"] > 0.0
    assert summary["compression_budget"] > 0.0


def test_hard_symbolic_nm_hard_suite(tmp_path) -> None:
    results = run_specs(
        specs=get_suite_specs("hard_symbolic_nm"),
        profile="hard",
        output_root=tmp_path / "hard_symbolic_nm",
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
    assert payload["summary"]["episode_count"] == 32


def test_hard_symbolic_registry_entries() -> None:
    assert "hard_symbolic_nm" in SUITES
    assert "nm_hard_symbolic_test_material" in SUITES["hard_symbolic_nm"]
    spec = SIMULATION_SPECS["nm_hard_symbolic_test_material"]
    assert spec.category == "hard_symbolic_nm"
    assert spec.hard_env is not None
    assert spec.hard_env["NM_HARD_SYMBOLIC_PROFILE"] == "hard"
