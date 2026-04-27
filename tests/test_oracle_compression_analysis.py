import sys
from pathlib import Path

from neuroloc.data.nm_worlds import ELIGIBILITY_COMMIT_FAMILIES, HARD_SYMBOLIC_FAMILIES
from neuroloc.simulations.memory.oracle_compression_analysis import build_summary, generate_records
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES, get_suite_specs
from neuroloc.simulations.suite_runner import run_specs
from neuroloc.simulations.shared import validate_metrics_file


def test_oracle_compression_smoke_suite(tmp_path) -> None:
    results = run_specs(
        specs=get_suite_specs("oracle_compression"),
        profile="smoke",
        output_root=tmp_path / "oracle_compression",
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
    assert summary["surface_count"] == 2
    assert summary["family_count"] == len(HARD_SYMBOLIC_FAMILIES) + len(ELIGIBILITY_COMMIT_FAMILIES)
    assert summary["episode_count"] == 8
    assert summary["contract_count"] == 56
    assert summary["operation_preservation_rate"] == 1.0
    assert summary["controls_preservation_rate"] == 1.0
    assert summary["leakage_free_rate"] == 1.0
    assert summary["no_memory_control_pass"] == 1.0
    assert summary["recency_control_pass"] == 1.0
    assert summary["shuffled_address_control_pass"] == 1.0
    assert summary["verbatim_control_available"] == 1.0
    assert summary["compressed_control_available"] == 1.0
    assert summary["overclaim_guard_pass"] == 1.0
    assert summary["strong_oracle_family_count"] > 0
    assert summary["weak_oracle_family_count"] > 0
    assert summary["kill_condition_count"] > 0
    assert summary["weak_oracle_ratio_count"] > 0
    assert summary["trainable_mirror_recommended"] == 0.0
    assert summary["imagination_branch_ratio_mean"] >= 50.0


def test_oracle_compression_records_are_deterministic_and_cover_families() -> None:
    first = generate_records("smoke")
    second = generate_records("smoke")
    assert first == second
    families = {record["family"] for record in first}
    assert families == set(HARD_SYMBOLIC_FAMILIES) | set(ELIGIBILITY_COMMIT_FAMILIES)
    surfaces = {record["surface"] for record in first}
    assert surfaces == {"hard_symbolic_nm", "eligibility_commit"}
    for record in first:
        assert record["profile"] == "smoke"
        assert record["verbatim_trace_bits"] >= record["latent_state_bits"] >= 1
        assert record["verbatim_trace_bits"] >= record["schema_residual_bits"] >= 1
        assert record["verbatim_trace_bits"] >= record["imagined_branch_program_bits"] >= 1
        assert record["operation_preserved"] == 1.0
        assert record["controls_preserved"] == 1.0
        assert record["leakage_free"] == 1.0
        assert not any(bool(value) for value in record["leakage_flags"].values())
        assert set(record["operation_flags"]) == {
            "state",
            "action",
            "joint",
            "address_separation",
            "replay_rewrite",
            "branch_reconstruction",
            "bounded_exposure",
        }
        if record["kill_condition"] == "none":
            assert record["accepted"] == 1.0
            assert record["kill_conditions_triggered"] == []
        else:
            assert record["accepted"] == 0.0
            assert record["kill_conditions_triggered"] == [record["kill_condition"]]


def test_oracle_compression_hard_suite_respects_registry_episode_counts(tmp_path) -> None:
    results = run_specs(
        specs=get_suite_specs("oracle_compression"),
        profile="hard",
        output_root=tmp_path / "oracle_compression",
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
    assert summary["episode_count"] == 64
    assert summary["contract_count"] == 448
    assert summary["accepted_rate"] == 4 / 7
    assert summary["kill_condition_count"] == 192
    assert 3.6 < summary["eligibility_commit_ratio_vs_always_write"] < 3.7
    assert summary["trainable_mirror_recommended"] == 0.0


def test_oracle_compression_summary_blocks_global_training_when_any_family_is_weak() -> None:
    records = generate_records("smoke")
    summary = build_summary(records)
    weak_families = [
        family
        for family, reasons in summary["kill_condition_by_family"].items()
        if reasons != ["none"]
    ]
    assert weak_families
    assert summary["trainable_mirror_recommended"] == 0.0
    assert summary["accepted_rate"] < 1.0
    assert summary["max_oracle_ratio"] >= 50.0
    assert summary["min_oracle_ratio"] < 10.0


def test_oracle_compression_registry_entries() -> None:
    assert "oracle_compression" in SUITES
    assert "oracle_compression_analysis" in SUITES["oracle_compression"]
    assert "oracle_compression_analysis" in SUITES["compression"]
    assert "oracle_compression_analysis" in SUITES["precompute"]
    spec = SIMULATION_SPECS["oracle_compression_analysis"]
    assert spec.category == "oracle_compression"
    assert spec.hard_env is not None
    assert spec.smoke_env["ORACLE_COMPRESSION_PROFILE"] == "smoke"
    assert spec.hard_env["ORACLE_COMPRESSION_PROFILE"] == "hard"
