from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from neuroloc.simulations.memory.compression_under_bit_budget_mirror import (
    BASELINE_POLICIES,
    build_dataset,
    build_summary,
    collect_forbidden_keys,
    evaluate_dataset,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES, get_suite_specs
from neuroloc.simulations.suite_runner import run_specs


def test_compression_mirror_dataset_is_deterministic_and_split_safe() -> None:
    first = build_dataset("smoke", seed=123, train_episodes=4, val_episodes=2, test_episodes=2)
    second = build_dataset("smoke", seed=123, train_episodes=4, val_episodes=2, test_episodes=2)
    assert first == second
    splits = {split: {row["seed"] for row in first if row["split"] == split} for split in ("train", "validation", "test")}
    assert len(splits["train"] & splits["validation"]) == 0
    assert len(splits["train"] & splits["test"]) == 0
    assert len(splits["validation"] & splits["test"]) == 0
    assert len(splits["train"]) == 4
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2


def test_compression_mirror_model_input_excludes_forbidden_fields() -> None:
    dataset = build_dataset("smoke", seed=124, train_episodes=2, val_episodes=1, test_episodes=1)
    assert all(row["family"] == "compression_under_bit_budget" for row in dataset)
    for row in dataset:
        assert row["forbidden_input_keys"] == []
        assert row["future_observation_violation_count"] == 0
        assert collect_forbidden_keys(row["model_input"]) == []
        assert max(event["time"] for event in row["model_input"]["observations"]) <= row["model_input"]["query"]["time"]
        assert "labels" in row
        assert "target" not in row["model_input"]
        assert "hidden_state" not in row["model_input"]


def test_compression_mirror_controls_and_bit_accounting() -> None:
    dataset = build_dataset("smoke", seed=125, train_episodes=4, val_episodes=2, test_episodes=4)
    rows = evaluate_dataset(dataset, "smoke", seed=125)
    summary = build_summary(dataset, rows)
    assert summary["family_count"] == 1
    assert summary["policy_count"] == len(BASELINE_POLICIES)
    assert summary["forbidden_input_violation_count"] == 0
    assert summary["future_observation_violation_count"] == 0
    assert summary["oracle_joint_success"] == 1.0
    assert summary["compressed_oracle_joint_success"] == 1.0
    assert summary["verbatim_joint_success"] == 1.0
    assert summary["no_memory_joint_success"] == 0.0
    assert summary["recency_only_joint_success"] == 0.0
    assert summary["shuffled_address_joint_success"] == 0.0
    assert summary["random_codebook_joint_success"] == 0.0
    assert summary["verbatim_within_budget"] == 0.0
    assert summary["compressed_oracle_within_budget"] == 1.0
    assert summary["compression_ratio_vs_verbatim"] > 1.0
    assert summary["paid_compute_authorized"] == 0.0
    assert summary["full_model_authorized"] == 0.0
    assert summary["blocked_authorization_violation_count"] == 0.0


def test_compression_mirror_evaluation_uses_record_contract() -> None:
    dataset = build_dataset("smoke", seed=129, train_episodes=1, val_episodes=1, test_episodes=1)
    dataset[0]["evaluation_contract"]["bit_budget"]["compressed_bits"] = 5
    rows = evaluate_dataset(dataset, "smoke", seed=129)
    matching = [
        row
        for row in rows
        if row["episode_id"] == dataset[0]["episode_id"] and row["policy"] == "compressed_oracle_store"
    ]
    assert len(matching) == 1
    assert matching[0]["total_committed_bits"] == 5


def test_compression_mirror_statistics_cover_every_policy() -> None:
    dataset = build_dataset("smoke", seed=126, train_episodes=2, val_episodes=1, test_episodes=1)
    rows = evaluate_dataset(dataset, "smoke", seed=126)
    policies = {row["policy"] for row in rows}
    assert policies == set(BASELINE_POLICIES)
    assert all("committed_bits_by_field" in row for row in rows)
    assert all("memory_output_vs_residual_norm" in row for row in rows)


def test_compression_mirror_registry_entries() -> None:
    assert "compression_under_bit_budget_mirror" in SIMULATION_SPECS
    assert "compression_mirror" in SUITES
    assert "compression_under_bit_budget_mirror" in SUITES["compression_mirror"]
    assert "compression_under_bit_budget_mirror" in SUITES["precompute"]
    spec = SIMULATION_SPECS["compression_under_bit_budget_mirror"]
    assert spec.category == "compression_mirror"
    maximums = dict(spec.maximum_summary_values)
    assert maximums["full_model_authorized"] == 0.0
    assert maximums["paid_compute_authorized"] == 0.0
    assert maximums["future_observation_violation_count"] == 0.0


def test_compression_mirror_smoke_suite(tmp_path: Path) -> None:
    results = run_specs(
        specs=get_suite_specs("compression_mirror"),
        profile="smoke",
        output_root=tmp_path / "compression_mirror",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [(result.simulation_id, result.validation_error, result.stderr_tail) for result in results if not result.ok]
    assert not failures, failures
    metrics_path = Path(results[0].metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["summary"]["forbidden_input_violation_count"] == 0
    assert payload["summary"]["future_observation_violation_count"] == 0
    assert payload["summary"]["local_mirror_code_authorized"] == 1.0
    assert payload["summary"]["paid_compute_authorized"] == 0.0


def test_compression_mirror_invalid_profile_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown hard symbolic profile"):
        build_dataset("bad", seed=127, train_episodes=1, val_episodes=1, test_episodes=1)
