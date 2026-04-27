from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from neuroloc.simulations.memory.compression_under_bit_budget_mirror import (
    ALL_POLICIES,
    BASELINE_POLICIES,
    DIAGNOSTIC_POLICIES,
    build_dataset,
    build_summary,
    collect_forbidden_keys,
    evaluate_dataset,
    learned_bits_by_field,
    oracle_code_fields,
    train_learned_codec,
    profile_caps,
    action_ambiguity_rate,
    source_observation_audit,
    source_observation_code_fields,
    source_event_for_record,
    source_signature_for_action,
    vectorize_record,
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


def test_compression_mirror_vector_features_ignore_label_fields() -> None:
    dataset = build_dataset("smoke", seed=131, train_episodes=1, val_episodes=1, test_episodes=1)
    caps = profile_caps("smoke")
    original = vectorize_record(dataset[0], caps)
    dataset[0]["labels"]["budget_bits"] += 1000
    dataset[0]["labels"]["verbatim_bits"] += 1000
    dataset[0]["labels"]["compressed_bits"] += 1000
    mutated = vectorize_record(dataset[0], caps)
    assert np.array_equal(original, mutated)


def test_compression_mirror_oracle_code_fields_match_contract_target() -> None:
    dataset = build_dataset("smoke", seed=132, train_episodes=1, val_episodes=1, test_episodes=1)
    caps = profile_caps("smoke")
    fields = oracle_code_fields(dataset[0], caps)
    target = dataset[0]["labels"]["state"]
    source_time = dataset[0]["evaluation_contract"]["memory_relevant_positions"][0]["time"]
    assert fields["address"] == target["color"] * caps["n_shapes"] + target["shape"]
    assert fields["schema"] == target["vel"] + caps["max_speed"]
    assert fields["residual"] == target["pos"]
    assert fields["action"] == dataset[0]["labels"]["action"]
    assert fields["provenance"] == source_time
    assert fields["provenance"] != dataset[0]["model_input"]["query"]["time"]


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
    assert summary["learned_result_count"] == 0


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


def test_compression_mirror_learned_codec_emits_trainable_rows() -> None:
    dataset = build_dataset("smoke", seed=130, train_episodes=8, val_episodes=2, test_episodes=2)
    learned = train_learned_codec(dataset, "smoke", seed=130, epochs=3)
    rows = evaluate_dataset(dataset, "smoke", seed=130, learned=learned)
    summary = build_summary(dataset, rows)
    learned_rows = [row for row in rows if row["policy"] == "learned_codec"]
    assert learned_rows
    assert summary["policy_count"] == len(ALL_POLICIES)
    assert summary["learned_result_count"] == len(dataset)
    assert summary["learned_codec_compression_ratio_vs_verbatim"] > 1.0
    assert summary["paid_compute_authorized"] == 0.0
    assert all(row["policy_is_learned_result"] == 1.0 for row in learned_rows)
    assert all(row["within_budget"] == 1.0 for row in learned_rows)
    assert all("predicted_state" in row for row in learned_rows)
    assert all("compact_code_fields" in row for row in learned_rows)
    assert all("oracle_code_fields" not in row for row in learned_rows)
    assert all(row["committed_bits_by_field"] == learned_bits_by_field(profile_caps("smoke")) for row in learned_rows)
    assert all(row["model_parameter_count"] == learned["parameter_count"] for row in learned_rows)
    assert all(row["source_required_fields_visible"] == 0.0 for row in learned_rows)
    assert all(row["source_state_reconstructable"] == 0.0 for row in learned_rows)


def test_compression_mirror_diagnostic_rows_localize_without_becoming_results() -> None:
    dataset = build_dataset("smoke", seed=133, train_episodes=8, val_episodes=2, test_episodes=2)
    learned = train_learned_codec(dataset, "smoke", seed=133, epochs=3)
    rows = evaluate_dataset(dataset, "smoke", seed=133, learned=learned)
    summary = build_summary(dataset, rows)
    diagnostic_rows = [row for row in rows if row["policy"] in DIAGNOSTIC_POLICIES]
    assert diagnostic_rows
    assert {row["policy"] for row in diagnostic_rows} == set(DIAGNOSTIC_POLICIES)
    assert all(row["policy_is_learned_result"] == 0.0 for row in diagnostic_rows)
    assert all(row["policy_is_diagnostic_control"] == 1.0 for row in diagnostic_rows)
    assert summary["diagnostic_result_count"] == len(dataset) * len(DIAGNOSTIC_POLICIES)
    assert summary["learned_result_count"] == len(dataset)
    assert summary["diagnostic_oracle_input_used"] == 1.0
    assert "oracle_address_payload_rescue_delta" in summary
    assert "provenance_exposure_rescue_delta" in summary
    assert "visible_source_state_oracle_action_oracle_decoder_joint_success" in summary
    assert "provenance_exposed_oracle_decoder_joint_success" in summary
    assert "learned_state_oracle_action_oracle_decoder_joint_success" in summary
    assert "oracle_state_learned_action_oracle_decoder_joint_success" in summary
    assert "source_signature_action_ambiguity_rate" in summary
    for row in diagnostic_rows:
        assert "encoder_address_accuracy" in row
        assert "encoder_payload_accuracy" in row
        assert "encoder_provenance_accuracy" in row
        assert "source_required_fields_visible" in row
        assert "source_state_reconstructable" in row


def test_compression_mirror_provenance_uses_memory_relevant_time_not_query_time() -> None:
    dataset = build_dataset("smoke", seed=134, train_episodes=2, val_episodes=1, test_episodes=1)
    caps = profile_caps("smoke")
    for row in dataset:
        event = source_event_for_record(row)
        fields = oracle_code_fields(row, caps)
        assert fields["provenance"] == row["evaluation_contract"]["memory_relevant_positions"][0]["time"]
        assert fields["provenance"] == event["time"]
        assert fields["provenance"] <= row["model_input"]["query"]["time"]


def test_compression_mirror_source_observation_audit_separates_availability_from_labels() -> None:
    dataset = build_dataset("smoke", seed=136, train_episodes=2, val_episodes=1, test_episodes=1)
    caps = profile_caps("smoke")
    row = dataset[0]
    before = source_observation_audit(row, caps)
    row["labels"]["state"]["color"] = (int(row["labels"]["state"]["color"]) + 1) % caps["n_colors"]
    row["labels"]["state"]["shape"] = (int(row["labels"]["state"]["shape"]) + 1) % caps["n_shapes"]
    row["labels"]["state"]["pos"] = (int(row["labels"]["state"]["pos"]) + 1) % caps["track_length"]
    after = source_observation_audit(row, caps)
    assert before["source_event_present"] == after["source_event_present"]
    assert before["source_event_observed"] == after["source_event_observed"]
    assert before["source_required_fields_visible"] == after["source_required_fields_visible"]
    assert before["source_query_gap"] == after["source_query_gap"]


def test_compression_mirror_visible_source_extractor_uses_model_input_and_contract_pointer() -> None:
    dataset = build_dataset("smoke", seed=137, train_episodes=12, val_episodes=1, test_episodes=1)
    caps = profile_caps("smoke")
    row = next(item for item in dataset if source_observation_audit(item, caps)["source_event_complete"] == 1.0)
    action = int(row["labels"]["action"])
    fields = source_observation_code_fields(row, caps, action)
    event = source_event_for_record(row)
    assert fields["address"] == int(event["color"]) * caps["n_shapes"] + int(event["shape"])
    assert fields["residual"] == int(event["pos"])
    assert fields["action"] == action
    row["labels"]["state"]["pos"] = (int(row["labels"]["state"]["pos"]) + 2) % caps["track_length"]
    assert source_observation_code_fields(row, caps, action) == fields
    for item in row["model_input"]["observations"]:
        if int(item["time"]) == int(event["time"]) and int(item["object_index"]) == int(event["object_index"]):
            item["pos"] = (int(item["pos"]) + 1) % caps["track_length"]
            break
    assert source_observation_code_fields(row, caps, action)["residual"] != fields["residual"]


def test_compression_mirror_payload_action_split_diagnostics_are_registered() -> None:
    assert "visible_source_state_oracle_action_oracle_decoder" in DIAGNOSTIC_POLICIES
    assert "provenance_exposed_oracle_decoder" in DIAGNOSTIC_POLICIES
    assert "learned_state_oracle_action_oracle_decoder" in DIAGNOSTIC_POLICIES
    assert "oracle_state_learned_action_oracle_decoder" in DIAGNOSTIC_POLICIES
    assert len(ALL_POLICIES) == len(BASELINE_POLICIES) + len(DIAGNOSTIC_POLICIES) + 1


def test_compression_mirror_decoder_generalization_summary_reports_train_validation_test() -> None:
    dataset = build_dataset("smoke", seed=138, train_episodes=8, val_episodes=2, test_episodes=2)
    learned = train_learned_codec(dataset, "smoke", seed=138, epochs=3)
    rows = evaluate_dataset(dataset, "smoke", seed=138, learned=learned)
    summary = build_summary(dataset, rows)
    for split in ("train", "validation", "test"):
        assert f"oracle_code_learned_decoder_{split}_joint_success" in summary
        assert f"oracle_code_learned_decoder_{split}_state_success" in summary
        assert f"oracle_code_learned_decoder_{split}_action_success" in summary
    assert "oracle_code_learned_decoder_train_test_joint_gap" in summary


def test_compression_mirror_action_ambiguity_metric_is_deterministic() -> None:
    dataset = build_dataset("smoke", seed=139, train_episodes=8, val_episodes=4, test_episodes=4)
    caps = profile_caps("smoke")
    first = action_ambiguity_rate(dataset, caps, "test")
    second = action_ambiguity_rate(dataset, caps, "test")
    signatures = [source_signature_for_action(row, caps) for row in dataset if row["split"] == "test"]
    assert first == second
    assert 0.0 <= first <= 1.0
    assert signatures == [source_signature_for_action(row, caps) for row in dataset if row["split"] == "test"]


def test_compression_mirror_diagnostic_controls_are_deterministic() -> None:
    dataset = build_dataset("smoke", seed=135, train_episodes=8, val_episodes=2, test_episodes=2)
    learned_a = train_learned_codec(dataset, "smoke", seed=135, epochs=3)
    learned_b = train_learned_codec(dataset, "smoke", seed=135, epochs=3)
    rows_a = evaluate_dataset(dataset, "smoke", seed=135, learned=learned_a)
    rows_b = evaluate_dataset(dataset, "smoke", seed=135, learned=learned_b)
    selected_a = [
        (row["episode_id"], row["policy"], row["joint_success"], row.get("compact_code_fields"))
        for row in rows_a
        if row["policy"] in DIAGNOSTIC_POLICIES
    ]
    selected_b = [
        (row["episode_id"], row["policy"], row["joint_success"], row.get("compact_code_fields"))
        for row in rows_b
        if row["policy"] in DIAGNOSTIC_POLICIES
    ]
    assert selected_a == selected_b


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
    assert dict(spec.minimum_summary_values)["learned_result_count"] == 1.0
    assert dict(spec.minimum_summary_values)["diagnostic_result_count"] == 1.0


def test_compression_mirror_smoke_suite() -> None:
    output_root = Path.cwd() / "reports" / "test_outputs" / f"compression_mirror_smoke_{uuid.uuid4().hex}"
    try:
        results = run_specs(
            specs=get_suite_specs("compression_mirror"),
            profile="smoke",
            output_root=output_root,
            python_executable=sys.executable,
            timeout_sec=300,
        )
        failures = [(result.simulation_id, result.validation_error, result.stderr_tail) for result in results if not result.ok]
        assert not failures, failures
        metrics_path = Path(results[0].metrics_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
    assert payload["summary"]["forbidden_input_violation_count"] == 0
    assert payload["summary"]["future_observation_violation_count"] == 0
    assert payload["summary"]["local_mirror_code_authorized"] == 1.0
    assert payload["summary"]["paid_compute_authorized"] == 0.0
    assert payload["summary"]["learned_result_count"] > 0
    assert payload["summary"]["diagnostic_result_count"] > 0
    assert "learned_codec_joint_success" in payload["summary"]
    assert "oracle_address_payload_rescue_delta" in payload["summary"]


def test_compression_mirror_invalid_profile_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown hard symbolic profile"):
        build_dataset("bad", seed=127, train_episodes=1, val_episodes=1, test_episodes=1)
