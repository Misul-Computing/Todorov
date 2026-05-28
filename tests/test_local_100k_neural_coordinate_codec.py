from neuroloc.simulations.memory.local_100k_neural_coordinate_codec import (
    MAX_PARAMETER_TARGET,
    PROFILES,
    NeuralCoordinateCodecCell,
    account_bits,
    build_facts,
    build_random_twin,
    build_summary,
    evaluate_cell,
)


PREFIX = "local_100k_neural_coordinate_codec"


def test_coordinate_codec_facts_are_source_authored_and_stable() -> None:
    first_facts, first_block, first_manifest = build_facts(6203, 32)
    second_facts, second_block, second_manifest = build_facts(6203, 32)
    assert first_facts == second_facts
    assert first_block == second_block
    assert first_manifest == second_manifest
    assert len(first_facts) == 32
    assert len(first_block) > 1024
    assert len({row["domain"] for row in first_manifest}) >= 4
    assert all({"role", "row", "source_id", "source_name", "domain", "key", "value", "provenance"} <= set(row) for row in first_facts)


def test_coordinate_codec_is_trainable_torch_model_not_payload_pack() -> None:
    facts, block, manifest = build_facts(6203, 32)
    cell = NeuralCoordinateCodecCell(facts, manifest, 12, 6203)
    metrics = evaluate_cell(cell, facts, block, manifest)
    account = account_bits(cell, facts, block)
    assert 0 < cell.parameter_count() < int(MAX_PARAMETER_TARGET)
    assert cell.trainable_neural_predictor_used == 1.0
    assert cell.mantissa_payload_packing_used == 0.0
    assert cell.raw_source_block_retained == 0.0
    assert cell.per_fact_payload_row_used == 0.0
    assert account["physical_fp32_parameter_bits"] == cell.parameter_count() * 32
    assert account["charged_parameter_count"] == cell.parameter_count()
    assert 0.0 <= metrics["exact_answer_success"] <= 1.0
    assert metrics["same_block_content_scan_success"] == 1.0


def test_random_label_twin_is_reported_as_control_not_claim() -> None:
    facts, block, manifest = build_facts(6203, 32)
    random_twin = build_random_twin(6203, facts)
    twin = NeuralCoordinateCodecCell(random_twin, manifest, 12, 9204)
    metrics = evaluate_cell(twin, random_twin, block, manifest)
    assert metrics["exact_answer_success"] < 0.95
    assert metrics["same_block_content_scan_success"] == 0.0


def test_summary_demotes_if_scan_or_exact_gate_fails() -> None:
    summary = build_summary("smoke", seed=6203)
    assert summary[f"{PREFIX}_evaluated"] == 1.0
    assert summary[f"{PREFIX}_charged_parameter_count"] < float(MAX_PARAMETER_TARGET)
    assert summary[f"{PREFIX}_physical_fp32_parameter_bits"] == summary[f"{PREFIX}_charged_parameter_count"] * 32.0
    assert summary[f"{PREFIX}_same_block_content_scan_success"] == 1.0
    assert summary[f"{PREFIX}_same_block_content_scan_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_paid_compute_authorized"] == 0.0
    assert summary[f"{PREFIX}_mantissa_payload_packing_used"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_retained_in_model"] == 0.0
    if summary[f"{PREFIX}_exact_answer_success"] < 0.95:
        assert summary[f"{PREFIX}_engineering_pass"] == 0.0
        assert summary[f"{PREFIX}_learned_exact_retrieval_authorized"] == 0.0


def test_profiles_are_ordered_for_smoke_and_hard() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])
    assert int(PROFILES["hard"]["train_steps"]) >= int(PROFILES["smoke"]["train_steps"])
