from neuroloc.simulations.memory.local_100k_shared_predictor_exact_codec import (
    SharedPredictorExactCodecCell,
    build_facts,
    build_random_twin,
    build_summary,
    score_reads,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_shared_predictor_fact_pack_is_deterministic_and_source_heldout() -> None:
    first_train, first_test, _, first_manifest = build_facts(827, 128, 96)
    second_train, second_test, _, second_manifest = build_facts(827, 128, 96)
    third_train, third_test, _, _ = build_facts(828, 128, 96)
    assert first_train == second_train
    assert first_test == second_test
    assert first_manifest == second_manifest
    assert (first_train, first_test) != (third_train, third_test)
    assert {row["source"] for row in first_train}.isdisjoint({row["source"] for row in first_test})
    assert not {tuple(row["key"]) for row in first_train}.intersection({tuple(row["key"]) for row in first_test})


def test_shared_predictor_codec_uses_one_block_stream_not_value_rows() -> None:
    train, facts, _, manifest = build_facts(827, 128, 96)
    cell = SharedPredictorExactCodecCell(train, facts, manifest)
    assert cell.block_stream_count == 1
    assert cell.per_fact_value_slice_count == 0
    assert cell.shared_predictor_used == 1.0
    assert cell.independent_value_slice_path_used == 0.0
    assert cell.raw_payload_retained == 0.0
    assert cell.reads_from_compressed_block == 1.0
    assert not hasattr(cell, "raw_payload")


def test_shared_predictor_codec_roundtrips_heldout_chunks_exactly() -> None:
    train, facts, _, manifest = build_facts(827, 128, 96)
    cell = SharedPredictorExactCodecCell(train, facts, manifest)
    reads = [cell.read(tuple(row["key"])) for row in facts]
    success = sum(row["exact_success"] for row in score_reads(facts, reads)) / len(facts)
    assert success == 1.0


def test_shared_predictor_random_twin_pays_entropy_and_loses_density() -> None:
    train, facts, _, manifest = build_facts(827, 128, 96)
    random_twin = build_random_twin(827, facts)
    real_cell = SharedPredictorExactCodecCell(train, facts, manifest)
    random_cell = SharedPredictorExactCodecCell(train, random_twin, manifest)
    assert random_cell.payload_bits > real_cell.payload_bits
    assert random_cell.strict_multiplier < real_cell.strict_multiplier


def test_shared_predictor_summary_reports_product_without_breakthrough_overclaim() -> None:
    summary = build_summary("smoke", seed=827)
    assert summary["local_100k_shared_predictor_exact_codec_engineering_pass"] == 1.0
    assert summary["local_100k_shared_predictor_exact_codec_product_pass"] == 1.0
    assert summary["local_100k_shared_predictor_exact_codec_exact_retrieval_success"] == 1.0
    assert summary["local_100k_shared_predictor_exact_codec_controls_collapse"] == 1.0
    assert summary["local_100k_shared_predictor_exact_codec_block_stream_count"] == 1.0
    assert summary["local_100k_shared_predictor_exact_codec_per_fact_value_slice_count"] == 0.0
    assert summary["local_100k_shared_predictor_exact_codec_random_label_density_control_collapse"] == 1.0
    assert summary["local_100k_shared_predictor_exact_codec_strict_breakthrough_authorized"] == 0.0
    assert summary["local_100k_shared_predictor_exact_codec_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary["local_100k_shared_predictor_exact_codec_full_nm_authorized"] == 0.0


def test_shared_predictor_accounting_charges_every_surface() -> None:
    summary = build_summary("smoke", seed=827)
    assert summary["local_100k_shared_predictor_exact_codec_parameter_count"] < 100000.0
    assert summary["local_100k_shared_predictor_exact_codec_predictor_model_bits"] > 0.0
    assert summary["local_100k_shared_predictor_exact_codec_payload_bits"] > 0.0
    assert summary["local_100k_shared_predictor_exact_codec_key_assignment_bits"] > 0.0
    assert summary["local_100k_shared_predictor_exact_codec_manifest_bits"] > 0.0
    assert summary["local_100k_shared_predictor_exact_codec_training_supervision_bits"] > 0.0
    assert summary["local_100k_shared_predictor_exact_codec_formula_or_schema_labels_present"] == 0.0
    assert summary["local_100k_shared_predictor_exact_codec_seed_oracle_authorized"] == 0.0


def test_shared_predictor_registry_entry() -> None:
    assert "local_100k_shared_predictor_exact_codec" in SIMULATION_SPECS
    assert "local_100k_shared_predictor_exact_codec" in SUITES["compression_mirror"]
    assert "local_100k_shared_predictor_exact_codec" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_shared_predictor_exact_codec"]
    assert spec.metrics_filename == "local_100k_shared_predictor_exact_codec_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_shared_predictor_exact_codec_product_pass"] == 1.0
    assert dict(spec.maximum_summary_values)["local_100k_shared_predictor_exact_codec_strict_breakthrough_authorized"] == 0.0
