from neuroloc.simulations.memory.local_100k_source_block_codec import (
    SourceBlockExactCodecCell,
    build_facts,
    build_random_twin,
    build_summary,
    score_reads,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


CHARGED_CORPUS_CODEC_BASELINE_MULTIPLIER = 13.941917871967359


def test_source_block_fact_pack_is_deterministic_and_source_heldout() -> None:
    first_train, first_test, first_block, first_manifest = build_facts(941, 128, 96)
    second_train, second_test, second_block, second_manifest = build_facts(941, 128, 96)
    third_train, third_test, _, _ = build_facts(942, 128, 96)
    assert first_train == second_train
    assert first_test == second_test
    assert first_block == second_block
    assert first_manifest == second_manifest
    assert (first_train, first_test) != (third_train, third_test)
    assert {row["source"] for row in first_train}.isdisjoint({row["source"] for row in first_test})
    assert len({(row["source"], int(row["offset"])) for row in first_test}) == len(first_test)
    assert len({tuple(row["key"]) for row in first_test}) == len(first_test)


def test_source_block_codec_uses_single_compressed_source_block() -> None:
    train, facts, source_block, manifest = build_facts(941, 128, 96)
    cell = SourceBlockExactCodecCell(train, facts, source_block, manifest)
    assert cell.source_block_count == 1
    assert cell.block_stream_count == 1
    assert cell.per_fact_value_slice_count == 0
    assert cell.source_offset_routing_used == 1.0
    assert cell.independent_value_slice_path_used == 0.0
    assert cell.raw_source_block_retained == 0.0
    assert cell.reads_from_compressed_block == 1.0
    assert not hasattr(cell, "decoded_block")
    before = cell.decompression_count
    cell.read(tuple(facts[0]["key"]))
    assert cell.decompression_count == before + 1


def test_source_block_codec_roundtrips_source_heldout_chunks_exactly() -> None:
    train, facts, source_block, manifest = build_facts(941, 128, 96)
    cell = SourceBlockExactCodecCell(train, facts, source_block, manifest)
    reads = [cell.read(tuple(row["key"])) for row in facts]
    success = sum(row["exact_success"] for row in score_reads(facts, reads)) / len(facts)
    assert success == 1.0


def test_source_block_codec_does_not_solve_random_label_twin() -> None:
    train, facts, source_block, manifest = build_facts(941, 128, 96)
    random_twin = build_random_twin(941, facts)
    cell = SourceBlockExactCodecCell(train, facts, source_block, manifest)
    twin_reads = [cell.read(tuple(row["key"])) for row in random_twin]
    twin_success = sum(row["exact_success"] for row in score_reads(random_twin, twin_reads)) / len(random_twin)
    assert twin_success == 0.0


def test_source_block_summary_reports_product_without_breakthrough_overclaim() -> None:
    summary = build_summary("smoke", seed=941)
    assert summary["local_100k_source_block_codec_engineering_pass"] == 1.0
    assert summary["local_100k_source_block_codec_product_pass"] == 1.0
    assert summary["local_100k_source_block_codec_exact_retrieval_success"] == 1.0
    assert summary["local_100k_source_block_codec_random_label_twin_success"] == 0.0
    assert summary["local_100k_source_block_codec_controls_collapse"] == 1.0
    assert summary["local_100k_source_block_codec_source_holdout_used"] == 1.0
    assert summary["local_100k_source_block_codec_source_offset_routing_used"] == 1.0
    assert summary["local_100k_source_block_codec_reads_from_compressed_block"] == 1.0
    assert summary["local_100k_source_block_codec_raw_source_block_retained"] == 0.0
    assert summary["local_100k_source_block_codec_block_stream_count"] == 1.0
    assert summary["local_100k_source_block_codec_per_fact_value_slice_count"] == 0.0
    assert summary["local_100k_source_block_codec_strict_multiplier"] > CHARGED_CORPUS_CODEC_BASELINE_MULTIPLIER
    assert summary["local_100k_source_block_codec_strict_breakthrough_authorized"] == 0.0
    assert summary["local_100k_source_block_codec_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary["local_100k_source_block_codec_full_nm_authorized"] == 0.0
    assert summary["local_100k_source_block_codec_paid_compute_authorized"] == 0.0


def test_source_block_accounting_charges_every_surface() -> None:
    summary = build_summary("smoke", seed=941)
    assert summary["local_100k_source_block_codec_parameter_count"] < 100000.0
    assert summary["local_100k_source_block_codec_block_payload_bits"] > 0.0
    assert summary["local_100k_source_block_codec_source_offset_bits"] > 0.0
    assert summary["local_100k_source_block_codec_decoder_bits"] > 0.0
    assert summary["local_100k_source_block_codec_manifest_bits"] > 0.0
    assert summary["local_100k_source_block_codec_committed_state_bits"] > summary["local_100k_source_block_codec_block_payload_bits"]
    assert summary["local_100k_source_block_codec_formula_or_schema_labels_present"] == 0.0
    assert summary["local_100k_source_block_codec_seed_oracle_authorized"] == 0.0
    assert summary["local_100k_source_block_codec_associative_random_key_target"] == 0.0
    assert summary["local_100k_source_block_codec_source_offset_key_target"] == 1.0
    assert summary["local_100k_source_block_codec_raw_source_block_bits_charged"] == 0.0


def test_source_block_registry_entry() -> None:
    assert "local_100k_source_block_codec" in SIMULATION_SPECS
    assert "local_100k_source_block_codec" in SUITES["compression_mirror"]
    assert "local_100k_source_block_codec" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_source_block_codec"]
    assert spec.metrics_filename == "local_100k_source_block_codec_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_source_block_codec_product_pass"] == 1.0
    assert dict(spec.minimum_summary_values)["local_100k_source_block_codec_strict_multiplier"] > CHARGED_CORPUS_CODEC_BASELINE_MULTIPLIER
    assert dict(spec.maximum_summary_values)["local_100k_source_block_codec_strict_breakthrough_authorized"] == 0.0
