from neuroloc.simulations.memory.local_100k_content_addressed_source_codec import (
    ContentAddressedSourceCodecCell,
    build_facts,
    build_random_twin,
    build_summary,
    score_reads,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


CHARGED_CORPUS_CODEC_BASELINE_MULTIPLIER = 13.941917871967359


def test_content_addressed_facts_use_opaque_digest_keys_without_source_offsets() -> None:
    first_train, first_test, first_block, first_manifest = build_facts(941, 128, 96)
    second_train, second_test, second_block, second_manifest = build_facts(941, 128, 96)
    third_train, third_test, _, _ = build_facts(942, 128, 96)
    assert first_train == second_train
    assert first_test == second_test
    assert first_block == second_block
    assert first_manifest == second_manifest
    assert (first_train, first_test) != (third_train, third_test)
    assert len({tuple(row["key"]) for row in first_test}) == len(first_test)
    assert all("source" not in row for row in first_test)
    assert all("offset" not in row for row in first_test)
    assert all("content_window_digest" in row for row in first_test)
    assert all(tuple(row["key"]) == tuple(row["content_window_digest"]) for row in first_test)


def test_content_addressed_codec_uses_one_charged_block_without_assignment_rows() -> None:
    train, facts, source_block, manifest = build_facts(941, 128, 96)
    cell = ContentAddressedSourceCodecCell(train, facts, source_block, manifest)
    assert cell.source_block_count == 1
    assert cell.block_stream_count == 1
    assert cell.key_assignment_bits == 0
    assert cell.per_fact_value_slice_count == 0
    assert cell.source_offset_routing_used == 0.0
    assert cell.content_digest_key_target == 1.0
    assert cell.raw_source_block_retained == 0.0
    assert cell.reads_from_compressed_block == 1.0
    assert not hasattr(cell, "decoded_block")
    assert not hasattr(cell, "fact_assignment_rows")
    before = cell.decompression_count
    cell.read(tuple(facts[0]["key"]))
    assert cell.decompression_count == before + 1


def test_content_addressed_codec_roundtrips_exactly_by_decompressing_and_scanning() -> None:
    train, facts, source_block, manifest = build_facts(941, 128, 96)
    cell = ContentAddressedSourceCodecCell(train, facts, source_block, manifest)
    reads = [cell.read(tuple(row["key"])) for row in facts]
    success = sum(row["exact_success"] for row in score_reads(facts, reads)) / len(facts)
    assert success == 1.0
    assert cell.scan_count == len(facts)


def test_content_addressed_codec_controls_fail() -> None:
    train, facts, source_block, manifest = build_facts(941, 128, 96)
    random_twin = build_random_twin(941, facts)
    cell = ContentAddressedSourceCodecCell(train, facts, source_block, manifest)
    shuffled_reads = [cell.read(tuple(reversed(row["key"]))) for row in facts]
    wrong_digest_reads = [cell.read(tuple(row["key"][:-1]) + ((int(row["key"][-1]) + 1) % 256,)) for row in facts]
    random_twin_reads = [cell.read(tuple(row["key"])) for row in random_twin]
    read_disabled_reads = [cell.read(tuple(row["key"]), read_enabled=False) for row in facts]
    decoder_disabled_reads = [cell.read(tuple(row["key"]), decoder_enabled=False) for row in facts]
    assert sum(row["exact_success"] for row in score_reads(facts, shuffled_reads)) == 0
    assert sum(row["exact_success"] for row in score_reads(facts, wrong_digest_reads)) == 0
    assert sum(row["exact_success"] for row in score_reads(random_twin, random_twin_reads)) == 0
    assert sum(row["exact_success"] for row in score_reads(facts, read_disabled_reads)) == 0
    assert sum(row["exact_success"] for row in score_reads(facts, decoder_disabled_reads)) == 0


def test_content_addressed_summary_reports_product_without_breakthrough_overclaim() -> None:
    summary = build_summary("smoke", seed=941)
    assert summary["local_100k_content_addressed_source_codec_product_pass"] == 1.0
    assert summary["local_100k_content_addressed_source_codec_exact_retrieval_success"] == 1.0
    assert summary["local_100k_content_addressed_source_codec_controls_collapse"] == 1.0
    assert summary["local_100k_content_addressed_source_codec_source_offset_routing_used"] == 0.0
    assert summary["local_100k_content_addressed_source_codec_content_digest_key_target"] == 1.0
    assert summary["local_100k_content_addressed_source_codec_key_assignment_bits"] == 0.0
    assert summary["local_100k_content_addressed_source_codec_per_fact_value_slice_count"] == 0.0
    assert summary["local_100k_content_addressed_source_codec_reads_from_compressed_block"] == 1.0
    assert summary["local_100k_content_addressed_source_codec_raw_source_block_retained"] == 0.0
    assert summary["local_100k_content_addressed_source_codec_strict_multiplier"] > CHARGED_CORPUS_CODEC_BASELINE_MULTIPLIER
    assert summary["local_100k_content_addressed_source_codec_strict_breakthrough_authorized"] == 0.0


def test_content_addressed_registry_entry() -> None:
    assert "local_100k_content_addressed_source_codec" in SIMULATION_SPECS
    assert "local_100k_content_addressed_source_codec" in SUITES["compression_mirror"]
    assert "local_100k_content_addressed_source_codec" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_content_addressed_source_codec"]
    assert spec.metrics_filename == "local_100k_content_addressed_source_codec_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_content_addressed_source_codec_product_pass"] == 1.0
    assert dict(spec.minimum_summary_values)["local_100k_content_addressed_source_codec_strict_multiplier"] > CHARGED_CORPUS_CODEC_BASELINE_MULTIPLIER
    assert dict(spec.maximum_summary_values)["local_100k_content_addressed_source_codec_strict_breakthrough_authorized"] == 0.0
