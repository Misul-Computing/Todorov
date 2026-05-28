from neuroloc.simulations.memory.local_100k_learned_block_codec_frontier import (
    LearnedBlockCodecFrontier,
    build_summary,
    compression_trial,
    current_payload_transform,
    heldout_block,
    inverse_current_payload_transform,
    load_authored_sources,
    random_label_block,
    run_profile,
    standard_codec_sweep,
    train_phrase_dictionary,
    train_test_overlap_counts,
)


PREFIX = "local_100k_learned_block_codec_frontier"


def test_frontier_sources_are_authored_and_source_heldout() -> None:
    train_blob, test_blob, train_manifest, test_manifest = load_authored_sources()
    block = heldout_block(test_blob, 16384, 2459)
    overlap = train_test_overlap_counts(train_blob, block, train_manifest, test_manifest)
    assert len(train_blob) > 4096
    assert len(test_blob) > 4096
    assert len(block) == 16384
    assert {row["path"] for row in train_manifest}.isdisjoint({row["path"] for row in test_manifest})
    assert overlap["source_train_test_path_overlap_count"] == 0.0
    assert overlap["source_train_test_hash_overlap_count"] == 0.0


def test_frontier_dictionary_is_train_only_and_decodes_exactly() -> None:
    train_blob, test_blob, train_manifest, _test_manifest = load_authored_sources()
    block = heldout_block(test_blob, 16384, 2459)
    dictionary = train_phrase_dictionary(train_blob, 128)
    cell = LearnedBlockCodecFrontier(train_blob, block, train_manifest)
    assert dictionary
    assert cell.train_only_dictionary_used == 1.0
    assert cell.learned_or_phrase_codec_used == 1.0
    assert cell.per_fact_value_slice_count == 0
    assert cell.block_stream_count == 1
    assert cell.raw_source_block_retained == 0.0
    assert not hasattr(cell, "raw_source_block")
    assert cell.decode() == block


def test_frontier_standard_sweep_and_transform_roundtrip() -> None:
    _train_blob, test_blob, _train_manifest, _test_manifest = load_authored_sources()
    block = heldout_block(test_blob, 16384, 2459)
    transformed = current_payload_transform(block)
    assert inverse_current_payload_transform(transformed) == block
    sweep = standard_codec_sweep(block)
    assert len(sweep) == 12
    assert all(row["charged_bits"] > 0 for row in sweep)


def test_frontier_reports_honest_failure_when_not_two_percent_better() -> None:
    summary = build_summary("smoke", seed=2459)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_reconstruction"] == 1.0
    assert summary[f"{PREFIX}_standard_reconstruction"] == 1.0
    assert summary[f"{PREFIX}_no_per_fact_rows"] == 1.0
    assert summary[f"{PREFIX}_per_fact_value_slice_count"] == 0.0
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_beats_best_fair_standard_by_2pct"] == 0.0
    assert summary[f"{PREFIX}_honest_failure_reported"] == 1.0
    assert summary[f"{PREFIX}_publishable"] == 0.0
    assert summary[f"{PREFIX}_paper_ready_local_candidate_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0


def test_frontier_static_retrieval_dominance_certificate() -> None:
    row = run_profile("smoke", seed=2459)
    assert row["static_retrieval_dominance_certificate"] == 1.0
    assert row["same_payload_scan_success"] == 1.0
    assert row["same_payload_qa_wrapper_success"] == 1.0
    assert row["same_payload_qa_wrapper_bits"] > row["same_payload_scan_bits"]
    assert row["same_payload_qa_wrapper_cannot_beat_scan"] == 1.0
    assert row["payload_bits_must_improve_for_wrapper_to_win"] > 0.0
    assert row["qa_wrapper_promoted"] == 0.0


def test_frontier_random_label_control_costs_like_entropy() -> None:
    train_blob, test_blob, train_manifest, _test_manifest = load_authored_sources()
    block = heldout_block(test_blob, 16384, 2459)
    random_block = random_label_block(2459, len(block))
    real = compression_trial(block, train_blob, train_manifest, 2459)
    random_trial = compression_trial(random_block, train_blob, train_manifest, 2459)
    assert real["exact_reconstruction"] == 1.0
    assert random_trial["exact_reconstruction"] == 1.0
    assert random_trial["random_label_payload_incompressible"] == 1.0
    assert random_trial["learned_charged_bits"] >= len(random_block) * 8
