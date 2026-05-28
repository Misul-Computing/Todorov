from pathlib import Path

from neuroloc.simulations.memory.local_100k_source_structure_block_codec import (
    PROJECT_ROOT,
    SourceStructurePayloadModule,
    build_summary,
    decompress_best,
    learn_indent_unit,
    learned_codec,
    measure_block,
    read_joined,
    restore_learned,
    restore_structure,
    target_paths,
    train_paths,
    transform_structure,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_structure_block_codec"


def test_source_structure_round_trip_is_exact_with_fallback_rows() -> None:
    block = b"def f():\n    return 1\n  odd\n\t\ttabbed\n      mixed\n"
    encoded = transform_structure(block, 4)
    restored = restore_structure(encoded["count_stream"], encoded["body_stream"], encoded["line_count"], encoded["indent_unit"])
    assert restored == block
    assert encoded["count_stream"] != block
    assert encoded["body_stream"] != block


def test_source_structure_indent_unit_is_learned_from_disjoint_train_files() -> None:
    train = train_paths()
    targets = target_paths("hard")
    train_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in train if path.exists()}
    target_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in targets if path.exists()}
    assert learn_indent_unit(read_joined(train)) == 4
    assert train_rel.isdisjoint(target_rel)


def test_source_structure_codec_beats_indent_token_margin_on_hard_block() -> None:
    train_block = read_joined(train_paths())
    target_block = read_joined(target_paths("hard"))
    metrics = measure_block(train_block, target_block, 8123)
    assert metrics["exact_reconstruction_success"] == 1.0
    assert metrics["random_label_exact_reconstruction_success"] == 1.0
    assert metrics["random_label_payload_incompressible"] == 1.0
    assert metrics["best_standard_payload_bits"] == 128816.0
    assert metrics["learned_count_payload_bits"] == 4416.0
    assert metrics["learned_body_payload_bits"] == 119528.0
    assert metrics["learned_structure_header_bits"] == 256.0
    assert metrics["learned_payload_bits"] == 124200.0
    assert metrics["payload_improvement_over_best_standard"] >= 0.0358
    assert metrics["strict_improvement_over_best_standard"] >= 0.0285
    assert metrics["beats_indent_token_strict_margin"] == 1.0
    assert metrics["strict_improvement_delta_over_indent_token"] > 0.008
    assert metrics["random_label_payload_improvement_over_best_standard"] <= 0.0
    assert metrics["compressed_stream_read_success"] == 1.0
    assert metrics["codec_state_has_raw_target_block"] == 0.0
    assert metrics["codec_state_has_uncompressed_count_stream"] == 0.0
    assert metrics["codec_state_has_uncompressed_body_stream"] == 0.0
    assert metrics["codec_state_has_restored_block"] == 0.0
    assert metrics["compressed_count_payload_retained"] == 1.0
    assert metrics["compressed_body_payload_retained"] == 1.0
    assert metrics["wrong_indent_unit_exact_reconstruction_success"] == 0.0
    assert metrics["decoder_disabled_exact_reconstruction_success"] == 0.0


def test_source_structure_learned_payload_restores_without_raw_cache() -> None:
    train_block = read_joined(train_paths())
    target_block = read_joined(target_paths("hard"))
    payload = learned_codec(train_block, target_block)
    restored = restore_learned(payload)
    count_stream = decompress_best(payload["count_codec_name"], payload["count_payload"])
    body_stream = decompress_best(payload["body_codec_name"], payload["body_payload"])
    assert restored == target_block
    assert count_stream != target_block
    assert body_stream != target_block
    assert "target_block" not in payload
    assert "restored_block" not in payload
    assert "count_stream" not in payload
    assert "body_stream" not in payload


def test_source_structure_payload_survives_state_dict_reload() -> None:
    train_block = read_joined(train_paths())
    target_block = read_joined(target_paths("hard"))
    payload = learned_codec(train_block, target_block)
    module = SourceStructurePayloadModule.from_learned(payload)
    reloaded = SourceStructurePayloadModule.empty_like(module)
    reloaded.load_state_dict(module.state_dict())
    assert module.restore() == target_block
    assert reloaded.restore() == target_block
    assert len(list(module.parameters())) == 0
    assert "count_payload" in module.state_dict()
    assert "body_payload" in module.state_dict()
    assert "count_codec_code" in module.state_dict()
    assert "body_codec_code" in module.state_dict()


def test_source_structure_summary_has_category_guards() -> None:
    summary = build_summary("hard", seed=8123)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_target_file_count"] == 4.0
    assert summary[f"{PREFIX}_source_structure_block_codec_candidate"] == 1.0
    assert summary[f"{PREFIX}_publishable_block_codec_candidate"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_model_state_restore_success"] == 1.0
    assert summary[f"{PREFIX}_model_state_reload_success"] == 1.0
    assert summary[f"{PREFIX}_model_state_payload_used"] == 1.0
    assert summary[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert summary[f"{PREFIX}_state_dict_count_payload_used"] == 1.0
    assert summary[f"{PREFIX}_state_dict_body_payload_used"] == 1.0
    assert summary[f"{PREFIX}_state_dict_codec_selectors_used"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_per_fact_value_row_count"] == 0.0
    assert summary[f"{PREFIX}_assignment_row_count"] == 0.0
    assert summary[f"{PREFIX}_hidden_fact_value_row_detected"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_compressed_stream_read_success"] == 1.0
    assert summary[f"{PREFIX}_codec_state_has_raw_target_block"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_uncompressed_count_stream"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_uncompressed_body_stream"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_restored_block"] == 0.0
    assert summary[f"{PREFIX}_wrong_indent_unit_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_decoder_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_body_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_count_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_ngram_width_bytes"] == 64.0
    assert summary[f"{PREFIX}_source_train_test_ngram_overlap_count"] == 3104.0


def test_source_structure_registry_entry() -> None:
    assert PREFIX in SIMULATION_SPECS
    assert PREFIX in SUITES["compression_mirror"]
    assert PREFIX in SUITES["precompute"]
    spec = SIMULATION_SPECS[PREFIX]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_strict_improvement_over_best_standard"] == 0.028
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_general_unknown_structure_breakthrough_authorized"] == 0.0
