from pathlib import Path

from neuroloc.simulations.memory.local_100k_indent_token_block_codec import (
    PROJECT_ROOT,
    build_summary,
    learn_token,
    measure_block,
    read_joined,
    restore_block,
    target_paths,
    train_paths,
    transform_block,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_indent_token_block_codec"


def test_indent_token_round_trip_is_exact() -> None:
    pattern = b"    "
    block = b"def f():\n    return 1\n\x80\xff\n"
    transformed = transform_block(block, pattern)
    assert transformed != block
    assert restore_block(transformed, pattern) == block


def test_indent_token_is_learned_from_disjoint_train_files() -> None:
    train = train_paths()
    targets = target_paths("hard")
    train_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in train if path.exists()}
    target_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in targets if path.exists()}
    token = learn_token(read_joined(train))
    assert token == b"    "
    assert train_rel.isdisjoint(target_rel)


def test_indent_token_codec_beats_best_standard_on_hard_block() -> None:
    train_block = read_joined(train_paths())
    target_block = read_joined(target_paths("hard"))
    metrics = measure_block(train_block, target_block, 7219)
    assert metrics["exact_reconstruction_success"] == 1.0
    assert metrics["random_label_payload_incompressible"] == 1.0
    assert metrics["payload_improvement_over_best_standard"] >= 0.02
    assert metrics["strict_improvement_over_best_standard"] >= 0.02
    assert metrics["random_label_payload_improvement_over_best_standard"] <= 0.0
    assert metrics["compressed_stream_read_success"] == 1.0
    assert metrics["codec_state_has_raw_target_block"] == 0.0
    assert metrics["codec_state_has_transformed_block"] == 0.0
    assert metrics["codec_state_has_restored_block"] == 0.0
    assert metrics["wrong_token_exact_reconstruction_success"] == 0.0
    assert metrics["decoder_disabled_exact_reconstruction_success"] == 0.0


def test_indent_token_summary_has_category_guards() -> None:
    summary = build_summary("hard", seed=7219)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_target_file_count"] == 4.0
    assert summary[f"{PREFIX}_strict_improvement_over_best_standard"] >= 0.02
    assert summary[f"{PREFIX}_learned_token_block_codec_candidate"] == 1.0
    assert summary[f"{PREFIX}_publishable_block_codec_candidate"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_static_retrieval_dominance_certificate"] == 1.0
    assert summary[f"{PREFIX}_static_retrieval_wrapper_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_per_fact_value_row_count"] == 0.0
    assert summary[f"{PREFIX}_assignment_row_count"] == 0.0
    assert summary[f"{PREFIX}_hidden_fact_value_row_detected"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_compressed_stream_read_success"] == 1.0
    assert summary[f"{PREFIX}_codec_state_has_raw_target_block"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_transformed_block"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_restored_block"] == 0.0
    assert summary[f"{PREFIX}_wrong_token_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_decoder_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_token_map_disabled_strict_improvement_over_best_standard"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_ngram_width_bytes"] == 64.0
    assert summary[f"{PREFIX}_source_train_test_ngram_overlap_count"] == 3104.0


def test_indent_token_registry_entry() -> None:
    assert PREFIX in SIMULATION_SPECS
    assert PREFIX in SUITES["compression_mirror"]
    assert PREFIX in SUITES["precompute"]
    spec = SIMULATION_SPECS[PREFIX]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_strict_improvement_over_best_standard"] == 0.009
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_general_unknown_structure_breakthrough_authorized"] == 0.0
