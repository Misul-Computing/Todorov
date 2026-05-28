from neuroloc.simulations.memory.local_100k_zstd_trained_dictionary_baseline_audit import (
    SIMULATION_ID,
    build_summary,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_zstd_trained_dictionary_baseline_audit"


def test_zstd_trained_dictionary_baseline_reconstructs_and_charges_dictionary() -> None:
    summary = build_summary("hard", seed=11933)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_public_trained_dictionary_baseline_used"] == 1.0
    assert summary[f"{PREFIX}_block_public_dict_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_corpus_public_dict_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_block_public_dict_dictionary_bits"] > 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_dictionary_bits"] > 0.0
    assert summary[f"{PREFIX}_block_public_dict_charged_bits"] > summary[f"{PREFIX}_block_public_dict_undercharged_bits"]
    assert summary[f"{PREFIX}_corpus_public_dict_charged_bits"] > summary[f"{PREFIX}_corpus_public_dict_undercharged_bits"]
    assert summary[f"{PREFIX}_block_public_dict_header_bits"] == 64.0
    assert summary[f"{PREFIX}_block_public_dict_selector_bits"] == 16.0


def test_zstd_trained_dictionary_baseline_is_train_only() -> None:
    summary = build_summary("smoke", seed=11933)
    assert summary[f"{PREFIX}_block_public_dict_train_only"] == 1.0
    assert summary[f"{PREFIX}_corpus_public_dict_train_only"] == 1.0
    assert summary[f"{PREFIX}_block_public_dict_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_block_public_dict_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_train_test_hash_overlap_count"] == 0.0


def test_zstd_trained_dictionary_random_label_is_not_helped() -> None:
    summary = build_summary("hard", seed=11933)
    assert summary[f"{PREFIX}_block_public_dict_random_label_charged_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_block_public_dict_random_label_undercharged_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_random_label_charged_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_random_label_undercharged_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_block_public_dict_disabled_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_block_public_dict_shuffled_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_disabled_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_corpus_public_dict_shuffled_reconstruction_success"] == 0.0


def test_zstd_trained_dictionary_compares_against_current_subtoken_codec() -> None:
    summary = build_summary("hard", seed=11933)
    assert summary[f"{PREFIX}_block_current_subtoken_payload_bits"] == 120952.0
    assert summary[f"{PREFIX}_block_public_dict_charged_bits"] == 151376.0
    assert summary[f"{PREFIX}_block_public_dict_undercharged_bits"] == 147200.0
    assert summary[f"{PREFIX}_block_current_subtoken_beats_public_dict_charged"] == 1.0
    assert summary[f"{PREFIX}_block_current_subtoken_beats_public_dict_undercharged"] == 1.0
    assert summary[f"{PREFIX}_block_current_subtoken_margin_over_public_dict_charged_bits"] >= 30000.0
    assert summary[f"{PREFIX}_block_current_subtoken_margin_over_public_dict_undercharged_bits"] >= 26000.0


def test_zstd_trained_dictionary_corpus_reports_aggregate_public_baseline() -> None:
    summary = build_summary("hard", seed=11933)
    assert summary[f"{PREFIX}_corpus_current_subtoken_payload_bits"] == 812688.0
    assert summary[f"{PREFIX}_corpus_public_dict_charged_bits"] == 982840.0
    assert summary[f"{PREFIX}_corpus_public_dict_undercharged_bits"] == 949992.0
    assert summary[f"{PREFIX}_corpus_current_subtoken_beats_public_dict_charged"] == 1.0
    assert summary[f"{PREFIX}_corpus_current_subtoken_beats_public_dict_undercharged"] == 1.0
    assert summary[f"{PREFIX}_corpus_current_subtoken_margin_over_public_dict_charged_bits"] >= 170000.0
    assert summary[f"{PREFIX}_corpus_current_subtoken_margin_over_public_dict_undercharged_bits"] >= 137000.0
    assert summary[f"{PREFIX}_source_code_public_baseline_audit_authorized"] == 1.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0


def test_zstd_trained_dictionary_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_block_current_subtoken_beats_public_dict_charged"] == 1.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_corpus_current_subtoken_beats_public_dict_charged"] == 1.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
