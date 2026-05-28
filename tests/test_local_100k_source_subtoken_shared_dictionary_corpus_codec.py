from neuroloc.simulations.memory.local_100k_source_subtoken_shared_dictionary_corpus_codec import (
    SIMULATION_ID,
    build_summary,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_subtoken_shared_dictionary_corpus_codec"


def test_source_subtoken_shared_dictionary_corpus_summary_passes_product_gate() -> None:
    summary = build_summary("hard", seed=12641)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_block_count"] == 5.0
    assert summary[f"{PREFIX}_exact_reconstruction_success_min"] == 1.0
    assert summary[f"{PREFIX}_frozen_manifest_hash_success_min"] == 1.0
    assert summary[f"{PREFIX}_aggregate_standard_payload_bits"] == 849752.0
    assert summary[f"{PREFIX}_prior_subtoken_corpus_payload_bits"] == 812688.0
    assert summary[f"{PREFIX}_aggregate_selected_payload_bits"] == 803400.0
    assert summary[f"{PREFIX}_aggregate_payload_improvement"] > 0.054
    assert summary[f"{PREFIX}_aggregate_payload_margin_over_prior_bits"] == 9288.0
    assert summary[f"{PREFIX}_aggregate_payload_improvement_delta_over_prior"] > 0.011
    assert summary[f"{PREFIX}_shared_token_count"] == 112.0
    assert summary[f"{PREFIX}_local_token_count_per_block"] == 16.0
    assert summary[f"{PREFIX}_shared_dictionary_payload_bits"] == 4360.0
    assert summary[f"{PREFIX}_shared_header_bits"] == 896.0
    assert summary[f"{PREFIX}_local_header_bits_per_block"] == 16.0
    assert summary[f"{PREFIX}_selector_bits_per_block"] == 16.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_product_authorized"] == 1.0


def test_source_subtoken_shared_dictionary_corpus_controls_and_random_label() -> None:
    summary = build_summary("hard", seed=12641)
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_wrong_indent_unit_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shared_dictionary_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_shared_dictionary_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_body_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 0.0
    assert summary[f"{PREFIX}_seed_oracle_authorized"] == 0.0


def test_source_subtoken_shared_dictionary_beats_public_dictionary_audit_lines() -> None:
    summary = build_summary("hard", seed=12641)
    assert summary[f"{PREFIX}_zstd_charged_public_baseline_bits"] == 982840.0
    assert summary[f"{PREFIX}_zstd_undercharged_public_baseline_bits"] == 949992.0
    assert summary[f"{PREFIX}_margin_over_zstd_charged_public_bits"] == 179440.0
    assert summary[f"{PREFIX}_margin_over_zstd_undercharged_public_bits"] == 146592.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_paid_compute_authorized"] == 0.0


def test_source_subtoken_shared_dictionary_corpus_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_aggregate_payload_margin_over_prior_bits"] == 1000.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
