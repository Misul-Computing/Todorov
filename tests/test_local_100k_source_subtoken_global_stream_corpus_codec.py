from neuroloc.simulations.memory.local_100k_source_subtoken_global_stream_corpus_codec import (
    SIMULATION_ID,
    SourceSubtokenGlobalStreamCorpusModule,
    build_summary,
    global_codec,
    read_limited_block,
    target_rows,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_subtoken_global_stream_corpus_codec"


def test_source_subtoken_global_stream_corpus_summary_passes_product_gate() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_block_count"] == 5.0
    assert summary[f"{PREFIX}_exact_reconstruction_success_min"] == 1.0
    assert summary[f"{PREFIX}_frozen_manifest_hash_success_min"] == 1.0
    assert summary[f"{PREFIX}_aggregate_standard_payload_bits"] == 849752.0
    assert summary[f"{PREFIX}_global_raw_standard_payload_bits"] == 736504.0
    assert summary[f"{PREFIX}_prior_shared_dictionary_corpus_payload_bits"] == 803400.0
    assert summary[f"{PREFIX}_prior_subtoken_corpus_payload_bits"] == 812688.0
    assert summary[f"{PREFIX}_aggregate_selected_payload_bits"] == 699144.0
    assert summary[f"{PREFIX}_aggregate_payload_improvement"] > 0.17
    assert summary[f"{PREFIX}_global_raw_standard_payload_improvement"] > 0.045
    assert summary[f"{PREFIX}_margin_over_global_raw_standard_bits"] == 37360.0
    assert summary[f"{PREFIX}_aggregate_payload_margin_over_prior_shared_bits"] == 104256.0
    assert summary[f"{PREFIX}_aggregate_payload_margin_over_prior_subtoken_bits"] == 113544.0
    assert summary[f"{PREFIX}_aggregate_payload_improvement_delta_over_prior_shared"] > 0.12
    assert summary[f"{PREFIX}_shared_token_count"] == 256.0
    assert summary[f"{PREFIX}_one_byte_token_count"] == 120.0
    assert summary[f"{PREFIX}_local_token_count_per_block"] == 0.0
    assert summary[f"{PREFIX}_shared_dictionary_payload_bits"] == 10400.0
    assert summary[f"{PREFIX}_global_count_payload_bits"] == 21272.0
    assert summary[f"{PREFIX}_global_body_payload_bits"] == 665112.0
    assert summary[f"{PREFIX}_global_length_payload_bits"] == 312.0
    assert summary[f"{PREFIX}_global_header_bits"] == 2048.0
    assert summary[f"{PREFIX}_global_raw_standard_header_bits"] == 64.0
    assert summary[f"{PREFIX}_model_state_codec_payload_used"] == 1.0
    assert summary[f"{PREFIX}_state_dict_buffer_payload_used"] == 1.0
    assert summary[f"{PREFIX}_model_state_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_reload_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_product_authorized"] == 1.0


def test_source_subtoken_global_stream_controls_and_random_label() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard"] <= 0.0
    assert summary[f"{PREFIX}_random_label_global_raw_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_global_raw_payload_improvement"] <= 0.0
    assert summary[f"{PREFIX}_random_label_global_raw_standard_payload_bits"] == 6420936.0
    assert summary[f"{PREFIX}_random_label_selected_payload_bits"] == 6422920.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_wrong_indent_unit_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shared_dictionary_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_shared_dictionary_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_body_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_count_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_length_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 0.0
    assert summary[f"{PREFIX}_seed_oracle_authorized"] == 0.0


def test_source_subtoken_global_stream_beats_public_dictionary_audit_lines() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_zstd_charged_public_baseline_bits"] == 982840.0
    assert summary[f"{PREFIX}_zstd_undercharged_public_baseline_bits"] == 949992.0
    assert summary[f"{PREFIX}_margin_over_zstd_charged_public_bits"] == 283696.0
    assert summary[f"{PREFIX}_margin_over_zstd_undercharged_public_bits"] == 250848.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_paid_compute_authorized"] == 0.0


def test_source_subtoken_global_stream_corpus_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_margin_over_global_raw_standard_bits"] == 15000.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_aggregate_payload_margin_over_prior_shared_bits"] == 90000.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_shared_token_count"] == 256.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_one_byte_token_count"] == 120.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_state_dict_reload_reconstruction_success"] == 1.0
    assert dict(spec.hard_minimum_summary_values)[f"{PREFIX}_block_count"] == 5.0
    assert dict(spec.hard_minimum_summary_values)[f"{PREFIX}_margin_over_global_raw_standard_bits"] == 30000.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0


def test_source_subtoken_global_stream_reconstructs_from_state_dict_only() -> None:
    blocks = [read_limited_block(row) for row in target_rows("hard")]
    codec = global_codec(blocks)
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    assert reload_module.reconstruct() == blocks
    assert set(state.keys()) == {"global_header", "shared_dictionary_payload", "count_payload", "body_payload", "length_payload"}
