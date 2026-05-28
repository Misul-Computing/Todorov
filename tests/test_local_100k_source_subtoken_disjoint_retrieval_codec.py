from neuroloc.simulations.memory.local_100k_source_subtoken_disjoint_retrieval_codec import (
    SIMULATION_ID,
    SourceSubtokenGlobalStreamCorpusModule,
    build_summary,
    global_codec,
    read_limited_block,
    retrieval_rows,
    target_rows,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_subtoken_disjoint_retrieval_codec"


def test_disjoint_retrieval_summary_passes_product_gate() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_block_count"] == 3.0
    assert summary[f"{PREFIX}_retrieval_fact_count"] == 14715.0
    assert summary[f"{PREFIX}_chunk_bytes"] == 32.0
    assert summary[f"{PREFIX}_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_heldout_chunk_retrieval_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_reload_chunk_retrieval_success"] == 1.0
    assert summary[f"{PREFIX}_source_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_selected_payload_bits"] == 429488.0
    assert summary[f"{PREFIX}_selected_retrieval_accounted_bits"] == 431536.0
    assert summary[f"{PREFIX}_standard_retrieval_accounted_bits"] == 473008.0
    assert summary[f"{PREFIX}_raw_content_scan_accounted_bits"] == 451128.0
    assert summary[f"{PREFIX}_undercharged_mph_accounted_bits"] == 451144.0
    assert summary[f"{PREFIX}_margin_over_standard_retrieval_bits"] == 41472.0
    assert summary[f"{PREFIX}_margin_over_raw_content_scan_bits"] == 19592.0
    assert summary[f"{PREFIX}_margin_over_undercharged_mph_bits"] == 19608.0
    assert summary[f"{PREFIX}_raw_content_scan_beaten"] == 1.0
    assert summary[f"{PREFIX}_undercharged_mph_beaten"] == 1.0
    assert summary[f"{PREFIX}_strict_multiplier"] > 55.0
    assert summary[f"{PREFIX}_source_code_retrieval_codec_product_authorized"] == 1.0


def test_disjoint_retrieval_controls_and_limits() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_selected_retrieval_bits"] >= summary[f"{PREFIX}_random_label_standard_retrieval_bits"]
    assert summary[f"{PREFIX}_random_label_selected_retrieval_bits"] >= summary[f"{PREFIX}_random_label_raw_content_scan_bits"]
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_wrong_indent_unit_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shared_dictionary_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_shared_dictionary_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_body_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_count_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_length_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_state_dict_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_source_code_retrieval_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_static_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_paid_compute_authorized"] == 0.0


def test_disjoint_retrieval_reconstructs_chunks_from_state_dict_only() -> None:
    blocks = [read_limited_block(row) for row in target_rows("hard")]
    rows = retrieval_rows(blocks)
    codec = global_codec(blocks)
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    restored = reload_module.reconstruct()
    assert set(state.keys()) == {"global_header", "shared_dictionary_payload", "count_payload", "body_payload", "length_payload"}
    assert restored == blocks
    for block_index, offset, expected in rows:
        assert restored[block_index][offset : offset + 32] == expected


def test_disjoint_retrieval_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert minimum[f"{PREFIX}_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_heldout_chunk_retrieval_success"] == 1.0
    assert minimum[f"{PREFIX}_raw_content_scan_beaten"] == 1.0
    assert minimum[f"{PREFIX}_undercharged_mph_beaten"] == 1.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
