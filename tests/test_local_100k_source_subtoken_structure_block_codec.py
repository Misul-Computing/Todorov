from neuroloc.simulations.memory.local_100k_source_subtoken_structure_block_codec import (
    SIMULATION_ID,
    build_summary,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_subtoken_structure_block_codec"


def test_source_subtoken_structure_block_summary_passes_product_gate() -> None:
    summary = build_summary("hard", seed=9419)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_model_state_restore_success"] == 1.0
    assert summary[f"{PREFIX}_model_state_reload_success"] == 1.0
    assert summary[f"{PREFIX}_target_block_bytes"] == 99761.0
    assert summary[f"{PREFIX}_best_standard_payload_bits"] == 128816.0
    assert summary[f"{PREFIX}_source_token_payload_bits"] == 123088.0
    assert summary[f"{PREFIX}_learned_payload_bits"] == 120952.0
    assert summary[f"{PREFIX}_strict_improvement_over_best_standard"] > 0.048
    assert summary[f"{PREFIX}_payload_improvement_over_best_standard"] > 0.061
    assert summary[f"{PREFIX}_strict_improvement_delta_over_source_token"] > 0.013
    assert summary[f"{PREFIX}_payload_improvement_delta_over_source_token"] > 0.017
    assert summary[f"{PREFIX}_beats_source_token_strict_margin"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard"] < 0.0
    assert summary[f"{PREFIX}_target_charged_dictionary_accounted"] == 1.0
    assert summary[f"{PREFIX}_train_free_dictionary_bits"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_token_dictionary_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_body_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_count_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_dictionary_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_raw_target_block"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_uncompressed_count_stream"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_uncompressed_body_stream"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_restored_block"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_block_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0


def test_source_subtoken_structure_block_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_strict_improvement_delta_over_source_token"] == 0.013
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
