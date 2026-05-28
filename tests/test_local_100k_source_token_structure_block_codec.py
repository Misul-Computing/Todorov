from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import (
    SIMULATION_ID,
    SourceTokenStructurePayloadModule,
    build_summary,
    learned_codec,
    read_joined,
    restore_learned,
    target_paths,
    train_paths,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_token_structure_block_codec"


def source_blocks() -> tuple[bytes, bytes]:
    return read_joined(train_paths()), read_joined(target_paths("hard"))


def test_source_token_structure_codec_restores_exact_bytes() -> None:
    train, target = source_blocks()
    learned = learned_codec(train, target)
    assert restore_learned(learned) == target
    module = SourceTokenStructurePayloadModule(learned)
    reload_module = SourceTokenStructurePayloadModule.empty_like(module)
    reload_module.load_state_dict(module.state_dict())
    assert module.restore() == target
    assert reload_module.restore() == target


def test_source_token_structure_codec_charges_token_dictionary_and_beats_source_structure() -> None:
    summary = build_summary("hard", seed=9311)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_model_state_reload_success"] == 1.0
    assert summary[f"{PREFIX}_best_standard_payload_bits"] == 128816.0
    assert summary[f"{PREFIX}_source_structure_payload_bits"] == 124200.0
    assert summary[f"{PREFIX}_learned_payload_bits"] == 123088.0
    assert summary[f"{PREFIX}_target_charged_token_count"] == 120.0
    assert summary[f"{PREFIX}_target_charged_dictionary_accounted"] == 1.0
    assert summary[f"{PREFIX}_train_free_dictionary_bits"] == 0.0
    assert summary[f"{PREFIX}_strict_improvement_over_best_standard"] > summary[f"{PREFIX}_source_structure_strict_improvement_baseline"]
    assert summary[f"{PREFIX}_beats_source_structure_strict_margin"] == 1.0


def test_source_token_structure_controls_and_limits() -> None:
    summary = build_summary("hard", seed=9311)
    assert summary[f"{PREFIX}_random_label_payload_incompressible"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard"] < 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_decoder_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_wrong_indent_unit_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_token_dictionary_disabled_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_body_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_count_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_shuffle_dictionary_payload_exact_reconstruction_success"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_raw_target_block"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_uncompressed_count_stream"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_uncompressed_body_stream"] == 0.0
    assert summary[f"{PREFIX}_codec_state_has_restored_block"] == 0.0
    assert summary[f"{PREFIX}_source_block_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0


def test_source_token_structure_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
