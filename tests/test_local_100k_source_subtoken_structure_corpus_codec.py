from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import (
    SIMULATION_ID,
    build_summary,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_subtoken_structure_corpus_codec"


def test_source_subtoken_structure_corpus_summary_passes_frozen_corpus_gate() -> None:
    summary = build_summary("hard", seed=10547)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_block_count"] == 5.0
    assert summary[f"{PREFIX}_exact_reconstruction_success_min"] == 1.0
    assert summary[f"{PREFIX}_frozen_manifest_hash_success_min"] == 1.0
    assert summary[f"{PREFIX}_aggregate_standard_payload_bits"] == 849752.0
    assert summary[f"{PREFIX}_aggregate_selected_payload_bits"] == 812688.0
    assert summary[f"{PREFIX}_aggregate_payload_improvement"] > 0.043
    assert summary[f"{PREFIX}_subtoken_structure_selected_block_count"] == 5.0
    assert summary[f"{PREFIX}_standard_fallback_selected_block_count"] == 0.0
    assert summary[f"{PREFIX}_selector_bits_per_block"] == 16.0
    assert summary[f"{PREFIX}_standard_codec_header_bits_per_block"] == 16.0
    assert summary[f"{PREFIX}_random_label_payload_control_required"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_incompressible_min"] == 1.0
    assert summary[f"{PREFIX}_random_label_payload_improvement_over_best_standard_max"] <= 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_paid_compute_authorized"] == 0.0


def test_source_subtoken_structure_corpus_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_frozen_manifest_hash_success_min"] == 1.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
