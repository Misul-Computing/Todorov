from neuroloc.simulations.memory.local_100k_source_token_structure_corpus_codec import (
    SIMULATION_ID,
    build_summary,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_token_structure_corpus_codec"


def test_source_token_structure_corpus_summary_passes_broad_codec_gate() -> None:
    summary = build_summary("hard", seed=10433)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_block_count"] == 5.0
    assert summary[f"{PREFIX}_exact_reconstruction_success_min"] == 1.0
    assert summary[f"{PREFIX}_aggregate_standard_payload_bits"] == 857648.0
    assert summary[f"{PREFIX}_aggregate_selected_payload_bits"] == 832512.0
    assert summary[f"{PREFIX}_aggregate_payload_improvement"] > 0.029
    assert summary[f"{PREFIX}_token_structure_selected_block_count"] == 4.0
    assert summary[f"{PREFIX}_standard_fallback_selected_block_count"] == 1.0
    assert summary[f"{PREFIX}_selector_bits_per_block"] == 16.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_product_authorized"] == 0.0
    assert summary[f"{PREFIX}_random_label_payload_control_required"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 0.0
    assert summary[f"{PREFIX}_source_code_corpus_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0


def test_source_token_structure_corpus_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_engineering_pass"] == 1.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_source_code_corpus_codec_product_authorized"] == 0.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
