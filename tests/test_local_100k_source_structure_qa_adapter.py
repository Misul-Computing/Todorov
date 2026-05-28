from neuroloc.simulations.memory.local_100k_source_structure_qa_adapter import (
    SIMULATION_ID,
    SourceStructureQAAdapterCell,
    build_facts,
    build_summary,
    raw_baseline_metrics,
    score_answers,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_structure_qa_adapter"


def questions_for(facts: list[dict]) -> list[str]:
    return [str(row["question"]) for row in facts]


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_source_structure_qa_cell_answers_from_compressed_state() -> None:
    train, facts, source_block, profile = build_facts(2137, 512)
    cell = SourceStructureQAAdapterCell(train, facts, source_block, profile)
    answers = cell.answer_many(questions_for(facts))
    assert exact_success_count(facts, answers) == len(facts)
    assert cell.structure_codec_used == 1.0
    assert cell.model_state_adapter_payload_used == 1.0
    assert cell.external_payload_store_used == 0.0
    assert cell.raw_source_block_retained == 0.0
    assert not hasattr(cell, "raw_source_block")


def test_source_structure_qa_payload_beats_raw_scan_but_not_same_structure_scan() -> None:
    train, facts, source_block, profile = build_facts(2137, 512)
    cell = SourceStructureQAAdapterCell(train, facts, source_block, profile)
    useful_bits = len(facts) * 32 * 8
    raw = raw_baseline_metrics(useful_bits, source_block)
    assert cell.block_payload_bits < raw["raw_best_standard_payload_bits"]
    assert cell.block_payload_bits == 246328
    assert raw["raw_best_standard_payload_bits"] == 262336


def test_source_structure_qa_summary_reports_product_and_limits() -> None:
    summary = build_summary("smoke", seed=2137)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_structure_qa_product_candidate"] == 1.0
    assert summary[f"{PREFIX}_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_heldout_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_paraphrase_stable_answer_success"] == 1.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_raw_content_scan_beaten"] == 1.0
    assert summary[f"{PREFIX}_raw_undercharged_mph_beaten"] == 1.0
    assert summary[f"{PREFIX}_same_structure_content_scan_beaten"] == 0.0
    assert summary[f"{PREFIX}_same_structure_content_scan_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_adapter_strict_multiplier"] > summary[f"{PREFIX}_raw_executable_content_scan_multiplier"]
    assert summary[f"{PREFIX}_adapter_strict_multiplier"] <= summary[f"{PREFIX}_same_structure_content_scan_multiplier"]
    assert summary[f"{PREFIX}_model_state_adapter_payload_used"] == 1.0
    assert summary[f"{PREFIX}_state_dict_buffer_payload_used"] == 1.0
    assert summary[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert summary[f"{PREFIX}_trainable_recompression_update_success"] == 1.0
    assert summary[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_source_block_codec_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0


def test_source_structure_qa_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert dict(spec.minimum_summary_values)[f"{PREFIX}_raw_content_scan_beaten"] == 1.0
    assert dict(spec.maximum_summary_values)[f"{PREFIX}_same_structure_content_scan_beaten"] == 0.0
