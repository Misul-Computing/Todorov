from neuroloc.simulations.memory.local_100k_source_subtoken_delta_update_adapter import (
    DELTA_PATCH_HEADER_BITS,
    SIMULATION_ID,
    SourceSubtokenDeltaUpdateAdapterCell,
    build_facts,
    build_summary,
    expected_fact_rows,
    score_answers,
    state_dict_reload_probe,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_subtoken_delta_update_adapter"


def questions_for(facts: list[dict]) -> list[str]:
    return [str(row["question"]) for row in facts]


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_delta_update_cell_answers_updated_and_unchanged_from_patch_state() -> None:
    train, facts, source_block, profile = build_facts(2137, 512)
    cell = SourceSubtokenDeltaUpdateAdapterCell(train, facts, source_block, profile, update_fact_count=16)
    updated, unchanged = expected_fact_rows(cell, facts)
    updated_answers = cell.answer_many(questions_for(updated))
    unchanged_answers = cell.answer_many(questions_for(unchanged))
    assert len(updated) == 16
    assert exact_success_count(updated, updated_answers) == len(updated)
    assert exact_success_count(unchanged, unchanged_answers) == len(unchanged)
    assert exact_success_count(updated, cell.answer_many(questions_for(updated), patch_disabled=True)) == 0
    assert exact_success_count(updated, cell.answer_many(questions_for(updated), random_patch=True)) == 0
    assert exact_success_count(updated, cell.answer_many(questions_for(updated), shuffled_patch=True)) == 0
    assert cell.model_state_patch_payload_used == 1.0
    assert "delta_patch_payload" in cell.module.state_dict()
    assert "delta_patch_header" in cell.module.state_dict()
    assert cell.module.state_dict()["delta_patch_header"].element_size() * cell.module.state_dict()["delta_patch_header"].numel() * 8 <= DELTA_PATCH_HEADER_BITS
    assert cell.delta_patch_bits < int(DELTA_PATCH_HEADER_BITS + 16 * (4 + 32) * 8)
    assert cell.delta_patch_bits > int(DELTA_PATCH_HEADER_BITS + 16 * 32 * 8)
    assert cell.total_updated_adapter_bits == cell.base_payload_bits + cell.delta_patch_bits
    assert len(cell.patch_payload_bytes()) < 16 * (4 + 32)
    assert len(cell.patch_payload_bytes()) > 16 * 32


def test_delta_update_state_dict_reload_uses_patch_payload() -> None:
    train, facts, source_block, profile = build_facts(2137, 512)
    cell = SourceSubtokenDeltaUpdateAdapterCell(train, facts, source_block, profile, update_fact_count=16)
    probe = state_dict_reload_probe(cell, train, facts, source_block, profile, 16)
    assert probe["state_dict_preload_success"] == 0.0
    assert probe["state_dict_reload_success"] == 1.0
    assert probe["patch_payload_in_state_dict"] == 1.0
    assert probe["patch_header_in_state_dict"] == 1.0


def test_delta_update_summary_reports_product_and_category_limits() -> None:
    summary = build_summary("smoke", seed=2137)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_source_subtoken_delta_update_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_update_fact_count"] == 64.0
    assert summary[f"{PREFIX}_exact_updated_answer_success"] == 1.0
    assert summary[f"{PREFIX}_unchanged_answer_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_reload_success"] == 1.0
    assert summary[f"{PREFIX}_random_patch_control_success"] == 0.0
    assert summary[f"{PREFIX}_patch_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_patch_success"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_model_state_patch_payload_used"] == 1.0
    assert summary[f"{PREFIX}_delta_patch_bits"] < int(DELTA_PATCH_HEADER_BITS + 64 * (4 + 32) * 8)
    assert summary[f"{PREFIX}_delta_patch_bits"] > int(DELTA_PATCH_HEADER_BITS + 64 * 32 * 8)
    assert summary[f"{PREFIX}_total_updated_adapter_bits"] == summary[f"{PREFIX}_base_payload_bits"] + summary[f"{PREFIX}_delta_patch_bits"]
    assert summary[f"{PREFIX}_margin_over_full_recompress_bits"] > 0.0
    assert summary[f"{PREFIX}_same_block_content_scan_update_bits"] == summary[f"{PREFIX}_full_recompress_updated_bits"]
    assert summary[f"{PREFIX}_margin_over_same_block_content_scan_update_bits"] == summary[f"{PREFIX}_margin_over_full_recompress_bits"]
    assert summary[f"{PREFIX}_same_block_content_scan_update_beaten"] == 1.0
    assert summary[f"{PREFIX}_margin_over_undercharged_mph_update_bits"] > 0.0
    assert summary[f"{PREFIX}_undercharged_mph_update_beaten"] == 1.0
    assert summary[f"{PREFIX}_matched_delta_patch_content_scan_bits"] == summary[f"{PREFIX}_delta_patch_bits"]
    assert summary[f"{PREFIX}_margin_over_matched_delta_patch_content_scan_bits"] == 0.0
    assert summary[f"{PREFIX}_matched_delta_patch_content_scan_beaten"] == 0.0
    assert summary[f"{PREFIX}_total_static_margin_over_full_recompress_bits"] > 0.0
    assert summary[f"{PREFIX}_source_subtoken_total_static_compression_authorized"] == 0.0
    assert summary[f"{PREFIX}_static_compression_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0


def test_delta_update_hard_boundary_remains_not_static_compression() -> None:
    summary = build_summary("hard", seed=2137)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_update_fact_count"] == 512.0
    assert summary[f"{PREFIX}_total_static_margin_over_full_recompress_bits"] < 0.0
    assert summary[f"{PREFIX}_matched_delta_patch_content_scan_bits"] == summary[f"{PREFIX}_delta_patch_bits"]
    assert summary[f"{PREFIX}_matched_delta_patch_content_scan_beaten"] == 0.0
    assert summary[f"{PREFIX}_source_subtoken_total_static_compression_authorized"] == 0.0
    assert summary[f"{PREFIX}_static_compression_breakthrough_authorized"] == 0.0


def test_delta_update_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert minimum[f"{PREFIX}_source_subtoken_delta_update_product_authorized"] == 1.0
    assert minimum[f"{PREFIX}_exact_updated_answer_success"] == 1.0
    assert minimum[f"{PREFIX}_state_dict_reload_success"] == 1.0
    assert minimum[f"{PREFIX}_margin_over_full_recompress_bits"] == 1.0
    assert maximum[f"{PREFIX}_random_patch_control_success"] == 0.0
    assert maximum[f"{PREFIX}_patch_disabled_success"] == 0.0
    assert maximum[f"{PREFIX}_shuffled_patch_success"] == 0.0
    assert maximum[f"{PREFIX}_full_nm_authorized"] == 0.0
