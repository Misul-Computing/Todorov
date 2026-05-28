from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import (
    PROFILES,
    MarginRecompressionAdapterCell,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    build_facts,
    build_summary,
    paraphrase_questions,
    score_answers,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_margin_recompression_adapter"
PREFIX = "local_100k_margin_recompression_adapter"


def questions_for(facts: list[dict]) -> list[str]:
    return [str(row["question"]) for row in facts]


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_margin_facts_are_disjoint_stable_and_field_safe() -> None:
    first_train, first_test, first_block, first_profile = build_facts(2137, 512)
    second_train, second_test, second_block, second_profile = build_facts(2137, 512)
    forbidden = {
        "source",
        "source_id",
        "offset",
        "block_offset",
        "content_window_digest",
        "answer_digest",
        "key",
        "assignment_key",
        "routing_key",
        "payload_row",
        "manifest_row",
        "stored_manifest",
        "external_payload",
        "semantic_handle",
    }
    assert first_train == []
    assert first_train == second_train
    assert first_test == second_test
    assert first_block == second_block
    assert first_profile == second_profile
    assert len({str(row["domain"]) for row in first_test}) >= 4
    assert len({str(row["path"]) for row in first_profile}) == len(first_profile)
    assert len({str(row["sha256"]) for row in first_profile}) == len(first_profile)
    assert all({"role", "row", "domain", "question", "value", "provenance"} <= set(row) for row in first_test)
    assert all(forbidden.isdisjoint(row) for row in first_test)


def test_margin_cell_answers_and_rejects_invalid_queries() -> None:
    train, facts, source_block, profile = build_facts(2137, 512)
    cell = MarginRecompressionAdapterCell(train, facts, source_block, profile)
    exact_answers = cell.answer_many(questions_for(facts))
    paraphrase_answers = cell.answer_many(paraphrase_questions(facts))
    partial_answers = cell.answer_many(["partial overlap evidence terms: memory state" for _fact in facts])
    assert exact_success_count(facts, exact_answers) == len(facts)
    assert exact_success_count(facts, paraphrase_answers) == len(facts)
    assert sum(int(row["hit"]) for row in partial_answers) == 0
    assert cell.source_train_test_path_overlap_count == 0
    assert cell.source_train_test_hash_overlap_count == 0
    assert cell.source_train_test_ngram_overlap_count == 0
    assert cell.trainable_recompression_controller_used == 1.0
    assert cell.model_state_adapter_payload_used == 1.0
    assert not hasattr(cell, "raw_source_block")


def test_margin_hosts_carry_payload_and_update_controller() -> None:
    train, facts, source_block, profile = build_facts(2137, 512)
    cell = MarginRecompressionAdapterCell(train, facts, source_block, profile)
    transformer = TinyTransformerAdapterHost(cell)
    recurrent = TinyRecurrentStateAdapterHost(cell)
    transformer_keys = set(transformer.module.state_dict().keys())
    recurrent_keys = set(recurrent.module.state_dict().keys())
    assert transformer.parameter_count() < 100000
    assert recurrent.parameter_count() < 100000
    assert "adapter_module.adapter_payload" in transformer_keys
    assert "adapter_module.adapter_payload" in recurrent_keys
    assert any("update_controller" in key for key in transformer_keys)
    assert any("update_controller" in key for key in recurrent_keys)
    assert exact_success_count(facts, transformer.answer_many(paraphrase_questions(facts))) == len(facts)
    assert exact_success_count(facts, recurrent.answer_many(paraphrase_questions(facts))) == len(facts)


def test_margin_summary_reports_high_margin_candidate_and_limits() -> None:
    summary = build_summary("smoke", seed=2137)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_paper_ready_candidate"] == 0.0
    assert summary[f"{PREFIX}_paper_ready_local_candidate_authorized"] == 0.0
    assert summary[f"{PREFIX}_bounded_adapter_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_static_compression_publishable_candidate"] == 0.0
    assert summary[f"{PREFIX}_paper_ready_requirement_count"] == 5.0
    assert summary[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_static_public_baseline_pass"] == 1.0
    assert summary[f"{PREFIX}_multi_domain_pass"] == 1.0
    assert summary[f"{PREFIX}_source_holdout_pass"] == 1.0
    assert summary[f"{PREFIX}_large_margin_over_mph_pass"] == 1.0
    assert summary[f"{PREFIX}_paraphrase_or_update_pass"] == 1.0
    assert summary[f"{PREFIX}_ablation_controls_pass"] == 1.0
    assert summary[f"{PREFIX}_paper_surface_strict_multiplier"] >= 22.0
    assert summary[f"{PREFIX}_adapter_strict_multiplier"] >= 22.0
    assert summary[f"{PREFIX}_trainable_recompression_update_success"] == 1.0
    assert summary[f"{PREFIX}_matched_update_recompress_baseline_success"] == 1.0
    assert summary[f"{PREFIX}_matched_update_baseline_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_matched_update_recompress_baseline_bits"] < summary[f"{PREFIX}_trainable_recompression_update_bits"]
    assert summary[f"{PREFIX}_publishable_update_adapter_candidate"] == 0.0
    assert summary[f"{PREFIX}_wrong_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_partial_overlap_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_content_scan_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_beats_same_block_undercharged_mph_baseline"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_learned_semantic_retrieval_authorized"] == 0.0


def test_margin_profiles_and_registry_contract() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == "local_100k_margin_recompression_adapter_metrics.json"
    assert minimum[f"{PREFIX}_bounded_adapter_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_paper_ready_requirement_count"] == 5.0
    assert minimum[f"{PREFIX}_large_margin_over_mph_pass"] == 1.0
    assert minimum[f"{PREFIX}_paper_surface_strict_multiplier"] >= 22.0
    assert minimum[f"{PREFIX}_trainable_recompression_update_success"] == 1.0
    assert minimum[f"{PREFIX}_matched_update_recompress_baseline_success"] == 1.0
    assert minimum[f"{PREFIX}_matched_update_baseline_not_beaten"] == 1.0
    assert maximum[f"{PREFIX}_paper_ready_candidate"] == 0.0
    assert maximum[f"{PREFIX}_paper_ready_local_candidate_authorized"] == 0.0
    assert maximum[f"{PREFIX}_static_compression_publishable_candidate"] == 0.0
    assert maximum[f"{PREFIX}_publishable_update_adapter_candidate"] == 0.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_learned_semantic_retrieval_authorized"] == 0.0
    assert maximum[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
