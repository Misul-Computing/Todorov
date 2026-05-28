from neuroloc.simulations.memory.local_100k_semantic_alias_payload_adapter import (
    PROFILES,
    SemanticAliasPayloadAdapterCell,
    alias_content_scan_answers,
    build_facts,
    build_summary,
    evidence_leakage_rate,
    lexical_content_scan_answers,
    public_facts,
    score_answers,
    transform_payload,
    compress_payload,
    decompress_payload,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import TinyRecurrentStateAdapterHost, TinyTransformerAdapterHost
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_semantic_alias_payload_adapter"
PREFIX = "local_100k_semantic_alias_payload_adapter"


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_semantic_alias_payload_transform_is_reversible_and_smaller() -> None:
    _train, facts, source_block, _profile = build_facts(2719, 512)
    encoded = transform_payload(source_block)
    restored = decompress_payload(compress_payload(source_block))
    assert restored == source_block
    assert len(encoded) < len(source_block)
    assert evidence_leakage_rate(facts, source_block) == 0.0


def test_semantic_alias_facts_are_public_field_safe() -> None:
    first_train, first_test, first_block, first_profile = build_facts(2719, 512)
    second_train, second_test, second_block, second_profile = build_facts(2719, 512)
    public = public_facts(first_test)
    forbidden = {
        "source",
        "source_id",
        "offset",
        "offset_for_test_only",
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
    assert len({str(row["domain"]) for row in public}) >= 4
    assert all({"role", "row", "domain", "question", "value", "provenance"} <= set(row) for row in public)
    assert all(forbidden.isdisjoint(row) for row in public)


def test_semantic_alias_cell_answers_and_lexical_scan_fails() -> None:
    train, facts_with_private, source_block, profile = build_facts(2719, 512)
    facts = public_facts(facts_with_private)
    cell = SemanticAliasPayloadAdapterCell(train, facts, source_block, profile)
    questions = [str(row["question"]) for row in facts]
    exact_answers = cell.answer_many(questions)
    lexical_answers = lexical_content_scan_answers(source_block, questions)
    alias_answers = alias_content_scan_answers(source_block, questions)
    invalid_answers = cell.answer_many([str(row["question"]) + " injected" for row in facts])
    assert exact_success_count(facts, exact_answers) == len(facts)
    assert exact_success_count(facts, lexical_answers) == 0
    assert exact_success_count(facts, alias_answers) == len(facts)
    assert sum(int(row["hit"]) for row in invalid_answers) == 0
    assert cell.model_state_adapter_payload_used == 1.0
    assert cell.raw_source_block_retained == 0.0
    assert not hasattr(cell, "raw_source_block")
    assert not hasattr(cell, "alias_to_token")


def test_semantic_alias_hosts_reload_payload() -> None:
    train, facts_with_private, source_block, profile = build_facts(2719, 512)
    facts = public_facts(facts_with_private)
    cell = SemanticAliasPayloadAdapterCell(train, facts, source_block, profile)
    transformer = TinyTransformerAdapterHost(cell)
    recurrent = TinyRecurrentStateAdapterHost(cell)
    transformer_keys = set(transformer.module.state_dict().keys())
    recurrent_keys = set(recurrent.module.state_dict().keys())
    questions = [str(row["question"]) for row in facts]
    assert transformer.parameter_count() < 100000
    assert recurrent.parameter_count() < 100000
    assert "adapter_module.adapter_payload" in transformer_keys
    assert "adapter_module.adapter_payload" in recurrent_keys
    assert exact_success_count(facts, transformer.answer_many(questions)) == len(facts)
    assert exact_success_count(facts, recurrent.answer_many(questions)) == len(facts)


def test_semantic_alias_summary_reports_breakthrough_candidate() -> None:
    summary = build_summary("smoke", seed=2719)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_publishable_breakthrough_candidate"] == 0.0
    assert summary[f"{PREFIX}_semantic_alias_diagnostic_candidate"] == 1.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_paper_ready_requirement_count"] == 4.0
    assert summary[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_static_public_baseline_pass"] == 0.0
    assert summary[f"{PREFIX}_multi_domain_pass"] == 1.0
    assert summary[f"{PREFIX}_lexical_content_scan_beaten"] == 1.0
    assert summary[f"{PREFIX}_fair_alias_content_scan_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_alias_content_scan_success"] == 1.0
    assert summary[f"{PREFIX}_same_block_undercharged_mph_beaten"] == 1.0
    assert summary[f"{PREFIX}_previous_content_scan_line_beaten"] == 1.0
    assert summary[f"{PREFIX}_evidence_token_leakage_rate"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_wrong_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_partial_overlap_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_paper_surface_strict_multiplier"] > 22.737
    assert summary[f"{PREFIX}_adapter_strict_multiplier"] > summary[f"{PREFIX}_mph_undercharged_strict_multiplier"]
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 1.0
    assert summary[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0


def test_semantic_alias_profiles_and_registry_contract() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == "local_100k_semantic_alias_payload_adapter_metrics.json"
    assert minimum[f"{PREFIX}_semantic_alias_diagnostic_candidate"] == 1.0
    assert minimum[f"{PREFIX}_lexical_content_scan_beaten"] == 1.0
    assert minimum[f"{PREFIX}_fair_alias_content_scan_not_beaten"] == 1.0
    assert minimum[f"{PREFIX}_same_block_undercharged_mph_beaten"] == 1.0
    assert minimum[f"{PREFIX}_previous_content_scan_line_beaten"] == 1.0
    assert maximum[f"{PREFIX}_publishable_breakthrough_candidate"] == 0.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert maximum[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
