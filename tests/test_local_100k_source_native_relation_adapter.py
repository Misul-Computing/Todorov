from neuroloc.simulations.memory.local_100k_source_native_relation_adapter import (
    PROFILES,
    SourceNativeRelationAdapterCell,
    build_facts,
    build_summary,
    public_facts,
    relation_paraphrases,
    relation_terms_from_question,
    relationless_content_scan_answers,
    score_answers,
    stride_aware_content_scan_answers,
    transform_payload,
    compress_payload,
    decompress_payload,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import TinyRecurrentStateAdapterHost, TinyTransformerAdapterHost
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_source_native_relation_adapter"
PREFIX = "local_100k_source_native_relation_adapter"


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_source_native_relation_generation_is_deterministic_and_transform_reversible() -> None:
    first = build_facts(3251, 512)
    second = build_facts(3251, 512)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
    encoded = transform_payload(first[2])
    restored = decompress_payload(compress_payload(first[2]))
    assert restored == first[2]
    assert len(encoded) < len(first[2])
    assert first[6] == 7


def test_source_native_public_facts_do_not_expose_offsets_or_aliases() -> None:
    _train, private_facts, _source_block, _profile, _train_manifest, _train_block, _stride = build_facts(3251, 512)
    public = public_facts(private_facts)
    forbidden = {
        "source",
        "source_id",
        "offset",
        "anchor_offset_for_test_only",
        "target_offset_for_test_only",
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
    assert len({str(row["domain"]) for row in public}) >= 4
    assert all({"role", "row", "domain", "question", "value", "provenance"} <= set(row) for row in public)
    assert all(forbidden.isdisjoint(row) for row in public)
    assert all(not str(row["question"]).startswith("q:") for row in public)


def test_source_native_cell_answers_and_relationless_scan_fails() -> None:
    train, private_facts, source_block, profile, _train_manifest, _train_block, stride = build_facts(3251, 512)
    facts = public_facts(private_facts)
    cell = SourceNativeRelationAdapterCell(train, facts, source_block, profile, learned_stride=stride)
    questions = [str(row["question"]) for row in facts]
    exact_answers = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(relation_paraphrases(facts))
    relationless_answers = relationless_content_scan_answers(source_block, questions)
    stride_aware_answers = stride_aware_content_scan_answers(source_block, questions, stride)
    wrong_stride_answers = cell.answer_many(questions, wrong_stride=True)
    router_disabled_answers = cell.answer_many(questions, router_disabled=True)
    injected_answers = cell.answer_many([str(row["question"]) + " injected" for row in facts])
    assert exact_success_count(facts, exact_answers) == len(facts)
    assert exact_success_count(facts, paraphrase_answers) == len(facts)
    assert exact_success_count(facts, relationless_answers) == 0
    assert exact_success_count(facts, stride_aware_answers) == len(facts)
    assert exact_success_count(facts, wrong_stride_answers) == 0
    assert exact_success_count(facts, router_disabled_answers) == 0
    assert sum(int(row["hit"]) for row in injected_answers) == 0
    assert cell.parameter_count() == 0
    assert cell.model_state_adapter_payload_used == 1.0
    assert cell.raw_source_block_retained == 0.0
    assert not hasattr(cell, "raw_source_block")


def test_source_native_question_parser_rejects_partial_and_extra_terms() -> None:
    assert relation_terms_from_question("source relation after evidence terms: alpha beta gamma") == ("alpha", "beta", "gamma")
    assert relation_terms_from_question("source relation after evidence terms: alpha beta") == tuple()
    assert relation_terms_from_question("source relation after evidence terms: alpha beta gamma delta") == tuple()


def test_source_native_hosts_reload_payload_and_router_code() -> None:
    train, private_facts, source_block, profile, _train_manifest, _train_block, stride = build_facts(3251, 512)
    facts = public_facts(private_facts)
    cell = SourceNativeRelationAdapterCell(train, facts, source_block, profile, learned_stride=stride)
    transformer = TinyTransformerAdapterHost(cell)
    recurrent = TinyRecurrentStateAdapterHost(cell)
    transformer_keys = set(transformer.module.state_dict().keys())
    recurrent_keys = set(recurrent.module.state_dict().keys())
    questions = [str(row["question"]) for row in facts]
    assert transformer.parameter_count() < 100000
    assert recurrent.parameter_count() < 100000
    assert "adapter_module.adapter_payload" in transformer_keys
    assert "adapter_module.relation_stride_code" in transformer_keys
    assert "adapter_module.adapter_payload" in recurrent_keys
    assert "adapter_module.relation_stride_code" in recurrent_keys
    assert exact_success_count(facts, transformer.answer_many(questions)) == len(facts)
    assert exact_success_count(facts, recurrent.answer_many(questions)) == len(facts)


def test_source_native_summary_reports_relation_breakthrough_candidate() -> None:
    summary = build_summary("smoke", seed=3251)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_publishable_relation_breakthrough_candidate"] == 0.0
    assert summary[f"{PREFIX}_formula_relation_diagnostic_candidate"] == 1.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_paper_ready_requirement_count"] == 5.0
    assert summary[f"{PREFIX}_legacy_static_public_baseline_pass"] == 1.0
    assert summary[f"{PREFIX}_static_public_baseline_pass"] == 0.0
    assert summary[f"{PREFIX}_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_paraphrase_stable_answer_success"] == 1.0
    assert summary[f"{PREFIX}_relationless_content_scan_success"] == 0.0
    assert summary[f"{PREFIX}_stride_aware_content_scan_success"] == 1.0
    assert summary[f"{PREFIX}_fair_stride_content_scan_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_wrong_stride_success"] == 0.0
    assert summary[f"{PREFIX}_router_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_source_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_ngram_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_paper_surface_strict_multiplier"] > 22.73766839237796
    assert summary[f"{PREFIX}_adapter_strict_multiplier"] > summary[f"{PREFIX}_same_block_undercharged_mph_multiplier"]
    assert summary[f"{PREFIX}_beats_same_block_undercharged_mph_baseline"] == 1.0
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 1.0
    assert summary[f"{PREFIX}_fixed_relation_constant_used"] == 1.0
    assert summary[f"{PREFIX}_learned_relation_router_used"] == 0.0
    assert summary[f"{PREFIX}_generated_alias_labels_present"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0


def test_source_native_profiles_and_registry_contract() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == "local_100k_source_native_relation_adapter_metrics.json"
    assert minimum[f"{PREFIX}_formula_relation_diagnostic_candidate"] == 1.0
    assert minimum[f"{PREFIX}_legacy_static_public_baseline_pass"] == 1.0
    assert minimum[f"{PREFIX}_relationless_content_scan_beaten"] == 1.0
    assert minimum[f"{PREFIX}_fair_stride_content_scan_not_beaten"] == 1.0
    assert minimum[f"{PREFIX}_beats_same_block_undercharged_mph_baseline"] == 1.0
    assert minimum[f"{PREFIX}_formula_or_schema_labels_present"] == 1.0
    assert maximum[f"{PREFIX}_publishable_relation_breakthrough_candidate"] == 0.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_static_public_baseline_pass"] == 0.0
    assert maximum[f"{PREFIX}_learned_relation_router_used"] == 0.0
    assert maximum[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert maximum[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
