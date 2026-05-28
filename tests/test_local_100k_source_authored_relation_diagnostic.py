from neuroloc.simulations.memory.local_100k_source_authored_relation_diagnostic import (
    SIMULATION_ID,
    SourceSubtokenGlobalStreamCorpusModule,
    answer_questions,
    authored_relation_facts,
    build_summary,
    global_codec,
    read_limited_block,
    target_rows,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_authored_relation_diagnostic"


def test_source_authored_relation_diagnostic_reports_safe_limits() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_work_bounded_relation_diagnostic_candidate"] == 1.0
    assert summary[f"{PREFIX}_relation_fact_count"] == 337.0
    assert summary[f"{PREFIX}_definition_relation_count"] == 303.0
    assert summary[f"{PREFIX}_import_relation_count"] == 34.0
    assert summary[f"{PREFIX}_exact_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_relation_aware_unlimited_scanner_success"] == 1.0
    assert summary[f"{PREFIX}_relation_aware_unlimited_scanner_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_read_limited_scanner_success"] == 0.0
    assert summary[f"{PREFIX}_read_work_gain_over_unlimited_scan"] == 337.0
    assert summary[f"{PREFIX}_selected_relation_accounted_bits"] == 437680.0
    assert summary[f"{PREFIX}_raw_relation_content_scan_bits"] == 457272.0
    assert summary[f"{PREFIX}_undercharged_relation_mph_bits"] == 457288.0
    assert summary[f"{PREFIX}_honest_mph_relation_index_bits"] == 305016.0
    assert summary[f"{PREFIX}_margin_over_raw_relation_content_scan_bits"] == 19592.0
    assert summary[f"{PREFIX}_margin_over_undercharged_relation_mph_bits"] == 19608.0
    assert summary[f"{PREFIX}_margin_over_honest_mph_relation_index_bits"] == -132664.0
    assert summary[f"{PREFIX}_honest_mph_index_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_source_authored_relation_product_authorized"] == 0.0
    assert summary[f"{PREFIX}_static_relation_breakthrough_authorized"] == 0.0


def test_source_authored_relation_controls_collapse() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_value_success"] == 0.0
    assert summary[f"{PREFIX}_relation_decoder_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_wrong_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_source_train_test_path_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_source_train_test_hash_overlap_count"] == 0.0
    assert summary[f"{PREFIX}_state_dict_reload_reconstruction_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_generated_alias_labels_present"] == 0.0
    assert summary[f"{PREFIX}_fixed_stride_relation_used"] == 0.0
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0


def test_source_authored_relation_answers_from_reloaded_state() -> None:
    blocks = [read_limited_block(row) for row in target_rows("hard")]
    facts = authored_relation_facts(blocks)
    codec = global_codec(blocks)
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    answers = answer_questions(reload_module, [str(fact["question"]) for fact in facts])
    assert len(facts) == 337
    assert reload_module.reconstruct() == blocks
    assert all(answer["hit"] == 1 for answer in answers)
    assert all(str(answer["value"]) == str(fact["value"]) for fact, answer in zip(facts, answers))


def test_source_authored_relation_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert minimum[f"{PREFIX}_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_work_bounded_relation_diagnostic_candidate"] == 1.0
    assert minimum[f"{PREFIX}_relation_aware_unlimited_scanner_success"] == 1.0
    assert maximum[f"{PREFIX}_source_authored_relation_product_authorized"] == 0.0
    assert maximum[f"{PREFIX}_static_relation_breakthrough_authorized"] == 0.0
