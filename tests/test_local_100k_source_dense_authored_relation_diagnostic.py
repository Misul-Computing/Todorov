from neuroloc.simulations.memory.local_100k_source_dense_authored_relation_diagnostic import (
    SIMULATION_ID,
    SourceSubtokenGlobalStreamCorpusModule,
    answer_questions,
    build_summary,
    dense_authored_relation_facts,
    global_codec,
    read_limited_block,
    target_rows,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_dense_authored_relation_diagnostic"


def test_source_dense_authored_relation_reports_amortized_index_win_and_paq_limit() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_work_bounded_dense_relation_diagnostic_candidate"] == 1.0
    assert summary[f"{PREFIX}_relation_fact_count"] == 3741.0
    assert summary[f"{PREFIX}_definition_parent_relation_count"] == 314.0
    assert summary[f"{PREFIX}_definition_signature_relation_count"] == 0.0
    assert summary[f"{PREFIX}_statement_enclosing_relation_count"] == 2808.0
    assert summary[f"{PREFIX}_control_statement_enclosing_relation_count"] == 619.0
    assert summary[f"{PREFIX}_exact_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_selected_relation_accounted_bits"] == 437680.0
    assert summary[f"{PREFIX}_honest_mph_relation_index_bits"] == 3366080.0
    assert summary[f"{PREFIX}_margin_over_honest_mph_relation_index_bits"] == 2928400.0
    assert summary[f"{PREFIX}_dense_relation_amortizes_honest_mph_index"] == 1.0
    assert summary[f"{PREFIX}_paq8px_level2_relation_accounted_bits"] == 261144.0
    assert summary[f"{PREFIX}_margin_over_paq8px_level2_relation_bits"] == -176536.0
    assert summary[f"{PREFIX}_public_context_mixing_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_source_dense_authored_relation_product_authorized"] == 0.0
    assert summary[f"{PREFIX}_static_relation_breakthrough_authorized"] == 0.0


def test_source_dense_authored_relation_controls_collapse() -> None:
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


def test_source_dense_authored_relation_answers_from_reloaded_state() -> None:
    blocks = [read_limited_block(row) for row in target_rows("hard")]
    facts = dense_authored_relation_facts(blocks)
    codec = global_codec(blocks)
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    answers = answer_questions(reload_module, [str(fact["question"]) for fact in facts])
    assert len(facts) == 3741
    assert reload_module.reconstruct() == blocks
    assert all(str(fact["relation"]) != "definition_signature" for fact in facts)
    assert all(str(fact["value"]) not in str(fact["question"]) for fact in facts if len(str(fact["value"])) > 6)
    assert all(answer["hit"] == 1 for answer in answers)
    assert all(str(answer["value"]) == str(fact["value"]) for fact, answer in zip(facts, answers))


def test_source_dense_authored_relation_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert minimum[f"{PREFIX}_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_work_bounded_dense_relation_diagnostic_candidate"] == 1.0
    assert minimum[f"{PREFIX}_dense_relation_amortizes_honest_mph_index"] == 1.0
    assert minimum[f"{PREFIX}_public_context_mixing_not_beaten"] == 1.0
    assert maximum[f"{PREFIX}_source_dense_authored_relation_product_authorized"] == 0.0
    assert maximum[f"{PREFIX}_static_relation_breakthrough_authorized"] == 0.0
