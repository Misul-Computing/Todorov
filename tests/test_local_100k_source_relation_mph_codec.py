from neuroloc.simulations.memory.local_100k_source_dense_authored_relation_diagnostic import dense_authored_relation_facts
from neuroloc.simulations.memory.local_100k_source_relation_mph_codec import (
    SIMULATION_ID,
    SourceRelationMPHCodecModule,
    build_relation_codec,
    build_summary,
    random_label_facts,
    relation_blocks,
    score_answers,
    wrong_query_variants,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_source_relation_mph_codec"


def test_source_relation_mph_codec_beats_public_and_index_baselines() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_source_relation_mph_codec_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_source_relation_index_product_candidate"] == 1.0
    assert summary[f"{PREFIX}_source_relation_static_breakthrough_candidate"] == 0.0
    assert summary[f"{PREFIX}_relation_fact_count"] == 3741.0
    assert summary[f"{PREFIX}_definition_parent_relation_count"] == 314.0
    assert summary[f"{PREFIX}_statement_enclosing_relation_count"] == 2808.0
    assert summary[f"{PREFIX}_control_statement_enclosing_relation_count"] == 619.0
    assert summary[f"{PREFIX}_fingerprint_bits_per_key"] == 17.0
    assert summary[f"{PREFIX}_selected_relation_accounted_bits"] == 248784.0
    assert summary[f"{PREFIX}_paq8px_level2_relation_accounted_bits"] == 261144.0
    assert summary[f"{PREFIX}_margin_over_paq8px_level2_relation_bits"] == 12360.0
    assert summary[f"{PREFIX}_raw_source_paq_content_scan_bits"] == 413888.0
    assert summary[f"{PREFIX}_margin_over_raw_source_paq_content_scan_bits"] == 165104.0
    assert summary[f"{PREFIX}_undercharged_mph_relation_bits"] == 2771261.0
    assert summary[f"{PREFIX}_honest_mph_relation_index_bits"] == 3366080.0
    assert summary[f"{PREFIX}_margin_over_undercharged_mph_relation_bits"] == 2522477.0
    assert summary[f"{PREFIX}_margin_over_honest_mph_relation_index_bits"] == 3117296.0
    assert summary[f"{PREFIX}_public_context_mixing_beaten"] == 1.0
    assert summary[f"{PREFIX}_raw_source_content_scan_beaten"] == 1.0
    assert summary[f"{PREFIX}_undercharged_mph_beaten"] == 1.0
    assert summary[f"{PREFIX}_honest_mph_index_beaten"] == 1.0
    assert summary[f"{PREFIX}_strict_multiplier"] > 67.0


def test_source_relation_mph_controls_and_category_limits() -> None:
    summary = build_summary("hard", seed=12829)
    assert summary[f"{PREFIX}_exact_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_random_label_cross_label_success"] == 0.0
    assert summary[f"{PREFIX}_random_label_rebuild_exact_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_rebuild_selected_relation_accounted_bits"] == 934984.0
    assert summary[f"{PREFIX}_random_label_rebuild_selected_bits_delta"] == 686200.0
    assert summary[f"{PREFIX}_random_label_rebuild_density_control_collapse"] == 1.0
    assert summary[f"{PREFIX}_decoder_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_fingerprint_success"] == 0.0
    assert summary[f"{PREFIX}_wrong_query_variant_count"] == 7482.0
    assert summary[f"{PREFIX}_wrong_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_state_dict_reload_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_exact_reload_answer_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_payload_keys_present"] == 1.0
    assert summary[f"{PREFIX}_header_raw_bits_within_charged_budget"] == 1.0
    assert summary[f"{PREFIX}_model_state_relation_payload_used"] == 1.0
    assert summary[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_full_question_table_stored"] == 0.0
    assert summary[f"{PREFIX}_stored_question_substring_hit_count"] == 0.0
    assert summary[f"{PREFIX}_raw_source_block_substring_hit_count"] == 0.0
    assert summary[f"{PREFIX}_paq8px_baseline_external_constant_used"] == 1.0
    assert summary[f"{PREFIX}_paq8px_baseline_recomputed_in_run"] == 1.0
    assert summary[f"{PREFIX}_paq8px_relation_recomputed_payload_bits"] == 252952.0
    assert summary[f"{PREFIX}_paq8px_relation_recomputed_accounted_bits"] == 261144.0
    assert summary[f"{PREFIX}_paq8px_relation_recomputed_archive_bytes"] == 31619.0
    assert summary[f"{PREFIX}_paq8px_relation_recomputed_matches_constant"] == 1.0
    assert summary[f"{PREFIX}_self_contained_paq8px_baseline_win_authorized"] == 1.0
    assert summary[f"{PREFIX}_generated_alias_labels_present"] == 0.0
    assert summary[f"{PREFIX}_fixed_stride_relation_used"] == 0.0
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 0.0
    assert summary[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_nm_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0


def test_source_relation_mph_answers_after_state_dict_reload() -> None:
    blocks = relation_blocks("hard")
    facts = dense_authored_relation_facts(blocks)
    codec = build_relation_codec(facts)
    module = SourceRelationMPHCodecModule(codec)
    state = module.state_dict()
    reload_module = SourceRelationMPHCodecModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    questions = [str(fact["question"]) for fact in facts]
    answers = reload_module.answer_many(questions)
    wrong_answers = reload_module.answer_many(wrong_query_variants(facts))
    assert score_answers(facts, answers) == 1.0
    assert score_answers(random_label_facts(12829, facts), answers) == 0.0
    assert sum(int(answer["hit"]) for answer in wrong_answers) == 0
    assert set(state.keys()) == {
        "relation_header",
        "displacement_payload",
        "value_id_payload",
        "provenance_id_payload",
        "value_dictionary_payload",
        "provenance_dictionary_payload",
        "fingerprint_payload",
    }


def test_source_relation_mph_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert minimum[f"{PREFIX}_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_source_relation_mph_codec_product_authorized"] == 1.0
    assert minimum[f"{PREFIX}_source_relation_index_product_candidate"] == 1.0
    assert minimum[f"{PREFIX}_paq8px_baseline_recomputed_in_run"] == 1.0
    assert minimum[f"{PREFIX}_self_contained_paq8px_baseline_win_authorized"] == 1.0
    assert minimum[f"{PREFIX}_public_context_mixing_beaten"] == 1.0
    assert minimum[f"{PREFIX}_raw_source_content_scan_beaten"] == 1.0
    assert minimum[f"{PREFIX}_undercharged_mph_beaten"] == 1.0
    assert minimum[f"{PREFIX}_honest_mph_index_beaten"] == 1.0
    hard_minimum = dict(spec.hard_minimum_summary_values)
    assert hard_minimum[f"{PREFIX}_paq8px_relation_recomputed_payload_bits"] == 252952.0
    assert hard_minimum[f"{PREFIX}_paq8px_relation_recomputed_accounted_bits"] == 261144.0
    assert hard_minimum[f"{PREFIX}_paq8px_relation_recomputed_archive_bytes"] == 31619.0
    assert hard_minimum[f"{PREFIX}_paq8px_relation_recomputed_matches_constant"] == 1.0
    assert hard_minimum[f"{PREFIX}_margin_over_paq8px_level2_relation_bits"] == 10000.0
    assert maximum[f"{PREFIX}_source_relation_static_breakthrough_candidate"] == 0.0
    assert maximum[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_full_nm_authorized"] == 0.0
