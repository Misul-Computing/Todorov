from neuroloc.simulations.memory.local_100k_external_relation_adapter import (
    SIMULATION_ID,
    ExternalRelationAdapterCell,
    build_summary,
    external_blocks,
    paraphrase_questions,
)
from neuroloc.simulations.memory.local_100k_source_dense_authored_relation_diagnostic import dense_authored_relation_facts
from neuroloc.simulations.memory.local_100k_source_relation_mph_codec import score_answers, wrong_query_variants
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_external_relation_adapter"


def test_external_relation_adapter_hard_product_metrics() -> None:
    summary = build_summary("hard", seed=14387)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_external_relation_adapter_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_llm_adoptable_relation_adapter_candidate"] == 1.0
    assert summary[f"{PREFIX}_external_public_corpus_used"] == 1.0
    assert summary[f"{PREFIX}_external_source_count"] == 7.0
    assert summary[f"{PREFIX}_external_source_total_bytes"] == 596308.0
    assert summary[f"{PREFIX}_relation_fact_count"] == 6247.0
    assert summary[f"{PREFIX}_selected_relation_accounted_bits"] == 413600.0
    assert summary[f"{PREFIX}_model_package_accounted_bits"] == 417696.0
    assert summary[f"{PREFIX}_useful_retrievable_bits"] == 2304104.0
    assert summary[f"{PREFIX}_strict_multiplier"] > 35.0
    assert summary[f"{PREFIX}_model_package_strict_multiplier"] > 35.0
    assert summary[f"{PREFIX}_paq8px_relation_recomputed_accounted_bits"] == 606072.0
    assert summary[f"{PREFIX}_margin_over_paq8px_level2_source_scan_bits"] == 192472.0
    assert summary[f"{PREFIX}_margin_over_honest_mph_relation_index_bits"] == 3098120.0
    assert summary[f"{PREFIX}_margin_over_undercharged_mph_relation_bits"] == 2104847.0


def test_external_relation_adapter_controls_and_limits() -> None:
    summary = build_summary("hard", seed=14387)
    assert summary[f"{PREFIX}_exact_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_paraphrased_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_random_label_rebuild_exact_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_rebuild_selected_relation_accounted_bits"] == 1558016.0
    assert summary[f"{PREFIX}_random_label_rebuild_density_control_collapse"] == 1.0
    assert summary[f"{PREFIX}_decoder_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_read_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_adapter_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_code_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_parser_disabled_prefixed_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_fingerprint_success"] == 0.0
    assert summary[f"{PREFIX}_wrong_query_variant_count"] == 24988.0
    assert summary[f"{PREFIX}_wrong_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_state_dict_reload_success"] == 1.0
    assert summary[f"{PREFIX}_state_dict_exact_reload_answer_success"] == 1.0
    assert summary[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_state_space_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_raw_source_block_retained"] == 0.0
    assert summary[f"{PREFIX}_full_question_table_stored"] == 0.0
    assert summary[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert summary[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
    assert summary[f"{PREFIX}_source_relation_static_breakthrough_candidate"] == 0.0
    assert summary[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0


def test_external_relation_adapter_answers_after_state_dict_reload() -> None:
    blocks = external_blocks("smoke")
    facts = dense_authored_relation_facts(blocks)
    cell = ExternalRelationAdapterCell(facts)
    state = cell.module.state_dict()
    reload_cell = ExternalRelationAdapterCell.empty_from_state_dict(state)
    reload_cell.module.load_state_dict(state)
    questions = paraphrase_questions(facts)
    wrong_answers = reload_cell.answer_many(wrong_query_variants(facts))
    assert score_answers(facts, reload_cell.answer_many(questions)) == 1.0
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


def test_external_relation_adapter_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    hard_minimum = dict(spec.hard_minimum_summary_values)
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert minimum[f"{PREFIX}_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_external_relation_adapter_product_authorized"] == 1.0
    assert minimum[f"{PREFIX}_llm_adoptable_relation_adapter_candidate"] == 1.0
    assert minimum[f"{PREFIX}_public_context_mixing_beaten"] == 1.0
    assert minimum[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert minimum[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert minimum[f"{PREFIX}_state_space_surface_pass"] == 1.0
    assert hard_minimum[f"{PREFIX}_relation_fact_count"] == 6000.0
    assert hard_minimum[f"{PREFIX}_margin_over_paq8px_level2_source_scan_bits"] == 150000.0
    assert maximum[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
