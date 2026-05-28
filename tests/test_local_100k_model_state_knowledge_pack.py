from neuroloc.simulations.memory.local_100k_model_state_knowledge_pack import (
    SIMULATION_ID,
    ModelStateKnowledgePackCell,
    build_summary,
    knowledge_pack_facts,
    paraphrase_questions,
    surface_rows,
)
from neuroloc.simulations.memory.local_100k_source_relation_mph_codec import score_answers
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


PREFIX = "local_100k_model_state_knowledge_pack"


def hard_summary() -> dict[str, float]:
    return build_summary("hard", seed=15131)


def test_knowledge_pack_generation_has_three_public_surfaces() -> None:
    facts = knowledge_pack_facts("smoke")
    rows = surface_rows("smoke")
    relations = {str(fact["relation"]) for fact in facts}
    assert len(rows) == 6
    assert sum(1 for row in rows if row["kind"] == "source") == 3
    assert sum(1 for row in rows if row["kind"] == "document") == 2
    assert sum(1 for row in rows if row["kind"] == "config") == 1
    assert "definition_parent" in relations
    assert "statement_enclosing_signature" in relations
    assert "document_heading" in relations
    assert "document_context" in relations
    assert "config_assignment_value" in relations
    assert "config_macro_name" in relations
    assert len(facts) == 3875


def test_knowledge_pack_non_source_questions_do_not_contain_answers() -> None:
    facts = knowledge_pack_facts("smoke")
    checked = [
        fact
        for fact in facts
        if fact["relation"] in {"document_heading", "document_context", "config_macro_name"}
    ]
    assert checked
    assert all(str(fact["value"]) not in str(fact["question"]) for fact in checked)


def test_knowledge_pack_hard_product_metrics() -> None:
    summary = hard_summary()
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_model_state_knowledge_pack_product_authorized"] == 1.0
    assert summary[f"{PREFIX}_paper_ready_bounded_knowledge_pack_candidate"] == 1.0
    assert summary[f"{PREFIX}_public_surface_count"] == 15.0
    assert summary[f"{PREFIX}_public_surface_total_bytes"] == 1323186.0
    assert summary[f"{PREFIX}_source_surface_count"] == 7.0
    assert summary[f"{PREFIX}_document_surface_count"] == 7.0
    assert summary[f"{PREFIX}_config_surface_count"] == 1.0
    assert summary[f"{PREFIX}_relation_fact_count"] == 9754.0
    assert summary[f"{PREFIX}_selected_relation_accounted_bits"] == 679400.0
    assert summary[f"{PREFIX}_model_package_accounted_bits"] == 687592.0
    assert summary[f"{PREFIX}_useful_retrievable_bits"] == 3992464.0
    assert summary[f"{PREFIX}_strict_multiplier"] > 37.0
    assert summary[f"{PREFIX}_model_package_strict_multiplier"] > 37.0
    assert summary[f"{PREFIX}_paq8px_relation_recomputed_accounted_bits"] == 1385264.0
    assert summary[f"{PREFIX}_zstd_level19_source_scan_accounted_bits"] == 2195016.0
    assert summary[f"{PREFIX}_strongest_baseline_accounted_bits"] == 1385264.0
    assert summary[f"{PREFIX}_margin_over_strongest_baseline_bits"] == 705864.0
    assert summary[f"{PREFIX}_margin_over_honest_mph_relation_index_bits"] == 5194024.0
    assert summary[f"{PREFIX}_margin_over_product_key_memory_bits"] == 13268336.0
    assert summary[f"{PREFIX}_margin_over_rag_knn_retrieval_bits"] == 1962568.0
    assert summary[f"{PREFIX}_margin_over_lora_exact_payload_lower_bound_bits"] == 3633384.0
    assert summary[f"{PREFIX}_margin_over_model_edit_exact_payload_lower_bound_bits"] == 3945512.0


def test_knowledge_pack_controls_lifecycle_and_limits() -> None:
    summary = hard_summary()
    assert summary[f"{PREFIX}_exact_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_paraphrased_relation_answer_success"] == 1.0
    assert summary[f"{PREFIX}_same_interface_scanner_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_random_label_rebuild_exact_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_rebuild_selected_relation_accounted_bits"] == 2429560.0
    assert summary[f"{PREFIX}_random_label_rebuild_density_control_collapse"] == 1.0
    assert summary[f"{PREFIX}_decoder_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_read_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_adapter_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_code_disabled_success"] == 0.0
    assert summary[f"{PREFIX}_parser_disabled_prefixed_success"] == 0.0
    assert summary[f"{PREFIX}_shuffled_fingerprint_success"] == 0.0
    assert summary[f"{PREFIX}_wrong_query_hit_rate"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_state_dict_reload_success"] == 1.0
    assert summary[f"{PREFIX}_adapter_export_reload_success"] == 1.0
    assert summary[f"{PREFIX}_update_lifecycle_pass"] == 1.0
    assert summary[f"{PREFIX}_update_patch_beats_full_recompress"] == 1.0
    assert summary[f"{PREFIX}_update_patch_accounted_bits"] == 198312.0
    assert summary[f"{PREFIX}_updated_full_recompress_accounted_bits"] == 780432.0
    assert summary[f"{PREFIX}_changed_value_before_update_success"] == 0.0
    assert summary[f"{PREFIX}_rollback_state_dict_reload_success"] == 1.0
    assert summary[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_state_space_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_raw_public_surface_retained"] == 0.0
    assert summary[f"{PREFIX}_full_question_table_stored"] == 0.0
    assert summary[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_authorized"] == 0.0
    assert summary[f"{PREFIX}_broad_knowledge_authorized"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_full_nm_authorized"] == 0.0


def test_knowledge_pack_answers_after_state_dict_reload() -> None:
    facts = knowledge_pack_facts("smoke")
    cell = ModelStateKnowledgePackCell(facts)
    state = cell.module.state_dict()
    reload_cell = ModelStateKnowledgePackCell.empty_from_state_dict(state)
    reload_cell.module.load_state_dict(state)
    questions = paraphrase_questions(facts)
    assert score_answers(facts, reload_cell.answer_many(questions)) == 1.0
    assert set(state.keys()) == {
        "relation_header",
        "displacement_payload",
        "value_id_payload",
        "provenance_id_payload",
        "value_dictionary_payload",
        "provenance_dictionary_payload",
        "fingerprint_payload",
    }


def test_knowledge_pack_registry_entry() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    hard_minimum = dict(spec.hard_minimum_summary_values)
    assert spec.metrics_filename == f"{PREFIX}_metrics.json"
    assert minimum[f"{PREFIX}_engineering_pass"] == 1.0
    assert minimum[f"{PREFIX}_model_state_knowledge_pack_product_authorized"] == 1.0
    assert minimum[f"{PREFIX}_paper_ready_bounded_knowledge_pack_candidate"] == 1.0
    assert minimum[f"{PREFIX}_source_surface_count"] == 3.0
    assert minimum[f"{PREFIX}_document_surface_count"] == 2.0
    assert minimum[f"{PREFIX}_config_surface_count"] == 1.0
    assert minimum[f"{PREFIX}_all_storage_baselines_beaten"] == 1.0
    assert minimum[f"{PREFIX}_update_lifecycle_pass"] == 1.0
    assert minimum[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert minimum[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert minimum[f"{PREFIX}_state_space_surface_pass"] == 1.0
    assert hard_minimum[f"{PREFIX}_public_surface_count"] == 15.0
    assert hard_minimum[f"{PREFIX}_relation_fact_count"] == 9000.0
    assert hard_minimum[f"{PREFIX}_margin_over_paq8px_level2_source_scan_bits"] == 500000.0
    assert maximum[f"{PREFIX}_broad_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
