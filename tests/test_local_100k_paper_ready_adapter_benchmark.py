from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import (
    PROFILES,
    PaperReadyAdapterCell,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    build_facts,
    build_summary,
    corrupt_adapter_payload,
    paraphrase_questions,
    score_answers,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_paper_ready_adapter_benchmark"
PREFIX = "local_100k_paper_ready_adapter_benchmark"


def questions_for(facts: list[dict]) -> list[str]:
    return [str(row["question"]) for row in facts]


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_paper_ready_facts_are_deterministic_multi_domain_and_field_safe() -> None:
    first_train, first_test, first_block, first_profile = build_facts(1901, 256, 128)
    second_train, second_test, second_block, second_profile = build_facts(1901, 256, 128)
    third_train, third_test, _, _ = build_facts(1902, 256, 128)
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
    assert first_train == second_train
    assert first_test == second_test
    assert first_block == second_block
    assert first_profile == second_profile
    assert (first_train, first_test) != (third_train, third_test)
    assert len({str(row["domain"]) for row in first_test}) >= 4
    assert all({"role", "row", "domain", "question", "value", "provenance"} <= set(row) for row in first_test)
    assert all(str(row["question"]).strip() for row in first_test)
    assert all(forbidden.isdisjoint(row) for row in first_test)


def test_paper_ready_cell_answers_exact_and_paraphrased_questions() -> None:
    train, facts, source_block, profile = build_facts(1901, 256, 128)
    cell = PaperReadyAdapterCell(train, facts, source_block, profile)
    exact_answers = cell.answer_many(questions_for(facts))
    paraphrase_answers = cell.answer_many(paraphrase_questions(facts))
    assert exact_success_count(facts, exact_answers) == len(facts)
    assert exact_success_count(facts, paraphrase_answers) == len(facts)
    assert cell.model_state_adapter_payload_used == 1.0
    assert cell.external_payload_store_used == 0.0
    assert cell.stored_manifest_used == 0.0
    assert not hasattr(cell, "raw_source_block")
    assert not hasattr(cell, "external_payload_store")


def test_paper_ready_transformer_and_recurrent_hosts_carry_adapter_state() -> None:
    train, facts, source_block, profile = build_facts(1901, 256, 128)
    cell = PaperReadyAdapterCell(train, facts, source_block, profile)
    transformer = TinyTransformerAdapterHost(cell)
    recurrent = TinyRecurrentStateAdapterHost(cell)
    transformer_keys = set(transformer.module.state_dict().keys())
    recurrent_keys = set(recurrent.module.state_dict().keys())
    assert transformer.parameter_count() < 100000
    assert recurrent.parameter_count() < 100000
    assert "adapter_module.adapter_payload" in transformer_keys
    assert "adapter_module.adapter_payload" in recurrent_keys
    assert exact_success_count(facts, transformer.answer_many(paraphrase_questions(facts))) == len(facts)
    assert exact_success_count(facts, recurrent.answer_many(paraphrase_questions(facts))) == len(facts)
    reload_cell = PaperReadyAdapterCell(train, facts, source_block, profile)
    corrupt_adapter_payload(reload_cell.module)
    reload_transformer = TinyTransformerAdapterHost(reload_cell)
    assert exact_success_count(facts, reload_transformer.answer_many(paraphrase_questions(facts))) == 0
    reload_transformer.module.load_state_dict(transformer.module.state_dict())
    assert exact_success_count(facts, reload_transformer.answer_many(paraphrase_questions(facts))) == len(facts)


def test_paper_ready_controls_collapse() -> None:
    train, facts, source_block, profile = build_facts(1901, 256, 128)
    cell = PaperReadyAdapterCell(train, facts, source_block, profile)
    shifted_questions = questions_for(facts[1:] + facts[:1])
    wrong_questions = [f"not the stored paper ready question {index}" for index, _row in enumerate(facts)]
    shuffled_answers = cell.answer_many(shifted_questions)
    wrong_answers = cell.answer_many(wrong_questions)
    read_disabled_answers = cell.answer_many(questions_for(facts), read_disabled=True)
    decoder_disabled_answers = cell.answer_many(questions_for(facts), decoder_disabled=True)
    parser_disabled_answers = cell.answer_many(questions_for(facts), parser_disabled=True)
    code_disabled_answers = cell.answer_many(questions_for(facts), code_disabled=True)
    adapter_disabled_answers = cell.answer_many(questions_for(facts), adapter_disabled=True)
    assert exact_success_count(facts, shuffled_answers) == 0
    assert exact_success_count(facts, wrong_answers) == 0
    assert exact_success_count(facts, read_disabled_answers) == 0
    assert exact_success_count(facts, decoder_disabled_answers) == 0
    assert exact_success_count(facts, parser_disabled_answers) == 0
    assert exact_success_count(facts, code_disabled_answers) == 0
    assert exact_success_count(facts, adapter_disabled_answers) == 0


def test_paper_ready_summary_reports_five_requirement_candidate() -> None:
    summary = build_summary("smoke", seed=1901)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_paper_ready_candidate"] == 1.0
    assert summary[f"{PREFIX}_paper_ready_requirement_count"] == 5.0
    assert summary[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert summary[f"{PREFIX}_transformer_state_dict_preload_success"] == 0.0
    assert summary[f"{PREFIX}_recurrent_state_dict_preload_success"] == 0.0
    assert summary[f"{PREFIX}_adapter_state_dict_preload_success"] == 0.0
    assert summary[f"{PREFIX}_public_baseline_stack_pass"] == 1.0
    assert summary[f"{PREFIX}_multi_domain_pass"] == 1.0
    assert summary[f"{PREFIX}_paraphrase_or_update_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_paraphrase_stable_answer_success"] == 1.0
    assert summary[f"{PREFIX}_adapter_strict_multiplier"] >= 16.0
    assert summary[f"{PREFIX}_paper_surface_strict_multiplier"] >= 16.0
    assert summary[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_learned_semantic_retrieval_authorized"] == 0.0


def test_paper_ready_profiles_and_registry_contract() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == "local_100k_paper_ready_adapter_benchmark_metrics.json"
    assert minimum[f"{PREFIX}_paper_ready_candidate"] == 1.0
    assert minimum[f"{PREFIX}_paper_ready_requirement_count"] == 5.0
    assert minimum[f"{PREFIX}_transformer_surface_pass"] == 1.0
    assert minimum[f"{PREFIX}_recurrent_surface_pass"] == 1.0
    assert minimum[f"{PREFIX}_public_baseline_stack_pass"] == 1.0
    assert minimum[f"{PREFIX}_adapter_strict_multiplier"] >= 16.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_learned_semantic_retrieval_authorized"] == 0.0
    assert maximum[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
