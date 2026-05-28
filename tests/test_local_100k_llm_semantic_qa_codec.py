from neuroloc.simulations.memory.local_100k_llm_semantic_qa_codec import (
    LLMSemanticQACodecCell,
    build_facts,
    build_random_twin,
    build_summary,
    score_answers,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_llm_semantic_qa_codec"
PREFIX = "local_100k_llm_semantic_qa_codec"


def test_semantic_qa_facts_are_deterministic_and_do_not_expose_forbidden_query_fields() -> None:
    first_train, first_test, first_block, first_manifest = build_facts(1207, 128, 96)
    second_train, second_test, second_block, second_manifest = build_facts(1207, 128, 96)
    third_train, third_test, _, _ = build_facts(1208, 128, 96)
    forbidden = {"source", "offset", "content_window_digest", "answer_digest", "key"}
    assert first_train == second_train
    assert first_test == second_test
    assert first_block == second_block
    assert first_manifest == second_manifest
    assert (first_train, first_test) != (third_train, third_test)
    assert len({tuple(row["semantic_handle"]) for row in first_test}) == len(first_test)
    assert all(str(row["question"]).strip() for row in first_test)
    assert all({"role", "row", "question", "semantic_handle", "value", "provenance"} <= set(row) for row in first_test)
    assert all(forbidden.isdisjoint(row) for row in first_test)


def test_semantic_qa_cell_uses_one_charged_compressed_block_without_rows_or_raw_source() -> None:
    train, facts, source_block, manifest = build_facts(1207, 128, 96)
    cell = LLMSemanticQACodecCell(train, facts, source_block, manifest)
    assert cell.source_block_count == 1
    assert cell.block_stream_count == 1
    assert cell.assignment_row_count == 0
    assert cell.per_fact_value_row_count == 0
    assert cell.raw_source_block_retained == 0.0
    assert cell.reads_from_compressed_block == 1.0
    assert cell.semantic_question_handle_target == 1.0
    assert cell.source_offset_routing_used == 0.0
    assert cell.content_digest_key_target == 0.0
    assert not hasattr(cell, "decoded_block")
    assert not hasattr(cell, "raw_source_block")
    assert not hasattr(cell, "assignment_rows")
    assert not hasattr(cell, "per_fact_value_rows")
    decompressions_before = cell.decompression_count
    scans_before = cell.scan_count
    answers = cell.answer_many([row["question"] for row in facts])
    assert len(answers) == len(facts)
    assert cell.decompression_count == decompressions_before + 1
    assert cell.scan_count == scans_before + 1


def test_semantic_qa_roundtrips_exactly_from_natural_language_questions() -> None:
    train, facts, source_block, manifest = build_facts(1207, 128, 96)
    cell = LLMSemanticQACodecCell(train, facts, source_block, manifest)
    answers = cell.answer_many([row["question"] for row in facts])
    success = sum(row["exact_success"] for row in score_answers(facts, answers)) / len(facts)
    assert success == 1.0
    assert cell.answer(facts[0]["question"])["hit"] == 1


def test_semantic_qa_controls_collapse() -> None:
    train, facts, source_block, manifest = build_facts(1207, 128, 96)
    random_twin = build_random_twin(1207, facts)
    cell = LLMSemanticQACodecCell(train, facts, source_block, manifest)
    shifted_questions = [row["question"] for row in facts[1:] + facts[:1]]
    wrong_questions = [f"not the stored semantic question {index}" for index, _row in enumerate(facts)]
    shuffled_answers = cell.answer_many(shifted_questions)
    wrong_answers = cell.answer_many(wrong_questions)
    random_twin_answers = cell.answer_many([row["question"] for row in random_twin])
    read_disabled_answers = cell.answer_many([row["question"] for row in facts], read_disabled=True)
    decoder_disabled_answers = cell.answer_many([row["question"] for row in facts], decoder_disabled=True)
    parser_disabled_answers = cell.answer_many([row["question"] for row in facts], parser_disabled=True)
    code_disabled_answers = cell.answer_many([row["question"] for row in facts], code_disabled=True)
    assert sum(row["exact_success"] for row in score_answers(facts, shuffled_answers)) == 0
    assert sum(row["exact_success"] for row in score_answers(facts, wrong_answers)) == 0
    assert sum(row["exact_success"] for row in score_answers(random_twin, random_twin_answers)) == 0
    assert sum(row["exact_success"] for row in score_answers(facts, read_disabled_answers)) == 0
    assert sum(row["exact_success"] for row in score_answers(facts, decoder_disabled_answers)) == 0
    assert sum(row["exact_success"] for row in score_answers(facts, parser_disabled_answers)) == 0
    assert sum(row["exact_success"] for row in score_answers(facts, code_disabled_answers)) == 0


def test_semantic_qa_summary_reports_bounded_product_without_overclaim() -> None:
    summary = build_summary("smoke", seed=1207)
    assert summary[f"{PREFIX}_product_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_beats_content_addressed_codec_baseline"] == 1.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert summary[f"{PREFIX}_semantic_question_handle_target"] == 1.0
    assert summary[f"{PREFIX}_source_offset_routing_used"] == 0.0
    assert summary[f"{PREFIX}_content_digest_key_target"] == 0.0
    assert summary[f"{PREFIX}_fixed_parser_bits"] > 0.0
    assert summary[f"{PREFIX}_decoder_bits"] >= summary[f"{PREFIX}_fixed_parser_bits"]
    assert summary[f"{PREFIX}_fixed_parser_charged_through_decoder_bits"] == 1.0


def test_semantic_qa_registry_entry_has_required_suite_and_gate_contract() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.category == "compression_mirror"
    assert spec.metrics_filename == "local_100k_llm_semantic_qa_codec_metrics.json"
    assert minimum[f"{PREFIX}_product_pass"] == 1.0
    assert minimum[f"{PREFIX}_exact_answer_success"] == 1.0
    assert minimum[f"{PREFIX}_controls_collapse"] == 1.0
    assert minimum[f"{PREFIX}_beats_content_addressed_codec_baseline"] == 1.0
    assert minimum[f"{PREFIX}_semantic_question_handle_target"] == 1.0
    assert minimum[f"{PREFIX}_reads_from_compressed_block"] == 1.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
    assert maximum[f"{PREFIX}_source_offset_routing_used"] == 0.0
    assert maximum[f"{PREFIX}_content_digest_key_target"] == 0.0
    assert maximum[f"{PREFIX}_assignment_row_count"] == 0.0
    assert maximum[f"{PREFIX}_per_fact_value_row_count"] == 0.0
