from neuroloc.simulations.memory.local_100k_answer_surface_codec import (
    AnswerSurfaceCodecCell,
    build_facts,
    build_random_twin,
    build_summary,
    decompress_payload,
    evaluate_controls,
    same_interface_answer_surface_scan,
    score_answers,
    unpack_payload,
)


SIMULATION_ID = "local_100k_answer_surface_codec"
PREFIX = "local_100k_answer_surface_codec"


def questions_for(facts: list[dict]) -> list[str]:
    return [str(row["question"]) for row in facts]


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_answer_surface_facts_are_deterministic_and_source_authored() -> None:
    first_train, first_facts = build_facts(3221, 256)
    second_train, second_facts = build_facts(3221, 256)
    third_train, third_facts = build_facts(3222, 256)
    assert first_train == []
    assert first_train == second_train
    assert first_facts == second_facts
    assert first_facts != third_facts
    assert len({str(row["key"]) for row in first_facts}) == len(first_facts)
    assert len({str(row["value"]) for row in first_facts}) == len(first_facts)
    assert len({str(row["source_path"]) for row in first_facts}) >= 4
    assert all(str(row["question"]).startswith("answer authored surface key ") for row in first_facts)
    assert all({"role", "row", "question", "key", "value", "provenance", "source_path"} <= set(row) for row in first_facts)


def test_answer_surface_cell_keeps_only_one_charged_payload_stream() -> None:
    train, facts = build_facts(3221, 128)
    cell = AnswerSurfaceCodecCell(train, facts)
    assert cell.single_charged_payload_stream_used == 1.0
    assert cell.external_payload_store_used == 0.0
    assert cell.hidden_dict_used == 0.0
    assert cell.raw_decoded_cache_retained == 0.0
    assert cell.raw_source_block_retained == 0.0
    assert cell.reads_from_charged_payload_stream == 1.0
    assert not hasattr(cell, "facts")
    assert not hasattr(cell, "test_facts")
    assert not hasattr(cell, "decoded_payload")
    assert not hasattr(cell, "decoded_rows_cache")
    assert not hasattr(cell, "records_by_key")
    rows = unpack_payload(decompress_payload(cell.payload_stream))
    assert len(rows) == len(facts)
    assert {"k", "q", "a", "p"} == set(rows[0])
    before_decode = cell.decode_count
    before_map = cell.transient_map_build_count
    answers = cell.answer_many(questions_for(facts))
    assert exact_success_count(facts, answers) == len(facts)
    assert cell.decode_count == before_decode + 1
    assert cell.transient_map_build_count == before_map + 1
    assert not hasattr(cell, "records_by_key")


def test_answer_surface_controls_and_random_label_twin_collapse() -> None:
    train, facts = build_facts(3221, 128)
    twin = build_random_twin(3221, facts)
    cell = AnswerSurfaceCodecCell(train, facts)
    controls = evaluate_controls(cell, facts, twin)
    assert controls["exact_success"] == 1.0
    assert controls["random_label_twin_success"] == 0.0
    assert controls["read_disabled_success"] == 0.0
    assert controls["decoder_disabled_success"] == 0.0
    assert controls["parser_disabled_success"] == 0.0
    assert controls["adapter_disabled_success"] == 0.0
    assert controls["code_disabled_success"] == 0.0
    assert controls["shuffled_question_success"] == 0.0
    assert controls["same_interface_answer_surface_scan_success"] == 1.0


def test_same_interface_answer_surface_scan_is_reported_and_not_beaten() -> None:
    train, facts = build_facts(3221, 128)
    cell = AnswerSurfaceCodecCell(train, facts)
    scan_answers = same_interface_answer_surface_scan(cell, questions_for(facts))
    assert exact_success_count(facts, scan_answers) == len(facts)
    summary = build_summary("smoke", seed=3221)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_random_label_twin_collapse"] == 1.0
    assert summary[f"{PREFIX}_same_interface_answer_surface_scan_success"] == 1.0
    assert summary[f"{PREFIX}_same_interface_answer_surface_scan_multiplier"] >= summary[f"{PREFIX}_adapter_multiplier"]
    assert summary[f"{PREFIX}_same_interface_answer_surface_scan_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_undercharged_mph_multiplier"] >= summary[f"{PREFIX}_adapter_multiplier"]
    assert summary[f"{PREFIX}_undercharged_mph_not_beaten"] == 1.0
    assert summary[f"{PREFIX}_per_fact_rows_in_payload"] == 1.0
    assert summary[f"{PREFIX}_table_diagnostic"] == 1.0
    assert summary[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert summary[f"{PREFIX}_publishable_auth"] == 0.0
    assert summary[f"{PREFIX}_breakthrough_authorized"] == 0.0
