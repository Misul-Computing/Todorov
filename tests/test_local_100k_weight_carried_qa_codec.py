from neuroloc.simulations.memory.local_100k_weight_carried_qa_codec import (
    PROFILES,
    WeightCarriedQACodecCell,
    build_facts,
    build_random_twin,
    build_summary,
    offset_for_fact,
    provenance_for_block,
    score_answers,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_weight_carried_qa_codec"
PREFIX = "local_100k_weight_carried_qa_codec"


def questions_for(facts: list[dict]) -> list[str]:
    return [str(row["question"]) for row in facts]


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_weight_carried_facts_are_deterministic_and_do_not_expose_hidden_routing_fields() -> None:
    first_train, first_test, first_block, first_manifest = build_facts(1703, 128, 96)
    second_train, second_test, second_block, second_manifest = build_facts(1703, 128, 96)
    third_train, third_test, _, _ = build_facts(1704, 128, 96)
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
    }
    assert first_train == second_train
    assert first_test == second_test
    assert first_block == second_block
    assert first_manifest == second_manifest
    assert (first_train, first_test) != (third_train, third_test)
    assert all({"role", "row", "question", "value", "provenance"} <= set(row) for row in first_test)
    assert all(str(row["question"]).strip() for row in first_test)
    assert all(forbidden.isdisjoint(row) for row in first_test)


def test_weight_carried_cell_uses_adapter_payload_without_external_payload_or_stored_manifest() -> None:
    train, facts, source_block, manifest = build_facts(1703, 128, 96)
    cell = WeightCarriedQACodecCell(train, facts, source_block, manifest)
    assert cell.model_state_adapter_payload_used == 1.0
    assert cell.external_payload_store_used == 0.0
    assert cell.stored_manifest_used == 0.0
    assert cell.adapter_recompression_update_path == 1.0
    assert cell.true_base_weight_implicit_storage_authorized == 0.0
    assert not hasattr(cell, "external_payload_store")
    assert not hasattr(cell, "stored_manifest")
    assert not hasattr(cell, "raw_source_block")
    answers = cell.answer_many(questions_for(facts))
    assert len(answers) == len(facts)


def test_weight_carried_roundtrips_exactly_from_natural_language_questions() -> None:
    train, facts, source_block, manifest = build_facts(1703, 128, 96)
    cell = WeightCarriedQACodecCell(train, facts, source_block, manifest)
    answers = cell.answer_many(questions_for(facts))
    assert exact_success_count(facts, answers) == len(facts)
    assert cell.answer(facts[0]["question"])["hit"] == 1


def test_weight_carried_recompression_update_changes_adapter_state_and_survives_reload() -> None:
    train, facts, source_block, manifest = build_facts(1703, 128, 96)
    cell = WeightCarriedQACodecCell(train, facts, source_block, manifest)
    offset = offset_for_fact(source_block, facts[0])
    old_value = bytes.fromhex(str(facts[0]["value"]))
    new_value = bytes((byte ^ 0x5A) for byte in old_value)
    updated_block = bytearray(source_block)
    updated_block[offset : offset + len(new_value)] = new_value
    cell.recompress_adapter_block(bytes(updated_block))
    answer = cell.answer(facts[0]["question"])
    assert cell.adapter_recompression_update_count == 1
    assert answer["value"] == new_value.hex()
    assert answer["value"] != facts[0]["value"]
    assert answer["provenance"] == provenance_for_block(offset, new_value)
    reload_cell = WeightCarriedQACodecCell(train, facts, bytes(updated_block), manifest)
    reload_cell.module.load_state_dict(cell.module.state_dict())
    assert reload_cell.answer(facts[0]["question"]) == answer


def test_weight_carried_controls_collapse() -> None:
    train, facts, source_block, manifest = build_facts(1703, 128, 96)
    random_twin = build_random_twin(1703, facts)
    cell = WeightCarriedQACodecCell(train, facts, source_block, manifest)
    shifted_questions = questions_for(facts[1:] + facts[:1])
    wrong_questions = [f"not the stored weight carried question {index}" for index, _row in enumerate(facts)]
    shuffled_answers = cell.answer_many(shifted_questions)
    wrong_answers = cell.answer_many(wrong_questions)
    random_twin_answers = cell.answer_many(questions_for(random_twin))
    read_disabled_answers = cell.answer_many(questions_for(facts), read_disabled=True)
    decoder_disabled_answers = cell.answer_many(questions_for(facts), decoder_disabled=True)
    parser_disabled_answers = cell.answer_many(questions_for(facts), parser_disabled=True)
    code_disabled_answers = cell.answer_many(questions_for(facts), code_disabled=True)
    adapter_disabled_answers = cell.answer_many(questions_for(facts), adapter_disabled=True)
    assert exact_success_count(facts, shuffled_answers) == 0
    assert exact_success_count(facts, wrong_answers) == 0
    assert exact_success_count(random_twin, random_twin_answers) == 0
    assert exact_success_count(facts, read_disabled_answers) == 0
    assert exact_success_count(facts, decoder_disabled_answers) == 0
    assert exact_success_count(facts, parser_disabled_answers) == 0
    assert exact_success_count(facts, code_disabled_answers) == 0
    assert exact_success_count(facts, adapter_disabled_answers) == 0


def test_weight_carried_summary_reports_adapter_path_and_strict_density_boundary() -> None:
    summary = build_summary("smoke", seed=1703)
    assert summary[f"{PREFIX}_product_pass"] == 1.0
    assert summary[f"{PREFIX}_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_strict_multiplier"] >= 15.0
    assert summary[f"{PREFIX}_model_state_adapter_payload_used"] == 1.0
    assert summary[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert summary[f"{PREFIX}_stored_manifest_used"] == 0.0
    assert summary[f"{PREFIX}_adapter_recompression_update_path"] == 1.0
    assert summary[f"{PREFIX}_adapter_recompression_update_success"] == 1.0
    assert summary[f"{PREFIX}_adapter_state_dict_reload_success"] == 1.0
    assert summary[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_pass"] == 0.0


def test_weight_carried_profiles_expose_smoke_and_hard_contracts() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["smoke"]["fact_count"]) > 0
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])


def test_weight_carried_registry_entry_has_required_suite_and_gate_contract() -> None:
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.category == "compression_mirror"
    assert spec.metrics_filename == "local_100k_weight_carried_qa_codec_metrics.json"
    assert minimum[f"{PREFIX}_product_pass"] == 1.0
    assert minimum[f"{PREFIX}_exact_answer_success"] == 1.0
    assert minimum[f"{PREFIX}_controls_collapse"] == 1.0
    assert minimum[f"{PREFIX}_strict_multiplier"] >= 15.0
    assert minimum[f"{PREFIX}_model_state_adapter_payload_used"] == 1.0
    assert minimum[f"{PREFIX}_adapter_recompression_update_path"] == 1.0
    assert minimum[f"{PREFIX}_adapter_recompression_update_success"] == 1.0
    assert minimum[f"{PREFIX}_adapter_state_dict_reload_success"] == 1.0
    assert maximum[f"{PREFIX}_external_payload_store_used"] == 0.0
    assert maximum[f"{PREFIX}_stored_manifest_used"] == 0.0
    assert maximum[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
    assert maximum[f"{PREFIX}_strict_600x_pass"] == 0.0
