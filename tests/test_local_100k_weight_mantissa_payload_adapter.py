from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import build_facts
from neuroloc.simulations.memory.local_100k_weight_mantissa_payload_adapter import (
    PROFILES,
    WeightMantissaPayloadAdapterCell,
    bits_to_bytes,
    build_summary,
    bytes_to_bits,
    float_tensor_to_words,
    pack_payload_words,
    score_answers,
    unpack_payload_words,
    words_to_float_tensor,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


SIMULATION_ID = "local_100k_weight_mantissa_payload_adapter"
PREFIX = "local_100k_weight_mantissa_payload_adapter"


def exact_success_count(facts: list[dict], answers: list[dict]) -> float:
    return sum(row["exact_success"] for row in score_answers(facts, answers))


def test_mantissa_bitpacking_round_trips_payload_bytes() -> None:
    payload = bytes(range(251)) + b"neuroloc-weight-mantissa"
    bits = bytes_to_bits(payload)
    assert bits_to_bytes(bits, len(payload)) == payload
    words = pack_payload_words(payload)
    restored = unpack_payload_words(words, len(payload))
    assert restored == payload
    tensor = words_to_float_tensor(words)
    assert float_tensor_to_words(tensor) == words


def test_weight_mantissa_cell_answers_from_parameters_not_payload_buffer() -> None:
    train, facts, source_block, profile = build_facts(5279, 512)
    cell = WeightMantissaPayloadAdapterCell(train, facts, source_block, profile)
    questions = [str(row["question"]) for row in facts]
    answers = cell.answer_many(questions)
    state_keys = set(cell.module.state_dict().keys())
    assert exact_success_count(facts, answers) == len(facts)
    assert cell.parameter_count() == cell.carrier_parameter_count
    assert cell.parameter_count() < 100000
    assert "carrier" in state_keys
    assert "adapter_payload" not in state_keys
    assert cell.model_weight_payload_used == 1.0
    assert cell.state_dict_buffer_payload_used == 0.0
    assert cell.raw_source_block_retained == 0.0
    assert not hasattr(cell, "raw_source_block")


def test_weight_mantissa_summary_reports_publishable_weight_payload_candidate() -> None:
    summary = build_summary("smoke", seed=5279)
    assert summary[f"{PREFIX}_engineering_pass"] == 1.0
    assert summary[f"{PREFIX}_publishable_weight_payload_candidate"] == 0.0
    assert summary[f"{PREFIX}_mantissa_payload_diagnostic_candidate"] == 1.0
    assert summary[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert summary[f"{PREFIX}_static_public_baseline_pass"] == 0.0
    assert summary[f"{PREFIX}_exact_answer_success"] == 1.0
    assert summary[f"{PREFIX}_random_label_twin_success"] == 0.0
    assert summary[f"{PREFIX}_controls_collapse"] == 1.0
    assert summary[f"{PREFIX}_model_weight_payload_used"] == 1.0
    assert summary[f"{PREFIX}_state_dict_buffer_payload_used"] == 0.0
    assert summary[f"{PREFIX}_adapter_parameter_count"] < 100000
    assert 16.0 < summary[f"{PREFIX}_paper_surface_strict_multiplier"] < 17.0
    assert 30.0 < summary[f"{PREFIX}_apparent_mantissa_paper_surface_multiplier"] < 32.0
    assert summary[f"{PREFIX}_payload_bit_paper_surface_multiplier"] < 23.0
    assert summary[f"{PREFIX}_fp32_paper_surface_multiplier"] < 17.0
    assert summary[f"{PREFIX}_same_block_content_scan_multiplier"] < 24.0
    assert summary[f"{PREFIX}_paper_surface_strict_multiplier"] < summary[f"{PREFIX}_same_block_content_scan_multiplier"]
    assert summary[f"{PREFIX}_beats_same_block_content_scan_baseline"] == 0.0
    assert summary[f"{PREFIX}_beats_same_block_undercharged_mph_baseline"] == 0.0
    assert summary[f"{PREFIX}_apparent_mantissa_multiplier_beats_mph"] == 1.0
    assert summary[f"{PREFIX}_mantissa_payload_carrier_used"] == 1.0
    assert summary[f"{PREFIX}_mantissa_steganography_diagnostic"] == 1.0
    assert summary[f"{PREFIX}_true_base_weight_implicit_storage_authorized"] == 0.0
    assert summary[f"{PREFIX}_formula_or_schema_labels_present"] == 0.0
    assert summary[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert summary[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0


def test_weight_mantissa_profiles_and_registry_contract() -> None:
    assert {"smoke", "hard"} <= set(PROFILES)
    assert int(PROFILES["hard"]["fact_count"]) >= int(PROFILES["smoke"]["fact_count"])
    assert SIMULATION_ID in SIMULATION_SPECS
    assert SIMULATION_ID in SUITES["compression_mirror"]
    assert SIMULATION_ID in SUITES["precompute"]
    spec = SIMULATION_SPECS[SIMULATION_ID]
    minimum = dict(spec.minimum_summary_values)
    maximum = dict(spec.maximum_summary_values)
    assert spec.metrics_filename == "local_100k_weight_mantissa_payload_adapter_metrics.json"
    assert minimum[f"{PREFIX}_mantissa_payload_diagnostic_candidate"] == 1.0
    assert minimum[f"{PREFIX}_mantissa_payload_carrier_used"] == 1.0
    assert minimum[f"{PREFIX}_mantissa_steganography_diagnostic"] == 1.0
    assert maximum[f"{PREFIX}_publishable_weight_payload_candidate"] == 0.0
    assert maximum[f"{PREFIX}_strict_breakthrough_authorized"] == 0.0
    assert maximum[f"{PREFIX}_static_public_baseline_pass"] == 0.0
    assert maximum[f"{PREFIX}_beats_same_block_content_scan_baseline"] == 0.0
    assert maximum[f"{PREFIX}_beats_same_block_undercharged_mph_baseline"] == 0.0
    assert maximum[f"{PREFIX}_strict_600x_pass"] == 0.0
    assert maximum[f"{PREFIX}_arbitrary_chat_authorized"] == 0.0
