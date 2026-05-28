from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_llm_semantic_qa_codec import (
    CHUNK_BYTES,
    anchor_text_for,
    mean_metric,
    score_answers,
    semantic_handle_for_anchor,
    semantic_handle_for_question,
)
from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import (
    MODEL_HEADER_BITS,
    ORDINARY_BITS_PER_PARAMETER,
    PAPER_READY_BASELINE_MULTIPLIER,
    SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER,
    TARGET_MULTIPLIER,
    WEIGHT_CARRIED_BASELINE_MULTIPLIER,
    build_facts,
    build_random_twin,
    hidden_state_inspection,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER, LLM_SEMANTIC_QA_BASELINE_MULTIPLIER
from neuroloc.simulations.memory.local_100k_semantic_alias_payload_adapter import CONTENT_SCAN_BASELINE_MULTIPLIER, MARGIN_BASELINE_MULTIPLIER, compress_payload, decompress_payload
from neuroloc.simulations.memory.local_100k_weight_carried_qa_codec import candidate_offsets_for_block, provenance_for_block

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_weight_mantissa_payload_adapter"
SEED = env_int("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_SEED", 5279)
FACTS_SMOKE = env_int("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_FACTS_HARD", 4096)
DECODER_BITS = env_int("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_DECODER_BITS", 32768)
SURFACE_CONTRACT_BITS = env_int("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_SURFACE_CONTRACT_BITS", 4096)
MANTISSA_BITS_PER_PARAMETER = env_int("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_MANTISSA_BITS_PER_PARAMETER", 23)
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))

require_positive("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_FACTS_SMOKE", FACTS_SMOKE)
require_positive("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_FACTS_HARD", FACTS_HARD)
require_positive("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_DECODER_BITS", DECODER_BITS)
require_positive("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_MANTISSA_BITS_PER_PARAMETER", MANTISSA_BITS_PER_PARAMETER)

PROFILES = {"smoke": {"fact_count": FACTS_SMOKE}, "hard": {"fact_count": FACTS_HARD}}
EXPONENT_BITS = 127 << 23
MANTISSA_MASK = (1 << 23) - 1


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("WEIGHT_MANTISSA_PAYLOAD_ADAPTER_PROFILE must be smoke or hard")
    return value


def bytes_to_bits(payload: bytes) -> list[int]:
    bits = []
    for byte in payload:
        for shift in range(8):
            bits.append((int(byte) >> shift) & 1)
    return bits


def bits_to_bytes(bits: list[int], length: int) -> bytes:
    out = bytearray()
    for index in range(int(length)):
        value = 0
        for shift in range(8):
            bit_index = index * 8 + shift
            if bit_index < len(bits):
                value |= int(bits[bit_index]) << shift
        out.append(value)
    return bytes(out)


def pack_payload_words(payload: bytes) -> list[int]:
    bits = bytes_to_bits(payload)
    words = []
    width = int(MANTISSA_BITS_PER_PARAMETER)
    for index in range(0, len(bits), width):
        word = 0
        for shift, bit in enumerate(bits[index : index + width]):
            word |= int(bit) << shift
        words.append(word)
    return words


def unpack_payload_words(words: list[int], payload_length: int) -> bytes:
    bits = []
    width = int(MANTISSA_BITS_PER_PARAMETER)
    for word in words:
        for shift in range(width):
            bits.append((int(word) >> shift) & 1)
    return bits_to_bytes(bits, int(payload_length))


def words_to_float_tensor(words: list[int]) -> Any:
    import numpy as np
    import torch

    uints = np.array([EXPONENT_BITS | (int(word) & MANTISSA_MASK) for word in words], dtype=np.uint32)
    floats = uints.view(np.float32).copy()
    return torch.tensor(floats, dtype=torch.float32)


def float_tensor_to_words(tensor: Any) -> list[int]:
    import numpy as np

    floats = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    uints = floats.view(np.uint32)
    return [int(value) & MANTISSA_MASK for value in uints.tolist()]


def build_weight_module(payload: bytes) -> Any:
    import torch
    import torch.nn as nn

    class WeightPayloadModule(nn.Module):
        def __init__(self, stream: bytes) -> None:
            super().__init__()
            words = pack_payload_words(stream)
            self.carrier = nn.Parameter(words_to_float_tensor(words), requires_grad=True)
            self.register_buffer("adapter_header", torch.tensor([int(len(stream)), int(len(words)), int(MANTISSA_BITS_PER_PARAMETER)], dtype=torch.int64), persistent=True)

        def forward(self, value: Any) -> Any:
            return value

    return WeightPayloadModule(payload)


class WeightMantissaPayloadAdapterCell:
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> None:
        self.model_weight_payload_used = 1.0
        self.state_dict_buffer_payload_used = 0.0
        self.external_payload_store_used = 0.0
        self.stored_manifest_used = 0.0
        self.block_stream_count = 1
        self.adapter_state_stream_count = 0
        self.per_fact_value_slice_count = 0
        self.assignment_row_count = 0
        self.per_fact_value_row_count = 0
        self.source_offset_routing_used = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_model_weights = 1.0
        self.reads_from_compressed_block = 1.0
        self.question_parser_in_decoder_bits = 1.0
        self.prompt_context_storage_used = 0.0
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        payload = compress_payload(source_block)
        self.module = build_weight_module(payload)
        self.block_payload_bits = int(len(payload) * 8)
        self.payload_length = int(len(payload))
        self.carrier_parameter_count = int(self.module.carrier.numel())
        self.candidate_count = len(candidate_offsets_for_block(len(source_block)))

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def payload_bytes(self) -> bytes:
        payload_length = int(self.module.adapter_header[0].item())
        words = float_tensor_to_words(self.module.carrier)
        return unpack_payload_words(words, payload_length)

    def decoded_adapter_block(self) -> bytes:
        return decompress_payload(self.payload_bytes())

    def answer_many(self, questions: list[str], read_disabled: bool = False, decoder_disabled: bool = False, parser_disabled: bool = False, adapter_disabled: bool = False, code_disabled: bool = False, **_kwargs: Any) -> list[dict[str, str | int]]:
        if read_disabled or decoder_disabled or parser_disabled or adapter_disabled or code_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        handles = [semantic_handle_for_question(str(question)) for question in questions]
        wanted = {tuple(handle) for handle in handles if handle}
        if not wanted:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        try:
            block = self.decoded_adapter_block()
        except Exception:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        found: dict[tuple[int, ...], dict[str, str | int]] = {}
        for offset in candidate_offsets_for_block(len(block)):
            handle = semantic_handle_for_anchor(anchor_text_for(block, int(offset)))
            if handle not in wanted or handle in found:
                continue
            value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            found[handle] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
            if len(found) == len(wanted):
                break
        return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]

    def answer(self, question: str, **kwargs: Any) -> dict[str, str | int]:
        return self.answer_many([str(question)], **kwargs)[0]


def shifted(rows: list[Any]) -> list[Any]:
    return rows[1:] + rows[:1] if rows else []


def evaluate_controls(cell: WeightMantissaPayloadAdapterCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]]) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    exact = cell.answer_many(questions)
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _ in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shuffled_question = cell.answer_many([str(fact["question"]) for fact in shifted(facts)])
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact), exact)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact), exact)]
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
    adapter_disabled = cell.answer_many(questions, adapter_disabled=True)
    code_disabled = cell.answer_many(questions, code_disabled=True)
    return {
        "exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "heldout_exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "random_label_twin_success": mean_metric(score_answers(random_twin, twin_reads), "exact_success"),
        "no_memory_success": mean_metric(score_answers(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_answers(facts, recency), "exact_success"),
        "shuffled_question_success": mean_metric(score_answers(facts, shuffled_question), "exact_success"),
        "shuffled_value_success": mean_metric(score_answers(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_answers(facts, shuffled_provenance), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, decoder_disabled), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, parser_disabled), "exact_success"),
        "adapter_disabled_success": mean_metric(score_answers(facts, adapter_disabled), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, code_disabled), "exact_success"),
    }


def account_bits(cell: WeightMantissaPayloadAdapterCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    useful_bits = int(len(facts) * int(CHUNK_BYTES) * 8)
    carrier_params = int(cell.carrier_parameter_count)
    apparent_equiv_params = float(carrier_params) + float(DECODER_BITS + MODEL_HEADER_BITS) / 16.0
    apparent_paper_equiv_params = apparent_equiv_params + float(SURFACE_CONTRACT_BITS) / 16.0
    payload_bit_strict_bits = float(cell.block_payload_bits + DECODER_BITS + MODEL_HEADER_BITS)
    payload_bit_paper_bits = float(cell.block_payload_bits + DECODER_BITS + MODEL_HEADER_BITS + SURFACE_CONTRACT_BITS)
    fp32_state_strict_bits = float(carrier_params * 32 + DECODER_BITS + MODEL_HEADER_BITS)
    fp32_state_paper_bits = float(carrier_params * 32 + DECODER_BITS + MODEL_HEADER_BITS + SURFACE_CONTRACT_BITS)
    payload_bit_equiv_params = payload_bit_strict_bits / 16.0
    payload_bit_paper_equiv_params = payload_bit_paper_bits / 16.0
    fp32_state_equiv_params = fp32_state_strict_bits / 16.0
    fp32_state_paper_equiv_params = fp32_state_paper_bits / 16.0
    content_scan_equiv_params = payload_bit_equiv_params
    mph_equiv_params = float(cell.block_payload_bits + DECODER_BITS + MODEL_HEADER_BITS + 16) / 16.0
    return {
        "block_payload_bits": float(cell.block_payload_bits),
        "carrier_parameter_count": float(carrier_params),
        "carrier_fp32_state_bits": float(carrier_params * 32),
        "mantissa_bits_per_parameter": float(MANTISSA_BITS_PER_PARAMETER),
        "decoder_bits": float(DECODER_BITS),
        "model_header_bits": float(MODEL_HEADER_BITS),
        "useful_retrievable_bits": float(useful_bits),
        "committed_state_bits": fp32_state_strict_bits,
        "paper_surface_accounted_bits": fp32_state_paper_bits,
        "payload_bit_strict_accounted_bits": payload_bit_strict_bits,
        "payload_bit_paper_surface_accounted_bits": payload_bit_paper_bits,
        "fp32_committed_state_bits": fp32_state_strict_bits,
        "fp32_paper_surface_accounted_bits": fp32_state_paper_bits,
        "adapter_strict_density": float(useful_bits) / max(fp32_state_equiv_params, 1.0),
        "adapter_strict_multiplier": float(useful_bits) / max(fp32_state_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "paper_surface_strict_density": float(useful_bits) / max(fp32_state_paper_equiv_params, 1.0),
        "paper_surface_strict_multiplier": float(useful_bits) / max(fp32_state_paper_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "apparent_mantissa_parameter_surface_multiplier": float(useful_bits) / max(apparent_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "apparent_mantissa_paper_surface_multiplier": float(useful_bits) / max(apparent_paper_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "payload_bit_strict_multiplier": float(useful_bits) / max(payload_bit_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "payload_bit_paper_surface_multiplier": float(useful_bits) / max(payload_bit_paper_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "fp32_strict_multiplier": float(useful_bits) / max(fp32_state_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "fp32_paper_surface_multiplier": float(useful_bits) / max(fp32_state_paper_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "same_block_content_scan_multiplier": float(useful_bits) / max(content_scan_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "same_block_undercharged_mph_multiplier": float(useful_bits) / max(mph_equiv_params, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    train, facts, source_block, source_profile = build_facts(int(seed), int(PROFILES[profile]["fact_count"]))
    cell = WeightMantissaPayloadAdapterCell(train, facts, source_block, source_profile)
    random_twin = build_random_twin(int(seed), facts)
    controls = evaluate_controls(cell, facts, random_twin)
    account = account_bits(cell, facts)
    inspection = hidden_state_inspection(cell, facts, source_block)
    controls_collapse = float(int(all(controls[key] <= 0.01 for key in ("random_label_twin_success", "no_memory_success", "shuffled_question_success", "shuffled_value_success", "shuffled_provenance_success", "read_disabled_success", "decoder_disabled_success", "parser_disabled_success", "adapter_disabled_success", "code_disabled_success"))))
    static_public_baseline_pass = float(int(account["paper_surface_strict_multiplier"] > max(float(CONTENT_SCAN_BASELINE_MULTIPLIER), float(MARGIN_BASELINE_MULTIPLIER), float(PAPER_READY_BASELINE_MULTIPLIER), float(WEIGHT_CARRIED_BASELINE_MULTIPLIER), float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER), float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER), float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER), float(CHARGED_CODEC_BASELINE_MULTIPLIER))))
    apparent_mantissa_multiplier_beats_mph = float(int(account["apparent_mantissa_parameter_surface_multiplier"] > account["same_block_undercharged_mph_multiplier"]))
    mantissa_diagnostic_candidate = float(int(controls["exact_answer_success"] >= 0.95 and controls_collapse == 1.0 and apparent_mantissa_multiplier_beats_mph == 1.0 and inspection["hidden_raw_source_prefix_detected"] == 0.0 and inspection["hidden_fact_value_row_detected"] == 0.0 and cell.parameter_count() < 100000))
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "source_domain_count": float(len({str(row["domain"]) for row in facts})),
        "adapter_parameter_count": float(cell.parameter_count()),
        "host_parameter_count_max": float(cell.parameter_count()),
        "publishable_weight_payload_candidate": 0.0,
        "mantissa_payload_diagnostic_candidate": mantissa_diagnostic_candidate,
        "strict_breakthrough_authorized": 0.0,
        "strict_600x_pass": 0.0,
        "static_public_baseline_pass": static_public_baseline_pass,
        "beats_same_block_content_scan_baseline": 0.0,
        "beats_same_block_undercharged_mph_baseline": 0.0,
        "apparent_mantissa_multiplier_beats_mph": apparent_mantissa_multiplier_beats_mph,
        "model_weight_payload_used": float(cell.model_weight_payload_used),
        "mantissa_payload_carrier_used": 1.0,
        "mantissa_steganography_diagnostic": 1.0,
        "true_base_weight_implicit_storage_authorized": 0.0,
        "state_dict_buffer_payload_used": float(cell.state_dict_buffer_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "assignment_row_count": float(cell.assignment_row_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "source_offset_routing_used": float(cell.source_offset_routing_used),
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "controls_collapse": controls_collapse,
        **account,
        **controls,
        **inspection,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_publishable_weight_payload_candidate": float(row["publishable_weight_payload_candidate"]),
        f"{SIMULATION_ID}_strict_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_general_unknown_structure_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
        f"{SIMULATION_ID}_external_simulator_authorized": 0.0,
        f"{SIMULATION_ID}_arbitrary_chat_authorized": 0.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["mantissa_payload_diagnostic_candidate"]),
    }
    for key, value in row.items():
        if key in {"profile"}:
            continue
        summary[f"{SIMULATION_ID}_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_weight_mantissa_payload_adapter_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "facts_smoke": int(FACTS_SMOKE), "facts_hard": int(FACTS_HARD), "mantissa_bits_per_parameter": int(MANTISSA_BITS_PER_PARAMETER)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_weight_mantissa_payload_adapter_metrics.json", "path": metrics_path}],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} exact={summary[f'{SIMULATION_ID}_exact_answer_success']:.3f} multiplier={summary[f'{SIMULATION_ID}_paper_surface_strict_multiplier']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
