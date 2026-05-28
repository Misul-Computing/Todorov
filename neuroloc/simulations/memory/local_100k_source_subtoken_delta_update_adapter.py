from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_llm_semantic_qa_codec import CHUNK_BYTES, SEMANTIC_HANDLE_BYTES
from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import (
    DECODER_BITS,
    FACTS_HARD,
    FACTS_SMOKE,
    MODEL_HEADER_BITS,
    mean_metric,
    paraphrase_questions,
    provenance_for_block,
    score_answers,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import (
    QUESTION_PREFIX,
    anchor_text_for,
    candidate_offsets_for_block,
    semantic_handle_for_anchor,
    semantic_handle_for_any_question,
)
from neuroloc.simulations.memory.local_100k_source_subtoken_qa_adapter import (
    SUBTOKEN_HEADER_BITS,
    SourceSubtokenQAAdapterCell,
    build_facts,
    encode_subtoken_block,
)
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import decode_unsigned_varint, encode_unsigned_varint

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_subtoken_delta_update_adapter"
SEED = env_int("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_SEED", 2137)
UPDATE_FACTS_SMOKE = env_int("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_UPDATE_FACTS_SMOKE", 64)
UPDATE_FACTS_HARD = env_int("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_UPDATE_FACTS_HARD", 512)
DELTA_PATCH_HEADER_BITS = env_int("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_PATCH_HEADER_BITS", 128)

require_positive("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_UPDATE_FACTS_SMOKE", UPDATE_FACTS_SMOKE)
require_positive("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_UPDATE_FACTS_HARD", UPDATE_FACTS_HARD)
require_positive("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_PATCH_HEADER_BITS", DELTA_PATCH_HEADER_BITS)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE, "update_fact_count": UPDATE_FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD, "update_fact_count": UPDATE_FACTS_HARD},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_SUBTOKEN_DELTA_UPDATE_ADAPTER_PROFILE must be smoke or hard")
    return value


def fact_offsets_for_block(block: bytes, facts: list[dict[str, Any]]) -> dict[int, int]:
    handles: dict[tuple[int, ...], list[int]] = {}
    for index, fact in enumerate(facts):
        handle = semantic_handle_for_any_question(str(fact["question"]))
        if handle:
            handles.setdefault(tuple(handle), []).append(int(index))
    found: dict[int, int] = {}
    for offset in candidate_offsets_for_block(len(block)):
        anchor = anchor_text_for(block, int(offset))
        handle = tuple(semantic_handle_for_anchor(anchor))
        if handle not in handles:
            continue
        value = block[int(offset) : int(offset) + int(CHUNK_BYTES)].hex()
        for index in handles[handle]:
            if index not in found and value == str(facts[index]["value"]):
                found[int(index)] = int(offset)
        if len(found) == len(facts):
            break
    if len(found) != len(facts):
        raise ValueError("fact offset recovery failed")
    return found


def replacement_value(index: int, old_value: bytes) -> bytes:
    rng = random.Random(int(SEED) + 55103 + int(index) * 9973)
    value = bytes(rng.randrange(0, 256) for _item in range(int(CHUNK_BYTES)))
    if value == old_value:
        return bytes((byte ^ 0x5A) for byte in value)
    return value


def encode_patch_payload(rows: list[tuple[int, bytes]]) -> bytes:
    payload = bytearray()
    previous = 0
    for offset, value in rows:
        delta = int(offset) - int(previous)
        if delta < 0:
            raise ValueError("patch rows must be offset-sorted")
        payload.extend(encode_unsigned_varint(delta))
        if len(value) != int(CHUNK_BYTES):
            raise ValueError("replacement value width mismatch")
        payload.extend(value)
        previous = int(offset)
    return bytes(payload)


def decode_patch_payload(payload: bytes, count: int) -> list[tuple[int, bytes]]:
    rows = []
    index = 0
    previous = 0
    for _item in range(int(count)):
        delta, index = decode_unsigned_varint(payload, index)
        offset = int(previous) + int(delta)
        value = payload[index : index + int(CHUNK_BYTES)]
        if len(value) != int(CHUNK_BYTES):
            raise ValueError("patch payload truncated")
        rows.append((offset, bytes(value)))
        index += int(CHUNK_BYTES)
        previous = int(offset)
    if index != len(payload):
        raise ValueError("unused patch payload bytes")
    return rows


def select_patch_rows(block: bytes, facts: list[dict[str, Any]], update_count: int) -> list[tuple[int, bytes]]:
    offsets = fact_offsets_for_block(block, facts)
    rows = []
    for index in range(min(int(update_count), len(facts))):
        offset = int(offsets[int(index)])
        old_value = block[offset : offset + int(CHUNK_BYTES)]
        rows.append((offset, replacement_value(index, old_value)))
    return sorted(rows, key=lambda row: int(row[0]))


class SourceSubtokenDeltaUpdateAdapterCell(SourceSubtokenQAAdapterCell):
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]], update_fact_count: int = UPDATE_FACTS_SMOKE) -> None:
        super().__init__(train_facts, test_facts, source_block, source_profile)
        base_block = self.base_decoded_adapter_block()
        patch_rows = select_patch_rows(base_block, test_facts, int(update_fact_count))
        patch_payload = encode_patch_payload(patch_rows)
        self.module.register_buffer("delta_patch_payload", torch.tensor(list(patch_payload), dtype=torch.uint8), persistent=True)
        self.module.register_buffer(
            "delta_patch_header",
            torch.tensor([int(len(patch_rows)), int(CHUNK_BYTES), int(DELTA_PATCH_HEADER_BITS)], dtype=torch.int32),
            persistent=True,
        )
        self.update_fact_count = int(len(patch_rows))
        self.base_payload_bits = int(self.block_payload_bits)
        self.delta_patch_bits = int(len(patch_payload) * 8 + int(DELTA_PATCH_HEADER_BITS))
        self.total_updated_adapter_bits = int(self.base_payload_bits + self.delta_patch_bits)
        self.model_state_patch_payload_used = 1.0
        self.patch_stream_transform_count = 1
        self.patch_stream_offset_bytes_per_row = float(len(patch_payload) - len(patch_rows) * int(CHUNK_BYTES)) / max(float(len(patch_rows)), 1.0)
        self.patch_stream_value_bytes_per_row = int(CHUNK_BYTES)

    def base_decoded_adapter_block(self) -> bytes:
        return super().decoded_adapter_block()

    def patch_payload_bytes(self) -> bytes:
        return bytes(int(item) for item in self.module.delta_patch_payload.tolist())

    def patch_rows_from_state(self, random_patch: bool = False, shuffled_patch: bool = False) -> list[tuple[int, bytes]]:
        count = int(self.module.delta_patch_header[0].item())
        rows = decode_patch_payload(self.patch_payload_bytes(), count)
        if random_patch:
            rng = random.Random(int(SEED) + 9173)
            return [(offset, bytes(rng.randrange(0, 256) for _item in range(int(CHUNK_BYTES)))) for offset, _value in rows]
        if shuffled_patch and len(rows) > 1:
            values = [value for _offset, value in rows[1:] + rows[:1]]
            return [(offset, bytes(value)) for (offset, _old), value in zip(rows, values)]
        return rows

    def updated_adapter_block(self, random_patch: bool = False, shuffled_patch: bool = False, patch_disabled: bool = False) -> bytes:
        block = bytearray(self.base_decoded_adapter_block())
        if patch_disabled:
            return bytes(block)
        for offset, value in self.patch_rows_from_state(random_patch=random_patch, shuffled_patch=shuffled_patch):
            if 0 <= int(offset) <= len(block) - int(CHUNK_BYTES):
                block[int(offset) : int(offset) + int(CHUNK_BYTES)] = value
        return bytes(block)

    def answer_many(
        self,
        questions: list[str],
        read_enabled: bool = True,
        decoder_enabled: bool = True,
        parser_enabled: bool = True,
        adapter_enabled: bool = True,
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        adapter_disabled: bool = False,
        code_disabled: bool = False,
        patch_disabled: bool = False,
        random_patch: bool = False,
        shuffled_patch: bool = False,
    ) -> list[dict[str, str | int]]:
        if not read_enabled:
            read_disabled = True
        if not decoder_enabled:
            decoder_disabled = True
        if not parser_enabled:
            parser_disabled = True
        if not adapter_enabled:
            adapter_disabled = True
        if read_disabled or decoder_disabled or parser_disabled or code_disabled or adapter_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _question in questions]
        valid_markers = (
            QUESTION_PREFIX,
            "which exact bytes follow these evidence tokens:",
            "retrieve the exact following passage for evidence terms:",
            "from the model state adapter, answer after evidence signature:",
            "what comes immediately after signature:",
        )
        handles = [semantic_handle_for_any_question(str(question)) if any(str(question).startswith(marker) for marker in valid_markers) else tuple() for question in questions]
        wanted = {tuple(handle) for handle in handles if handle}
        if not wanted:
            return [{"value": "", "provenance": "", "hit": 0} for _question in questions]
        try:
            block = self.base_decoded_adapter_block()
        except Exception:
            return [{"value": "", "provenance": "", "hit": 0} for _question in questions]
        patch_map: dict[int, bytes] = {}
        if not patch_disabled:
            patch_map = {int(offset): bytes(value) for offset, value in self.patch_rows_from_state(random_patch=random_patch, shuffled_patch=shuffled_patch)}
        self.decompression_count += 1
        self.scan_count += 1
        found: dict[tuple[int, ...], dict[str, str | int]] = {}
        for offset in candidate_offsets_for_block(len(block)):
            anchor = anchor_text_for(block, int(offset))
            handle = tuple(semantic_handle_for_anchor(anchor))
            if handle not in wanted or handle in found:
                continue
            value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            if int(offset) in patch_map:
                value = patch_map[int(offset)]
            found[handle] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
            if len(found) == len(wanted):
                break
        return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]


def expected_fact_rows(cell: SourceSubtokenDeltaUpdateAdapterCell, facts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_block = cell.base_decoded_adapter_block()
    offsets = fact_offsets_for_block(base_block, facts)
    patch_map = {int(offset): bytes(value) for offset, value in cell.patch_rows_from_state()}
    updated = []
    unchanged = []
    for index, fact in enumerate(facts):
        offset = int(offsets[int(index)])
        row = dict(fact)
        if offset in patch_map:
            value = patch_map[offset]
            row["value"] = value.hex()
            row["provenance"] = provenance_for_block(offset, value)
            updated.append(row)
        else:
            unchanged.append(row)
    return updated, unchanged


def recompressed_updated_bits(updated_block: bytes) -> int:
    learned = encode_subtoken_block(updated_block)
    payload_bits = int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"])) + len(bytes(learned["dictionary_payload"]))) * 8)
    return int(payload_bits + int(SUBTOKEN_HEADER_BITS))


def state_dict_reload_probe(cell: SourceSubtokenDeltaUpdateAdapterCell, train_facts: list[dict[str, Any]], facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]], update_count: int) -> dict[str, float]:
    updated, _unchanged = expected_fact_rows(cell, facts)
    probe_facts = updated
    questions = [str(fact["question"]) for fact in probe_facts]
    reload_cell = SourceSubtokenDeltaUpdateAdapterCell(train_facts, facts, source_block, source_profile, update_count)
    reload_cell.module.delta_patch_payload = torch.tensor([(int(item) ^ 0x7F) for item in reload_cell.module.delta_patch_payload.tolist()], dtype=torch.uint8)
    preload = reload_cell.answer_many(questions)
    preload_score = mean_metric(score_answers(probe_facts, preload), "exact_success")
    reload_cell.module.load_state_dict(cell.module.state_dict())
    reloaded = reload_cell.answer_many(questions)
    reload_score = mean_metric(score_answers(probe_facts, reloaded), "exact_success")
    state_keys = set(cell.module.state_dict().keys())
    return {
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "patch_payload_in_state_dict": float(int("delta_patch_payload" in state_keys)),
        "patch_header_in_state_dict": float(int("delta_patch_header" in state_keys)),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(PROFILES[profile]["fact_count"])
    update_count = int(PROFILES[profile]["update_fact_count"])
    train_facts, facts, source_block, source_profile = build_facts(seed, fact_count)
    cell = SourceSubtokenDeltaUpdateAdapterCell(train_facts, facts, source_block, source_profile, update_count)
    updated_facts, unchanged_facts = expected_fact_rows(cell, facts)
    updated_questions = [str(fact["question"]) for fact in updated_facts]
    unchanged_questions = paraphrase_questions(unchanged_facts)
    exact_updated_answer_success = mean_metric(score_answers(updated_facts, cell.answer_many(updated_questions)), "exact_success")
    unchanged_answer_success = mean_metric(score_answers(unchanged_facts, cell.answer_many(unchanged_questions)), "exact_success")
    random_patch_control_success = mean_metric(score_answers(updated_facts, cell.answer_many(updated_questions, random_patch=True)), "exact_success")
    patch_disabled_success = mean_metric(score_answers(updated_facts, cell.answer_many(updated_questions, patch_disabled=True)), "exact_success")
    shuffled_patch_success = mean_metric(score_answers(updated_facts, cell.answer_many(updated_questions, shuffled_patch=True)), "exact_success")
    reload_probe = state_dict_reload_probe(cell, train_facts, facts, source_block, source_profile, update_count)
    full_recompress_updated_bits = recompressed_updated_bits(cell.updated_adapter_block())
    undercharged_mph_update_bits = int(int(cell.update_fact_count) * (int(SEMANTIC_HANDLE_BYTES) * 8 + int(CHUNK_BYTES) * 8 + 64 + 16))
    matched_delta_patch_content_scan_bits = int(cell.delta_patch_bits)
    margin_over_full_recompress_bits = int(full_recompress_updated_bits - int(cell.delta_patch_bits))
    margin_over_undercharged_mph_update_bits = int(undercharged_mph_update_bits - int(cell.delta_patch_bits))
    margin_over_matched_delta_patch_content_scan_bits = int(matched_delta_patch_content_scan_bits - int(cell.delta_patch_bits))
    total_static_margin_over_full_recompress_bits = int(full_recompress_updated_bits - int(cell.total_updated_adapter_bits))
    controls_collapse = float(int(random_patch_control_success == 0.0 and patch_disabled_success == 0.0 and shuffled_patch_success == 0.0))
    margins_pass = float(int(margin_over_full_recompress_bits > 0 and margin_over_undercharged_mph_update_bits > 0))
    exact_pass = float(int(exact_updated_answer_success == 1.0 and unchanged_answer_success == 1.0 and reload_probe["state_dict_reload_success"] == 1.0 and controls_collapse == 1.0))
    product_authorized = float(int(exact_pass == 1.0 and margins_pass == 1.0 and cell.model_state_patch_payload_used == 1.0))
    engineering_pass = product_authorized
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "update_fact_count": float(cell.update_fact_count),
        "unchanged_fact_count": float(len(unchanged_facts)),
        "base_payload_bits": float(cell.base_payload_bits),
        "delta_patch_bits": float(cell.delta_patch_bits),
        "total_updated_adapter_bits": float(cell.total_updated_adapter_bits),
        "full_recompress_updated_bits": float(full_recompress_updated_bits),
        "same_block_content_scan_update_bits": float(full_recompress_updated_bits),
        "undercharged_mph_update_bits": float(undercharged_mph_update_bits),
        "matched_delta_patch_content_scan_bits": float(matched_delta_patch_content_scan_bits),
        "margin_over_full_recompress_bits": float(margin_over_full_recompress_bits),
        "margin_over_same_block_content_scan_update_bits": float(margin_over_full_recompress_bits),
        "margin_over_undercharged_mph_update_bits": float(margin_over_undercharged_mph_update_bits),
        "margin_over_matched_delta_patch_content_scan_bits": float(margin_over_matched_delta_patch_content_scan_bits),
        "total_static_margin_over_full_recompress_bits": float(total_static_margin_over_full_recompress_bits),
        "exact_updated_answer_success": float(exact_updated_answer_success),
        "unchanged_answer_success": float(unchanged_answer_success),
        "state_dict_reload_success": float(reload_probe["state_dict_reload_success"]),
        "state_dict_preload_success": float(reload_probe["state_dict_preload_success"]),
        "patch_payload_in_state_dict": float(reload_probe["patch_payload_in_state_dict"]),
        "patch_header_in_state_dict": float(reload_probe["patch_header_in_state_dict"]),
        "random_patch_control_success": float(random_patch_control_success),
        "patch_disabled_success": float(patch_disabled_success),
        "shuffled_patch_success": float(shuffled_patch_success),
        "controls_collapse": float(controls_collapse),
        "model_state_patch_payload_used": float(cell.model_state_patch_payload_used),
        "model_state_adapter_payload_used": float(cell.model_state_adapter_payload_used),
        "state_dict_buffer_payload_used": float(cell.state_dict_buffer_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "assignment_row_count": float(cell.assignment_row_count),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "delta_update_margins_pass": float(margins_pass),
        "same_block_content_scan_update_beaten": float(int(margin_over_full_recompress_bits > 0)),
        "undercharged_mph_update_beaten": float(int(margin_over_undercharged_mph_update_bits > 0)),
        "matched_delta_patch_content_scan_beaten": float(int(margin_over_matched_delta_patch_content_scan_bits > 0)),
        "source_subtoken_delta_update_product_authorized": float(product_authorized),
        "delta_update_product_not_static_compression": 1.0,
        "source_subtoken_total_static_compression_authorized": 0.0,
        "static_compression_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "broad_breakthrough_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "engineering_pass": float(engineering_pass),
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["engineering_pass"]),
    }
    for key, value in row.items():
        if key == "profile":
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
    metrics_path = output_dir / "local_100k_source_subtoken_delta_update_adapter_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "seed": int(SEED),
            "update_fact_count": int(PROFILES[profile]["update_fact_count"]),
            "delta_patch_header_bits": int(DELTA_PATCH_HEADER_BITS),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_subtoken_delta_update_adapter_metrics.json", "path": metrics_path}],
        warnings=["delta-update product only; not a total static knowledge-compression breakthrough"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} update_facts={summary[f'{SIMULATION_ID}_update_fact_count']:.0f} margin_full={summary[f'{SIMULATION_ID}_margin_over_full_recompress_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
