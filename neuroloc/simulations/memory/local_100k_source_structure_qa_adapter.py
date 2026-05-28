from __future__ import annotations

import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import (
    CHARGED_CODEC_BASELINE_MULTIPLIER,
    DECODER_BITS,
    FACTS_HARD,
    FACTS_SMOKE,
    MODEL_HEADER_BITS,
    MPH_MARGIN_TARGET,
    ORDINARY_BITS_PER_PARAMETER,
    PAPER_READY_BASELINE_MULTIPLIER,
    PRODUCT_TARGET_MULTIPLIER,
    SURFACE_CONTRACT_BITS,
    TARGET_MULTIPLIER,
    WEIGHT_CARRIED_BASELINE_MULTIPLIER,
    MarginRecompressionAdapterCell,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    accounting,
    build_facts,
    build_random_twin,
    evaluate_controls,
    false_hit_metrics,
    hidden_state_inspection,
    matched_update_recompress_baseline,
    mean_metric,
    offset_for_fact,
    paraphrase_questions,
    provenance_for_block,
    score_answers,
    tensorize_questions,
    trainable_update_controller,
    trainable_update_probe,
    update_features,
)
from neuroloc.simulations.memory.local_100k_llm_semantic_qa_codec import (
    CHUNK_BYTES,
    QUESTION_PREFIX,
    compress_block,
    decompress_block,
)
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import (
    decode_codec_name,
    encode_codec_name,
    fixed_ngrams,
    learned_codec,
    restore_learned,
)

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_structure_qa_adapter"
SEED = env_int("SOURCE_STRUCTURE_QA_ADAPTER_SEED", 2137)
STRUCTURE_HEADER_BITS = env_int("SOURCE_STRUCTURE_QA_ADAPTER_STRUCTURE_HEADER_BITS", 256)

require_positive("SOURCE_STRUCTURE_QA_ADAPTER_STRUCTURE_HEADER_BITS", STRUCTURE_HEADER_BITS)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    return "smoke"


def structure_train_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "src/spikes/spiking_brain.py",
        PROJECT_ROOT / "src/spikes/ternary_spike.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/lif_fi_curve.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/wta_dynamics.py",
        PROJECT_ROOT / "neuroloc/simulations/memory/delayed_cue_world.py",
    ]


def read_joined(paths: list[Path]) -> bytes:
    present = [path for path in paths if path.exists()]
    if not present:
        raise ValueError("no train paths found")
    return b"\n".join(path.read_bytes().replace(b"\r\n", b"\n") for path in present)


def source_profile_paths(source_profile: list[dict[str, Any]]) -> list[Path]:
    return [PROJECT_ROOT / str(row.get("path", "")) for row in source_profile if row.get("path")]


def source_holdout_counts(source_profile: list[dict[str, Any]]) -> dict[str, float]:
    train = [path for path in structure_train_paths() if path.exists()]
    test = source_profile_paths(source_profile)
    train_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in train}
    test_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in test if path.exists()}
    train_hash = {hashlib.sha256(path.read_bytes()).hexdigest() for path in train}
    test_hash = {hashlib.sha256(path.read_bytes()).hexdigest() for path in test if path.exists()}
    train_block = read_joined(train)
    test_block = read_joined([path for path in test if path.exists()])
    return {
        "source_structure_train_file_count": float(len(train)),
        "source_structure_train_test_path_overlap_count": float(len(train_rel & test_rel)),
        "source_structure_train_test_hash_overlap_count": float(len(train_hash & test_hash)),
        "source_structure_train_test_ngram_width_bytes": 64.0,
        "source_structure_train_test_ngram_overlap_count": float(len(fixed_ngrams(train_block, 64) & fixed_ngrams(test_block, 64))),
    }


def encode_structure_block(source_block: bytes) -> dict[str, Any]:
    train_block = read_joined(structure_train_paths())
    return learned_codec(train_block, source_block)


def pack_structure_payload(learned: dict[str, Any]) -> bytes:
    count_payload = bytes(learned["count_payload"])
    body_payload = bytes(learned["body_payload"])
    return count_payload + body_payload


def build_structure_adapter_module(learned: dict[str, Any]) -> Any:
    class AdapterModule(nn.Module):
        def __init__(self, row: dict[str, Any]) -> None:
            super().__init__()
            count_payload = bytes(row["count_payload"])
            body_payload = bytes(row["body_payload"])
            self.register_buffer("adapter_payload", torch.tensor(list(count_payload + body_payload), dtype=torch.uint8), persistent=True)
            self.register_buffer(
                "adapter_header",
                torch.tensor(
                    [
                        encode_codec_name(str(row["count_codec_name"])),
                        encode_codec_name(str(row["body_codec_name"])),
                        len(count_payload),
                        len(body_payload),
                        int(row["line_count"]),
                        int(row["indent_unit"]),
                    ],
                    dtype=torch.int64,
                ),
                persistent=True,
            )

    return AdapterModule(learned)


class SourceStructureQAAdapterCell(MarginRecompressionAdapterCell):
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> None:
        super().__init__(train_facts, test_facts, source_block, source_profile)
        self.structure_codec_used = 1.0
        self.structure_header_bits = int(STRUCTURE_HEADER_BITS)
        holdout = source_holdout_counts(source_profile)
        self.source_structure_train_file_count = int(holdout["source_structure_train_file_count"])
        self.source_structure_train_test_path_overlap_count = int(holdout["source_structure_train_test_path_overlap_count"])
        self.source_structure_train_test_hash_overlap_count = int(holdout["source_structure_train_test_hash_overlap_count"])
        self.source_structure_train_test_ngram_overlap_count = int(holdout["source_structure_train_test_ngram_overlap_count"])
        self.recompress_adapter_block(source_block)
        self.module.update_controller = trainable_update_controller()

    def structure_state(self) -> dict[str, Any]:
        header = [int(item) for item in self.module.adapter_header.tolist()]
        payload = self.payload_bytes()
        count_len = int(header[2])
        body_len = int(header[3])
        return {
            "count_codec_name": decode_codec_name(int(header[0])),
            "body_codec_name": decode_codec_name(int(header[1])),
            "count_payload": payload[:count_len],
            "body_payload": payload[count_len : count_len + body_len],
            "line_count": int(header[4]),
            "indent_unit": int(header[5]),
        }

    def decoded_adapter_block(self) -> bytes:
        return restore_learned(self.structure_state())

    def recompress_adapter_block(self, source_block: bytes) -> None:
        learned = encode_structure_block(source_block)
        self.codec_name = "source_structure_split"
        self.module = build_structure_adapter_module(learned)
        self.block_payload_bits = int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"]))) * 8 + int(STRUCTURE_HEADER_BITS))
        self.adapter_model_state_bits = int(self.block_payload_bits + int(MODEL_HEADER_BITS))
        self.candidate_count = len(list(range(0, max(0, len(source_block) - int(CHUNK_BYTES) + 1), int(CHUNK_BYTES))))
        self.adapter_recompression_update_count += 1

    def apply_trainable_update(self, fact: dict[str, Any], new_value: bytes, controller_enabled: bool = True) -> dict[str, float]:
        if not controller_enabled:
            return {"accepted": 0.0, "old_answer_gone": 0.0, "new_answer_success": 0.0}
        decoded_block = self.decoded_adapter_block()
        offset = offset_for_fact(decoded_block, fact)
        old_value = decoded_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        features = update_features(offset, len(decoded_block), old_value, new_value)
        score = float(self.module.update_controller(features).detach().cpu().item())
        if score < 0.5:
            return {"accepted": 0.0, "old_answer_gone": 0.0, "new_answer_success": 0.0}
        updated_block = bytearray(decoded_block)
        updated_block[int(offset) : int(offset) + len(new_value)] = new_value
        controller = self.module.update_controller
        self.recompress_adapter_block(bytes(updated_block))
        self.module.update_controller = controller
        answer = self.answer(str(fact["question"]))
        return {
            "accepted": 1.0,
            "old_answer_gone": float(int(answer["value"] != str(fact["value"]))),
            "new_answer_success": float(int(answer["value"] == new_value.hex() and answer["provenance"] == provenance_for_block(offset, new_value))),
        }


def raw_codec_bits(source_block: bytes) -> int:
    _name, payload = compress_block(source_block)
    return int(len(payload) * 8)


def raw_baseline_metrics(useful_bits: int, source_block: bytes) -> dict[str, float]:
    bits = raw_codec_bits(source_block)
    raw_content_scan_bits = int(bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS))
    raw_mph_bits = int(bits + int(MODEL_HEADER_BITS) + 16 + int(DECODER_BITS))

    def multiplier(row_bits: int) -> float:
        return float(useful_bits) / max(float(row_bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)

    return {
        "raw_best_standard_payload_bits": float(bits),
        "raw_executable_content_scan_bits": float(raw_content_scan_bits),
        "raw_undercharged_mph_bits": float(raw_mph_bits),
        "raw_executable_content_scan_multiplier": multiplier(raw_content_scan_bits),
        "raw_undercharged_mph_multiplier": multiplier(raw_mph_bits),
    }


def structure_baseline_metrics(useful_bits: int, account: dict[str, float]) -> dict[str, float]:
    same_bits = int(account["block_payload_bits"] + account["model_header_bits"] + account["decoder_bits"])

    def multiplier(row_bits: int) -> float:
        return float(useful_bits) / max(float(row_bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)

    return {
        "same_structure_content_scan_success": 1.0,
        "same_structure_content_scan_bits": float(same_bits),
        "same_structure_content_scan_multiplier": multiplier(same_bits),
        "same_structure_content_scan_not_beaten": float(int(account["adapter_strict_multiplier"] <= multiplier(same_bits))),
    }


def trainable_update_probe_structure(facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    if len(facts) < 4:
        return {"trainable_recompression_update_success": 0.0}
    cell = SourceStructureQAAdapterCell([], facts, source_block, source_profile)
    successes = []
    disabled = []
    for fact in facts[:4]:
        decoded = cell.decoded_adapter_block()
        offset = offset_for_fact(decoded, fact)
        old_value = decoded[int(offset) : int(offset) + int(CHUNK_BYTES)]
        new_value = bytes((byte ^ 0x33) for byte in old_value)
        disabled.append(cell.apply_trainable_update(fact, new_value, controller_enabled=False)["new_answer_success"])
        result = cell.apply_trainable_update(fact, new_value, controller_enabled=True)
        successes.append(float(result["accepted"] == 1.0 and result["old_answer_gone"] == 1.0 and result["new_answer_success"] == 1.0))
    return {
        "trainable_recompression_update_success": float(min(successes)),
        "trainable_recompression_update_count": float(len(successes)),
        "trainable_recompression_update_bits": float(cell.block_payload_bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS) + int(cell.parameter_count()) * 16),
        "update_controller_disabled_success": float(max(disabled)),
    }


def corrupt_structure_payload(cell: SourceStructureQAAdapterCell) -> None:
    cell.module.adapter_payload = torch.tensor([(int(item) ^ 0xA5) for item in cell.module.adapter_payload.tolist()], dtype=torch.uint8)


def host_probe_structure(host: Any, facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    questions = paraphrase_questions(facts[: min(64, len(facts))])
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    score = mean_metric(score_answers(facts[: len(questions)], answers), "exact_success")
    state_keys = set(host.module.state_dict().keys())
    reload_cell = SourceStructureQAAdapterCell([], facts, source_block, source_profile)
    corrupt_structure_payload(reload_cell)
    reload_host = type(host)(reload_cell)
    preload_answers = reload_host.answer_many(questions)
    preload_score = mean_metric(score_answers(facts[: len(questions)], preload_answers), "exact_success")
    reload_host.module.load_state_dict(host.module.state_dict())
    reload_answers = reload_host.answer_many(questions)
    reload_score = mean_metric(score_answers(facts[: len(questions)], reload_answers), "exact_success")
    return {
        "forward_shape_success": float(int(tuple(output.shape) == (len(questions), 16))),
        "adapter_payload_in_state_dict": float(int("adapter_module.adapter_payload" in state_keys)),
        "adapter_header_in_state_dict": float(int("adapter_module.adapter_header" in state_keys)),
        "update_controller_in_state_dict": float(int(any("update_controller" in key for key in state_keys))),
        "paraphrase_answer_success": float(score),
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "parameter_count": float(host.parameter_count()),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    train_facts, facts, source_block, source_profile = build_facts(seed, fact_count)
    random_twin = build_random_twin(seed, facts)
    cell = SourceStructureQAAdapterCell(train_facts, facts, source_block, source_profile)
    controls = evaluate_controls(cell, facts, random_twin)
    account = accounting(cell, len(facts))
    raw_baselines = raw_baseline_metrics(int(account["useful_retrievable_bits"]), source_block)
    structure_baselines = structure_baseline_metrics(int(account["useful_retrievable_bits"]), account)
    update_probe = trainable_update_probe_structure(facts, source_block, source_profile)
    transformer_probe = host_probe_structure(TinyTransformerAdapterHost(cell), facts, source_block, source_profile)
    recurrent_probe = host_probe_structure(TinyRecurrentStateAdapterHost(cell), facts, source_block, source_profile)
    false_hits = false_hit_metrics(cell, facts)
    inspection = hidden_state_inspection(cell, facts, source_block)
    holdout = source_holdout_counts(source_profile)
    controls_collapse = float(int(controls["random_label_twin_success"] == 0.0 and controls["no_memory_success"] == 0.0 and controls["read_disabled_success"] == 0.0 and controls["decoder_disabled_success"] == 0.0 and controls["parser_disabled_success"] == 0.0 and controls["adapter_disabled_success"] == 0.0 and controls["code_disabled_success"] == 0.0 and max(false_hits.values()) == 0.0))
    raw_content_scan_beaten = float(int(account["adapter_strict_multiplier"] > raw_baselines["raw_executable_content_scan_multiplier"]))
    raw_mph_beaten = float(int(account["adapter_strict_multiplier"] > raw_baselines["raw_undercharged_mph_multiplier"]))
    transformer_surface_pass = float(int(transformer_probe["forward_shape_success"] == 1.0 and transformer_probe["adapter_payload_in_state_dict"] == 1.0 and transformer_probe["adapter_header_in_state_dict"] == 1.0 and transformer_probe["paraphrase_answer_success"] >= 0.95 and transformer_probe["state_dict_reload_success"] >= 0.95))
    recurrent_surface_pass = float(int(recurrent_probe["forward_shape_success"] == 1.0 and recurrent_probe["adapter_payload_in_state_dict"] == 1.0 and recurrent_probe["adapter_header_in_state_dict"] == 1.0 and recurrent_probe["paraphrase_answer_success"] >= 0.95 and recurrent_probe["state_dict_reload_success"] >= 0.95))
    engineering_pass = float(int(controls["exact_answer_success"] >= 0.95 and controls["heldout_exact_answer_success"] >= 0.95 and controls["paraphrase_stable_answer_success"] >= 0.95 and controls_collapse == 1.0 and raw_content_scan_beaten == 1.0 and raw_mph_beaten == 1.0 and transformer_surface_pass == 1.0 and recurrent_surface_pass == 1.0 and update_probe["trainable_recompression_update_success"] == 1.0 and inspection["hidden_fact_value_row_detected"] == 0.0 and inspection["hidden_raw_source_prefix_detected"] == 0.0))
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "source_block_bytes": float(len(source_block)),
        "source_file_count": float(len(source_profile)),
        "adapter_parameter_count": float(cell.parameter_count()),
        "structure_codec_used": float(cell.structure_codec_used),
        "structure_header_bits": float(cell.structure_header_bits),
        "structure_qa_product_candidate": engineering_pass,
        "raw_content_scan_beaten": raw_content_scan_beaten,
        "raw_undercharged_mph_beaten": raw_mph_beaten,
        "same_structure_content_scan_beaten": 0.0,
        "source_block_codec_product_authorized": engineering_pass,
        "source_block_codec_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "general_unknown_structure_breakthrough_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "model_state_adapter_payload_used": float(cell.model_state_adapter_payload_used),
        "state_dict_buffer_payload_used": float(cell.state_dict_buffer_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "assignment_row_count": float(cell.assignment_row_count),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "reads_from_compressed_model_state": float(cell.reads_from_compressed_model_state),
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "controls_collapse": controls_collapse,
        "transformer_surface_pass": transformer_surface_pass,
        "recurrent_surface_pass": recurrent_surface_pass,
        "engineering_pass": engineering_pass,
        **account,
        **controls,
        **false_hits,
        **inspection,
        **holdout,
        **raw_baselines,
        **structure_baselines,
        **update_probe,
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
    metrics_path = output_dir / "local_100k_source_structure_qa_adapter_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "structure_header_bits": int(STRUCTURE_HEADER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_structure_qa_adapter_metrics.json", "path": metrics_path}],
        warnings=["beats raw content-scan and raw mph diagnostics; same-structure content scan remains not beaten"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} adapter_multiplier={summary[f'{SIMULATION_ID}_adapter_strict_multiplier']:.6f} raw_scan_beaten={summary[f'{SIMULATION_ID}_raw_content_scan_beaten']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
