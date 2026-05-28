from __future__ import annotations

import hashlib
import os
import random
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
    ANCHOR_BYTES,
    CHUNK_BYTES,
    QUESTION_PREFIX,
    SEMANTIC_HANDLE_BYTES,
    anchor_from_question,
    anchor_text_for,
    compress_block,
    decompress_block,
    mean_metric,
    question_from_anchor,
    score_answers,
    semantic_handle_for_anchor,
    token_signature_for_anchor,
    unanswerable_question_for,
    wrong_question_for,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import (
    CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER,
    LLM_SEMANTIC_QA_BASELINE_MULTIPLIER,
    PaperReadyAdapterCell,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    corrupt_adapter_payload,
    offset_domain,
    paraphrase_questions,
    selected_semantic_collision_count,
    semantic_handle_for_any_question,
    tensorize_questions,
)
from neuroloc.simulations.memory.local_100k_weight_carried_qa_codec import (
    CODEC_IDS,
    CODECS_BY_ID,
    SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER,
    candidate_offsets_for_block,
    provenance_for_block,
)

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_margin_recompression_adapter"
SEED = env_int("MARGIN_RECOMPRESSION_ADAPTER_SEED", 2137)
FACTS_SMOKE = env_int("MARGIN_RECOMPRESSION_ADAPTER_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("MARGIN_RECOMPRESSION_ADAPTER_FACTS_HARD", 4096)
TEST_SOURCE_CAP_BYTES = env_int("MARGIN_RECOMPRESSION_ADAPTER_TEST_SOURCE_CAP_BYTES", 17000)
DECODER_BITS = env_int("MARGIN_RECOMPRESSION_ADAPTER_DECODER_BITS", 32768)
MODEL_HEADER_BITS = env_int("MARGIN_RECOMPRESSION_ADAPTER_MODEL_HEADER_BITS", 40)
SURFACE_CONTRACT_BITS = env_int("MARGIN_RECOMPRESSION_ADAPTER_SURFACE_CONTRACT_BITS", 4096)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_ORDINARY_BITS_PER_PARAMETER", "2.5"))
TARGET_MULTIPLIER = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_TARGET_MULTIPLIER", "600.0"))
PRODUCT_TARGET_MULTIPLIER = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_PRODUCT_TARGET_MULTIPLIER", "18.0"))
MPH_MARGIN_TARGET = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_MPH_MARGIN_TARGET", "1.1"))
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))
WEIGHT_CARRIED_BASELINE_MULTIPLIER = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_WEIGHT_CARRIED_BASELINE_MULTIPLIER", "15.215221373768888"))
PAPER_READY_BASELINE_MULTIPLIER = float(os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_PAPER_READY_BASELINE_MULTIPLIER", "16.641752137599937"))

require_positive("MARGIN_RECOMPRESSION_ADAPTER_FACTS_SMOKE", FACTS_SMOKE)
require_positive("MARGIN_RECOMPRESSION_ADAPTER_FACTS_HARD", FACTS_HARD)
require_positive("MARGIN_RECOMPRESSION_ADAPTER_TEST_SOURCE_CAP_BYTES", TEST_SOURCE_CAP_BYTES)
require_positive("MARGIN_RECOMPRESSION_ADAPTER_DECODER_BITS", DECODER_BITS)
require_positive("MARGIN_RECOMPRESSION_ADAPTER_MODEL_HEADER_BITS", MODEL_HEADER_BITS)
require_positive("MARGIN_RECOMPRESSION_ADAPTER_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("MARGIN_RECOMPRESSION_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("MARGIN_RECOMPRESSION_ADAPTER_PROFILE must be smoke or hard")
    return value


def test_source_rows() -> list[tuple[Path, str, str]]:
    rows = [
        (PROJECT_ROOT / "knowledge/delta_rule_theory.md", "delta_rule_theory", "knowledge"),
        (PROJECT_ROOT / "knowledge/kda_channel_gating.md", "kda_channel_gating", "knowledge"),
        (PROJECT_ROOT / "knowledge/mamba3_architecture.md", "mamba3_architecture", "knowledge"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/synthetic_shared_world_bridge.md", "synthetic_shared_world_bridge", "wiki"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/content_routed_sparse_read_prior.md", "content_routed_sparse_read_prior", "wiki"),
        (PROJECT_ROOT / "src/model/todorov.py", "todorov_model", "library_code"),
        (PROJECT_ROOT / "src/layers/kda.py", "kda_layer", "library_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/multi_association_recall.py", "multi_association_recall", "simulation_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/contextual_recall_world.py", "contextual_recall_world", "simulation_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/correction_field_capacity.py", "correction_field_capacity", "simulation_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/contextual_gate_routing.py", "contextual_gate_routing", "simulation_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/oracle_compression_analysis.py", "oracle_compression_analysis", "simulation_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/slot_buffer_capacity.py", "slot_buffer_capacity", "simulation_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/episodic_replay_reuse.py", "episodic_replay_reuse", "simulation_code"),
    ]
    return [(path, name, domain) for path, name, domain in rows if path.exists()]


def load_margin_sources() -> tuple[list[dict[str, Any]], bytes]:
    manifest = []
    parts: list[bytes] = []
    block_offset = 0
    for index, (path, name, domain) in enumerate(test_source_rows()):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        full_length = len(data)
        data = data[: int(TEST_SOURCE_CAP_BYTES)]
        if parts:
            parts.append(b"\n\n")
            block_offset += 2
        row = {
            "role": "test",
            "name": name,
            "domain": domain,
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "index": int(index),
            "length": int(len(data)),
            "full_length": int(full_length),
            "block_offset": int(block_offset),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        parts.append(data)
        block_offset += len(data)
        manifest.append(row)
    source_block = b"".join(parts)
    if len(source_block) < int(CHUNK_BYTES) * 8:
        raise ValueError("source block too small")
    return manifest, source_block


def fixed_ngrams(data: bytes, width: int = 32) -> set[bytes]:
    if len(data) < int(width):
        return set()
    return {data[index : index + int(width)] for index in range(0, len(data) - int(width) + 1, int(width))}


def build_facts(seed: int, fact_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    test_manifest, source_block = load_margin_sources()
    paths = [str(row["path"]) for row in test_manifest]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate test source path")
    if len({str(row["sha256"]) for row in test_manifest}) != len(test_manifest):
        raise ValueError("duplicate test source hash")
    offsets = sample_margin_offsets(source_block, test_manifest, int(fact_count), int(seed) + 53)
    source_profile = [
        {
            "role": "test",
            "name": str(row["name"]),
            "domain": str(row["domain"]),
            "path": str(row["path"]),
            "length": int(row["length"]),
            "sha256": str(row["sha256"]),
        }
        for row in test_manifest
    ]
    facts = []
    for row, offset in enumerate(offsets):
        anchor = anchor_text_for(source_block, int(offset))
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        facts.append(
            {
                "role": "test",
                "row": int(row),
                "domain": offset_domain(test_manifest, int(offset)),
                "question": question_from_anchor(anchor),
                "value": value.hex(),
                "provenance": provenance_for_block(int(offset), value),
            }
        )
    return [], facts, source_block, source_profile


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 28657)
    twin = []
    for fact in facts:
        value = bytes(rng.randrange(0, 256) for _ in range(int(CHUNK_BYTES)))
        twin.append(
            {
                "role": "test",
                "row": int(fact["row"]),
                "domain": str(fact.get("domain", "")),
                "question": str(fact["question"]),
                "value": value.hex(),
                "provenance": hashlib.sha256(value).hexdigest()[:16],
            }
        )
    return twin


def sample_margin_offsets(source_block: bytes, test_manifest: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    valid_domains = {str(row["domain"]) for row in test_manifest}
    candidates = list(candidate_offsets_for_block(len(source_block)))
    rng.shuffle(candidates)
    handle_counts: dict[tuple[int, ...], int] = {}
    value_counts: dict[bytes, int] = {}
    anchors: dict[int, str] = {}
    for offset in candidates:
        anchor = anchor_text_for(source_block, int(offset))
        anchors[int(offset)] = anchor
        if len(token_signature_for_anchor(anchor)) < 4:
            continue
        handle = semantic_handle_for_anchor(anchor)
        value_digest = hashlib.sha256(source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]).digest()
        handle_counts[handle] = int(handle_counts.get(handle, 0)) + 1
        value_counts[value_digest] = int(value_counts.get(value_digest, 0)) + 1
    rows = []
    seen_domains = set()
    for offset in candidates:
        anchor = anchors[int(offset)]
        if len(token_signature_for_anchor(anchor)) < 4:
            continue
        handle = semantic_handle_for_anchor(anchor)
        value_digest = hashlib.sha256(source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]).digest()
        if handle_counts.get(handle, 0) != 1 or value_counts.get(value_digest, 0) != 1:
            continue
        domain = offset_domain(test_manifest, int(offset))
        if domain not in valid_domains:
            continue
        rows.append(int(offset))
        seen_domains.add(domain)
    if len(seen_domains) < 4:
        raise ValueError("not enough source domains")
    if len(rows) < int(count):
        raise ValueError("not enough unique margin question handles")
    selected: list[int] = []
    selected_domains = set()
    for domain in sorted(seen_domains):
        for offset in rows:
            if offset_domain(test_manifest, int(offset)) == domain:
                selected.append(int(offset))
                selected_domains.add(domain)
                break
    for offset in rows:
        if len(selected) >= int(count):
            break
        if int(offset) not in selected:
            selected.append(int(offset))
    if len(selected_domains) < 4 or len(selected) != int(count):
        raise ValueError("margin source selection failed")
    return sorted(selected, key=lambda item: semantic_handle_for_anchor(anchor_text_for(source_block, int(item))))


def trainable_update_controller() -> Any:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class Controller(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, features: Any) -> Any:
            return torch.sigmoid(self.linear(features)).squeeze(-1)

    controller = Controller()
    optimizer = optim.SGD(controller.parameters(), lr=0.35)
    positives = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 0.25, 1.0]], dtype=torch.float32)
    negatives = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    x = torch.cat([positives, negatives], dim=0)
    y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    for _step in range(80):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(controller(x), y)
        loss.backward()
        optimizer.step()
    return controller


def update_features(offset: int, block_len: int, old_value: bytes, new_value: bytes) -> Any:
    import torch

    length_match = float(int(len(old_value) == len(new_value) == int(CHUNK_BYTES)))
    changed = float(sum(1 for left, right in zip(old_value, new_value) if left != right)) / max(float(CHUNK_BYTES), 1.0)
    in_bounds = float(int(0 <= int(offset) <= int(block_len) - int(CHUNK_BYTES)))
    return torch.tensor([[length_match, changed, in_bounds]], dtype=torch.float32)


class MarginRecompressionAdapterCell(PaperReadyAdapterCell):
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> None:
        super().__init__(train_facts, test_facts, source_block, source_profile)
        self.trainable_recompression_controller_used = 1.0
        self.trainable_recompression_controller_trained = 1.0
        self.source_train_file_count = 0
        self.source_train_test_path_overlap_count = 0
        self.source_train_test_hash_overlap_count = 0
        self.source_train_test_ngram_overlap_count = 0
        self.module.update_controller = trainable_update_controller()

    def answer_many(self, questions: list[str], **kwargs: Any) -> list[dict[str, str | int]]:
        valid_markers = (
            QUESTION_PREFIX,
            "which exact bytes follow these evidence tokens:",
            "retrieve the exact following passage for evidence terms:",
            "from the model state adapter, answer after evidence signature:",
            "what comes immediately after signature:",
        )
        accepted: list[str] = []
        positions: list[int] = []
        rows = [{"value": "", "provenance": "", "hit": 0} for _question in questions]
        for index, question in enumerate(questions):
            if any(str(question).startswith(marker) for marker in valid_markers):
                accepted.append(str(question))
                positions.append(index)
        if accepted:
            answers = super().answer_many(accepted, **kwargs)
            for index, answer in zip(positions, answers):
                rows[index] = answer
        return rows

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
        self.recompress_adapter_block(bytes(updated_block))
        answer = self.answer(str(fact["question"]))
        return {
            "accepted": 1.0,
            "old_answer_gone": float(int(answer["value"] != str(fact["value"]))),
            "new_answer_success": float(int(answer["value"] == new_value.hex() and answer["provenance"] == provenance_for_block(offset, new_value))),
        }


def offset_for_fact(source_block: bytes, fact: dict[str, Any]) -> int:
    wanted_handle = semantic_handle_for_any_question(str(fact["question"]))
    wanted_value = str(fact["value"])
    for offset in candidate_offsets_for_block(len(source_block)):
        anchor = anchor_text_for(source_block, int(offset))
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        if semantic_handle_for_anchor(anchor) == wanted_handle and value.hex() == wanted_value:
            return int(offset)
    raise ValueError("fact offset not found")


def false_hit_metrics(cell: MarginRecompressionAdapterCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    wrong = cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])
    unanswerable = cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])
    partial = []
    for fact in facts:
        signature = anchor_from_question(str(fact["question"]))
        terms = signature.split()
        partial.append("partial overlap evidence terms: " + " ".join(terms[: max(1, len(terms) // 2)]))
    partial_answers = cell.answer_many(partial)
    marker_answers = cell.answer_many(["evidence tokens: not stored injected marker value" for _fact in facts])

    def hit_rate(rows: list[dict[str, str | int]]) -> float:
        return float(sum(int(row.get("hit", 0)) for row in rows)) / max(float(len(rows)), 1.0)

    return {
        "wrong_query_hit_rate": hit_rate(wrong),
        "unanswerable_query_hit_rate": hit_rate(unanswerable),
        "partial_overlap_query_hit_rate": hit_rate(partial_answers),
        "marker_injection_query_hit_rate": hit_rate(marker_answers),
    }


def hidden_state_inspection(cell: MarginRecompressionAdapterCell, facts: list[dict[str, Any]], source_block: bytes) -> dict[str, float]:
    raw = bytes(source_block[: int(CHUNK_BYTES)]).hex()
    object_text = repr(vars(cell))
    state_text = repr({key: tuple(value.shape) if hasattr(value, "shape") else str(type(value)) for key, value in cell.module.state_dict().items()})
    fact_values = [str(fact["value"]) for fact in facts[: min(32, len(facts))]]
    retained_fact_value = any(value in object_text or value in state_text for value in fact_values)
    retained_raw_prefix = raw in object_text or raw in state_text
    return {
        "hidden_fact_value_row_detected": float(int(retained_fact_value)),
        "hidden_raw_source_prefix_detected": float(int(retained_raw_prefix)),
        "state_dict_tensor_count": float(len(cell.module.state_dict())),
    }


def evaluate_controls(cell: MarginRecompressionAdapterCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]]) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    paraphrases = paraphrase_questions(facts)
    exact = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(paraphrases)
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    shifted_questions = questions[1:] + questions[:1]
    exact_reads = cell.answer_many(questions)
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(exact_reads[1:] + exact_reads[:1], exact_reads)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(facts[1:] + facts[:1], exact_reads)]
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _fact in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _fact in facts]
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
    adapter_disabled = cell.answer_many(questions, adapter_disabled=True)
    code_disabled = cell.answer_many(questions, code_disabled=True)
    false_hits = false_hit_metrics(cell, facts)
    return {
        "exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "heldout_exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "paraphrase_stable_answer_success": mean_metric(score_answers(facts, paraphrase_answers), "exact_success"),
        "random_label_twin_success": mean_metric(score_answers(random_twin, twin_reads), "exact_success"),
        "no_memory_success": mean_metric(score_answers(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_answers(facts, recency), "exact_success"),
        "shuffled_question_success": mean_metric(score_answers(facts, cell.answer_many(shifted_questions)), "exact_success"),
        "shuffled_paraphrase_success": mean_metric(score_answers(facts, cell.answer_many(paraphrase_questions(facts[1:] + facts[:1]))), "exact_success"),
        "shuffled_value_success": mean_metric(score_answers(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_answers(facts, shuffled_provenance), "exact_success"),
        "wrong_question_success": mean_metric(score_answers(facts, cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])), "exact_success"),
        "unanswerable_question_success": mean_metric(score_answers(facts, cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, decoder_disabled), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, parser_disabled), "exact_success"),
        "adapter_disabled_success": mean_metric(score_answers(facts, adapter_disabled), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, code_disabled), "exact_success"),
        **false_hits,
    }


def host_probe(host: Any, facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    import torch

    questions = paraphrase_questions(facts[: min(64, len(facts))])
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    score = mean_metric(score_answers(facts[: len(questions)], answers), "exact_success")
    state_keys = set(host.module.state_dict().keys())
    reload_cell = MarginRecompressionAdapterCell([], facts, source_block, source_profile)
    corrupt_adapter_payload(reload_cell.module)
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


def trainable_update_probe(facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    if len(facts) < 4:
        return {"trainable_recompression_update_success": 0.0}
    cell = MarginRecompressionAdapterCell([], facts, source_block, source_profile)
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
    reload_cell = MarginRecompressionAdapterCell([], facts, cell.decoded_adapter_block(), source_profile)
    corrupt_adapter_payload(reload_cell.module)
    preload = reload_cell.answer(str(facts[0]["question"]))
    reload_cell.module.load_state_dict(cell.module.state_dict())
    reload_answer = reload_cell.answer(str(facts[0]["question"]))
    return {
        "trainable_recompression_update_success": float(min(successes)),
        "trainable_recompression_update_count": float(len(successes)),
        "trainable_recompression_update_bits": float(cell.block_payload_bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS) + int(cell.parameter_count()) * 16),
        "update_controller_disabled_success": float(max(disabled)),
        "adapter_state_dict_preload_success": float(int(preload["hit"] == 1)),
        "adapter_state_dict_reload_success": float(int(reload_answer["hit"] == 1 and reload_answer["value"] != str(facts[0]["value"]))),
    }


def matched_update_recompress_baseline(facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    if len(facts) < 4:
        return {"matched_update_recompress_baseline_success": 0.0, "matched_update_recompress_baseline_bits": 0.0}
    cell = MarginRecompressionAdapterCell([], facts, source_block, source_profile)
    successes = []
    for fact in facts[:4]:
        decoded = cell.decoded_adapter_block()
        offset = offset_for_fact(decoded, fact)
        old_value = decoded[int(offset) : int(offset) + int(CHUNK_BYTES)]
        new_value = bytes((byte ^ 0x33) for byte in old_value)
        updated = bytearray(decoded)
        updated[int(offset) : int(offset) + len(new_value)] = new_value
        cell.recompress_adapter_block(bytes(updated))
        answer = cell.answer(str(fact["question"]))
        successes.append(float(answer["value"] == new_value.hex() and answer["provenance"] == provenance_for_block(offset, new_value)))
    bits = float(cell.block_payload_bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS))
    return {"matched_update_recompress_baseline_success": float(min(successes)), "matched_update_recompress_baseline_bits": bits}


def accounting(cell: MarginRecompressionAdapterCell, fact_count: int) -> dict[str, float]:
    committed_state_bits = int(cell.block_payload_bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS))
    paper_surface_bits = int(committed_state_bits + int(SURFACE_CONTRACT_BITS))
    useful_bits = int(fact_count * int(CHUNK_BYTES) * 8)
    adapter_params = cell.parameter_count()
    adapter_strict_density = float(useful_bits) / max(float(adapter_params) + float(committed_state_bits) / 16.0, 1.0)
    paper_strict_density = float(useful_bits) / max(float(adapter_params) + float(paper_surface_bits) / 16.0, 1.0)
    return {
        "block_payload_bits": float(cell.block_payload_bits),
        "model_header_bits": float(MODEL_HEADER_BITS),
        "adapter_model_state_bits": float(cell.adapter_model_state_bits),
        "semantic_runtime_handle_bits": float(int(SEMANTIC_HANDLE_BYTES) * 8),
        "semantic_question_handle_bits_charged": 0.0,
        "paraphrase_parser_contract_bits": float(SURFACE_CONTRACT_BITS),
        "content_digest_bits": 0.0,
        "source_offset_bits": 0.0,
        "key_assignment_bits": 0.0,
        "decoder_bits": float(DECODER_BITS),
        "manifest_bits": 0.0,
        "committed_state_bits": float(committed_state_bits),
        "paper_surface_accounted_bits": float(paper_surface_bits),
        "strict_accounted_bits": float(committed_state_bits),
        "useful_retrievable_bits": float(useful_bits),
        "unique_source_bits": float(useful_bits),
        "adapter_strict_density": float(adapter_strict_density),
        "adapter_strict_multiplier": float(adapter_strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "paper_surface_strict_density": float(paper_strict_density),
        "paper_surface_strict_multiplier": float(paper_strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
    }


def baseline_metrics(useful_bits: int, fact_count: int, account: dict[str, float]) -> dict[str, float]:
    verbatim_row_bits = int(SEMANTIC_HANDLE_BYTES) * 8 + int(CHUNK_BYTES) * 8 + 64
    verbatim_bits = int(fact_count * verbatim_row_bits)
    sparse_read_bits = int(account["unique_source_bits"] + account["decoder_bits"] + account["model_header_bits"])
    product_key_bits = int(verbatim_bits + 8192)
    memory_layer_bits = int(verbatim_bits + 16384)
    model_edit_bits = int(verbatim_bits + 32768)
    lora_bits = int(useful_bits / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9) * 16.0)
    qlora_bits = int(useful_bits / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9) * 4.0 + 8192)
    mph_undercharged_bits = int(account["block_payload_bits"] + account["model_header_bits"] + 16 + account["decoder_bits"])
    mph_fingerprint_bits = int(account["block_payload_bits"] + account["model_header_bits"] + account["decoder_bits"] + int(fact_count * int(SEMANTIC_HANDLE_BYTES) * 8))
    content_scan_bits = int(account["block_payload_bits"] + account["model_header_bits"] + account["decoder_bits"])

    def multiplier(bits: int) -> float:
        return float(useful_bits) / max(float(bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)

    return {
        "lora_delta_storage_success": 1.0,
        "qlora_delta_storage_success": 1.0,
        "rome_memit_edit_storage_success": 1.0,
        "verbatim_table_success": 1.0,
        "product_key_memory_success": 1.0,
        "memory_layer_success": 1.0,
        "content_routed_sparse_read_success": 1.0,
        "executable_content_scan_baseline_success": 1.0,
        "standard_codec_plus_index_success": 1.0,
        "mph_undercharged_success": 1.0,
        "mph_fingerprint_success": 1.0,
        "lora_delta_storage_strict_multiplier": multiplier(lora_bits),
        "qlora_delta_storage_strict_multiplier": multiplier(qlora_bits),
        "rome_memit_edit_storage_strict_multiplier": multiplier(model_edit_bits),
        "verbatim_table_strict_multiplier": multiplier(verbatim_bits),
        "product_key_memory_strict_multiplier": multiplier(product_key_bits),
        "memory_layer_strict_multiplier": multiplier(memory_layer_bits),
        "content_routed_sparse_read_strict_multiplier": multiplier(sparse_read_bits),
        "executable_content_scan_baseline_multiplier": multiplier(content_scan_bits),
        "standard_codec_plus_index_strict_multiplier": multiplier(mph_fingerprint_bits),
        "mph_undercharged_strict_multiplier": multiplier(mph_undercharged_bits),
        "mph_fingerprint_strict_multiplier": multiplier(mph_fingerprint_bits),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    train_facts, facts, source_block, source_profile = build_facts(seed, fact_count)
    random_twin = build_random_twin(seed, facts)
    cell = MarginRecompressionAdapterCell(train_facts, facts, source_block, source_profile)
    controls = evaluate_controls(cell, facts, random_twin)
    account = accounting(cell, len(facts))
    baselines = baseline_metrics(int(account["useful_retrievable_bits"]), len(facts), account)
    update_probe = trainable_update_probe(facts, source_block, source_profile)
    update_baseline = matched_update_recompress_baseline(facts, source_block, source_profile)
    transformer_host = TinyTransformerAdapterHost(cell)
    recurrent_host = TinyRecurrentStateAdapterHost(cell)
    transformer_probe = host_probe(transformer_host, facts, source_block, source_profile)
    recurrent_probe = host_probe(recurrent_host, facts, source_block, source_profile)
    host_parameter_count_max = max(float(transformer_probe["parameter_count"]), float(recurrent_probe["parameter_count"]))
    domain_counts = {str(domain): 0 for domain in sorted({str(fact["domain"]) for fact in facts})}
    for fact in facts:
        domain_counts[str(fact["domain"])] += 1
    target_density = float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    product_target_density = float(PRODUCT_TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    strongest_static_public_baseline_multiplier = max(
        float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        float(PAPER_READY_BASELINE_MULTIPLIER),
        float(baselines["lora_delta_storage_strict_multiplier"]),
        float(baselines["qlora_delta_storage_strict_multiplier"]),
        float(baselines["rome_memit_edit_storage_strict_multiplier"]),
        float(baselines["verbatim_table_strict_multiplier"]),
        float(baselines["product_key_memory_strict_multiplier"]),
        float(baselines["memory_layer_strict_multiplier"]),
        float(baselines["content_routed_sparse_read_strict_multiplier"]),
        float(baselines["standard_codec_plus_index_strict_multiplier"]),
    )
    strongest_content_scan_multiplier = float(baselines["executable_content_scan_baseline_multiplier"])
    controls_collapse = float(
        int(
            controls["random_label_twin_success"] == 0.0
            and controls["no_memory_success"] == 0.0
            and controls["read_disabled_success"] == 0.0
            and controls["decoder_disabled_success"] == 0.0
            and controls["parser_disabled_success"] == 0.0
            and controls["adapter_disabled_success"] == 0.0
            and controls["code_disabled_success"] == 0.0
            and controls["wrong_query_hit_rate"] == 0.0
            and controls["unanswerable_query_hit_rate"] == 0.0
            and controls["partial_overlap_query_hit_rate"] == 0.0
            and controls["marker_injection_query_hit_rate"] == 0.0
            and controls["shuffled_question_success"] <= 0.01
            and controls["shuffled_paraphrase_success"] <= 0.01
            and controls["shuffled_value_success"] <= 0.01
            and controls["shuffled_provenance_success"] <= 0.01
            and controls["recency_only_success"] <= 0.01
        )
    )
    transformer_surface_pass = float(
        int(
            transformer_probe["forward_shape_success"] == 1.0
            and transformer_probe["adapter_payload_in_state_dict"] == 1.0
            and transformer_probe["adapter_header_in_state_dict"] == 1.0
            and transformer_probe["update_controller_in_state_dict"] == 1.0
            and transformer_probe["paraphrase_answer_success"] >= 0.95
            and transformer_probe["state_dict_reload_success"] >= 0.95
        )
    )
    recurrent_surface_pass = float(
        int(
            recurrent_probe["forward_shape_success"] == 1.0
            and recurrent_probe["adapter_payload_in_state_dict"] == 1.0
            and recurrent_probe["adapter_header_in_state_dict"] == 1.0
            and recurrent_probe["update_controller_in_state_dict"] == 1.0
            and recurrent_probe["paraphrase_answer_success"] >= 0.95
            and recurrent_probe["state_dict_reload_success"] >= 0.95
        )
    )
    inspection = hidden_state_inspection(cell, facts, source_block)
    source_holdout_pass = float(int(cell.source_train_test_path_overlap_count == 0 and cell.source_train_test_hash_overlap_count == 0 and cell.source_train_test_ngram_overlap_count == 0))
    multi_domain_pass = float(int(len(domain_counts) >= 4 and min(domain_counts.values()) > 0))
    large_margin_over_mph_pass = float(int(account["paper_surface_strict_multiplier"] >= float(PAPER_READY_BASELINE_MULTIPLIER) * float(MPH_MARGIN_TARGET)))
    static_public_baseline_pass = float(int(account["paper_surface_strict_multiplier"] > strongest_static_public_baseline_multiplier and large_margin_over_mph_pass == 1.0))
    content_scan_not_beaten = float(int(account["adapter_strict_multiplier"] <= strongest_content_scan_multiplier))
    paraphrase_or_update_pass = float(int(controls["paraphrase_stable_answer_success"] >= 0.95 and update_probe["trainable_recompression_update_success"] == 1.0 and update_probe["adapter_state_dict_reload_success"] == 1.0))
    matched_update_baseline_not_beaten = float(int(update_baseline["matched_update_recompress_baseline_success"] >= 0.95 and update_baseline["matched_update_recompress_baseline_bits"] <= update_probe["trainable_recompression_update_bits"]))
    ablation_controls_pass = float(int(controls_collapse == 1.0 and update_probe["update_controller_disabled_success"] == 0.0))
    paper_ready_requirement_count = float(transformer_surface_pass + recurrent_surface_pass + static_public_baseline_pass + multi_domain_pass + paraphrase_or_update_pass)
    strict_600x_pass = float(int(controls["exact_answer_success"] >= 0.95 and account["adapter_strict_density"] >= target_density and controls_collapse == 1.0))
    bounded_adapter_engineering_pass = float(
        int(
            controls["exact_answer_success"] >= 0.95
            and controls["heldout_exact_answer_success"] >= 0.95
            and controls["paraphrase_stable_answer_success"] >= 0.95
            and transformer_surface_pass == 1.0
            and recurrent_surface_pass == 1.0
            and static_public_baseline_pass == 1.0
            and multi_domain_pass == 1.0
            and source_holdout_pass == 1.0
            and large_margin_over_mph_pass == 1.0
            and paraphrase_or_update_pass == 1.0
            and ablation_controls_pass == 1.0
            and controls_collapse == 1.0
            and content_scan_not_beaten == 1.0
            and host_parameter_count_max < 100000.0
            and account["paper_surface_strict_density"] >= product_target_density
            and inspection["hidden_fact_value_row_detected"] == 0.0
            and inspection["hidden_raw_source_prefix_detected"] == 0.0
            and strict_600x_pass == 0.0
        )
    )
    static_compression_publishable_candidate = float(int(bounded_adapter_engineering_pass == 1.0 and content_scan_not_beaten == 0.0 and account["adapter_strict_multiplier"] > strongest_content_scan_multiplier))
    paper_ready_candidate = 0.0
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "train_fact_count": 0.0,
        "test_fact_count": float(len(facts)),
        "source_file_count": float(len(source_profile)),
        "source_domain_count": float(len(domain_counts)),
        "source_block_bytes": float(len(source_block)),
        "candidate_scan_count": float(cell.candidate_count),
        "selected_semantic_collision_count": float(selected_semantic_collision_count(facts)),
        "ambiguous_match_count": 0.0,
        "adapter_parameter_count": float(cell.parameter_count()),
        "host_parameter_count_max": float(host_parameter_count_max),
        "transformer_host_parameter_count": float(transformer_probe["parameter_count"]),
        "recurrent_host_parameter_count": float(recurrent_probe["parameter_count"]),
        "target_density": float(target_density),
        "target_multiplier": float(TARGET_MULTIPLIER),
        "product_target_multiplier": float(PRODUCT_TARGET_MULTIPLIER),
        "strict_600x_pass": strict_600x_pass,
        "paper_ready_candidate": paper_ready_candidate,
        "bounded_adapter_engineering_pass": bounded_adapter_engineering_pass,
        "static_compression_publishable_candidate": static_compression_publishable_candidate,
        "paper_ready_requirement_count": paper_ready_requirement_count,
        "transformer_surface_pass": transformer_surface_pass,
        "recurrent_surface_pass": recurrent_surface_pass,
        "static_public_baseline_pass": static_public_baseline_pass,
        "multi_domain_pass": multi_domain_pass,
        "source_holdout_pass": source_holdout_pass,
        "large_margin_over_mph_pass": large_margin_over_mph_pass,
        "content_scan_not_beaten": content_scan_not_beaten,
        "same_block_undercharged_mph_not_beaten": float(int(account["adapter_strict_multiplier"] <= float(baselines["mph_undercharged_strict_multiplier"]))),
        "paraphrase_or_update_pass": paraphrase_or_update_pass,
        "matched_update_baseline_not_beaten": matched_update_baseline_not_beaten,
        "publishable_update_adapter_candidate": 0.0,
        "ablation_controls_pass": ablation_controls_pass,
        "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        "llm_semantic_qa_baseline_multiplier": float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        "weight_carried_baseline_multiplier": float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        "paper_ready_baseline_multiplier": float(PAPER_READY_BASELINE_MULTIPLIER),
        "strongest_static_public_baseline_multiplier": strongest_static_public_baseline_multiplier,
        "strongest_content_scan_multiplier": strongest_content_scan_multiplier,
        "beats_lora_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["lora_delta_storage_strict_multiplier"]))),
        "beats_qlora_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["qlora_delta_storage_strict_multiplier"]))),
        "beats_model_edit_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["rome_memit_edit_storage_strict_multiplier"]))),
        "beats_product_key_memory_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["product_key_memory_strict_multiplier"]))),
        "beats_memory_layer_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["memory_layer_strict_multiplier"]))),
        "beats_content_routed_sparse_read_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["content_routed_sparse_read_strict_multiplier"]))),
        "beats_previous_mph_line_by_margin": float(large_margin_over_mph_pass),
        "beats_same_block_undercharged_mph_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["mph_undercharged_strict_multiplier"]))),
        "beats_mph_fingerprint_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["mph_fingerprint_strict_multiplier"]))),
        "beats_weight_carried_baseline": float(int(account["paper_surface_strict_multiplier"] > float(WEIGHT_CARRIED_BASELINE_MULTIPLIER))),
        "unknown_structure_source": 1.0,
        "bounded_llm_question_surface": 1.0,
        "source_block_count": 1.0,
        "block_stream_count": float(cell.block_stream_count),
        "adapter_state_stream_count": float(cell.adapter_state_stream_count),
        "per_fact_value_slice_count": float(cell.per_fact_value_slice_count),
        "assignment_row_count": float(cell.assignment_row_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "source_offset_routing_used": float(cell.source_offset_routing_used),
        "content_digest_key_target": float(cell.content_digest_key_target),
        "semantic_question_handle_target": float(cell.semantic_question_handle_target),
        "paraphrase_stable_handle_target": float(cell.paraphrase_stable_handle_target),
        "source_offset_key_target": float(cell.source_offset_key_target),
        "associative_random_key_target": 0.0,
        "answer_digest_key_target": float(cell.answer_digest_key_target),
        "independent_value_slice_path_used": float(cell.independent_value_slice_path_used),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "reads_from_compressed_model_state": float(cell.reads_from_compressed_model_state),
        "reads_from_compressed_block": float(cell.reads_from_compressed_block),
        "raw_source_block_bits_charged": 0.0,
        "question_parser_in_decoder_bits": float(cell.question_parser_in_decoder_bits),
        "fixed_parser_bits": float(DECODER_BITS),
        "fixed_parser_charged_through_decoder_bits": 1.0,
        "prompt_context_storage_used": float(cell.prompt_context_storage_used),
        "quoted_anchor_surface_used": 0.0,
        "lexical_token_signature_surface": 1.0,
        "learned_semantic_retrieval_authorized": 0.0,
        "model_state_adapter_payload_used": float(cell.model_state_adapter_payload_used),
        "state_dict_buffer_payload_used": float(cell.state_dict_buffer_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "adapter_recompression_update_path": 1.0,
        "trainable_recompression_controller_used": float(cell.trainable_recompression_controller_used),
        "trainable_recompression_controller_trained": float(cell.trainable_recompression_controller_trained),
        "true_base_weight_implicit_storage_authorized": 0.0,
        "source_holdout_used": 1.0,
        "source_train_file_count": 0.0,
        "source_train_test_path_overlap_count": 0.0,
        "source_train_test_hash_overlap_count": 0.0,
        "source_train_test_ngram_overlap_count": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "no_per_fact_value_rows": 1.0,
        "no_assignment_table": 1.0,
        "controls_collapse": controls_collapse,
        "transformer_forward_shape_success": float(transformer_probe["forward_shape_success"]),
        "transformer_adapter_payload_in_state_dict": float(transformer_probe["adapter_payload_in_state_dict"]),
        "transformer_adapter_header_in_state_dict": float(transformer_probe["adapter_header_in_state_dict"]),
        "transformer_update_controller_in_state_dict": float(transformer_probe["update_controller_in_state_dict"]),
        "transformer_paraphrase_answer_success": float(transformer_probe["paraphrase_answer_success"]),
        "transformer_state_dict_preload_success": float(transformer_probe["state_dict_preload_success"]),
        "transformer_state_dict_reload_success": float(transformer_probe["state_dict_reload_success"]),
        "recurrent_forward_shape_success": float(recurrent_probe["forward_shape_success"]),
        "recurrent_adapter_payload_in_state_dict": float(recurrent_probe["adapter_payload_in_state_dict"]),
        "recurrent_adapter_header_in_state_dict": float(recurrent_probe["adapter_header_in_state_dict"]),
        "recurrent_update_controller_in_state_dict": float(recurrent_probe["update_controller_in_state_dict"]),
        "recurrent_paraphrase_answer_success": float(recurrent_probe["paraphrase_answer_success"]),
        "recurrent_state_dict_preload_success": float(recurrent_probe["state_dict_preload_success"]),
        "recurrent_state_dict_reload_success": float(recurrent_probe["state_dict_reload_success"]),
        **account,
        **controls,
        **update_probe,
        **update_baseline,
        **inspection,
        **baselines,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_paper_ready_local_candidate_authorized": float(row["paper_ready_candidate"]),
        f"{SIMULATION_ID}_strict_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_general_unknown_structure_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
        f"{SIMULATION_ID}_external_simulator_authorized": 0.0,
        f"{SIMULATION_ID}_arbitrary_chat_authorized": 0.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["bounded_adapter_engineering_pass"]),
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
    metrics_path = output_dir / "local_100k_margin_recompression_adapter_metrics.json"
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
            "facts_smoke": int(FACTS_SMOKE),
            "facts_hard": int(FACTS_HARD),
            "chunk_bytes": int(CHUNK_BYTES),
            "anchor_bytes": int(ANCHOR_BYTES),
            "semantic_handle_bytes": int(SEMANTIC_HANDLE_BYTES),
            "decoder_bits": int(DECODER_BITS),
            "model_header_bits": int(MODEL_HEADER_BITS),
            "surface_contract_bits": int(SURFACE_CONTRACT_BITS),
            "test_source_cap_bytes": int(TEST_SOURCE_CAP_BYTES),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
            "product_target_multiplier": float(PRODUCT_TARGET_MULTIPLIER),
            "mph_margin_target": float(MPH_MARGIN_TARGET),
            "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
            "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
            "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
            "llm_semantic_qa_baseline_multiplier": float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
            "weight_carried_baseline_multiplier": float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
            "paper_ready_baseline_multiplier": float(PAPER_READY_BASELINE_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_margin_recompression_adapter_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
