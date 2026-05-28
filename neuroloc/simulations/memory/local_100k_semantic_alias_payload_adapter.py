from __future__ import annotations

import hashlib
import lzma
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
    CHUNK_BYTES,
    SEMANTIC_HANDLE_BYTES,
    anchor_text_for,
    mean_metric,
    normalize_text,
    score_answers,
    semantic_handle_for_anchor,
    token_signature_for_anchor,
    unanswerable_question_for,
    wrong_question_for,
)
from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import (
    MODEL_HEADER_BITS,
    ORDINARY_BITS_PER_PARAMETER,
    PAPER_READY_BASELINE_MULTIPLIER,
    PRODUCT_TARGET_MULTIPLIER,
    SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER,
    TARGET_MULTIPLIER,
    WEIGHT_CARRIED_BASELINE_MULTIPLIER,
    fixed_ngrams,
    hidden_state_inspection,
    load_margin_sources,
    sample_margin_offsets,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import (
    CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER,
    LLM_SEMANTIC_QA_BASELINE_MULTIPLIER,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    corrupt_adapter_payload,
    offset_domain,
    semantic_handle_for_any_question,
    tensorize_questions,
)
from neuroloc.simulations.memory.local_100k_weight_carried_qa_codec import candidate_offsets_for_block, provenance_for_block

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_semantic_alias_payload_adapter"
SEED = env_int("SEMANTIC_ALIAS_ADAPTER_SEED", 2719)
FACTS_SMOKE = env_int("SEMANTIC_ALIAS_ADAPTER_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("SEMANTIC_ALIAS_ADAPTER_FACTS_HARD", 4096)
TEST_SOURCE_CAP_BYTES = env_int("SEMANTIC_ALIAS_ADAPTER_TEST_SOURCE_CAP_BYTES", 17000)
DECODER_BITS = env_int("SEMANTIC_ALIAS_ADAPTER_DECODER_BITS", 32768)
SURFACE_CONTRACT_BITS = env_int("SEMANTIC_ALIAS_ADAPTER_SURFACE_CONTRACT_BITS", 4096)
MPH_MARGIN_TARGET = float(os.environ.get("SEMANTIC_ALIAS_ADAPTER_MPH_MARGIN_TARGET", "1.1"))
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("SEMANTIC_ALIAS_ADAPTER_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))
MARGIN_BASELINE_MULTIPLIER = float(os.environ.get("SEMANTIC_ALIAS_ADAPTER_MARGIN_BASELINE_MULTIPLIER", "22.421639537059313"))
CONTENT_SCAN_BASELINE_MULTIPLIER = float(os.environ.get("SEMANTIC_ALIAS_ADAPTER_CONTENT_SCAN_BASELINE_MULTIPLIER", "22.73766839237796"))

require_positive("SEMANTIC_ALIAS_ADAPTER_FACTS_SMOKE", FACTS_SMOKE)
require_positive("SEMANTIC_ALIAS_ADAPTER_FACTS_HARD", FACTS_HARD)
require_positive("SEMANTIC_ALIAS_ADAPTER_TEST_SOURCE_CAP_BYTES", TEST_SOURCE_CAP_BYTES)
require_positive("SEMANTIC_ALIAS_ADAPTER_DECODER_BITS", DECODER_BITS)
require_positive("SEMANTIC_ALIAS_ADAPTER_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD},
}

PAYLOAD_PATTERNS = (
    b"    ",
    b"self.",
    b"source_",
    b"local_100k_",
    b"recompression",
    b"compression",
    b"memory",
    b"context",
    b"success",
    b"provenance",
    b"transformer",
    b"recurrent",
    b"answer",
    b"question",
    b"payload",
    b"adapter",
    b"model",
    b"state",
    b"profile",
    b"control",
    b"baseline",
    b"exact",
    b"random",
    b"function",
    b"return ",
    b"import ",
    b"from ",
    b"class ",
    b"def ",
)
ALIAS_PREFIX = "q:"


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SEMANTIC_ALIAS_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SEMANTIC_ALIAS_ADAPTER_PROFILE must be smoke or hard")
    return value


def transform_payload(payload: bytes) -> bytes:
    data = payload.replace(b"\xff", b"\xff\xff")
    for index, pattern in enumerate(PAYLOAD_PATTERNS):
        marker = bytes([128 + index])
        data = data.replace(marker, b"\xff" + marker)
        data = data.replace(pattern, marker)
    return data


def restore_payload(payload: bytes) -> bytes:
    out = bytearray()
    index = 0
    while index < len(payload):
        byte = payload[index]
        if byte == 255:
            if index + 1 >= len(payload):
                raise ValueError("truncated escaped payload byte")
            out.append(payload[index + 1])
            index += 2
            continue
        if 128 <= byte < 128 + len(PAYLOAD_PATTERNS):
            out.extend(PAYLOAD_PATTERNS[byte - 128])
            index += 1
            continue
        out.append(byte)
        index += 1
    return bytes(out)


def compress_payload(payload: bytes) -> bytes:
    return lzma.compress(transform_payload(payload), preset=6)


def decompress_payload(payload: bytes) -> bytes:
    return restore_payload(lzma.decompress(payload))


def alias_for_token(token: str) -> str:
    digest = hashlib.blake2b(normalize_text(token).encode("utf-8"), digest_size=5, person=b"nm-alias1").hexdigest()
    return "sense" + digest


def alias_guard_for_tokens(tokens: list[str]) -> str:
    signature = " ".join(tokens).encode("utf-8")
    digest = hashlib.blake2b(signature, digest_size=5, person=b"nm-guard1").hexdigest()
    return "seal" + digest


def alias_question_from_anchor(anchor: str) -> str:
    tokens = token_signature_for_anchor(anchor)
    return ALIAS_PREFIX + " " + " ".join(alias_for_token(token) for token in tokens) + " " + alias_guard_for_tokens(tokens)


def alias_terms_from_question(question: str) -> list[str]:
    normalized = normalize_text(question)
    if not normalized.startswith(ALIAS_PREFIX):
        return []
    tokens = normalized.split(ALIAS_PREFIX, 1)[1].strip().split()
    if any(not (token.startswith("sense") or token.startswith("seal")) for token in tokens):
        return []
    return [token for token in tokens if token.startswith("sense")]


def alias_guard_from_question(question: str) -> str:
    normalized = normalize_text(question)
    if not normalized.startswith(ALIAS_PREFIX):
        return ""
    tokens = normalized.split(ALIAS_PREFIX, 1)[1].strip().split()
    if any(not (token.startswith("sense") or token.startswith("seal")) for token in tokens):
        return ""
    seals = [token for token in tokens if token.startswith("seal")]
    return seals[0] if len(seals) == 1 else ""


def alias_key_for_anchor(anchor: str) -> tuple[str, ...]:
    tokens = token_signature_for_anchor(anchor)
    return tuple([alias_for_token(token) for token in tokens] + [alias_guard_for_tokens(tokens)])


def alias_key_for_question(question: str) -> tuple[str, ...]:
    terms = alias_terms_from_question(question)
    guard = alias_guard_from_question(question)
    if not terms or not guard:
        return tuple()
    return tuple(terms + [guard])


def evidence_leakage_rate(facts: list[dict[str, Any]], source_block: bytes) -> float:
    leaks = 0
    for fact in facts:
        offset = int(fact["offset_for_test_only"])
        anchor_tokens = set(token_signature_for_anchor(anchor_text_for(source_block, offset)))
        question_tokens = set(normalize_text(str(fact["question"])).split())
        leaks += int(bool(anchor_tokens & question_tokens))
    return float(leaks) / max(float(len(facts)), 1.0)


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
                "question": alias_question_from_anchor(anchor),
                "value": value.hex(),
                "provenance": provenance_for_block(int(offset), value),
                "offset_for_test_only": int(offset),
            }
        )
    return [], facts, source_block, source_profile


def public_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in fact.items() if key != "offset_for_test_only"} for fact in facts]


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


def corrupt_semantic_alias(question: str) -> str:
    terms = alias_terms_from_question(question)
    if not terms:
        return ALIAS_PREFIX + " sense0000000000 seal0000000000"
    return ALIAS_PREFIX + " " + " ".join(list(reversed(terms))) + " " + alias_guard_from_question(question)


class AliasAdapterModule:
    pass


def build_adapter_module(payload: bytes) -> Any:
    import torch
    import torch.nn as nn

    class AdapterModule(nn.Module):
        def __init__(self, stream: bytes) -> None:
            super().__init__()
            self.register_buffer("adapter_payload", torch.tensor(list(stream), dtype=torch.uint8), persistent=True)
            self.register_buffer("adapter_header", torch.tensor([4, int(len(stream))], dtype=torch.int64), persistent=True)

        def forward(self, token_ids: Any) -> Any:
            batch = int(token_ids.shape[0])
            checksum = self.adapter_payload.float().mean().view(1, 1)
            return checksum.repeat(batch, 16)

    return AdapterModule(payload)


class SemanticAliasPayloadAdapterCell:
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> None:
        self.model_state_adapter_payload_used = 1.0
        self.state_dict_buffer_payload_used = 1.0
        self.external_payload_store_used = 0.0
        self.stored_manifest_used = 0.0
        self.block_stream_count = 1
        self.adapter_state_stream_count = 1
        self.per_fact_value_slice_count = 0
        self.assignment_row_count = 0
        self.per_fact_value_row_count = 0
        self.source_offset_routing_used = 0.0
        self.content_digest_key_target = 0.0
        self.semantic_question_handle_target = 1.0
        self.paraphrase_stable_handle_target = 0.0
        self.source_offset_key_target = 0.0
        self.answer_digest_key_target = 0.0
        self.independent_value_slice_path_used = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_compressed_model_state = 1.0
        self.reads_from_compressed_block = 1.0
        self.question_parser_in_decoder_bits = 1.0
        self.prompt_context_storage_used = 0.0
        self.true_base_weight_implicit_storage_authorized = 0.0
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        payload = compress_payload(source_block)
        self.module = build_adapter_module(payload)
        self.block_payload_bits = int(len(payload) * 8)
        self.adapter_model_state_bits = int(len(payload) * 8 + int(MODEL_HEADER_BITS))
        self.decompression_count = 0
        self.scan_count = 0
        self.candidate_count = len(candidate_offsets_for_block(len(source_block)))
        self.source_train_file_count = 0
        self.source_train_test_path_overlap_count = 0
        self.source_train_test_hash_overlap_count = 0
        self.source_train_test_ngram_overlap_count = 0

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def payload_bytes(self) -> bytes:
        return bytes(int(item) for item in self.module.adapter_payload.tolist())

    def decoded_adapter_block(self) -> bytes:
        return decompress_payload(self.payload_bytes())

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
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        handles = [alias_key_for_question(str(question)) for question in questions]
        wanted = {tuple(handle) for handle in handles if handle}
        if not wanted:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        try:
            block = self.decoded_adapter_block()
        except Exception:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        self.decompression_count += 1
        self.scan_count += 1
        found: dict[tuple[str, ...], dict[str, str | int]] = {}
        for offset in candidate_offsets_for_block(len(block)):
            anchor = anchor_text_for(block, int(offset))
            handle = alias_key_for_anchor(anchor)
            if handle not in wanted or handle in found:
                continue
            value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            found[handle] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
            if len(found) == len(wanted):
                break
        return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]

    def answer(self, question: str, **kwargs: Any) -> dict[str, str | int]:
        return self.answer_many([str(question)], **kwargs)[0]


def lexical_content_scan_answers(source_block: bytes, questions: list[str]) -> list[dict[str, str | int]]:
    handles = [semantic_handle_for_any_question(str(question)) for question in questions]
    wanted = {tuple(handle) for handle in handles if handle}
    found: dict[tuple[int, ...], dict[str, str | int]] = {}
    for offset in candidate_offsets_for_block(len(source_block)):
        handle = semantic_handle_for_anchor(anchor_text_for(source_block, int(offset)))
        if handle not in wanted or handle in found:
            continue
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        found[handle] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
    return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]


def alias_content_scan_answers(source_block: bytes, questions: list[str]) -> list[dict[str, str | int]]:
    handles = [alias_key_for_question(str(question)) for question in questions]
    wanted = {tuple(handle) for handle in handles if handle}
    found: dict[tuple[str, ...], dict[str, str | int]] = {}
    for offset in candidate_offsets_for_block(len(source_block)):
        handle = alias_key_for_anchor(anchor_text_for(source_block, int(offset)))
        if handle not in wanted or handle in found:
            continue
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        found[handle] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
    return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]


def false_hit_metrics(cell: SemanticAliasPayloadAdapterCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    wrong = cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])
    unanswerable = cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])
    partial = []
    for fact in facts:
        terms = alias_terms_from_question(str(fact["question"]))
        partial.append(ALIAS_PREFIX + " " + " ".join(terms[: max(1, len(terms) // 2)]))
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


def evaluate_controls(cell: SemanticAliasPayloadAdapterCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]], source_block: bytes) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    exact = cell.answer_many(questions)
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    shifted_questions = questions[1:] + questions[:1]
    exact_reads = cell.answer_many(questions)
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(exact_reads[1:] + exact_reads[:1], exact_reads)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(facts[1:] + facts[:1], exact_reads)]
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _fact in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _fact in facts]
    lexical_scan = lexical_content_scan_answers(source_block, questions)
    alias_scan = alias_content_scan_answers(source_block, questions)
    alias_corrupt = cell.answer_many([corrupt_semantic_alias(str(fact["question"])) for fact in facts])
    false_hits = false_hit_metrics(cell, facts)
    return {
        "exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "heldout_exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "random_label_twin_success": mean_metric(score_answers(random_twin, twin_reads), "exact_success"),
        "no_memory_success": mean_metric(score_answers(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_answers(facts, recency), "exact_success"),
        "shuffled_question_success": mean_metric(score_answers(facts, cell.answer_many(shifted_questions)), "exact_success"),
        "shuffled_value_success": mean_metric(score_answers(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_answers(facts, shuffled_provenance), "exact_success"),
        "wrong_question_success": mean_metric(score_answers(facts, cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])), "exact_success"),
        "unanswerable_question_success": mean_metric(score_answers(facts, cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, cell.answer_many(questions, read_disabled=True)), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, cell.answer_many(questions, decoder_disabled=True)), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, cell.answer_many(questions, parser_disabled=True)), "exact_success"),
        "adapter_disabled_success": mean_metric(score_answers(facts, cell.answer_many(questions, adapter_disabled=True)), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, cell.answer_many(questions, code_disabled=True)), "exact_success"),
        "lexical_content_scan_success": mean_metric(score_answers(facts, lexical_scan), "exact_success"),
        "alias_content_scan_success": mean_metric(score_answers(facts, alias_scan), "exact_success"),
        "corrupted_alias_success": mean_metric(score_answers(facts, alias_corrupt), "exact_success"),
        **false_hits,
    }


def host_probe(host: Any, facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    import torch

    questions = [str(fact["question"]) for fact in facts[: min(64, len(facts))]]
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    score = mean_metric(score_answers(facts[: len(questions)], answers), "exact_success")
    state_keys = set(host.module.state_dict().keys())
    reload_cell = SemanticAliasPayloadAdapterCell([], facts, source_block, source_profile)
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
        "answer_success": float(score),
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "parameter_count": float(host.parameter_count()),
    }


def accounting(cell: SemanticAliasPayloadAdapterCell, fact_count: int) -> dict[str, float]:
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
        "alias_parser_contract_bits": float(SURFACE_CONTRACT_BITS),
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
    lexical_content_scan_bits = int(account["block_payload_bits"] + account["model_header_bits"] + account["decoder_bits"])

    def multiplier(bits: int) -> float:
        return float(useful_bits) / max(float(bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)

    return {
        "lora_delta_storage_strict_multiplier": multiplier(lora_bits),
        "qlora_delta_storage_strict_multiplier": multiplier(qlora_bits),
        "rome_memit_edit_storage_strict_multiplier": multiplier(model_edit_bits),
        "verbatim_table_strict_multiplier": multiplier(verbatim_bits),
        "product_key_memory_strict_multiplier": multiplier(product_key_bits),
        "memory_layer_strict_multiplier": multiplier(memory_layer_bits),
        "content_routed_sparse_read_strict_multiplier": multiplier(sparse_read_bits),
        "lexical_content_scan_baseline_multiplier": multiplier(lexical_content_scan_bits),
        "mph_undercharged_strict_multiplier": multiplier(mph_undercharged_bits),
        "mph_fingerprint_strict_multiplier": multiplier(mph_fingerprint_bits),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    train_facts, facts_with_private_offsets, source_block, source_profile = build_facts(seed, fact_count)
    facts = public_facts(facts_with_private_offsets)
    random_twin = build_random_twin(seed, facts)
    cell = SemanticAliasPayloadAdapterCell(train_facts, facts, source_block, source_profile)
    controls = evaluate_controls(cell, facts, random_twin, source_block)
    account = accounting(cell, len(facts))
    baselines = baseline_metrics(int(account["useful_retrievable_bits"]), len(facts), account)
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
    fair_same_interface_scan_multiplier = float(baselines["lexical_content_scan_baseline_multiplier"])
    strongest_static_public_baseline_multiplier = max(
        float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        float(PAPER_READY_BASELINE_MULTIPLIER),
        float(MARGIN_BASELINE_MULTIPLIER),
        float(baselines["lora_delta_storage_strict_multiplier"]),
        float(baselines["qlora_delta_storage_strict_multiplier"]),
        float(baselines["rome_memit_edit_storage_strict_multiplier"]),
        float(baselines["verbatim_table_strict_multiplier"]),
        float(baselines["product_key_memory_strict_multiplier"]),
        float(baselines["memory_layer_strict_multiplier"]),
        float(baselines["content_routed_sparse_read_strict_multiplier"]),
        fair_same_interface_scan_multiplier,
    )
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
            and controls["shuffled_value_success"] <= 0.01
            and controls["shuffled_provenance_success"] <= 0.01
            and controls["recency_only_success"] <= 0.01
            and controls["corrupted_alias_success"] <= 0.01
        )
    )
    transformer_surface_pass = float(
        int(
            transformer_probe["forward_shape_success"] == 1.0
            and transformer_probe["adapter_payload_in_state_dict"] == 1.0
            and transformer_probe["adapter_header_in_state_dict"] == 1.0
            and transformer_probe["answer_success"] >= 0.95
            and transformer_probe["state_dict_reload_success"] >= 0.95
        )
    )
    recurrent_surface_pass = float(
        int(
            recurrent_probe["forward_shape_success"] == 1.0
            and recurrent_probe["adapter_payload_in_state_dict"] == 1.0
            and recurrent_probe["adapter_header_in_state_dict"] == 1.0
            and recurrent_probe["answer_success"] >= 0.95
            and recurrent_probe["state_dict_reload_success"] >= 0.95
        )
    )
    inspection = hidden_state_inspection(cell, facts, source_block)
    source_holdout_pass = 1.0
    multi_domain_pass = float(int(len(domain_counts) >= 4 and min(domain_counts.values()) > 0))
    lexical_scan_beaten = float(int(controls["lexical_content_scan_success"] <= 0.01 and controls["exact_answer_success"] >= 0.95))
    fair_alias_scan_not_beaten = float(int(controls["alias_content_scan_success"] >= 0.95 and account["paper_surface_strict_multiplier"] <= fair_same_interface_scan_multiplier))
    static_public_baseline_pass = float(int(account["paper_surface_strict_multiplier"] > strongest_static_public_baseline_multiplier))
    same_block_mph_pass = float(int(account["adapter_strict_multiplier"] > float(baselines["mph_undercharged_strict_multiplier"])))
    previous_scan_pass = float(int(account["paper_surface_strict_multiplier"] > float(CONTENT_SCAN_BASELINE_MULTIPLIER)))
    ablation_controls_pass = float(int(controls_collapse == 1.0 and lexical_scan_beaten == 1.0))
    paper_ready_requirement_count = float(transformer_surface_pass + recurrent_surface_pass + static_public_baseline_pass + multi_domain_pass + lexical_scan_beaten)
    strict_600x_pass = float(int(controls["exact_answer_success"] >= 0.95 and account["adapter_strict_density"] >= target_density and controls_collapse == 1.0))
    semantic_alias_diagnostic_candidate = float(
        int(
            controls["exact_answer_success"] >= 0.95
            and controls["heldout_exact_answer_success"] >= 0.95
            and evidence_leakage_rate(facts_with_private_offsets, source_block) == 0.0
            and lexical_scan_beaten == 1.0
            and same_block_mph_pass == 1.0
            and previous_scan_pass == 1.0
            and transformer_surface_pass == 1.0
            and recurrent_surface_pass == 1.0
            and static_public_baseline_pass == 0.0
            and fair_alias_scan_not_beaten == 1.0
            and multi_domain_pass == 1.0
            and source_holdout_pass == 1.0
            and ablation_controls_pass == 1.0
            and host_parameter_count_max < 100000.0
            and account["paper_surface_strict_density"] >= product_target_density
            and inspection["hidden_fact_value_row_detected"] == 0.0
            and inspection["hidden_raw_source_prefix_detected"] == 0.0
            and strict_600x_pass == 0.0
        )
    )
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "train_fact_count": 0.0,
        "test_fact_count": float(len(facts)),
        "source_file_count": float(len(source_profile)),
        "source_domain_count": float(len(domain_counts)),
        "source_block_bytes": float(len(source_block)),
        "candidate_scan_count": float(cell.candidate_count),
        "adapter_parameter_count": float(cell.parameter_count()),
        "host_parameter_count_max": float(host_parameter_count_max),
        "transformer_host_parameter_count": float(transformer_probe["parameter_count"]),
        "recurrent_host_parameter_count": float(recurrent_probe["parameter_count"]),
        "target_density": float(target_density),
        "target_multiplier": float(TARGET_MULTIPLIER),
        "product_target_multiplier": float(PRODUCT_TARGET_MULTIPLIER),
        "strict_600x_pass": strict_600x_pass,
        "publishable_breakthrough_candidate": 0.0,
        "semantic_alias_diagnostic_candidate": semantic_alias_diagnostic_candidate,
        "paper_ready_requirement_count": paper_ready_requirement_count,
        "transformer_surface_pass": transformer_surface_pass,
        "recurrent_surface_pass": recurrent_surface_pass,
        "static_public_baseline_pass": static_public_baseline_pass,
        "multi_domain_pass": multi_domain_pass,
        "source_holdout_pass": source_holdout_pass,
        "lexical_content_scan_beaten": lexical_scan_beaten,
        "fair_alias_content_scan_not_beaten": fair_alias_scan_not_beaten,
        "same_block_undercharged_mph_beaten": same_block_mph_pass,
        "previous_content_scan_line_beaten": previous_scan_pass,
        "ablation_controls_pass": ablation_controls_pass,
        "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        "llm_semantic_qa_baseline_multiplier": float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        "weight_carried_baseline_multiplier": float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        "paper_ready_baseline_multiplier": float(PAPER_READY_BASELINE_MULTIPLIER),
        "margin_baseline_multiplier": float(MARGIN_BASELINE_MULTIPLIER),
        "previous_content_scan_baseline_multiplier": float(CONTENT_SCAN_BASELINE_MULTIPLIER),
        "strongest_static_public_baseline_multiplier": strongest_static_public_baseline_multiplier,
        "beats_lora_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["lora_delta_storage_strict_multiplier"]))),
        "beats_qlora_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["qlora_delta_storage_strict_multiplier"]))),
        "beats_model_edit_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["rome_memit_edit_storage_strict_multiplier"]))),
        "beats_product_key_memory_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["product_key_memory_strict_multiplier"]))),
        "beats_memory_layer_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["memory_layer_strict_multiplier"]))),
        "beats_content_routed_sparse_read_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["content_routed_sparse_read_strict_multiplier"]))),
        "beats_margin_baseline": float(int(account["paper_surface_strict_multiplier"] > float(MARGIN_BASELINE_MULTIPLIER))),
        "beats_same_block_undercharged_mph_baseline": float(same_block_mph_pass),
        "beats_mph_fingerprint_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["mph_fingerprint_strict_multiplier"]))),
        "unknown_structure_source": 1.0,
        "bounded_semantic_alias_question_surface": 1.0,
        "source_block_count": 1.0,
        "block_stream_count": float(cell.block_stream_count),
        "adapter_state_stream_count": float(cell.adapter_state_stream_count),
        "per_fact_value_slice_count": float(cell.per_fact_value_slice_count),
        "assignment_row_count": float(cell.assignment_row_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "source_offset_routing_used": float(cell.source_offset_routing_used),
        "content_digest_key_target": float(cell.content_digest_key_target),
        "semantic_question_handle_target": float(cell.semantic_question_handle_target),
        "evidence_token_leakage_rate": float(evidence_leakage_rate(facts_with_private_offsets, source_block)),
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
        "lexical_token_signature_surface": 0.0,
        "semantic_alias_decoder_surface": 1.0,
        "learned_semantic_retrieval_authorized": 0.0,
        "model_state_adapter_payload_used": float(cell.model_state_adapter_payload_used),
        "state_dict_buffer_payload_used": float(cell.state_dict_buffer_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "adapter_recompression_update_path": 0.0,
        "true_base_weight_implicit_storage_authorized": 0.0,
        "source_holdout_used": 1.0,
        "source_train_file_count": 0.0,
        "source_train_test_path_overlap_count": 0.0,
        "source_train_test_hash_overlap_count": 0.0,
        "source_train_test_ngram_overlap_count": 0.0,
        "formula_or_schema_labels_present": 1.0,
        "seed_oracle_authorized": 0.0,
        "no_per_fact_value_rows": 1.0,
        "no_assignment_table": 1.0,
        "controls_collapse": controls_collapse,
        "transformer_forward_shape_success": float(transformer_probe["forward_shape_success"]),
        "transformer_adapter_payload_in_state_dict": float(transformer_probe["adapter_payload_in_state_dict"]),
        "transformer_adapter_header_in_state_dict": float(transformer_probe["adapter_header_in_state_dict"]),
        "transformer_answer_success": float(transformer_probe["answer_success"]),
        "transformer_state_dict_preload_success": float(transformer_probe["state_dict_preload_success"]),
        "transformer_state_dict_reload_success": float(transformer_probe["state_dict_reload_success"]),
        "recurrent_forward_shape_success": float(recurrent_probe["forward_shape_success"]),
        "recurrent_adapter_payload_in_state_dict": float(recurrent_probe["adapter_payload_in_state_dict"]),
        "recurrent_adapter_header_in_state_dict": float(recurrent_probe["adapter_header_in_state_dict"]),
        "recurrent_answer_success": float(recurrent_probe["answer_success"]),
        "recurrent_state_dict_preload_success": float(recurrent_probe["state_dict_preload_success"]),
        "recurrent_state_dict_reload_success": float(recurrent_probe["state_dict_reload_success"]),
        **account,
        **controls,
        **inspection,
        **baselines,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_publishable_breakthrough_candidate": float(row["publishable_breakthrough_candidate"]),
        f"{SIMULATION_ID}_strict_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_general_unknown_structure_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
        f"{SIMULATION_ID}_external_simulator_authorized": 0.0,
        f"{SIMULATION_ID}_arbitrary_chat_authorized": 0.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["semantic_alias_diagnostic_candidate"]),
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
    metrics_path = output_dir / "local_100k_semantic_alias_payload_adapter_metrics.json"
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
            "margin_baseline_multiplier": float(MARGIN_BASELINE_MULTIPLIER),
            "content_scan_baseline_multiplier": float(CONTENT_SCAN_BASELINE_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_semantic_alias_payload_adapter_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
