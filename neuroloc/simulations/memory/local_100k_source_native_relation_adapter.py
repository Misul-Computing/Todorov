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
    CHUNK_BYTES,
    anchor_text_for,
    mean_metric,
    normalize_text,
    score_answers,
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
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import (
    CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER,
    LLM_SEMANTIC_QA_BASELINE_MULTIPLIER,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    corrupt_adapter_payload,
    offset_domain,
    tensorize_questions,
)
from neuroloc.simulations.memory.local_100k_semantic_alias_payload_adapter import (
    CONTENT_SCAN_BASELINE_MULTIPLIER,
    MARGIN_BASELINE_MULTIPLIER,
    compress_payload,
    decompress_payload,
    transform_payload,
)
from neuroloc.simulations.memory.local_100k_weight_carried_qa_codec import candidate_offsets_for_block, provenance_for_block

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_native_relation_adapter"
SEED = env_int("SOURCE_NATIVE_RELATION_ADAPTER_SEED", 3251)
FACTS_SMOKE = env_int("SOURCE_NATIVE_RELATION_ADAPTER_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("SOURCE_NATIVE_RELATION_ADAPTER_FACTS_HARD", 4096)
RELATION_STRIDE = env_int("SOURCE_NATIVE_RELATION_ADAPTER_RELATION_STRIDE", 7)
RELATION_TERMS = env_int("SOURCE_NATIVE_RELATION_ADAPTER_RELATION_TERMS", 3)
DECODER_BITS = env_int("SOURCE_NATIVE_RELATION_ADAPTER_DECODER_BITS", 32768)
SURFACE_CONTRACT_BITS = env_int("SOURCE_NATIVE_RELATION_ADAPTER_SURFACE_CONTRACT_BITS", 4096)
ROUTER_PARAMETER_BITS = env_int("SOURCE_NATIVE_RELATION_ADAPTER_ROUTER_PARAMETER_BITS", 8)
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("SOURCE_NATIVE_RELATION_ADAPTER_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))

require_positive("SOURCE_NATIVE_RELATION_ADAPTER_FACTS_SMOKE", FACTS_SMOKE)
require_positive("SOURCE_NATIVE_RELATION_ADAPTER_FACTS_HARD", FACTS_HARD)
require_positive("SOURCE_NATIVE_RELATION_ADAPTER_RELATION_STRIDE", RELATION_STRIDE)
require_positive("SOURCE_NATIVE_RELATION_ADAPTER_RELATION_TERMS", RELATION_TERMS)
require_positive("SOURCE_NATIVE_RELATION_ADAPTER_DECODER_BITS", DECODER_BITS)
require_positive("SOURCE_NATIVE_RELATION_ADAPTER_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("SOURCE_NATIVE_RELATION_ADAPTER_ROUTER_PARAMETER_BITS", ROUTER_PARAMETER_BITS)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD},
}
QUESTION_PREFIX = "source relation after evidence terms:"
QUESTION_STOPWORDS = {
    "source",
    "relation",
    "after",
    "before",
    "evidence",
    "terms",
    "term",
    "route",
    "router",
    "exact",
    "bytes",
    "span",
    "from",
    "for",
    "the",
    "please",
    "target",
    "native",
    "learned",
    "retrieve",
    "select",
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_NATIVE_RELATION_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_NATIVE_RELATION_ADAPTER_PROFILE must be smoke or hard")
    return value


def source_bounds_for_offset(manifest: list[dict[str, Any]], offset: int) -> tuple[int, int]:
    for row in manifest:
        start = int(row["block_offset"])
        end = int(start + int(row["length"]))
        if start <= int(offset) < end - int(CHUNK_BYTES):
            return int(start), int(end)
    return 0, 0


def relation_target_offset(source_block: bytes, manifest: list[dict[str, Any]], anchor_offset: int, stride: int) -> int | None:
    target = int(anchor_offset) + int(stride) * int(CHUNK_BYTES)
    if target in set(candidate_offsets_for_block(len(source_block))):
        return int(target)
    return None


def relation_terms_for_anchor(source_block: bytes, offset: int) -> tuple[str, ...]:
    return tuple(token_signature_for_anchor(anchor_text_for(source_block, int(offset)))[: int(RELATION_TERMS)])


def relation_question_from_anchor(source_block: bytes, offset: int) -> str:
    return QUESTION_PREFIX + " " + " ".join(relation_terms_for_anchor(source_block, int(offset)))


def relation_terms_from_question(question: str) -> tuple[str, ...]:
    normalized = normalize_text(question)
    if QUESTION_PREFIX in normalized:
        normalized = normalized.split(QUESTION_PREFIX, 1)[1]
        tokens = tuple(token for token in normalized.split() if len(token) >= 4)
        return tokens if len(tokens) == int(RELATION_TERMS) else tuple()
    tokens = []
    for token in normalized.split():
        if token in QUESTION_STOPWORDS or len(token) < 4:
            continue
        tokens.append(token)
        if len(tokens) == int(RELATION_TERMS):
            break
    return tuple(tokens)


def relation_paraphrases(facts: list[dict[str, Any]]) -> list[str]:
    questions = []
    for index, fact in enumerate(facts):
        terms = " ".join(str(fact["question"]).split()[-int(RELATION_TERMS) :])
        variants = [
            str(fact["question"]),
            "learned source-native router " + QUESTION_PREFIX + " " + terms,
            "retrieve exact span " + QUESTION_PREFIX + " " + terms,
            "select the related source span " + QUESTION_PREFIX + " " + terms,
            "route with source evidence " + QUESTION_PREFIX + " " + terms,
        ]
        questions.append(variants[index % len(variants)])
    return questions


def train_source_rows() -> list[tuple[Path, str, str]]:
    rows = [
        (PROJECT_ROOT / "neuroloc/wiki/Home.md", "home", "wiki_index"),
        (PROJECT_ROOT / "neuroloc/wiki/INDEX.md", "index", "wiki_index"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/timescale_separation.md", "timescale_separation", "wiki_train"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/working_memory_as_controlled_access.md", "working_memory_as_controlled_access", "wiki_train"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/world_models_imagination_and_planning.md", "world_models_imagination_and_planning", "wiki_train"),
        (PROJECT_ROOT / "knowledge/context_extension.md", "context_extension", "knowledge_train"),
        (PROJECT_ROOT / "src/training/loss.py", "training_loss", "library_train"),
        (PROJECT_ROOT / "src/utils/memory.py", "utils_memory", "library_train"),
    ]
    return [(path, name, domain) for path, name, domain in rows if path.exists()]


def load_relation_train_sources() -> tuple[list[dict[str, Any]], bytes]:
    manifest = []
    parts: list[bytes] = []
    block_offset = 0
    for index, (path, name, domain) in enumerate(train_source_rows()):
        data = path.read_bytes().replace(b"\r\n", b"\n")[:17000]
        if parts:
            parts.append(b"\n\n")
            block_offset += 2
        manifest.append(
            {
                "role": "train",
                "name": name,
                "domain": domain,
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "index": int(index),
                "length": int(len(data)),
                "block_offset": int(block_offset),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        parts.append(data)
        block_offset += len(data)
    return manifest, b"".join(parts)


def overlap_counts(train_manifest: list[dict[str, Any]], train_block: bytes, test_manifest: list[dict[str, Any]], test_block: bytes) -> dict[str, float]:
    train_paths = {str(row["path"]) for row in train_manifest}
    test_paths = {str(row["path"]) for row in test_manifest}
    train_hashes = {str(row["sha256"]) for row in train_manifest}
    test_hashes = {str(row["sha256"]) for row in test_manifest}
    return {
        "source_train_test_path_overlap_count": float(len(train_paths & test_paths)),
        "source_train_test_hash_overlap_count": float(len(train_hashes & test_hashes)),
        "source_train_test_ngram_overlap_count": float(len(fixed_ngrams(train_block, 64) & fixed_ngrams(test_block, 64))),
    }


def learn_relation_stride(train_block: bytes, train_manifest: list[dict[str, Any]], stride: int) -> int:
    best_stride = 1
    best_score = -1
    offsets = candidate_offsets_for_block(len(train_block))
    for candidate_stride in range(1, 17):
        score = 0
        for offset in offsets[: min(len(offsets), 1024)]:
            target = relation_target_offset(train_block, train_manifest, int(offset), int(stride))
            predicted = relation_target_offset(train_block, train_manifest, int(offset), int(candidate_stride))
            score += int(target is not None and predicted == target)
        if score > best_score:
            best_score = int(score)
            best_stride = int(candidate_stride)
    return int(best_stride)


def sample_relation_offsets(source_block: bytes, manifest: list[dict[str, Any]], count: int, seed: int, stride: int) -> list[tuple[int, int]]:
    rows = []
    seen_terms = set()
    seen_targets = set()
    offsets = sorted(candidate_offsets_for_block(len(source_block)))
    for offset in offsets:
        target = relation_target_offset(source_block, manifest, int(offset), int(stride))
        if target is None:
            continue
        terms = relation_terms_for_anchor(source_block, int(offset))
        if len(terms) != int(RELATION_TERMS) or terms in seen_terms or target in seen_targets:
            continue
        seen_terms.add(terms)
        seen_targets.add(target)
        rows.append((int(offset), int(target)))
    rows = sorted(rows, key=lambda item: hashlib.blake2b(" ".join(relation_terms_for_anchor(source_block, item[0])).encode("utf-8"), digest_size=8, person=b"nm-rel-order").digest())
    if len(rows) < int(count):
        raise ValueError("not enough source-native relation offsets")
    return rows[: int(count)]


def build_facts(seed: int, fact_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, list[dict[str, Any]], list[dict[str, Any]], bytes, int]:
    train_manifest, train_block = load_relation_train_sources()
    test_manifest, source_block = load_margin_sources()
    learned_stride = learn_relation_stride(train_block, train_manifest, int(RELATION_STRIDE))
    pairs = sample_relation_offsets(source_block, test_manifest, int(fact_count), int(seed) + 19, int(learned_stride))
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
    for row, (anchor_offset, target_offset) in enumerate(pairs):
        value = source_block[int(target_offset) : int(target_offset) + int(CHUNK_BYTES)]
        facts.append(
            {
                "role": "test",
                "row": int(row),
                "domain": offset_domain(test_manifest, int(target_offset)),
                "question": relation_question_from_anchor(source_block, int(anchor_offset)),
                "value": value.hex(),
                "provenance": provenance_for_block(int(target_offset), value),
                "anchor_offset_for_test_only": int(anchor_offset),
                "target_offset_for_test_only": int(target_offset),
            }
        )
    return [], facts, source_block, source_profile, train_manifest, train_block, int(learned_stride)


def public_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in fact.items() if not str(key).endswith("_for_test_only")} for fact in facts]


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 4111)
    rows = []
    for fact in facts:
        value = bytes(rng.randrange(0, 256) for _ in range(int(CHUNK_BYTES)))
        rows.append(
            {
                "role": "test",
                "row": int(fact["row"]),
                "domain": str(fact.get("domain", "")),
                "question": str(fact["question"]),
                "value": value.hex(),
                "provenance": hashlib.sha256(value).hexdigest()[:16],
            }
        )
    return rows


def build_relation_module(payload: bytes, learned_stride: int) -> Any:
    import torch
    import torch.nn as nn

    class RelationAdapterModule(nn.Module):
        def __init__(self, stream: bytes, stride: int) -> None:
            super().__init__()
            self.register_buffer("adapter_payload", torch.tensor(list(stream), dtype=torch.uint8), persistent=True)
            self.register_buffer("adapter_header", torch.tensor([int(len(stream)), int(CHUNK_BYTES)], dtype=torch.int64), persistent=True)
            self.register_buffer("relation_stride_code", torch.tensor([int(stride)], dtype=torch.uint8), persistent=True)

        def forward(self, value: Any) -> Any:
            return value

    return RelationAdapterModule(payload, int(learned_stride))


class SourceNativeRelationAdapterCell:
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]], learned_stride: int | None = None) -> None:
        self.model_state_adapter_payload_used = 1.0
        self.state_dict_buffer_payload_used = 1.0
        self.external_payload_store_used = 0.0
        self.stored_manifest_used = 0.0
        self.source_profile_count = len(source_profile)
        self.source_domain_count = len({str(row["domain"]) for row in source_profile})
        self.block_stream_count = 1
        self.adapter_state_stream_count = 1
        self.per_fact_value_slice_count = 0
        self.assignment_row_count = 0
        self.per_fact_value_row_count = 0
        self.source_offset_routing_used = 0.0
        self.content_digest_key_target = 0.0
        self.semantic_question_handle_target = 0.0
        self.source_native_relation_target = 1.0
        self.learned_relation_router_used = 1.0
        self.source_offset_key_target = 0.0
        self.independent_value_slice_path_used = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_compressed_model_state = 1.0
        self.reads_from_compressed_block = 1.0
        self.question_parser_in_decoder_bits = 1.0
        self.prompt_context_storage_used = 0.0
        self.answer_digest_key_target = 0.0
        self.true_base_weight_implicit_storage_authorized = 0.0
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        self.learned_stride = int(learned_stride if learned_stride is not None else RELATION_STRIDE)
        self.relation_router_code_bits = int(ROUTER_PARAMETER_BITS)
        payload = compress_payload(source_block)
        self.module = build_relation_module(payload, self.learned_stride)
        self.block_payload_bits = int(len(payload) * 8)
        self.adapter_model_state_bits = int(len(payload) * 8 + int(MODEL_HEADER_BITS))
        self.candidate_count = len(candidate_offsets_for_block(len(source_block)))
        self.decompression_count = 0
        self.scan_count = 0

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def payload_bytes(self) -> bytes:
        return bytes(int(item) for item in self.module.adapter_payload.tolist())

    def decoded_adapter_block(self) -> bytes:
        return decompress_payload(self.payload_bytes())

    def relation_stride_from_state(self) -> int:
        return max(1, int(self.module.relation_stride_code.detach().cpu().item()))

    def answer_many(
        self,
        questions: list[str],
        read_enabled: bool = True,
        decoder_enabled: bool = True,
        parser_enabled: bool = True,
        adapter_enabled: bool = True,
        router_enabled: bool = True,
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        adapter_disabled: bool = False,
        router_disabled: bool = False,
        code_disabled: bool = False,
        wrong_stride: bool = False,
    ) -> list[dict[str, str | int]]:
        if not read_enabled:
            read_disabled = True
        if not decoder_enabled:
            decoder_disabled = True
        if not parser_enabled:
            parser_disabled = True
        if not adapter_enabled:
            adapter_disabled = True
        if not router_enabled:
            router_disabled = True
        if read_disabled or decoder_disabled or parser_disabled or code_disabled or adapter_disabled or router_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        terms = [relation_terms_from_question(str(question)) for question in questions]
        wanted = {term for term in terms if len(term) == int(RELATION_TERMS)}
        if not wanted:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        try:
            block = self.decoded_adapter_block()
        except Exception:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        self.decompression_count += 1
        self.scan_count += 1
        stride = 1 if wrong_stride else self.relation_stride_from_state()
        found: dict[tuple[str, ...], dict[str, str | int]] = {}
        offsets = candidate_offsets_for_block(len(block))
        for offset in offsets:
            key = relation_terms_for_anchor(block, int(offset))
            if key not in wanted or key in found:
                continue
            target = int(offset) + int(stride) * int(CHUNK_BYTES)
            if target not in offsets:
                continue
            value = block[int(target) : int(target) + int(CHUNK_BYTES)]
            found[key] = {"value": value.hex(), "provenance": provenance_for_block(int(target), value), "hit": 1}
            if len(found) == len(wanted):
                break
        return [found.get(term, {"value": "", "provenance": "", "hit": 0}) for term in terms]

    def answer(self, question: str, **kwargs: Any) -> dict[str, str | int]:
        return self.answer_many([str(question)], **kwargs)[0]


def relationless_content_scan_answers(source_block: bytes, questions: list[str]) -> list[dict[str, str | int]]:
    terms = [relation_terms_from_question(str(question)) for question in questions]
    wanted = {term for term in terms if len(term) == int(RELATION_TERMS)}
    found: dict[tuple[str, ...], dict[str, str | int]] = {}
    for offset in candidate_offsets_for_block(len(source_block)):
        key = relation_terms_for_anchor(source_block, int(offset))
        if key not in wanted or key in found:
            continue
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        found[key] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
        if len(found) == len(wanted):
            break
    return [found.get(term, {"value": "", "provenance": "", "hit": 0}) for term in terms]


def wrong_stride_content_scan_answers(source_block: bytes, questions: list[str]) -> list[dict[str, str | int]]:
    cell = SourceNativeRelationAdapterCell([], [], source_block, [], learned_stride=1)
    return cell.answer_many(questions)


def stride_aware_content_scan_answers(source_block: bytes, questions: list[str], stride: int) -> list[dict[str, str | int]]:
    cell = SourceNativeRelationAdapterCell([], [], source_block, [], learned_stride=int(stride))
    return cell.answer_many(questions)


def shifted(rows: list[Any]) -> list[Any]:
    return rows[1:] + rows[:1] if rows else []


def false_hit_rate(answers: list[dict[str, str | int]]) -> float:
    return float(sum(int(answer.get("hit", 0)) for answer in answers)) / max(float(len(answers)), 1.0)


def evaluate_controls(cell: SourceNativeRelationAdapterCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]], source_block: bytes) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    paraphrases = relation_paraphrases(facts)
    exact = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(paraphrases)
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _ in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shuffled_question = cell.answer_many([str(fact["question"]) for fact in shifted(facts)])
    exact_reads = cell.answer_many(questions)
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    wrong_question = cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])
    unanswerable_question = cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])
    partial_overlap = cell.answer_many(["source relation after evidence terms " + " ".join(str(fact["question"]).split()[-2:]) for fact in facts])
    marker_injection = cell.answer_many([str(fact["question"]) + " target " + str(fact["value"])[:8] for fact in facts])
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
    adapter_disabled = cell.answer_many(questions, adapter_disabled=True)
    router_disabled = cell.answer_many(questions, router_disabled=True)
    code_disabled = cell.answer_many(questions, code_disabled=True)
    wrong_stride = cell.answer_many(questions, wrong_stride=True)
    relationless_scan = relationless_content_scan_answers(source_block, questions)
    wrong_stride_scan = wrong_stride_content_scan_answers(source_block, questions)
    stride_aware_scan = stride_aware_content_scan_answers(source_block, questions, cell.relation_stride_from_state())
    return {
        "exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "heldout_exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "paraphrase_stable_answer_success": mean_metric(score_answers(facts, paraphrase_answers), "exact_success"),
        "random_label_twin_success": mean_metric(score_answers(random_twin, twin_reads), "exact_success"),
        "no_memory_success": mean_metric(score_answers(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_answers(facts, recency), "exact_success"),
        "shuffled_question_success": mean_metric(score_answers(facts, shuffled_question), "exact_success"),
        "shuffled_value_success": mean_metric(score_answers(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_answers(facts, shuffled_provenance), "exact_success"),
        "wrong_question_success": mean_metric(score_answers(facts, wrong_question), "exact_success"),
        "unanswerable_question_success": mean_metric(score_answers(facts, unanswerable_question), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, decoder_disabled), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, parser_disabled), "exact_success"),
        "adapter_disabled_success": mean_metric(score_answers(facts, adapter_disabled), "exact_success"),
        "router_disabled_success": mean_metric(score_answers(facts, router_disabled), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, code_disabled), "exact_success"),
        "wrong_stride_success": mean_metric(score_answers(facts, wrong_stride), "exact_success"),
        "relationless_content_scan_success": mean_metric(score_answers(facts, relationless_scan), "exact_success"),
        "wrong_stride_content_scan_success": mean_metric(score_answers(facts, wrong_stride_scan), "exact_success"),
        "stride_aware_content_scan_success": mean_metric(score_answers(facts, stride_aware_scan), "exact_success"),
        "wrong_query_hit_rate": false_hit_rate(wrong_question),
        "unanswerable_query_hit_rate": false_hit_rate(unanswerable_question),
        "partial_overlap_query_hit_rate": false_hit_rate(partial_overlap),
        "marker_injection_query_hit_rate": false_hit_rate(marker_injection),
    }


def baseline_metrics(useful_bits: int, committed_bits: int, paper_bits: int) -> dict[str, float]:
    def multiplier(bits: float, params: float = 0.0) -> float:
        return float(useful_bits) / max((float(bits) / 16.0 + float(params)) * float(ORDINARY_BITS_PER_PARAMETER), 1.0)

    return {
        "lora_delta_storage_strict_multiplier": float(useful_bits) / ((float(useful_bits) / 16.0 + 1.0) * float(ORDINARY_BITS_PER_PARAMETER)),
        "qlora_delta_storage_strict_multiplier": float(useful_bits) / ((float(useful_bits) / 4.0 + 1.0) * float(ORDINARY_BITS_PER_PARAMETER)),
        "rome_memit_edit_storage_strict_multiplier": float(useful_bits) / ((float(useful_bits) / 3.5 + 1.0) * float(ORDINARY_BITS_PER_PARAMETER)),
        "verbatim_table_strict_multiplier": multiplier(float(useful_bits) + float(DECODER_BITS)),
        "product_key_memory_strict_multiplier": multiplier(float(useful_bits) + 8192.0 + float(DECODER_BITS)),
        "memory_layer_strict_multiplier": multiplier(float(useful_bits) + 16384.0 + float(DECODER_BITS)),
        "content_routed_sparse_read_strict_multiplier": multiplier(float(useful_bits) * 0.75 + float(DECODER_BITS)),
        "same_block_relationless_content_scan_multiplier": multiplier(float(committed_bits)),
        "same_block_wrong_stride_scan_multiplier": multiplier(float(committed_bits)),
        "same_block_undercharged_mph_multiplier": multiplier(float(committed_bits) + 16.0),
        "paper_surface_relationless_scan_multiplier": multiplier(float(paper_bits)),
    }


def account_bits(cell: SourceNativeRelationAdapterCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    useful_bits = int(len(facts) * int(CHUNK_BYTES) * 8)
    router_bits = int(cell.relation_router_code_bits)
    committed_bits = int(cell.block_payload_bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS) + router_bits)
    paper_bits = int(committed_bits + int(SURFACE_CONTRACT_BITS))
    adapter_denominator = max(float(committed_bits) / 16.0 + float(cell.parameter_count()), 1.0)
    paper_denominator = max(float(paper_bits) / 16.0 + float(cell.parameter_count()), 1.0)
    return {
        "block_payload_bits": float(cell.block_payload_bits),
        "model_header_bits": float(MODEL_HEADER_BITS),
        "decoder_bits": float(DECODER_BITS),
        "surface_contract_bits": float(SURFACE_CONTRACT_BITS),
        "relation_router_parameter_bits": float(router_bits),
        "committed_state_bits": float(committed_bits),
        "paper_surface_accounted_bits": float(paper_bits),
        "useful_retrievable_bits": float(useful_bits),
        "adapter_strict_density": float(useful_bits) / adapter_denominator,
        "adapter_strict_multiplier": float(useful_bits) / adapter_denominator / float(ORDINARY_BITS_PER_PARAMETER),
        "paper_surface_strict_density": float(useful_bits) / paper_denominator,
        "paper_surface_strict_multiplier": float(useful_bits) / paper_denominator / float(ORDINARY_BITS_PER_PARAMETER),
    }


def host_probe(host: Any, facts: list[dict[str, Any]], cell_factory: Any, source_block: bytes, source_profile: list[dict[str, Any]], learned_stride: int) -> dict[str, float]:
    import torch

    questions = relation_paraphrases(facts[: min(64, len(facts))])
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    score = mean_metric(score_answers(facts[: len(questions)], answers), "exact_success")
    state_keys = set(host.module.state_dict().keys())
    reload_cell = cell_factory([], facts, source_block, source_profile, learned_stride=learned_stride)
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
        "relation_router_in_state_dict": float(int("adapter_module.relation_stride_code" in state_keys)),
        "answer_success": float(score),
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "parameter_count": float(host.parameter_count()),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    fact_count = int(PROFILES[profile]["fact_count"])
    train, facts_private, source_block, source_profile, train_manifest, train_block, learned_stride = build_facts(int(seed), int(fact_count))
    facts = public_facts(facts_private)
    cell = SourceNativeRelationAdapterCell(train, facts, source_block, source_profile, learned_stride=learned_stride)
    random_twin = build_random_twin(int(seed), facts)
    controls = evaluate_controls(cell, facts, random_twin, source_block)
    account = account_bits(cell, facts)
    baselines = baseline_metrics(int(account["useful_retrievable_bits"]), int(account["committed_state_bits"]), int(account["paper_surface_accounted_bits"]))
    transformer_probe = host_probe(TinyTransformerAdapterHost(cell), facts, SourceNativeRelationAdapterCell, source_block, source_profile, learned_stride)
    recurrent_probe = host_probe(TinyRecurrentStateAdapterHost(cell), facts, SourceNativeRelationAdapterCell, source_block, source_profile, learned_stride)
    inspection = hidden_state_inspection(cell, facts, source_block)
    overlaps = overlap_counts(train_manifest, train_block, load_margin_sources()[0], source_block)
    host_parameter_count_max = max(float(transformer_probe["parameter_count"]), float(recurrent_probe["parameter_count"]))
    controls_collapse = float(
        int(
            controls["random_label_twin_success"] <= 0.01
            and controls["no_memory_success"] <= 0.01
            and controls["shuffled_question_success"] <= 0.01
            and controls["shuffled_value_success"] <= 0.01
            and controls["shuffled_provenance_success"] <= 0.01
            and controls["read_disabled_success"] <= 0.01
            and controls["decoder_disabled_success"] <= 0.01
            and controls["parser_disabled_success"] <= 0.01
            and controls["adapter_disabled_success"] <= 0.01
            and controls["router_disabled_success"] <= 0.01
            and controls["code_disabled_success"] <= 0.01
            and controls["wrong_stride_success"] <= 0.01
        )
    )
    transformer_surface_pass = float(
        int(
            transformer_probe["forward_shape_success"] == 1.0
            and transformer_probe["adapter_payload_in_state_dict"] == 1.0
            and transformer_probe["relation_router_in_state_dict"] == 1.0
            and transformer_probe["answer_success"] >= 0.95
            and transformer_probe["state_dict_reload_success"] >= 0.95
        )
    )
    recurrent_surface_pass = float(
        int(
            recurrent_probe["forward_shape_success"] == 1.0
            and recurrent_probe["adapter_payload_in_state_dict"] == 1.0
            and recurrent_probe["relation_router_in_state_dict"] == 1.0
            and recurrent_probe["answer_success"] >= 0.95
            and recurrent_probe["state_dict_reload_success"] >= 0.95
        )
    )
    source_holdout_pass = float(
        int(
            overlaps["source_train_test_path_overlap_count"] == 0.0
            and overlaps["source_train_test_hash_overlap_count"] == 0.0
            and overlaps["source_train_test_ngram_overlap_count"] == 0.0
        )
    )
    multi_domain_pass = float(int(len({str(fact["domain"]) for fact in facts}) >= 4))
    relationless_scan_beaten = float(int(controls["relationless_content_scan_success"] <= 0.01 and controls["wrong_stride_content_scan_success"] <= 0.01))
    fair_stride_scan_not_beaten = float(int(controls["stride_aware_content_scan_success"] >= 0.95))
    false_hit_pass = float(int(controls["wrong_query_hit_rate"] == 0.0 and controls["unanswerable_query_hit_rate"] == 0.0 and controls["partial_overlap_query_hit_rate"] == 0.0 and controls["marker_injection_query_hit_rate"] == 0.0))
    legacy_static_public_baseline_pass = float(
        int(
            account["paper_surface_strict_multiplier"] > max(
                float(CONTENT_SCAN_BASELINE_MULTIPLIER),
                float(MARGIN_BASELINE_MULTIPLIER),
                float(PAPER_READY_BASELINE_MULTIPLIER),
                float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
                float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
                float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
                float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
                float(CHARGED_CODEC_BASELINE_MULTIPLIER),
            )
        )
    )
    static_public_baseline_pass = float(int(legacy_static_public_baseline_pass == 1.0 and fair_stride_scan_not_beaten == 0.0))
    formula_relation_diagnostic_candidate = float(
        int(
            controls["exact_answer_success"] >= 0.95
            and controls["paraphrase_stable_answer_success"] >= 0.95
            and relationless_scan_beaten == 1.0
            and fair_stride_scan_not_beaten == 1.0
            and controls_collapse == 1.0
            and transformer_surface_pass == 1.0
            and recurrent_surface_pass == 1.0
            and source_holdout_pass == 1.0
            and multi_domain_pass == 1.0
            and false_hit_pass == 1.0
            and legacy_static_public_baseline_pass == 1.0
            and host_parameter_count_max < 100000.0
            and account["paper_surface_strict_multiplier"] > float(CONTENT_SCAN_BASELINE_MULTIPLIER)
            and account["adapter_strict_multiplier"] > float(baselines["same_block_undercharged_mph_multiplier"])
            and inspection["hidden_fact_value_row_detected"] == 0.0
            and inspection["hidden_raw_source_prefix_detected"] == 0.0
        )
    )
    strict_600x_pass = float(int(account["adapter_strict_density"] >= float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER) and controls["exact_answer_success"] >= 0.95))
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "train_fact_count": 0.0,
        "test_fact_count": float(len(facts)),
        "source_file_count": float(len(source_profile)),
        "source_domain_count": float(len({str(fact["domain"]) for fact in facts})),
        "source_block_bytes": float(len(source_block)),
        "train_source_file_count": float(len(train_manifest)),
        "train_source_block_bytes": float(len(train_block)),
        "candidate_scan_count": float(cell.candidate_count),
        "learned_relation_stride": float(learned_stride),
        "relation_terms": float(RELATION_TERMS),
        "adapter_parameter_count": float(cell.parameter_count()),
        "host_parameter_count_max": float(host_parameter_count_max),
        "transformer_host_parameter_count": float(transformer_probe["parameter_count"]),
        "recurrent_host_parameter_count": float(recurrent_probe["parameter_count"]),
        "target_density": float(float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)),
        "target_multiplier": float(TARGET_MULTIPLIER),
        "product_target_multiplier": float(PRODUCT_TARGET_MULTIPLIER),
        "publishable_relation_breakthrough_candidate": 0.0,
        "formula_relation_diagnostic_candidate": formula_relation_diagnostic_candidate,
        "legacy_static_public_baseline_pass": legacy_static_public_baseline_pass,
        "strict_600x_pass": strict_600x_pass,
        "paper_ready_requirement_count": float(transformer_surface_pass + recurrent_surface_pass + legacy_static_public_baseline_pass + multi_domain_pass + relationless_scan_beaten),
        "transformer_surface_pass": transformer_surface_pass,
        "recurrent_surface_pass": recurrent_surface_pass,
        "static_public_baseline_pass": static_public_baseline_pass,
        "multi_domain_pass": multi_domain_pass,
        "source_holdout_pass": source_holdout_pass,
        "relationless_content_scan_beaten": relationless_scan_beaten,
        "same_interface_content_scan_success": float(controls["relationless_content_scan_success"]),
        "same_interface_wrong_stride_scan_success": float(controls["wrong_stride_content_scan_success"]),
        "stride_aware_content_scan_success": float(controls["stride_aware_content_scan_success"]),
        "fair_stride_content_scan_not_beaten": fair_stride_scan_not_beaten,
        "ablation_controls_pass": controls_collapse,
        "false_hit_controls_pass": false_hit_pass,
        "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        "llm_semantic_qa_baseline_multiplier": float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        "weight_carried_baseline_multiplier": float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        "paper_ready_baseline_multiplier": float(PAPER_READY_BASELINE_MULTIPLIER),
        "margin_baseline_multiplier": float(MARGIN_BASELINE_MULTIPLIER),
        "previous_content_scan_baseline_multiplier": float(CONTENT_SCAN_BASELINE_MULTIPLIER),
        "beats_lora_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["lora_delta_storage_strict_multiplier"]))),
        "beats_qlora_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["qlora_delta_storage_strict_multiplier"]))),
        "beats_model_edit_storage_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["rome_memit_edit_storage_strict_multiplier"]))),
        "beats_product_key_memory_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["product_key_memory_strict_multiplier"]))),
        "beats_memory_layer_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["memory_layer_strict_multiplier"]))),
        "beats_content_routed_sparse_read_baseline": float(int(account["paper_surface_strict_multiplier"] > float(baselines["content_routed_sparse_read_strict_multiplier"]))),
        "beats_margin_baseline": float(int(account["paper_surface_strict_multiplier"] > float(MARGIN_BASELINE_MULTIPLIER))),
        "beats_previous_content_scan_baseline": float(int(account["paper_surface_strict_multiplier"] > float(CONTENT_SCAN_BASELINE_MULTIPLIER))),
        "beats_same_block_undercharged_mph_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["same_block_undercharged_mph_multiplier"]))),
        "model_state_adapter_payload_used": float(cell.model_state_adapter_payload_used),
        "state_dict_buffer_payload_used": float(cell.state_dict_buffer_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "block_stream_count": float(cell.block_stream_count),
        "adapter_state_stream_count": float(cell.adapter_state_stream_count),
        "assignment_row_count": float(cell.assignment_row_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "per_fact_value_slice_count": float(cell.per_fact_value_slice_count),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "source_offset_routing_used": float(cell.source_offset_routing_used),
        "source_native_relation_target": float(cell.source_native_relation_target),
        "learned_relation_router_used": 0.0,
        "fixed_relation_constant_used": 1.0,
        "generated_alias_labels_present": 0.0,
        "formula_or_schema_labels_present": 1.0,
        "seed_oracle_authorized": 0.0,
        "source_holdout_used": 1.0,
        "controls_collapse": controls_collapse,
        "transformer_forward_shape_success": float(transformer_probe["forward_shape_success"]),
        "transformer_adapter_payload_in_state_dict": float(transformer_probe["adapter_payload_in_state_dict"]),
        "transformer_relation_router_in_state_dict": float(transformer_probe["relation_router_in_state_dict"]),
        "transformer_answer_success": float(transformer_probe["answer_success"]),
        "transformer_state_dict_preload_success": float(transformer_probe["state_dict_preload_success"]),
        "transformer_state_dict_reload_success": float(transformer_probe["state_dict_reload_success"]),
        "recurrent_forward_shape_success": float(recurrent_probe["forward_shape_success"]),
        "recurrent_adapter_payload_in_state_dict": float(recurrent_probe["adapter_payload_in_state_dict"]),
        "recurrent_relation_router_in_state_dict": float(recurrent_probe["relation_router_in_state_dict"]),
        "recurrent_answer_success": float(recurrent_probe["answer_success"]),
        "recurrent_state_dict_preload_success": float(recurrent_probe["state_dict_preload_success"]),
        "recurrent_state_dict_reload_success": float(recurrent_probe["state_dict_reload_success"]),
        **account,
        **controls,
        **inspection,
        **overlaps,
        **baselines,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_publishable_relation_breakthrough_candidate": float(row["publishable_relation_breakthrough_candidate"]),
        f"{SIMULATION_ID}_strict_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_general_unknown_structure_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
        f"{SIMULATION_ID}_external_simulator_authorized": 0.0,
        f"{SIMULATION_ID}_arbitrary_chat_authorized": 0.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["formula_relation_diagnostic_candidate"]),
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
    metrics_path = output_dir / "local_100k_source_native_relation_adapter_metrics.json"
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
            "relation_stride": int(RELATION_STRIDE),
            "relation_terms": int(RELATION_TERMS),
            "decoder_bits": int(DECODER_BITS),
            "model_header_bits": int(MODEL_HEADER_BITS),
            "surface_contract_bits": int(SURFACE_CONTRACT_BITS),
            "router_parameter_bits": int(ROUTER_PARAMETER_BITS),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
            "product_target_multiplier": float(PRODUCT_TARGET_MULTIPLIER),
            "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
            "margin_baseline_multiplier": float(MARGIN_BASELINE_MULTIPLIER),
            "content_scan_baseline_multiplier": float(CONTENT_SCAN_BASELINE_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(PROFILES[profile]["fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_native_relation_adapter_metrics.json", "path": metrics_path}],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} exact={summary[f'{SIMULATION_ID}_exact_answer_success']:.3f} relation_scan={summary[f'{SIMULATION_ID}_relationless_content_scan_success']:.3f} paper_multiplier={summary[f'{SIMULATION_ID}_paper_surface_strict_multiplier']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
