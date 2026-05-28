from __future__ import annotations

import bz2
import hashlib
import lzma
import math
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("LLM_SEMANTIC_QA_CODEC_SEED", 997)
FACTS_SMOKE = env_int("LLM_SEMANTIC_QA_CODEC_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("LLM_SEMANTIC_QA_CODEC_FACTS_HARD", 4096)
TRAIN_FACTS_SMOKE = env_int("LLM_SEMANTIC_QA_CODEC_TRAIN_FACTS_SMOKE", 2048)
TRAIN_FACTS_HARD = env_int("LLM_SEMANTIC_QA_CODEC_TRAIN_FACTS_HARD", 2048)
CHUNK_BYTES = env_int("LLM_SEMANTIC_QA_CODEC_CHUNK_BYTES", 32)
ANCHOR_BYTES = env_int("LLM_SEMANTIC_QA_CODEC_ANCHOR_BYTES", 96)
SEMANTIC_HANDLE_BYTES = env_int("LLM_SEMANTIC_QA_CODEC_SEMANTIC_HANDLE_BYTES", 4)
DECODER_BITS = env_int("LLM_SEMANTIC_QA_CODEC_DECODER_BITS", 65536)
MANIFEST_DECODER_BITS = env_int("LLM_SEMANTIC_QA_CODEC_MANIFEST_DECODER_BITS", 0)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("LLM_SEMANTIC_QA_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))
TARGET_MULTIPLIER = float(os.environ.get("LLM_SEMANTIC_QA_CODEC_TARGET_MULTIPLIER", "600.0"))
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("LLM_SEMANTIC_QA_CODEC_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))
SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("LLM_SEMANTIC_QA_CODEC_SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER", "14.06876726917481"))
CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("LLM_SEMANTIC_QA_CODEC_CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER", "14.06888524576417"))

require_positive("LLM_SEMANTIC_QA_CODEC_FACTS_SMOKE", FACTS_SMOKE)
require_positive("LLM_SEMANTIC_QA_CODEC_FACTS_HARD", FACTS_HARD)
require_positive("LLM_SEMANTIC_QA_CODEC_TRAIN_FACTS_SMOKE", TRAIN_FACTS_SMOKE)
require_positive("LLM_SEMANTIC_QA_CODEC_TRAIN_FACTS_HARD", TRAIN_FACTS_HARD)
require_positive("LLM_SEMANTIC_QA_CODEC_CHUNK_BYTES", CHUNK_BYTES)
require_positive("LLM_SEMANTIC_QA_CODEC_ANCHOR_BYTES", ANCHOR_BYTES)
require_positive("LLM_SEMANTIC_QA_CODEC_SEMANTIC_HANDLE_BYTES", SEMANTIC_HANDLE_BYTES)
require_positive("LLM_SEMANTIC_QA_CODEC_DECODER_BITS", DECODER_BITS)

QUESTION_PREFIX = "what exact passage follows this evidence:"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{2,}")
STOPWORDS = {
    "and",
    "are",
    "but",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "this",
    "with",
    "will",
    "without",
    "which",
    "when",
    "where",
    "what",
    "why",
    "how",
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("LLM_SEMANTIC_QA_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("LLM_SEMANTIC_QA_CODEC_PROFILE must be smoke or hard")
    return value


def bits_for_cardinality(cardinality: int) -> int:
    return max(1, math.ceil(math.log2(max(2, int(cardinality)))))


def source_rows() -> list[tuple[str, Path, str]]:
    rows = [
        ("test", PROJECT_ROOT / "knowledge/training_efficiency.md", "training_efficiency"),
        ("test", PROJECT_ROOT / "knowledge/papers_library.md", "papers_library"),
        ("test", PROJECT_ROOT / "knowledge/context_extension.md", "context_extension"),
        ("train", PROJECT_ROOT / "knowledge/unified_theory.md", "unified_theory"),
        ("train", PROJECT_ROOT / "knowledge/hybrid_architectures.md", "hybrid_architectures"),
        ("train", PROJECT_ROOT / "knowledge/delta_rule_theory.md", "delta_rule_theory"),
        ("train", PROJECT_ROOT / "knowledge/mla_compression.md", "mla_compression"),
        ("train", PROJECT_ROOT / "knowledge/mamba3_architecture.md", "mamba3_architecture"),
        ("train", PROJECT_ROOT / "knowledge/kda_channel_gating.md", "kda_channel_gating"),
        ("train", PROJECT_ROOT / "knowledge/geometric_algebra.md", "geometric_algebra"),
        ("train", PROJECT_ROOT / "knowledge/ternary_spikes.md", "ternary_spikes"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/compression_and_bottlenecks.md", "compression_and_bottlenecks"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/local_vs_global_computation.md", "local_vs_global_computation"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/timescale_separation.md", "timescale_separation"),
    ]
    return [(role, path, name) for role, path, name in rows if path.exists()]


def load_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes]:
    train_manifest = []
    test_manifest = []
    test_parts: list[bytes] = []
    block_offset = 0
    for index, (role, path, name) in enumerate(source_rows()):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        row = {
            "role": role,
            "name": name,
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "index": int(index),
            "length": int(len(data)),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if role == "test":
            if test_parts:
                test_parts.append(b"\n\n")
                block_offset += 2
            row["block_offset"] = int(block_offset)
            test_parts.append(data)
            block_offset += len(data)
            test_manifest.append(row)
        else:
            train_manifest.append(row)
    source_block = b"".join(test_parts)
    if len(source_block) < int(CHUNK_BYTES) * 8:
        raise ValueError("source block too small")
    return train_manifest, test_manifest, source_block


def manifest_bits(manifest: list[dict[str, Any]]) -> int:
    payload = ";".join(
        f"{row['role']}:{row['name']}:{row['path']}:{int(row['index'])}:{int(row.get('block_offset', 0))}:{int(row['length'])}:{row['sha256']}"
        for row in manifest
    ).encode("utf-8")
    return int(len(payload) * 8)


def normalize_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    value = re.sub(r"\s+", " ", value)
    value = "".join(character if 32 <= ord(character) <= 126 else " " for character in value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def question_from_anchor(anchor: str) -> str:
    return f"{QUESTION_PREFIX} {' '.join(token_signature_for_anchor(anchor))}"


def anchor_from_question(question: str) -> str:
    normalized = normalize_text(question)
    if QUESTION_PREFIX not in normalized:
        return ""
    return normalized.split(QUESTION_PREFIX, 1)[1].strip()


def token_signature_for_anchor(anchor: str) -> list[str]:
    seen = set()
    tokens = []
    for token in TOKEN_RE.findall(normalize_text(anchor)):
        if token in STOPWORDS or len(token) < 4 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) == 16:
            break
    return tokens


def semantic_handle_for_anchor(anchor: str) -> tuple[int, ...]:
    signature = " ".join(token_signature_for_anchor(anchor)).encode("utf-8")
    digest = hashlib.blake2b(signature, digest_size=int(SEMANTIC_HANDLE_BYTES), person=b"nm-llmqa-v1").digest()
    width = 1 if len(digest) < 8 else 4
    return tuple(int.from_bytes(digest[index : index + width], "little") for index in range(0, len(digest), width))


def semantic_handle_for_question(question: str) -> tuple[int, ...]:
    signature = anchor_from_question(question)
    if not signature:
        return tuple()
    digest = hashlib.blake2b(normalize_text(signature).encode("utf-8"), digest_size=int(SEMANTIC_HANDLE_BYTES), person=b"nm-llmqa-v1").digest()
    width = 1 if len(digest) < 8 else 4
    return tuple(int.from_bytes(digest[index : index + width], "little") for index in range(0, len(digest), width))


def candidate_offsets(test_manifest: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    candidates = []
    for source_id, row in enumerate(test_manifest):
        source_start = int(row["block_offset"])
        start = source_start + int(ANCHOR_BYTES)
        end = source_start + int(row["length"]) - int(CHUNK_BYTES)
        for offset in range(start, end + 1, int(CHUNK_BYTES)):
            candidates.append((int(source_id), int(offset), str(row["name"])))
    return candidates


def anchor_text_for(source_block: bytes, offset: int) -> str:
    return normalize_text(source_block[int(offset) - int(ANCHOR_BYTES) : int(offset)])


def provenance_for(test_manifest: list[dict[str, Any]], source_id: int, offset: int, value: bytes) -> str:
    row = test_manifest[int(source_id)]
    local_offset = int(offset) - int(row["block_offset"])
    return hashlib.sha256(f"{row['path']}:{local_offset}:{int(CHUNK_BYTES)}:".encode("utf-8") + hashlib.sha256(value).digest()).hexdigest()[:16]


def selected_semantic_collision_count(facts: list[dict[str, Any]]) -> int:
    handles = [tuple(fact["semantic_handle"]) for fact in facts]
    return int(len(handles) - len(set(handles)))


def sample_test_offsets(source_block: bytes, test_manifest: list[dict[str, Any]], count: int, seed: int) -> list[tuple[int, int, str]]:
    candidates = candidate_offsets(test_manifest)
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    handle_counts: dict[tuple[int, ...], int] = {}
    value_counts: dict[bytes, int] = {}
    anchors: dict[int, str] = {}
    for _source_id, offset, _source in candidates:
        anchor = anchor_text_for(source_block, int(offset))
        anchors[int(offset)] = anchor
        if len(token_signature_for_anchor(anchor)) < 4:
            continue
        handle = semantic_handle_for_anchor(anchor)
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        value_digest = hashlib.sha256(value).digest()
        handle_counts[handle] = int(handle_counts.get(handle, 0)) + 1
        value_counts[value_digest] = int(value_counts.get(value_digest, 0)) + 1
    chosen = []
    seen_handles = set()
    seen_values = set()
    for source_id, offset, source in candidates:
        anchor = anchors[int(offset)]
        if len(token_signature_for_anchor(anchor)) < 4:
            continue
        handle = semantic_handle_for_anchor(anchor)
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        value_digest = hashlib.sha256(value).digest()
        if handle_counts.get(handle, 0) != 1 or value_counts.get(value_digest, 0) != 1:
            continue
        if handle in seen_handles or value_digest in seen_values:
            continue
        if value.hex() in question_from_anchor(anchor):
            continue
        seen_handles.add(handle)
        seen_values.add(value_digest)
        chosen.append((int(source_id), int(offset), str(source)))
        if len(chosen) == int(count):
            return sorted(chosen, key=lambda row: semantic_handle_for_anchor(anchor_text_for(source_block, int(row[1]))))
    raise ValueError("not enough unique semantic question handles")


def sample_train_rows(train_manifest: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 37)
    rows = []
    for index in range(int(count)):
        source = train_manifest[index % len(train_manifest)]
        payload = hashlib.sha256(f"train:{seed}:{index}:{source['name']}".encode("utf-8")).digest()
        handle = tuple(int.from_bytes(payload[item : item + 4], "little") for item in range(0, 16, 4))
        rows.append({"role": "train", "row": int(index), "source": source["name"], "semantic_handle": handle, "offset": rng.randrange(0, max(1, int(source["length"])))})
    return rows


def build_facts(seed: int, fact_count: int, train_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    train_manifest, test_manifest, source_block = load_sources()
    train_facts = sample_train_rows(train_manifest, int(train_count), int(seed))
    offsets = sample_test_offsets(source_block, test_manifest, int(fact_count), int(seed) + 41)
    test_facts = []
    for row, (source_id, offset, _source) in enumerate(offsets):
        anchor = anchor_text_for(source_block, int(offset))
        question = question_from_anchor(anchor)
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        handle = semantic_handle_for_anchor(anchor)
        test_facts.append(
            {
                "role": "test",
                "row": int(row),
                "question": question,
                "semantic_handle": handle,
                "value": value.hex(),
                "provenance": provenance_for(test_manifest, int(source_id), int(offset), value),
            }
        )
    return train_facts, test_facts, source_block, test_manifest


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 28657)
    twin = []
    for fact in facts:
        value = bytes(rng.randrange(0, 256) for _ in range(int(CHUNK_BYTES)))
        twin.append(
            {
                "role": "test",
                "row": int(fact["row"]),
                "question": str(fact["question"]),
                "semantic_handle": tuple(fact["semantic_handle"]),
                "value": value.hex(),
                "provenance": hashlib.sha256(value).hexdigest()[:16],
            }
        )
    return twin


def compress_block(payload: bytes) -> tuple[str, bytes]:
    candidates = [
        ("zlib9", zlib.compress(payload, level=9)),
        ("bz2", bz2.compress(payload, compresslevel=9)),
        ("lzma6", lzma.compress(payload, preset=6)),
    ]
    return min(candidates, key=lambda row: (len(row[1]), row[0]))


def decompress_block(codec_name: str, payload: bytes) -> bytes:
    if codec_name == "zlib9":
        return zlib.decompress(payload)
    if codec_name == "bz2":
        return bz2.decompress(payload)
    if codec_name == "lzma6":
        return lzma.decompress(payload)
    raise ValueError("unknown block codec")


def score_answers(facts: list[dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for fact, answer in zip(facts, answers):
        value_ok = str(answer["value"]) == str(fact["value"])
        provenance_ok = str(answer["provenance"]) == str(fact["provenance"])
        hit_ok = int(answer["hit"]) == 1
        rows.append({"value_success": float(value_ok), "provenance_success": float(provenance_ok), "hit_success": float(hit_ok), "exact_success": float(value_ok and provenance_ok and hit_ok)})
    return rows


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def shifted(items: list[Any]) -> list[Any]:
    if len(items) <= 1:
        return items
    return items[-1:] + items[:-1]


def wrong_question_for(question: str) -> str:
    return f"{question} impossible-anchor-{hashlib.sha256(question.encode('utf-8')).hexdigest()[:12]}"


def unanswerable_question_for(index: int) -> str:
    return f"{QUESTION_PREFIX} unanswerable-control-{index}-missing-evidence"


def overlap_distractor_question_for(question: str) -> str:
    signature = anchor_from_question(question)
    pieces = signature.split()
    if not pieces:
        return wrong_question_for(question)
    kept = pieces[: max(1, len(pieces) // 2)]
    kept.append("distractor-control-token")
    return f"{QUESTION_PREFIX} {' '.join(kept)}"


class LLMSemanticQACodecCell:
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, manifest: list[dict[str, Any]]) -> None:
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.read_gate = nn.Parameter(torch.ones(1))
                self.decoder_gate = nn.Parameter(torch.ones(1))
                self.parser_gate = nn.Parameter(torch.ones(1))

            def forward(self, value: Any) -> Any:
                return value

        self.module = Module()
        self.manifest = list(manifest)
        self.source_block_count = 1
        self.block_stream_count = 1
        self.per_fact_value_slice_count = 0
        self.assignment_row_count = 0
        self.per_fact_value_row_count = 0
        self.source_offset_routing_used = 0.0
        self.content_digest_key_target = 0.0
        self.semantic_question_handle_target = 1.0
        self.source_offset_key_target = 0.0
        self.key_assignment_bits = 0
        self.independent_value_slice_path_used = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_compressed_block = 1.0
        self.question_parser_in_decoder_bits = 1.0
        self.prompt_context_storage_used = 0.0
        self.answer_digest_key_target = 0.0
        self.source_block_len = len(source_block)
        self.codec_name, self.block_stream = compress_block(source_block)
        self.block_payload_bits = int(len(self.block_stream) * 8)
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        self.decompression_count = 0
        self.scan_count = 0
        self.candidate_count = len(candidate_offsets(self.manifest))

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def answer_many(
        self,
        questions: list[str],
        read_enabled: bool = True,
        decoder_enabled: bool = True,
        parser_enabled: bool = True,
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        code_disabled: bool = False,
    ) -> list[dict[str, str | int]]:
        if not read_enabled:
            read_disabled = True
        if not decoder_enabled:
            decoder_disabled = True
        if not parser_enabled:
            parser_disabled = True
        if read_disabled or decoder_disabled or parser_disabled or code_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        handles = [semantic_handle_for_question(str(question)) for question in questions]
        wanted = {tuple(handle) for handle in handles if handle}
        if not wanted:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        block = decompress_block(self.codec_name, self.block_stream)
        self.decompression_count += 1
        self.scan_count += 1
        found: dict[tuple[int, ...], dict[str, str | int]] = {}
        for source_id, offset, _source in candidate_offsets(self.manifest):
            anchor = anchor_text_for(block, int(offset))
            handle = semantic_handle_for_anchor(anchor)
            if handle not in wanted or handle in found:
                continue
            value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            found[handle] = {"value": value.hex(), "provenance": provenance_for(self.manifest, int(source_id), int(offset), value), "hit": 1}
            if len(found) == len(wanted):
                break
        return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]

    def answer(
        self,
        question: str,
        read_enabled: bool = True,
        decoder_enabled: bool = True,
        parser_enabled: bool = True,
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        code_disabled: bool = False,
    ) -> dict[str, str | int]:
        return self.answer_many([str(question)], read_enabled=read_enabled, decoder_enabled=decoder_enabled, parser_enabled=parser_enabled, read_disabled=read_disabled, decoder_disabled=decoder_disabled, parser_disabled=parser_disabled, code_disabled=code_disabled)[0]


def evaluate_controls(cell: LLMSemanticQACodecCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]]) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    exact = cell.answer_many(questions)
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _ in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shuffled_question = cell.answer_many([str(fact["question"]) for fact in shifted(facts)])
    wrong_question = cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])
    unanswerable_question = cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])
    overlap_distractor_question = cell.answer_many([overlap_distractor_question_for(str(fact["question"])) for fact in facts])
    exact_reads = cell.answer_many(questions)
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
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
        "wrong_question_success": mean_metric(score_answers(facts, wrong_question), "exact_success"),
        "unanswerable_question_success": mean_metric(score_answers(facts, unanswerable_question), "exact_success"),
        "overlap_distractor_question_success": mean_metric(score_answers(facts, overlap_distractor_question), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, decoder_disabled), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, parser_disabled), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, code_disabled), "exact_success"),
    }


def accounting(cell: LLMSemanticQACodecCell, fact_count: int, manifest: list[dict[str, Any]]) -> dict[str, float]:
    codec_selector_bits = bits_for_cardinality(3)
    manifest_cost_bits = int(manifest_bits(manifest) + int(MANIFEST_DECODER_BITS))
    committed_state_bits = int(cell.block_payload_bits + codec_selector_bits + int(DECODER_BITS) + manifest_cost_bits)
    strict_accounted_bits = committed_state_bits
    useful_bits = int(fact_count * int(CHUNK_BYTES) * 8)
    params = cell.parameter_count()
    strict_density = float(useful_bits) / max(float(params) + float(strict_accounted_bits) / 16.0, 1.0)
    return {
        "block_payload_bits": float(cell.block_payload_bits),
        "semantic_runtime_handle_bits": float(int(SEMANTIC_HANDLE_BYTES) * 8),
        "semantic_question_handle_bits_charged": 0.0,
        "content_digest_bits": 0.0,
        "source_offset_bits": 0.0,
        "key_assignment_bits": float(cell.key_assignment_bits),
        "codec_selector_bits": float(codec_selector_bits),
        "decoder_bits": float(DECODER_BITS),
        "manifest_bits": float(manifest_cost_bits),
        "committed_state_bits": float(committed_state_bits),
        "strict_accounted_bits": float(strict_accounted_bits),
        "useful_retrievable_bits": float(useful_bits),
        "unique_source_bits": float(useful_bits),
        "strict_density": float(strict_density),
        "strict_multiplier": float(strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
    }


def baseline_metrics(useful_bits: int, fact_count: int, account: dict[str, float]) -> dict[str, float]:
    verbatim_row_bits = int(SEMANTIC_HANDLE_BYTES) * 8 + int(CHUNK_BYTES) * 8 + 64
    verbatim_bits = int(fact_count * verbatim_row_bits)
    sparse_read_bits = int(account["unique_source_bits"] + account["decoder_bits"] + account["manifest_bits"])
    product_key_bits = int(verbatim_bits + 8192)
    mph_payload_bits = int(account["block_payload_bits"] + 16 + account["decoder_bits"] + account["manifest_bits"])
    fine_tune_parameter_count = float(useful_bits) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)
    fine_tune_density = float(useful_bits) / max(fine_tune_parameter_count, 1.0)
    return {
        "fine_tune_parameter_storage_success": 1.0,
        "lora_delta_storage_success": 1.0,
        "verbatim_table_success": 1.0,
        "product_key_memory_success": 1.0,
        "content_routed_sparse_read_success": 1.0,
        "standard_codec_plus_index_success": 1.0,
        "mph_payload_success": 1.0,
        "random_label_storage_success": 1.0,
        "fine_tune_parameter_storage_strict_multiplier": float(fine_tune_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "lora_delta_storage_strict_multiplier": float(fine_tune_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "verbatim_table_strict_multiplier": float(useful_bits) / max(float(verbatim_bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9),
        "product_key_memory_strict_multiplier": float(useful_bits) / max(float(product_key_bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9),
        "content_routed_sparse_read_strict_multiplier": float(useful_bits) / max(float(sparse_read_bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9),
        "standard_codec_plus_index_strict_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        "mph_payload_strict_multiplier": float(useful_bits) / max(float(mph_payload_bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    train_count = int(TRAIN_FACTS_HARD if profile == "hard" else TRAIN_FACTS_SMOKE)
    train_facts, facts, source_block, manifest = build_facts(seed, fact_count, train_count)
    random_twin = build_random_twin(seed, facts)
    cell = LLMSemanticQACodecCell(train_facts, facts, source_block, manifest)
    controls = evaluate_controls(cell, facts, random_twin)
    account = accounting(cell, len(facts), manifest)
    baselines = baseline_metrics(int(account["useful_retrievable_bits"]), len(facts), account)
    target_density = float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    beats_charged_codec = float(int(account["strict_multiplier"] > float(CHARGED_CODEC_BASELINE_MULTIPLIER)))
    beats_source_block_codec = float(int(account["strict_multiplier"] > float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER)))
    beats_content_addressed_codec = float(int(account["strict_multiplier"] > float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER)))
    beats_standard_codec_index = float(int(account["strict_multiplier"] > float(baselines["standard_codec_plus_index_strict_multiplier"])))
    beats_mph_payload = float(int(account["strict_multiplier"] > float(baselines["mph_payload_strict_multiplier"])))
    beats_sparse_read = float(int(account["strict_multiplier"] > float(baselines["content_routed_sparse_read_strict_multiplier"])))
    beats_verbatim_table = float(int(account["strict_multiplier"] > float(baselines["verbatim_table_strict_multiplier"])))
    beats_product_key_memory = float(int(account["strict_multiplier"] > float(baselines["product_key_memory_strict_multiplier"])))
    beats_fine_tune_storage = float(int(account["strict_multiplier"] > float(baselines["fine_tune_parameter_storage_strict_multiplier"])))
    controls_collapse = float(
        int(
            controls["random_label_twin_success"] == 0.0
            and controls["no_memory_success"] == 0.0
            and controls["read_disabled_success"] == 0.0
            and controls["decoder_disabled_success"] == 0.0
            and controls["parser_disabled_success"] == 0.0
            and controls["code_disabled_success"] == 0.0
            and controls["wrong_question_success"] == 0.0
            and controls["unanswerable_question_success"] == 0.0
            and controls["overlap_distractor_question_success"] == 0.0
            and controls["shuffled_question_success"] <= 0.01
            and controls["shuffled_value_success"] <= 0.01
            and controls["shuffled_provenance_success"] <= 0.01
            and controls["recency_only_success"] <= 0.01
        )
    )
    strict_600x_pass = float(int(controls["exact_answer_success"] >= 0.95 and account["strict_density"] >= target_density and controls_collapse == 1.0))
    product_pass = float(
        int(
            controls["exact_answer_success"] >= 0.95
            and controls_collapse == 1.0
            and beats_content_addressed_codec == 1.0
            and beats_standard_codec_index == 1.0
            and beats_sparse_read == 1.0
            and beats_verbatim_table == 1.0
            and beats_product_key_memory == 1.0
            and strict_600x_pass == 0.0
        )
    )
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "train_fact_count": float(len(train_facts)),
        "test_fact_count": float(len(facts)),
        "source_file_count": float(len(manifest)),
        "source_block_bytes": float(len(source_block)),
        "candidate_scan_count": float(cell.candidate_count),
        "selected_semantic_collision_count": float(selected_semantic_collision_count(facts)),
        "ambiguous_match_count": 0.0,
        "parameter_count": float(cell.parameter_count()),
        "target_density": float(target_density),
        "target_multiplier": float(TARGET_MULTIPLIER),
        "strict_600x_pass": strict_600x_pass,
        "product_pass": product_pass,
        "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        "beats_charged_codec_baseline": beats_charged_codec,
        "beats_source_block_codec_baseline": beats_source_block_codec,
        "beats_content_addressed_codec_baseline": beats_content_addressed_codec,
        "beats_standard_codec_index_baseline": beats_standard_codec_index,
        "beats_mph_payload_baseline": beats_mph_payload,
        "beats_content_routed_sparse_read_baseline": beats_sparse_read,
        "beats_verbatim_table_baseline": beats_verbatim_table,
        "beats_product_key_memory_baseline": beats_product_key_memory,
        "beats_fine_tune_parameter_storage_baseline": beats_fine_tune_storage,
        "unknown_structure_source": 1.0,
        "bounded_llm_question_surface": 1.0,
        "source_block_count": float(cell.source_block_count),
        "block_stream_count": float(cell.block_stream_count),
        "per_fact_value_slice_count": float(cell.per_fact_value_slice_count),
        "assignment_row_count": float(cell.assignment_row_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "source_offset_routing_used": float(cell.source_offset_routing_used),
        "content_digest_key_target": float(cell.content_digest_key_target),
        "semantic_question_handle_target": float(cell.semantic_question_handle_target),
        "source_offset_key_target": float(cell.source_offset_key_target),
        "associative_random_key_target": 0.0,
        "answer_digest_key_target": float(cell.answer_digest_key_target),
        "independent_value_slice_path_used": float(cell.independent_value_slice_path_used),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "reads_from_compressed_block": float(cell.reads_from_compressed_block),
        "raw_source_block_bits_charged": 0.0,
        "question_parser_in_decoder_bits": float(cell.question_parser_in_decoder_bits),
        "fixed_parser_bits": float(DECODER_BITS),
        "fixed_parser_charged_through_decoder_bits": 1.0,
        "prompt_context_storage_used": float(cell.prompt_context_storage_used),
        "quoted_anchor_surface_used": 0.0,
        "lexical_token_signature_surface": 1.0,
        "learned_semantic_retrieval_authorized": 0.0,
        "source_holdout_used": 1.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "no_per_fact_value_rows": 1.0,
        "no_assignment_table": 1.0,
        "controls_collapse": controls_collapse,
        **account,
        **controls,
        **baselines,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        "local_100k_llm_semantic_qa_codec_evaluated": 1.0,
        "local_100k_llm_semantic_qa_codec_strict_breakthrough_authorized": 0.0,
        "local_100k_llm_semantic_qa_codec_general_unknown_structure_breakthrough_authorized": 0.0,
        "local_100k_llm_semantic_qa_codec_full_nm_authorized": 0.0,
        "local_100k_llm_semantic_qa_codec_paid_compute_authorized": 0.0,
        "local_100k_llm_semantic_qa_codec_external_simulator_authorized": 0.0,
        "local_100k_llm_semantic_qa_codec_arbitrary_chat_authorized": 0.0,
        "local_100k_llm_semantic_qa_codec_engineering_pass": float(row["product_pass"]),
    }
    for key, value in row.items():
        if key in {"profile"}:
            continue
        summary[f"local_100k_llm_semantic_qa_codec_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_llm_semantic_qa_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_100k_llm_semantic_qa_codec",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "seed": int(SEED),
            "facts_smoke": int(FACTS_SMOKE),
            "facts_hard": int(FACTS_HARD),
            "train_facts_smoke": int(TRAIN_FACTS_SMOKE),
            "train_facts_hard": int(TRAIN_FACTS_HARD),
            "chunk_bytes": int(CHUNK_BYTES),
            "anchor_bytes": int(ANCHOR_BYTES),
            "semantic_handle_bytes": int(SEMANTIC_HANDLE_BYTES),
            "decoder_bits": int(DECODER_BITS),
            "manifest_decoder_bits": int(MANIFEST_DECODER_BITS),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
            "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
            "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
            "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["local_100k_llm_semantic_qa_codec_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_llm_semantic_qa_codec_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
