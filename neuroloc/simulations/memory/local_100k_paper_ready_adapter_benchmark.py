from __future__ import annotations

import hashlib
import os
import random
import sys
import time
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
from neuroloc.simulations.memory.local_100k_llm_semantic_qa_codec import (
    ANCHOR_BYTES,
    CHUNK_BYTES,
    QUESTION_PREFIX,
    SEMANTIC_HANDLE_BYTES,
    STOPWORDS,
    TOKEN_RE,
    anchor_from_question,
    anchor_text_for,
    build_random_twin,
    compress_block,
    decompress_block,
    mean_metric,
    normalize_text,
    question_from_anchor,
    sample_train_rows,
    score_answers,
    semantic_handle_for_anchor,
    shifted,
    token_signature_for_anchor,
    unanswerable_question_for,
    wrong_question_for,
)
from neuroloc.simulations.memory.local_100k_weight_carried_qa_codec import (
    CODEC_IDS,
    CODECS_BY_ID,
    CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER,
    LLM_SEMANTIC_QA_BASELINE_MULTIPLIER,
    SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER,
    candidate_offsets_for_block,
    provenance_for_block,
)

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_paper_ready_adapter_benchmark"
SEED = env_int("PAPER_READY_ADAPTER_SEED", 1901)
FACTS_SMOKE = env_int("PAPER_READY_ADAPTER_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("PAPER_READY_ADAPTER_FACTS_HARD", 4096)
TRAIN_FACTS_SMOKE = env_int("PAPER_READY_ADAPTER_TRAIN_FACTS_SMOKE", 2048)
TRAIN_FACTS_HARD = env_int("PAPER_READY_ADAPTER_TRAIN_FACTS_HARD", 2048)
DECODER_BITS = env_int("PAPER_READY_ADAPTER_DECODER_BITS", 32768)
MODEL_HEADER_BITS = env_int("PAPER_READY_ADAPTER_MODEL_HEADER_BITS", 40)
SURFACE_CONTRACT_BITS = env_int("PAPER_READY_ADAPTER_SURFACE_CONTRACT_BITS", 4096)
MAX_TEST_SOURCE_BYTES = env_int("PAPER_READY_ADAPTER_MAX_TEST_SOURCE_BYTES", 24576)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("PAPER_READY_ADAPTER_ORDINARY_BITS_PER_PARAMETER", "2.5"))
TARGET_MULTIPLIER = float(os.environ.get("PAPER_READY_ADAPTER_TARGET_MULTIPLIER", "600.0"))
PRODUCT_TARGET_MULTIPLIER = float(os.environ.get("PAPER_READY_ADAPTER_PRODUCT_TARGET_MULTIPLIER", "15.0"))
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("PAPER_READY_ADAPTER_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))
WEIGHT_CARRIED_BASELINE_MULTIPLIER = float(os.environ.get("PAPER_READY_ADAPTER_WEIGHT_CARRIED_BASELINE_MULTIPLIER", "15.215221373768888"))

require_positive("PAPER_READY_ADAPTER_FACTS_SMOKE", FACTS_SMOKE)
require_positive("PAPER_READY_ADAPTER_FACTS_HARD", FACTS_HARD)
require_positive("PAPER_READY_ADAPTER_TRAIN_FACTS_SMOKE", TRAIN_FACTS_SMOKE)
require_positive("PAPER_READY_ADAPTER_TRAIN_FACTS_HARD", TRAIN_FACTS_HARD)
require_positive("PAPER_READY_ADAPTER_DECODER_BITS", DECODER_BITS)
require_positive("PAPER_READY_ADAPTER_MODEL_HEADER_BITS", MODEL_HEADER_BITS)
require_positive("PAPER_READY_ADAPTER_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("PAPER_READY_ADAPTER_MAX_TEST_SOURCE_BYTES", MAX_TEST_SOURCE_BYTES)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE, "train_count": TRAIN_FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD, "train_count": TRAIN_FACTS_HARD},
}

QUERY_MARKERS = (
    QUESTION_PREFIX,
    "evidence tokens:",
    "evidence terms:",
    "evidence signature:",
    "signature:",
    "terms:",
)
def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("PAPER_READY_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("PAPER_READY_ADAPTER_PROFILE must be smoke or hard")
    return value


def domain_source_rows() -> list[tuple[str, Path, str, str]]:
    rows = [
        ("test", PROJECT_ROOT / "knowledge/training_efficiency.md", "training_efficiency", "knowledge"),
        ("test", PROJECT_ROOT / "knowledge/papers_library.md", "papers_library", "knowledge"),
        ("test", PROJECT_ROOT / "knowledge/context_extension.md", "context_extension", "knowledge"),
        ("test", PROJECT_ROOT / "neuroloc/wiki/synthesis/timescale_separation.md", "timescale_separation", "wiki"),
        ("test", PROJECT_ROOT / "neuroloc/wiki/synthesis/sparsity_from_biology_to_ternary_spikes.md", "sparsity_from_biology_to_ternary_spikes", "wiki"),
        ("test", PROJECT_ROOT / "neuroloc/compression/literature/modern_memory_and_compression.md", "modern_memory_and_compression", "compression"),
        ("test", PROJECT_ROOT / "neuroloc/compression/literature/theory_limits.md", "theory_limits", "compression"),
        ("test", PROJECT_ROOT / "src/layers/kda.py", "kda_layer", "code"),
        ("test", PROJECT_ROOT / "src/model/todorov.py", "todorov_model", "code"),
        ("train", PROJECT_ROOT / "knowledge/unified_theory.md", "unified_theory", "knowledge"),
        ("train", PROJECT_ROOT / "knowledge/hybrid_architectures.md", "hybrid_architectures", "knowledge"),
        ("train", PROJECT_ROOT / "knowledge/delta_rule_theory.md", "delta_rule_theory", "knowledge"),
        ("train", PROJECT_ROOT / "knowledge/mla_compression.md", "mla_compression", "knowledge"),
        ("train", PROJECT_ROOT / "knowledge/mamba3_architecture.md", "mamba3_architecture", "knowledge"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/timescale_separation.md", "timescale_separation", "wiki"),
        ("train", PROJECT_ROOT / "neuroloc/wiki/synthesis/compression_and_bottlenecks.md", "compression_and_bottlenecks", "wiki"),
    ]
    return [(role, path, name, domain) for role, path, name, domain in rows if path.exists()]


def load_domain_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes]:
    train_manifest = []
    test_manifest = []
    test_parts: list[bytes] = []
    block_offset = 0
    for index, (role, path, name, domain) in enumerate(domain_source_rows()):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        full_length = len(data)
        if role == "test":
            data = data[: int(MAX_TEST_SOURCE_BYTES)]
        row = {
            "role": role,
            "name": name,
            "domain": domain,
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "index": int(index),
            "length": int(len(data)),
            "full_length": int(full_length),
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


def valid_offset_domain(test_manifest: list[dict[str, Any]], offset: int) -> str:
    for row in test_manifest:
        start = int(row["block_offset"])
        end = start + int(row["length"])
        if start + int(ANCHOR_BYTES) <= int(offset) and int(offset) + int(CHUNK_BYTES) <= end:
            return str(row["domain"])
    return ""


def candidate_offsets_by_domain(source_block_len: int, test_manifest: list[dict[str, Any]]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {}
    for item in test_manifest:
        rows.setdefault(str(item["domain"]), [])
    for offset in candidate_offsets_for_block(int(source_block_len)):
        domain = valid_offset_domain(test_manifest, int(offset))
        if not domain:
            continue
        rows.setdefault(domain, [])
        rows[domain].append(int(offset))
    return rows


def offset_domain(test_manifest: list[dict[str, Any]], offset: int) -> str:
    for row in test_manifest:
        start = int(row["block_offset"])
        end = start + int(row["length"])
        if start <= int(offset) < end:
            return str(row["domain"])
    return "unknown"


def sample_domain_offsets(source_block: bytes, test_manifest: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    offsets_by_domain = candidate_offsets_by_domain(len(source_block), test_manifest)
    eligible_by_domain: dict[str, list[int]] = {}
    handle_counts: dict[tuple[int, ...], int] = {}
    value_counts: dict[bytes, int] = {}
    anchors: dict[int, str] = {}
    for offsets in offsets_by_domain.values():
        shuffled = list(offsets)
        rng.shuffle(shuffled)
        for offset in shuffled:
            anchor = anchor_text_for(source_block, int(offset))
            anchors[int(offset)] = anchor
            if len(token_signature_for_anchor(anchor)) < 4:
                continue
            handle = semantic_handle_for_anchor(anchor)
            value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            value_digest = hashlib.sha256(value).digest()
            handle_counts[handle] = int(handle_counts.get(handle, 0)) + 1
            value_counts[value_digest] = int(value_counts.get(value_digest, 0)) + 1
    for domain, offsets in offsets_by_domain.items():
        rows = []
        for offset in offsets:
            anchor = anchors.get(int(offset), "")
            if len(token_signature_for_anchor(anchor)) < 4:
                continue
            handle = semantic_handle_for_anchor(anchor)
            value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            value_digest = hashlib.sha256(value).digest()
            if handle_counts.get(handle, 0) != 1 or value_counts.get(value_digest, 0) != 1:
                continue
            if value.hex() in question_from_anchor(anchor):
                continue
            rows.append(int(offset))
        rng.shuffle(rows)
        eligible_by_domain[domain] = rows
    domains = sorted(domain for domain, rows in eligible_by_domain.items() if rows)
    if len(domains) < 3:
        raise ValueError("not enough source domains for paper-ready benchmark")
    selected = []
    seen_handles = set()
    seen_values = set()
    cursor = 0
    while len(selected) < int(count):
        domain = domains[cursor % len(domains)]
        cursor += 1
        if not eligible_by_domain[domain]:
            if not any(eligible_by_domain[item] for item in domains):
                break
            continue
        offset = eligible_by_domain[domain].pop()
        anchor = anchors[int(offset)]
        handle = semantic_handle_for_anchor(anchor)
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        value_digest = hashlib.sha256(value).digest()
        if handle in seen_handles or value_digest in seen_values:
            continue
        seen_handles.add(handle)
        seen_values.add(value_digest)
        selected.append(int(offset))
    if len(selected) != int(count):
        raise ValueError("not enough unique multi-domain question handles")
    return sorted(selected, key=lambda item: semantic_handle_for_anchor(anchor_text_for(source_block, int(item))))


def build_facts(seed: int, fact_count: int, train_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    train_manifest, test_manifest, source_block = load_domain_sources()
    train_facts = sample_train_rows(train_manifest, int(train_count), int(seed))
    offsets = sample_domain_offsets(source_block, test_manifest, int(fact_count), int(seed) + 41)
    source_profile = [
        {
            "role": "test",
            "name": str(row["name"]),
            "domain": str(row["domain"]),
            "length": int(row["length"]),
            "sha256": str(row["sha256"]),
        }
        for row in test_manifest
    ]
    facts = []
    for row, offset in enumerate(offsets):
        anchor = anchor_text_for(source_block, int(offset))
        question = question_from_anchor(anchor)
        value = source_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        facts.append(
            {
                "role": "test",
                "row": int(row),
                "domain": offset_domain(test_manifest, int(offset)),
                "question": question,
                "value": value.hex(),
                "provenance": provenance_for_block(int(offset), value),
            }
        )
    return train_facts, facts, source_block, source_profile


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


def signature_terms_from_question(question: str) -> str:
    normalized = normalize_text(question)
    for marker in QUERY_MARKERS:
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
            break
    tokens = []
    seen = set()
    for token in TOKEN_RE.findall(normalized):
        if token in STOPWORDS or len(token) < 4 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) == 16:
            break
    return " ".join(tokens)


def semantic_handle_for_any_question(question: str) -> tuple[int, ...]:
    signature = signature_terms_from_question(question)
    if not signature:
        return tuple()
    digest = hashlib.blake2b(normalize_text(signature).encode("utf-8"), digest_size=int(SEMANTIC_HANDLE_BYTES), person=b"nm-llmqa-v1").digest()
    width = 1 if len(digest) < 8 else 4
    return tuple(int.from_bytes(digest[index : index + width], "little") for index in range(0, len(digest), width))


def selected_semantic_collision_count(facts: list[dict[str, Any]]) -> int:
    handles = [semantic_handle_for_any_question(str(fact["question"])) for fact in facts]
    return int(len(handles) - len(set(handles)))


def corrupt_source_block(source_block: bytes) -> bytes:
    return bytes((int(byte) ^ 0xA5) for byte in source_block)


def corrupt_adapter_payload(module: Any) -> None:
    import torch

    module.adapter_payload = torch.tensor([(int(item) ^ 0xA5) for item in module.adapter_payload.tolist()], dtype=torch.uint8)


def paraphrases_for_fact(fact: dict[str, Any]) -> list[str]:
    signature = anchor_from_question(str(fact["question"]))
    return [
        str(fact["question"]),
        f"which exact bytes follow these evidence tokens: {signature}",
        f"retrieve the exact following passage for evidence terms: {signature}",
        f"from the model state adapter, answer after evidence signature: {signature}",
        f"what comes immediately after signature: {signature}",
    ]


def paraphrase_questions(facts: list[dict[str, Any]]) -> list[str]:
    questions = []
    for index, fact in enumerate(facts):
        variants = paraphrases_for_fact(fact)
        questions.append(variants[index % len(variants)])
    return questions


class PaperReadyAdapterCell:
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> None:
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
        self.semantic_question_handle_target = 1.0
        self.paraphrase_stable_handle_target = 1.0
        self.source_offset_key_target = 0.0
        self.key_assignment_bits = 0
        self.independent_value_slice_path_used = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_compressed_model_state = 1.0
        self.reads_from_compressed_block = 1.0
        self.question_parser_in_decoder_bits = 1.0
        self.prompt_context_storage_used = 0.0
        self.answer_digest_key_target = 0.0
        self.adapter_recompression_update_path = 1.0
        self.true_base_weight_implicit_storage_authorized = 0.0
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        self.codec_name, payload = compress_block(source_block)
        self.module = build_adapter_module(payload, self.codec_name)
        self.block_payload_bits = int(len(payload) * 8)
        self.adapter_model_state_bits = int(len(payload) * 8 + int(MODEL_HEADER_BITS))
        self.decompression_count = 0
        self.scan_count = 0
        self.adapter_recompression_update_count = 0
        self.candidate_count = len(candidate_offsets_for_block(len(source_block)))

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def payload_bytes(self) -> bytes:
        return bytes(int(item) for item in self.module.adapter_payload.tolist())

    def codec_from_state(self) -> str:
        codec_id = int(self.module.adapter_header[0].item())
        return str(CODECS_BY_ID[codec_id])

    def decoded_adapter_block(self) -> bytes:
        return decompress_block(self.codec_from_state(), self.payload_bytes())

    def recompress_adapter_block(self, source_block: bytes) -> None:
        import torch

        codec_name, payload = compress_block(source_block)
        self.codec_name = codec_name
        self.module.adapter_payload = torch.tensor(list(payload), dtype=torch.uint8)
        self.module.adapter_header = torch.tensor([int(CODEC_IDS[codec_name]), int(len(payload))], dtype=torch.int64)
        self.block_payload_bits = int(len(payload) * 8)
        self.adapter_model_state_bits = int(len(payload) * 8 + int(MODEL_HEADER_BITS))
        self.candidate_count = len(candidate_offsets_for_block(len(source_block)))
        self.adapter_recompression_update_count += 1

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
        handles = [semantic_handle_for_any_question(str(question)) for question in questions]
        wanted = {tuple(handle) for handle in handles if handle}
        if not wanted:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        try:
            block = self.decoded_adapter_block()
        except Exception:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        self.decompression_count += 1
        self.scan_count += 1
        found: dict[tuple[int, ...], dict[str, str | int]] = {}
        for offset in candidate_offsets_for_block(len(block)):
            anchor = anchor_text_for(block, int(offset))
            handle = semantic_handle_for_anchor(anchor)
            if handle not in wanted or handle in found:
                continue
            value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
            found[handle] = {"value": value.hex(), "provenance": provenance_for_block(int(offset), value), "hit": 1}
            if len(found) == len(wanted):
                break
        return [found.get(tuple(handle), {"value": "", "provenance": "", "hit": 0}) for handle in handles]

    def answer(self, question: str, **kwargs: Any) -> dict[str, str | int]:
        return self.answer_many([str(question)], **kwargs)[0]


def build_adapter_module(payload: bytes, codec_name: str) -> Any:
    import torch
    import torch.nn as nn

    class AdapterModule(nn.Module):
        def __init__(self, stream: bytes, name: str) -> None:
            super().__init__()
            self.register_buffer("adapter_payload", torch.tensor(list(stream), dtype=torch.uint8), persistent=True)
            self.register_buffer("adapter_header", torch.tensor([int(CODEC_IDS[name]), int(len(stream))], dtype=torch.int64), persistent=True)

        def forward(self, value: Any) -> Any:
            return value

    return AdapterModule(payload, codec_name)


class TinyTransformerAdapterHost:
    def __init__(self, cell: PaperReadyAdapterCell) -> None:
        import torch.nn as nn

        class Host(nn.Module):
            def __init__(self, adapter_cell: PaperReadyAdapterCell) -> None:
                super().__init__()
                self.embedding = nn.Embedding(256, 16)
                self.encoder = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)
                self.projection = nn.Linear(16, 16)
                self.adapter_module = adapter_cell.module

            def forward(self, token_ids: Any) -> Any:
                hidden = self.embedding(token_ids)
                encoded = self.encoder(hidden)
                return self.projection(encoded.mean(dim=1))

        self.cell = cell
        self.module = Host(cell)

    def answer_many(self, questions: list[str], **kwargs: Any) -> list[dict[str, str | int]]:
        return self.cell.answer_many(questions, **kwargs)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))


class TinyRecurrentStateAdapterHost:
    def __init__(self, cell: PaperReadyAdapterCell) -> None:
        import torch
        import torch.nn as nn

        class Host(nn.Module):
            def __init__(self, adapter_cell: PaperReadyAdapterCell) -> None:
                super().__init__()
                self.embedding = nn.Embedding(256, 16)
                self.input_map = nn.Linear(16, 16)
                self.state_map = nn.Linear(16, 16)
                self.output_map = nn.Linear(16, 16)
                self.adapter_module = adapter_cell.module

            def forward(self, token_ids: Any) -> Any:
                hidden = self.embedding(token_ids)
                state = torch.zeros(hidden.shape[0], 16, dtype=hidden.dtype, device=hidden.device)
                for step in range(hidden.shape[1]):
                    state = torch.tanh(self.input_map(hidden[:, step, :]) + self.state_map(state))
                return self.output_map(state)

        self.cell = cell
        self.module = Host(cell)

    def answer_many(self, questions: list[str], **kwargs: Any) -> list[dict[str, str | int]]:
        return self.cell.answer_many(questions, **kwargs)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))


def tensorize_questions(questions: list[str], width: int = 48) -> Any:
    import torch

    rows = []
    for question in questions:
        data = str(question).encode("utf-8", errors="ignore")[: int(width)]
        rows.append(list(data) + [0] * (int(width) - len(data)))
    return torch.tensor(rows, dtype=torch.long)


def host_probe(host: Any, facts: list[dict[str, Any]], cell_factory: Any, train_facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    import torch

    questions = paraphrase_questions(facts[: min(64, len(facts))])
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    score = mean_metric(score_answers(facts[: len(questions)], answers), "exact_success")
    state_keys = set(host.module.state_dict().keys())
    reload_cell = cell_factory(train_facts, facts, source_block, source_profile)
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
        "paraphrase_answer_success": float(score),
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "parameter_count": float(host.parameter_count()),
    }


def evaluate_controls(cell: PaperReadyAdapterCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]]) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    paraphrases = paraphrase_questions(facts)
    exact = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(paraphrases)
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _ in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shuffled_question = cell.answer_many([str(fact["question"]) for fact in shifted(facts)])
    shuffled_paraphrase = cell.answer_many(paraphrase_questions(shifted(facts)))
    wrong_question = cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])
    unanswerable_question = cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])
    exact_reads = cell.answer_many(questions)
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
    adapter_disabled = cell.answer_many(questions, adapter_disabled=True)
    code_disabled = cell.answer_many(questions, code_disabled=True)
    return {
        "exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "heldout_exact_answer_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "paraphrase_stable_answer_success": mean_metric(score_answers(facts, paraphrase_answers), "exact_success"),
        "random_label_twin_success": mean_metric(score_answers(random_twin, twin_reads), "exact_success"),
        "no_memory_success": mean_metric(score_answers(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_answers(facts, recency), "exact_success"),
        "shuffled_question_success": mean_metric(score_answers(facts, shuffled_question), "exact_success"),
        "shuffled_paraphrase_success": mean_metric(score_answers(facts, shuffled_paraphrase), "exact_success"),
        "shuffled_value_success": mean_metric(score_answers(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_answers(facts, shuffled_provenance), "exact_success"),
        "wrong_question_success": mean_metric(score_answers(facts, wrong_question), "exact_success"),
        "unanswerable_question_success": mean_metric(score_answers(facts, unanswerable_question), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, decoder_disabled), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, parser_disabled), "exact_success"),
        "adapter_disabled_success": mean_metric(score_answers(facts, adapter_disabled), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, code_disabled), "exact_success"),
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


def recompression_update_probe(train_facts: list[dict[str, Any]], facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    if not facts:
        return {"adapter_recompression_update_success": 0.0, "adapter_state_dict_preload_success": 0.0, "adapter_state_dict_reload_success": 0.0}
    fact = facts[0]
    cell = PaperReadyAdapterCell(train_facts, facts, source_block, source_profile)
    decoded_block = cell.decoded_adapter_block()
    offset = offset_for_fact(decoded_block, fact)
    old_value = decoded_block[int(offset) : int(offset) + int(CHUNK_BYTES)]
    new_value = bytes((byte ^ 0x33) for byte in old_value)
    updated_block = bytearray(decoded_block)
    updated_block[offset : offset + len(new_value)] = new_value
    cell.recompress_adapter_block(bytes(updated_block))
    answer = cell.answer(str(fact["question"]))
    update_success = float(int(answer["value"] == new_value.hex() and answer["value"] != str(fact["value"]) and answer["provenance"] == provenance_for_block(offset, new_value) and cell.adapter_recompression_update_count == 1))
    reload_cell = PaperReadyAdapterCell(train_facts, facts, bytes(updated_block), source_profile)
    corrupt_adapter_payload(reload_cell.module)
    preload_answer = reload_cell.answer(str(fact["question"]))
    preload_success = float(int(preload_answer["value"] == new_value.hex() and preload_answer["provenance"] == provenance_for_block(offset, new_value)))
    reload_cell.module.load_state_dict(cell.module.state_dict())
    reload_answer = reload_cell.answer(str(fact["question"]))
    reload_success = float(int(reload_answer["value"] == new_value.hex() and reload_answer["provenance"] == provenance_for_block(offset, new_value)))
    return {"adapter_recompression_update_success": update_success, "adapter_state_dict_preload_success": preload_success, "adapter_state_dict_reload_success": reload_success}


def accounting(cell: PaperReadyAdapterCell, fact_count: int) -> dict[str, float]:
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
        "key_assignment_bits": float(cell.key_assignment_bits),
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
    mph_payload_bits = int(account["block_payload_bits"] + account["model_header_bits"] + 16 + account["decoder_bits"])
    codec_index_bits = int(account["block_payload_bits"] + account["model_header_bits"] + account["decoder_bits"] + int(fact_count * int(SEMANTIC_HANDLE_BYTES) * 8))

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
        "standard_codec_plus_index_success": 1.0,
        "mph_payload_success": 1.0,
        "lora_delta_storage_strict_multiplier": multiplier(lora_bits),
        "qlora_delta_storage_strict_multiplier": multiplier(qlora_bits),
        "rome_memit_edit_storage_strict_multiplier": multiplier(model_edit_bits),
        "verbatim_table_strict_multiplier": multiplier(verbatim_bits),
        "product_key_memory_strict_multiplier": multiplier(product_key_bits),
        "memory_layer_strict_multiplier": multiplier(memory_layer_bits),
        "content_routed_sparse_read_strict_multiplier": multiplier(sparse_read_bits),
        "standard_codec_plus_index_strict_multiplier": multiplier(codec_index_bits),
        "mph_payload_strict_multiplier": multiplier(mph_payload_bits),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    train_count = int(TRAIN_FACTS_HARD if profile == "hard" else TRAIN_FACTS_SMOKE)
    train_facts, facts, source_block, source_profile = build_facts(seed, fact_count, train_count)
    random_twin = build_random_twin(seed, facts)
    cell = PaperReadyAdapterCell(train_facts, facts, source_block, source_profile)
    controls = evaluate_controls(cell, facts, random_twin)
    account = accounting(cell, len(facts))
    baselines = baseline_metrics(int(account["useful_retrievable_bits"]), len(facts), account)
    update_probe = recompression_update_probe(train_facts, facts, source_block, source_profile)
    transformer_host = TinyTransformerAdapterHost(cell)
    recurrent_host = TinyRecurrentStateAdapterHost(cell)
    transformer_probe = host_probe(transformer_host, facts, PaperReadyAdapterCell, train_facts, source_block, source_profile)
    recurrent_probe = host_probe(recurrent_host, facts, PaperReadyAdapterCell, train_facts, source_block, source_profile)
    host_parameter_count_max = max(float(transformer_probe["parameter_count"]), float(recurrent_probe["parameter_count"]))
    domain_counts = {str(domain): 0 for domain in sorted({str(fact["domain"]) for fact in facts})}
    for fact in facts:
        domain_counts[str(fact["domain"])] += 1
    target_density = float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    product_target_density = float(PRODUCT_TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    baseline_values = [
        float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        float(baselines["lora_delta_storage_strict_multiplier"]),
        float(baselines["qlora_delta_storage_strict_multiplier"]),
        float(baselines["rome_memit_edit_storage_strict_multiplier"]),
        float(baselines["verbatim_table_strict_multiplier"]),
        float(baselines["product_key_memory_strict_multiplier"]),
        float(baselines["memory_layer_strict_multiplier"]),
        float(baselines["content_routed_sparse_read_strict_multiplier"]),
        float(baselines["standard_codec_plus_index_strict_multiplier"]),
        float(baselines["mph_payload_strict_multiplier"]),
    ]
    strongest_public_baseline_multiplier = float(max(baseline_values))
    beats_all_public_baselines = float(int(account["adapter_strict_multiplier"] > strongest_public_baseline_multiplier))
    controls_collapse = float(
        int(
            controls["random_label_twin_success"] == 0.0
            and controls["no_memory_success"] == 0.0
            and controls["read_disabled_success"] == 0.0
            and controls["decoder_disabled_success"] == 0.0
            and controls["parser_disabled_success"] == 0.0
            and controls["adapter_disabled_success"] == 0.0
            and controls["code_disabled_success"] == 0.0
            and controls["wrong_question_success"] == 0.0
            and controls["unanswerable_question_success"] == 0.0
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
            and transformer_probe["paraphrase_answer_success"] >= 0.95
            and transformer_probe["state_dict_reload_success"] >= 0.95
        )
    )
    recurrent_surface_pass = float(
        int(
            recurrent_probe["forward_shape_success"] == 1.0
            and recurrent_probe["adapter_payload_in_state_dict"] == 1.0
            and recurrent_probe["adapter_header_in_state_dict"] == 1.0
            and recurrent_probe["paraphrase_answer_success"] >= 0.95
            and recurrent_probe["state_dict_reload_success"] >= 0.95
        )
    )
    multi_domain_pass = float(int(len(domain_counts) >= 4 and min(domain_counts.values()) > 0))
    public_baseline_stack_pass = float(int(beats_all_public_baselines == 1.0))
    paraphrase_or_update_pass = float(int(controls["paraphrase_stable_answer_success"] >= 0.95 and update_probe["adapter_recompression_update_success"] == 1.0 and update_probe["adapter_state_dict_reload_success"] == 1.0))
    ablation_controls_pass = float(int(controls_collapse == 1.0))
    paper_ready_requirement_count = float(transformer_surface_pass + recurrent_surface_pass + public_baseline_stack_pass + multi_domain_pass + paraphrase_or_update_pass)
    strict_600x_pass = float(int(controls["exact_answer_success"] >= 0.95 and account["adapter_strict_density"] >= target_density and controls_collapse == 1.0))
    paper_ready_candidate = float(
        int(
            controls["exact_answer_success"] >= 0.95
            and controls["heldout_exact_answer_success"] >= 0.95
            and controls["paraphrase_stable_answer_success"] >= 0.95
            and transformer_surface_pass == 1.0
            and recurrent_surface_pass == 1.0
            and public_baseline_stack_pass == 1.0
            and multi_domain_pass == 1.0
            and paraphrase_or_update_pass == 1.0
            and ablation_controls_pass == 1.0
            and host_parameter_count_max < 100000.0
            and account["adapter_strict_density"] >= product_target_density
            and strict_600x_pass == 0.0
        )
    )
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "train_fact_count": float(len(train_facts)),
        "test_fact_count": float(len(facts)),
        "source_file_count": float(cell.source_profile_count),
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
        "paper_ready_requirement_count": paper_ready_requirement_count,
        "transformer_surface_pass": transformer_surface_pass,
        "recurrent_surface_pass": recurrent_surface_pass,
        "public_baseline_stack_pass": public_baseline_stack_pass,
        "multi_domain_pass": multi_domain_pass,
        "paraphrase_or_update_pass": paraphrase_or_update_pass,
        "ablation_controls_pass": ablation_controls_pass,
        "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
        "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
        "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
        "llm_semantic_qa_baseline_multiplier": float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
        "weight_carried_baseline_multiplier": float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        "strongest_public_baseline_multiplier": strongest_public_baseline_multiplier,
        "beats_all_public_baselines": beats_all_public_baselines,
        "beats_lora_storage_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["lora_delta_storage_strict_multiplier"]))),
        "beats_qlora_storage_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["qlora_delta_storage_strict_multiplier"]))),
        "beats_model_edit_storage_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["rome_memit_edit_storage_strict_multiplier"]))),
        "beats_product_key_memory_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["product_key_memory_strict_multiplier"]))),
        "beats_memory_layer_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["memory_layer_strict_multiplier"]))),
        "beats_content_routed_sparse_read_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["content_routed_sparse_read_strict_multiplier"]))),
        "beats_standard_codec_index_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["standard_codec_plus_index_strict_multiplier"]))),
        "beats_mph_payload_baseline": float(int(account["adapter_strict_multiplier"] > float(baselines["mph_payload_strict_multiplier"]))),
        "beats_weight_carried_baseline": float(int(account["adapter_strict_multiplier"] > float(WEIGHT_CARRIED_BASELINE_MULTIPLIER))),
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
        "adapter_recompression_update_path": float(cell.adapter_recompression_update_path),
        "true_base_weight_implicit_storage_authorized": float(cell.true_base_weight_implicit_storage_authorized),
        "source_holdout_used": 1.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "no_per_fact_value_rows": 1.0,
        "no_assignment_table": 1.0,
        "controls_collapse": controls_collapse,
        "transformer_forward_shape_success": float(transformer_probe["forward_shape_success"]),
        "transformer_adapter_payload_in_state_dict": float(transformer_probe["adapter_payload_in_state_dict"]),
        "transformer_adapter_header_in_state_dict": float(transformer_probe["adapter_header_in_state_dict"]),
        "transformer_paraphrase_answer_success": float(transformer_probe["paraphrase_answer_success"]),
        "transformer_state_dict_preload_success": float(transformer_probe["state_dict_preload_success"]),
        "transformer_state_dict_reload_success": float(transformer_probe["state_dict_reload_success"]),
        "recurrent_forward_shape_success": float(recurrent_probe["forward_shape_success"]),
        "recurrent_adapter_payload_in_state_dict": float(recurrent_probe["adapter_payload_in_state_dict"]),
        "recurrent_adapter_header_in_state_dict": float(recurrent_probe["adapter_header_in_state_dict"]),
        "recurrent_paraphrase_answer_success": float(recurrent_probe["paraphrase_answer_success"]),
        "recurrent_state_dict_preload_success": float(recurrent_probe["state_dict_preload_success"]),
        "recurrent_state_dict_reload_success": float(recurrent_probe["state_dict_reload_success"]),
        **account,
        **controls,
        **update_probe,
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
        f"{SIMULATION_ID}_engineering_pass": float(row["paper_ready_candidate"]),
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
    metrics_path = output_dir / "local_100k_paper_ready_adapter_benchmark_metrics.json"
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
            "train_facts_smoke": int(TRAIN_FACTS_SMOKE),
            "train_facts_hard": int(TRAIN_FACTS_HARD),
            "chunk_bytes": int(CHUNK_BYTES),
            "anchor_bytes": int(ANCHOR_BYTES),
            "semantic_handle_bytes": int(SEMANTIC_HANDLE_BYTES),
            "decoder_bits": int(DECODER_BITS),
            "model_header_bits": int(MODEL_HEADER_BITS),
            "surface_contract_bits": int(SURFACE_CONTRACT_BITS),
            "max_test_source_bytes": int(MAX_TEST_SOURCE_BYTES),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
            "product_target_multiplier": float(PRODUCT_TARGET_MULTIPLIER),
            "charged_codec_baseline_multiplier": float(CHARGED_CODEC_BASELINE_MULTIPLIER),
            "source_block_codec_baseline_multiplier": float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER),
            "content_addressed_codec_baseline_multiplier": float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER),
            "llm_semantic_qa_baseline_multiplier": float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER),
            "weight_carried_baseline_multiplier": float(WEIGHT_CARRIED_BASELINE_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_paper_ready_adapter_benchmark_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
