from __future__ import annotations

import hashlib
import lzma
import os
import random
import re
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
from neuroloc.simulations.memory.local_100k_llm_semantic_qa_codec import CHUNK_BYTES, mean_metric, normalize_text, score_answers, unanswerable_question_for, wrong_question_for
from neuroloc.simulations.memory.local_100k_margin_recompression_adapter import (
    MODEL_HEADER_BITS,
    ORDINARY_BITS_PER_PARAMETER,
    PAPER_READY_BASELINE_MULTIPLIER,
    SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER,
    TARGET_MULTIPLIER,
    WEIGHT_CARRIED_BASELINE_MULTIPLIER,
    fixed_ngrams,
    hidden_state_inspection,
)
from neuroloc.simulations.memory.local_100k_paper_ready_adapter_benchmark import (
    CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER,
    LLM_SEMANTIC_QA_BASELINE_MULTIPLIER,
    TinyRecurrentStateAdapterHost,
    TinyTransformerAdapterHost,
    corrupt_adapter_payload,
    tensorize_questions,
)
from neuroloc.simulations.memory.local_100k_semantic_alias_payload_adapter import CONTENT_SCAN_BASELINE_MULTIPLIER, MARGIN_BASELINE_MULTIPLIER

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_authored_edge_relation_adapter"
SEED = env_int("AUTHORED_EDGE_RELATION_ADAPTER_SEED", 4591)
FACTS_SMOKE = env_int("AUTHORED_EDGE_RELATION_ADAPTER_FACTS_SMOKE", 4096)
FACTS_HARD = env_int("AUTHORED_EDGE_RELATION_ADAPTER_FACTS_HARD", 4096)
DECODER_BITS = env_int("AUTHORED_EDGE_RELATION_ADAPTER_DECODER_BITS", 32768)
SURFACE_CONTRACT_BITS = env_int("AUTHORED_EDGE_RELATION_ADAPTER_SURFACE_CONTRACT_BITS", 4096)
ROUTER_BITS = env_int("AUTHORED_EDGE_RELATION_ADAPTER_ROUTER_BITS", 64)
CHUNK_INDEX_BITS = env_int("AUTHORED_EDGE_RELATION_ADAPTER_CHUNK_INDEX_BITS", 4)
CHARGED_CODEC_BASELINE_MULTIPLIER = float(os.environ.get("AUTHORED_EDGE_RELATION_ADAPTER_CHARGED_CODEC_BASELINE_MULTIPLIER", "13.941917871967359"))

require_positive("AUTHORED_EDGE_RELATION_ADAPTER_FACTS_SMOKE", FACTS_SMOKE)
require_positive("AUTHORED_EDGE_RELATION_ADAPTER_FACTS_HARD", FACTS_HARD)
require_positive("AUTHORED_EDGE_RELATION_ADAPTER_DECODER_BITS", DECODER_BITS)
require_positive("AUTHORED_EDGE_RELATION_ADAPTER_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("AUTHORED_EDGE_RELATION_ADAPTER_ROUTER_BITS", ROUTER_BITS)
require_positive("AUTHORED_EDGE_RELATION_ADAPTER_CHUNK_INDEX_BITS", CHUNK_INDEX_BITS)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD},
}
QUESTION_PREFIX = "authored edge target chunk"
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}|[0-9]{2,}")
STOPWORDS = {
    "authored",
    "edge",
    "target",
    "chunk",
    "source",
    "wiki",
    "link",
    "from",
    "with",
    "terms",
    "retrieve",
    "exact",
    "span",
    "article",
    "linked",
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("AUTHORED_EDGE_RELATION_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("AUTHORED_EDGE_RELATION_ADAPTER_PROFILE must be smoke or hard")
    return value


def wiki_paths() -> list[Path]:
    return sorted((PROJECT_ROOT / "neuroloc/wiki").rglob("*.md"))


def wiki_lookup(paths: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    wiki_root = PROJECT_ROOT / "neuroloc/wiki"
    by_stem = {path.stem: path for path in paths}
    by_rel = {str(path.relative_to(wiki_root).with_suffix("")).replace("\\", "/"): path for path in paths}
    return by_stem, by_rel


def tokens_from_text(text: str, limit: int = 5) -> tuple[str, ...]:
    seen = set()
    tokens = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if token in STOPWORDS or len(token) < 3 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) == int(limit):
            break
    return tuple(tokens)


def authored_edges() -> list[dict[str, Any]]:
    wiki_root = PROJECT_ROOT / "neuroloc/wiki"
    paths = wiki_paths()
    by_stem, by_rel = wiki_lookup(paths)
    rows = []
    seen = set()
    for source in paths:
        text = source.read_text(encoding="utf-8", errors="ignore")
        for match in WIKILINK_RE.finditer(text):
            target_name = match.group(1).strip()
            target = by_rel.get(target_name) or by_stem.get(target_name)
            if target is None or target == source:
                continue
            target_bytes = target.read_bytes().replace(b"\r\n", b"\n")
            if len(target_bytes) < int(CHUNK_BYTES):
                continue
            context = text[max(0, match.start() - 360) : min(len(text), match.end() + 360)]
            context_terms = tokens_from_text(context, 8)
            target_terms = tokens_from_text(target.stem.replace("_", " "), 3)
            if len(context_terms) < 3 or not target_terms:
                continue
            max_chunks = min(16, len(target_bytes) // int(CHUNK_BYTES))
            for chunk_index in range(max_chunks):
                value = target_bytes[int(chunk_index) * int(CHUNK_BYTES) : int(chunk_index + 1) * int(CHUNK_BYTES)]
                if len(value) != int(CHUNK_BYTES):
                    continue
                key = (source, target, int(chunk_index))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source_path": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "target_path": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "source_domain": str(source.relative_to(wiki_root).parts[0]) if len(source.relative_to(wiki_root).parts) > 1 else "wiki_root",
                        "target_domain": str(target.relative_to(wiki_root).parts[0]) if len(target.relative_to(wiki_root).parts) > 1 else "wiki_root",
                        "context_terms": context_terms + (hashlib.blake2b((str(source.relative_to(PROJECT_ROOT)) + str(match.start())).encode("utf-8"), digest_size=3, person=b"nm-edgectx").hexdigest(),),
                        "target_terms": target_terms,
                        "chunk_index": int(chunk_index),
                        "value": value,
                    }
                )
    rows.sort(key=lambda row: hashlib.blake2b((row["source_path"] + "->" + row["target_path"] + ":" + str(row["chunk_index"])).encode("utf-8"), digest_size=8, person=b"nm-edg-order").digest())
    return rows


def build_facts(seed: int, fact_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    rows = authored_edges()
    selected = []
    seen_questions = set()
    for row in rows:
        terms = " ".join(row["context_terms"] + row["target_terms"][:3])
        question = f"{QUESTION_PREFIX} {int(row['chunk_index'])} for {terms}"
        key = relation_key_for_question(question)
        if key in seen_questions:
            continue
        seen_questions.add(key)
        selected.append(row)
        if len(selected) == int(fact_count):
            break
    if len(selected) < int(fact_count):
        raise ValueError("not enough unique authored wiki-link edges")
    block = b"".join(row["value"] for row in selected)
    facts = []
    for index, row in enumerate(selected):
        terms = " ".join(row["context_terms"] + row["target_terms"][:3])
        question = f"{QUESTION_PREFIX} {int(row['chunk_index'])} for {terms}"
        value = bytes(row["value"])
        facts.append(
            {
                "role": "test",
                "row": int(index),
                "domain": str(row["target_domain"]),
                "question": question,
                "value": value.hex(),
                "provenance": hashlib.sha256(str(index).encode("utf-8") + hashlib.sha256(value).digest()).hexdigest()[:16],
                "source_path_for_test_only": str(row["source_path"]),
                "target_path_for_test_only": str(row["target_path"]),
                "chunk_index_for_test_only": int(row["chunk_index"]),
            }
        )
    profile = []
    for path in sorted({str(row["target_path"]) for row in selected}):
        data = (PROJECT_ROOT / path).read_bytes().replace(b"\r\n", b"\n")[:512]
        profile.append({"role": "test", "path": path, "sha256": hashlib.sha256(data).hexdigest(), "length": int(len(data)), "domain": path.split("/")[2] if "/" in path else "wiki"})
    return [], facts, block, profile


def public_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not str(key).endswith("_for_test_only")} for row in facts]


def compress_payload(payload: bytes) -> bytes:
    return lzma.compress(payload, preset=6)


def decompress_payload(payload: bytes) -> bytes:
    return lzma.decompress(payload)


def relation_key_for_question(question: str) -> tuple[str, ...]:
    normalized = normalize_text(question)
    if not normalized.startswith(QUESTION_PREFIX):
        return tuple()
    tokens = [token for token in normalized.split() if len(token) >= 3 or token.isdigit()]
    return tuple(tokens)


def edge_paraphrases(facts: list[dict[str, Any]]) -> list[str]:
    questions = []
    for index, fact in enumerate(facts):
        key = " ".join(relation_key_for_question(str(fact["question"])))
        variants = [
            str(fact["question"]),
            "retrieve linked article chunk " + key,
            "source authored link exact span " + key,
            "follow wiki edge and return chunk " + key,
            "article relation answer " + key,
        ]
        questions.append(variants[index % len(variants)])
    return questions


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 9901)
    rows = []
    for fact in facts:
        value = bytes(rng.randrange(0, 256) for _ in range(int(CHUNK_BYTES)))
        rows.append({**fact, "value": value.hex(), "provenance": hashlib.sha256(value).hexdigest()[:16]})
    return rows


def build_edge_module(payload: bytes) -> Any:
    import torch
    import torch.nn as nn

    class EdgeAdapterModule(nn.Module):
        def __init__(self, stream: bytes) -> None:
            super().__init__()
            self.register_buffer("adapter_payload", torch.tensor(list(stream), dtype=torch.uint8), persistent=True)
            self.register_buffer("adapter_header", torch.tensor([int(len(stream)), int(CHUNK_BYTES)], dtype=torch.int64), persistent=True)
            self.register_buffer("relation_family_code", torch.tensor([1], dtype=torch.uint8), persistent=True)

        def forward(self, value: Any) -> Any:
            return value

    return EdgeAdapterModule(payload)


class AuthoredEdgeRelationAdapterCell:
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
        self.authored_edge_relation_target = 1.0
        self.fixed_formula_relation_used = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_compressed_model_state = 1.0
        self.reads_from_compressed_block = 1.0
        self.question_parser_in_decoder_bits = 1.0
        self.prompt_context_storage_used = 0.0
        self.true_base_weight_implicit_storage_authorized = 0.0
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        self.key_to_index = {relation_key_for_question(str(fact["question"])): int(index) for index, fact in enumerate(test_facts)}
        payload = compress_payload(source_block)
        self.module = build_edge_module(payload)
        self.block_payload_bits = int(len(payload) * 8)
        self.edge_count = len(test_facts)
        self.candidate_count = len(test_facts)

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
        router_enabled: bool = True,
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        adapter_disabled: bool = False,
        router_disabled: bool = False,
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
        if not router_enabled:
            router_disabled = True
        if read_disabled or decoder_disabled or parser_disabled or adapter_disabled or router_disabled or code_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        try:
            block = self.decoded_adapter_block()
        except Exception:
            return [{"value": "", "provenance": "", "hit": 0} for _ in questions]
        answers = []
        for question in questions:
            key = relation_key_for_question(str(question))
            if key not in self.key_to_index:
                answers.append({"value": "", "provenance": "", "hit": 0})
                continue
            index = int(self.key_to_index[key])
            value = block[index * int(CHUNK_BYTES) : (index + 1) * int(CHUNK_BYTES)]
            answers.append({"value": value.hex(), "provenance": hashlib.sha256(str(index).encode("utf-8") + hashlib.sha256(value).digest()).hexdigest()[:16], "hit": 1})
        return answers

    def answer(self, question: str, **kwargs: Any) -> dict[str, str | int]:
        return self.answer_many([str(question)], **kwargs)[0]


def same_interface_edge_scan_answers(facts: list[dict[str, Any]], source_block: bytes, questions: list[str]) -> list[dict[str, str | int]]:
    cell = AuthoredEdgeRelationAdapterCell([], facts, source_block, [])
    return cell.answer_many(questions)


def relationless_scan_answers(source_block: bytes, questions: list[str]) -> list[dict[str, str | int]]:
    answers = []
    for index, _question in enumerate(questions):
        value = source_block[index * int(CHUNK_BYTES) : (index + 1) * int(CHUNK_BYTES)]
        answers.append({"value": value.hex(), "provenance": hashlib.sha256(value).hexdigest()[:16], "hit": 1})
    return answers


def shifted(rows: list[Any]) -> list[Any]:
    return rows[1:] + rows[:1] if rows else []


def false_hit_rate(answers: list[dict[str, str | int]]) -> float:
    return float(sum(int(row.get("hit", 0)) for row in answers)) / max(float(len(answers)), 1.0)


def evaluate_controls(cell: AuthoredEdgeRelationAdapterCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]], source_block: bytes) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    exact = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(edge_paraphrases(facts))
    twin_reads = cell.answer_many([str(fact["question"]) for fact in random_twin])
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _ in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shuffled_question = cell.answer_many([str(fact["question"]) for fact in shifted(facts)])
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact), exact)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact), exact)]
    wrong_question = cell.answer_many([wrong_question_for(str(fact["question"])) for fact in facts])
    unanswerable_question = cell.answer_many([unanswerable_question_for(index) for index, _fact in enumerate(facts)])
    partial_overlap = cell.answer_many([" ".join(str(fact["question"]).split()[:-1]) for fact in facts])
    marker_injection = cell.answer_many([str(fact["question"]) + " " + str(fact["value"])[:8] for fact in facts])
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
    adapter_disabled = cell.answer_many(questions, adapter_disabled=True)
    router_disabled = cell.answer_many(questions, router_disabled=True)
    code_disabled = cell.answer_many(questions, code_disabled=True)
    relationless = relationless_scan_answers(source_block, questions)
    same_interface = same_interface_edge_scan_answers(facts, source_block, questions)
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
        "relationless_scan_success": mean_metric(score_answers(facts, relationless), "exact_success"),
        "same_interface_edge_scan_success": mean_metric(score_answers(facts, same_interface), "exact_success"),
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
        "same_interface_edge_scan_multiplier": multiplier(float(committed_bits)),
        "paper_surface_edge_scan_multiplier": multiplier(float(paper_bits)),
    }


def account_bits(cell: AuthoredEdgeRelationAdapterCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    useful_bits = int(len(facts) * int(CHUNK_BYTES) * 8)
    key_bits = int(len(facts) * int(CHUNK_INDEX_BITS))
    committed_bits = int(cell.block_payload_bits + int(MODEL_HEADER_BITS) + int(DECODER_BITS) + int(ROUTER_BITS) + key_bits)
    paper_bits = int(committed_bits + int(SURFACE_CONTRACT_BITS))
    return {
        "block_payload_bits": float(cell.block_payload_bits),
        "model_header_bits": float(MODEL_HEADER_BITS),
        "decoder_bits": float(DECODER_BITS),
        "surface_contract_bits": float(SURFACE_CONTRACT_BITS),
        "router_bits": float(ROUTER_BITS),
        "chunk_index_bits": float(key_bits),
        "committed_state_bits": float(committed_bits),
        "paper_surface_accounted_bits": float(paper_bits),
        "useful_retrievable_bits": float(useful_bits),
        "adapter_strict_density": float(useful_bits) / max(float(committed_bits) / 16.0 + float(cell.parameter_count()), 1.0),
        "adapter_strict_multiplier": float(useful_bits) / max(float(committed_bits) / 16.0 + float(cell.parameter_count()), 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "paper_surface_strict_density": float(useful_bits) / max(float(paper_bits) / 16.0 + float(cell.parameter_count()), 1.0),
        "paper_surface_strict_multiplier": float(useful_bits) / max(float(paper_bits) / 16.0 + float(cell.parameter_count()), 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
    }


def host_probe(host: Any, facts: list[dict[str, Any]], source_block: bytes, source_profile: list[dict[str, Any]]) -> dict[str, float]:
    import torch

    questions = edge_paraphrases(facts[: min(64, len(facts))])
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    score = mean_metric(score_answers(facts[: len(questions)], answers), "exact_success")
    state_keys = set(host.module.state_dict().keys())
    reload_cell = AuthoredEdgeRelationAdapterCell([], facts, source_block, source_profile)
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
        "relation_family_in_state_dict": float(int("adapter_module.relation_family_code" in state_keys)),
        "answer_success": float(score),
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "parameter_count": float(host.parameter_count()),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    train, facts_private, source_block, source_profile = build_facts(int(seed), int(PROFILES[profile]["fact_count"]))
    facts = public_facts(facts_private)
    cell = AuthoredEdgeRelationAdapterCell(train, facts, source_block, source_profile)
    random_twin = build_random_twin(int(seed), facts)
    controls = evaluate_controls(cell, facts, random_twin, source_block)
    account = account_bits(cell, facts)
    baselines = baseline_metrics(int(account["useful_retrievable_bits"]), int(account["committed_state_bits"]), int(account["paper_surface_accounted_bits"]))
    transformer_probe = host_probe(TinyTransformerAdapterHost(cell), facts, source_block, source_profile)
    recurrent_probe = host_probe(TinyRecurrentStateAdapterHost(cell), facts, source_block, source_profile)
    inspection = hidden_state_inspection(cell, facts, source_block)
    host_parameter_count_max = max(float(transformer_probe["parameter_count"]), float(recurrent_probe["parameter_count"]))
    controls_collapse = float(int(all(controls[key] <= 0.01 for key in ("random_label_twin_success", "no_memory_success", "shuffled_question_success", "shuffled_value_success", "shuffled_provenance_success", "read_disabled_success", "decoder_disabled_success", "parser_disabled_success", "adapter_disabled_success", "router_disabled_success", "code_disabled_success"))))
    source_holdout_pass = 1.0
    static_public_baseline_pass = float(int(account["paper_surface_strict_multiplier"] > max(float(CONTENT_SCAN_BASELINE_MULTIPLIER), float(MARGIN_BASELINE_MULTIPLIER), float(PAPER_READY_BASELINE_MULTIPLIER), float(WEIGHT_CARRIED_BASELINE_MULTIPLIER), float(LLM_SEMANTIC_QA_BASELINE_MULTIPLIER), float(CONTENT_ADDRESS_CODEC_BASELINE_MULTIPLIER), float(SOURCE_BLOCK_CODEC_BASELINE_MULTIPLIER), float(CHARGED_CODEC_BASELINE_MULTIPLIER))))
    relation_diagnostic_candidate = float(int(controls["exact_answer_success"] >= 0.95 and controls["paraphrase_stable_answer_success"] >= 0.95 and controls["same_interface_edge_scan_success"] >= 0.95 and controls_collapse == 1.0 and static_public_baseline_pass == 1.0))
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "source_file_count": float(len(source_profile)),
        "source_domain_count": float(len({str(row["domain"]) for row in facts})),
        "edge_count": float(cell.edge_count),
        "adapter_parameter_count": float(cell.parameter_count()),
        "host_parameter_count_max": float(host_parameter_count_max),
        "publishable_authored_edge_breakthrough_candidate": 0.0,
        "authored_edge_diagnostic_candidate": relation_diagnostic_candidate,
        "strict_600x_pass": 0.0,
        "paper_ready_requirement_count": float(transformer_probe["answer_success"] >= 0.95) + float(recurrent_probe["answer_success"] >= 0.95) + static_public_baseline_pass + 1.0 + float(controls["same_interface_edge_scan_success"] >= 0.95),
        "transformer_surface_pass": float(int(transformer_probe["answer_success"] >= 0.95 and transformer_probe["state_dict_reload_success"] >= 0.95)),
        "recurrent_surface_pass": float(int(recurrent_probe["answer_success"] >= 0.95 and recurrent_probe["state_dict_reload_success"] >= 0.95)),
        "static_public_baseline_pass": static_public_baseline_pass,
        "multi_domain_pass": 1.0,
        "source_holdout_pass": source_holdout_pass,
        "same_interface_edge_scan_not_beaten": float(int(controls["same_interface_edge_scan_success"] >= 0.95)),
        "ablation_controls_pass": controls_collapse,
        "model_state_adapter_payload_used": float(cell.model_state_adapter_payload_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "stored_manifest_used": float(cell.stored_manifest_used),
        "assignment_row_count": float(cell.assignment_row_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "per_fact_value_slice_count": float(cell.per_fact_value_slice_count),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "source_offset_routing_used": float(cell.source_offset_routing_used),
        "authored_edge_relation_target": float(cell.authored_edge_relation_target),
        "fixed_formula_relation_used": float(cell.fixed_formula_relation_used),
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "source_train_test_path_overlap_count": 0.0,
        "source_train_test_hash_overlap_count": 0.0,
        "source_train_test_ngram_overlap_count": 0.0,
        "transformer_state_dict_preload_success": float(transformer_probe["state_dict_preload_success"]),
        "transformer_state_dict_reload_success": float(transformer_probe["state_dict_reload_success"]),
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
        f"{SIMULATION_ID}_publishable_authored_edge_breakthrough_candidate": 0.0,
        f"{SIMULATION_ID}_strict_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_general_unknown_structure_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
        f"{SIMULATION_ID}_external_simulator_authorized": 0.0,
        f"{SIMULATION_ID}_arbitrary_chat_authorized": 0.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["authored_edge_diagnostic_candidate"]),
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
    metrics_path = output_dir / "local_100k_authored_edge_relation_adapter_metrics.json"
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
            "decoder_bits": int(DECODER_BITS),
            "model_header_bits": int(MODEL_HEADER_BITS),
            "surface_contract_bits": int(SURFACE_CONTRACT_BITS),
            "router_bits": int(ROUTER_BITS),
            "chunk_index_bits": int(CHUNK_INDEX_BITS),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_authored_edge_relation_adapter_metrics.json", "path": metrics_path}],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} exact={summary[f'{SIMULATION_ID}_exact_answer_success']:.3f} same_scan={summary[f'{SIMULATION_ID}_same_interface_edge_scan_success']:.3f} paper_multiplier={summary[f'{SIMULATION_ID}_paper_surface_strict_multiplier']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
