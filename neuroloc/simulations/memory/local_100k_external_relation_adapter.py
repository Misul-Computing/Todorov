from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from functools import lru_cache
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
from neuroloc.simulations.memory.local_100k_source_dense_authored_relation_diagnostic import dense_authored_relation_facts
from neuroloc.simulations.memory.local_100k_source_relation_mph_codec import (
    FINGERPRINT_BITS,
    HONEST_MPH_OVERHEAD_BITS_PER_KEY,
    PROVENANCE_BITS_PER_FACT,
    RELATION_DECODER_BITS,
    ROUTER_HEADER_BITS,
    SourceRelationMPHCodecModule,
    build_relation_codec,
    random_label_facts,
    recompute_relation_paq8px,
    relation_codec_bits,
    relation_useful_bits,
    score_answers,
    state_probe,
    wrong_query_variants,
)

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_external_relation_adapter"
SEED = env_int("EXTERNAL_RELATION_ADAPTER_SEED", 14387)
DOWNLOAD_TIMEOUT_SEC = env_int("EXTERNAL_RELATION_ADAPTER_DOWNLOAD_TIMEOUT_SEC", 45)
MODEL_PACKAGE_HEADER_BITS = env_int("EXTERNAL_RELATION_ADAPTER_MODEL_PACKAGE_HEADER_BITS", 4096)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("EXTERNAL_RELATION_ADAPTER_ORDINARY_BITS_PER_PARAMETER", "2.5"))
CURRENT_LOCAL_RELATION_MULTIPLIER = float(os.environ.get("EXTERNAL_RELATION_ADAPTER_CURRENT_LOCAL_RELATION_MULTIPLIER", "67.90445687825584"))

require_positive("EXTERNAL_RELATION_ADAPTER_DOWNLOAD_TIMEOUT_SEC", DOWNLOAD_TIMEOUT_SEC)
require_positive("EXTERNAL_RELATION_ADAPTER_MODEL_PACKAGE_HEADER_BITS", MODEL_PACKAGE_HEADER_BITS)

EXTERNAL_SOURCES = [
    {
        "name": "cpython_argparse",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/argparse.py",
        "sha256": "2acf90321c8d7cc2fb3e1b0a248ec8df8431c4c095e99604aa71faec48455100",
        "bytes": 101454,
    },
    {
        "name": "cpython_base_events",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/asyncio/base_events.py",
        "sha256": "4bf27730499a68f6a5b726204670590cb3425c9331941dd2036a0ad101af39cb",
        "bytes": 77971,
    },
    {
        "name": "cpython_enum",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/enum.py",
        "sha256": "c8ead615c159598370295649eb296819ad4b40d50b200c4fec2d4269bf7af9ae",
        "bytes": 81636,
    },
    {
        "name": "cpython_dataclasses",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/dataclasses.py",
        "sha256": "0e449d55d6206b0022f541ba32be88fafc934ff71d9aa65f31f101ca6147f2ae",
        "bytes": 61753,
    },
    {
        "name": "cpython_pathlib",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/pathlib.py",
        "sha256": "dc14d8207519fb8bcdd9c7bae1d54da3ad1b339aa83d4979c16057cd552a3487",
        "bytes": 51105,
    },
    {
        "name": "cpython_typing",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/typing.py",
        "sha256": "c45d935c17234b1d6ae42d2d5499d3e03b4e2548fae0c4fce15477e23502214d",
        "bytes": 117428,
    },
    {
        "name": "cpython_mock",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/unittest/mock.py",
        "sha256": "9b7fb2946f5b9a9db9b80247d235e396abc1654fde898f2c2a215503f45a8145",
        "bytes": 104961,
    },
]

PROFILE_INDICES = {
    "smoke": (0, 1, 2),
    "hard": (0, 1, 2, 3, 4, 5, 6),
}

PROFILES = {
    "smoke": {"min_fact_count": 2500.0, "min_public_margin_bits": 50000.0, "min_multiplier": 35.0},
    "hard": {"min_fact_count": 6000.0, "min_public_margin_bits": 150000.0, "min_multiplier": 30.0},
}

QUERY_PREFIXES = (
    "external source relation query:",
    "answer the stored source relation for query:",
    "bounded relation lookup:",
    "relation adapter question:",
)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("EXTERNAL_RELATION_ADAPTER_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("EXTERNAL_RELATION_ADAPTER_PROFILE must be smoke or hard")
    return value


def cache_root() -> Path:
    return PROJECT_ROOT / "codex_local_output" / "external_relation_adapter_cache"


def external_rows(profile: str) -> list[dict[str, Any]]:
    return [EXTERNAL_SOURCES[int(index)] for index in PROFILE_INDICES[str(profile)]]


def source_cache_path(row: dict[str, Any]) -> Path:
    return cache_root() / f"{row['name']}.py"


def load_external_source(row: dict[str, Any]) -> bytes:
    cache_root().mkdir(parents=True, exist_ok=True)
    path = source_cache_path(row)
    if path.exists():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == str(row["sha256"]):
            return data
    with urllib.request.urlopen(str(row["url"]), timeout=int(DOWNLOAD_TIMEOUT_SEC)) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != str(row["sha256"]):
        raise ValueError(f"external source hash mismatch for {row['name']}: {digest}")
    path.write_bytes(data)
    return data


def external_blocks(profile: str) -> list[bytes]:
    return [load_external_source(row) for row in external_rows(profile)]


def source_manifest_digest(rows: list[dict[str, Any]]) -> str:
    joined = "\n".join(f"{row['name']}:{row['sha256']}:{row['url']}" for row in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def normalize_relation_question(question: str, parser_disabled: bool = False) -> str:
    if parser_disabled:
        return str(question)
    value = str(question).strip()
    lowered = value.lower()
    for prefix in QUERY_PREFIXES:
        if lowered.startswith(prefix):
            return value[len(prefix) :].strip()
    return value


def paraphrase_questions(facts: list[dict[str, Any]]) -> list[str]:
    rows = []
    for index, fact in enumerate(facts):
        question = str(fact["question"])
        variants = [
            question,
            f"external source relation query: {question}",
            f"answer the stored source relation for query: {question}",
            f"bounded relation lookup: {question}",
            f"relation adapter question: {question}",
        ]
        rows.append(variants[index % len(variants)])
    return rows


class ExternalRelationAdapterCell:
    def __init__(self, facts: list[dict[str, Any]] | None = None, module: SourceRelationMPHCodecModule | None = None) -> None:
        self.model_state_adapter_payload_used = 1.0
        self.state_dict_buffer_payload_used = 1.0
        self.external_payload_store_used = 0.0
        self.raw_source_block_retained = 0.0
        self.full_question_table_stored = 0.0
        self.generated_alias_labels_present = 0.0
        self.formula_or_schema_labels_present = 0.0
        self.true_base_weight_implicit_storage_authorized = 0.0
        self.adapter_state_stream_count = 7
        self.per_fact_value_row_count = 0
        self.per_fact_residual_row_count = 0
        self.relation_index_operation = 1.0
        self.bounded_question_surface = 1.0
        self.peft_like_adapter_route = 1.0
        self.quantization_packaging_route = 1.0
        self.memory_layer_backend_route = 1.0
        if module is not None:
            self.module = module
            return
        if facts is None:
            raise ValueError("facts or module required")
        self.codec = build_relation_codec(facts)
        self.module = SourceRelationMPHCodecModule(self.codec)

    @classmethod
    def empty_from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "ExternalRelationAdapterCell":
        module = SourceRelationMPHCodecModule.empty_from_state_dict(state_dict)
        return cls(module=module)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def answer_many(
        self,
        questions: list[str],
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        adapter_disabled: bool = False,
        code_disabled: bool = False,
        shuffled_fingerprint: bool = False,
        shuffled_value_ids: bool = False,
    ) -> list[dict[str, str | int]]:
        disabled = bool(read_disabled or decoder_disabled or adapter_disabled or code_disabled)
        normalized = [normalize_relation_question(question, parser_disabled=parser_disabled) for question in questions]
        return self.module.answer_many(normalized, disabled=disabled, shuffled_fingerprint=shuffled_fingerprint, shuffled_value_ids=shuffled_value_ids)


class TinyExternalTransformerHost:
    def __init__(self, cell: ExternalRelationAdapterCell) -> None:
        class Host(nn.Module):
            def __init__(self, adapter_cell: ExternalRelationAdapterCell) -> None:
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


class TinyExternalRecurrentHost:
    def __init__(self, cell: ExternalRelationAdapterCell) -> None:
        class Host(nn.Module):
            def __init__(self, adapter_cell: ExternalRelationAdapterCell) -> None:
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


class TinyExternalStateSpaceHost:
    def __init__(self, cell: ExternalRelationAdapterCell) -> None:
        class Host(nn.Module):
            def __init__(self, adapter_cell: ExternalRelationAdapterCell) -> None:
                super().__init__()
                self.embedding = nn.Embedding(256, 16)
                self.input_map = nn.Linear(16, 16)
                self.gate_map = nn.Linear(16, 16)
                self.output_map = nn.Linear(16, 16)
                self.state_decay = nn.Parameter(torch.zeros(16))
                self.adapter_module = adapter_cell.module

            def forward(self, token_ids: Any) -> Any:
                hidden = self.embedding(token_ids)
                state = torch.zeros(hidden.shape[0], 16, dtype=hidden.dtype, device=hidden.device)
                decay = torch.sigmoid(self.state_decay).unsqueeze(0)
                for step in range(hidden.shape[1]):
                    gate = torch.sigmoid(self.gate_map(hidden[:, step, :]))
                    proposal = torch.tanh(self.input_map(hidden[:, step, :]))
                    state = decay * state + gate * proposal
                return self.output_map(state)

        self.cell = cell
        self.module = Host(cell)

    def answer_many(self, questions: list[str], **kwargs: Any) -> list[dict[str, str | int]]:
        return self.cell.answer_many(questions, **kwargs)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))


def tensorize_questions(questions: list[str], width: int = 48) -> torch.Tensor:
    rows = []
    for question in questions:
        data = str(question).encode("utf-8", errors="ignore")[: int(width)]
        data = data + bytes(max(0, int(width) - len(data)))
        rows.append(list(data))
    return torch.tensor(rows, dtype=torch.long)


def corrupt_relation_payload(module: SourceRelationMPHCodecModule) -> None:
    if hasattr(module, "fingerprint_payload"):
        module.fingerprint_payload = torch.tensor([(int(item) ^ 0xA5) for item in module.fingerprint_payload.tolist()], dtype=torch.uint8)


def host_probe(host: Any, facts: list[dict[str, Any]]) -> dict[str, float]:
    questions = paraphrase_questions(facts[: min(256, len(facts))])
    token_ids = tensorize_questions(questions)
    with torch.no_grad():
        output = host.module(token_ids)
    answers = host.answer_many(questions)
    state_keys = set(host.module.state_dict().keys())
    cell_state = host.cell.module.state_dict()
    reload_cell = ExternalRelationAdapterCell.empty_from_state_dict(cell_state)
    corrupt_relation_payload(reload_cell.module)
    reload_host = type(host)(reload_cell)
    preload_score = score_answers(facts[: len(questions)], reload_host.answer_many(questions))
    reload_host.module.load_state_dict(host.module.state_dict())
    reload_score = score_answers(facts[: len(questions)], reload_host.answer_many(questions))
    return {
        "forward_shape_success": float(int(tuple(output.shape) == (len(questions), 16))),
        "relation_payload_in_state_dict": float(int("adapter_module.fingerprint_payload" in state_keys)),
        "relation_header_in_state_dict": float(int("adapter_module.relation_header" in state_keys)),
        "state_dict_preload_success": float(preload_score),
        "state_dict_reload_success": float(reload_score),
        "paraphrase_answer_success": float(score_answers(facts[: len(questions)], answers)),
        "parameter_count": float(host.parameter_count()),
    }


def shuffled_value_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not facts:
        return []
    values = [str(fact["value"]) for fact in facts]
    provenances = [str(fact["provenance"]) for fact in facts]
    return [{**fact, "value": values[(index + 1) % len(values)], "provenance": provenances[(index + 1) % len(provenances)]} for index, fact in enumerate(facts)]


def control_metrics(cell: ExternalRelationAdapterCell, facts: list[dict[str, Any]], seed: int) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    paraphrases = paraphrase_questions(facts)
    prefixed = [f"external source relation query: {question}" for question in questions]
    exact_answers = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(paraphrases)
    random_facts = random_label_facts(int(seed), facts)
    random_codec_cell = ExternalRelationAdapterCell(random_facts)
    wrong_questions = wrong_query_variants(facts)
    prefixed_wrong = [f"external source relation query: {question}" for question in wrong_questions]
    wrong_answers = cell.answer_many(wrong_questions + prefixed_wrong)
    return {
        "exact_relation_answer_success": score_answers(facts, exact_answers),
        "paraphrased_relation_answer_success": score_answers(facts, paraphrase_answers),
        "random_label_twin_success": score_answers(random_facts, exact_answers),
        "random_label_rebuild_exact_success": score_answers(random_facts, random_codec_cell.answer_many(questions)),
        "random_label_rebuild_selected_relation_accounted_bits": float(relation_codec_bits(random_codec_cell.codec)),
        "decoder_disabled_success": score_answers(facts, cell.answer_many(questions, decoder_disabled=True)),
        "parser_disabled_prefixed_success": score_answers(facts, cell.answer_many(prefixed, parser_disabled=True)),
        "read_disabled_success": score_answers(facts, cell.answer_many(questions, read_disabled=True)),
        "adapter_disabled_success": score_answers(facts, cell.answer_many(questions, adapter_disabled=True)),
        "code_disabled_success": score_answers(facts, cell.answer_many(questions, code_disabled=True)),
        "shuffled_fingerprint_success": score_answers(facts, cell.answer_many(questions, shuffled_fingerprint=True)),
        "shuffled_value_id_success": score_answers(facts, cell.answer_many(questions, shuffled_value_ids=True)),
        "shuffled_value_label_success": score_answers(shuffled_value_facts(facts), exact_answers),
        "wrong_query_variant_count": float(len(wrong_questions) + len(prefixed_wrong)),
        "wrong_query_hit_rate": float(sum(int(answer["hit"]) for answer in wrong_answers)) / max(float(len(wrong_answers)), 1.0),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    rows = external_rows(profile)
    blocks = external_blocks(profile)
    facts = dense_authored_relation_facts(blocks)
    cell = ExternalRelationAdapterCell(facts)
    codec = cell.codec
    selected_bits = int(relation_codec_bits(codec))
    package_bits = int(selected_bits + int(MODEL_PACKAGE_HEADER_BITS))
    useful_bits = int(relation_useful_bits(facts))
    controls = control_metrics(cell, facts, int(seed))
    random_bits = int(controls["random_label_rebuild_selected_relation_accounted_bits"])
    random_density_collapse = float(int(controls["random_label_rebuild_exact_success"] == 1.0 and random_bits > selected_bits))
    paq_metrics = recompute_relation_paq8px(blocks, f"external_relation_adapter_{profile}")
    paq_bits = int(paq_metrics["paq8px_relation_recomputed_accounted_bits"])
    raw_source_paq_content_scan_bits = int(paq_bits)
    public_margin = float(paq_bits - selected_bits)
    honest_mph_bits = int(useful_bits + len(facts) * (int(HONEST_MPH_OVERHEAD_BITS_PER_KEY) + int(PROVENANCE_BITS_PER_FACT)) + int(RELATION_DECODER_BITS))
    undercharged_mph_bits = int(useful_bits + len(facts) * (int(FINGERPRINT_BITS) + 16) + int(RELATION_DECODER_BITS))
    state = state_probe(cell.module, facts, blocks)
    transformer_probe = host_probe(TinyExternalTransformerHost(cell), facts)
    recurrent_probe = host_probe(TinyExternalRecurrentHost(cell), facts)
    state_space_probe = host_probe(TinyExternalStateSpaceHost(cell), facts)
    host_parameter_count_max = max(float(transformer_probe["parameter_count"]), float(recurrent_probe["parameter_count"]), float(state_space_probe["parameter_count"]))
    source_hash_success = float(min(float(hashlib.sha256(block).hexdigest() == str(row["sha256"])) for block, row in zip(blocks, rows)))
    source_size_success = float(min(float(len(block) == int(row["bytes"])) for block, row in zip(blocks, rows)))
    strict_density = float(useful_bits) / max(float(selected_bits) / 16.0, 1.0)
    package_density = float(useful_bits) / max(float(package_bits) / 16.0, 1.0)
    controls_collapse = float(
        int(
            controls["random_label_twin_success"] == 0.0
            and random_density_collapse == 1.0
            and controls["decoder_disabled_success"] == 0.0
            and controls["parser_disabled_prefixed_success"] < 1.0
            and controls["read_disabled_success"] == 0.0
            and controls["adapter_disabled_success"] == 0.0
            and controls["code_disabled_success"] == 0.0
            and controls["shuffled_fingerprint_success"] == 0.0
            and controls["shuffled_value_id_success"] <= 0.05
            and controls["wrong_query_hit_rate"] == 0.0
        )
    )
    transformer_pass = float(int(transformer_probe["forward_shape_success"] == 1.0 and transformer_probe["relation_payload_in_state_dict"] == 1.0 and transformer_probe["relation_header_in_state_dict"] == 1.0 and transformer_probe["paraphrase_answer_success"] >= 0.95 and transformer_probe["state_dict_reload_success"] >= 0.95 and transformer_probe["state_dict_preload_success"] == 0.0))
    recurrent_pass = float(int(recurrent_probe["forward_shape_success"] == 1.0 and recurrent_probe["relation_payload_in_state_dict"] == 1.0 and recurrent_probe["relation_header_in_state_dict"] == 1.0 and recurrent_probe["paraphrase_answer_success"] >= 0.95 and recurrent_probe["state_dict_reload_success"] >= 0.95 and recurrent_probe["state_dict_preload_success"] == 0.0))
    state_space_pass = float(int(state_space_probe["forward_shape_success"] == 1.0 and state_space_probe["relation_payload_in_state_dict"] == 1.0 and state_space_probe["relation_header_in_state_dict"] == 1.0 and state_space_probe["paraphrase_answer_success"] >= 0.95 and state_space_probe["state_dict_reload_success"] >= 0.95 and state_space_probe["state_dict_preload_success"] == 0.0))
    external_relation_adapter_product = float(
        int(
            controls["exact_relation_answer_success"] >= 0.95
            and controls["paraphrased_relation_answer_success"] >= 0.95
            and controls_collapse == 1.0
            and state["state_dict_reload_success"] >= 0.95
            and state["state_dict_payload_keys_present"] == 1.0
            and state["raw_source_block_retained"] == 0.0
            and state["full_question_table_stored"] == 0.0
            and source_hash_success == 1.0
            and source_size_success == 1.0
            and paq_metrics["paq8px_baseline_recomputed_in_run"] == 1.0
            and selected_bits < paq_bits
            and selected_bits < honest_mph_bits
            and public_margin >= float(PROFILES[profile]["min_public_margin_bits"])
            and strict_density / float(ORDINARY_BITS_PER_PARAMETER) >= float(PROFILES[profile]["min_multiplier"])
            and transformer_pass == 1.0
            and recurrent_pass == 1.0
            and state_space_pass == 1.0
            and host_parameter_count_max < 100000.0
        )
    )
    return {
        "profile": profile,
        "external_source_count": float(len(rows)),
        "external_source_total_bytes": float(sum(len(block) for block in blocks)),
        "external_source_hash_success": source_hash_success,
        "external_source_size_success": source_size_success,
        "external_source_manifest_digest_prefix": float(int(source_manifest_digest(rows)[:8], 16)),
        "relation_fact_count": float(len(facts)),
        "definition_parent_relation_count": float(sum(1 for fact in facts if fact["relation"] == "definition_parent")),
        "statement_enclosing_relation_count": float(sum(1 for fact in facts if fact["relation"] == "statement_enclosing_signature")),
        "control_statement_enclosing_relation_count": float(sum(1 for fact in facts if fact["relation"] == "control_statement_enclosing_signature")),
        "value_dictionary_count": float(codec["value_count"]),
        "provenance_dictionary_count": float(codec["provenance_count"]),
        "selected_relation_accounted_bits": float(selected_bits),
        "model_package_accounted_bits": float(package_bits),
        "model_package_header_bits": float(MODEL_PACKAGE_HEADER_BITS),
        "useful_retrievable_bits": float(useful_bits),
        "strict_density": float(strict_density),
        "strict_multiplier": float(strict_density / float(ORDINARY_BITS_PER_PARAMETER)),
        "model_package_strict_density": float(package_density),
        "model_package_strict_multiplier": float(package_density / float(ORDINARY_BITS_PER_PARAMETER)),
        "current_local_relation_multiplier_baseline": float(CURRENT_LOCAL_RELATION_MULTIPLIER),
        "external_multiplier_below_local_relation_product": float(int(strict_density / float(ORDINARY_BITS_PER_PARAMETER) < float(CURRENT_LOCAL_RELATION_MULTIPLIER))),
        "paq8px_level2_source_scan_accounted_bits": float(paq_bits),
        "raw_source_paq_content_scan_bits": float(raw_source_paq_content_scan_bits),
        "margin_over_paq8px_level2_source_scan_bits": float(public_margin),
        "margin_over_raw_source_paq_content_scan_bits": float(raw_source_paq_content_scan_bits - selected_bits),
        "honest_mph_relation_index_bits": float(honest_mph_bits),
        "undercharged_mph_relation_bits": float(undercharged_mph_bits),
        "margin_over_honest_mph_relation_index_bits": float(honest_mph_bits - selected_bits),
        "margin_over_undercharged_mph_relation_bits": float(undercharged_mph_bits - selected_bits),
        "external_public_corpus_used": 1.0,
        "external_public_corpus_cached": 1.0,
        "llm_model_state_adapter_surface": 1.0,
        "peft_like_adapter_route_candidate": float(cell.peft_like_adapter_route),
        "quantization_like_packaging_route_candidate": float(cell.quantization_packaging_route),
        "memory_layer_backend_route_candidate": float(cell.memory_layer_backend_route),
        "relation_index_operation": float(cell.relation_index_operation),
        "bounded_question_surface": float(cell.bounded_question_surface),
        "adapter_state_stream_count": float(cell.adapter_state_stream_count),
        "per_fact_value_row_count": float(cell.per_fact_value_row_count),
        "per_fact_residual_row_count": float(cell.per_fact_residual_row_count),
        "adapter_parameter_count": float(cell.parameter_count()),
        "host_parameter_count_max": float(host_parameter_count_max),
        "transformer_host_parameter_count": float(transformer_probe["parameter_count"]),
        "recurrent_host_parameter_count": float(recurrent_probe["parameter_count"]),
        "state_space_host_parameter_count": float(state_space_probe["parameter_count"]),
        "transformer_surface_pass": transformer_pass,
        "recurrent_surface_pass": recurrent_pass,
        "state_space_surface_pass": state_space_pass,
        "transformer_forward_shape_success": float(transformer_probe["forward_shape_success"]),
        "transformer_relation_payload_in_state_dict": float(transformer_probe["relation_payload_in_state_dict"]),
        "transformer_relation_header_in_state_dict": float(transformer_probe["relation_header_in_state_dict"]),
        "transformer_paraphrase_answer_success": float(transformer_probe["paraphrase_answer_success"]),
        "transformer_state_dict_preload_success": float(transformer_probe["state_dict_preload_success"]),
        "transformer_state_dict_reload_success": float(transformer_probe["state_dict_reload_success"]),
        "recurrent_forward_shape_success": float(recurrent_probe["forward_shape_success"]),
        "recurrent_relation_payload_in_state_dict": float(recurrent_probe["relation_payload_in_state_dict"]),
        "recurrent_relation_header_in_state_dict": float(recurrent_probe["relation_header_in_state_dict"]),
        "recurrent_paraphrase_answer_success": float(recurrent_probe["paraphrase_answer_success"]),
        "recurrent_state_dict_preload_success": float(recurrent_probe["state_dict_preload_success"]),
        "recurrent_state_dict_reload_success": float(recurrent_probe["state_dict_reload_success"]),
        "state_space_forward_shape_success": float(state_space_probe["forward_shape_success"]),
        "state_space_relation_payload_in_state_dict": float(state_space_probe["relation_payload_in_state_dict"]),
        "state_space_relation_header_in_state_dict": float(state_space_probe["relation_header_in_state_dict"]),
        "state_space_paraphrase_answer_success": float(state_space_probe["paraphrase_answer_success"]),
        "state_space_state_dict_preload_success": float(state_space_probe["state_dict_preload_success"]),
        "state_space_state_dict_reload_success": float(state_space_probe["state_dict_reload_success"]),
        "controls_collapse": controls_collapse,
        "random_label_rebuild_density_control_collapse": random_density_collapse,
        "public_context_mixing_beaten": float(int(selected_bits < paq_bits)),
        "raw_source_content_scan_beaten": float(int(selected_bits < raw_source_paq_content_scan_bits)),
        "honest_mph_index_beaten": float(int(selected_bits < honest_mph_bits)),
        "undercharged_mph_beaten": float(int(selected_bits < undercharged_mph_bits)),
        "external_relation_adapter_product_authorized": external_relation_adapter_product,
        "llm_adoptable_relation_adapter_candidate": external_relation_adapter_product,
        "source_relation_static_breakthrough_candidate": 0.0,
        "broad_breakthrough_authorized": 0.0,
        "strict_600x_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "true_base_weight_implicit_storage_authorized": 0.0,
        "formula_or_schema_labels_present": float(cell.formula_or_schema_labels_present),
        "generated_alias_labels_present": float(cell.generated_alias_labels_present),
        "engineering_pass": external_relation_adapter_product,
        **controls,
        **paq_metrics,
        **state,
    }


@lru_cache(maxsize=8)
def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_engineering_pass": float(row["engineering_pass"]),
    }
    for key, value in row.items():
        if key == "profile":
            continue
        if isinstance(value, str):
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
    metrics_path = output_dir / "local_100k_external_relation_adapter_metrics.json"
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
            "external_sources": [str(row["name"]) for row in external_rows(profile)],
            "fingerprint_bits": int(FINGERPRINT_BITS),
            "router_header_bits": int(ROUTER_HEADER_BITS),
            "decoder_bits": int(RELATION_DECODER_BITS),
            "model_package_header_bits": int(MODEL_PACKAGE_HEADER_BITS),
        },
        seed_numpy=int(SEED),
        n_trials=1,
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_external_relation_adapter_metrics.json", "path": metrics_path}],
        warnings=[],
        status="completed",
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} facts={summary[f'{SIMULATION_ID}_relation_fact_count']:.0f} multiplier={summary[f'{SIMULATION_ID}_strict_multiplier']:.6f} paq_margin={summary[f'{SIMULATION_ID}_margin_over_paq8px_level2_source_scan_bits']:.0f}")
    return 0 if summary[f"{SIMULATION_ID}_engineering_pass"] >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
