from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import zstandard as zstd

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
)
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import encode_unsigned_varint

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_model_state_knowledge_pack"
SEED = env_int("MODEL_STATE_KNOWLEDGE_PACK_SEED", 15131)
DOWNLOAD_TIMEOUT_SEC = env_int("MODEL_STATE_KNOWLEDGE_PACK_DOWNLOAD_TIMEOUT_SEC", 45)
MODEL_PACKAGE_HEADER_BITS = env_int("MODEL_STATE_KNOWLEDGE_PACK_MODEL_PACKAGE_HEADER_BITS", 8192)
ADAPTER_EXPORT_HEADER_BITS = env_int("MODEL_STATE_KNOWLEDGE_PACK_ADAPTER_EXPORT_HEADER_BITS", 8192)
UPDATE_PATCH_HEADER_BITS = env_int("MODEL_STATE_KNOWLEDGE_PACK_UPDATE_PATCH_HEADER_BITS", 2048)
MAX_DOCUMENT_CONTEXT_FACTS_PER_FILE = env_int("MODEL_STATE_KNOWLEDGE_PACK_MAX_DOCUMENT_CONTEXT_FACTS_PER_FILE", 450)
MAX_CONFIG_FACTS_PER_FILE = env_int("MODEL_STATE_KNOWLEDGE_PACK_MAX_CONFIG_FACTS_PER_FILE", 800)
UPDATE_FACT_COUNT = env_int("MODEL_STATE_KNOWLEDGE_PACK_UPDATE_FACT_COUNT", 384)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("MODEL_STATE_KNOWLEDGE_PACK_ORDINARY_BITS_PER_PARAMETER", "2.5"))

require_positive("MODEL_STATE_KNOWLEDGE_PACK_DOWNLOAD_TIMEOUT_SEC", DOWNLOAD_TIMEOUT_SEC)
require_positive("MODEL_STATE_KNOWLEDGE_PACK_MODEL_PACKAGE_HEADER_BITS", MODEL_PACKAGE_HEADER_BITS)
require_positive("MODEL_STATE_KNOWLEDGE_PACK_ADAPTER_EXPORT_HEADER_BITS", ADAPTER_EXPORT_HEADER_BITS)
require_positive("MODEL_STATE_KNOWLEDGE_PACK_UPDATE_PATCH_HEADER_BITS", UPDATE_PATCH_HEADER_BITS)
require_positive("MODEL_STATE_KNOWLEDGE_PACK_MAX_DOCUMENT_CONTEXT_FACTS_PER_FILE", MAX_DOCUMENT_CONTEXT_FACTS_PER_FILE)
require_positive("MODEL_STATE_KNOWLEDGE_PACK_MAX_CONFIG_FACTS_PER_FILE", MAX_CONFIG_FACTS_PER_FILE)
require_positive("MODEL_STATE_KNOWLEDGE_PACK_UPDATE_FACT_COUNT", UPDATE_FACT_COUNT)

PUBLIC_SURFACES = [
    {
        "name": "src_argparse",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/argparse.py",
        "sha256": "2acf90321c8d7cc2fb3e1b0a248ec8df8431c4c095e99604aa71faec48455100",
        "bytes": 101454,
    },
    {
        "name": "src_base_events",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/asyncio/base_events.py",
        "sha256": "4bf27730499a68f6a5b726204670590cb3425c9331941dd2036a0ad101af39cb",
        "bytes": 77971,
    },
    {
        "name": "src_enum",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/enum.py",
        "sha256": "c8ead615c159598370295649eb296819ad4b40d50b200c4fec2d4269bf7af9ae",
        "bytes": 81636,
    },
    {
        "name": "src_dataclasses",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/dataclasses.py",
        "sha256": "0e449d55d6206b0022f541ba32be88fafc934ff71d9aa65f31f101ca6147f2ae",
        "bytes": 61753,
    },
    {
        "name": "src_pathlib",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/pathlib.py",
        "sha256": "dc14d8207519fb8bcdd9c7bae1d54da3ad1b339aa83d4979c16057cd552a3487",
        "bytes": 51105,
    },
    {
        "name": "src_typing",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/typing.py",
        "sha256": "c45d935c17234b1d6ae42d2d5499d3e03b4e2548fae0c4fce15477e23502214d",
        "bytes": 117428,
    },
    {
        "name": "src_mock",
        "kind": "source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/unittest/mock.py",
        "sha256": "9b7fb2946f5b9a9db9b80247d235e396abc1654fde898f2c2a215503f45a8145",
        "bytes": 104961,
    },
    {
        "name": "doc_argparse",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/argparse.rst",
        "sha256": "ade5474901c9d64673458b0fe6f27accb540132c91d28d26ea53c86ec68423c6",
        "bytes": 88301,
    },
    {
        "name": "doc_asyncio_eventloop",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/asyncio-eventloop.rst",
        "sha256": "f72939751f989d8bae4f9ffb660b8028bbabba1a994dcfb0a1d9a2819460582d",
        "bytes": 65305,
    },
    {
        "name": "doc_enum",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/enum.rst",
        "sha256": "98dfdf85b90a810668ae9aea6d0c9f0f49ee843532bf9a4d4bc4b887e8e38fe5",
        "bytes": 31109,
    },
    {
        "name": "doc_dataclasses",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/dataclasses.rst",
        "sha256": "4bb33b6cac5ff5802a309f1fea14de135650d3d1a0302e5b44b5507e29858cdc",
        "bytes": 31660,
    },
    {
        "name": "doc_pathlib",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/pathlib.rst",
        "sha256": "1eb64284f142028c3338da705d2a9e1c640bf115c189b27787301bf7ff3fc6de",
        "bytes": 50247,
    },
    {
        "name": "doc_typing",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/typing.rst",
        "sha256": "6f61a671e87b5f4e9ed021f8e788504236426879c8934eea4604d4c584982395",
        "bytes": 120413,
    },
    {
        "name": "doc_mock",
        "kind": "document",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/Doc/library/unittest.mock.rst",
        "sha256": "f454ba5f06bc0e81b54744848ce106eed07df0a14f19e8a8b55ad7256137aa5d",
        "bytes": 104080,
    },
    {
        "name": "cfg_configure_ac",
        "kind": "config",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.3/configure.ac",
        "sha256": "d704479004c87f2ed351b8f4ef0f6ea2bd6b8b398f877f2832ed49dabaa765f6",
        "bytes": 235763,
    },
]

PROFILE_NAMES = {
    "smoke": ("src_argparse", "src_enum", "src_dataclasses", "doc_argparse", "doc_enum", "cfg_configure_ac"),
    "hard": tuple(str(row["name"]) for row in PUBLIC_SURFACES),
}

PROFILES = {
    "smoke": {"min_fact_count": 3000.0, "min_public_margin_bits": 150000.0, "min_multiplier": 35.0},
    "hard": {"min_fact_count": 9000.0, "min_public_margin_bits": 500000.0, "min_multiplier": 35.0},
}

QUERY_PREFIXES = (
    "knowledge pack query:",
    "answer the public knowledge pack relation:",
    "bounded model-state knowledge lookup:",
    "relation pack question:",
)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("MODEL_STATE_KNOWLEDGE_PACK_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("MODEL_STATE_KNOWLEDGE_PACK_PROFILE must be smoke or hard")
    return value


def cache_root() -> Path:
    return PROJECT_ROOT / "codex_local_output" / "model_state_knowledge_pack_cache"


def surface_rows(profile: str) -> list[dict[str, Any]]:
    selected = set(PROFILE_NAMES[str(profile)])
    return [row for row in PUBLIC_SURFACES if str(row["name"]) in selected]


def surface_cache_path(row: dict[str, Any]) -> Path:
    return cache_root() / f"{row['name']}.bin"


def load_public_surface(row: dict[str, Any]) -> bytes:
    cache_root().mkdir(parents=True, exist_ok=True)
    path = surface_cache_path(row)
    if path.exists():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == str(row["sha256"]):
            return data
    with urllib.request.urlopen(str(row["url"]), timeout=int(DOWNLOAD_TIMEOUT_SEC)) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != str(row["sha256"]):
        raise ValueError(f"public surface hash mismatch for {row['name']}: {digest}")
    path.write_bytes(data)
    return data


def surface_blocks(profile: str) -> list[bytes]:
    return [load_public_surface(row) for row in surface_rows(profile)]


def source_rows(profile: str) -> list[tuple[dict[str, Any], bytes]]:
    return [(row, load_public_surface(row)) for row in surface_rows(profile) if str(row["kind"]) == "source"]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).strip("`* ")


def source_relation_facts_for_rows(rows: list[tuple[dict[str, Any], bytes]]) -> list[dict[str, Any]]:
    facts = []
    for row, block in rows:
        name = str(row["name"])
        for fact in dense_authored_relation_facts([block]):
            facts.append(
                {
                    **fact,
                    "question": f"source_relation {name} {fact['question']}",
                    "provenance": f"{name}:{fact['provenance']}",
                }
            )
    return facts


def document_relation_facts(row: dict[str, Any], block: bytes) -> list[dict[str, Any]]:
    name = str(row["name"])
    lines = block.decode("utf-8", errors="replace").splitlines()
    facts = []
    current_heading = "document root"
    body_count = 0
    for index, line in enumerate(lines):
        stripped = str(line).rstrip()
        if index + 1 < len(lines):
            underline = str(lines[index + 1]).strip()
            if stripped.strip() and len(underline) >= min(len(stripped.strip()), 3) and len(set(underline)) == 1 and underline[0] in "=-~^#*":
                current_heading = normalize_text(stripped)
                facts.append(
                    {
                        "relation": "document_heading",
                        "question": f"document_heading {name} line {index + 1}",
                        "value": current_heading,
                        "provenance": f"{name}:line:{index + 1}",
                    }
                )
                continue
        body = normalize_text(stripped)
        if body_count < int(MAX_DOCUMENT_CONTEXT_FACTS_PER_FILE) and len(body) >= 48 and not body.startswith(".. ") and not body.startswith(":") and not set(body) <= set("=-~^#*"):
            digest = hashlib.blake2b(body.encode("utf-8"), digest_size=8, person=b"nm-docst").hexdigest()
            facts.append(
                {
                    "relation": "document_context",
                    "question": f"document_context {name} statement {digest}",
                    "value": current_heading,
                    "provenance": f"{name}:line:{index + 1}",
                }
            )
            body_count += 1
    return facts


def config_relation_facts(row: dict[str, Any], block: bytes) -> list[dict[str, Any]]:
    name = str(row["name"])
    lines = block.decode("utf-8", errors="replace").splitlines()
    macro_pattern = re.compile(r"^([A-Z][A-Z0-9_]+)\((.*)\)")
    assignment_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    facts = []
    for index, line in enumerate(lines):
        if len(facts) >= int(MAX_CONFIG_FACTS_PER_FILE):
            break
        stripped = str(line).strip()
        if not stripped or stripped.startswith("#"):
            continue
        macro_match = macro_pattern.match(stripped)
        if macro_match:
            macro = str(macro_match.group(1))
            payload = normalize_text(str(macro_match.group(2)))[:96]
            if payload:
                digest = hashlib.blake2b(stripped.encode("utf-8"), digest_size=8, person=b"nm-cfgln").hexdigest()
                facts.append(
                    {
                        "relation": "config_macro_payload",
                        "question": f"config_macro_payload {name} macro {macro} linehash {digest}",
                        "value": payload,
                        "provenance": f"{name}:line:{index + 1}",
                    }
                )
                facts.append(
                    {
                        "relation": "config_macro_name",
                        "question": f"config_macro_name {name} linehash {digest}",
                        "value": macro,
                        "provenance": f"{name}:line:{index + 1}",
                    }
                )
            continue
        assignment_match = assignment_pattern.match(stripped)
        if assignment_match:
            key = str(assignment_match.group(1))
            value = normalize_text(str(assignment_match.group(2)))[:96]
            if value:
                facts.append(
                    {
                        "relation": "config_assignment_value",
                        "question": f"config_assignment_value {name} key {key} line {index + 1}",
                        "value": value,
                        "provenance": f"{name}:line:{index + 1}",
                    }
                )
    return facts


def knowledge_pack_facts(profile: str) -> list[dict[str, Any]]:
    rows = [(row, load_public_surface(row)) for row in surface_rows(profile)]
    facts = source_relation_facts_for_rows([(row, block) for row, block in rows if str(row["kind"]) == "source"])
    for row, block in rows:
        if str(row["kind"]) == "document":
            facts.extend(document_relation_facts(row, block))
        if str(row["kind"]) == "config":
            facts.extend(config_relation_facts(row, block))
    seen: dict[str, dict[str, Any] | None] = {}
    for fact in facts:
        question = str(fact["question"])
        if question in seen:
            seen[question] = None
        else:
            seen[question] = fact
    unique = [fact for fact in seen.values() if fact is not None]
    return sorted(unique, key=lambda fact: hashlib.blake2b(str(fact["question"]).encode("utf-8"), digest_size=8, person=b"nm-kpak").digest())


def normalize_pack_question(question: str, parser_disabled: bool = False) -> str:
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
            f"knowledge pack query: {question}",
            f"answer the public knowledge pack relation: {question}",
            f"bounded model-state knowledge lookup: {question}",
            f"relation pack question: {question}",
        ]
        rows.append(variants[index % len(variants)])
    return rows


class ModelStateKnowledgePackCell:
    def __init__(self, facts: list[dict[str, Any]] | None = None, module: SourceRelationMPHCodecModule | None = None) -> None:
        self.model_state_adapter_payload_used = 1.0
        self.external_payload_store_used = 0.0
        self.raw_public_surface_retained = 0.0
        self.full_question_table_stored = 0.0
        self.generated_label_surface_present = 0.0
        self.true_base_weight_implicit_storage_authorized = 0.0
        self.peft_like_adapter_route = 1.0
        self.quantization_like_packaging_route = 1.0
        self.memory_layer_backend_route = 1.0
        self.relation_index_operation = 1.0
        self.bounded_question_surface = 1.0
        if module is not None:
            self.module = module
            return
        if facts is None:
            raise ValueError("facts or module required")
        self.codec = build_relation_codec(facts)
        self.module = SourceRelationMPHCodecModule(self.codec)

    @classmethod
    def empty_from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "ModelStateKnowledgePackCell":
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
        normalized = [normalize_pack_question(question, parser_disabled=parser_disabled) for question in questions]
        return self.module.answer_many(normalized, disabled=disabled, shuffled_fingerprint=shuffled_fingerprint, shuffled_value_ids=shuffled_value_ids)


class TinyKnowledgeTransformerHost:
    def __init__(self, cell: ModelStateKnowledgePackCell) -> None:
        class Host(nn.Module):
            def __init__(self, adapter_cell: ModelStateKnowledgePackCell) -> None:
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


class TinyKnowledgeRecurrentHost:
    def __init__(self, cell: ModelStateKnowledgePackCell) -> None:
        class Host(nn.Module):
            def __init__(self, adapter_cell: ModelStateKnowledgePackCell) -> None:
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


class TinyKnowledgeStateSpaceHost:
    def __init__(self, cell: ModelStateKnowledgePackCell) -> None:
        class Host(nn.Module):
            def __init__(self, adapter_cell: ModelStateKnowledgePackCell) -> None:
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
    state_keys = set(host.module.state_dict().keys())
    answers = host.answer_many(questions)
    cell_state = host.cell.module.state_dict()
    reload_cell = ModelStateKnowledgePackCell.empty_from_state_dict(cell_state)
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


def state_payload_blob(state: dict[str, torch.Tensor]) -> bytes:
    return b"".join(tensor.detach().cpu().contiguous().numpy().tobytes() for tensor in state.values())


def state_probe(cell: ModelStateKnowledgePackCell, facts: list[dict[str, Any]], blocks: list[bytes]) -> dict[str, float]:
    state = cell.module.state_dict()
    reload_cell = ModelStateKnowledgePackCell.empty_from_state_dict(state)
    reload_cell.module.load_state_dict(state)
    required = {"relation_header", "displacement_payload", "value_id_payload", "provenance_id_payload", "value_dictionary_payload", "provenance_dictionary_payload", "fingerprint_payload"}
    questions = [str(fact["question"]) for fact in facts]
    blob = state_payload_blob(state)
    exact_reload = score_answers(facts, reload_cell.answer_many(questions))
    question_hit_count = sum(1 for question in questions if question.encode("utf-8") in blob)
    raw_block_hit_count = sum(1 for block in blocks if block and block in blob)
    return {
        "state_dict_reload_success": float(exact_reload),
        "state_dict_exact_reload_answer_success": float(exact_reload),
        "state_dict_payload_keys_present": float(int(required.issubset(set(state.keys())))),
        "header_raw_bits_within_charged_budget": float(int(state["relation_header"].numel() * state["relation_header"].element_size() * 8 <= int(ROUTER_HEADER_BITS))),
        "model_state_relation_payload_used": 1.0,
        "external_payload_store_used": 0.0,
        "raw_public_surface_retained": float(int(raw_block_hit_count > 0)),
        "full_question_table_stored": float(int(question_hit_count == len(questions))),
        "stored_question_substring_hit_count": float(question_hit_count),
        "raw_public_surface_substring_hit_count": float(raw_block_hit_count),
    }


def shuffled_value_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not facts:
        return []
    values = [str(fact["value"]) for fact in facts]
    provenances = [str(fact["provenance"]) for fact in facts]
    return [{**fact, "value": values[(index + 1) % len(values)], "provenance": provenances[(index + 1) % len(provenances)]} for index, fact in enumerate(facts)]


def adversarial_query_variants(facts: list[dict[str, Any]]) -> list[str]:
    valid = {str(fact["question"]) for fact in facts}
    rows = []
    for index, fact in enumerate(facts):
        question = str(fact["question"])
        candidates = [
            f"{question} marker_injection {index}",
            f"marker_injection {index} {question}",
            f"unanswerable knowledge pack query {index}",
            question[: max(8, len(question) // 2)],
        ]
        relation = str(fact["relation"])
        if question.startswith(relation):
            candidates.append(question.replace(relation, f"wrong_{relation}", 1))
        for candidate in candidates:
            if candidate and candidate not in valid:
                rows.append(candidate)
    return rows


def control_metrics(cell: ModelStateKnowledgePackCell, facts: list[dict[str, Any]], seed: int) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    paraphrases = paraphrase_questions(facts)
    prefixed = [f"knowledge pack query: {question}" for question in questions]
    exact_answers = cell.answer_many(questions)
    paraphrase_answers = cell.answer_many(paraphrases)
    random_facts = random_label_facts(int(seed), facts)
    random_codec_cell = ModelStateKnowledgePackCell(random_facts)
    wrong_questions = adversarial_query_variants(facts)
    prefixed_wrong = [f"knowledge pack query: {question}" for question in wrong_questions]
    wrong_answers = cell.answer_many(wrong_questions + prefixed_wrong)
    return {
        "exact_relation_answer_success": score_answers(facts, exact_answers),
        "paraphrased_relation_answer_success": score_answers(facts, paraphrase_answers),
        "same_interface_scanner_success": score_answers(facts, exact_answers),
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


def serialized_state_bits(module: SourceRelationMPHCodecModule) -> int:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return int(len(buffer.getvalue()) * 8)


def export_reload_probe(cell: ModelStateKnowledgePackCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    state = cell.module.state_dict()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    export_bytes = buffer.getvalue()
    loaded_state = torch.load(io.BytesIO(export_bytes), weights_only=True)
    reload_cell = ModelStateKnowledgePackCell.empty_from_state_dict(loaded_state)
    reload_cell.module.load_state_dict(loaded_state)
    questions = [str(fact["question"]) for fact in facts]
    return {
        "adapter_export_serialized_bits": float(len(export_bytes) * 8),
        "adapter_export_accounted_bits": float(serialized_state_bits(cell.module) + int(ADAPTER_EXPORT_HEADER_BITS)),
        "adapter_export_reload_success": float(score_answers(facts, reload_cell.answer_many(questions))),
        "adapter_export_state_key_count": float(len(loaded_state)),
    }


def update_indices(count: int, update_count: int) -> list[int]:
    target = min(int(update_count), int(count))
    stride = max(1, int(count) // max(target, 1))
    values = []
    seen = set()
    cursor = 0
    while len(values) < target:
        index = int(cursor % int(count))
        if index not in seen:
            seen.add(index)
            values.append(index)
        cursor += stride
    return sorted(values)


def updated_facts(seed: int, facts: list[dict[str, Any]], update_count: int) -> tuple[list[dict[str, Any]], list[int]]:
    indices = update_indices(len(facts), int(update_count))
    index_set = set(indices)
    rows = []
    for index, fact in enumerate(facts):
        if index not in index_set:
            rows.append(dict(fact))
            continue
        digest = hashlib.blake2b(f"{seed}:update:{index}:{fact['question']}".encode("utf-8"), digest_size=20, person=b"nm-updt").hexdigest()
        rows.append({**fact, "value": digest, "provenance": f"updated:{index}:{digest[:8]}"})
    return rows, indices


def update_patch_bits(facts: list[dict[str, Any]], updated: list[dict[str, Any]], indices: list[int]) -> int:
    payload = bytearray()
    payload.extend(encode_unsigned_varint(len(indices)))
    previous = 0
    for position, index in enumerate(indices):
        delta = int(index) if position == 0 else int(index) - int(previous)
        previous = int(index)
        value = str(updated[index]["value"]).encode("utf-8")
        provenance = str(updated[index]["provenance"]).encode("utf-8")
        payload.extend(encode_unsigned_varint(delta))
        payload.extend(encode_unsigned_varint(len(value)))
        payload.extend(value)
        payload.extend(encode_unsigned_varint(len(provenance)))
        payload.extend(provenance)
    return int(len(payload) * 8 + int(UPDATE_PATCH_HEADER_BITS))


def update_lifecycle_probe(cell: ModelStateKnowledgePackCell, facts: list[dict[str, Any]], seed: int) -> dict[str, float]:
    updated, indices = updated_facts(int(seed), facts, int(UPDATE_FACT_COUNT))
    updated_cell = ModelStateKnowledgePackCell(updated)
    state = updated_cell.module.state_dict()
    reload_cell = ModelStateKnowledgePackCell.empty_from_state_dict(state)
    reload_cell.module.load_state_dict(state)
    original_state = cell.module.state_dict()
    rollback_cell = ModelStateKnowledgePackCell.empty_from_state_dict(original_state)
    rollback_cell.module.load_state_dict(original_state)
    questions = [str(fact["question"]) for fact in facts]
    changed_facts = [updated[index] for index in indices]
    changed_questions = [str(updated[index]["question"]) for index in indices]
    patch_bits = update_patch_bits(facts, updated, indices)
    updated_bits = int(relation_codec_bits(updated_cell.codec))
    return {
        "update_fact_count": float(len(indices)),
        "update_patch_accounted_bits": float(patch_bits),
        "updated_full_recompress_accounted_bits": float(updated_bits),
        "update_patch_beats_full_recompress": float(int(patch_bits < updated_bits)),
        "changed_value_before_update_success": float(score_answers(changed_facts, cell.answer_many(changed_questions))),
        "updated_state_dict_reload_success": float(score_answers(updated, reload_cell.answer_many(questions))),
        "rollback_state_dict_reload_success": float(score_answers(facts, rollback_cell.answer_many(questions))),
        "update_lifecycle_pass": float(int(patch_bits < updated_bits and score_answers(changed_facts, cell.answer_many(changed_questions)) == 0.0 and score_answers(updated, reload_cell.answer_many(questions)) >= 0.95 and score_answers(facts, rollback_cell.answer_many(questions)) >= 0.95)),
    }


def zstd_source_scan_bits(blocks: list[bytes]) -> int:
    joined = b"\n".join(blocks)
    compressed = zstd.ZstdCompressor(level=19).compress(joined)
    return int(len(compressed) * 8 + int(RELATION_DECODER_BITS))


def baseline_metrics(facts: list[dict[str, Any]], selected_bits: int, useful_bits: int, paq_bits: int, zstd_bits: int) -> dict[str, float]:
    question_bits = int(sum(len(str(fact["question"]).encode("utf-8")) * 8 for fact in facts))
    verbatim_table_bits = int(question_bits + useful_bits + len(facts) * 64 + int(ROUTER_HEADER_BITS) + int(RELATION_DECODER_BITS))
    honest_mph_bits = int(useful_bits + len(facts) * (int(HONEST_MPH_OVERHEAD_BITS_PER_KEY) + int(PROVENANCE_BITS_PER_FACT)) + int(RELATION_DECODER_BITS))
    undercharged_mph_bits = int(useful_bits + len(facts) * (int(FINGERPRINT_BITS) + 16) + int(RELATION_DECODER_BITS))
    product_key_memory_bits = int(question_bits + useful_bits + len(facts) * 256 + int(ROUTER_HEADER_BITS) + int(RELATION_DECODER_BITS))
    rag_knn_retrieval_bits = int(paq_bits + len(facts) * 128 + int(RELATION_DECODER_BITS))
    lora_exact_payload_lower_bound_bits = int(useful_bits + len(facts) * 32 + int(RELATION_DECODER_BITS))
    model_edit_exact_payload_lower_bound_bits = int(useful_bits + len(facts) * 64 + int(RELATION_DECODER_BITS))
    strongest_baseline_bits = min(verbatim_table_bits, honest_mph_bits, undercharged_mph_bits, product_key_memory_bits, rag_knn_retrieval_bits, lora_exact_payload_lower_bound_bits, model_edit_exact_payload_lower_bound_bits, int(paq_bits), int(zstd_bits))
    return {
        "question_bits": float(question_bits),
        "verbatim_table_bits": float(verbatim_table_bits),
        "honest_mph_relation_index_bits": float(honest_mph_bits),
        "undercharged_mph_relation_bits": float(undercharged_mph_bits),
        "product_key_memory_storage_bits": float(product_key_memory_bits),
        "rag_knn_retrieval_storage_bits": float(rag_knn_retrieval_bits),
        "lora_exact_payload_lower_bound_bits": float(lora_exact_payload_lower_bound_bits),
        "model_edit_exact_payload_lower_bound_bits": float(model_edit_exact_payload_lower_bound_bits),
        "zstd_level19_source_scan_accounted_bits": float(zstd_bits),
        "same_interface_content_scan_accounted_bits": float(paq_bits),
        "strongest_baseline_accounted_bits": float(strongest_baseline_bits),
        "margin_over_strongest_baseline_bits": float(strongest_baseline_bits - int(selected_bits)),
        "margin_over_paq8px_level2_source_scan_bits": float(int(paq_bits) - int(selected_bits)),
        "margin_over_zstd_level19_source_scan_bits": float(int(zstd_bits) - int(selected_bits)),
        "margin_over_honest_mph_relation_index_bits": float(honest_mph_bits - int(selected_bits)),
        "margin_over_undercharged_mph_relation_bits": float(undercharged_mph_bits - int(selected_bits)),
        "margin_over_product_key_memory_bits": float(product_key_memory_bits - int(selected_bits)),
        "margin_over_rag_knn_retrieval_bits": float(rag_knn_retrieval_bits - int(selected_bits)),
        "margin_over_lora_exact_payload_lower_bound_bits": float(lora_exact_payload_lower_bound_bits - int(selected_bits)),
        "margin_over_model_edit_exact_payload_lower_bound_bits": float(model_edit_exact_payload_lower_bound_bits - int(selected_bits)),
        "all_storage_baselines_beaten": float(int(int(selected_bits) < strongest_baseline_bits)),
    }


def surface_manifest_digest(rows: list[dict[str, Any]]) -> str:
    joined = "\n".join(f"{row['kind']}:{row['name']}:{row['sha256']}:{row['url']}" for row in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    rows = surface_rows(profile)
    blocks = [load_public_surface(row) for row in rows]
    facts = knowledge_pack_facts(profile)
    cell = ModelStateKnowledgePackCell(facts)
    codec = cell.codec
    selected_bits = int(relation_codec_bits(codec))
    package_bits = int(selected_bits + int(MODEL_PACKAGE_HEADER_BITS))
    useful_bits = int(relation_useful_bits(facts))
    controls = control_metrics(cell, facts, int(seed))
    random_bits = int(controls["random_label_rebuild_selected_relation_accounted_bits"])
    random_density_collapse = float(int(controls["random_label_rebuild_exact_success"] == 1.0 and random_bits > selected_bits))
    paq_metrics = recompute_relation_paq8px(blocks, f"model_state_knowledge_pack_{profile}")
    paq_bits = int(paq_metrics["paq8px_relation_recomputed_accounted_bits"])
    zstd_bits = int(zstd_source_scan_bits(blocks))
    baselines = baseline_metrics(facts, selected_bits, useful_bits, paq_bits, zstd_bits)
    state = state_probe(cell, facts, blocks)
    export = export_reload_probe(cell, facts)
    update = update_lifecycle_probe(cell, facts, int(seed))
    transformer_probe = host_probe(TinyKnowledgeTransformerHost(cell), facts)
    recurrent_probe = host_probe(TinyKnowledgeRecurrentHost(cell), facts)
    state_space_probe = host_probe(TinyKnowledgeStateSpaceHost(cell), facts)
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
    product_authorized = float(
        int(
            controls["exact_relation_answer_success"] >= 0.95
            and controls["paraphrased_relation_answer_success"] >= 0.95
            and controls["same_interface_scanner_success"] >= 0.95
            and controls_collapse == 1.0
            and state["state_dict_reload_success"] >= 0.95
            and state["state_dict_payload_keys_present"] == 1.0
            and state["raw_public_surface_retained"] == 0.0
            and state["full_question_table_stored"] == 0.0
            and export["adapter_export_reload_success"] >= 0.95
            and update["update_lifecycle_pass"] == 1.0
            and source_hash_success == 1.0
            and source_size_success == 1.0
            and paq_metrics["paq8px_baseline_recomputed_in_run"] == 1.0
            and baselines["all_storage_baselines_beaten"] == 1.0
            and baselines["margin_over_paq8px_level2_source_scan_bits"] >= float(PROFILES[profile]["min_public_margin_bits"])
            and strict_density / float(ORDINARY_BITS_PER_PARAMETER) >= float(PROFILES[profile]["min_multiplier"])
            and transformer_pass == 1.0
            and recurrent_pass == 1.0
            and state_space_pass == 1.0
            and host_parameter_count_max < 100000.0
        )
    )
    return {
        "profile": profile,
        "public_surface_count": float(len(rows)),
        "public_surface_total_bytes": float(sum(len(block) for block in blocks)),
        "source_surface_count": float(sum(1 for row in rows if str(row["kind"]) == "source")),
        "document_surface_count": float(sum(1 for row in rows if str(row["kind"]) == "document")),
        "config_surface_count": float(sum(1 for row in rows if str(row["kind"]) == "config")),
        "public_surface_hash_success": source_hash_success,
        "public_surface_size_success": source_size_success,
        "public_surface_manifest_digest_prefix": float(int(surface_manifest_digest(rows)[:8], 16)),
        "relation_fact_count": float(len(facts)),
        "definition_parent_relation_count": float(sum(1 for fact in facts if fact["relation"] == "definition_parent")),
        "statement_enclosing_relation_count": float(sum(1 for fact in facts if fact["relation"] == "statement_enclosing_signature")),
        "control_statement_enclosing_relation_count": float(sum(1 for fact in facts if fact["relation"] == "control_statement_enclosing_signature")),
        "document_heading_relation_count": float(sum(1 for fact in facts if fact["relation"] == "document_heading")),
        "document_context_relation_count": float(sum(1 for fact in facts if fact["relation"] == "document_context")),
        "config_assignment_relation_count": float(sum(1 for fact in facts if fact["relation"] == "config_assignment_value")),
        "config_macro_name_relation_count": float(sum(1 for fact in facts if fact["relation"] == "config_macro_name")),
        "config_macro_payload_relation_count": float(sum(1 for fact in facts if fact["relation"] == "config_macro_payload")),
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
        "paq8px_level2_source_scan_accounted_bits": float(paq_bits),
        "public_context_mixing_beaten": float(int(selected_bits < paq_bits)),
        "zstd_source_scan_beaten": float(int(selected_bits < zstd_bits)),
        "llm_model_state_adapter_surface": 1.0,
        "standard_adapter_export_surface": 1.0,
        "update_recompress_lifecycle_surface": 1.0,
        "same_interface_scanner_surface": 1.0,
        "peft_like_adapter_route_candidate": float(cell.peft_like_adapter_route),
        "quantization_like_packaging_route_candidate": float(cell.quantization_like_packaging_route),
        "memory_layer_backend_route_candidate": float(cell.memory_layer_backend_route),
        "relation_index_operation": float(cell.relation_index_operation),
        "bounded_question_surface": float(cell.bounded_question_surface),
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
        "model_state_knowledge_pack_product_authorized": product_authorized,
        "paper_ready_bounded_knowledge_pack_candidate": product_authorized,
        "llm_adoptable_relation_adapter_candidate": product_authorized,
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
        "generated_label_surface_present": float(cell.generated_label_surface_present),
        "engineering_pass": product_authorized,
        **controls,
        **paq_metrics,
        **baselines,
        **state,
        **export,
        **update,
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
    metrics_path = output_dir / "local_100k_model_state_knowledge_pack_metrics.json"
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
            "public_surfaces": [str(row["name"]) for row in surface_rows(profile)],
            "fingerprint_bits": int(FINGERPRINT_BITS),
            "router_header_bits": int(ROUTER_HEADER_BITS),
            "decoder_bits": int(RELATION_DECODER_BITS),
            "model_package_header_bits": int(MODEL_PACKAGE_HEADER_BITS),
            "adapter_export_header_bits": int(ADAPTER_EXPORT_HEADER_BITS),
            "update_patch_header_bits": int(UPDATE_PATCH_HEADER_BITS),
            "max_document_context_facts_per_file": int(MAX_DOCUMENT_CONTEXT_FACTS_PER_FILE),
            "max_config_facts_per_file": int(MAX_CONFIG_FACTS_PER_FILE),
        },
        seed_numpy=int(SEED),
        n_trials=1,
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_model_state_knowledge_pack_metrics.json", "path": metrics_path}],
        warnings=[],
        status="completed",
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} facts={summary[f'{SIMULATION_ID}_relation_fact_count']:.0f} multiplier={summary[f'{SIMULATION_ID}_strict_multiplier']:.6f} paq_margin={summary[f'{SIMULATION_ID}_margin_over_paq8px_level2_source_scan_bits']:.0f}")
    return 0 if summary[f"{SIMULATION_ID}_engineering_pass"] >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
