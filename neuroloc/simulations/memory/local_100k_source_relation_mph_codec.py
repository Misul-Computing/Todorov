from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from collections import defaultdict
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
from neuroloc.simulations.memory.local_100k_source_dense_authored_relation_diagnostic import dense_authored_relation_facts, read_limited_block, source_overlap_metrics
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import best_codec, decompress_best, decode_codec_name, encode_codec_name, shuffle_bytes
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import FROZEN_BLOCKS
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import decode_unsigned_varint, encode_unsigned_varint

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_relation_mph_codec"
SEED = env_int("SOURCE_RELATION_MPH_CODEC_SEED", 12829)
RELATION_DECODER_BITS = env_int("SOURCE_RELATION_MPH_CODEC_DECODER_BITS", 8192)
ROUTER_HEADER_BITS = env_int("SOURCE_RELATION_MPH_CODEC_ROUTER_HEADER_BITS", 4096)
FINGERPRINT_BITS = env_int("SOURCE_RELATION_MPH_CODEC_FINGERPRINT_BITS", 17)
UNDERCHARGED_MPH_OVERHEAD_BITS_PER_KEY = env_int("SOURCE_RELATION_MPH_CODEC_UNDERCHARGED_MPH_OVERHEAD_BITS_PER_KEY", 16)
HONEST_MPH_OVERHEAD_BITS_PER_KEY = env_int("SOURCE_RELATION_MPH_CODEC_HONEST_MPH_OVERHEAD_BITS_PER_KEY", 128)
PROVENANCE_BITS_PER_FACT = env_int("SOURCE_RELATION_MPH_CODEC_PROVENANCE_BITS_PER_FACT", 64)
PAQ8PX_LEVEL2_RELATION_ACCOUNTED_BITS = env_int("SOURCE_RELATION_MPH_CODEC_PAQ8PX_LEVEL2_RELATION_BITS", 261144)
PAQ8PX_LEVEL2_RAW_SOURCE_PAYLOAD_BITS = env_int("SOURCE_RELATION_MPH_CODEC_PAQ8PX_LEVEL2_RAW_SOURCE_PAYLOAD_BITS", 405696)
RECOMPUTE_PAQ8PX = env_int("SOURCE_RELATION_MPH_CODEC_RECOMPUTE_PAQ8PX", 1)
PAQ8PX_TIMEOUT_SEC = env_int("SOURCE_RELATION_MPH_CODEC_PAQ8PX_TIMEOUT_SEC", 300)
PAQ8PX_EXE = os.environ.get("SOURCE_RELATION_MPH_CODEC_PAQ8PX_EXE", "codex_local_output/compression_tools/paq8px_v214/paq8px.exe")
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("SOURCE_RELATION_MPH_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))

require_positive("SOURCE_RELATION_MPH_CODEC_DECODER_BITS", RELATION_DECODER_BITS)
require_positive("SOURCE_RELATION_MPH_CODEC_ROUTER_HEADER_BITS", ROUTER_HEADER_BITS)
require_positive("SOURCE_RELATION_MPH_CODEC_FINGERPRINT_BITS", FINGERPRINT_BITS)
require_positive("SOURCE_RELATION_MPH_CODEC_UNDERCHARGED_MPH_OVERHEAD_BITS_PER_KEY", UNDERCHARGED_MPH_OVERHEAD_BITS_PER_KEY)
require_positive("SOURCE_RELATION_MPH_CODEC_HONEST_MPH_OVERHEAD_BITS_PER_KEY", HONEST_MPH_OVERHEAD_BITS_PER_KEY)
require_positive("SOURCE_RELATION_MPH_CODEC_PROVENANCE_BITS_PER_FACT", PROVENANCE_BITS_PER_FACT)
require_positive("SOURCE_RELATION_MPH_CODEC_PAQ8PX_LEVEL2_RELATION_BITS", PAQ8PX_LEVEL2_RELATION_ACCOUNTED_BITS)
require_positive("SOURCE_RELATION_MPH_CODEC_PAQ8PX_LEVEL2_RAW_SOURCE_PAYLOAD_BITS", PAQ8PX_LEVEL2_RAW_SOURCE_PAYLOAD_BITS)
require_positive("SOURCE_RELATION_MPH_CODEC_PAQ8PX_TIMEOUT_SEC", PAQ8PX_TIMEOUT_SEC)

PROFILES = {
    "smoke": {"profile": "smoke", "min_fact_count": 2200.0, "min_public_margin_bits": 8000.0},
    "hard": {"profile": "hard", "min_fact_count": 3500.0, "min_public_margin_bits": 10000.0},
}
RELATION_PROFILE_INDICES = {
    "smoke": (0, 3),
    "hard": (0, 3, 4),
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_RELATION_MPH_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_RELATION_MPH_CODEC_PROFILE must be smoke or hard")
    return value


def relation_rows(profile: str) -> list[dict[str, Any]]:
    return [FROZEN_BLOCKS[int(index)] for index in RELATION_PROFILE_INDICES[str(PROFILES[profile]["profile"])]]


def relation_blocks(profile: str) -> list[bytes]:
    return [read_limited_block(row) for row in relation_rows(profile)]


def hash_mod(key: bytes, person: bytes, mod: int, extra: bytes = b"") -> int:
    return int.from_bytes(hashlib.blake2b(key + extra, digest_size=8, person=person).digest(), "little") % int(mod)


def key_fingerprint(key: bytes, bits: int = FINGERPRINT_BITS) -> int:
    mask = (1 << int(bits)) - 1
    return int.from_bytes(hashlib.blake2b(key, digest_size=8, person=b"nm-fing").digest(), "little") & mask


def build_mph(keys: list[bytes]) -> tuple[list[int], list[int]]:
    count = len(keys)
    buckets: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
    for index, key in enumerate(keys):
        buckets[hash_mod(key, b"nm-buck", count)].append((int(index), key))
    ordered = sorted(buckets.items(), key=lambda item: -len(item[1]))
    displacements = [0 for _item in range(count)]
    slots: list[int | None] = [None for _item in range(count)]
    for bucket, rows in ordered:
        displacement = 0
        while True:
            seen: list[int] = []
            ok = True
            for _index, key in rows:
                slot = hash_mod(key, b"nm-slot", count, encode_unsigned_varint(displacement))
                if slots[slot] is not None or slot in seen:
                    ok = False
                    break
                seen.append(slot)
            if ok:
                displacements[int(bucket)] = int(displacement)
                for (index, _key), slot in zip(rows, seen):
                    slots[int(slot)] = int(index)
                break
            displacement += 1
            if displacement > 1000000:
                raise ValueError("minimal perfect hash construction failed")
    if any(slot is None for slot in slots):
        raise ValueError("minimal perfect hash left empty slots")
    return displacements, [int(slot) for slot in slots if slot is not None]


def encode_varint_rows(values: list[int]) -> bytes:
    out = bytearray()
    for value in values:
        out.extend(encode_unsigned_varint(int(value)))
    return bytes(out)


def decode_varint_rows(payload: bytes, count: int) -> list[int]:
    out = []
    index = 0
    for _item in range(int(count)):
        value, index = decode_unsigned_varint(payload, index)
        out.append(int(value))
    if index != len(payload):
        raise ValueError("unused varint payload bytes")
    return out


def encode_string_table(values: list[str]) -> bytes:
    for value in values:
        if b"\x00" in str(value).encode("utf-8"):
            raise ValueError("string table value contains separator")
    return b"\x00".join(str(value).encode("utf-8") for value in values)


def decode_string_table(payload: bytes, count: int) -> list[str]:
    if int(count) == 0:
        if payload:
            raise ValueError("nonempty zero-count string table")
        return []
    rows = payload.split(b"\x00")
    if len(rows) != int(count):
        raise ValueError("string table count mismatch")
    return [row.decode("utf-8") for row in rows]


def pack_fingerprints(values: list[int], bits: int = FINGERPRINT_BITS) -> bytes:
    out = bytearray()
    bit_buffer = 0
    bit_count = 0
    for value in values:
        bit_buffer |= int(value) << bit_count
        bit_count += int(bits)
        while bit_count >= 8:
            out.append(bit_buffer & 255)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count:
        out.append(bit_buffer & 255)
    return bytes(out)


def unpack_fingerprints(payload: bytes, count: int, bits: int = FINGERPRINT_BITS) -> list[int]:
    out = []
    bit_buffer = 0
    bit_count = 0
    index = 0
    mask = (1 << int(bits)) - 1
    while len(out) < int(count):
        while bit_count < int(bits):
            if index >= len(payload):
                raise ValueError("truncated fingerprint payload")
            bit_buffer |= int(payload[index]) << bit_count
            bit_count += 8
            index += 1
        out.append(int(bit_buffer & mask))
        bit_buffer >>= int(bits)
        bit_count -= int(bits)
    if any(byte != 0 for byte in payload[index:]) or bit_buffer:
        raise ValueError("unused fingerprint payload bytes")
    return out


def codec_name_value(name: str) -> int:
    return int(encode_codec_name(str(name)))


def relation_useful_bits(facts: list[dict[str, Any]]) -> int:
    return int(sum(len(str(fact["value"]).encode("utf-8")) * 8 + len(str(fact["provenance"]).encode("utf-8")) * 8 for fact in facts))


def build_relation_codec(facts: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [str(fact["question"]).encode("utf-8") for fact in facts]
    displacements, slots = build_mph(keys)
    value_map: dict[str, int] = {}
    provenance_map: dict[str, int] = {}
    values: list[str] = []
    provenances: list[str] = []
    for fact in facts:
        value = str(fact["value"])
        provenance = str(fact["provenance"])
        if value not in value_map:
            value_map[value] = len(values)
            values.append(value)
        if provenance not in provenance_map:
            provenance_map[provenance] = len(provenances)
            provenances.append(provenance)
    value_ids = []
    provenance_ids = []
    fingerprints = []
    for fact_index in slots:
        fact = facts[int(fact_index)]
        key = str(fact["question"]).encode("utf-8")
        value_ids.append(int(value_map[str(fact["value"])]))
        provenance_ids.append(int(provenance_map[str(fact["provenance"])]))
        fingerprints.append(key_fingerprint(key))
    displacement_raw = encode_varint_rows(displacements)
    value_id_raw = encode_varint_rows(value_ids)
    provenance_id_raw = encode_varint_rows(provenance_ids)
    value_dictionary_raw = encode_string_table(values)
    provenance_dictionary_raw = encode_string_table(provenances)
    fingerprint_payload = pack_fingerprints(fingerprints)
    displacement_codec_name, displacement_payload = best_codec(displacement_raw)
    value_id_codec_name, value_id_payload = best_codec(value_id_raw)
    provenance_id_codec_name, provenance_id_payload = best_codec(provenance_id_raw)
    value_dictionary_codec_name, value_dictionary_payload = best_codec(value_dictionary_raw)
    provenance_dictionary_codec_name, provenance_dictionary_payload = best_codec(provenance_dictionary_raw)
    return {
        "fact_count": int(len(facts)),
        "value_count": int(len(values)),
        "provenance_count": int(len(provenances)),
        "fingerprint_bits": int(FINGERPRINT_BITS),
        "displacement_codec_name": displacement_codec_name,
        "value_id_codec_name": value_id_codec_name,
        "provenance_id_codec_name": provenance_id_codec_name,
        "value_dictionary_codec_name": value_dictionary_codec_name,
        "provenance_dictionary_codec_name": provenance_dictionary_codec_name,
        "displacement_payload": displacement_payload,
        "value_id_payload": value_id_payload,
        "provenance_id_payload": provenance_id_payload,
        "value_dictionary_payload": value_dictionary_payload,
        "provenance_dictionary_payload": provenance_dictionary_payload,
        "fingerprint_payload": fingerprint_payload,
        "raw_displacement_bytes": int(len(displacement_raw)),
        "raw_value_id_bytes": int(len(value_id_raw)),
        "raw_provenance_id_bytes": int(len(provenance_id_raw)),
        "raw_value_dictionary_bytes": int(len(value_dictionary_raw)),
        "raw_provenance_dictionary_bytes": int(len(provenance_dictionary_raw)),
    }


def relation_codec_bits(codec: dict[str, Any]) -> int:
    return int(
        int(ROUTER_HEADER_BITS)
        + int(RELATION_DECODER_BITS)
        + len(bytes(codec["displacement_payload"])) * 8
        + len(bytes(codec["value_id_payload"])) * 8
        + len(bytes(codec["provenance_id_payload"])) * 8
        + len(bytes(codec["value_dictionary_payload"])) * 8
        + len(bytes(codec["provenance_dictionary_payload"])) * 8
        + len(bytes(codec["fingerprint_payload"])) * 8
    )


def paq8px_path() -> Path:
    path = Path(PAQ8PX_EXE)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def recompute_relation_paq8px(blocks: list[bytes], profile: str) -> dict[str, float]:
    constant_payload_bits = int(PAQ8PX_LEVEL2_RELATION_ACCOUNTED_BITS) - int(RELATION_DECODER_BITS)
    if int(RECOMPUTE_PAQ8PX) != 1:
        return {
            "paq8px_baseline_external_constant_used": 1.0,
            "paq8px_baseline_recomputed_in_run": 0.0,
            "paq8px_relation_recomputed_payload_bits": float(constant_payload_bits),
            "paq8px_relation_recomputed_accounted_bits": float(PAQ8PX_LEVEL2_RELATION_ACCOUNTED_BITS),
            "paq8px_relation_recomputed_archive_bytes": float(constant_payload_bits // 8),
            "paq8px_relation_recomputed_matches_constant": 0.0,
        }
    exe = paq8px_path()
    if not exe.exists():
        return {
            "paq8px_baseline_external_constant_used": 1.0,
            "paq8px_baseline_recomputed_in_run": 0.0,
            "paq8px_relation_recomputed_payload_bits": float(constant_payload_bits),
            "paq8px_relation_recomputed_accounted_bits": float(PAQ8PX_LEVEL2_RELATION_ACCOUNTED_BITS),
            "paq8px_relation_recomputed_archive_bytes": float(constant_payload_bits // 8),
            "paq8px_relation_recomputed_matches_constant": 0.0,
        }
    out_dir = PROJECT_ROOT / "codex_local_output" / "source_relation_mph_paq8px_recompute" / str(profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "relation_raw_joined.bin"
    archive_path = out_dir / "relation_raw_joined_paq2.paq8px214"
    input_path.write_bytes(b"\n".join(blocks))
    if archive_path.exists():
        archive_path.unlink()
    try:
        completed = subprocess.run([str(exe), "-2", str(input_path), str(archive_path)], cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=int(PAQ8PX_TIMEOUT_SEC), check=False)
        success = float(int(completed.returncode == 0 and archive_path.exists()))
    except Exception:
        success = 0.0
    if success == 1.0:
        payload_bits = int(archive_path.stat().st_size) * 8
    else:
        payload_bits = constant_payload_bits
    accounted_bits = int(payload_bits) + int(RELATION_DECODER_BITS)
    return {
        "paq8px_baseline_external_constant_used": 1.0,
        "paq8px_baseline_recomputed_in_run": float(success),
        "paq8px_relation_recomputed_payload_bits": float(payload_bits),
        "paq8px_relation_recomputed_accounted_bits": float(accounted_bits),
        "paq8px_relation_recomputed_archive_bytes": float(payload_bits // 8),
        "paq8px_relation_recomputed_matches_constant": float(int(success == 1.0 and accounted_bits == int(PAQ8PX_LEVEL2_RELATION_ACCOUNTED_BITS))),
    }


class SourceRelationMPHCodecModule(nn.Module):
    def __init__(self, codec: dict[str, Any] | None = None, state_shapes: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        if codec is None and state_shapes is None:
            raise ValueError("codec or state shapes required")
        if codec is not None:
            header = torch.tensor(
                [
                    int(codec["fact_count"]),
                    int(codec["value_count"]),
                    int(codec["provenance_count"]),
                    int(codec["fingerprint_bits"]),
                    codec_name_value(str(codec["displacement_codec_name"])),
                    codec_name_value(str(codec["value_id_codec_name"])),
                    codec_name_value(str(codec["provenance_id_codec_name"])),
                    codec_name_value(str(codec["value_dictionary_codec_name"])),
                    codec_name_value(str(codec["provenance_dictionary_codec_name"])),
                    int(ROUTER_HEADER_BITS),
                ],
                dtype=torch.int32,
            )
            self.register_buffer("relation_header", header, persistent=True)
            self.register_buffer("displacement_payload", torch.tensor(list(bytes(codec["displacement_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("value_id_payload", torch.tensor(list(bytes(codec["value_id_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("provenance_id_payload", torch.tensor(list(bytes(codec["provenance_id_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("value_dictionary_payload", torch.tensor(list(bytes(codec["value_dictionary_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("provenance_dictionary_payload", torch.tensor(list(bytes(codec["provenance_dictionary_payload"])), dtype=torch.uint8), persistent=True)
            self.register_buffer("fingerprint_payload", torch.tensor(list(bytes(codec["fingerprint_payload"])), dtype=torch.uint8), persistent=True)
        else:
            for name in ("relation_header", "displacement_payload", "value_id_payload", "provenance_id_payload", "value_dictionary_payload", "provenance_dictionary_payload", "fingerprint_payload"):
                self.register_buffer(name, torch.empty_like(state_shapes[name]), persistent=True)

    @classmethod
    def empty_from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "SourceRelationMPHCodecModule":
        return cls(state_shapes=state_dict)

    def payload_bytes(self, name: str) -> bytes:
        return bytes(int(item) for item in getattr(self, name).tolist())

    def decoded_state(self) -> dict[str, Any]:
        header = [int(item) for item in self.relation_header.tolist()]
        fact_count = int(header[0])
        value_count = int(header[1])
        provenance_count = int(header[2])
        fingerprint_bits = int(header[3])
        displacement_raw = decompress_best(decode_codec_name(header[4]), self.payload_bytes("displacement_payload"))
        value_id_raw = decompress_best(decode_codec_name(header[5]), self.payload_bytes("value_id_payload"))
        provenance_id_raw = decompress_best(decode_codec_name(header[6]), self.payload_bytes("provenance_id_payload"))
        value_dictionary_raw = decompress_best(decode_codec_name(header[7]), self.payload_bytes("value_dictionary_payload"))
        provenance_dictionary_raw = decompress_best(decode_codec_name(header[8]), self.payload_bytes("provenance_dictionary_payload"))
        return {
            "fact_count": fact_count,
            "value_count": value_count,
            "provenance_count": provenance_count,
            "fingerprint_bits": fingerprint_bits,
            "displacements": decode_varint_rows(displacement_raw, fact_count),
            "value_ids": decode_varint_rows(value_id_raw, fact_count),
            "provenance_ids": decode_varint_rows(provenance_id_raw, fact_count),
            "values": decode_string_table(value_dictionary_raw, value_count),
            "provenances": decode_string_table(provenance_dictionary_raw, provenance_count),
            "fingerprints": unpack_fingerprints(self.payload_bytes("fingerprint_payload"), fact_count, fingerprint_bits),
        }

    def answer_many(self, questions: list[str], disabled: bool = False, shuffled_fingerprint: bool = False, shuffled_value_ids: bool = False) -> list[dict[str, str | int]]:
        if disabled:
            return [{"hit": 0, "value": "", "provenance": ""} for _question in questions]
        try:
            state = self.decoded_state()
        except Exception:
            return [{"hit": 0, "value": "", "provenance": ""} for _question in questions]
        fingerprints = list(state["fingerprints"])
        value_ids = list(state["value_ids"])
        if shuffled_fingerprint and len(fingerprints) > 1:
            fingerprints = fingerprints[1:] + fingerprints[:1]
        if shuffled_value_ids and len(value_ids) > 1:
            value_ids = value_ids[1:] + value_ids[:1]
        out = []
        count = int(state["fact_count"])
        for question in questions:
            key = str(question).encode("utf-8")
            bucket = hash_mod(key, b"nm-buck", count)
            displacement = int(state["displacements"][bucket])
            slot = hash_mod(key, b"nm-slot", count, encode_unsigned_varint(displacement))
            if int(fingerprints[slot]) != key_fingerprint(key, int(state["fingerprint_bits"])):
                out.append({"hit": 0, "value": "", "provenance": ""})
                continue
            value_id = int(value_ids[slot])
            provenance_id = int(state["provenance_ids"][slot])
            if value_id < 0 or value_id >= len(state["values"]) or provenance_id < 0 or provenance_id >= len(state["provenances"]):
                out.append({"hit": 0, "value": "", "provenance": ""})
                continue
            out.append({"hit": 1, "value": str(state["values"][value_id]), "provenance": str(state["provenances"][provenance_id])})
        return out


def score_answers(facts: list[dict[str, Any]], answers: list[dict[str, str | int]]) -> float:
    if not facts:
        return 0.0
    values = []
    for fact, answer in zip(facts, answers):
        values.append(float(int(int(answer["hit"]) == 1 and str(answer["value"]) == str(fact["value"]) and str(answer["provenance"]) == str(fact["provenance"]))))
    return float(sum(values) / max(float(len(values)), 1.0))


def random_label_facts(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, fact in enumerate(facts):
        digest = hashlib.blake2b(f"{seed}:{index}:{fact['question']}".encode("utf-8"), digest_size=24, person=b"nm-rand").hexdigest()
        rows.append({**fact, "value": digest, "provenance": f"random:{index}"})
    return rows


def wrong_query_variants(facts: list[dict[str, Any]]) -> list[str]:
    relations = ("definition_parent", "statement_enclosing_signature", "control_statement_enclosing_signature")
    valid = {str(fact["question"]) for fact in facts}
    rows = []
    for index, fact in enumerate(facts):
        question = str(fact["question"])
        candidates = [f"{question} injected {index}", f"injected {index} {question}"]
        for relation in relations:
            prefix = f"{relation} "
            if question.startswith(prefix):
                alternate = relations[(relations.index(relation) + 1) % len(relations)]
                candidates.append(f"{alternate} {question[len(prefix):]}")
                break
        for candidate in candidates:
            if candidate not in valid:
                rows.append(candidate)
    return rows


def state_payload_blob(state: dict[str, torch.Tensor]) -> bytes:
    return b"".join(tensor.detach().cpu().contiguous().numpy().tobytes() for tensor in state.values())


def state_probe(module: SourceRelationMPHCodecModule, facts: list[dict[str, Any]], blocks: list[bytes]) -> dict[str, float]:
    state = module.state_dict()
    reload_module = SourceRelationMPHCodecModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    required = {"relation_header", "displacement_payload", "value_id_payload", "provenance_id_payload", "value_dictionary_payload", "provenance_dictionary_payload", "fingerprint_payload"}
    questions = [str(fact["question"]) for fact in facts]
    blob = state_payload_blob(state)
    exact_reload = score_answers(facts, reload_module.answer_many(questions))
    question_hit_count = sum(1 for question in questions if question.encode("utf-8") in blob)
    raw_block_hit_count = sum(1 for block in blocks if block and block in blob)
    return {
        "state_dict_reload_success": float(exact_reload),
        "state_dict_exact_reload_answer_success": float(exact_reload),
        "state_dict_payload_keys_present": float(int(required.issubset(set(state.keys())))),
        "header_raw_bits_within_charged_budget": float(int(state["relation_header"].numel() * state["relation_header"].element_size() * 8 <= int(ROUTER_HEADER_BITS))),
        "model_state_relation_payload_used": 1.0,
        "external_payload_store_used": 0.0,
        "raw_source_block_retained": float(int(raw_block_hit_count > 0)),
        "full_question_table_stored": float(int(question_hit_count == len(questions))),
        "stored_question_substring_hit_count": float(question_hit_count),
        "raw_source_block_substring_hit_count": float(raw_block_hit_count),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    blocks = relation_blocks(profile)
    facts = dense_authored_relation_facts(blocks)
    codec = build_relation_codec(facts)
    module = SourceRelationMPHCodecModule(codec)
    questions = [str(fact["question"]) for fact in facts]
    exact_success = score_answers(facts, module.answer_many(questions))
    random_facts = random_label_facts(int(seed), facts)
    random_codec = build_relation_codec(random_facts)
    random_module = SourceRelationMPHCodecModule(random_codec)
    random_label_success = score_answers(random_facts, module.answer_many(questions))
    random_label_rebuild_success = score_answers(random_facts, random_module.answer_many(questions))
    random_label_rebuild_bits = relation_codec_bits(random_codec)
    disabled_success = score_answers(facts, module.answer_many(questions, disabled=True))
    shuffled_fingerprint_success = score_answers(facts, module.answer_many(questions, shuffled_fingerprint=True))
    shuffled_value_success = score_answers(facts, module.answer_many(questions, shuffled_value_ids=True))
    wrong_questions = wrong_query_variants(facts)
    wrong_answers = module.answer_many(wrong_questions)
    wrong_query_hit_rate = float(sum(int(answer["hit"]) for answer in wrong_answers)) / max(float(len(wrong_answers)), 1.0)
    useful_bits = relation_useful_bits(facts)
    selected_bits = relation_codec_bits(codec)
    random_label_rebuild_density_control_collapse = float(int(random_label_rebuild_success == 1.0 and random_label_rebuild_bits > selected_bits))
    undercharged_mph_bits = int(useful_bits + len(facts) * (int(FINGERPRINT_BITS) + int(UNDERCHARGED_MPH_OVERHEAD_BITS_PER_KEY)) + int(RELATION_DECODER_BITS))
    honest_mph_bits = int(useful_bits + len(facts) * (int(HONEST_MPH_OVERHEAD_BITS_PER_KEY) + int(PROVENANCE_BITS_PER_FACT)) + int(RELATION_DECODER_BITS))
    raw_source_content_scan_bits = int(PAQ8PX_LEVEL2_RAW_SOURCE_PAYLOAD_BITS + RELATION_DECODER_BITS)
    paq_metrics = recompute_relation_paq8px(blocks, profile)
    paq_relation_bits = int(paq_metrics["paq8px_relation_recomputed_accounted_bits"])
    controls_collapse = float(int(random_label_success == 0.0 and random_label_rebuild_density_control_collapse == 1.0 and disabled_success == 0.0 and shuffled_fingerprint_success == 0.0 and wrong_query_hit_rate == 0.0))
    state = state_probe(module, facts, blocks)
    overlaps = source_overlap_metrics(relation_rows(profile))
    public_margin = float(paq_relation_bits - selected_bits)
    exact_pass = float(int(exact_success == 1.0 and state["state_dict_reload_success"] == 1.0 and state["state_dict_payload_keys_present"] == 1.0 and state["header_raw_bits_within_charged_budget"] == 1.0))
    self_contained_paq_win = float(int(paq_metrics["paq8px_baseline_recomputed_in_run"] == 1.0 and selected_bits < paq_relation_bits))
    baseline_pass = float(int(selected_bits < undercharged_mph_bits and selected_bits < honest_mph_bits and selected_bits < raw_source_content_scan_bits and selected_bits < paq_relation_bits and public_margin >= float(PROFILES[profile]["min_public_margin_bits"]) and self_contained_paq_win == 1.0))
    product_authorized = float(int(exact_pass == 1.0 and controls_collapse == 1.0 and baseline_pass == 1.0 and overlaps["source_train_test_path_overlap_count"] == 0.0 and overlaps["source_train_test_hash_overlap_count"] == 0.0))
    return {
        "profile": profile,
        "block_count": float(len(blocks)),
        "relation_fact_count": float(len(facts)),
        "definition_parent_relation_count": float(sum(1 for fact in facts if fact["relation"] == "definition_parent")),
        "statement_enclosing_relation_count": float(sum(1 for fact in facts if fact["relation"] == "statement_enclosing_signature")),
        "control_statement_enclosing_relation_count": float(sum(1 for fact in facts if fact["relation"] == "control_statement_enclosing_signature")),
        "value_dictionary_count": float(codec["value_count"]),
        "provenance_dictionary_count": float(codec["provenance_count"]),
        "fingerprint_bits_per_key": float(FINGERPRINT_BITS),
        "useful_retrievable_bits": float(useful_bits),
        "selected_relation_accounted_bits": float(selected_bits),
        "raw_source_paq_content_scan_bits": float(raw_source_content_scan_bits),
        "paq8px_level2_relation_accounted_bits": float(paq_relation_bits),
        "undercharged_mph_relation_bits": float(undercharged_mph_bits),
        "honest_mph_relation_index_bits": float(honest_mph_bits),
        "margin_over_raw_source_paq_content_scan_bits": float(raw_source_content_scan_bits - selected_bits),
        "margin_over_paq8px_level2_relation_bits": float(paq_relation_bits - selected_bits),
        "margin_over_undercharged_mph_relation_bits": float(undercharged_mph_bits - selected_bits),
        "margin_over_honest_mph_relation_index_bits": float(honest_mph_bits - selected_bits),
        "strict_density": float(useful_bits) / max(float(selected_bits) / 16.0, 1.0),
        "strict_multiplier": float(useful_bits) / max(float(selected_bits) / 16.0, 1.0) / float(ORDINARY_BITS_PER_PARAMETER),
        "displacement_payload_bits": float(len(bytes(codec["displacement_payload"])) * 8),
        "value_id_payload_bits": float(len(bytes(codec["value_id_payload"])) * 8),
        "provenance_id_payload_bits": float(len(bytes(codec["provenance_id_payload"])) * 8),
        "value_dictionary_payload_bits": float(len(bytes(codec["value_dictionary_payload"])) * 8),
        "provenance_dictionary_payload_bits": float(len(bytes(codec["provenance_dictionary_payload"])) * 8),
        "fingerprint_payload_bits": float(len(bytes(codec["fingerprint_payload"])) * 8),
        "router_header_bits": float(ROUTER_HEADER_BITS),
        "relation_decoder_bits": float(RELATION_DECODER_BITS),
        "raw_displacement_bytes": float(codec["raw_displacement_bytes"]),
        "raw_value_id_bytes": float(codec["raw_value_id_bytes"]),
        "raw_provenance_id_bytes": float(codec["raw_provenance_id_bytes"]),
        "raw_value_dictionary_bytes": float(codec["raw_value_dictionary_bytes"]),
        "raw_provenance_dictionary_bytes": float(codec["raw_provenance_dictionary_bytes"]),
        "exact_relation_answer_success": float(exact_success),
        "random_label_twin_success": float(random_label_success),
        "random_label_cross_label_success": float(random_label_success),
        "random_label_rebuild_exact_success": float(random_label_rebuild_success),
        "random_label_rebuild_selected_relation_accounted_bits": float(random_label_rebuild_bits),
        "random_label_rebuild_selected_bits_delta": float(random_label_rebuild_bits - selected_bits),
        "random_label_rebuild_density_control_collapse": float(random_label_rebuild_density_control_collapse),
        "decoder_disabled_success": float(disabled_success),
        "shuffled_fingerprint_success": float(shuffled_fingerprint_success),
        "shuffled_value_id_success": float(shuffled_value_success),
        "wrong_query_variant_count": float(len(wrong_questions)),
        "wrong_query_hit_rate": float(wrong_query_hit_rate),
        "controls_collapse": float(controls_collapse),
        **paq_metrics,
        "self_contained_paq8px_baseline_win_authorized": float(self_contained_paq_win),
        "public_context_mixing_beaten": float(int(selected_bits < paq_relation_bits)),
        "raw_source_content_scan_beaten": float(int(selected_bits < raw_source_content_scan_bits)),
        "undercharged_mph_beaten": float(int(selected_bits < undercharged_mph_bits)),
        "honest_mph_index_beaten": float(int(selected_bits < honest_mph_bits)),
        "source_relation_mph_codec_product_authorized": float(product_authorized),
        "source_relation_index_product_candidate": float(product_authorized),
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
        "generated_alias_labels_present": 0.0,
        "fixed_stride_relation_used": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "engineering_pass": float(product_authorized),
        **state,
        **overlaps,
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
    metrics_path = output_dir / "local_100k_source_relation_mph_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "fingerprint_bits": int(FINGERPRINT_BITS), "router_header_bits": int(ROUTER_HEADER_BITS), "relation_decoder_bits": int(RELATION_DECODER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_relation_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_relation_mph_codec_metrics.json", "path": metrics_path}],
        warnings=["source-authored relation mph codec only; not broad knowledge compression, chat, full nm, paid compute, or 600x proof"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} relations={summary[f'{SIMULATION_ID}_relation_fact_count']:.0f} selected_bits={summary[f'{SIMULATION_ID}_selected_relation_accounted_bits']:.0f} paq_margin_bits={summary[f'{SIMULATION_ID}_margin_over_paq8px_level2_relation_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
