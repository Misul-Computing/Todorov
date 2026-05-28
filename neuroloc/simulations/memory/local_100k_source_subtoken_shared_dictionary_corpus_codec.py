from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import best_codec, random_block, read_joined, train_paths
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_block_codec import learned_codec, substitute_subtokens
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import FROZEN_BLOCKS, corpus_blocks, read_block
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import delta_encode, dictionary_stream, tokens_from_stream
from neuroloc.simulations.memory.local_100k_source_token_structure_block_codec import delta_decode, restore_body, shuffle_bytes
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import decompress_best, learn_indent_unit, restore_structure, transform_structure

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_subtoken_shared_dictionary_corpus_codec"
SEED = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SEED", 12641)
MAX_BLOCK_BYTES = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_MAX_BLOCK_BYTES", 250000)
SHARED_TOKEN_COUNT = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SHARED_TOKEN_COUNT", 112)
LOCAL_TOKEN_COUNT = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_LOCAL_TOKEN_COUNT", 16)
SHARED_HEADER_BITS = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SHARED_HEADER_BITS", 896)
LOCAL_HEADER_BITS_PER_BLOCK = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_LOCAL_HEADER_BITS_PER_BLOCK", 16)
SELECTOR_BITS_PER_BLOCK = env_int("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SELECTOR_BITS_PER_BLOCK", 16)
MIN_AGGREGATE_PAYLOAD_IMPROVEMENT = float(os.environ.get("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_MIN_AGGREGATE_PAYLOAD_IMPROVEMENT", "0.054"))
PRIOR_CORPUS_SELECTED_PAYLOAD_BITS = float(os.environ.get("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_PRIOR_PAYLOAD_BITS", "812688"))
ZSTD_CHARGED_PUBLIC_BITS = float(os.environ.get("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_ZSTD_CHARGED_PUBLIC_BITS", "982840"))
ZSTD_UNDERCHARGED_PUBLIC_BITS = float(os.environ.get("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_ZSTD_UNDERCHARGED_PUBLIC_BITS", "949992"))
IDENTIFIER_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")

require_positive("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_MAX_BLOCK_BYTES", MAX_BLOCK_BYTES)
require_positive("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SHARED_TOKEN_COUNT", SHARED_TOKEN_COUNT)
require_positive("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_LOCAL_TOKEN_COUNT", LOCAL_TOKEN_COUNT)
require_positive("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SHARED_HEADER_BITS", SHARED_HEADER_BITS)
require_positive("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_LOCAL_HEADER_BITS_PER_BLOCK", LOCAL_HEADER_BITS_PER_BLOCK)
require_positive("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_SELECTOR_BITS_PER_BLOCK", SELECTOR_BITS_PER_BLOCK)

PROFILES = {
    "smoke": {"block_count": 3, "min_improvement": 0.052},
    "hard": {"block_count": 5, "min_improvement": 0.054},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_SUBTOKEN_SHARED_DICTIONARY_CORPUS_CODEC_PROFILE must be smoke or hard")
    return value


def target_blocks(profile: str) -> list[dict[str, Any]]:
    return FROZEN_BLOCKS[: int(PROFILES[profile]["block_count"])]


def read_limited_block(row: dict[str, Any]) -> bytes:
    return bytes(read_block(row)[: int(MAX_BLOCK_BYTES)])


def token_candidates(bodies: list[bytes], token_limit: int, exclude: set[bytes] | None = None) -> list[bytes]:
    excluded = set() if exclude is None else set(exclude)
    counts: Counter[bytes] = Counter()
    for body in bodies:
        counts.update(match.group(0) for match in IDENTIFIER_RE.finditer(body))
    candidates = [token for token, count in counts.items() if token not in excluded and len(token) >= 3 and count >= 2]
    candidates.sort(key=lambda token: (-((len(token) - 1) * counts[token] - len(token)), -counts[token], token))
    return candidates[: int(token_limit)]


def shared_codec(blocks: list[bytes]) -> dict[str, Any]:
    train_block = read_joined(train_paths())
    indent_unit = learn_indent_unit(train_block)
    transformed = [transform_structure(block, indent_unit) for block in blocks]
    shared_tokens = token_candidates([bytes(row["body_stream"]) for row in transformed], int(SHARED_TOKEN_COUNT))
    shared_dictionary = dictionary_stream(shared_tokens)
    shared_dictionary_codec_name, shared_dictionary_payload = best_codec(shared_dictionary)
    rows = []
    for row in transformed:
        local_tokens = token_candidates([bytes(row["body_stream"])], int(LOCAL_TOKEN_COUNT), set(shared_tokens))
        tokens = shared_tokens + local_tokens
        count_delta = delta_encode(bytes(row["count_stream"]))
        substituted_body = substitute_subtokens(bytes(row["body_stream"]), tokens)
        local_dictionary = dictionary_stream(local_tokens)
        count_codec_name, count_payload = best_codec(count_delta)
        body_codec_name, body_payload = best_codec(substituted_body)
        if local_dictionary:
            local_dictionary_codec_name, local_dictionary_payload = best_codec(local_dictionary)
        else:
            local_dictionary_codec_name, local_dictionary_payload = "none0", b""
        rows.append(
            {
                "line_count": int(row["line_count"]),
                "count_codec_name": count_codec_name,
                "body_codec_name": body_codec_name,
                "local_dictionary_codec_name": local_dictionary_codec_name,
                "count_payload": count_payload,
                "body_payload": body_payload,
                "local_dictionary_payload": local_dictionary_payload,
                "local_token_count": int(len(local_tokens)),
                "count_delta_len": len(count_delta),
                "body_stream_len": len(bytes(row["body_stream"])),
                "substituted_body_stream_len": len(substituted_body),
                "local_dictionary_stream_len": len(local_dictionary),
            }
        )
    return {
        "indent_unit": int(indent_unit),
        "shared_token_count": int(len(shared_tokens)),
        "shared_dictionary_codec_name": shared_dictionary_codec_name,
        "shared_dictionary_payload": shared_dictionary_payload,
        "shared_dictionary_stream_len": len(shared_dictionary),
        "rows": rows,
    }


def restore_one(codec: dict[str, Any], row: dict[str, Any]) -> bytes:
    shared_dictionary = decompress_best(str(codec["shared_dictionary_codec_name"]), bytes(codec["shared_dictionary_payload"]))
    shared_tokens = tokens_from_stream(shared_dictionary, int(codec["shared_token_count"]))
    if int(row["local_token_count"]) > 0:
        local_dictionary = decompress_best(str(row["local_dictionary_codec_name"]), bytes(row["local_dictionary_payload"]))
    else:
        local_dictionary = b""
    local_tokens = tokens_from_stream(local_dictionary, int(row["local_token_count"]))
    tokens = shared_tokens + local_tokens
    count_delta = decompress_best(str(row["count_codec_name"]), bytes(row["count_payload"]))
    substituted_body = decompress_best(str(row["body_codec_name"]), bytes(row["body_payload"]))
    count_stream = delta_decode(count_delta)
    body_stream = restore_body(substituted_body, tokens)
    return restore_structure(count_stream, body_stream, int(row["line_count"]), int(codec["indent_unit"]))


def restore_all(codec: dict[str, Any]) -> list[bytes]:
    return [restore_one(codec, row) for row in codec["rows"]]


def codec_payload_bits(codec: dict[str, Any]) -> int:
    total = int(len(bytes(codec["shared_dictionary_payload"])) * 8 + int(SHARED_HEADER_BITS))
    for row in codec["rows"]:
        total += int(len(bytes(row["count_payload"])) * 8)
        total += int(len(bytes(row["body_payload"])) * 8)
        total += int(len(bytes(row["local_dictionary_payload"])) * 8)
        total += int(LOCAL_HEADER_BITS_PER_BLOCK)
        total += int(SELECTOR_BITS_PER_BLOCK)
    return int(total)


def standard_payload_bits(blocks: list[bytes]) -> int:
    return int(sum(len(best_codec(block)[1]) * 8 for block in blocks))


def block_hashes(rows: list[dict[str, Any]], blocks: list[bytes]) -> dict[str, float]:
    values = []
    for row, block in zip(rows, blocks):
        values.append(float(hashlib.sha256(block).hexdigest() == str(row["sha256"])))
    return {"frozen_manifest_hash_success_min": float(min(values))}


def random_blocks(seed: int, blocks: list[bytes]) -> list[bytes]:
    return [random_block(int(seed) + index * 19, len(block)) for index, block in enumerate(blocks)]


def prior_subtoken_payload_bits(profile: str, blocks: list[bytes]) -> int:
    if profile == "hard":
        return int(PRIOR_CORPUS_SELECTED_PAYLOAD_BITS)
    train_block = read_joined(train_paths())
    total = 0
    for block in blocks:
        row = learned_codec(train_block, block)
        total += int((len(bytes(row["count_payload"])) + len(bytes(row["body_payload"])) + len(bytes(row["dictionary_payload"]))) * 8 + 896 + 16)
    return int(total)


def control_success(codec: dict[str, Any], blocks: list[bytes], seed: int) -> dict[str, float]:
    wrong_indent = dict(codec)
    wrong_indent["indent_unit"] = int(wrong_indent["indent_unit"]) + 1
    try:
        wrong_indent_success = float(restore_all(wrong_indent) == blocks)
    except Exception:
        wrong_indent_success = 0.0
    shared_disabled = dict(codec)
    shared_disabled["shared_token_count"] = 0
    shared_disabled["shared_dictionary_payload"] = b""
    shared_disabled["shared_dictionary_codec_name"] = "none0"
    try:
        shared_disabled_success = float(restore_all(shared_disabled) == blocks)
    except Exception:
        shared_disabled_success = 0.0
    shuffled_shared = dict(codec)
    shuffled_shared["shared_dictionary_payload"] = shuffle_bytes(bytes(codec["shared_dictionary_payload"]), int(seed))
    try:
        shuffled_shared_success = float(restore_all(shuffled_shared) == blocks)
    except Exception:
        shuffled_shared_success = 0.0
    shuffled_body = dict(codec)
    shuffled_rows = []
    for index, row in enumerate(codec["rows"]):
        new_row = dict(row)
        new_row["body_payload"] = shuffle_bytes(bytes(row["body_payload"]), int(seed) + index + 31)
        shuffled_rows.append(new_row)
    shuffled_body["rows"] = shuffled_rows
    try:
        shuffled_body_success = float(restore_all(shuffled_body) == blocks)
    except Exception:
        shuffled_body_success = 0.0
    return {
        "wrong_indent_unit_exact_reconstruction_success": wrong_indent_success,
        "shared_dictionary_disabled_exact_reconstruction_success": shared_disabled_success,
        "shuffled_shared_dictionary_exact_reconstruction_success": shuffled_shared_success,
        "shuffled_body_payload_exact_reconstruction_success": shuffled_body_success,
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    rows = target_blocks(profile)
    blocks = [read_limited_block(row) for row in rows]
    codec = shared_codec(blocks)
    restored = restore_all(codec)
    random_payloads = random_blocks(int(seed), blocks)
    random_codec = shared_codec(random_payloads)
    standard_bits = standard_payload_bits(blocks)
    selected_bits = codec_payload_bits(codec)
    prior_bits = float(prior_subtoken_payload_bits(profile, blocks))
    random_standard_bits = standard_payload_bits(random_payloads)
    random_selected_bits = codec_payload_bits(random_codec)
    aggregate_improvement = float(standard_bits - selected_bits) / max(float(standard_bits), 1.0)
    prior_margin = float(prior_bits - selected_bits)
    hash_success = block_hashes(rows, blocks)["frozen_manifest_hash_success_min"]
    controls = control_success(codec, blocks, int(seed))
    controls_collapse = float(int(controls["wrong_indent_unit_exact_reconstruction_success"] == 0.0 and controls["shared_dictionary_disabled_exact_reconstruction_success"] == 0.0 and controls["shuffled_shared_dictionary_exact_reconstruction_success"] == 0.0 and controls["shuffled_body_payload_exact_reconstruction_success"] == 0.0))
    random_improvement = float(random_standard_bits - random_selected_bits) / max(float(random_standard_bits), 1.0)
    engineering_pass = float(int(restored == blocks and hash_success == 1.0 and aggregate_improvement >= float(PROFILES[profile]["min_improvement"]) and prior_margin > 0.0 and random_improvement <= 0.0 and controls_collapse == 1.0))
    return {
        "profile": profile,
        "block_count": float(len(blocks)),
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "source_subtoken_shared_dictionary_corpus_codec_candidate": engineering_pass,
        "source_code_corpus_codec_product_authorized": engineering_pass,
        "source_code_corpus_codec_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "general_unknown_structure_breakthrough_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "exact_reconstruction_success_min": float(int(restored == blocks)),
        "frozen_manifest_hash_success_min": float(hash_success),
        "useful_retrievable_bits": float(sum(len(block) * 8 for block in blocks)),
        "aggregate_standard_payload_bits": float(standard_bits),
        "prior_subtoken_corpus_payload_bits": float(prior_bits),
        "aggregate_selected_payload_bits": float(selected_bits),
        "aggregate_payload_improvement": float(aggregate_improvement),
        "aggregate_payload_margin_over_prior_bits": float(prior_margin),
        "aggregate_payload_improvement_delta_over_prior": float(prior_margin / max(float(prior_bits), 1.0)),
        "shared_token_count": float(codec["shared_token_count"]),
        "local_token_count_per_block": float(LOCAL_TOKEN_COUNT),
        "shared_dictionary_payload_bits": float(len(bytes(codec["shared_dictionary_payload"])) * 8),
        "shared_header_bits": float(SHARED_HEADER_BITS),
        "local_header_bits_per_block": float(LOCAL_HEADER_BITS_PER_BLOCK),
        "selector_bits_per_block": float(SELECTOR_BITS_PER_BLOCK),
        "zstd_charged_public_baseline_bits": float(ZSTD_CHARGED_PUBLIC_BITS),
        "zstd_undercharged_public_baseline_bits": float(ZSTD_UNDERCHARGED_PUBLIC_BITS),
        "margin_over_zstd_charged_public_bits": float(float(ZSTD_CHARGED_PUBLIC_BITS) - selected_bits),
        "margin_over_zstd_undercharged_public_bits": float(float(ZSTD_UNDERCHARGED_PUBLIC_BITS) - selected_bits),
        "random_label_payload_incompressible": float(int(random_selected_bits >= random_standard_bits)),
        "random_label_payload_improvement_over_best_standard": float(random_improvement),
        "random_label_selected_payload_bits": float(random_selected_bits),
        "random_label_best_standard_payload_bits": float(random_standard_bits),
        "controls_collapse": controls_collapse,
        "shared_dictionary_payload_retained": 1.0,
        "per_block_local_dictionary_payload_retained": 1.0,
        "raw_source_block_retained": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "engineering_pass": engineering_pass,
        **controls,
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
    metrics_path = output_dir / "local_100k_source_subtoken_shared_dictionary_corpus_codec_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "shared_token_count": int(SHARED_TOKEN_COUNT), "local_token_count": int(LOCAL_TOKEN_COUNT), "shared_header_bits": int(SHARED_HEADER_BITS), "local_header_bits_per_block": int(LOCAL_HEADER_BITS_PER_BLOCK), "selector_bits_per_block": int(SELECTOR_BITS_PER_BLOCK)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_block_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_subtoken_shared_dictionary_corpus_codec_metrics.json", "path": metrics_path}],
        warnings=["source-code corpus shared-dictionary codec only; no nm, chat, knowledge, paid-compute, or broad breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} aggregate_improvement={summary[f'{SIMULATION_ID}_aggregate_payload_improvement']:.6f} prior_margin_bits={summary[f'{SIMULATION_ID}_aggregate_payload_margin_over_prior_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
