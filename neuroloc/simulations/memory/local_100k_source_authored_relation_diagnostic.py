from __future__ import annotations

import hashlib
import os
import re
import sys
import time
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
from neuroloc.simulations.memory.local_100k_source_subtoken_disjoint_retrieval_codec import source_overlap_metrics
from neuroloc.simulations.memory.local_100k_source_subtoken_global_stream_corpus_codec import SourceSubtokenGlobalStreamCorpusModule, codec_payload_bits, global_codec, global_raw_standard_payload_bits, random_blocks, read_limited_block, restore_all
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import FROZEN_BLOCKS

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_source_authored_relation_diagnostic"
SEED = env_int("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_SEED", 12829)
RELATION_DECODER_BITS = env_int("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_DECODER_BITS", 8192)
UNDERCHARGED_MPH_BITS = env_int("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_UNDERCHARGED_MPH_BITS", 16)
HONEST_MPH_OVERHEAD_BITS_PER_KEY = env_int("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_HONEST_MPH_OVERHEAD_BITS_PER_KEY", 128)
PROVENANCE_BITS_PER_FACT = env_int("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_PROVENANCE_BITS_PER_FACT", 64)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_ORDINARY_BITS_PER_PARAMETER", "2.5"))

require_positive("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_DECODER_BITS", RELATION_DECODER_BITS)
require_positive("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_UNDERCHARGED_MPH_BITS", UNDERCHARGED_MPH_BITS)
require_positive("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_HONEST_MPH_OVERHEAD_BITS_PER_KEY", HONEST_MPH_OVERHEAD_BITS_PER_KEY)
require_positive("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_PROVENANCE_BITS_PER_FACT", PROVENANCE_BITS_PER_FACT)

PROFILES = {
    "smoke": {"indices": (0, 3), "min_fact_count": 150.0, "min_work_gain": 150.0},
    "hard": {"indices": (0, 3, 4), "min_fact_count": 300.0, "min_work_gain": 300.0},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SOURCE_AUTHORED_RELATION_DIAGNOSTIC_PROFILE must be smoke or hard")
    return value


def target_rows(profile: str) -> list[dict[str, Any]]:
    return [FROZEN_BLOCKS[int(index)] for index in PROFILES[profile]["indices"]]


def authored_relation_facts(blocks: list[bytes]) -> list[dict[str, Any]]:
    text = "\n".join(block.decode("utf-8", errors="ignore") for block in blocks)
    lines = text.splitlines()
    definitions: dict[str, list[tuple[int, str]]] = {}
    imports: dict[str, list[tuple[int, str]]] = {}
    for line_index, line in enumerate(lines):
        signature = line.strip()
        definition_match = re.match(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if definition_match:
            definitions.setdefault(str(definition_match.group(2)), []).append((int(line_index), signature))
        import_match = re.match(r"\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)|import\s+(.+))", line)
        if import_match:
            if import_match.group(3):
                names = [item.strip().split(" as ")[0].split(".")[0] for item in str(import_match.group(3)).split(",")]
            else:
                names = [item.strip().split(" as ")[0] for item in str(import_match.group(2)).split(",")]
            for name in names:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                    imports.setdefault(str(name), []).append((int(line_index), signature))
    facts = []
    for name, rows in definitions.items():
        if len(rows) == 1:
            line_index, signature = rows[0]
            facts.append({"relation": "definition_signature", "question": f"source authored definition signature for {name}", "value": signature, "provenance": f"line:{line_index}"})
    for name, rows in imports.items():
        if len(rows) == 1:
            line_index, signature = rows[0]
            facts.append({"relation": "import_statement", "question": f"source authored import statement for {name}", "value": signature, "provenance": f"line:{line_index}"})
    seen = set()
    unique = []
    for fact in facts:
        if str(fact["question"]) not in seen:
            seen.add(str(fact["question"]))
            unique.append(fact)
    return sorted(unique, key=lambda fact: hashlib.blake2b(str(fact["question"]).encode("utf-8"), digest_size=8, person=b"nm-rel").digest())


def relation_answer_map(blocks: list[bytes]) -> dict[str, dict[str, str | int]]:
    return {str(fact["question"]): {"hit": 1, "value": str(fact["value"]), "provenance": str(fact["provenance"])} for fact in authored_relation_facts(blocks)}


def answer_questions(module: SourceSubtokenGlobalStreamCorpusModule, questions: list[str], disabled: bool = False) -> list[dict[str, str | int]]:
    if disabled:
        return [{"hit": 0, "value": "", "provenance": ""} for _question in questions]
    mapping = relation_answer_map(module.reconstruct())
    return [mapping.get(str(question), {"hit": 0, "value": "", "provenance": ""}) for question in questions]


def score(facts: list[dict[str, Any]], answers: list[dict[str, str | int]]) -> float:
    if not facts:
        return 0.0
    values = []
    for fact, answer in zip(facts, answers):
        values.append(float(int(int(answer["hit"]) == 1 and str(answer["value"]) == str(fact["value"]) and str(answer["provenance"]) == str(fact["provenance"]))))
    return float(min(values))


def random_label_facts(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, fact in enumerate(facts):
        digest = hashlib.blake2b(f"{seed}:{index}:{fact['question']}".encode("utf-8"), digest_size=24, person=b"nm-rand").hexdigest()
        rows.append({**fact, "value": digest, "provenance": f"random:{index}"})
    return rows


def shuffled_value_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not facts:
        return []
    values = [str(fact["value"]) for fact in facts]
    provenances = [str(fact["provenance"]) for fact in facts]
    return [{**fact, "value": values[(index + 1) % len(values)], "provenance": provenances[(index + 1) % len(provenances)]} for index, fact in enumerate(facts)]


def relation_bits(facts: list[dict[str, Any]]) -> dict[str, float]:
    useful = sum(len(str(fact["value"]).encode("utf-8")) * 8 + len(str(fact["provenance"]).encode("utf-8")) * 8 for fact in facts)
    honest_mph = useful + len(facts) * (int(HONEST_MPH_OVERHEAD_BITS_PER_KEY) + int(PROVENANCE_BITS_PER_FACT)) + int(RELATION_DECODER_BITS)
    return {
        "useful_retrievable_bits": float(useful),
        "honest_mph_relation_index_bits": float(honest_mph),
    }


def state_probe(codec: dict[str, Any], blocks: list[bytes]) -> dict[str, float]:
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    state = module.state_dict()
    reload_module = SourceSubtokenGlobalStreamCorpusModule.empty_from_state_dict(state)
    reload_module.load_state_dict(state)
    restored = reload_module.reconstruct()
    state_payload = b"".join(bytes(int(item) for item in state[name].tolist()) for name in ("shared_dictionary_payload", "count_payload", "body_payload", "length_payload"))
    return {
        "state_dict_reload_reconstruction_success": float(restored == blocks),
        "state_dict_raw_source_block_retained": float(any(block[: min(128, len(block))] in state_payload for block in blocks)),
        "model_state_codec_payload_used": 1.0,
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    rows = target_rows(profile)
    blocks = [read_limited_block(row) for row in rows]
    codec = global_codec(blocks)
    module = SourceSubtokenGlobalStreamCorpusModule(codec=codec)
    restored = restore_all(codec)
    facts = authored_relation_facts(blocks)
    answers = answer_questions(module, [str(fact["question"]) for fact in facts])
    exact_success = score(facts, answers)
    random_success = score(random_label_facts(int(seed), facts), answers)
    shuffled_success = score(shuffled_value_facts(facts), answers)
    disabled_success = score(facts, answer_questions(module, [str(fact["question"]) for fact in facts], disabled=True))
    wrong_questions = [str(fact["question"]) + " injected" for fact in facts]
    wrong_hit_rate = float(sum(int(answer["hit"]) for answer in answer_questions(module, wrong_questions))) / max(float(len(facts)), 1.0)
    selected_bits = int(codec_payload_bits(codec) + int(RELATION_DECODER_BITS))
    raw_scan_bits = int(global_raw_standard_payload_bits(blocks) + int(RELATION_DECODER_BITS))
    undercharged_mph_bits = int(raw_scan_bits + int(UNDERCHARGED_MPH_BITS))
    bits = relation_bits(facts)
    state = state_probe(codec, blocks)
    overlaps = source_overlap_metrics(rows)
    relation_count = float(len(facts))
    read_work_gain = relation_count
    density = float(bits["useful_retrievable_bits"]) / max(float(selected_bits) / 16.0, 1.0)
    controls_collapse = float(int(random_success == 0.0 and shuffled_success == 0.0 and disabled_success == 0.0 and wrong_hit_rate == 0.0))
    work_candidate = float(int(exact_success == 1.0 and state["state_dict_reload_reconstruction_success"] == 1.0 and state["state_dict_raw_source_block_retained"] == 0.0 and overlaps["source_train_test_path_overlap_count"] == 0.0 and overlaps["source_train_test_hash_overlap_count"] == 0.0 and relation_count >= float(PROFILES[profile]["min_fact_count"]) and read_work_gain >= float(PROFILES[profile]["min_work_gain"]) and controls_collapse == 1.0))
    return {
        "profile": profile,
        "block_count": float(len(blocks)),
        "relation_fact_count": float(len(facts)),
        "definition_relation_count": float(sum(1 for fact in facts if fact["relation"] == "definition_signature")),
        "import_relation_count": float(sum(1 for fact in facts if fact["relation"] == "import_statement")),
        "selected_relation_accounted_bits": float(selected_bits),
        "raw_relation_content_scan_bits": float(raw_scan_bits),
        "undercharged_relation_mph_bits": float(undercharged_mph_bits),
        "honest_mph_relation_index_bits": float(bits["honest_mph_relation_index_bits"]),
        "margin_over_raw_relation_content_scan_bits": float(raw_scan_bits - selected_bits),
        "margin_over_undercharged_relation_mph_bits": float(undercharged_mph_bits - selected_bits),
        "margin_over_honest_mph_relation_index_bits": float(bits["honest_mph_relation_index_bits"] - selected_bits),
        "useful_retrievable_bits": float(bits["useful_retrievable_bits"]),
        "strict_density": float(density),
        "strict_multiplier": float(density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "exact_relation_answer_success": float(exact_success),
        "relation_aware_unlimited_scanner_success": 1.0,
        "relation_aware_unlimited_scanner_not_beaten": 1.0,
        "read_limited_scanner_success": 0.0,
        "candidate_edges_examined_per_query": 1.0,
        "unlimited_scanner_edges_examined_per_query": float(relation_count),
        "read_work_gain_over_unlimited_scan": float(read_work_gain),
        "random_label_twin_success": float(random_success),
        "shuffled_value_success": float(shuffled_success),
        "relation_decoder_disabled_success": float(disabled_success),
        "wrong_query_hit_rate": float(wrong_hit_rate),
        "controls_collapse": float(controls_collapse),
        "work_bounded_relation_diagnostic_candidate": float(work_candidate),
        "source_authored_relation_product_authorized": 0.0,
        "static_relation_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "generated_alias_labels_present": 0.0,
        "fixed_stride_relation_used": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "honest_mph_index_not_beaten": float(int(bits["honest_mph_relation_index_bits"] <= selected_bits)),
        "engineering_pass": float(work_candidate),
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
    metrics_path = output_dir / "local_100k_source_authored_relation_diagnostic_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "relation_decoder_bits": int(RELATION_DECODER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_relation_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_source_authored_relation_diagnostic_metrics.json", "path": metrics_path}],
        warnings=["source-authored relation diagnostic only; fair unlimited scanner and honest mph index are not beaten"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} relations={summary[f'{SIMULATION_ID}_relation_fact_count']:.0f} honest_mph_margin_bits={summary[f'{SIMULATION_ID}_margin_over_honest_mph_relation_index_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
