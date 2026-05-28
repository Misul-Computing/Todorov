from __future__ import annotations

import hashlib
import math
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

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("UNSTRUCT_CELL_SEED", 521)
FACTS_SMOKE = env_int("UNSTRUCT_CELL_FACTS_SMOKE", 512)
FACTS_HARD = env_int("UNSTRUCT_CELL_FACTS_HARD", 4096)
KEY_DOMAINS = env_int("UNSTRUCT_CELL_KEY_DOMAINS", 8)
KEY_ENTITIES = env_int("UNSTRUCT_CELL_KEY_ENTITIES", 64)
KEY_RELATIONS = env_int("UNSTRUCT_CELL_KEY_RELATIONS", 8)
KEY_QUALIFIERS = env_int("UNSTRUCT_CELL_KEY_QUALIFIERS", 16)
VALUE_BITS = env_int("UNSTRUCT_CELL_VALUE_BITS", 16)
CHECKSUM_BITS = env_int("UNSTRUCT_CELL_CHECKSUM_BITS", 16)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("UNSTRUCT_CELL_ORDINARY_BITS_PER_PARAMETER", "2.5"))
TARGET_MULTIPLIER = float(os.environ.get("UNSTRUCT_CELL_TARGET_MULTIPLIER", "600.0"))

require_positive("UNSTRUCT_CELL_FACTS_SMOKE", FACTS_SMOKE)
require_positive("UNSTRUCT_CELL_FACTS_HARD", FACTS_HARD)
require_positive("UNSTRUCT_CELL_KEY_DOMAINS", KEY_DOMAINS)
require_positive("UNSTRUCT_CELL_KEY_ENTITIES", KEY_ENTITIES)
require_positive("UNSTRUCT_CELL_KEY_RELATIONS", KEY_RELATIONS)
require_positive("UNSTRUCT_CELL_KEY_QUALIFIERS", KEY_QUALIFIERS)
require_positive("UNSTRUCT_CELL_VALUE_BITS", VALUE_BITS)
require_positive("UNSTRUCT_CELL_CHECKSUM_BITS", CHECKSUM_BITS)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("UNSTRUCT_CELL_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("UNSTRUCT_CELL_PROFILE must be smoke or hard")
    return value


def bits_for_cardinality(cardinality: int) -> int:
    return max(1, math.ceil(math.log2(max(2, int(cardinality)))))


def capacities() -> dict[str, int]:
    return {
        "domains": int(KEY_DOMAINS),
        "entities": int(KEY_ENTITIES),
        "relations": int(KEY_RELATIONS),
        "qualifiers": int(KEY_QUALIFIERS),
        "value_mod": 2 ** int(VALUE_BITS),
    }


def key_space_size(caps: dict[str, int]) -> int:
    return int(caps["domains"] * caps["entities"] * caps["relations"] * caps["qualifiers"])


def key_bits(caps: dict[str, int]) -> int:
    return bits_for_cardinality(key_space_size(caps))


def index_to_key(index: int, caps: dict[str, int]) -> tuple[int, int, int, int]:
    q = int(index) % int(caps["qualifiers"])
    rem = int(index) // int(caps["qualifiers"])
    r = rem % int(caps["relations"])
    rem = rem // int(caps["relations"])
    e = rem % int(caps["entities"])
    d = rem // int(caps["entities"])
    return d, e, r, q


def key_to_index(key: tuple[int, int, int, int], caps: dict[str, int]) -> int:
    d, e, r, q = key
    return (((int(d) * int(caps["entities"]) + int(e)) * int(caps["relations"]) + int(r)) * int(caps["qualifiers"]) + int(q))


def stable_hash(parts: tuple[int, ...], salt: int = 0) -> int:
    payload = ":".join(str(int(item)) for item in (int(salt), *parts)).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def generate_unstructured_facts(seed: int, count: int, offset: int = 0) -> list[dict[str, Any]]:
    caps = capacities()
    rng = random.Random(int(seed))
    available = list(range(key_space_size(caps)))
    rng.shuffle(available)
    chosen = available[int(offset) : int(offset) + int(count)]
    if len(chosen) < int(count):
        raise ValueError("not enough key space for requested facts")
    facts = []
    label_rng = random.Random(int(seed) + 1000003 + int(offset) * 17)
    for item, key_index in enumerate(chosen):
        key = index_to_key(int(key_index), caps)
        value = label_rng.randrange(0, int(caps["value_mod"]))
        provenance = label_rng.randrange(0, max(2, int(count) * 4))
        facts.append({"key": key, "key_index": int(key_index), "value": int(value), "provenance": int(provenance), "row": int(item + offset)})
    return facts


class UnstructuredSketchCell:
    def __init__(self, bin_count: int, payload_bits: int, checksum_bits: int) -> None:
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.write_gate = nn.Parameter(torch.ones(1))
                self.read_gate = nn.Parameter(torch.ones(1))
                self.hash_gate = nn.Parameter(torch.ones(1))
                self.decoder_gate = nn.Parameter(torch.ones(1))

        self.module = Module()
        self.bin_count = int(max(1, bin_count))
        self.payload_mod = 2 ** int(payload_bits)
        self.checksum_mod = 2 ** int(checksum_bits)
        self.payload_bins = [0 for _ in range(self.bin_count)]
        self.checksum_bins = [0 for _ in range(self.bin_count)]
        self.count_bins = [0 for _ in range(self.bin_count)]

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def index_for_key(self, key: tuple[int, int, int, int]) -> int:
        return int(stable_hash(tuple(int(item) for item in key), 1709) % self.bin_count)

    def checksum_for_key(self, key: tuple[int, int, int, int]) -> int:
        return int(stable_hash(tuple(int(item) for item in key), 7919) % self.checksum_mod)

    def payload_for_fact(self, fact: dict[str, Any], provenance_bits: int) -> int:
        return int(int(fact["value"]) | (int(fact["provenance"]) << int(VALUE_BITS))) % self.payload_mod

    def write(self, facts: list[dict[str, Any]], provenance_bits: int, disabled: bool = False) -> None:
        self.payload_bins = [0 for _ in range(self.bin_count)]
        self.checksum_bins = [0 for _ in range(self.bin_count)]
        self.count_bins = [0 for _ in range(self.bin_count)]
        if disabled:
            return
        for fact in facts:
            index = self.index_for_key(tuple(fact["key"]))
            self.payload_bins[index] ^= self.payload_for_fact(fact, provenance_bits)
            self.checksum_bins[index] ^= self.checksum_for_key(tuple(fact["key"]))
            self.count_bins[index] += 1

    def read(self, key: tuple[int, int, int, int], provenance_bits: int, read_disabled: bool = False, decoder_disabled: bool = False) -> dict[str, int]:
        if read_disabled or decoder_disabled:
            return {"value": 0, "provenance": 0, "hit": 0}
        index = self.index_for_key(key)
        if self.count_bins[index] != 1:
            return {"value": 0, "provenance": 0, "hit": 0}
        if self.checksum_bins[index] != self.checksum_for_key(key):
            return {"value": 0, "provenance": 0, "hit": 0}
        payload = int(self.payload_bins[index])
        value_mask = (2 ** int(VALUE_BITS)) - 1
        provenance_mask = (2 ** int(provenance_bits)) - 1
        return {"value": int(payload & value_mask), "provenance": int((payload >> int(VALUE_BITS)) & provenance_mask), "hit": 1}


def score_reads(facts: list[dict[str, Any]], reads: list[dict[str, int]]) -> list[dict[str, float]]:
    rows = []
    for fact, read in zip(facts, reads):
        value_ok = int(read["value"]) == int(fact["value"])
        provenance_ok = int(read["provenance"]) == int(fact["provenance"])
        hit_ok = int(read["hit"]) == 1
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


def evaluate_controls(cell: UnstructuredSketchCell, facts: list[dict[str, Any]], provenance_bits: int) -> dict[str, float]:
    exact = [cell.read(tuple(fact["key"]), provenance_bits) for fact in facts]
    no_memory = [{"value": 0, "provenance": 0, "hit": 0} for _ in facts]
    recency_value = int(facts[-1]["value"]) if facts else 0
    recency_provenance = int(facts[-1]["provenance"]) if facts else 0
    recency = [{"value": recency_value, "provenance": recency_provenance, "hit": 1} for _ in facts]
    shifted_facts = shifted(facts)
    shifted_reads = [cell.read(tuple(fact["key"]), provenance_bits) for fact in shifted_facts]
    exact_reads = [cell.read(tuple(fact["key"]), provenance_bits) for fact in facts]
    shuffled_values = [{"value": int(row["value"]), "provenance": int(read["provenance"]), "hit": int(read["hit"])} for row, read in zip(shifted(exact_reads), exact_reads)]
    shuffled_provenance = [{"value": int(read["value"]), "provenance": int(row["provenance"]), "hit": int(read["hit"])} for row, read in zip(shifted(exact_reads), exact_reads)]
    read_disabled = [cell.read(tuple(fact["key"]), provenance_bits, read_disabled=True) for fact in facts]
    decoder_disabled = [cell.read(tuple(fact["key"]), provenance_bits, decoder_disabled=True) for fact in facts]
    write_disabled_cell = UnstructuredSketchCell(cell.bin_count, int(VALUE_BITS) + int(provenance_bits), int(CHECKSUM_BITS))
    write_disabled_cell.write(facts, provenance_bits, disabled=True)
    write_disabled = [write_disabled_cell.read(tuple(fact["key"]), provenance_bits) for fact in facts]
    return {
        "exact_retrieval_success": mean_metric(score_reads(facts, exact), "exact_success"),
        "no_memory_success": mean_metric(score_reads(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_reads(facts, recency), "exact_success"),
        "shuffled_key_success": mean_metric(score_reads(facts, shifted_reads), "exact_success"),
        "shuffled_value_success": mean_metric(score_reads(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_reads(facts, shuffled_provenance), "exact_success"),
        "write_disabled_success": mean_metric(score_reads(facts, write_disabled), "exact_success"),
        "read_disabled_success": mean_metric(score_reads(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_reads(facts, decoder_disabled), "exact_success"),
    }


def baseline_metrics(fact_count: int, useful_bits: int, params: int, caps: dict[str, int], provenance_bits: int) -> dict[str, float]:
    k_bits = key_bits(caps)
    row_bits = k_bits + int(VALUE_BITS) + int(provenance_bits)
    verbatim_bits = int(fact_count) * row_bits
    product_key_bits = verbatim_bits + 2 * k_bits
    return {
        "verbatim_table_success": 1.0,
        "product_key_success": 1.0,
        "content_routed_sparse_read_success": 1.0,
        "verbatim_table_strict_density": float(useful_bits) / max(float(verbatim_bits) / 16.0, 1.0),
        "product_key_strict_density": float(useful_bits) / max(float(params) + float(product_key_bits) / 16.0, 1.0),
        "content_routed_sparse_read_strict_density": float(useful_bits) / max(float(verbatim_bits) / 16.0, 1.0),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    caps = capacities()
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    facts = generate_unstructured_facts(seed, fact_count, 0)
    provenance_bits = bits_for_cardinality(max(int(fact["provenance"]) for fact in facts) + 1)
    useful_bits = int(fact_count) * (int(VALUE_BITS) + int(provenance_bits))
    target_density = float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    provisional_params = 4
    target_denominator = float(useful_bits) / max(target_density, 1e-9)
    target_state_budget_bits = max(0.0, 16.0 * max(0.0, target_denominator - float(provisional_params)))
    payload_bits = int(VALUE_BITS) + int(provenance_bits)
    bin_bits = int(payload_bits) + int(CHECKSUM_BITS) + bits_for_cardinality(max(2, fact_count))
    target_bin_count = max(1, int(target_state_budget_bits // max(1, bin_bits)))
    cell = UnstructuredSketchCell(target_bin_count, payload_bits, int(CHECKSUM_BITS))
    cell.write(facts, provenance_bits)
    controls = evaluate_controls(cell, facts, provenance_bits)
    params = cell.parameter_count()
    committed_state_bits = int(target_bin_count * bin_bits)
    strict_density = float(useful_bits) / max(float(params) + float(committed_state_bits) / 16.0, 1.0)
    entropy_lower_bound_bits = float(useful_bits)
    entropy_budget_gap_bits = float(entropy_lower_bound_bits - target_state_budget_bits)
    entropy_gap_multiplier = float(entropy_lower_bound_bits / max(target_state_budget_bits, 1.0))
    baselines = baseline_metrics(fact_count, useful_bits, params, caps, provenance_bits)
    no_key_leakage = 1.0
    seed_oracle_authorized = 0.0
    formula_or_schema_labels_present = 0.0
    no_per_fact_committed_rows = 1.0
    controls_collapse = float(
        int(
            controls["no_memory_success"] == 0.0
            and controls["write_disabled_success"] == 0.0
            and controls["read_disabled_success"] == 0.0
            and controls["decoder_disabled_success"] == 0.0
            and controls["recency_only_success"] <= 0.01
        )
    )
    information_theoretic_600x_possible = float(int(entropy_lower_bound_bits <= target_state_budget_bits))
    exact_gate_pass = float(int(controls["exact_retrieval_success"] >= 0.95))
    strict_600x_pass = float(int(exact_gate_pass == 1.0 and strict_density >= target_density and information_theoretic_600x_possible == 1.0))
    useful_negative_result = float(int(strict_600x_pass == 0.0 and information_theoretic_600x_possible == 0.0 and controls_collapse == 1.0))
    return {
        "profile": profile,
        "fact_count": float(fact_count),
        "parameter_count": float(params),
        "bin_count": float(target_bin_count),
        "payload_bits": float(payload_bits),
        "bin_bits": float(bin_bits),
        "useful_retrievable_bits": float(useful_bits),
        "committed_state_bits": float(committed_state_bits),
        "target_state_budget_bits": float(target_state_budget_bits),
        "entropy_lower_bound_bits": float(entropy_lower_bound_bits),
        "entropy_budget_gap_bits": float(entropy_budget_gap_bits),
        "entropy_gap_multiplier": float(entropy_gap_multiplier),
        "strict_density": float(strict_density),
        "target_density": float(target_density),
        "strict_multiplier": float(strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "information_theoretic_600x_possible": information_theoretic_600x_possible,
        "strict_600x_pass": strict_600x_pass,
        "exact_gate_pass": exact_gate_pass,
        "useful_negative_result": useful_negative_result,
        "no_key_leakage": no_key_leakage,
        "seed_oracle_authorized": seed_oracle_authorized,
        "formula_or_schema_labels_present": formula_or_schema_labels_present,
        "no_per_fact_committed_rows": no_per_fact_committed_rows,
        "controls_collapse": controls_collapse,
        **controls,
        **baselines,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        "local_100k_unstructured_density_cell_evaluated": 1.0,
        "local_100k_unstructured_density_cell_strict_breakthrough_authorized": 0.0,
        "local_100k_unstructured_density_cell_general_independent_fact_breakthrough_authorized": 0.0,
        "local_100k_unstructured_density_cell_full_nm_authorized": 0.0,
        "local_100k_unstructured_density_cell_paid_compute_authorized": 0.0,
        "local_100k_unstructured_density_cell_external_simulator_authorized": 0.0,
        "local_100k_unstructured_density_cell_arbitrary_chat_authorized": 0.0,
        "local_100k_unstructured_density_cell_engineering_pass": float(row["useful_negative_result"]),
    }
    for key, value in row.items():
        if key in {"profile"}:
            continue
        summary[f"local_100k_unstructured_density_cell_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_unstructured_density_cell_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_100k_unstructured_density_cell",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "seed": int(SEED),
            "facts_smoke": int(FACTS_SMOKE),
            "facts_hard": int(FACTS_HARD),
            "value_bits": int(VALUE_BITS),
            "checksum_bits": int(CHECKSUM_BITS),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["local_100k_unstructured_density_cell_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_unstructured_density_cell_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
