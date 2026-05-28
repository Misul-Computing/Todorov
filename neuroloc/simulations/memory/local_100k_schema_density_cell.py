from __future__ import annotations

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
SEED = env_int("SCHEMA_CELL_SEED", 419)
FACTS_SMOKE = env_int("SCHEMA_CELL_FACTS_SMOKE", 8192)
FACTS_HARD = env_int("SCHEMA_CELL_FACTS_HARD", 32768)
TRAIN_FACTS = env_int("SCHEMA_CELL_TRAIN_FACTS", 24)
KEY_DOMAINS = env_int("SCHEMA_CELL_KEY_DOMAINS", 8)
KEY_ENTITIES = env_int("SCHEMA_CELL_KEY_ENTITIES", 64)
KEY_RELATIONS = env_int("SCHEMA_CELL_KEY_RELATIONS", 8)
KEY_QUALIFIERS = env_int("SCHEMA_CELL_KEY_QUALIFIERS", 16)
VALUE_MOD = env_int("SCHEMA_CELL_VALUE_MOD", 65521)
PROVENANCE_MOD = env_int("SCHEMA_CELL_PROVENANCE_MOD", 16381)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("SCHEMA_CELL_ORDINARY_BITS_PER_PARAMETER", "2.5"))
TARGET_MULTIPLIER = float(os.environ.get("SCHEMA_CELL_TARGET_MULTIPLIER", "600.0"))

require_positive("SCHEMA_CELL_FACTS_SMOKE", FACTS_SMOKE)
require_positive("SCHEMA_CELL_FACTS_HARD", FACTS_HARD)
require_positive("SCHEMA_CELL_TRAIN_FACTS", TRAIN_FACTS)
require_positive("SCHEMA_CELL_KEY_DOMAINS", KEY_DOMAINS)
require_positive("SCHEMA_CELL_KEY_ENTITIES", KEY_ENTITIES)
require_positive("SCHEMA_CELL_KEY_RELATIONS", KEY_RELATIONS)
require_positive("SCHEMA_CELL_KEY_QUALIFIERS", KEY_QUALIFIERS)
require_positive("SCHEMA_CELL_VALUE_MOD", VALUE_MOD)
require_positive("SCHEMA_CELL_PROVENANCE_MOD", PROVENANCE_MOD)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("SCHEMA_CELL_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("SCHEMA_CELL_PROFILE must be smoke or hard")
    return value


def bits_for_cardinality(cardinality: int) -> int:
    return max(1, math.ceil(math.log2(max(2, int(cardinality)))))


def capacities() -> dict[str, int]:
    return {
        "domains": int(KEY_DOMAINS),
        "entities": int(KEY_ENTITIES),
        "relations": int(KEY_RELATIONS),
        "qualifiers": int(KEY_QUALIFIERS),
        "value_mod": int(VALUE_MOD),
        "provenance_mod": int(PROVENANCE_MOD),
    }


def key_space_size(caps: dict[str, int]) -> int:
    return int(caps["domains"] * caps["entities"] * caps["relations"] * caps["qualifiers"])


def key_bits(caps: dict[str, int]) -> int:
    return bits_for_cardinality(key_space_size(caps))


def key_to_index(key: tuple[int, int, int, int], caps: dict[str, int]) -> int:
    d, e, r, q = key
    return (((int(d) * int(caps["entities"]) + int(e)) * int(caps["relations"]) + int(r)) * int(caps["qualifiers"]) + int(q))


def index_to_key(index: int, caps: dict[str, int]) -> tuple[int, int, int, int]:
    q = int(index) % int(caps["qualifiers"])
    rem = int(index) // int(caps["qualifiers"])
    r = rem % int(caps["relations"])
    rem = rem // int(caps["relations"])
    e = rem % int(caps["entities"])
    d = rem // int(caps["entities"])
    return d, e, r, q


def feature_vector(key: tuple[int, int, int, int], caps: dict[str, int], mod: int) -> list[int]:
    d, e, r, q = [int(item) for item in key]
    idx = key_to_index(key, caps)
    return [
        1 % mod,
        d % mod,
        e % mod,
        r % mod,
        q % mod,
        (d * e) % mod,
        (d * r) % mod,
        (d * q) % mod,
        (e * r) % mod,
        (e * q) % mod,
        (r * q) % mod,
        (idx * idx + 3 * idx + 7) % mod,
    ]


def schema_coefficients(seed: int, width: int, mod: int) -> list[int]:
    rng = random.Random(int(seed) + int(mod) + int(width) * 17)
    return [rng.randrange(1, int(mod)) for _ in range(int(width))]


def dot_mod(values: list[int], coefficients: list[int], mod: int) -> int:
    return int(sum((int(a) * int(b)) % int(mod) for a, b in zip(values, coefficients)) % int(mod))


def selected_key_indices(seed: int, count: int, offset: int, caps: dict[str, int]) -> list[int]:
    rng = random.Random(int(seed))
    available = list(range(key_space_size(caps)))
    rng.shuffle(available)
    chosen = available[int(offset) : int(offset) + int(count)]
    if len(chosen) < int(count):
        raise ValueError("not enough key space for requested facts")
    return [int(item) for item in chosen]


def generate_schema_facts(seed: int, count: int, offset: int = 0) -> list[dict[str, Any]]:
    caps = capacities()
    width = len(feature_vector((0, 0, 0, 0), caps, int(caps["value_mod"])))
    value_coeffs = schema_coefficients(seed + 101, width, int(caps["value_mod"]))
    provenance_coeffs = schema_coefficients(seed + 211, width, int(caps["provenance_mod"]))
    facts = []
    for key_index in selected_key_indices(seed, count, offset, caps):
        key = index_to_key(key_index, caps)
        value = dot_mod(feature_vector(key, caps, int(caps["value_mod"])), value_coeffs, int(caps["value_mod"]))
        provenance = dot_mod(feature_vector(key, caps, int(caps["provenance_mod"])), provenance_coeffs, int(caps["provenance_mod"]))
        facts.append({"key": key, "key_index": int(key_index), "value": int(value), "provenance": int(provenance)})
    return facts


def generate_random_facts(seed: int, count: int, offset: int = 0) -> list[dict[str, Any]]:
    caps = capacities()
    rng = random.Random(int(seed) + 9301)
    facts = []
    for key_index in selected_key_indices(seed, count, offset, caps):
        key = index_to_key(key_index, caps)
        facts.append(
            {
                "key": key,
                "key_index": int(key_index),
                "value": int(rng.randrange(0, int(caps["value_mod"]))),
                "provenance": int(rng.randrange(0, int(caps["provenance_mod"]))),
            }
        )
    return facts


def split_schema_sets(seed: int, train_count: int, test_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return generate_schema_facts(seed, train_count, 0), generate_schema_facts(seed, test_count, train_count)


def solve_modular_linear(rows: list[list[int]], targets: list[int], width: int, mod: int) -> list[int]:
    matrix = [[int(value) % int(mod) for value in row[:width]] + [int(target) % int(mod)] for row, target in zip(rows, targets)]
    rank = 0
    pivots: list[int] = []
    for col in range(int(width)):
        pivot = None
        for row in range(rank, len(matrix)):
            if matrix[row][col] % int(mod) != 0:
                pivot = row
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inv = pow(matrix[rank][col] % int(mod), -1, int(mod))
        matrix[rank] = [(item * inv) % int(mod) for item in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank:
                continue
            factor = matrix[row][col] % int(mod)
            if factor:
                matrix[row] = [(a - factor * b) % int(mod) for a, b in zip(matrix[row], matrix[rank])]
        pivots.append(col)
        rank += 1
        if rank == int(width):
            break
    if rank < int(width):
        raise ValueError("schema rows are rank deficient")
    for row in matrix:
        if all((row[col] % int(mod)) == 0 for col in range(int(width))) and row[-1] % int(mod) != 0:
            raise ValueError("schema rows are inconsistent")
    solution = [0 for _ in range(int(width))]
    for row, col in enumerate(pivots):
        solution[col] = int(matrix[row][-1] % int(mod))
    return solution


class SchemaDensityNeuronCell:
    def __init__(self, caps: dict[str, int]) -> None:
        import torch
        import torch.nn as nn

        width = len(feature_vector((0, 0, 0, 0), caps, int(caps["value_mod"])))

        class Module(nn.Module):
            def __init__(self, size: int) -> None:
                super().__init__()
                self.value_coefficients = nn.Parameter(torch.zeros(size))
                self.provenance_coefficients = nn.Parameter(torch.zeros(size))

        self.module = Module(width)
        self.caps = dict(caps)
        self.width = int(width)
        self.fitted = False

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def fit(self, facts: list[dict[str, Any]], disabled: bool = False) -> None:
        import torch

        if disabled:
            self.fitted = False
            return
        value_rows = [feature_vector(tuple(fact["key"]), self.caps, int(self.caps["value_mod"])) for fact in facts]
        value_targets = [int(fact["value"]) for fact in facts]
        provenance_rows = [feature_vector(tuple(fact["key"]), self.caps, int(self.caps["provenance_mod"])) for fact in facts]
        provenance_targets = [int(fact["provenance"]) for fact in facts]
        value_solution = solve_modular_linear(value_rows, value_targets, self.width, int(self.caps["value_mod"]))
        provenance_solution = solve_modular_linear(provenance_rows, provenance_targets, self.width, int(self.caps["provenance_mod"]))
        with torch.no_grad():
            self.module.value_coefficients.copy_(torch.tensor(value_solution, dtype=self.module.value_coefficients.dtype))
            self.module.provenance_coefficients.copy_(torch.tensor(provenance_solution, dtype=self.module.provenance_coefficients.dtype))
        self.fitted = True

    def coefficients(self, name: str) -> list[int]:
        tensor = getattr(self.module, name).detach().cpu().numpy()
        return [int(round(float(item))) for item in tensor.tolist()]

    def read(
        self,
        key: tuple[int, int, int, int],
        schema_disabled: bool = False,
        decoder_disabled: bool = False,
        shuffled_schema: bool = False,
    ) -> dict[str, int]:
        if decoder_disabled or schema_disabled or not self.fitted:
            return {"value": 0, "provenance": 0, "hit": 0}
        value_coeffs = self.coefficients("value_coefficients")
        provenance_coeffs = self.coefficients("provenance_coefficients")
        if shuffled_schema:
            value_coeffs = value_coeffs[-1:] + value_coeffs[:-1]
            provenance_coeffs = provenance_coeffs[-1:] + provenance_coeffs[:-1]
        value = dot_mod(feature_vector(key, self.caps, int(self.caps["value_mod"])), value_coeffs, int(self.caps["value_mod"]))
        provenance = dot_mod(feature_vector(key, self.caps, int(self.caps["provenance_mod"])), provenance_coeffs, int(self.caps["provenance_mod"]))
        return {"value": int(value), "provenance": int(provenance), "hit": 1}


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


def evaluate_cell(cell: SchemaDensityNeuronCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    exact = [cell.read(tuple(fact["key"])) for fact in facts]
    no_memory = [{"value": 0, "provenance": 0, "hit": 0} for _ in facts]
    recency_value = int(facts[-1]["value"]) if facts else 0
    recency_provenance = int(facts[-1]["provenance"]) if facts else 0
    recency = [{"value": recency_value, "provenance": recency_provenance, "hit": 1} for _ in facts]
    shuffled_keys = [cell.read(tuple(fact["key"])) for fact in shifted(facts)]
    exact_reads = [cell.read(tuple(fact["key"])) for fact in facts]
    shuffled_values = [{"value": int(row["value"]), "provenance": int(read["provenance"]), "hit": int(read["hit"])} for row, read in zip(shifted(exact_reads), exact_reads)]
    shuffled_provenance = [{"value": int(read["value"]), "provenance": int(row["provenance"]), "hit": int(read["hit"])} for row, read in zip(shifted(exact_reads), exact_reads)]
    schema_disabled = [cell.read(tuple(fact["key"]), schema_disabled=True) for fact in facts]
    shuffled_schema = [cell.read(tuple(fact["key"]), shuffled_schema=True) for fact in facts]
    decoder_disabled = [cell.read(tuple(fact["key"]), decoder_disabled=True) for fact in facts]
    read_disabled = [{"value": 0, "provenance": 0, "hit": 0} for _ in facts]
    return {
        "exact_retrieval_success": mean_metric(score_reads(facts, exact), "exact_success"),
        "no_memory_success": mean_metric(score_reads(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_reads(facts, recency), "exact_success"),
        "shuffled_key_success": mean_metric(score_reads(facts, shuffled_keys), "exact_success"),
        "shuffled_value_success": mean_metric(score_reads(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_reads(facts, shuffled_provenance), "exact_success"),
        "schema_disabled_success": mean_metric(score_reads(facts, schema_disabled), "exact_success"),
        "shuffled_schema_success": mean_metric(score_reads(facts, shuffled_schema), "exact_success"),
        "decoder_disabled_success": mean_metric(score_reads(facts, decoder_disabled), "exact_success"),
        "read_disabled_success": mean_metric(score_reads(facts, read_disabled), "exact_success"),
    }


def baseline_success_and_bits(fact_count: int, params: int, useful_bits: int, caps: dict[str, int], value_bits: int, provenance_bits: int) -> dict[str, float]:
    k_bits = key_bits(caps)
    scalar_capacity = max(1, int(params))
    scalar_success = min(1.0, float(scalar_capacity) / float(max(1, fact_count)))
    mini_memory_capacity = 64
    mini_success = min(1.0, float(mini_memory_capacity) / float(max(1, fact_count)))
    verbatim_bits = int(fact_count) * (k_bits + int(value_bits) + int(provenance_bits))
    product_key_bits = verbatim_bits + k_bits * 2
    mini_memory_bits = mini_memory_capacity * (k_bits + int(value_bits) + int(provenance_bits))
    return {
        "scalar_mlp_success": float(scalar_success),
        "product_key_success": 1.0,
        "verbatim_table_success": 1.0,
        "content_routed_sparse_read_success": 1.0,
        "mini_titans_miras_success": float(mini_success),
        "scalar_mlp_strict_density": float(useful_bits) * scalar_success / max(float(params), 1.0),
        "product_key_strict_density": float(useful_bits) / max(float(params) + float(product_key_bits) / 16.0, 1.0),
        "verbatim_table_strict_density": float(useful_bits) / max(float(verbatim_bits) / 16.0, 1.0),
        "content_routed_sparse_read_strict_density": float(useful_bits) / max(float(verbatim_bits) / 16.0, 1.0),
        "mini_titans_miras_strict_density": float(useful_bits) * mini_success / max(float(params) + float(mini_memory_bits) / 16.0, 1.0),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    caps = capacities()
    fact_count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    train_facts, test_facts = split_schema_sets(seed, int(TRAIN_FACTS), fact_count)
    cell = SchemaDensityNeuronCell(caps)
    cell.fit(train_facts)
    metrics = evaluate_cell(cell, test_facts)
    random_train = generate_random_facts(seed + 701, int(TRAIN_FACTS), 0)
    random_test = generate_random_facts(seed + 701, fact_count, int(TRAIN_FACTS))
    random_cell = SchemaDensityNeuronCell(caps)
    random_cell.fit(random_train[: random_cell.width])
    random_metrics = evaluate_cell(random_cell, random_test)
    disabled_cell = SchemaDensityNeuronCell(caps)
    disabled_cell.fit(train_facts, disabled=True)
    write_disabled_success = mean_metric(score_reads(test_facts, [disabled_cell.read(tuple(fact["key"])) for fact in test_facts]), "exact_success")
    params = cell.parameter_count()
    value_bits = bits_for_cardinality(int(caps["value_mod"]))
    provenance_bits = bits_for_cardinality(int(caps["provenance_mod"]))
    useful_bits = int(fact_count) * (value_bits + provenance_bits)
    generator_schema_bits = 64
    decoder_bits = 64
    key_schema_bits = key_bits(caps) + sum(bits_for_cardinality(caps[key]) for key in ("domains", "entities", "relations", "qualifiers"))
    coefficient_bits = cell.width * value_bits + cell.width * provenance_bits
    committed_state_bits = int(coefficient_bits + generator_schema_bits + decoder_bits + key_schema_bits)
    training_supervision_bits = int(TRAIN_FACTS) * (key_bits(caps) + value_bits + provenance_bits)
    params_only_density = float(useful_bits) / max(float(params), 1.0)
    strict_density = float(useful_bits) / max(float(params) + float(committed_state_bits) / 16.0, 1.0)
    strict_with_supervision_density = float(useful_bits) / max(float(params) + float(committed_state_bits + training_supervision_bits) / 16.0, 1.0)
    target_density = float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    baselines = baseline_success_and_bits(fact_count, params, useful_bits, caps, value_bits, provenance_bits)
    baseline_best = max(
        float(baselines["product_key_strict_density"]),
        float(baselines["verbatim_table_strict_density"]),
        float(baselines["content_routed_sparse_read_strict_density"]),
        float(baselines["mini_titans_miras_strict_density"]),
    )
    controls_collapse = float(
        int(
            metrics["no_memory_success"] == 0.0
            and metrics["recency_only_success"] <= 0.01
            and metrics["shuffled_key_success"] == 0.0
            and metrics["shuffled_value_success"] <= 0.01
            and metrics["shuffled_provenance_success"] <= 0.01
            and metrics["schema_disabled_success"] == 0.0
            and metrics["shuffled_schema_success"] == 0.0
            and metrics["decoder_disabled_success"] == 0.0
            and metrics["read_disabled_success"] == 0.0
            and write_disabled_success == 0.0
        )
    )
    train_keys = {int(row["key_index"]) for row in train_facts}
    test_keys = {int(row["key_index"]) for row in test_facts}
    no_key_leakage = float(int(not bool(train_keys.intersection(test_keys))))
    random_entropy_control_success = float(random_metrics["exact_retrieval_success"])
    no_per_fact_committed_rows = 1.0
    structured_strict_pass = float(
        int(
            metrics["exact_retrieval_success"] >= 0.95
            and strict_density >= target_density
            and strict_with_supervision_density >= target_density
            and strict_density > baseline_best
            and random_entropy_control_success <= 0.01
            and controls_collapse == 1.0
            and no_key_leakage == 1.0
            and no_per_fact_committed_rows == 1.0
        )
    )
    target_valid_for_high_density_knowledge = 0.0
    structured_boundary_result = structured_strict_pass
    independent_random_600x_pass = 0.0
    return {
        "profile": profile,
        "fact_count": float(fact_count),
        "train_fact_count": float(TRAIN_FACTS),
        "parameter_count": float(params),
        "feature_count": float(cell.width),
        "value_bits": float(value_bits),
        "provenance_bits": float(provenance_bits),
        "useful_retrievable_bits": float(useful_bits),
        "committed_state_bits": float(committed_state_bits),
        "coefficient_bits": float(coefficient_bits),
        "generator_schema_bits": float(generator_schema_bits),
        "decoder_bits": float(decoder_bits),
        "key_schema_bits": float(key_schema_bits),
        "training_supervision_bits": float(training_supervision_bits),
        "params_only_density": float(params_only_density),
        "strict_density": float(strict_density),
        "strict_with_supervision_density": float(strict_with_supervision_density),
        "target_density": float(target_density),
        "params_only_multiplier": float(params_only_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "strict_multiplier": float(strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "strict_with_supervision_multiplier": float(strict_with_supervision_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "strict_density_advantage_over_best_baseline": float(strict_density - baseline_best),
        "no_key_leakage": no_key_leakage,
        "controls_collapse": controls_collapse,
        "random_entropy_control_success": random_entropy_control_success,
        "independent_random_600x_pass": independent_random_600x_pass,
        "no_per_fact_committed_rows": no_per_fact_committed_rows,
        "structured_strict_600x_pass": structured_strict_pass,
        "structured_boundary_result": structured_boundary_result,
        "target_valid_for_high_density_knowledge": target_valid_for_high_density_knowledge,
        "write_disabled_success": float(write_disabled_success),
        **metrics,
        **baselines,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        "local_100k_schema_density_cell_evaluated": 1.0,
        "local_100k_schema_density_cell_structured_strict_compression_authorized": 0.0,
        "local_100k_schema_density_cell_general_independent_fact_breakthrough_authorized": 0.0,
        "local_100k_schema_density_cell_full_nm_authorized": 0.0,
        "local_100k_schema_density_cell_paid_compute_authorized": 0.0,
        "local_100k_schema_density_cell_external_simulator_authorized": 0.0,
        "local_100k_schema_density_cell_arbitrary_chat_authorized": 0.0,
        "local_100k_schema_density_cell_engineering_pass": float(row["structured_boundary_result"]),
        "local_100k_schema_density_cell_claim_limited_to_structured_facts": 1.0,
        "local_100k_schema_density_cell_target_rejected_by_user": 1.0,
    }
    for key, value in row.items():
        if key in {"profile"}:
            continue
        summary[f"local_100k_schema_density_cell_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_schema_density_cell_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_100k_schema_density_cell",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "seed": int(SEED),
            "facts_smoke": int(FACTS_SMOKE),
            "facts_hard": int(FACTS_HARD),
            "train_facts": int(TRAIN_FACTS),
            "value_mod": int(VALUE_MOD),
            "provenance_mod": int(PROVENANCE_MOD),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["local_100k_schema_density_cell_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_schema_density_cell_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
