from __future__ import annotations

import hashlib
import math
import os
import random
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
SEED = env_int("UNKNOWN_CELL_SEED", 631)
FACTS_SMOKE = env_int("UNKNOWN_CELL_FACTS_SMOKE", 256)
FACTS_HARD = env_int("UNKNOWN_CELL_FACTS_HARD", 4096)
CHUNK_BYTES = env_int("UNKNOWN_CELL_CHUNK_BYTES", 32)
DECODER_BITS = env_int("UNKNOWN_CELL_DECODER_BITS", 65536)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("UNKNOWN_CELL_ORDINARY_BITS_PER_PARAMETER", "2.5"))
TARGET_MULTIPLIER = float(os.environ.get("UNKNOWN_CELL_TARGET_MULTIPLIER", "600.0"))

require_positive("UNKNOWN_CELL_FACTS_SMOKE", FACTS_SMOKE)
require_positive("UNKNOWN_CELL_FACTS_HARD", FACTS_HARD)
require_positive("UNKNOWN_CELL_CHUNK_BYTES", CHUNK_BYTES)
require_positive("UNKNOWN_CELL_DECODER_BITS", DECODER_BITS)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("UNKNOWN_CELL_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("UNKNOWN_CELL_PROFILE must be smoke or hard")
    return value


def bits_for_cardinality(cardinality: int) -> int:
    return max(1, math.ceil(math.log2(max(2, int(cardinality)))))


def corpus_paths() -> list[Path]:
    names = [
        "neuroloc/wiki/PROJECT_PLAN.md",
        "neuroloc/wiki/synthesis/neural_model_lane_operation_preserving_compression.md",
        "neuroloc/wiki/synthesis/high_density_neuron_cell_related_work_pressure_matrix.md",
        "neuroloc/wiki/tests/local_100k_high_density_cell.md",
        "neuroloc/wiki/tests/local_100k_unstructured_density_cell.md",
        "neuroloc/wiki/mistakes/schema_density_cell_structured_target_category_error.md",
        "neuroloc/wiki/mistakes/unstructured_exact_600x_entropy_wall.md",
    ]
    return [PROJECT_ROOT / name for name in names if (PROJECT_ROOT / name).exists()]


def load_corpus() -> tuple[bytes, list[dict[str, Any]]]:
    blobs: list[bytes] = []
    manifest = []
    offset = 0
    for index, path in enumerate(corpus_paths()):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if blobs:
            blobs.append(b"\n\n")
            offset += 2
        digest = hashlib.sha256(data).hexdigest()
        blobs.append(data)
        manifest.append({"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "index": int(index), "offset": int(offset), "length": int(len(data)), "sha256": digest})
        offset += len(data)
    corpus = b"".join(blobs)
    if len(corpus) < int(CHUNK_BYTES) * 4:
        raise ValueError("corpus too small")
    return corpus, manifest


def sample_offsets(corpus: bytes, manifest: list[dict[str, Any]], count: int, chunk_bytes: int, seed: int) -> list[int]:
    stride = int(chunk_bytes)
    candidates = []
    for item in manifest:
        start = int(item["offset"])
        end = start + int(item["length"]) - int(chunk_bytes)
        candidates.extend(range(start, end + 1, stride))
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    chosen = []
    seen = set()
    for offset in candidates:
        chunk = corpus[int(offset) : int(offset) + int(chunk_bytes)]
        digest = hashlib.sha256(chunk).digest()
        if digest in seen:
            continue
        seen.add(digest)
        chosen.append(int(offset))
        if len(chosen) == int(count):
            return sorted(chosen)
    raise ValueError("not enough unique non-overlapping chunks")


def manifest_bits(manifest: list[dict[str, Any]]) -> int:
    payload = ";".join(
        f"{row['path']}:{int(row['index'])}:{int(row['offset'])}:{int(row['length'])}:{row['sha256']}"
        for row in manifest
    ).encode("utf-8")
    return int(len(payload) * 8)


def provenance_for_offset(manifest: list[dict[str, Any]], offset: int) -> str:
    for row in manifest:
        start = int(row["offset"])
        end = start + int(row["length"])
        if start <= int(offset) < end:
            local_offset = int(offset) - start
            return hashlib.sha256(f"{row['path']}:{local_offset}:{int(CHUNK_BYTES)}".encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(f"separator:{int(offset)}:{int(CHUNK_BYTES)}".encode("utf-8")).hexdigest()[:16]


def build_unknown_facts(seed: int, count: int) -> tuple[list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    corpus, manifest = load_corpus()
    offsets = sample_offsets(corpus, manifest, count, int(CHUNK_BYTES), seed)
    facts = []
    for row, offset in enumerate(offsets):
        chunk = corpus[int(offset) : int(offset) + int(CHUNK_BYTES)]
        facts.append({"key": int(offset), "value": chunk.hex(), "provenance": provenance_for_offset(manifest, int(offset)), "row": int(row)})
    return facts, corpus, manifest


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 4441)
    twin = []
    for fact in facts:
        value = bytes(rng.randrange(0, 256) for _ in range(int(CHUNK_BYTES)))
        twin.append({"key": int(fact["key"]), "value": value.hex(), "provenance": hashlib.sha256(value).hexdigest()[:16], "row": int(fact["row"])})
    return twin


class UnknownStructureCompressedCell:
    def __init__(self, corpus: bytes, manifest: list[dict[str, Any]]) -> None:
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.read_gate = nn.Parameter(torch.ones(1))
                self.decoder_gate = nn.Parameter(torch.ones(1))
                self.provenance_gate = nn.Parameter(torch.ones(1))
                self.address_gate = nn.Parameter(torch.ones(1))

        self.module = Module()
        self.compressed = zlib.compress(corpus, level=9)
        self.length = len(corpus)
        self.manifest = list(manifest)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def committed_state_bits(self) -> int:
        return int(len(self.compressed) * 8 + bits_for_cardinality(self.length) + int(DECODER_BITS) + manifest_bits(self.manifest))

    def read(self, key: int, disabled: bool = False, decoder_disabled: bool = False) -> dict[str, str | int]:
        if disabled or decoder_disabled:
            return {"value": "", "provenance": "", "hit": 0}
        corpus = zlib.decompress(self.compressed)
        start = int(key)
        if start < 0 or start + int(CHUNK_BYTES) > len(corpus):
            return {"value": "", "provenance": "", "hit": 0}
        chunk = corpus[start : start + int(CHUNK_BYTES)]
        return {"value": chunk.hex(), "provenance": provenance_for_offset(self.manifest, start), "hit": 1}


def score_reads(facts: list[dict[str, Any]], reads: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for fact, read in zip(facts, reads):
        value_ok = str(read["value"]) == str(fact["value"])
        provenance_ok = str(read["provenance"]) == str(fact["provenance"])
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


def evaluate(cell: UnknownStructureCompressedCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    exact = [cell.read(int(fact["key"])) for fact in facts]
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _ in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shifted_reads = [cell.read(int(fact["key"])) for fact in shifted(facts)]
    exact_reads = [cell.read(int(fact["key"])) for fact in facts]
    shuffled_values = [{"value": row["value"], "provenance": read["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    shuffled_provenance = [{"value": read["value"], "provenance": row["provenance"], "hit": read["hit"]} for row, read in zip(shifted(exact_reads), exact_reads)]
    read_disabled = [cell.read(int(fact["key"]), disabled=True) for fact in facts]
    decoder_disabled = [cell.read(int(fact["key"]), decoder_disabled=True) for fact in facts]
    return {
        "exact_retrieval_success": mean_metric(score_reads(facts, exact), "exact_success"),
        "no_memory_success": mean_metric(score_reads(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_reads(facts, recency), "exact_success"),
        "shuffled_key_success": mean_metric(score_reads(facts, shifted_reads), "exact_success"),
        "shuffled_value_success": mean_metric(score_reads(facts, shuffled_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_reads(facts, shuffled_provenance), "exact_success"),
        "read_disabled_success": mean_metric(score_reads(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_reads(facts, decoder_disabled), "exact_success"),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    count = int(FACTS_HARD if profile == "hard" else FACTS_SMOKE)
    facts, corpus, manifest = build_unknown_facts(seed, count)
    random_twin = build_random_twin(seed, facts)
    cell = UnknownStructureCompressedCell(corpus, manifest)
    metrics = evaluate(cell, facts)
    twin_metrics = evaluate(cell, random_twin)
    params = cell.parameter_count()
    committed_state_bits = cell.committed_state_bits()
    useful_bits = int(len(facts)) * int(CHUNK_BYTES) * 8
    unique_source_bits = int(len({int(fact["key"]) for fact in facts})) * int(CHUNK_BYTES) * 8
    query_key_bits = int(len(facts)) * bits_for_cardinality(len(corpus))
    target_density = float(TARGET_MULTIPLIER) * float(ORDINARY_BITS_PER_PARAMETER)
    strict_density = float(unique_source_bits) / max(float(params) + float(committed_state_bits) / 16.0, 1.0)
    raw_corpus_bits = len(corpus) * 8
    compressed_ratio = float(raw_corpus_bits) / max(float(committed_state_bits), 1.0)
    zlib_payload_ratio = float(raw_corpus_bits) / max(float(len(cell.compressed) * 8), 1.0)
    controls_collapse = float(
        int(
            metrics["no_memory_success"] == 0.0
            and metrics["read_disabled_success"] == 0.0
            and metrics["decoder_disabled_success"] == 0.0
            and metrics["recency_only_success"] <= 0.01
            and metrics["shuffled_key_success"] <= 0.01
            and metrics["shuffled_value_success"] <= 0.01
            and metrics["shuffled_provenance_success"] <= 0.01
        )
    )
    strict_600x_pass = float(int(metrics["exact_retrieval_success"] >= 0.95 and strict_density >= target_density and twin_metrics["exact_retrieval_success"] <= 0.01 and controls_collapse == 1.0))
    corpus_probe_pass = float(int(metrics["exact_retrieval_success"] >= 0.95 and strict_600x_pass == 0.0 and twin_metrics["exact_retrieval_success"] <= 0.01 and controls_collapse == 1.0))
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "corpus_file_count": float(len(manifest)),
        "corpus_bytes": float(len(corpus)),
        "compressed_bytes": float(len(cell.compressed)),
        "parameter_count": float(params),
        "committed_state_bits": float(committed_state_bits),
        "decoder_bits": float(DECODER_BITS),
        "manifest_bits": float(manifest_bits(manifest)),
        "query_key_bits": float(query_key_bits),
        "useful_retrievable_bits": float(useful_bits),
        "unique_source_bits": float(unique_source_bits),
        "raw_corpus_bits": float(raw_corpus_bits),
        "compressed_ratio": float(compressed_ratio),
        "zlib_payload_ratio": float(zlib_payload_ratio),
        "strict_density": float(strict_density),
        "target_density": float(target_density),
        "strict_multiplier": float(strict_density / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)),
        "strict_600x_pass": strict_600x_pass,
        "corpus_probe_pass": corpus_probe_pass,
        "useful_negative_result": corpus_probe_pass,
        "random_label_twin_success": float(twin_metrics["exact_retrieval_success"]),
        "unknown_structure_source": 1.0,
        "standard_codec_dependency": 1.0,
        "sequence_offset_key_target": 1.0,
        "associative_random_key_target": 0.0,
        "provenance_independent_of_key_and_manifest": 0.0,
        "formula_or_schema_labels_present": 0.0,
        "seed_oracle_authorized": 0.0,
        "no_per_fact_committed_rows": 1.0,
        "controls_collapse": controls_collapse,
        **metrics,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        "local_100k_unknown_structure_density_probe_evaluated": 1.0,
        "local_100k_unknown_structure_density_probe_strict_breakthrough_authorized": 0.0,
        "local_100k_unknown_structure_density_probe_general_unknown_structure_breakthrough_authorized": 0.0,
        "local_100k_unknown_structure_density_probe_full_nm_authorized": 0.0,
        "local_100k_unknown_structure_density_probe_paid_compute_authorized": 0.0,
        "local_100k_unknown_structure_density_probe_external_simulator_authorized": 0.0,
        "local_100k_unknown_structure_density_probe_arbitrary_chat_authorized": 0.0,
        "local_100k_unknown_structure_density_probe_engineering_pass": float(row["corpus_probe_pass"]),
    }
    for key, value in row.items():
        if key in {"profile"}:
            continue
        summary[f"local_100k_unknown_structure_density_probe_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_unknown_structure_density_probe_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_100k_unknown_structure_density_probe",
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
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
            "target_multiplier": float(TARGET_MULTIPLIER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["local_100k_unknown_structure_density_probe_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_unknown_structure_density_probe_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
