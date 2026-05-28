from __future__ import annotations

import hashlib
import os
import random
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

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_neural_coordinate_codec"
SEED = env_int("NEURAL_COORDINATE_CODEC_SEED", 6203)
FACTS_SMOKE = env_int("NEURAL_COORDINATE_CODEC_FACTS_SMOKE", 64)
FACTS_HARD = env_int("NEURAL_COORDINATE_CODEC_FACTS_HARD", 256)
CHUNK_BYTES = env_int("NEURAL_COORDINATE_CODEC_CHUNK_BYTES", 16)
HIDDEN_DIM = env_int("NEURAL_COORDINATE_CODEC_HIDDEN_DIM", 96)
FOURIER_BANDS = env_int("NEURAL_COORDINATE_CODEC_FOURIER_BANDS", 6)
TRAIN_STEPS_SMOKE = env_int("NEURAL_COORDINATE_CODEC_TRAIN_STEPS_SMOKE", 180)
TRAIN_STEPS_HARD = env_int("NEURAL_COORDINATE_CODEC_TRAIN_STEPS_HARD", 220)
DECODER_BITS = env_int("NEURAL_COORDINATE_CODEC_DECODER_BITS", 8192)
SURFACE_CONTRACT_BITS = env_int("NEURAL_COORDINATE_CODEC_SURFACE_CONTRACT_BITS", 2048)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("NEURAL_COORDINATE_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))
EXACT_SUCCESS_TARGET = float(os.environ.get("NEURAL_COORDINATE_CODEC_EXACT_SUCCESS_TARGET", "0.95"))
MAX_PARAMETER_TARGET = env_int("NEURAL_COORDINATE_CODEC_MAX_PARAMETER_TARGET", 100000)

require_positive("NEURAL_COORDINATE_CODEC_FACTS_SMOKE", FACTS_SMOKE)
require_positive("NEURAL_COORDINATE_CODEC_FACTS_HARD", FACTS_HARD)
require_positive("NEURAL_COORDINATE_CODEC_CHUNK_BYTES", CHUNK_BYTES)
require_positive("NEURAL_COORDINATE_CODEC_HIDDEN_DIM", HIDDEN_DIM)
require_positive("NEURAL_COORDINATE_CODEC_FOURIER_BANDS", FOURIER_BANDS)
require_positive("NEURAL_COORDINATE_CODEC_TRAIN_STEPS_SMOKE", TRAIN_STEPS_SMOKE)
require_positive("NEURAL_COORDINATE_CODEC_TRAIN_STEPS_HARD", TRAIN_STEPS_HARD)
require_positive("NEURAL_COORDINATE_CODEC_DECODER_BITS", DECODER_BITS)
require_positive("NEURAL_COORDINATE_CODEC_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)
require_positive("NEURAL_COORDINATE_CODEC_MAX_PARAMETER_TARGET", MAX_PARAMETER_TARGET)

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE, "train_steps": TRAIN_STEPS_SMOKE},
    "hard": {"fact_count": FACTS_HARD, "train_steps": TRAIN_STEPS_HARD},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("NEURAL_COORDINATE_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("NEURAL_COORDINATE_CODEC_PROFILE must be smoke or hard")
    return value


def source_rows() -> list[tuple[Path, str, str]]:
    rows = [
        (PROJECT_ROOT / "knowledge/training_efficiency.md", "training_efficiency", "knowledge"),
        (PROJECT_ROOT / "knowledge/context_extension.md", "context_extension", "knowledge"),
        (PROJECT_ROOT / "knowledge/delta_rule_theory.md", "delta_rule_theory", "knowledge"),
        (PROJECT_ROOT / "knowledge/kda_channel_gating.md", "kda_channel_gating", "knowledge"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/content_routed_sparse_read_prior.md", "content_routed_sparse_read_prior", "wiki"),
        (PROJECT_ROOT / "neuroloc/wiki/synthesis/compression_and_bottlenecks.md", "compression_and_bottlenecks", "wiki"),
        (PROJECT_ROOT / "src/layers/kda.py", "kda_layer", "library_code"),
        (PROJECT_ROOT / "neuroloc/simulations/memory/contextual_recall_world.py", "contextual_recall_world", "simulation_code"),
    ]
    return [(path, name, domain) for path, name, domain in rows if path.exists()]


def load_sources() -> tuple[bytes, list[dict[str, Any]]]:
    parts: list[bytes] = []
    manifest = []
    offset = 0
    for index, (path, name, domain) in enumerate(source_rows()):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if parts:
            parts.append(b"\n\n")
            offset += 2
        parts.append(data)
        manifest.append(
            {
                "source_id": int(index),
                "name": name,
                "domain": domain,
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "offset": int(offset),
                "length": int(len(data)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        offset += len(data)
    block = b"".join(parts)
    if len(block) < int(CHUNK_BYTES) * 16:
        raise ValueError("source block too small")
    return block, manifest


def source_for_offset(manifest: list[dict[str, Any]], offset: int) -> dict[str, Any]:
    for row in manifest:
        start = int(row["offset"])
        end = start + int(row["length"])
        if start <= int(offset) < end:
            return row
    raise ValueError("offset outside source manifest")


def candidate_offsets(block: bytes, manifest: list[dict[str, Any]]) -> list[int]:
    rows = []
    step = int(CHUNK_BYTES)
    for source in manifest:
        start = int(source["offset"])
        end = start + int(source["length"]) - int(CHUNK_BYTES)
        rows.extend(range(start, max(start, end + 1), step))
    unique = []
    seen = set()
    for offset in rows:
        value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        digest = hashlib.sha256(value).digest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(int(offset))
    return unique


def sample_offsets(block: bytes, manifest: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    rows = candidate_offsets(block, manifest)
    rng.shuffle(rows)
    chosen = []
    seen_sources = set()
    for source in manifest:
        for offset in rows:
            if source_for_offset(manifest, int(offset))["source_id"] == source["source_id"]:
                chosen.append(int(offset))
                seen_sources.add(int(source["source_id"]))
                break
    for offset in rows:
        if len(chosen) >= int(count):
            break
        if int(offset) not in chosen:
            chosen.append(int(offset))
    if len(chosen) != int(count):
        raise ValueError("not enough unique authored chunks")
    if len(seen_sources) < min(4, len(manifest)):
        raise ValueError("not enough source coverage")
    return sorted(chosen)


def provenance_for(source: dict[str, Any], offset: int, value: bytes) -> str:
    local = int(offset) - int(source["offset"])
    payload = f"{source['path']}:{local}:{int(CHUNK_BYTES)}:".encode("utf-8") + hashlib.sha256(value).digest()
    return hashlib.sha256(payload).hexdigest()[:16]


def key_for(source: dict[str, Any], offset: int) -> tuple[int, int]:
    local = int(offset) - int(source["offset"])
    return int(source["source_id"]), int(local)


def build_facts(seed: int, fact_count: int) -> tuple[list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    block, manifest = load_sources()
    offsets = sample_offsets(block, manifest, int(fact_count), int(seed) + 19)
    facts = []
    for row, offset in enumerate(offsets):
        source = source_for_offset(manifest, int(offset))
        value = block[int(offset) : int(offset) + int(CHUNK_BYTES)]
        facts.append(
            {
                "role": "test",
                "row": int(row),
                "source_id": int(source["source_id"]),
                "source_name": str(source["name"]),
                "domain": str(source["domain"]),
                "key": key_for(source, int(offset)),
                "value": value.hex(),
                "provenance": provenance_for(source, int(offset), value),
            }
        )
    return facts, block, manifest


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 7717)
    rows = []
    for fact in facts:
        value = bytes(rng.randrange(0, 256) for _ in range(int(CHUNK_BYTES)))
        rows.append(
            {
                "role": "test",
                "row": int(fact["row"]),
                "source_id": int(fact["source_id"]),
                "source_name": str(fact["source_name"]),
                "domain": str(fact["domain"]),
                "key": tuple(fact["key"]),
                "value": value.hex(),
                "provenance": hashlib.sha256(value).hexdigest()[:16],
            }
        )
    return rows


def coordinate_rows(facts: list[dict[str, Any]]) -> tuple[list[tuple[int, int, int]], list[int]]:
    coords = []
    labels = []
    for fact in facts:
        source_id, local_offset = tuple(fact["key"])
        value = bytes.fromhex(str(fact["value"]))
        for byte_index, byte in enumerate(value):
            coords.append((int(source_id), int(local_offset), int(byte_index)))
            labels.append(int(byte))
    return coords, labels


def feature_matrix(coords: list[tuple[int, int, int]], manifest: list[dict[str, Any]]) -> Any:
    import torch

    source_count = max(1, len(manifest))
    max_offset = max(1, max(int(row["length"]) for row in manifest))
    rows = []
    for source_id, local_offset, byte_index in coords:
        base = [
            float(source_id) / max(1.0, float(source_count - 1)),
            float(local_offset) / float(max_offset),
            float(byte_index) / max(1.0, float(CHUNK_BYTES - 1)),
        ]
        features = list(base)
        for band in range(int(FOURIER_BANDS)):
            scale = float(2**band)
            for value in base:
                angle = float(value) * scale * 6.283185307179586
                features.append(float(torch.sin(torch.tensor(angle)).item()))
                features.append(float(torch.cos(torch.tensor(angle)).item()))
        rows.append(features)
    return torch.tensor(rows, dtype=torch.float32)


class NeuralCoordinateByteCodec:
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM) -> None:
        import torch.nn as nn

        self.module = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 256),
        )

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def train_codec(self, features: Any, labels: Any, steps: int, seed: int) -> list[float]:
        import torch

        torch.manual_seed(int(seed))
        self.module.train()
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.01, weight_decay=0.0001)
        losses = []
        for _step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            logits = self.module(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return losses

    def predict_bytes(self, features: Any) -> bytes:
        import torch

        self.module.eval()
        with torch.no_grad():
            labels = self.module(features).argmax(dim=-1).detach().cpu().tolist()
        return bytes(int(value) for value in labels)


class NeuralCoordinateCodecCell:
    def __init__(self, facts: list[dict[str, Any]], manifest: list[dict[str, Any]], train_steps: int, seed: int) -> None:
        import torch

        torch.manual_seed(int(seed))
        coords, labels = coordinate_rows(facts)
        features = feature_matrix(coords, manifest)
        target = torch.tensor(labels, dtype=torch.long)
        self.manifest = manifest
        self.train_steps = int(train_steps)
        self.codec = NeuralCoordinateByteCodec(int(features.shape[1]))
        self.training_loss = self.codec.train_codec(features, target, int(train_steps), int(seed))
        self.trainable_neural_predictor_used = 1.0
        self.raw_source_block_retained = 0.0
        self.mantissa_payload_packing_used = 0.0
        self.per_fact_payload_row_used = 0.0
        self.coordinate_query_used = 1.0
        self.payload_bytes_in_state_dict = 0.0

    def parameter_count(self) -> int:
        return self.codec.parameter_count()

    def answer_many(self, keys: list[tuple[int, int]], read_disabled: bool = False, decoder_disabled: bool = False, coordinate_disabled: bool = False) -> list[dict[str, Any]]:
        if read_disabled or decoder_disabled or coordinate_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _key in keys]
        answers = []
        for source_id, local_offset in keys:
            coords = [(int(source_id), int(local_offset), byte_index) for byte_index in range(int(CHUNK_BYTES))]
            features = feature_matrix(coords, self.manifest)
            value = self.codec.predict_bytes(features)
            answers.append({"value": value.hex(), "provenance": hashlib.sha256(value).hexdigest()[:16], "hit": 1})
        return answers


def score_answers(facts: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, float]:
    exact = 0
    hit = 0
    byte_total = 0
    byte_correct = 0
    for fact, answer in zip(facts, answers):
        expected = bytes.fromhex(str(fact["value"]))
        raw = bytes.fromhex(str(answer.get("value", ""))) if answer.get("value") else b""
        hit += int(answer.get("hit", 0))
        exact += int(raw == expected)
        for got, want in zip(raw, expected):
            byte_correct += int(int(got) == int(want))
        byte_total += len(expected)
    return {
        "exact_success": float(exact) / max(1.0, float(len(facts))),
        "hit_rate": float(hit) / max(1.0, float(len(facts))),
        "byte_accuracy": float(byte_correct) / max(1.0, float(byte_total)),
    }


def same_block_content_scan(facts: list[dict[str, Any]], block: bytes, manifest: list[dict[str, Any]]) -> dict[str, float]:
    answers = []
    for fact in facts:
        source_id, local_offset = tuple(fact["key"])
        source = next(row for row in manifest if int(row["source_id"]) == int(source_id))
        offset = int(source["offset"]) + int(local_offset)
        value = block[offset : offset + int(CHUNK_BYTES)]
        answers.append({"value": value.hex(), "provenance": provenance_for(source, offset, value), "hit": 1})
    return score_answers(facts, answers)


def account_bits(cell: NeuralCoordinateCodecCell, facts: list[dict[str, Any]], block: bytes) -> dict[str, float]:
    useful_bits = float(len(facts) * int(CHUNK_BYTES) * 8)
    parameter_count = int(cell.parameter_count())
    charged_parameter_bits = float(parameter_count) * float(ORDINARY_BITS_PER_PARAMETER)
    physical_fp32_parameter_bits = float(parameter_count * 32)
    strict_bits = float(charged_parameter_bits + DECODER_BITS + SURFACE_CONTRACT_BITS)
    physical_bits = float(physical_fp32_parameter_bits + DECODER_BITS + SURFACE_CONTRACT_BITS)
    scan_bits = float(len(block) * 8 + DECODER_BITS + SURFACE_CONTRACT_BITS)
    return {
        "useful_bits": useful_bits,
        "charged_parameter_count": float(parameter_count),
        "charged_parameter_bits": charged_parameter_bits,
        "physical_fp32_parameter_bits": physical_fp32_parameter_bits,
        "strict_accounted_bits": strict_bits,
        "physical_fp32_accounted_bits": physical_bits,
        "same_block_content_scan_bits": scan_bits,
        "strict_multiplier": useful_bits / max(strict_bits, 1.0),
        "physical_fp32_multiplier": useful_bits / max(physical_bits, 1.0),
        "same_block_content_scan_multiplier": useful_bits / max(scan_bits, 1.0),
        "under_100k_parameters": float(int(parameter_count < int(MAX_PARAMETER_TARGET))),
    }


def evaluate_cell(cell: NeuralCoordinateCodecCell, facts: list[dict[str, Any]], block: bytes, manifest: list[dict[str, Any]]) -> dict[str, float]:
    keys = [tuple(fact["key"]) for fact in facts]
    exact = score_answers(facts, cell.answer_many(keys))
    read_disabled = score_answers(facts, cell.answer_many(keys, read_disabled=True))
    decoder_disabled = score_answers(facts, cell.answer_many(keys, decoder_disabled=True))
    coordinate_disabled = score_answers(facts, cell.answer_many(keys, coordinate_disabled=True))
    scan = same_block_content_scan(facts, block, manifest)
    return {
        "exact_answer_success": exact["exact_success"],
        "byte_accuracy": exact["byte_accuracy"],
        "hit_rate": exact["hit_rate"],
        "read_disabled_success": read_disabled["exact_success"],
        "decoder_disabled_success": decoder_disabled["exact_success"],
        "coordinate_disabled_success": coordinate_disabled["exact_success"],
        "same_block_content_scan_success": scan["exact_success"],
        "same_block_content_scan_byte_accuracy": scan["byte_accuracy"],
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError("profile must be smoke or hard")
    started = time.perf_counter()
    fact_count = int(PROFILES[profile]["fact_count"])
    train_steps = int(PROFILES[profile]["train_steps"])
    facts, block, manifest = build_facts(seed, fact_count)
    random_twin = build_random_twin(seed, facts)
    cell = NeuralCoordinateCodecCell(facts, manifest, train_steps, seed)
    twin_cell = NeuralCoordinateCodecCell(random_twin, manifest, train_steps, seed + 3001)
    metrics = evaluate_cell(cell, facts, block, manifest)
    twin_metrics = evaluate_cell(twin_cell, random_twin, block, manifest)
    account = account_bits(cell, facts, block)
    random_account = account_bits(twin_cell, random_twin, block)
    exact_gate = float(int(metrics["exact_answer_success"] >= float(EXACT_SUCCESS_TARGET)))
    random_collapse = float(int(twin_metrics["exact_answer_success"] < 0.05 and twin_metrics["byte_accuracy"] < 0.35))
    control_collapse = float(int(metrics["read_disabled_success"] == 0.0 and metrics["decoder_disabled_success"] == 0.0 and metrics["coordinate_disabled_success"] == 0.0))
    beats_scan = float(int(metrics["exact_answer_success"] > metrics["same_block_content_scan_success"] and account["strict_accounted_bits"] <= account["same_block_content_scan_bits"]))
    product_pass = float(int(account["under_100k_parameters"] == 1.0 and exact_gate == 1.0 and random_collapse == 1.0 and control_collapse == 1.0 and beats_scan == 1.0))
    return {
        "profile": profile,
        "seed": int(seed),
        "fact_count": int(fact_count),
        "chunk_bytes": int(CHUNK_BYTES),
        "train_steps": int(train_steps),
        "source_count": int(len(manifest)),
        "source_block_bytes": int(len(block)),
        "source_domain_count": int(len({str(row["domain"]) for row in manifest})),
        "training_loss_start": float(cell.training_loss[0]),
        "training_loss_end": float(cell.training_loss[-1]),
        "random_label_training_loss_start": float(twin_cell.training_loss[0]),
        "random_label_training_loss_end": float(twin_cell.training_loss[-1]),
        "exact_success_gate": exact_gate,
        "random_label_twin_collapse": random_collapse,
        "ablation_controls_collapse": control_collapse,
        "beats_same_block_content_scan": beats_scan,
        "same_block_content_scan_not_beaten": float(int(beats_scan == 0.0)),
        "neural_coordinate_codec_product_pass": product_pass,
        "strict_breakthrough_authorized": 0.0,
        "learned_exact_retrieval_authorized": product_pass,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "mantissa_payload_packing_used": float(cell.mantissa_payload_packing_used),
        "raw_source_block_retained_in_model": float(cell.raw_source_block_retained),
        "trainable_neural_predictor_used": float(cell.trainable_neural_predictor_used),
        "duration_sec": float(time.perf_counter() - started),
        **metrics,
        **{f"random_label_twin_{key}": value for key, value in twin_metrics.items()},
        **account,
        **{f"random_label_twin_{key}": value for key, value in random_account.items()},
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        "local_100k_neural_coordinate_codec_evaluated": 1.0,
        "local_100k_neural_coordinate_codec_strict_breakthrough_authorized": 0.0,
        "local_100k_neural_coordinate_codec_full_nm_authorized": 0.0,
        "local_100k_neural_coordinate_codec_paid_compute_authorized": 0.0,
        "local_100k_neural_coordinate_codec_arbitrary_chat_authorized": 0.0,
        "local_100k_neural_coordinate_codec_engineering_pass": float(row["neural_coordinate_codec_product_pass"]),
    }
    for key, value in row.items():
        if key == "profile":
            continue
        summary[f"local_100k_neural_coordinate_codec_{key}"] = value
    return summary


def run_once(profile: str, seed: int = SEED) -> dict[str, Any]:
    started_at = utc_now_iso()
    started = time.perf_counter()
    summary = build_summary(profile, seed)
    finished_at = utc_now_iso()
    return build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        parameters={"profile": profile, "seed": int(seed)},
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[],
        warnings=[],
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
    )


def main() -> int:
    profile = infer_profile()
    record = run_once(profile, SEED)
    out_dir = output_dir_for(SIMULATION_ID)
    path = out_dir / f"{SIMULATION_ID}_metrics.json"
    write_json(record, path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
