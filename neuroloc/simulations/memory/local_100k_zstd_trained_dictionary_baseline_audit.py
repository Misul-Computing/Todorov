from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import zstandard as zstd

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.local_100k_source_structure_block_codec import best_codec, fixed_ngrams, random_block, read_joined, target_paths
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_block_codec import learned_codec
from neuroloc.simulations.memory.local_100k_source_subtoken_structure_corpus_codec import FROZEN_BLOCKS, corpus_blocks, read_block

SCRIPT_PATH = Path(__file__).resolve()
SIMULATION_ID = "local_100k_zstd_trained_dictionary_baseline_audit"
SEED = env_int("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_SEED", 11933)
PUBLIC_DICT_HEADER_BITS = env_int("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_HEADER_BITS", 64)
PUBLIC_DICT_SELECTOR_BITS = env_int("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_SELECTOR_BITS", 16)
SUBTOKEN_STRUCTURE_HEADER_BITS = env_int("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_SUBTOKEN_HEADER_BITS", 896)
MIN_BLOCK_CHARGED_MARGIN_BITS = float(os.environ.get("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_MIN_BLOCK_CHARGED_MARGIN_BITS", "20000"))
MIN_BLOCK_UNDERCHARGED_MARGIN_BITS = float(os.environ.get("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_MIN_BLOCK_UNDERCHARGED_MARGIN_BITS", "5000"))
MIN_CORPUS_CHARGED_MARGIN_BITS = float(os.environ.get("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_MIN_CORPUS_CHARGED_MARGIN_BITS", "150000"))
MIN_CORPUS_UNDERCHARGED_MARGIN_BITS = float(os.environ.get("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_MIN_CORPUS_UNDERCHARGED_MARGIN_BITS", "50000"))

require_positive("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_HEADER_BITS", PUBLIC_DICT_HEADER_BITS)
require_positive("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_SELECTOR_BITS", PUBLIC_DICT_SELECTOR_BITS)
require_positive("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_SUBTOKEN_HEADER_BITS", SUBTOKEN_STRUCTURE_HEADER_BITS)

PROFILES = {
    "smoke": {"corpus_block_count": 3, "dict_sizes": (512, 1024, 2048, 4096), "levels": (1, 10, 15, 20, 22), "block_charged_margin_bits": 20000.0, "block_undercharged_margin_bits": 5000.0, "corpus_charged_margin_bits": 100000.0, "corpus_undercharged_margin_bits": 50000.0},
    "hard": {"corpus_block_count": 5, "dict_sizes": (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536), "levels": tuple(range(1, 23)), "block_charged_margin_bits": float(MIN_BLOCK_CHARGED_MARGIN_BITS), "block_undercharged_margin_bits": float(MIN_BLOCK_UNDERCHARGED_MARGIN_BITS), "corpus_charged_margin_bits": float(MIN_CORPUS_CHARGED_MARGIN_BITS), "corpus_undercharged_margin_bits": float(MIN_CORPUS_UNDERCHARGED_MARGIN_BITS)},
}

PUBLIC_TRAIN_PATHS = (
    "src/layers/kda.py",
    "src/layers/mamba3.py",
    "src/layers/mla.py",
    "src/model/todorov.py",
    "neuroloc/simulations/memory/slot_buffer_capacity.py",
    "neuroloc/simulations/memory/asymmetric_outer_product_recall.py",
    "notebooks/autoresearch/train.py",
    "neuroloc/model/neural_machine.py",
    "neuroloc/model/god_machine.py",
    "neuroloc/data/nm_worlds.py",
    "neuroloc/data/nm_3d_worlds.py",
    "neuroloc/simulations/reasoning/thinking_loop_prototype.py",
    "neuroloc/simulations/prototypes/rate_coded_spike.py",
    "neuroloc/simulations/sparse_coding/hierarchical_ternary.py",
    "neuroloc/simulations/plasticity/bcm_alpha_pilot.py",
    "neuroloc/simulations/dendritic/multicompartment_neuron.py",
    "scripts/kaggle_exec.py",
    "scripts/evaluate_gates.py",
    "scripts/generate_dossier.py",
    "scripts/register_run.py",
    "config.py",
)


@dataclass(frozen=True)
class DictionaryCandidate:
    requested_size: int
    level: int
    dictionary_bytes: bytes
    payloads: tuple[bytes, ...]

    @property
    def payload_bits(self) -> int:
        return int(sum(len(payload) for payload in self.payloads) * 8)

    @property
    def dictionary_bits(self) -> int:
        return int(len(self.dictionary_bytes) * 8)

    @property
    def charged_bits(self) -> int:
        return int(self.payload_bits + self.dictionary_bits + int(PUBLIC_DICT_HEADER_BITS) + int(PUBLIC_DICT_SELECTOR_BITS))

    @property
    def undercharged_bits(self) -> int:
        return int(self.payload_bits)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("ZSTD_TRAINED_DICTIONARY_BASELINE_AUDIT_PROFILE must be smoke or hard")
    return value


def candidate_public_train_paths() -> list[Path]:
    return [PROJECT_ROOT / path for path in PUBLIC_TRAIN_PATHS if (PROJECT_ROOT / path).exists()]


def select_train_paths(targets: list[Path]) -> list[Path]:
    target_resolved = {path.resolve() for path in targets if path.exists()}
    target_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in targets if path.exists()}
    selected = []
    for path in candidate_public_train_paths():
        if path.resolve() in target_resolved:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() in target_hashes:
            continue
        selected.append(path)
    if not selected:
        raise ValueError("no public train paths remain after target exclusion")
    return selected


def sample_chunks(paths: list[Path]) -> list[bytes]:
    rows = []
    for path in paths:
        data = path.read_bytes().replace(b"\r\n", b"\n")
        for size in (256, 512, 1024, 2048, 4096, 8192):
            end = max(0, len(data) - int(size) + 1)
            for index in range(0, end, int(size)):
                rows.append(data[index : index + int(size)])
        rows.append(data)
    return [row for row in rows if len(row) >= 64]


def train_dictionaries(samples: list[bytes], sizes: tuple[int, ...]) -> list[tuple[int, bytes]]:
    rows = []
    for size in sizes:
        try:
            dictionary = zstd.train_dictionary(int(size), samples)
            rows.append((int(size), dictionary.as_bytes()))
        except zstd.ZstdError:
            continue
    if not rows:
        raise ValueError("zstd dictionary training produced no candidates")
    return rows


def compress_payloads(blocks: list[bytes], dictionary_bytes: bytes, level: int) -> tuple[bytes, ...]:
    dictionary = zstd.ZstdCompressionDict(bytes(dictionary_bytes))
    compressor = zstd.ZstdCompressor(level=int(level), dict_data=dictionary)
    return tuple(compressor.compress(block) for block in blocks)


def best_trained_dictionary(blocks: list[bytes], dictionaries: list[tuple[int, bytes]], levels: tuple[int, ...]) -> DictionaryCandidate:
    best: DictionaryCandidate | None = None
    for requested_size, dictionary_bytes in dictionaries:
        for level in levels:
            candidate = DictionaryCandidate(int(requested_size), int(level), bytes(dictionary_bytes), compress_payloads(blocks, dictionary_bytes, int(level)))
            if best is None or (candidate.charged_bits, candidate.payload_bits, candidate.requested_size, candidate.level) < (best.charged_bits, best.payload_bits, best.requested_size, best.level):
                best = candidate
    if best is None:
        raise ValueError("no zstd dictionary candidates found")
    return best


def restore_payloads(candidate: DictionaryCandidate) -> tuple[bytes, ...]:
    dictionary = zstd.ZstdCompressionDict(bytes(candidate.dictionary_bytes))
    decompressor = zstd.ZstdDecompressor(dict_data=dictionary)
    return tuple(decompressor.decompress(payload) for payload in candidate.payloads)


def disabled_restore_success(candidate: DictionaryCandidate, targets: list[bytes]) -> float:
    try:
        decompressor = zstd.ZstdDecompressor()
        decoded = tuple(decompressor.decompress(payload) for payload in candidate.payloads)
        return float(decoded == tuple(targets))
    except zstd.ZstdError:
        return 0.0


def shuffled_dictionary_success(candidate: DictionaryCandidate, targets: list[bytes], seed: int) -> float:
    values = list(candidate.dictionary_bytes)
    random.Random(int(seed) + 733).shuffle(values)
    try:
        shuffled = DictionaryCandidate(candidate.requested_size, candidate.level, bytes(values), candidate.payloads)
        return float(restore_payloads(shuffled) == tuple(targets))
    except zstd.ZstdError:
        return 0.0


def subtoken_payload_bits(train_block: bytes, block: bytes) -> int:
    learned = learned_codec(train_block, block)
    return int((len(bytes(learned["count_payload"])) + len(bytes(learned["body_payload"])) + len(bytes(learned["dictionary_payload"]))) * 8 + int(SUBTOKEN_STRUCTURE_HEADER_BITS))


def overlap_counts(train: list[Path], targets: list[Path], prefix: str) -> dict[str, float]:
    train_present = [path for path in train if path.exists()]
    target_present = [path for path in targets if path.exists()]
    train_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in train_present}
    target_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in target_present}
    train_hash = {hashlib.sha256(path.read_bytes()).hexdigest() for path in train_present}
    target_hash = {hashlib.sha256(path.read_bytes()).hexdigest() for path in target_present}
    train_block = read_joined(train_present)
    target_block = read_joined(target_present)
    ngram_width = 64
    return {
        f"{prefix}_public_dict_train_test_path_overlap_count": float(len(train_rel & target_rel)),
        f"{prefix}_public_dict_train_test_hash_overlap_count": float(len(train_hash & target_hash)),
        f"{prefix}_public_dict_train_test_ngram_width_bytes": float(ngram_width),
        f"{prefix}_public_dict_train_test_ngram_overlap_count": float(len(fixed_ngrams(train_block, ngram_width) & fixed_ngrams(target_block, ngram_width))),
    }


def block_target_paths(profile: str) -> list[Path]:
    return target_paths(profile)


def corpus_target_paths(profile: str) -> list[Path]:
    paths = []
    for row in corpus_blocks(profile):
        paths.extend(PROJECT_ROOT / str(path) for path in row["paths"])
    return paths


def best_standard_bits(blocks: list[bytes]) -> int:
    return int(sum(len(best_codec(block)[1]) * 8 for block in blocks))


def random_blocks(blocks: list[bytes], seed: int) -> list[bytes]:
    return [random_block(int(seed) + index * 17, len(block)) for index, block in enumerate(blocks)]


def improvement(reference_bits: int, candidate_bits: int) -> float:
    return float(reference_bits - candidate_bits) / max(float(reference_bits), 1.0)


def measure_surface(prefix: str, blocks: list[bytes], train: list[Path], target_files: list[Path], current_bits: int, seed: int, profile: str) -> dict[str, float]:
    samples = sample_chunks(train)
    dictionaries = train_dictionaries(samples, tuple(PROFILES[profile]["dict_sizes"]))
    levels = tuple(PROFILES[profile]["levels"])
    best_public = best_trained_dictionary(blocks, dictionaries, levels)
    restored = restore_payloads(best_public)
    random_payloads = random_blocks(blocks, int(seed))
    best_public_random = best_trained_dictionary(random_payloads, dictionaries, levels)
    standard_bits = best_standard_bits(blocks)
    random_standard_bits = best_standard_bits(random_payloads)
    overlap = overlap_counts(train, target_files, prefix)
    exact_success = float(restored == tuple(blocks))
    disabled_success = disabled_restore_success(best_public, blocks)
    shuffled_success = shuffled_dictionary_success(best_public, blocks, int(seed))
    charged_margin = float(best_public.charged_bits - int(current_bits))
    undercharged_margin = float(best_public.undercharged_bits - int(current_bits))
    train_only = float(int(overlap[f"{prefix}_public_dict_train_test_path_overlap_count"] == 0.0 and overlap[f"{prefix}_public_dict_train_test_hash_overlap_count"] == 0.0))
    return {
        f"{prefix}_public_dict_train_only": train_only,
        f"{prefix}_public_dict_train_file_count": float(len(train)),
        f"{prefix}_public_dict_training_sample_count": float(len(samples)),
        f"{prefix}_public_dict_exact_reconstruction_success": exact_success,
        f"{prefix}_public_dict_disabled_reconstruction_success": disabled_success,
        f"{prefix}_public_dict_shuffled_reconstruction_success": shuffled_success,
        f"{prefix}_current_subtoken_payload_bits": float(current_bits),
        f"{prefix}_best_generic_standard_payload_bits": float(standard_bits),
        f"{prefix}_public_dict_payload_bits": float(best_public.payload_bits),
        f"{prefix}_public_dict_dictionary_bits": float(best_public.dictionary_bits),
        f"{prefix}_public_dict_header_bits": float(PUBLIC_DICT_HEADER_BITS),
        f"{prefix}_public_dict_selector_bits": float(PUBLIC_DICT_SELECTOR_BITS),
        f"{prefix}_public_dict_charged_bits": float(best_public.charged_bits),
        f"{prefix}_public_dict_undercharged_bits": float(best_public.undercharged_bits),
        f"{prefix}_public_dict_requested_size": float(best_public.requested_size),
        f"{prefix}_public_dict_actual_size_bytes": float(len(best_public.dictionary_bytes)),
        f"{prefix}_public_dict_level": float(best_public.level),
        f"{prefix}_current_subtoken_beats_public_dict_charged": float(int(current_bits < best_public.charged_bits)),
        f"{prefix}_current_subtoken_beats_public_dict_undercharged": float(int(current_bits < best_public.undercharged_bits)),
        f"{prefix}_current_subtoken_margin_over_public_dict_charged_bits": charged_margin,
        f"{prefix}_current_subtoken_margin_over_public_dict_undercharged_bits": undercharged_margin,
        f"{prefix}_current_subtoken_improvement_over_public_dict_charged": improvement(best_public.charged_bits, int(current_bits)),
        f"{prefix}_current_subtoken_improvement_over_public_dict_undercharged": improvement(best_public.undercharged_bits, int(current_bits)),
        f"{prefix}_public_dict_random_label_standard_bits": float(random_standard_bits),
        f"{prefix}_public_dict_random_label_payload_bits": float(best_public_random.payload_bits),
        f"{prefix}_public_dict_random_label_charged_bits": float(best_public_random.charged_bits),
        f"{prefix}_public_dict_random_label_undercharged_bits": float(best_public_random.undercharged_bits),
        f"{prefix}_public_dict_random_label_charged_improvement_over_best_standard": improvement(random_standard_bits, best_public_random.charged_bits),
        f"{prefix}_public_dict_random_label_undercharged_improvement_over_best_standard": improvement(random_standard_bits, best_public_random.undercharged_bits),
        **overlap,
    }


def current_block_payload_bits(profile: str) -> int:
    target_block = read_joined(block_target_paths(profile))
    baseline_train = read_joined([PROJECT_ROOT / path for path in PUBLIC_TRAIN_PATHS[:6] if (PROJECT_ROOT / path).exists()])
    return subtoken_payload_bits(baseline_train, target_block)


def current_corpus_payload_bits(profile: str) -> int:
    baseline_train = read_joined([PROJECT_ROOT / path for path in PUBLIC_TRAIN_PATHS[:6] if (PROJECT_ROOT / path).exists()])
    total = 0
    for row in corpus_blocks(profile):
        block = read_block(row)
        total += subtoken_payload_bits(baseline_train, block) + 16
    return int(total)


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    block_targets = block_target_paths(profile)
    block_train = select_train_paths(block_targets)
    block = read_joined(block_targets)
    block_metrics = measure_surface("block", [block], block_train, block_targets, current_block_payload_bits(profile), int(seed), profile)
    corpus_rows = corpus_blocks(profile)
    corpus_targets = corpus_target_paths(profile)
    corpus_train = select_train_paths(corpus_targets)
    corpus_payloads = [read_block(row) for row in corpus_rows]
    corpus_metrics = measure_surface("corpus", corpus_payloads, corpus_train, corpus_targets, current_corpus_payload_bits(profile), int(seed) + 101, profile)
    block_controls = float(int(block_metrics["block_public_dict_exact_reconstruction_success"] == 1.0 and block_metrics["block_public_dict_disabled_reconstruction_success"] == 0.0 and block_metrics["block_public_dict_shuffled_reconstruction_success"] == 0.0 and block_metrics["block_public_dict_random_label_charged_improvement_over_best_standard"] <= 0.0 and block_metrics["block_public_dict_random_label_undercharged_improvement_over_best_standard"] <= 0.0 and block_metrics["block_public_dict_train_only"] == 1.0))
    corpus_controls = float(int(corpus_metrics["corpus_public_dict_exact_reconstruction_success"] == 1.0 and corpus_metrics["corpus_public_dict_disabled_reconstruction_success"] == 0.0 and corpus_metrics["corpus_public_dict_shuffled_reconstruction_success"] == 0.0 and corpus_metrics["corpus_public_dict_random_label_charged_improvement_over_best_standard"] <= 0.0 and corpus_metrics["corpus_public_dict_random_label_undercharged_improvement_over_best_standard"] <= 0.0 and corpus_metrics["corpus_public_dict_train_only"] == 1.0))
    margin_pass = float(int(block_metrics["block_current_subtoken_margin_over_public_dict_charged_bits"] >= float(PROFILES[profile]["block_charged_margin_bits"]) and block_metrics["block_current_subtoken_margin_over_public_dict_undercharged_bits"] >= float(PROFILES[profile]["block_undercharged_margin_bits"]) and corpus_metrics["corpus_current_subtoken_margin_over_public_dict_charged_bits"] >= float(PROFILES[profile]["corpus_charged_margin_bits"]) and corpus_metrics["corpus_current_subtoken_margin_over_public_dict_undercharged_bits"] >= float(PROFILES[profile]["corpus_undercharged_margin_bits"])))
    engineering_pass = float(int(block_controls == 1.0 and corpus_controls == 1.0 and margin_pass == 1.0 and block_metrics["block_current_subtoken_beats_public_dict_charged"] == 1.0 and block_metrics["block_current_subtoken_beats_public_dict_undercharged"] == 1.0 and corpus_metrics["corpus_current_subtoken_beats_public_dict_charged"] == 1.0 and corpus_metrics["corpus_current_subtoken_beats_public_dict_undercharged"] == 1.0))
    return {
        "profile": profile,
        "parameter_count": 0.0,
        "trainable_parameter_count": 0.0,
        "public_trained_dictionary_baseline_used": 1.0,
        "zstd_trained_dictionary_baseline_audit_candidate": engineering_pass,
        "zstd_trained_dictionary_baseline_audit_authorized": engineering_pass,
        "source_code_public_baseline_audit_authorized": engineering_pass,
        "source_code_codec_product_authorized": 0.0,
        "source_code_codec_breakthrough_authorized": 0.0,
        "strict_breakthrough_authorized": 0.0,
        "general_unknown_structure_breakthrough_authorized": 0.0,
        "static_retrieval_wrapper_authorized": 0.0,
        "broad_nm_authorized": 0.0,
        "broad_chat_authorized": 0.0,
        "broad_knowledge_authorized": 0.0,
        "arbitrary_chat_authorized": 0.0,
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "external_simulator_authorized": 0.0,
        "block_public_dict_controls_collapse": block_controls,
        "corpus_public_dict_controls_collapse": corpus_controls,
        "public_dict_margin_pass": margin_pass,
        "controls_collapse": float(int(block_controls == 1.0 and corpus_controls == 1.0)),
        "engineering_pass": engineering_pass,
        **block_metrics,
        **corpus_metrics,
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
    metrics_path = output_dir / "local_100k_zstd_trained_dictionary_baseline_audit_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name=SIMULATION_ID,
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "seed": int(SEED), "public_dict_header_bits": int(PUBLIC_DICT_HEADER_BITS), "public_dict_selector_bits": int(PUBLIC_DICT_SELECTOR_BITS), "subtoken_structure_header_bits": int(SUBTOKEN_STRUCTURE_HEADER_BITS)},
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_corpus_public_dict_payload_bits"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"name": "local_100k_zstd_trained_dictionary_baseline_audit_metrics.json", "path": metrics_path}],
        warnings=["public trained zstd dictionary audit only; dictionary bytes are charged; no broad compression, nm, chat, knowledge, or static breakthrough authorization"],
    )
    write_json(metrics_path, record)
    print(f"{SIMULATION_ID} profile={profile} engineering_pass={summary[f'{SIMULATION_ID}_engineering_pass']:.3f} block_charged_margin_bits={summary[f'{SIMULATION_ID}_block_current_subtoken_margin_over_public_dict_charged_bits']:.0f} corpus_charged_margin_bits={summary[f'{SIMULATION_ID}_corpus_current_subtoken_margin_over_public_dict_charged_bits']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
