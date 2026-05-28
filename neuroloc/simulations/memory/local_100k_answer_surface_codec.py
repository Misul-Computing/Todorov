from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import os
import random
import re
import sys
import time
import zlib
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
SIMULATION_ID = "local_100k_answer_surface_codec"
SEED = env_int("ANSWER_SURFACE_CODEC_SEED", 3221)
FACTS_SMOKE = env_int("ANSWER_SURFACE_CODEC_FACTS_SMOKE", 1024)
FACTS_HARD = env_int("ANSWER_SURFACE_CODEC_FACTS_HARD", 4096)
ANSWER_BYTES = env_int("ANSWER_SURFACE_CODEC_ANSWER_BYTES", 96)
WINDOW_STEP_BYTES = env_int("ANSWER_SURFACE_CODEC_WINDOW_STEP_BYTES", 31)
DECODER_BITS = env_int("ANSWER_SURFACE_CODEC_DECODER_BITS", 32768)
MODEL_HEADER_BITS = env_int("ANSWER_SURFACE_CODEC_MODEL_HEADER_BITS", 64)
SURFACE_CONTRACT_BITS = env_int("ANSWER_SURFACE_CODEC_SURFACE_CONTRACT_BITS", 4096)
ADAPTER_CONTROL_BITS = env_int("ANSWER_SURFACE_CODEC_ADAPTER_CONTROL_BITS", 1024)
ORDINARY_BITS_PER_PARAMETER = float(os.environ.get("ANSWER_SURFACE_CODEC_ORDINARY_BITS_PER_PARAMETER", "2.5"))

require_positive("ANSWER_SURFACE_CODEC_FACTS_SMOKE", FACTS_SMOKE)
require_positive("ANSWER_SURFACE_CODEC_FACTS_HARD", FACTS_HARD)
require_positive("ANSWER_SURFACE_CODEC_ANSWER_BYTES", ANSWER_BYTES)
require_positive("ANSWER_SURFACE_CODEC_WINDOW_STEP_BYTES", WINDOW_STEP_BYTES)
require_positive("ANSWER_SURFACE_CODEC_DECODER_BITS", DECODER_BITS)
require_positive("ANSWER_SURFACE_CODEC_MODEL_HEADER_BITS", MODEL_HEADER_BITS)
require_positive("ANSWER_SURFACE_CODEC_SURFACE_CONTRACT_BITS", SURFACE_CONTRACT_BITS)

QUESTION_PREFIX = "answer authored surface key"
CODEC_IDS = {"zlib9": 1, "bz2": 2, "lzma6": 3}
CODECS_BY_ID = {value: key for key, value in CODEC_IDS.items()}

PROFILES = {
    "smoke": {"fact_count": FACTS_SMOKE},
    "hard": {"fact_count": FACTS_HARD},
}


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("ANSWER_SURFACE_CODEC_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("ANSWER_SURFACE_CODEC_PROFILE must be smoke or hard")
    return value


def normalize_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    value = value.replace("\r\n", "\n")
    value = re.sub(r"`+", "", value)
    value = re.sub(r"\s+", " ", value)
    value = "".join(character if 32 <= ord(character) <= 126 else " " for character in value)
    return re.sub(r"\s+", " ", value).strip().lower()


def source_rows() -> list[tuple[Path, str]]:
    paths = [
        PROJECT_ROOT / "knowledge/context_extension.md",
        PROJECT_ROOT / "knowledge/delta_rule_theory.md",
        PROJECT_ROOT / "knowledge/geometric_algebra.md",
        PROJECT_ROOT / "knowledge/hybrid_architectures.md",
        PROJECT_ROOT / "knowledge/kda_channel_gating.md",
        PROJECT_ROOT / "knowledge/mamba3_architecture.md",
        PROJECT_ROOT / "knowledge/mla_compression.md",
        PROJECT_ROOT / "knowledge/training_efficiency.md",
        PROJECT_ROOT / "knowledge/unified_theory.md",
        PROJECT_ROOT / "neuroloc/wiki/PROJECT_PLAN.md",
        PROJECT_ROOT / "neuroloc/wiki/synthesis/content_routed_sparse_read_prior.md",
        PROJECT_ROOT / "neuroloc/wiki/synthesis/neural_model_compression_stack.md",
        PROJECT_ROOT / "neuroloc/wiki/synthesis/neural_model_dossier_compression.md",
        PROJECT_ROOT / "neuroloc/wiki/synthesis/neural_model_related_work_pressure_matrix.md",
        PROJECT_ROOT / "src/layers/kda.py",
        PROJECT_ROOT / "src/layers/mamba3.py",
        PROJECT_ROOT / "src/layers/mla.py",
    ]
    return [(path, path.relative_to(PROJECT_ROOT).as_posix()) for path in paths if path.exists()]


def authored_windows() -> list[dict[str, Any]]:
    rows = []
    for path, label in source_rows():
        data = normalize_text(path.read_bytes())
        encoded = data.encode("utf-8")
        if len(encoded) < int(ANSWER_BYTES):
            continue
        for offset in range(0, len(encoded) - int(ANSWER_BYTES) + 1, int(WINDOW_STEP_BYTES)):
            raw = encoded[offset : offset + int(ANSWER_BYTES)]
            answer = normalize_text(raw)
            if len(answer) < 40:
                continue
            digest = hashlib.blake2b(f"{label}:{offset}:".encode("utf-8") + answer.encode("utf-8"), digest_size=8, person=b"ansurf1").hexdigest()
            rows.append({"source_path": label, "source_offset": int(offset), "answer": answer, "key": digest})
    return rows


def provenance_for(source_path: str, answer: str) -> str:
    return hashlib.sha256(f"{source_path}:".encode("utf-8") + answer.encode("utf-8")).hexdigest()[:20]


def question_for_key(key: str) -> str:
    return f"{QUESTION_PREFIX} {key}"


def build_facts(seed: int, fact_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = authored_windows()
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    seen_keys = set()
    seen_answers = set()
    facts = []
    for row in rows:
        key = str(row["key"])
        answer = str(row["answer"])
        answer_digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if key in seen_keys or answer_digest in seen_answers:
            continue
        seen_keys.add(key)
        seen_answers.add(answer_digest)
        facts.append(
            {
                "role": "test",
                "row": int(len(facts)),
                "question": question_for_key(key),
                "key": key,
                "value": answer,
                "provenance": provenance_for(str(row["source_path"]), answer),
                "source_path": str(row["source_path"]),
            }
        )
        if len(facts) == int(fact_count):
            return [], sorted(facts, key=lambda item: str(item["key"]))
    raise ValueError("not enough authored answer windows")


def build_random_twin(seed: int, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 8101)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    twin = []
    for fact in facts:
        value = "".join(rng.choice(alphabet) for _index in range(max(16, len(str(fact["value"])))))
        twin.append(
            {
                "role": "test",
                "row": int(fact["row"]),
                "question": str(fact["question"]),
                "key": str(fact["key"]),
                "value": value,
                "provenance": hashlib.sha256(value.encode("utf-8")).hexdigest()[:20],
                "source_path": str(fact["source_path"]),
            }
        )
    return twin


def key_from_question(question: str) -> str:
    normalized = normalize_text(question)
    marker = f"{QUESTION_PREFIX} "
    if not normalized.startswith(marker):
        return ""
    key = normalized.split(marker, 1)[1].strip().split()[0]
    return key if re.fullmatch(r"[0-9a-f]{16}", key) else ""


def pack_payload(facts: list[dict[str, Any]]) -> bytes:
    rows = [
        {
            "k": str(fact["key"]),
            "q": str(fact["question"]),
            "a": str(fact["value"]),
            "p": str(fact["provenance"]),
        }
        for fact in facts
    ]
    payload = {"surface": "authored_answer_only", "version": 1, "rows": rows}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def unpack_payload(payload: bytes) -> list[dict[str, str]]:
    raw = json.loads(payload.decode("utf-8"))
    if raw.get("surface") != "authored_answer_only":
        raise ValueError("wrong payload surface")
    rows = raw.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("payload rows must be a list")
    return [{"k": str(row["k"]), "q": str(row["q"]), "a": str(row["a"]), "p": str(row["p"])} for row in rows]


def compress_payload(payload: bytes) -> bytes:
    candidates = [
        ("zlib9", zlib.compress(payload, level=9)),
        ("bz2", bz2.compress(payload, compresslevel=9)),
        ("lzma6", lzma.compress(payload, preset=6)),
    ]
    codec_name, stream = min(candidates, key=lambda item: (len(item[1]), item[0]))
    return bytes([int(CODEC_IDS[codec_name])]) + stream


def decompress_payload(stream: bytes) -> bytes:
    if not stream:
        raise ValueError("empty payload stream")
    codec_name = CODECS_BY_ID.get(int(stream[0]))
    if codec_name == "zlib9":
        return zlib.decompress(stream[1:])
    if codec_name == "bz2":
        return bz2.decompress(stream[1:])
    if codec_name == "lzma6":
        return lzma.decompress(stream[1:])
    raise ValueError("unknown payload codec")


def score_answers(facts: list[dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for fact, answer in zip(facts, answers):
        value_ok = str(answer.get("value", "")) == str(fact["value"])
        provenance_ok = str(answer.get("provenance", "")) == str(fact["provenance"])
        hit_ok = int(answer.get("hit", 0)) == 1
        rows.append({"value_success": float(value_ok), "provenance_success": float(provenance_ok), "hit_success": float(hit_ok), "exact_success": float(value_ok and provenance_ok and hit_ok)})
    return rows


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / float(len(rows)))


def shifted(items: list[Any]) -> list[Any]:
    if len(items) <= 1:
        return list(items)
    return list(items[1:] + items[:1])


class AnswerSurfaceCodecCell:
    def __init__(self, train_facts: list[dict[str, Any]], test_facts: list[dict[str, Any]]) -> None:
        self.payload_stream = compress_payload(pack_payload(test_facts))
        self.single_charged_payload_stream_used = 1.0
        self.external_payload_store_used = 0.0
        self.hidden_dict_used = 0.0
        self.raw_decoded_cache_retained = 0.0
        self.raw_source_block_retained = 0.0
        self.reads_from_charged_payload_stream = 1.0
        self.per_fact_rows_in_payload = 1.0
        self.table_diagnostic = 1.0
        self.answer_only_surface = 1.0
        self.train_fact_count = len(train_facts)
        self.test_fact_count = len(test_facts)
        self.decode_count = 0
        self.transient_map_build_count = 0

    def parameter_count(self) -> int:
        return 0

    def decoded_rows(self) -> list[dict[str, str]]:
        self.decode_count += 1
        return unpack_payload(decompress_payload(self.payload_stream))

    def answer_many(
        self,
        questions: list[str],
        read_enabled: bool = True,
        decoder_enabled: bool = True,
        parser_enabled: bool = True,
        adapter_enabled: bool = True,
        read_disabled: bool = False,
        decoder_disabled: bool = False,
        parser_disabled: bool = False,
        adapter_disabled: bool = False,
        code_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        if not read_enabled:
            read_disabled = True
        if not decoder_enabled:
            decoder_disabled = True
        if not parser_enabled:
            parser_disabled = True
        if not adapter_enabled:
            adapter_disabled = True
        if read_disabled or decoder_disabled or parser_disabled or adapter_disabled or code_disabled:
            return [{"value": "", "provenance": "", "hit": 0} for _question in questions]
        rows = self.decoded_rows()
        self.transient_map_build_count += 1
        by_key = {str(row["k"]): row for row in rows}
        answers = []
        for question in questions:
            key = key_from_question(str(question))
            row = by_key.get(key)
            if row is None:
                answers.append({"value": "", "provenance": "", "hit": 0})
            else:
                answers.append({"value": str(row["a"]), "provenance": str(row["p"]), "hit": 1})
        return answers

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.answer_many([str(question)], **kwargs)[0]


def same_interface_answer_surface_scan(cell: AnswerSurfaceCodecCell, questions: list[str]) -> list[dict[str, Any]]:
    rows = unpack_payload(decompress_payload(cell.payload_stream))
    answers = []
    for question in questions:
        key = key_from_question(str(question))
        matched = None
        for row in rows:
            if str(row["k"]) == key:
                matched = row
                break
        if matched is None:
            answers.append({"value": "", "provenance": "", "hit": 0})
        else:
            answers.append({"value": str(matched["a"]), "provenance": str(matched["p"]), "hit": 1})
    return answers


def evaluate_controls(cell: AnswerSurfaceCodecCell, facts: list[dict[str, Any]], random_twin: list[dict[str, Any]]) -> dict[str, float]:
    questions = [str(fact["question"]) for fact in facts]
    exact = cell.answer_many(questions)
    twin = cell.answer_many([str(fact["question"]) for fact in random_twin])
    no_memory = [{"value": "", "provenance": "", "hit": 0} for _fact in facts]
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _fact in facts]
    shifted_question = cell.answer_many([str(fact["question"]) for fact in shifted(facts)])
    wrong_question = cell.answer_many([f"{QUESTION_PREFIX} deadbeefdeadbeef" for _fact in facts])
    exact_rows = cell.answer_many(questions)
    shifted_values = [{"value": row["value"], "provenance": answer["provenance"], "hit": answer["hit"]} for row, answer in zip(shifted(exact_rows), exact_rows)]
    shifted_provenance = [{"value": answer["value"], "provenance": row["provenance"], "hit": answer["hit"]} for row, answer in zip(facts[1:] + facts[:1], exact_rows)]
    read_disabled = cell.answer_many(questions, read_disabled=True)
    decoder_disabled = cell.answer_many(questions, decoder_disabled=True)
    parser_disabled = cell.answer_many(questions, parser_disabled=True)
    adapter_disabled = cell.answer_many(questions, adapter_disabled=True)
    code_disabled = cell.answer_many(questions, code_disabled=True)
    same_scan = same_interface_answer_surface_scan(cell, questions)
    return {
        "exact_success": mean_metric(score_answers(facts, exact), "exact_success"),
        "random_label_twin_success": mean_metric(score_answers(random_twin, twin), "exact_success"),
        "no_memory_success": mean_metric(score_answers(facts, no_memory), "exact_success"),
        "recency_only_success": mean_metric(score_answers(facts, recency), "exact_success"),
        "shuffled_question_success": mean_metric(score_answers(facts, shifted_question), "exact_success"),
        "shuffled_value_success": mean_metric(score_answers(facts, shifted_values), "exact_success"),
        "shuffled_provenance_success": mean_metric(score_answers(facts, shifted_provenance), "exact_success"),
        "wrong_question_success": mean_metric(score_answers(facts, wrong_question), "exact_success"),
        "read_disabled_success": mean_metric(score_answers(facts, read_disabled), "exact_success"),
        "decoder_disabled_success": mean_metric(score_answers(facts, decoder_disabled), "exact_success"),
        "parser_disabled_success": mean_metric(score_answers(facts, parser_disabled), "exact_success"),
        "adapter_disabled_success": mean_metric(score_answers(facts, adapter_disabled), "exact_success"),
        "code_disabled_success": mean_metric(score_answers(facts, code_disabled), "exact_success"),
        "same_interface_answer_surface_scan_success": mean_metric(score_answers(facts, same_scan), "exact_success"),
    }


def accounting(cell: AnswerSurfaceCodecCell, facts: list[dict[str, Any]]) -> dict[str, float]:
    useful_bits = int(sum(len(str(fact["value"]).encode("utf-8")) * 8 for fact in facts))
    payload_bits = int(len(cell.payload_stream) * 8)
    committed_bits = int(payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS) + int(ADAPTER_CONTROL_BITS))
    paper_bits = int(committed_bits + int(SURFACE_CONTRACT_BITS))
    same_interface_scan_bits = int(payload_bits + int(DECODER_BITS) + int(MODEL_HEADER_BITS))
    undercharged_mph_bits = int(payload_bits + int(DECODER_BITS) + 16)

    def multiplier(bits: int, params: int = 0) -> float:
        return float(useful_bits) / max(float(params) + float(bits) / 16.0, 1.0) / max(float(ORDINARY_BITS_PER_PARAMETER), 1e-9)

    return {
        "payload_stream_bits": float(payload_bits),
        "committed_bits": float(committed_bits),
        "paper_bits": float(paper_bits),
        "paper_surface_accounted_bits": float(paper_bits),
        "useful_answer_bits": float(useful_bits),
        "adapter_multiplier": multiplier(committed_bits, cell.parameter_count()),
        "paper_multiplier": multiplier(paper_bits, cell.parameter_count()),
        "same_interface_answer_surface_scan_bits": float(same_interface_scan_bits),
        "same_interface_answer_surface_scan_multiplier": multiplier(same_interface_scan_bits),
        "undercharged_mph_bits": float(undercharged_mph_bits),
        "undercharged_mph_multiplier": multiplier(undercharged_mph_bits),
        "parameter_count": float(cell.parameter_count()),
    }


def run_profile(profile: str, seed: int = SEED) -> dict[str, Any]:
    fact_count = int(PROFILES[profile]["fact_count"])
    train_facts, facts = build_facts(seed, fact_count)
    random_twin = build_random_twin(seed, facts)
    cell = AnswerSurfaceCodecCell(train_facts, facts)
    controls = evaluate_controls(cell, facts, random_twin)
    account = accounting(cell, facts)
    controls_collapse = float(
        int(
            controls["random_label_twin_success"] == 0.0
            and controls["no_memory_success"] == 0.0
            and controls["read_disabled_success"] == 0.0
            and controls["decoder_disabled_success"] == 0.0
            and controls["parser_disabled_success"] == 0.0
            and controls["adapter_disabled_success"] == 0.0
            and controls["code_disabled_success"] == 0.0
            and controls["wrong_question_success"] == 0.0
            and controls["shuffled_question_success"] <= 0.01
            and controls["shuffled_value_success"] <= 0.01
            and controls["shuffled_provenance_success"] <= 0.01
            and controls["recency_only_success"] <= 0.01
        )
    )
    same_interface_not_beaten = float(int(account["adapter_multiplier"] <= account["same_interface_answer_surface_scan_multiplier"]))
    undercharged_mph_not_beaten = float(int(account["adapter_multiplier"] <= account["undercharged_mph_multiplier"]))
    engineering_pass = float(
        int(
            controls["exact_success"] == 1.0
            and controls_collapse == 1.0
            and controls["same_interface_answer_surface_scan_success"] == 1.0
            and cell.single_charged_payload_stream_used == 1.0
            and cell.external_payload_store_used == 0.0
            and cell.raw_decoded_cache_retained == 0.0
            and same_interface_not_beaten == 1.0
        )
    )
    return {
        "profile": profile,
        "fact_count": float(len(facts)),
        "source_file_count": float(len({str(fact["source_path"]) for fact in facts})),
        "engineering_pass": engineering_pass,
        "exact_success": float(controls["exact_success"]),
        "controls_collapse": controls_collapse,
        "random_label_twin_collapse": float(int(controls["random_label_twin_success"] == 0.0)),
        "publishable_auth": 0.0,
        "publishable_candidate": 0.0,
        "same_interface_answer_surface_scan_not_beaten": same_interface_not_beaten,
        "undercharged_mph_not_beaten": undercharged_mph_not_beaten,
        "single_charged_payload_stream_used": float(cell.single_charged_payload_stream_used),
        "external_payload_store_used": float(cell.external_payload_store_used),
        "hidden_dict_used": float(cell.hidden_dict_used),
        "raw_decoded_cache_retained": float(cell.raw_decoded_cache_retained),
        "raw_source_block_retained": float(cell.raw_source_block_retained),
        "reads_from_charged_payload_stream": float(cell.reads_from_charged_payload_stream),
        "per_fact_rows_in_payload": float(cell.per_fact_rows_in_payload),
        "table_diagnostic": float(cell.table_diagnostic),
        "answer_only_surface": float(cell.answer_only_surface),
        "full_nm_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "breakthrough_authorized": 0.0,
        **controls,
        **account,
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    row = run_profile(profile, seed)
    summary: dict[str, Any] = {
        f"{SIMULATION_ID}_evaluated": 1.0,
        f"{SIMULATION_ID}_publishable_auth": 0.0,
        f"{SIMULATION_ID}_breakthrough_authorized": 0.0,
        f"{SIMULATION_ID}_full_nm_authorized": 0.0,
        f"{SIMULATION_ID}_paid_compute_authorized": 0.0,
    }
    for key, value in row.items():
        if key == "profile":
            continue
        summary[f"{SIMULATION_ID}_{key}"] = float(value)
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    metrics_path = output_dir / "local_100k_answer_surface_codec_metrics.json"
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
            "facts_smoke": int(FACTS_SMOKE),
            "facts_hard": int(FACTS_HARD),
            "answer_bytes": int(ANSWER_BYTES),
            "window_step_bytes": int(WINDOW_STEP_BYTES),
            "decoder_bits": int(DECODER_BITS),
            "model_header_bits": int(MODEL_HEADER_BITS),
            "surface_contract_bits": int(SURFACE_CONTRACT_BITS),
            "adapter_control_bits": int(ADAPTER_CONTROL_BITS),
            "ordinary_bits_per_parameter": float(ORDINARY_BITS_PER_PARAMETER),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary[f"{SIMULATION_ID}_fact_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=["exploratory authored answer surface; same-interface scan succeeds; no breakthrough claim"],
        artifacts=[{"name": "local_100k_answer_surface_codec_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
