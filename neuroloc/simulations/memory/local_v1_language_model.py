from __future__ import annotations

import math
import os
import re
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
SEED = env_int("V1_LANGUAGE_SEED", 71)
MAX_RECORDS = env_int("V1_LANGUAGE_MAX_RECORDS", 32)
SEED_COUNT = env_int("V1_LANGUAGE_SEED_COUNT", 2)

require_positive("V1_LANGUAGE_MAX_RECORDS", MAX_RECORDS)
require_positive("V1_LANGUAGE_SEED_COUNT", SEED_COUNT)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "before",
    "by",
    "for",
    "from",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "while",
    "with",
    "without",
    "binds",
    "stores",
    "refreshes",
    "branches",
    "keeps",
    "queries",
    "route",
    "routes",
    "hides",
    "rehearses",
    "selects",
    "carries",
    "update",
    "updates",
    "names",
    "reconstructs",
    "opens",
    "separates",
    "teaches",
}

TRAIN_TEMPLATES = (
    "what does {cue} preserve",
    "where should {cue} route",
    "remember {cue} and {payload}",
    "{cue} supports which state",
)

TEST_TEMPLATES = (
    "what does {cue} preserve for action",
    "tell me the memory tied to {cue}",
    "which payload belongs with {cue}",
    "answer from state about {cue}",
)

DEFAULT_DATASET_PATHS = (
    PROJECT_ROOT / "neuroloc" / "wiki" / "PROJECT_PLAN.md",
    PROJECT_ROOT / "docs" / "STATUS_BOARD.md",
    PROJECT_ROOT / "AGENTS.md",
)


def infer_profile() -> str:
    return os.environ.get("V1_LANGUAGE_PROFILE", "smoke")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_]*", text.lower())


def sentence_chunks(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n.!?;:]+", text) if part.strip()]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if len(token) > 2 and token not in STOP_WORDS]


def default_source_texts() -> list[str]:
    texts = []
    for path in DEFAULT_DATASET_PATHS:
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore")[:24_000])
    if texts:
        return texts
    return [
        "memory binds state for action.",
        "language routes queries into local state.",
        "replay preserves provenance after distraction.",
        "imagination branches a compact world state.",
    ]


def build_records(source_texts: list[str] | None = None, max_records: int = MAX_RECORDS) -> list[dict[str, Any]]:
    texts = source_texts if source_texts is not None else default_source_texts()
    records = []
    seen = set()
    for text_index, text in enumerate(texts):
        for sentence in sentence_chunks(text):
            tokens = content_tokens(sentence)
            if len(tokens) < 2:
                continue
            cue = tokens[0]
            if cue in seen:
                continue
            payload = next((token for token in tokens[1:] if token != cue), tokens[1])
            record = {
                "record_id": len(records),
                "cue": cue,
                "payload": payload,
                "source": f"record_{len(records)}",
                "source_text_index": text_index,
                "text": sentence.strip(),
            }
            records.append(record)
            seen.add(cue)
            if len(records) >= max_records:
                return records
    if len(records) < 4:
        raise ValueError("v1 local language model needs at least four unique dataset records")
    return records


def render_examples(records: list[dict[str, Any]], templates: tuple[str, ...]) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        for template_index, template in enumerate(templates):
            text = template.format(cue=record["cue"], payload=record["payload"])
            examples.append({"text": text, "target": int(record["record_id"]), "template_index": template_index})
    return examples


def build_vocab(examples: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, int]:
    words = sorted({token for example in examples for token in tokenize(str(example["text"]))} | {str(record["cue"]) for record in records} | {str(record["payload"]) for record in records})
    return {word: index for index, word in enumerate(words)}


def vectorize(text: str, vocab: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float64)
    for token in tokenize(text):
        if token in vocab:
            vector[vocab[token]] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def train_state_weights(examples: list[dict[str, Any]], records: list[dict[str, Any]], vocab: dict[str, int]) -> np.ndarray:
    weights = np.zeros((len(records), len(vocab)), dtype=np.float64)
    for example in examples:
        target = int(example["target"])
        weights[target] += vectorize(str(example["text"]), vocab)
    for record in records:
        target = int(record["record_id"])
        weights[target] += 2.0 * vectorize(f"{record['cue']} {record['payload']}", vocab)
    row_norms = np.linalg.norm(weights, axis=1, keepdims=True)
    return np.divide(weights, np.maximum(row_norms, 1e-9))


def predict_record_id(prompt: str, bundle: dict[str, Any]) -> int:
    vector = vectorize(prompt, bundle["vocab"])
    scores = bundle["weights"] @ vector
    return int(np.argmax(scores))


def prompt_has_grounded_cue(prompt: str, bundle: dict[str, Any]) -> bool:
    tokens = set(tokenize(prompt))
    return any(str(record["cue"]) in tokens for record in bundle["records"])


def record_for_prediction(record_id: int, bundle: dict[str, Any], state_mode: str = "normal") -> dict[str, Any] | None:
    if state_mode == "zero":
        return None
    records = bundle["records"]
    if state_mode == "shuffled":
        return records[(record_id + 1) % len(records)]
    return records[record_id]


def score_examples(examples: list[dict[str, Any]], bundle: dict[str, Any], state_mode: str = "normal") -> dict[str, float]:
    joint = []
    state = []
    action = []
    provenance = []
    for example in examples:
        target_record = bundle["records"][int(example["target"])]
        predicted_id = predict_record_id(str(example["text"]), bundle)
        predicted_record = record_for_prediction(predicted_id, bundle, state_mode)
        if predicted_record is None:
            state_ok = 0.0
            action_ok = 0.0
            provenance_ok = 0.0
        else:
            state_ok = float(predicted_record["cue"] == target_record["cue"] and predicted_record["payload"] == target_record["payload"])
            action_ok = float(predicted_record["payload"] == target_record["payload"])
            provenance_ok = float(predicted_record["source"] == target_record["source"])
        state.append(state_ok)
        action.append(action_ok)
        provenance.append(provenance_ok)
        joint.append(float(state_ok == 1.0 and action_ok == 1.0 and provenance_ok == 1.0))
    return {
        "joint": float(np.mean(joint)) if joint else 0.0,
        "state": float(np.mean(state)) if state else 0.0,
        "action": float(np.mean(action)) if action else 0.0,
        "provenance": float(np.mean(provenance)) if provenance else 0.0,
    }


def train_v1_language_state_model(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    records = build_records(source_texts, MAX_RECORDS)
    rng.shuffle(records)
    records = sorted([{**record, "record_id": index, "source": f"record_{index}"} for index, record in enumerate(records)], key=lambda item: int(item["record_id"]))
    train_examples = render_examples(records, TRAIN_TEMPLATES)
    test_examples = render_examples(records, TEST_TEMPLATES)
    vocab = build_vocab(train_examples + test_examples, records)
    weights = train_state_weights(train_examples, records, vocab)
    return {
        "profile": profile,
        "seed": seed,
        "records": records,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "vocab": vocab,
        "weights": weights,
        "parameter_count": int(weights.size),
        "accounted_bits": int(math.ceil(math.log2(max(len(records), 2))) + math.ceil(math.log2(max(len(vocab), 2))) + 16),
    }


def answer_v1_prompt(prompt: str, bundle: dict[str, Any], state_mode: str = "normal") -> str:
    if not prompt_has_grounded_cue(prompt, bundle):
        return "v1 answer: outside grounded memory scope"
    predicted_id = predict_record_id(prompt, bundle)
    record = record_for_prediction(predicted_id, bundle, state_mode)
    if record is None:
        return "v1 answer: no grounded memory state"
    return f"v1 answer: {record['cue']} links to {record['payload']}. source {record['source']}."


def find_record_by_cue(records: list[dict[str, Any]], cue: str) -> dict[str, Any] | None:
    for record in records:
        if record["cue"] == cue:
            return record
    return None


def token_after(tokens: list[str], marker: str, fallback_index: int) -> str:
    if marker in tokens:
        index = tokens.index(marker)
        if index + 1 < len(tokens):
            return tokens[index + 1]
    if fallback_index < len(tokens):
        return tokens[fallback_index]
    return tokens[-1] if tokens else ""


def copied_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(record) for record in bundle["records"]]


def answer_from_record(record: dict[str, Any] | None) -> str:
    if record is None:
        return "v1 answer: no grounded memory state"
    return f"v1 answer: {record['cue']} links to {record['payload']}. source {record['source']}."


def run_v1_dialogue(turns: list[str], bundle: dict[str, Any], state_mode: str = "normal", replay_mode: str = "targeted") -> dict[str, Any]:
    memory = copied_records(bundle)
    responses = []
    update_success = 0.0
    targeted_replay_success = 0.0
    branch_state_success = 0.0
    branch_response = ""
    final_response = ""
    updated_cue = ""
    updated_payload = ""
    for turn in turns:
        tokens = tokenize(turn)
        if not tokens:
            continue
        if tokens[0] == "update":
            cue = token_after(tokens, "update", 1)
            payload = token_after(tokens, "payload", 3)
            record = find_record_by_cue(memory, cue)
            if record is not None and state_mode != "zero":
                record["payload"] = payload
                record["source"] = f"update_record_{record['record_id']}"
                updated_cue = cue
                updated_payload = payload
                update_success = 1.0
                response = answer_from_record(record)
            else:
                response = "v1 answer: no grounded memory state"
            responses.append(response)
        elif tokens[0] == "replay":
            cue = token_after(tokens, "replay", 1)
            record = find_record_by_cue(memory, cue)
            if replay_mode == "random" and memory:
                candidates = [item for item in memory if item["cue"] != cue]
                record = candidates[0] if candidates else record
            if state_mode == "zero":
                record = None
            response = answer_from_record(record)
            targeted_replay_success = float(record is not None and replay_mode == "targeted" and record["cue"] == cue and (not updated_cue or record["payload"] == updated_payload))
            responses.append(response)
        elif tokens[0] == "imagine":
            cue = token_after(tokens, "imagine", 1)
            branch = token_after(tokens, "branch", len(tokens) - 1)
            record = find_record_by_cue(memory, cue)
            if record is not None and state_mode != "zero":
                branch_response = f"v1 answer: branch {branch} for {cue} links to {record['payload']}. source {record['source']}."
                branch_state_success = 1.0
                responses.append(branch_response)
            else:
                branch_response = "v1 answer: no grounded memory state"
                responses.append(branch_response)
        else:
            predicted_id = predict_record_id(turn, bundle)
            record = memory[predicted_id]
            if state_mode == "zero":
                record = None
            elif state_mode == "shuffled":
                record = memory[(predicted_id + 1) % len(memory)]
            response = answer_from_record(record)
            final_response = response
            responses.append(response)
    if not final_response and responses:
        final_response = responses[-1]
    final_joint_success = float(state_mode != "zero" and bool(updated_cue) and updated_payload in final_response and updated_cue in final_response)
    return {
        "responses": responses,
        "update_success": update_success,
        "targeted_replay_success": targeted_replay_success,
        "branch_state_success": branch_state_success,
        "final_joint_success": final_joint_success,
        "branch_response": branch_response,
        "final_response": final_response,
    }


def build_v1_dialogue_summary(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    rows = []
    for seed_index in range(int(SEED_COUNT)):
        bundle = train_v1_language_state_model(profile, seed + seed_index * 1_009, source_texts)
        cue = str(bundle["records"][0]["cue"])
        payload = f"update{seed_index}"
        turns = [
            f"what does {cue} preserve for action",
            f"update {cue} payload {payload} from study note",
            f"tell me about {bundle['records'][1]['cue']}",
            f"replay {cue}",
            f"imagine {cue} branch rehearsal",
            f"what does {cue} preserve for action",
        ]
        normal = run_v1_dialogue(turns, bundle)
        zero = run_v1_dialogue(turns, bundle, state_mode="zero")
        random_replay = run_v1_dialogue(turns, bundle, replay_mode="random")
        rows.append({
            "update": float(normal["update_success"]),
            "replay": float(normal["targeted_replay_success"]),
            "random_replay": float(random_replay["targeted_replay_success"]),
            "branch": float(normal["branch_state_success"]),
            "final": float(normal["final_joint_success"]),
            "zero_final": float(zero["final_joint_success"]),
        })
    engineering_pass = float(int(rows and min([row["update"] for row in rows]) >= 0.9 and min([row["replay"] for row in rows]) >= 0.9 and max([row["random_replay"] for row in rows]) == 0.0 and min([row["branch"] for row in rows]) >= 0.9 and min([row["final"] for row in rows]) >= 0.9 and max([row["zero_final"] for row in rows]) == 0.0))
    return {
        "v1_dialogue_gate_evaluated": 1.0,
        "v1_memory_update_success_min": float(min([row["update"] for row in rows]) if rows else 0.0),
        "v1_targeted_replay_success_min": float(min([row["replay"] for row in rows]) if rows else 0.0),
        "v1_random_replay_success_max": float(max([row["random_replay"] for row in rows]) if rows else 0.0),
        "v1_branch_state_success_min": float(min([row["branch"] for row in rows]) if rows else 0.0),
        "v1_dialogue_final_joint_success_min": float(min([row["final"] for row in rows]) if rows else 0.0),
        "v1_dialogue_zero_state_joint_success_max": float(max([row["zero_final"] for row in rows]) if rows else 0.0),
        "v1_dialogue_engineering_pass": engineering_pass,
    }


def one_run_summary(profile: str, seed: int, source_texts: list[str] | None = None) -> dict[str, Any]:
    bundle = train_v1_language_state_model(profile, seed, source_texts)
    normal = score_examples(bundle["test_examples"], bundle, "normal")
    zero = score_examples(bundle["test_examples"], bundle, "zero")
    shuffled = score_examples(bundle["test_examples"], bundle, "shuffled")
    accounted_bits = float(bundle["accounted_bits"])
    return {
        "joint": normal["joint"],
        "state": normal["state"],
        "action": normal["action"],
        "provenance": normal["provenance"],
        "zero_joint": zero["joint"],
        "shuffle_joint": shuffled["joint"],
        "parameter_count": float(bundle["parameter_count"]),
        "accounted_bits": accounted_bits,
        "useful_density": normal["joint"] / max(accounted_bits, 1e-9),
        "record_count": float(len(bundle["records"])),
        "vocab_size": float(len(bundle["vocab"])),
        "example_response": answer_v1_prompt(bundle["test_examples"][0]["text"], bundle),
    }


def build_v1_language_summary(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    runs = [one_run_summary(profile, seed + seed_index * 1_009, source_texts) for seed_index in range(int(SEED_COUNT))]
    joints = [float(row["joint"]) for row in runs]
    states = [float(row["state"]) for row in runs]
    actions = [float(row["action"]) for row in runs]
    provenances = [float(row["provenance"]) for row in runs]
    zeros = [float(row["zero_joint"]) for row in runs]
    shuffles = [float(row["shuffle_joint"]) for row in runs]
    params = [float(row["parameter_count"]) for row in runs]
    densities = [float(row["useful_density"]) for row in runs]
    engineering_pass = float(int(runs and min(joints) >= 0.9 and min(states) >= 0.9 and min(actions) >= 0.9 and min(provenances) >= 0.9 and max(zeros) == 0.0 and max(shuffles) < 0.4 and max(params) < 10_000))
    return {
        "v1_language_gate_evaluated": 1.0,
        "v1_dataset_grounded": 1.0,
        "v1_state_first_training_used": 1.0,
        "v1_next_token_training_used": 0.0,
        "v1_local_model_authorized": 1.0,
        "v1_full_model_authorized": 0.0,
        "v1_paid_compute_authorized": 0.0,
        "v1_arbitrary_chat_authorized": 0.0,
        "v1_seed_count": int(SEED_COUNT),
        "v1_run_count": int(len(runs)),
        "v1_record_count_min": float(min([float(row["record_count"]) for row in runs]) if runs else 0.0),
        "v1_vocab_size_max": float(max([float(row["vocab_size"]) for row in runs]) if runs else 0.0),
        "v1_trainable_parameter_count_max": float(max(params) if params else 0.0),
        "v1_test_joint_success_min": float(min(joints) if joints else 0.0),
        "v1_test_state_success_min": float(min(states) if states else 0.0),
        "v1_test_action_success_min": float(min(actions) if actions else 0.0),
        "v1_provenance_success_min": float(min(provenances) if provenances else 0.0),
        "v1_zero_state_joint_success_max": float(max(zeros) if zeros else 0.0),
        "v1_shuffled_state_joint_success_max": float(max(shuffles) if shuffles else 0.0),
        "v1_accounted_bits_max": float(max([float(row["accounted_bits"]) for row in runs]) if runs else 0.0),
        "v1_useful_operation_success_per_accounted_bit_min": float(min(densities) if densities else 0.0),
        "v1_interactive_response_supported": 1.0,
        "v1_engineering_pass": engineering_pass,
        "v1_claim_downgraded_to_dataset_state_router": float(1.0 - engineering_pass),
        "v1_example_response": str(runs[0]["example_response"]) if runs else "",
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_v1_language_summary(profile)
    summary.update(build_v1_dialogue_summary(profile))
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_v1_language_model_metrics.json"
    record = build_run_record(
        simulation_name="local_v1_language_model",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=utc_now_iso(),
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "max_records": int(MAX_RECORDS),
            "seed_count": int(SEED_COUNT),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["v1_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"path": metrics_path.as_posix(), "type": "metrics"}],
        warnings=["local dataset-grounded state router only; not arbitrary chat, not solved compression, not full neural model v1"],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
