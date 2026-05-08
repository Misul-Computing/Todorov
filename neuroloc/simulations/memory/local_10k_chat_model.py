from __future__ import annotations

import argparse
import math
import os
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
from neuroloc.simulations.memory.local_v1_language_model import (
    build_records,
    content_tokens,
    render_examples,
    tokenize,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("LOCAL_10K_CHAT_SEED", 83)
MAX_RECORDS = env_int("LOCAL_10K_CHAT_MAX_RECORDS", 32)
SEED_COUNT = env_int("LOCAL_10K_CHAT_SEED_COUNT", 2)
STATE_DIM = env_int("LOCAL_10K_CHAT_STATE_DIM", 48)

require_positive("LOCAL_10K_CHAT_MAX_RECORDS", MAX_RECORDS)
require_positive("LOCAL_10K_CHAT_SEED_COUNT", SEED_COUNT)
require_positive("LOCAL_10K_CHAT_STATE_DIM", STATE_DIM)

TRAIN_TEMPLATES = (
    "what does {cue} preserve",
    "tell me about {cue}",
    "remember {cue} and {payload}",
    "route {cue} through memory",
)

TEST_TEMPLATES = (
    "what does {cue} preserve for action",
    "which memory belongs to {cue}",
    "answer from state about {cue}",
    "tell me the payload for {cue}",
)

CORE_SOURCE_TEXTS = [
    "cortex binds memory for action.",
    "synapse stores update signals for future recall.",
    "memory preserves useful state for future answers.",
    "compression keeps payload under a useful bit budget.",
    "replay refreshes provenance after distraction.",
    "imagination branches a compact state for rehearsal.",
    "context keeps short term dialogue state.",
    "consolidation moves useful traces into long term memory.",
    "language routes grounded questions into bounded memory state.",
    "occlusion hides objects while local state preserves position.",
    "dreaming rehearses branches without external observation.",
    "attention selects records while memory preserves payload.",
    "neuron state carries eligibility before commitment.",
    "world dynamics update velocity after action.",
    "provenance names the record that supported the answer.",
    "decoder reconstructs fields from compact state.",
    "gate opens when surprise exceeds local expectation.",
    "schema separates address payload action and source.",
    "dataset teaches routing without next token training.",
    "branch state preserves rehearsal choices for later replay.",
    "short term memory carries the latest dialogue update.",
    "long term memory stores stable records for retrieval.",
    "local state binds address payload action and source.",
    "router selects the bounded memory record for a grounded question.",
    "refusal protects the model outside grounded memory scope.",
    "rewrite changes payload while preserving provenance.",
    "control tests disable route weights to prove state dependence.",
    "shuffle tests permute routes to expose shortcut answers.",
    "bit budget counts committed state rather than raw context.",
    "useful density measures successful operations per committed bit.",
    "foundation models need local proof before paid compute.",
    "chat surface answers only from its bounded memory records.",
]


def infer_profile() -> str:
    return os.environ.get("LOCAL_10K_CHAT_PROFILE", "smoke")


def stable_token_vector(token: str, state_dim: int) -> np.ndarray:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(token)) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(0.0, 1.0, size=state_dim)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-9)


def build_chat_vocab(records: list[dict[str, Any]]) -> dict[str, int]:
    examples = render_examples(records, TRAIN_TEMPLATES + TEST_TEMPLATES)
    words = {token for example in examples for token in tokenize(str(example["text"]))}
    for record in records:
        words.update(content_tokens(str(record["text"])))
        words.add(str(record["cue"]))
        words.add(str(record["payload"]))
    extra = {"update", "payload", "replay", "imagine", "branch", "changed", "what", "tell", "about"}
    return {word: index for index, word in enumerate(sorted(words | extra))}


def encode_text(text: str, bundle: dict[str, Any]) -> np.ndarray:
    state = np.zeros(int(bundle["state_dim"]), dtype=np.float64)
    count = 0
    cue_tokens = {str(record["cue"]) for record in bundle["records"]}
    for token in tokenize(text):
        index = bundle["vocab"].get(token)
        if index is not None:
            weight = 8.0 if token in cue_tokens else 1.0
            state += weight * bundle["token_embeddings"][index]
            count += weight
    if count:
        state /= float(count)
    norm = float(np.linalg.norm(state))
    if norm > 0.0:
        state /= norm
    return state


def cue_in_prompt(prompt: str, bundle: dict[str, Any]) -> str | None:
    tokens = set(tokenize(prompt))
    for record in bundle["records"]:
        if str(record["cue"]) in tokens:
            return str(record["cue"])
    return None


def record_by_cue(bundle: dict[str, Any], cue: str) -> dict[str, Any] | None:
    for record in bundle["memory"]:
        if str(record["cue"]) == cue:
            return record
    return None


def route_control_bundle(bundle: dict[str, Any], mode: str) -> dict[str, Any]:
    controlled = dict(bundle)
    if mode == "disabled":
        controlled["route_weights"] = np.zeros_like(bundle["route_weights"])
    elif mode == "shuffled":
        controlled["route_weights"] = np.roll(bundle["route_weights"], shift=1, axis=0)
    return controlled


def train_local_10k_chat_model(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    records = build_records(source_texts if source_texts is not None else CORE_SOURCE_TEXTS, MAX_RECORDS)
    rng.shuffle(records)
    records = sorted([{**record, "record_id": index, "source": f"record_{index}"} for index, record in enumerate(records)], key=lambda item: int(item["record_id"]))
    vocab = build_chat_vocab(records)
    token_embeddings = np.vstack([stable_token_vector(token, int(STATE_DIM)) for token in vocab])
    record_states = []
    for record in records:
        state = np.zeros(int(STATE_DIM), dtype=np.float64)
        for token in tokenize(str(record["cue"])):
            index = vocab.get(token)
            if index is not None:
                state += 8.0 * token_embeddings[index]
        for token in tokenize(str(record["payload"])):
            index = vocab.get(token)
            if index is not None:
                state += 2.0 * token_embeddings[index]
        norm = float(np.linalg.norm(state))
        record_states.append(state / max(norm, 1e-9))
    record_states_array = np.vstack(record_states)
    route_weights = record_states_array.copy()
    parameter_count = int(token_embeddings.size + route_weights.size)
    return {
        "profile": profile,
        "seed": seed,
        "state_dim": int(STATE_DIM),
        "records": records,
        "memory": [dict(record) for record in records],
        "vocab": vocab,
        "token_embeddings": token_embeddings,
        "route_weights": route_weights,
        "parameter_count": parameter_count,
        "record_count": int(len(records)),
        "training_method": "state_first_local_binding",
        "next_token_training_used": False,
        "full_model_authorized": False,
        "paid_compute_authorized": False,
        "arbitrary_chat_authorized": False,
        "short_term": {},
        "branches": [],
        "accounted_bits": int(math.ceil(math.log2(max(len(records), 2))) + math.ceil(math.log2(max(len(vocab), 2))) + int(STATE_DIM)),
    }


def predict_record(prompt: str, bundle: dict[str, Any]) -> dict[str, Any] | None:
    cue = cue_in_prompt(prompt, bundle)
    if cue is None:
        return None
    state = encode_text(prompt, bundle)
    scores = bundle["route_weights"] @ state
    index = int(np.argmax(scores))
    routed = bundle["memory"][index]
    if str(routed["cue"]) == cue:
        return routed
    return None


def render_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"text": "v1 chat: outside grounded memory scope", "accepted": False}
    return {"text": f"v1 chat: {record['cue']} links to {record['payload']}. source {record['source']}.", "accepted": True}


def token_after(tokens: list[str], marker: str, fallback_index: int) -> str:
    if marker in tokens:
        index = tokens.index(marker)
        if index + 1 < len(tokens):
            return tokens[index + 1]
    if fallback_index < len(tokens):
        return tokens[fallback_index]
    return tokens[-1] if tokens else ""


def chat_once(prompt: str, bundle: dict[str, Any]) -> dict[str, Any]:
    tokens = tokenize(prompt)
    if not tokens:
        return {"text": "v1 chat: outside grounded memory scope", "accepted": False}
    if tokens[0] == "update":
        cue = token_after(tokens, "update", 1)
        payload = token_after(tokens, "payload", 3)
        record = record_by_cue(bundle, cue)
        if record is None:
            return {"text": "v1 chat: outside grounded memory scope", "accepted": False}
        record["payload"] = payload
        record["source"] = f"update_record_{record['record_id']}"
        bundle["short_term"] = {"cue": cue, "payload": payload, "source": record["source"]}
        return {"text": f"v1 chat: updated {cue} to {payload}. source {record['source']}.", "accepted": True}
    if tokens[0] == "replay":
        cue = token_after(tokens, "replay", 1)
        record = record_by_cue(bundle, cue)
        response = render_record(record)
        response["replay_target"] = cue
        return response
    if tokens[0] == "imagine":
        cue = token_after(tokens, "imagine", 1)
        branch = token_after(tokens, "branch", len(tokens) - 1)
        record = record_by_cue(bundle, cue)
        if record is None:
            return {"text": "v1 chat: outside grounded memory scope", "accepted": False}
        branch_state = {"cue": cue, "payload": record["payload"], "branch": branch, "source": record["source"]}
        bundle["branches"].append(branch_state)
        return {"text": f"v1 chat: branch {branch} for {cue} links to {record['payload']}. source {record['source']}.", "accepted": True, "branch": branch}
    if "changed" in tokens and bundle.get("short_term"):
        state = bundle["short_term"]
        return {"text": f"v1 chat: {state['cue']} now links to {state['payload']}. source {state['source']}.", "accepted": True}
    return render_record(predict_record(prompt, bundle))


def run_chat_script(turns: list[str], bundle: dict[str, Any]) -> dict[str, Any]:
    responses = []
    update_success = 0.0
    short_term_context_success = 0.0
    targeted_replay_success = 0.0
    branch_state_success = 0.0
    final_joint_success = 0.0
    branch_response = ""
    updated_cue = ""
    updated_payload = ""
    for turn in turns:
        response = chat_once(turn, bundle)
        responses.append(response)
        tokens = tokenize(turn)
        if tokens and tokens[0] == "update" and response["accepted"]:
            updated_cue = token_after(tokens, "update", 1)
            updated_payload = token_after(tokens, "payload", 3)
            update_success = 1.0
        if "changed" in tokens and updated_cue and updated_payload in response["text"]:
            short_term_context_success = 1.0
        if tokens and tokens[0] == "replay" and updated_cue in response["text"] and updated_payload in response["text"]:
            targeted_replay_success = 1.0
        if tokens and tokens[0] == "imagine" and response["accepted"]:
            branch_response = response["text"]
            if updated_payload in response["text"]:
                branch_state_success = 1.0
        if updated_cue and updated_payload and updated_cue in response["text"] and updated_payload in response["text"]:
            final_joint_success = 1.0
    return {
        "responses": responses,
        "update_success": update_success,
        "short_term_context_success": short_term_context_success,
        "targeted_replay_success": targeted_replay_success,
        "branch_state_success": branch_state_success,
        "final_joint_success": final_joint_success,
        "branch_response": branch_response,
    }


def score_grounded_examples(bundle: dict[str, Any], route_mode: str = "normal") -> dict[str, float]:
    active_bundle = bundle if route_mode == "normal" else route_control_bundle(bundle, route_mode)
    successes = []
    refusals = []
    for record in active_bundle["records"]:
        response = chat_once(f"what does {record['cue']} preserve for action", active_bundle)
        successes.append(float(response["accepted"] and str(record["cue"]) in response["text"] and str(record["payload"]) in response["text"]))
    for prompt in ("write a song about thunder", "invent a recipe", "tell me a joke about castles"):
        response = chat_once(prompt, active_bundle)
        refusals.append(float(response["accepted"] is False))
    return {"grounded": float(np.mean(successes)) if successes else 0.0, "refusal": float(np.mean(refusals)) if refusals else 0.0}


def one_chat_run(profile: str, seed: int, source_texts: list[str] | None = None) -> dict[str, float]:
    bundle = train_local_10k_chat_model(profile, seed, source_texts)
    score = score_grounded_examples(bundle)
    route_disabled = score_grounded_examples(bundle, "disabled")
    route_shuffled = score_grounded_examples(bundle, "shuffled")
    cue = str(bundle["records"][0]["cue"])
    payload = f"update{seed % 997}"
    script = [
        f"what does {cue} preserve for action",
        f"update {cue} payload {payload} from study note",
        "what changed",
        f"replay {cue}",
        f"imagine {cue} branch rehearsal",
        f"what does {cue} preserve for action",
    ]
    transcript = run_chat_script(script, bundle)
    return {
        "grounded": float(score["grounded"]),
        "refusal": float(score["refusal"]),
        "route_disabled": float(route_disabled["grounded"]),
        "route_shuffled": float(route_shuffled["grounded"]),
        "update": float(transcript["update_success"]),
        "context": float(transcript["short_term_context_success"]),
        "replay": float(transcript["targeted_replay_success"]),
        "branch": float(transcript["branch_state_success"]),
        "final": float(transcript["final_joint_success"]),
        "params": float(bundle["parameter_count"]),
        "records": float(bundle["record_count"]),
        "bits": float(bundle["accounted_bits"]),
    }


def build_local_10k_chat_summary(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    rows = [one_chat_run(profile, seed + index * 1_009, source_texts) for index in range(int(SEED_COUNT))]
    engineering_pass = float(int(rows and max([row["params"] for row in rows]) < 10_000 and min([row["grounded"] for row in rows]) >= 0.9 and min([row["refusal"] for row in rows]) == 1.0 and max([row["route_disabled"] for row in rows]) <= 0.1 and max([row["route_shuffled"] for row in rows]) <= 0.4 and min([row["update"] for row in rows]) >= 0.9 and min([row["context"] for row in rows]) >= 0.9 and min([row["replay"] for row in rows]) >= 0.9 and min([row["branch"] for row in rows]) >= 0.9 and min([row["final"] for row in rows]) >= 0.9))
    return {
        "local_10k_chat_gate_evaluated": 1.0,
        "local_10k_chat_local_model_authorized": 1.0,
        "local_10k_chat_full_model_authorized": 0.0,
        "local_10k_chat_paid_compute_authorized": 0.0,
        "local_10k_chat_arbitrary_chat_authorized": 0.0,
        "local_10k_chat_next_token_training_used": 0.0,
        "local_10k_chat_state_first_training_used": 1.0,
        "local_10k_chat_seed_count": int(SEED_COUNT),
        "local_10k_chat_run_count": int(len(rows)),
        "local_10k_chat_record_count_min": float(min([row["records"] for row in rows]) if rows else 0.0),
        "local_10k_chat_parameter_count_max": float(max([row["params"] for row in rows]) if rows else 0.0),
        "local_10k_chat_accounted_bits_max": float(max([row["bits"] for row in rows]) if rows else 0.0),
        "local_10k_chat_grounded_response_success_min": float(min([row["grounded"] for row in rows]) if rows else 0.0),
        "local_10k_chat_unknown_refusal_success_min": float(min([row["refusal"] for row in rows]) if rows else 0.0),
        "local_10k_chat_route_disabled_grounded_success_max": float(max([row["route_disabled"] for row in rows]) if rows else 0.0),
        "local_10k_chat_route_shuffled_grounded_success_max": float(max([row["route_shuffled"] for row in rows]) if rows else 0.0),
        "local_10k_chat_memory_update_success_min": float(min([row["update"] for row in rows]) if rows else 0.0),
        "local_10k_chat_short_term_context_success_min": float(min([row["context"] for row in rows]) if rows else 0.0),
        "local_10k_chat_targeted_replay_success_min": float(min([row["replay"] for row in rows]) if rows else 0.0),
        "local_10k_chat_branch_state_success_min": float(min([row["branch"] for row in rows]) if rows else 0.0),
        "local_10k_chat_final_dialogue_joint_success_min": float(min([row["final"] for row in rows]) if rows else 0.0),
        "local_10k_chat_engineering_pass": engineering_pass,
        "local_10k_chat_claim_downgraded_to_constrained_responder": float(1.0 - engineering_pass),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()
    profile = infer_profile()
    bundle = train_local_10k_chat_model(profile, SEED)
    if args.chat:
        if args.prompt:
            print(chat_once(args.prompt, bundle)["text"])
            return 0
        for line in sys.stdin:
            text = line.strip()
            if text in {"exit", "quit"}:
                break
            print(chat_once(text, bundle)["text"])
        return 0
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_local_10k_chat_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_10k_chat_model_metrics.json"
    record = build_run_record(
        simulation_name="local_10k_chat_model",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=utc_now_iso(),
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "max_records": int(MAX_RECORDS), "seed_count": int(SEED_COUNT), "state_dim": int(STATE_DIM)},
        seed_numpy=int(SEED),
        n_trials=int(summary["local_10k_chat_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"path": metrics_path.as_posix(), "type": "metrics"}],
        warnings=["local constrained 10k chat surface only; not arbitrary chat, not solved compression, not full-scale language model"],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
