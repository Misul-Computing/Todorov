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
from neuroloc.simulations.memory.local_10k_chat_model import CORE_SOURCE_TEXTS, stable_token_vector
from neuroloc.simulations.memory.local_v1_language_model import build_records, content_tokens, tokenize

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("LOCAL_FOUNDATION_NM_SEED", 131)
MAX_RECORDS = env_int("LOCAL_FOUNDATION_NM_MAX_RECORDS", 24)
SEED_COUNT = env_int("LOCAL_FOUNDATION_NM_SEED_COUNT", 2)
STATE_DIM = env_int("LOCAL_FOUNDATION_NM_STATE_DIM", 40)

require_positive("LOCAL_FOUNDATION_NM_MAX_RECORDS", MAX_RECORDS)
require_positive("LOCAL_FOUNDATION_NM_SEED_COUNT", SEED_COUNT)
require_positive("LOCAL_FOUNDATION_NM_STATE_DIM", STATE_DIM)

PROMPT_TEMPLATES = (
    "what does {cue} preserve",
    "tell me about {cue}",
    "which action belongs to {cue}",
    "answer from working memory about {cue}",
)

ACTION_BY_CUE = {
    "cortex": "bind",
    "synapse": "store",
    "memory": "preserve",
    "compression": "budget",
    "replay": "refresh",
    "imagination": "branch",
    "context": "carry",
    "consolidation": "stabilize",
    "language": "route",
    "occlusion": "hide",
    "dreaming": "rehearse",
    "attention": "select",
    "neuron": "commit",
    "world": "update",
    "provenance": "name",
    "decoder": "reconstruct",
    "gate": "open",
    "schema": "separate",
    "dataset": "teach",
    "branch": "preserve",
    "short": "carry",
    "long": "store",
    "local": "bind",
    "router": "select",
    "refusal": "protect",
    "rewrite": "change",
    "control": "disable",
    "shuffle": "permute",
    "bit": "count",
    "useful": "measure",
    "foundation": "prove",
    "chat": "answer",
}


def infer_profile() -> str:
    return os.environ.get("LOCAL_FOUNDATION_NM_PROFILE", "smoke")


def build_vocab(records: list[dict[str, Any]]) -> dict[str, int]:
    words = set()
    for record in records:
        words.update(content_tokens(str(record["text"])))
        words.add(str(record["cue"]))
        words.add(str(record["payload"]))
    for template in PROMPT_TEMPLATES:
        words.update(tokenize(template))
    words.update({"update", "payload", "replay", "imagine", "branch", "changed", "act", "action", "context", "working", "memory", "long", "term"})
    words.update(ACTION_BY_CUE.values())
    return {word: index for index, word in enumerate(sorted(words))}


def encode_text(text: str, bundle: dict[str, Any]) -> np.ndarray:
    state = np.zeros(int(bundle["state_dim"]), dtype=np.float64)
    count = 0.0
    cue_tokens = {str(record["cue"]) for record in bundle["records"]}
    for token in tokenize(text):
        index = bundle["vocab"].get(token)
        if index is not None:
            weight = 9.0 if token in cue_tokens else 1.0
            state += weight * bundle["token_embeddings"][index]
            count += weight
    if count > 0.0:
        state /= count
    norm = float(np.linalg.norm(state))
    if norm > 0.0:
        state /= norm
    return state


def action_for_record(record: dict[str, Any]) -> str:
    cue = str(record["cue"])
    return ACTION_BY_CUE.get(cue, str(record["payload"]))


def memory_vector(record: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray:
    state = np.zeros(int(bundle["state_dim"]), dtype=np.float64)
    for token in tokenize(str(record["cue"])):
        index = bundle["vocab"].get(token)
        if index is not None:
            state += 9.0 * bundle["token_embeddings"][index]
    for token in tokenize(str(record["payload"])):
        index = bundle["vocab"].get(token)
        if index is not None:
            state += 2.0 * bundle["token_embeddings"][index]
    for token in tokenize(action_for_record(record)):
        index = bundle["vocab"].get(token)
        if index is not None:
            state += 1.5 * bundle["token_embeddings"][index]
    norm = float(np.linalg.norm(state))
    return state / max(norm, 1e-9)


def encode_memory_object(record: dict[str, Any], bundle: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    vector = memory_vector(record, bundle)
    schema = str(record["cue"]).split("_")[0]
    return {
        "address": str(record["cue"]),
        "schema": schema,
        "payload": str(record["payload"]),
        "action": action_for_record(record),
        "source": source or str(record["source"]),
        "record_id": int(record["record_id"]),
        "version": int(record.get("version", 0)),
        "vector": vector,
    }


def decode_memory_code(code: dict[str, Any] | None, bundle: dict[str, Any]) -> dict[str, Any] | None:
    if code is None or not bundle.get("codec_enabled", True):
        return None
    return {
        "cue": str(code["address"]),
        "payload": str(code["payload"]),
        "action": str(code["action"]),
        "source": str(code["source"]),
        "record_id": int(code["record_id"]),
        "version": int(code["version"]),
    }


def cue_in_prompt(prompt: str, bundle: dict[str, Any]) -> str | None:
    tokens = set(tokenize(prompt))
    for code in bundle["long_term_memory"]:
        if str(code["address"]) in tokens:
            return str(code["address"])
    return None


def route_memory(prompt: str, bundle: dict[str, Any]) -> dict[str, Any] | None:
    cue = cue_in_prompt(prompt, bundle)
    if cue is None:
        return None
    query = encode_text(prompt, bundle)
    scores = bundle["route_weights"] @ query
    index = int(np.argmax(scores))
    code = bundle["long_term_memory"][index]
    if str(code["address"]) != cue:
        return None
    bundle["working_memory"] = dict(code)
    bundle["context_state"] = {"cue": cue, "source": str(code["source"]), "operation": "route"}
    return code


def control_bundle(bundle: dict[str, Any], mode: str) -> dict[str, Any]:
    controlled = dict(bundle)
    controlled["long_term_memory"] = [dict(code) for code in bundle["long_term_memory"]]
    controlled["episodic_memory"] = [dict(code) for code in bundle["episodic_memory"]]
    controlled["replay_buffer"] = list(bundle["replay_buffer"])
    controlled["branch_memory"] = list(bundle["branch_memory"])
    controlled["context_state"] = dict(bundle["context_state"])
    controlled["working_memory"] = dict(bundle["working_memory"]) if bundle["working_memory"] else None
    if mode == "disabled_route":
        controlled["route_weights"] = np.zeros_like(bundle["route_weights"])
    elif mode == "shuffled_route":
        controlled["route_weights"] = np.roll(bundle["route_weights"], shift=1, axis=0)
    elif mode == "disabled_codec":
        controlled["codec_enabled"] = False
    elif mode == "disabled_replay_buffer":
        controlled["replay_enabled"] = False
    elif mode == "disabled_branch_memory":
        controlled["branch_enabled"] = False
    return controlled


def train_foundation_neural_model(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    records = build_records(source_texts if source_texts is not None else CORE_SOURCE_TEXTS, int(MAX_RECORDS))
    rng.shuffle(records)
    records = sorted([{**record, "record_id": index, "source": f"record_{index}", "version": 0} for index, record in enumerate(records)], key=lambda item: int(item["record_id"]))
    vocab = build_vocab(records)
    token_embeddings = np.vstack([stable_token_vector(token, int(STATE_DIM)) for token in vocab])
    partial = {"state_dim": int(STATE_DIM), "vocab": vocab, "token_embeddings": token_embeddings}
    codes = [encode_memory_object(record, {**partial, "records": records}, str(record["source"])) for record in records]
    route_weights = np.vstack([code["vector"] for code in codes])
    schemas = {str(code["schema"]) for code in codes}
    actions = {str(code["action"]) for code in codes}
    parameter_count = int(token_embeddings.size + route_weights.size)
    accounted_bits = int(math.ceil(math.log2(max(len(records), 2))) + math.ceil(math.log2(max(len(vocab), 2))) + math.ceil(math.log2(max(len(schemas), 2))) + math.ceil(math.log2(max(len(actions), 2))) + int(STATE_DIM))
    return {
        "profile": profile,
        "seed": seed,
        "state_dim": int(STATE_DIM),
        "records": records,
        "vocab": vocab,
        "token_embeddings": token_embeddings,
        "route_weights": route_weights,
        "long_term_memory": [dict(code) for code in codes],
        "episodic_memory": [dict(code) for code in codes],
        "working_memory": None,
        "context_state": {},
        "replay_buffer": [],
        "branch_memory": [],
        "codec_name": "schema_residual_identity_v0",
        "codec_enabled": True,
        "replay_enabled": True,
        "branch_enabled": True,
        "training_method": "state_first_memory_object_binding",
        "next_token_training_used": False,
        "full_model_authorized": False,
        "paid_compute_authorized": False,
        "arbitrary_chat_authorized": False,
        "parameter_count": parameter_count,
        "record_count": int(len(records)),
        "accounted_bits": accounted_bits,
    }


def render_answer(decoded: dict[str, Any] | None, mode: str = "answer") -> dict[str, Any]:
    if decoded is None:
        return {"text": "foundation chat: outside grounded memory scope", "accepted": False}
    if mode == "action":
        return {"text": f"foundation chat: action {decoded['action']} for {decoded['cue']}. source {decoded['source']}.", "accepted": True}
    return {"text": f"foundation chat: {decoded['cue']} -> {decoded['payload']}; action {decoded['action']}; source {decoded['source']}.", "accepted": True}


def token_after(tokens: list[str], marker: str, fallback: int) -> str:
    if marker in tokens:
        index = tokens.index(marker)
        if index + 1 < len(tokens):
            return tokens[index + 1]
    if fallback < len(tokens):
        return tokens[fallback]
    return tokens[-1] if tokens else ""


def code_by_cue(bundle: dict[str, Any], cue: str) -> dict[str, Any] | None:
    for code in bundle["long_term_memory"]:
        if str(code["address"]) == cue:
            return code
    return None


def foundation_chat_once(prompt: str, bundle: dict[str, Any]) -> dict[str, Any]:
    tokens = tokenize(prompt)
    if not tokens:
        return {"text": "foundation chat: outside grounded memory scope", "accepted": False}
    if tokens[0] == "update":
        cue = token_after(tokens, "update", 1)
        payload = token_after(tokens, "payload", 3)
        code = code_by_cue(bundle, cue)
        if code is None:
            return {"text": "foundation chat: outside grounded memory scope", "accepted": False}
        code["payload"] = payload
        code["source"] = f"update_record_{code['record_id']}"
        code["version"] = int(code["version"]) + 1
        bundle["episodic_memory"].append(dict(code))
        bundle["working_memory"] = dict(code)
        bundle["context_state"] = {"cue": cue, "payload": payload, "source": code["source"], "operation": "update"}
        decoded = decode_memory_code(code, bundle)
        return {"text": f"foundation chat: updated {cue} -> {payload}; source {decoded['source']}.", "accepted": True}
    if tokens[0] == "replay":
        cue = token_after(tokens, "replay", 1)
        code = code_by_cue(bundle, cue)
        if code is None:
            return {"text": "foundation chat: outside grounded memory scope", "accepted": False}
        if bundle.get("replay_enabled", True):
            bundle["replay_buffer"].append(dict(code))
        replay_code = bundle["replay_buffer"][-1] if bundle["replay_buffer"] else None
        return render_answer(decode_memory_code(replay_code, bundle))
    if tokens[0] == "imagine":
        cue = token_after(tokens, "imagine", 1)
        branch = token_after(tokens, "branch", len(tokens) - 1)
        code = code_by_cue(bundle, cue)
        if code is None:
            return {"text": "foundation chat: outside grounded memory scope", "accepted": False}
        if bundle.get("branch_enabled", True):
            branch_code = dict(code)
            branch_code["branch"] = branch
            branch_code["source"] = f"branch_from_{code['source']}"
            bundle["branch_memory"].append(branch_code)
        branch_code = bundle["branch_memory"][-1] if bundle["branch_memory"] else None
        decoded = decode_memory_code(branch_code, bundle)
        if decoded is None:
            return {"text": "foundation chat: outside grounded memory scope", "accepted": False}
        return {"text": f"foundation chat: branch {branch} for {decoded['cue']} -> {decoded['payload']}; action {decoded['action']}; source {decoded['source']}.", "accepted": True}
    if tokens[0] in {"act", "action"}:
        cue = token_after(tokens, tokens[0], 1)
        code = code_by_cue(bundle, cue)
        return render_answer(decode_memory_code(code, bundle), "action")
    if "changed" in tokens and bundle.get("context_state"):
        state = bundle["context_state"]
        if "payload" in state:
            return {"text": f"foundation chat: {state['cue']} now maps to {state['payload']}. source {state['source']}.", "accepted": True}
    code = route_memory(prompt, bundle)
    return render_answer(decode_memory_code(code, bundle))


def run_architecture_script(bundle: dict[str, Any]) -> dict[str, Any]:
    cue = str(bundle["records"][0]["cue"])
    payload = f"foundation{int(bundle['seed']) % 997}"
    turns = [
        f"what does {cue} preserve",
        f"action {cue}",
        f"update {cue} payload {payload}",
        "what changed",
        f"replay {cue}",
        f"imagine {cue} branch rehearsal",
        f"what does {cue} preserve",
    ]
    responses = [foundation_chat_once(turn, bundle) for turn in turns]
    return {
        "cue": cue,
        "payload": payload,
        "responses": responses,
        "grounded": float(responses[0]["accepted"] and cue in responses[0]["text"]),
        "action": float(responses[1]["accepted"] and "action" in responses[1]["text"]),
        "update": float(responses[2]["accepted"] and payload in responses[2]["text"]),
        "context": float(responses[3]["accepted"] and payload in responses[3]["text"]),
        "replay": float(responses[4]["accepted"] and payload in responses[4]["text"]),
        "branch": float(responses[5]["accepted"] and payload in responses[5]["text"] and "branch" in responses[5]["text"]),
        "final": float(responses[6]["accepted"] and payload in responses[6]["text"]),
        "working": float(bundle["working_memory"] is not None),
        "episodic": float(len(bundle["episodic_memory"]) > len(bundle["records"])),
        "long_term": float(len(bundle["long_term_memory"]) == len(bundle["records"])),
        "replay_buffer": float(len(bundle["replay_buffer"]) > 0),
        "branch_memory": float(len(bundle["branch_memory"]) > 0),
    }


def score_grounded(bundle: dict[str, Any], mode: str = "normal") -> float:
    active = bundle if mode == "normal" else control_bundle(bundle, mode)
    successes = []
    for record in active["records"]:
        response = foundation_chat_once(f"what does {record['cue']} preserve", active)
        successes.append(float(response["accepted"] and str(record["cue"]) in response["text"] and str(record["payload"]) in response["text"]))
    return float(np.mean(successes)) if successes else 0.0


def score_refusal(bundle: dict[str, Any]) -> float:
    prompts = ("write a sonnet about rain", "invent a meal", "explain ocean tides")
    return float(np.mean([float(foundation_chat_once(prompt, bundle)["accepted"] is False) for prompt in prompts]))


def disabled_surface_success(profile: str, seed: int, source_texts: list[str] | None, mode: str, prompt_prefix: str) -> float:
    bundle = control_bundle(train_foundation_neural_model(profile, seed, source_texts), mode)
    cue = str(bundle["records"][0]["cue"])
    response = foundation_chat_once(f"{prompt_prefix} {cue} branch rehearsal", bundle)
    return float(response["accepted"])


def one_foundation_run(profile: str, seed: int, source_texts: list[str] | None = None) -> dict[str, float]:
    bundle = train_foundation_neural_model(profile, seed, source_texts)
    script = run_architecture_script(bundle)
    return {
        "grounded": score_grounded(train_foundation_neural_model(profile, seed, source_texts)),
        "refusal": score_refusal(train_foundation_neural_model(profile, seed, source_texts)),
        "route_disabled": score_grounded(train_foundation_neural_model(profile, seed, source_texts), "disabled_route"),
        "route_shuffled": score_grounded(train_foundation_neural_model(profile, seed, source_texts), "shuffled_route"),
        "codec_disabled": score_grounded(train_foundation_neural_model(profile, seed, source_texts), "disabled_codec"),
        "replay_disabled": disabled_surface_success(profile, seed, source_texts, "disabled_replay_buffer", "replay"),
        "branch_disabled": disabled_surface_success(profile, seed, source_texts, "disabled_branch_memory", "imagine"),
        "action": float(script["action"]),
        "update": float(script["update"]),
        "context": float(script["context"]),
        "replay": float(script["replay"]),
        "branch": float(script["branch"]),
        "final": float(script["final"]),
        "working": float(script["working"]),
        "episodic": float(script["episodic"]),
        "long_term": float(script["long_term"]),
        "replay_buffer": float(script["replay_buffer"]),
        "branch_memory": float(script["branch_memory"]),
        "params": float(bundle["parameter_count"]),
        "records": float(bundle["record_count"]),
        "bits": float(bundle["accounted_bits"]),
    }


def build_foundation_neural_model_summary(profile: str = "smoke", seed: int = SEED, source_texts: list[str] | None = None) -> dict[str, Any]:
    rows = [one_foundation_run(profile, seed + index * 1_009, source_texts) for index in range(int(SEED_COUNT))]
    engineering_pass = float(int(rows and max(row["params"] for row in rows) < 10_000 and min(row["grounded"] for row in rows) >= 0.9 and min(row["refusal"] for row in rows) == 1.0 and max(row["route_disabled"] for row in rows) <= 0.1 and max(row["route_shuffled"] for row in rows) <= 0.4 and max(row["codec_disabled"] for row in rows) == 0.0 and max(row["replay_disabled"] for row in rows) == 0.0 and max(row["branch_disabled"] for row in rows) == 0.0 and min(row["action"] for row in rows) >= 0.9 and min(row["update"] for row in rows) >= 0.9 and min(row["context"] for row in rows) >= 0.9 and min(row["replay"] for row in rows) >= 0.9 and min(row["branch"] for row in rows) >= 0.9 and min(row["final"] for row in rows) >= 0.9 and min(row["working"] for row in rows) >= 0.9 and min(row["episodic"] for row in rows) >= 0.9 and min(row["long_term"] for row in rows) >= 0.9 and min(row["replay_buffer"] for row in rows) >= 0.9 and min(row["branch_memory"] for row in rows) >= 0.9))
    return {
        "foundation_nm_gate_evaluated": 1.0,
        "foundation_nm_local_model_authorized": 1.0,
        "foundation_nm_full_model_authorized": 0.0,
        "foundation_nm_paid_compute_authorized": 0.0,
        "foundation_nm_arbitrary_chat_authorized": 0.0,
        "foundation_nm_next_token_training_used": 0.0,
        "foundation_nm_state_first_training_used": 1.0,
        "foundation_nm_codec_boundary_used": 1.0,
        "foundation_nm_simple_codec_used": 1.0,
        "foundation_nm_extreme_compression_claimed": 0.0,
        "foundation_nm_seed_count": int(SEED_COUNT),
        "foundation_nm_run_count": int(len(rows)),
        "foundation_nm_record_count_min": float(min(row["records"] for row in rows) if rows else 0.0),
        "foundation_nm_parameter_count_max": float(max(row["params"] for row in rows) if rows else 0.0),
        "foundation_nm_accounted_bits_max": float(max(row["bits"] for row in rows) if rows else 0.0),
        "foundation_nm_grounded_response_success_min": float(min(row["grounded"] for row in rows) if rows else 0.0),
        "foundation_nm_unknown_refusal_success_min": float(min(row["refusal"] for row in rows) if rows else 0.0),
        "foundation_nm_route_disabled_grounded_success_max": float(max(row["route_disabled"] for row in rows) if rows else 0.0),
        "foundation_nm_route_shuffled_grounded_success_max": float(max(row["route_shuffled"] for row in rows) if rows else 0.0),
        "foundation_nm_codec_disabled_grounded_success_max": float(max(row["codec_disabled"] for row in rows) if rows else 0.0),
        "foundation_nm_replay_disabled_success_max": float(max(row["replay_disabled"] for row in rows) if rows else 0.0),
        "foundation_nm_branch_disabled_success_max": float(max(row["branch_disabled"] for row in rows) if rows else 0.0),
        "foundation_nm_action_success_min": float(min(row["action"] for row in rows) if rows else 0.0),
        "foundation_nm_memory_update_success_min": float(min(row["update"] for row in rows) if rows else 0.0),
        "foundation_nm_context_success_min": float(min(row["context"] for row in rows) if rows else 0.0),
        "foundation_nm_targeted_replay_success_min": float(min(row["replay"] for row in rows) if rows else 0.0),
        "foundation_nm_branch_state_success_min": float(min(row["branch"] for row in rows) if rows else 0.0),
        "foundation_nm_final_dialogue_joint_success_min": float(min(row["final"] for row in rows) if rows else 0.0),
        "foundation_nm_working_memory_used_min": float(min(row["working"] for row in rows) if rows else 0.0),
        "foundation_nm_episodic_memory_used_min": float(min(row["episodic"] for row in rows) if rows else 0.0),
        "foundation_nm_long_term_memory_used_min": float(min(row["long_term"] for row in rows) if rows else 0.0),
        "foundation_nm_replay_buffer_used_min": float(min(row["replay_buffer"] for row in rows) if rows else 0.0),
        "foundation_nm_branch_memory_used_min": float(min(row["branch_memory"] for row in rows) if rows else 0.0),
        "foundation_nm_engineering_pass": engineering_pass,
        "foundation_nm_claim_downgraded_to_foundation_surface": float(1.0 - engineering_pass),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()
    profile = infer_profile()
    bundle = train_foundation_neural_model(profile, SEED)
    if args.chat:
        if args.prompt:
            print(foundation_chat_once(args.prompt, bundle)["text"])
            return 0
        for line in sys.stdin:
            text = line.strip()
            if text in {"exit", "quit"}:
                break
            print(foundation_chat_once(text, bundle)["text"])
        return 0
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_foundation_neural_model_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_foundation_neural_model_metrics.json"
    record = build_run_record(
        simulation_name="local_foundation_neural_model",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=utc_now_iso(),
        duration_sec=float(time.perf_counter() - started),
        parameters={"profile": profile, "max_records": int(MAX_RECORDS), "seed_count": int(SEED_COUNT), "state_dim": int(STATE_DIM)},
        seed_numpy=int(SEED),
        n_trials=int(summary["foundation_nm_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"path": metrics_path.as_posix(), "type": "metrics"}],
        warnings=["local foundation neural-model surface only; simple codec boundary, not solved compression, not arbitrary chat, not paid compute"],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
