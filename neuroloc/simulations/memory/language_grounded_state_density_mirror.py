from __future__ import annotations

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
from neuroloc.simulations.memory.compression_under_bit_budget_mirror import (
    bits_for_cardinality,
    build_factor_heldout_distributed_dataset,
    distributed_evidence_sparse_read_result,
    estimate_velocity,
    factor_holdout_bucket_for_key,
    first_known,
    learned_code_bits,
    last_known,
    matched_budget_sparse_read_result,
    one_hot,
    profile_caps,
    visible_source_action,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("LGSD_SEED", 52)
TRAIN_EPISODES = env_int("LGSD_TRAIN_EPISODES", 2048)
VAL_EPISODES = env_int("LGSD_VAL_EPISODES", 96)
TEST_EPISODES = env_int("LGSD_TEST_EPISODES", 96)
EPOCHS = env_int("LGSD_EPOCHS", 180)
SEED_COUNT = env_int("LGSD_SEED_COUNT", 2)
HEAD_RANK = env_int("LGSD_HEAD_RANK", 16)

require_positive("LGSD_TRAIN_EPISODES", TRAIN_EPISODES)
require_positive("LGSD_VAL_EPISODES", VAL_EPISODES)
require_positive("LGSD_TEST_EPISODES", TEST_EPISODES)
require_positive("LGSD_EPOCHS", EPOCHS)
require_positive("LGSD_SEED_COUNT", SEED_COUNT)
require_positive("LGSD_HEAD_RANK", HEAD_RANK)

AXES = ("color_shape_pair_band", "color_velocity_pair_band", "shape_velocity_pair_band", "position_velocity_phase_band")
SLOT_WORDS = ("ada", "bex", "cato", "dima", "evan", "fara", "goro", "hema")
COLOR_WORDS = ("crimson", "amber", "teal", "violet", "olive", "silver", "indigo", "coral")
SHAPE_WORDS = ("cube", "ring", "pyramid", "sphere", "prism", "wedge", "cone", "bar")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
EVENT_TEMPLATE_COUNT = 4
QUERY_TEMPLATE_COUNT = 4


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("LGSD_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("LGSD_PROFILE must be smoke or hard")
    return value


def word_at(values: tuple[str, ...], index: int) -> str:
    return values[int(index) % len(values)]


def randomized_event_text(event: dict[str, int], template_index: int) -> str:
    slot = word_at(SLOT_WORDS, int(event["object_index"]))
    time_word = word_at(NUMBER_WORDS, int(event["time"]))
    parts: list[str] = []
    if int(event["color"]) >= 0:
        parts.append(f"hue {word_at(COLOR_WORDS, int(event['color']))}")
    if int(event["shape"]) >= 0:
        parts.append(f"form {word_at(SHAPE_WORDS, int(event['shape']))}")
    if int(event["pos"]) >= 0:
        parts.append(f"place {word_at(NUMBER_WORDS, int(event['pos']))}")
    body = ", ".join(parts) if parts else "no visible detail"
    if int(template_index) % EVENT_TEMPLATE_COUNT == 0:
        return f"at moment {time_word}, object {slot} showed {body}"
    if int(template_index) % EVENT_TEMPLATE_COUNT == 1:
        return f"report for {slot}: {body} during step {time_word}"
    if int(template_index) % EVENT_TEMPLATE_COUNT == 2:
        return f"when the beat was {time_word}, {slot} carried {body}"
    return f"{slot} was logged with {body} near clock {time_word}"


def randomized_record_prompt(record: dict[str, Any], seed: int) -> str:
    segments, query = randomized_record_parts(record, seed)
    return " | ".join([segment for segment, _ in segments] + [query])


def randomized_record_parts(record: dict[str, Any], seed: int) -> tuple[list[tuple[str, dict[str, int] | None]], str]:
    rng = np.random.default_rng(int(seed))
    events = list(record["model_input"]["observations"])
    order = rng.permutation(len(events)) if events else []
    rendered = [(randomized_event_text(events[int(index)], int(rng.integers(0, EVENT_TEMPLATE_COUNT))), events[int(index)]) for index in order]
    if int(rng.integers(0, 2)) == 1:
        rendered.insert(int(rng.integers(0, len(rendered) + 1)), ("background note: the room stayed quiet and this clause is irrelevant", None))
    focus = word_at(SLOT_WORDS, int(record["model_input"]["query"]["focus_local_index"]))
    query_style = int(rng.integers(0, QUERY_TEMPLATE_COUNT))
    if query_style == 0:
        query = f"which action and state should {focus} answer with"
    elif query_style == 1:
        query = f"for object {focus}, return action plus remembered state"
    elif query_style == 2:
        query = f"decide the response for {focus} using the prior reports"
    else:
        query = f"give {focus} the correct action, hue, form, place, and motion"
    return rendered, query


def tokenize_randomized_prompt(prompt: str) -> list[str]:
    return re.findall(r"[a-z]+", prompt.lower())


def event_text(event: dict[str, int]) -> str:
    parts = [f"time_{int(event['time'])}", f"slot_{int(event['object_index'])}"]
    if int(event["color"]) >= 0:
        parts.append(f"color_{int(event['color'])}")
    if int(event["shape"]) >= 0:
        parts.append(f"shape_{int(event['shape'])}")
    if int(event["pos"]) >= 0:
        parts.append(f"pos_{int(event['pos'])}")
    return " ".join(parts)


def record_prompt(record: dict[str, Any]) -> str:
    events = sorted(record["model_input"]["observations"], key=lambda item: (int(item["time"]), int(item["object_index"])))
    focus = int(record["model_input"]["query"]["focus_local_index"])
    observations = " ; ".join(event_text(event) for event in events)
    return f"observations {observations} question action_for slot_{focus}"


def answer_text(record: dict[str, Any]) -> str:
    state = record["labels"]["state"]
    return f"answer action_{int(record['labels']['action'])} color_{int(state['color'])} shape_{int(state['shape'])} pos_{int(state['pos'])} vel_{int(state['vel'])}"


def parse_index(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    return int(token[len(prefix) :])


def parse_prompt(prompt: str) -> tuple[list[dict[str, int]], int]:
    events: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    focus = 0
    tokens = prompt.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "question":
            if current is not None:
                events.append(current)
                current = None
            if index + 2 < len(tokens) and tokens[index + 1] == "action_for":
                parsed = parse_index(tokens[index + 2], "slot_")
                focus = int(parsed if parsed is not None else 0)
            break
        if token == ";":
            if current is not None:
                events.append(current)
                current = None
            index += 1
            continue
        parsed_time = parse_index(token, "time_")
        if parsed_time is not None:
            if current is not None:
                events.append(current)
            current = {"time": int(parsed_time), "object_index": 0, "color": -1, "shape": -1, "pos": -1, "observed": 1}
            index += 1
            continue
        if current is not None:
            for key, prefix in (("object_index", "slot_"), ("color", "color_"), ("shape", "shape_"), ("pos", "pos_")):
                parsed = parse_index(token, prefix)
                if parsed is not None:
                    current[key] = int(parsed)
        index += 1
    if current is not None:
        events.append(current)
    return events, focus


def vectorize_message_fields(prompt: str, caps: dict[str, int]) -> dict[str, np.ndarray]:
    events, focus = parse_prompt(prompt)
    focus_events = [event for event in events if int(event["object_index"]) == int(focus)]
    color_missing = int(caps["n_colors"])
    shape_missing = int(caps["n_shapes"])
    pos_missing = int(caps["track_length"])
    position_events = sorted([(int(event["time"]), int(event["pos"])) for event in focus_events if int(event["pos"]) >= 0])
    commit_time = position_events[len(position_events) // 2][0] if position_events else 0
    prior_positions = [(int(event["time"]), int(event["pos"])) for event in focus_events if int(event["pos"]) >= 0 and int(event["time"]) <= commit_time]
    later_positions = [(int(event["time"]), int(event["pos"])) for event in focus_events if int(event["pos"]) >= 0 and int(event["time"]) >= commit_time]
    commit_positions = [pos for time_index, pos in prior_positions if int(time_index) == int(commit_time)]
    prior_pos = sorted(prior_positions)[-1][1] if prior_positions else pos_missing
    later_pos = sorted(later_positions)[0][1] if later_positions else pos_missing
    commit_pos = commit_positions[0] if commit_positions else prior_pos
    velocity = estimate_velocity(focus_events, int(caps["max_speed"]))
    color_values: list[float] = []
    color_values.extend(one_hot(last_known(focus_events, "color", color_missing), int(caps["n_colors"]) + 1))
    color_values.extend(one_hot(first_known(focus_events, "color", color_missing), int(caps["n_colors"]) + 1))
    shape_values: list[float] = []
    shape_values.extend(one_hot(last_known(focus_events, "shape", shape_missing), int(caps["n_shapes"]) + 1))
    shape_values.extend(one_hot(first_known(focus_events, "shape", shape_missing), int(caps["n_shapes"]) + 1))
    pos_values: list[float] = []
    pos_values.extend(one_hot(commit_pos, int(caps["track_length"]) + 1))
    pos_values.extend(one_hot(prior_pos, int(caps["track_length"]) + 1))
    pos_values.extend(one_hot(later_pos, int(caps["track_length"]) + 1))
    pos_values.extend(one_hot(last_known(focus_events, "pos", pos_missing), int(caps["track_length"]) + 1))
    pos_values.extend(one_hot(first_known(focus_events, "pos", pos_missing), int(caps["track_length"]) + 1))
    velocity_values: list[float] = []
    velocity_values.extend(one_hot(velocity + int(caps["max_speed"]), int(caps["max_speed"]) * 2 + 1))
    provenance_values: list[float] = []
    provenance_values.extend(one_hot(commit_time, int(caps["seq_len"])))
    return {
        "color": np.asarray(color_values, dtype=np.float32),
        "shape": np.asarray(shape_values, dtype=np.float32),
        "pos": np.asarray(pos_values, dtype=np.float32),
        "vel": np.asarray(velocity_values, dtype=np.float32),
        "action": np.concatenate([np.asarray(color_values, dtype=np.float32), np.asarray(shape_values, dtype=np.float32), np.asarray(velocity_values, dtype=np.float32)]),
        "provenance": np.asarray(provenance_values, dtype=np.float32),
    }


def natural_events_from_prompt(prompt: str) -> tuple[list[dict[str, int]], int]:
    events: list[dict[str, int]] = []
    segments = [segment.strip() for segment in prompt.split("|") if segment.strip()]
    query_tokens = tokenize_randomized_prompt(segments[-1] if segments else "")
    focus = 0
    for token in query_tokens:
        if token in SLOT_WORDS:
            focus = int(SLOT_WORDS.index(token))
            break
    for segment in segments[:-1]:
        tokens = tokenize_randomized_prompt(segment)
        slot_tokens = [int(SLOT_WORDS.index(token)) for token in tokens if token in SLOT_WORDS]
        if not slot_tokens:
            continue
        slot = int(slot_tokens[0])
        time_value = 0
        for index, token in enumerate(tokens[:-1]):
            if token in {"moment", "step", "beat", "clock"} and tokens[index + 1] in NUMBER_WORDS:
                time_value = int(NUMBER_WORDS.index(tokens[index + 1]))
            if token in {"beat", "clock"} and index + 2 < len(tokens) and tokens[index + 1] in {"was", "near"} and tokens[index + 2] in NUMBER_WORDS:
                time_value = int(NUMBER_WORDS.index(tokens[index + 2]))
        event = {"time": int(time_value), "object_index": int(slot), "color": -1, "shape": -1, "pos": -1, "observed": 1}
        for index, token in enumerate(tokens[:-1]):
            value = tokens[index + 1]
            if token == "hue" and value in COLOR_WORDS:
                event["color"] = int(COLOR_WORDS.index(value))
            if token == "form" and value in SHAPE_WORDS:
                event["shape"] = int(SHAPE_WORDS.index(value))
            if token == "place" and value in NUMBER_WORDS:
                event["pos"] = int(NUMBER_WORDS.index(value))
        events.append(event)
    return events, focus


def natural_rule_code_fields(prompt: str, caps: dict[str, int]) -> dict[str, int]:
    events, focus = natural_events_from_prompt(prompt)
    focus_events = [event for event in events if int(event["object_index"]) == int(focus)]
    position_events = sorted([(int(event["time"]), int(event["pos"])) for event in focus_events if int(event["pos"]) >= 0])
    commit_time = position_events[len(position_events) // 2][0] if position_events else 0
    commit_pos = [pos for time_value, pos in position_events if int(time_value) == int(commit_time)]
    pos = int(commit_pos[0] if commit_pos else (position_events[-1][1] if position_events else 0))
    color = int(last_known(focus_events, "color", 0))
    shape = int(last_known(focus_events, "shape", 0))
    velocity = int(estimate_velocity(focus_events, int(caps["max_speed"])))
    fields = {
        "address": int(color * int(caps["n_shapes"]) + shape),
        "schema": int(velocity + int(caps["max_speed"])),
        "residual": int(pos),
        "action": 0,
        "provenance": int(commit_time),
    }
    fields["action"] = int(visible_source_action(fields, caps))
    return fields


def parser_schema_cost_bits(caps: dict[str, int]) -> int:
    return int(
        bits_for_cardinality(len(SLOT_WORDS))
        + bits_for_cardinality(len(COLOR_WORDS))
        + bits_for_cardinality(len(SHAPE_WORDS))
        + bits_for_cardinality(len(NUMBER_WORDS))
        + bits_for_cardinality(EVENT_TEMPLATE_COUNT)
        + bits_for_cardinality(QUERY_TEMPLATE_COUNT)
        + learned_code_bits(caps)
    )


def randomized_prompts_for_records(records: list[dict[str, Any]], seed: int) -> list[str]:
    return [randomized_record_prompt(row, seed + index * 104_729) for index, row in enumerate(records)]


def build_vocab(prompts: list[str]) -> dict[str, int]:
    tokens = sorted({token for prompt in prompts for token in tokenize_randomized_prompt(prompt)})
    return {token: index for index, token in enumerate(tokens)}


def segment_feature_tokens(text: str) -> list[str]:
    tokens = tokenize_randomized_prompt(text)
    role_tokens = {"beat", "clock", "near", "place", "moment", "tick", "at", "around", "position", "marked", "marker", "slot", "unit", "object", "index", "source", "target", "query", "ask", "tell", "report", "answer", "where", "which", "what"}
    pairs = [f"{tokens[index]}_{tokens[index + 1]}" for index in range(max(0, len(tokens) - 1)) if tokens[index] in role_tokens or tokens[index + 1] in role_tokens]
    return tokens + pairs


def build_segment_vocab(texts: list[str]) -> dict[str, int]:
    tokens = sorted({token for text in texts for token in segment_feature_tokens(text)})
    return {token: index for index, token in enumerate(tokens)}


def segment_features(texts: list[str], vocab: dict[str, int], blocked_tokens: set[str] | None = None) -> np.ndarray:
    blocked = blocked_tokens or set()
    values = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_index, text in enumerate(texts):
        for token in segment_feature_tokens(text):
            if token in blocked or any(part in blocked for part in token.split("_")):
                continue
            if token in vocab:
                values[row_index, int(vocab[token])] += 1.0
    return values


def event_binding_head_blocked_tokens(head: str) -> set[str]:
    if head == "color":
        return set(SHAPE_WORDS)
    if head == "shape":
        return set(COLOR_WORDS)
    return set()


def bag_features(prompts: list[str], vocab: dict[str, int]) -> np.ndarray:
    values = np.zeros((len(prompts), len(vocab)), dtype=np.float32)
    for row_index, prompt in enumerate(prompts):
        for token in tokenize_randomized_prompt(prompt):
            if token in vocab:
                values[row_index, int(vocab[token])] += 1.0
    return values


def evaluate_fields(fields: dict[str, int], record: dict[str, Any], caps: dict[str, int]) -> dict[str, float]:
    target_state = record["labels"]["state"]
    predicted_state = {
        "color": int(int(fields["address"]) // int(caps["n_shapes"])),
        "shape": int(int(fields["address"]) % int(caps["n_shapes"])),
        "pos": int(fields["residual"]),
        "vel": int(int(fields["schema"]) - int(caps["max_speed"])),
    }
    state_ok = float(int(all(int(predicted_state[key]) == int(target_state[key]) for key in predicted_state)))
    action_ok = float(int(int(fields["action"]) == int(record["labels"]["action"])))
    provenance_ok = float(int(int(fields["provenance"]) == int(record["model_input"]["query"]["commit_time"])))
    return {
        "joint_success": float(int(state_ok == 1.0 and action_ok == 1.0)),
        "state_success": state_ok,
        "action_success": action_ok,
        "provenance_success": provenance_ok,
        "color_success": float(int(predicted_state["color"] == int(target_state["color"]))),
        "shape_success": float(int(predicted_state["shape"] == int(target_state["shape"]))),
        "pos_success": float(int(predicted_state["pos"] == int(target_state["pos"]))),
        "vel_success": float(int(predicted_state["vel"] == int(target_state["vel"]))),
    }


def labels_for_records(records: list[dict[str, Any]], caps: dict[str, int]) -> dict[str, np.ndarray]:
    max_speed = int(caps["max_speed"])
    return {
        "color": np.asarray([int(row["labels"]["state"]["color"]) for row in records], dtype=np.int64),
        "shape": np.asarray([int(row["labels"]["state"]["shape"]) for row in records], dtype=np.int64),
        "pos": np.asarray([int(row["labels"]["state"]["pos"]) for row in records], dtype=np.int64),
        "vel": np.asarray([int(row["labels"]["state"]["vel"]) + max_speed for row in records], dtype=np.int64),
        "action": np.asarray([int(row["labels"]["action"]) for row in records], dtype=np.int64),
        "provenance": np.asarray([int(row["model_input"]["query"]["commit_time"]) for row in records], dtype=np.int64),
    }


def split_records(dataset: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in dataset if row["split"] == split]


def train_language_model(dataset: list[dict[str, Any]], profile: str, seed: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    caps = profile_caps(profile)
    torch.manual_seed(int(seed))
    train_records = split_records(dataset, "train")
    feature_fields = {key: np.stack([vectorize_message_fields(record_prompt(row), caps)[key] for row in train_records], axis=0) for key in ("color", "shape", "pos", "vel", "action", "provenance")}
    label_values = labels_for_records(train_records, caps)
    x_train = {key: torch.tensor(value, dtype=torch.float32) for key, value in feature_fields.items()}
    y_train = {key: torch.tensor(value, dtype=torch.long) for key, value in label_values.items()}
    def head(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, int(HEAD_RANK)), nn.ReLU(), nn.Linear(int(HEAD_RANK), output_dim))
    heads = nn.ModuleDict(
        {
            "color": head(int(feature_fields["color"].shape[1]), int(caps["n_colors"])),
            "shape": head(int(feature_fields["shape"].shape[1]), int(caps["n_shapes"])),
            "pos": head(int(feature_fields["pos"].shape[1]), int(caps["track_length"])),
            "vel": head(int(feature_fields["vel"].shape[1]), int(caps["max_speed"]) * 2 + 1),
            "action": head(int(feature_fields["action"].shape[1]), int(caps["action_count"])),
            "provenance": head(int(feature_fields["provenance"].shape[1]), int(caps["seq_len"])),
        }
    )
    optimizer = torch.optim.Adam(heads.parameters(), lr=0.05)
    losses = []
    for _ in range(int(EPOCHS)):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(functional.cross_entropy(heads[key](x_train[key]), y_train[key]) for key in heads)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "heads": heads,
        "caps": caps,
        "parameter_count": int(sum(parameter.numel() for parameter in heads.parameters())),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
    }


def predict(record: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    import torch

    caps = learned["caps"]
    prompt = record_prompt(record)
    field_features = vectorize_message_fields(prompt, caps)
    x_value = {key: torch.tensor(value[None, :], dtype=torch.float32) for key, value in field_features.items()}
    heads = learned["heads"]
    heads.eval()
    with torch.no_grad():
        logits = {key: head(x_value[key]) for key, head in heads.items()}
        pred = {key: int(value.argmax(dim=-1).cpu().item()) for key, value in logits.items()}
    predicted_state = {
        "color": int(pred["color"]),
        "shape": int(pred["shape"]),
        "pos": int(pred["pos"]),
        "vel": int(pred["vel"] - int(caps["max_speed"])),
    }
    computed_action = visible_source_action({"address": int(predicted_state["color"] * int(caps["n_shapes"]) + predicted_state["shape"]), "schema": int(predicted_state["vel"] + int(caps["max_speed"])), "residual": int(predicted_state["pos"]), "action": 0, "provenance": int(pred["provenance"])}, caps)
    target_state = record["labels"]["state"]
    state_ok = float(int(all(int(predicted_state[key]) == int(target_state[key]) for key in predicted_state)))
    action_ok = float(int(int(computed_action) == int(record["labels"]["action"])))
    provenance_ok = float(int(int(pred["provenance"]) == int(record["model_input"]["query"]["commit_time"])))
    return {
        "state_success": state_ok,
        "action_success": action_ok,
        "joint_success": float(int(state_ok == 1.0 and action_ok == 1.0)),
        "provenance_success": provenance_ok,
        "color_success": float(int(predicted_state["color"] == int(target_state["color"]))),
        "shape_success": float(int(predicted_state["shape"] == int(target_state["shape"]))),
        "pos_success": float(int(predicted_state["pos"] == int(target_state["pos"]))),
        "vel_success": float(int(predicted_state["vel"] == int(target_state["vel"]))),
        "response": f"answer action_{int(computed_action)} color_{int(predicted_state['color'])} shape_{int(predicted_state['shape'])} pos_{int(predicted_state['pos'])} vel_{int(predicted_state['vel'])}",
    }


def code_labels_for_records(records: list[dict[str, Any]], caps: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        "address": np.asarray([int(row["labels"]["state"]["color"]) * int(caps["n_shapes"]) + int(row["labels"]["state"]["shape"]) for row in records], dtype=np.int64),
        "schema": np.asarray([int(row["labels"]["state"]["vel"]) + int(caps["max_speed"]) for row in records], dtype=np.int64),
        "residual": np.asarray([int(row["labels"]["state"]["pos"]) for row in records], dtype=np.int64),
        "action": np.asarray([int(row["labels"]["action"]) for row in records], dtype=np.int64),
        "provenance": np.asarray([int(row["model_input"]["query"]["commit_time"]) for row in records], dtype=np.int64),
    }


def train_parser_resistant_model(train_records: list[dict[str, Any]], caps: dict[str, int], seed: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    prompts = randomized_prompts_for_records(train_records, seed)
    vocab = build_vocab(prompts)
    x_train = torch.tensor(bag_features(prompts, vocab), dtype=torch.float32)
    labels = code_labels_for_records(train_records, caps)
    y_train = {key: torch.tensor(value, dtype=torch.long) for key, value in labels.items()}
    torch.manual_seed(int(seed))
    state_dim = 12
    encoder = nn.Sequential(nn.Linear(int(x_train.shape[1]), state_dim), nn.Tanh())
    heads = nn.ModuleDict(
        {
            "address": nn.Linear(state_dim, int(caps["n_colors"]) * int(caps["n_shapes"])),
            "schema": nn.Linear(state_dim, int(caps["max_speed"]) * 2 + 1),
            "residual": nn.Linear(state_dim, int(caps["track_length"])),
            "action": nn.Linear(state_dim, int(caps["action_count"])),
            "provenance": nn.Linear(state_dim, int(caps["seq_len"])),
        }
    )
    modules = nn.ModuleDict({"encoder": encoder, "heads": heads})
    optimizer = torch.optim.Adam(modules.parameters(), lr=0.03)
    losses = []
    for _ in range(min(int(EPOCHS), 120)):
        optimizer.zero_grad(set_to_none=True)
        state = encoder(x_train)
        loss = sum(functional.cross_entropy(heads[key](state), y_train[key]) for key in heads)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "vocab": vocab,
        "encoder": encoder,
        "heads": heads,
        "parameter_count": int(sum(parameter.numel() for parameter in modules.parameters())),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
        "state_dim": state_dim,
    }


def predict_parser_resistant(records: list[dict[str, Any]], learned: dict[str, Any], caps: dict[str, int], seed: int, state_mode: str = "normal") -> list[dict[str, float]]:
    import torch

    prompts = randomized_prompts_for_records(records, seed)
    x_value = torch.tensor(bag_features(prompts, learned["vocab"]), dtype=torch.float32)
    encoder = learned["encoder"]
    heads = learned["heads"]
    encoder.eval()
    heads.eval()
    with torch.no_grad():
        state = encoder(x_value)
        if state_mode == "zero":
            state = torch.zeros_like(state)
        elif state_mode == "shuffle" and int(state.shape[0]) > 1:
            state = torch.roll(state, shifts=1, dims=0)
        predictions = {key: head(state).argmax(dim=-1).cpu().numpy().astype(int) for key, head in heads.items()}
    results = []
    for index, record in enumerate(records):
        fields = {
            "address": int(predictions["address"][index]),
            "schema": int(predictions["schema"][index]),
            "residual": int(predictions["residual"][index]),
            "action": int(predictions["action"][index]),
            "provenance": int(predictions["provenance"][index]),
        }
        results.append(evaluate_fields(fields, record, caps))
    return results


def parser_resistant_axis_summary(profile: str, axis: str, seed: int) -> dict[str, Any]:
    caps = profile_caps(profile)
    dataset = build_factor_heldout_distributed_dataset(profile, seed, TRAIN_EPISODES, VAL_EPISODES, TEST_EPISODES, key=axis)
    for row in dataset:
        row["factor_holdout_bucket"] = factor_holdout_bucket_for_key(row, caps, axis)
    train_records = split_records(dataset, "train")
    test_records = split_records(dataset, "test")
    learned = train_parser_resistant_model(train_records, caps, seed + 191)
    test_results = predict_parser_resistant(test_records, learned, caps, seed + 293, "normal")
    zero_results = predict_parser_resistant(test_records, learned, caps, seed + 293, "zero")
    shuffle_results = predict_parser_resistant(test_records, learned, caps, seed + 293, "shuffle")
    rule_results = [evaluate_fields(natural_rule_code_fields(randomized_record_prompt(row, seed + index * 104_729), caps), row, caps) for index, row in enumerate(test_records)]
    matched_sparse_results = [matched_budget_sparse_read_result(row, caps) for row in test_records]
    uncapped_sparse_results = [distributed_evidence_sparse_read_result(row, caps, max_records=32) for row in test_records]
    field_floor = min(
        mean_metric(test_results, "color_success"),
        mean_metric(test_results, "shape_success"),
        mean_metric(test_results, "pos_success"),
        mean_metric(test_results, "vel_success"),
        mean_metric(test_results, "provenance_success"),
    )
    return {
        "joint": mean_metric(test_results, "joint_success"),
        "state": mean_metric(test_results, "state_success"),
        "action": mean_metric(test_results, "action_success"),
        "field_floor": float(field_floor),
        "zero_joint": mean_metric(zero_results, "joint_success"),
        "shuffle_joint": mean_metric(shuffle_results, "joint_success"),
        "rule_joint": mean_metric(rule_results, "joint_success"),
        "matched_sparse_joint": mean_metric(matched_sparse_results, "joint_correct"),
        "uncapped_sparse_joint": mean_metric(uncapped_sparse_results, "joint_correct"),
        "parameter_count": float(learned["parameter_count"]),
        "learned_bits": float(learned_code_bits(caps)),
        "matched_sparse_bits": mean_metric(matched_sparse_results, "bits_committed"),
        "parser_bits": float(parser_schema_cost_bits(caps)),
        "prompt": randomized_record_prompt(test_records[0], seed + 293) if test_records else "",
        "train_loss_start": float(learned["train_loss_start"]),
        "train_loss_final": float(learned["train_loss_final"]),
    }


def code_fields_from_bound_events(events: list[dict[str, int]], focus: int, caps: dict[str, int]) -> dict[str, int]:
    focus_events = [event for event in events if int(event["object_index"]) == int(focus)]
    position_events = sorted([(int(event["time"]), int(event["pos"])) for event in focus_events if int(event["pos"]) >= 0])
    commit_time = position_events[len(position_events) // 2][0] if position_events else 0
    pos_at_commit = [pos for time_value, pos in position_events if int(time_value) == int(commit_time)]
    residual = int(pos_at_commit[0] if pos_at_commit else (position_events[-1][1] if position_events else 0))
    velocity = 0
    if len(position_events) >= 2:
        first_time, first_pos = position_events[0]
        last_time, last_pos = position_events[-1]
        velocity = int(round(float(last_pos - first_pos) / float(max(1, last_time - first_time))))
        velocity = int(max(-int(caps["max_speed"]), min(int(caps["max_speed"]), velocity)))
    color = int(first_known(focus_events, "color", 0))
    shape = int(first_known(focus_events, "shape", 0))
    fields = {
        "address": int(color * int(caps["n_shapes"]) + shape),
        "schema": int(velocity + int(caps["max_speed"])),
        "residual": int(residual),
        "action": 0,
        "provenance": int(commit_time),
    }
    fields["action"] = int(visible_source_action(fields, caps))
    return fields


def event_binding_code_fields(prompt: str, caps: dict[str, int]) -> dict[str, int]:
    events, focus = natural_events_from_prompt(prompt)
    return code_fields_from_bound_events(events, focus, caps)


def answer_event_binding_prompt(prompt: str, profile: str = "smoke") -> str:
    caps = profile_caps(profile)
    fields = event_binding_code_fields(prompt, caps)
    state = {
        "color": int(int(fields["address"]) // int(caps["n_shapes"])),
        "shape": int(int(fields["address"]) % int(caps["n_shapes"])),
        "pos": int(fields["residual"]),
        "vel": int(int(fields["schema"]) - int(caps["max_speed"])),
    }
    return f"answer action_{int(fields['action'])} color_{int(state['color'])} shape_{int(state['shape'])} pos_{int(state['pos'])} vel_{int(state['vel'])}"


def event_binding_results(records: list[dict[str, Any]], caps: dict[str, int], seed: int, state_mode: str = "normal") -> list[dict[str, float]]:
    prompts = randomized_prompts_for_records(records, seed)
    fields = [event_binding_code_fields(prompt, caps) for prompt in prompts]
    if state_mode == "zero":
        fields = [
            {
                "address": 0,
                "schema": int(caps["max_speed"]),
                "residual": 0,
                "action": 0,
                "provenance": 0,
            }
            for _ in fields
        ]
    elif state_mode == "shuffle" and len(fields) > 1:
        fields = fields[-1:] + fields[:-1]
    return [evaluate_fields(fields[index], record, caps) for index, record in enumerate(records)]


def event_binding_training_tables(records: list[dict[str, Any]], seed: int, caps: dict[str, int]) -> dict[str, Any]:
    segment_texts: list[str] = []
    event_labels: list[int] = []
    slot_labels: list[int] = []
    time_labels: list[int] = []
    color_labels: list[int] = []
    shape_labels: list[int] = []
    pos_labels: list[int] = []
    query_texts: list[str] = []
    query_labels: list[int] = []
    for index, record in enumerate(records):
        segments, query = randomized_record_parts(record, seed + index * 104_729)
        query_texts.append(query)
        query_labels.append(int(record["model_input"]["query"]["focus_local_index"]))
        for text, event in segments:
            segment_texts.append(text)
            event_labels.append(int(event is not None))
            if event is None:
                slot_labels.append(0)
                time_labels.append(0)
                color_labels.append(int(caps["n_colors"]))
                shape_labels.append(int(caps["n_shapes"]))
                pos_labels.append(int(caps["track_length"]))
            else:
                slot_labels.append(int(event["object_index"]))
                time_labels.append(int(event["time"]))
                color_labels.append(int(event["color"]) if int(event["color"]) >= 0 else int(caps["n_colors"]))
                shape_labels.append(int(event["shape"]) if int(event["shape"]) >= 0 else int(caps["n_shapes"]))
                pos_labels.append(int(event["pos"]) if int(event["pos"]) >= 0 else int(caps["track_length"]))
    return {
        "segment_texts": segment_texts,
        "event_labels": np.asarray(event_labels, dtype=np.int64),
        "slot_labels": np.asarray(slot_labels, dtype=np.int64),
        "time_labels": np.asarray(time_labels, dtype=np.int64),
        "color_labels": np.asarray(color_labels, dtype=np.int64),
        "shape_labels": np.asarray(shape_labels, dtype=np.int64),
        "pos_labels": np.asarray(pos_labels, dtype=np.int64),
        "query_texts": query_texts,
        "query_labels": np.asarray(query_labels, dtype=np.int64),
    }


def train_event_binding_segment_model(records: list[dict[str, Any]], caps: dict[str, int], seed: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    tables = event_binding_training_tables(records, seed, caps)
    vocab = build_segment_vocab(list(tables["segment_texts"]) + list(tables["query_texts"]))
    segment_inputs = {key: torch.tensor(segment_features(tables["segment_texts"], vocab, event_binding_head_blocked_tokens(key)), dtype=torch.float32) for key in ("event", "slot", "time", "color", "shape", "pos")}
    x_queries = torch.tensor(segment_features(tables["query_texts"], vocab), dtype=torch.float32)
    labels = {key: torch.tensor(tables[f"{key}_labels"], dtype=torch.long) for key in ("event", "slot", "time", "color", "shape", "pos")}
    y_query = torch.tensor(tables["query_labels"], dtype=torch.long)
    torch.manual_seed(int(seed))
    feature_count = int(len(vocab))
    heads = nn.ModuleDict(
        {
            "event": nn.Linear(feature_count, 2),
            "slot": nn.Linear(feature_count, int(caps["n_active"])),
            "time": nn.Linear(feature_count, int(caps["seq_len"])),
            "color": nn.Linear(feature_count, int(caps["n_colors"]) + 1),
            "shape": nn.Linear(feature_count, int(caps["n_shapes"]) + 1),
            "pos": nn.Linear(feature_count, int(caps["track_length"]) + 1),
            "query": nn.Linear(feature_count, int(caps["n_active"])),
        }
    )
    optimizer = torch.optim.Adam(heads.parameters(), lr=0.04)
    losses = []
    event_mask = labels["event"] == 1
    for _ in range(min(int(EPOCHS), 120)):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(heads["event"](segment_inputs["event"]), labels["event"])
        loss = loss + sum(functional.cross_entropy(heads[key](segment_inputs[key][event_mask]), labels[key][event_mask]) for key in ("slot", "time", "color", "shape", "pos"))
        loss = loss + functional.cross_entropy(heads["query"](x_queries), y_query)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "vocab": vocab,
        "heads": heads,
        "parameter_count": int(sum(parameter.numel() for parameter in heads.parameters())),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
    }


def predict_trainable_event_binding(records: list[dict[str, Any]], learned: dict[str, Any], caps: dict[str, int], seed: int, state_mode: str = "normal") -> list[dict[str, float]]:
    import torch

    prompts = randomized_prompts_for_records(records, seed)
    segment_groups: list[list[str]] = []
    query_texts: list[str] = []
    flat_segments: list[str] = []
    for prompt in prompts:
        parts = [segment.strip() for segment in prompt.split("|") if segment.strip()]
        event_segments = parts[:-1]
        segment_groups.append(event_segments)
        flat_segments.extend(event_segments)
        query_texts.append(parts[-1] if parts else "")
    segment_inputs = {key: torch.tensor(segment_features(flat_segments, learned["vocab"], event_binding_head_blocked_tokens(key)), dtype=torch.float32) for key in ("event", "slot", "time", "color", "shape", "pos")}
    x_queries = torch.tensor(segment_features(query_texts, learned["vocab"]), dtype=torch.float32)
    heads = learned["heads"]
    heads.eval()
    with torch.no_grad():
        if flat_segments:
            pred = {key: heads[key](segment_inputs[key]).argmax(dim=-1).cpu().numpy().astype(int) for key in ("event", "slot", "time", "color", "shape", "pos")}
        else:
            pred = {key: np.asarray([], dtype=int) for key in ("event", "slot", "time", "color", "shape", "pos")}
        query_pred = heads["query"](x_queries).argmax(dim=-1).cpu().numpy().astype(int) if query_texts else np.asarray([], dtype=int)
    fields_list = []
    cursor = 0
    for record_index, segments in enumerate(segment_groups):
        events = []
        for _ in segments:
            if int(pred["event"][cursor]) == 1:
                events.append(
                    {
                        "time": int(pred["time"][cursor]),
                        "object_index": int(pred["slot"][cursor]),
                        "color": int(pred["color"][cursor]) if int(pred["color"][cursor]) < int(caps["n_colors"]) else -1,
                        "shape": int(pred["shape"][cursor]) if int(pred["shape"][cursor]) < int(caps["n_shapes"]) else -1,
                        "pos": int(pred["pos"][cursor]) if int(pred["pos"][cursor]) < int(caps["track_length"]) else -1,
                        "observed": 1,
                    }
                )
            cursor += 1
        fields_list.append(code_fields_from_bound_events(events, int(query_pred[record_index]), caps))
    if state_mode == "zero":
        fields_list = [{"address": 0, "schema": int(caps["max_speed"]), "residual": 0, "action": 0, "provenance": 0} for _ in fields_list]
    elif state_mode == "shuffle" and len(fields_list) > 1:
        fields_list = fields_list[-1:] + fields_list[:-1]
    return [evaluate_fields(fields_list[index], record, caps) for index, record in enumerate(records)]


def event_binding_axis_summary(profile: str, axis: str, seed: int) -> dict[str, Any]:
    caps = profile_caps(profile)
    dataset = build_factor_heldout_distributed_dataset(profile, seed, TRAIN_EPISODES, VAL_EPISODES, TEST_EPISODES, key=axis)
    for row in dataset:
        row["factor_holdout_bucket"] = factor_holdout_bucket_for_key(row, caps, axis)
    train_records = split_records(dataset, "train")
    test_records = split_records(dataset, "test")
    learned = train_event_binding_segment_model(train_records, caps, seed + 503)
    test_results = event_binding_results(test_records, caps, seed + 401, "normal")
    zero_results = event_binding_results(test_records, caps, seed + 401, "zero")
    shuffle_results = event_binding_results(test_records, caps, seed + 401, "shuffle")
    trainable_results = predict_trainable_event_binding(test_records, learned, caps, seed + 401, "normal")
    trainable_zero_results = predict_trainable_event_binding(test_records, learned, caps, seed + 401, "zero")
    trainable_shuffle_results = predict_trainable_event_binding(test_records, learned, caps, seed + 401, "shuffle")
    matched_sparse_results = [matched_budget_sparse_read_result(row, caps) for row in test_records]
    uncapped_sparse_results = [distributed_evidence_sparse_read_result(row, caps, max_records=32) for row in test_records]
    field_floor = min(
        mean_metric(test_results, "color_success"),
        mean_metric(test_results, "shape_success"),
        mean_metric(test_results, "pos_success"),
        mean_metric(test_results, "vel_success"),
        mean_metric(test_results, "provenance_success"),
    )
    trainable_field_floor = min(
        mean_metric(trainable_results, "color_success"),
        mean_metric(trainable_results, "shape_success"),
        mean_metric(trainable_results, "pos_success"),
        mean_metric(trainable_results, "vel_success"),
        mean_metric(trainable_results, "provenance_success"),
    )
    return {
        "joint": mean_metric(test_results, "joint_success"),
        "state": mean_metric(test_results, "state_success"),
        "action": mean_metric(test_results, "action_success"),
        "field_floor": float(field_floor),
        "zero_joint": mean_metric(zero_results, "joint_success"),
        "shuffle_joint": mean_metric(shuffle_results, "joint_success"),
        "trainable_joint": mean_metric(trainable_results, "joint_success"),
        "trainable_state": mean_metric(trainable_results, "state_success"),
        "trainable_action": mean_metric(trainable_results, "action_success"),
        "trainable_field_floor": float(trainable_field_floor),
        "trainable_zero_joint": mean_metric(trainable_zero_results, "joint_success"),
        "trainable_shuffle_joint": mean_metric(trainable_shuffle_results, "joint_success"),
        "trainable_parameter_count": float(learned["parameter_count"]),
        "trainable_loss_start": float(learned["train_loss_start"]),
        "trainable_loss_final": float(learned["train_loss_final"]),
        "matched_sparse_joint": mean_metric(matched_sparse_results, "joint_correct"),
        "uncapped_sparse_joint": mean_metric(uncapped_sparse_results, "joint_correct"),
        "rule_cost_score": float(parser_schema_cost_bits(caps) * 17),
        "rule_bits": float(parser_schema_cost_bits(caps)),
        "accounted_bits": float(parser_schema_cost_bits(caps) + learned_code_bits(caps)),
        "learned_bits": float(learned_code_bits(caps)),
        "matched_sparse_bits": mean_metric(matched_sparse_results, "bits_committed"),
        "prompt": randomized_record_prompt(test_records[0], seed + 401) if test_records else "",
        "response": answer_event_binding_prompt(randomized_record_prompt(test_records[0], seed + 401), profile) if test_records else "",
    }


def mean_metric(results: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in results])) if results else 0.0


def bucket_sets(dataset: list[dict[str, Any]]) -> tuple[set[int], set[int], set[int]]:
    return (
        {int(row["factor_holdout_bucket"]) for row in dataset if row["split"] == "train"},
        {int(row["factor_holdout_bucket"]) for row in dataset if row["split"] == "validation"},
        {int(row["factor_holdout_bucket"]) for row in dataset if row["split"] == "test"},
    )


def axis_summary(profile: str, axis: str, seed: int) -> dict[str, Any]:
    caps = profile_caps(profile)
    dataset = build_factor_heldout_distributed_dataset(profile, seed, TRAIN_EPISODES, VAL_EPISODES, TEST_EPISODES, key=axis)
    for row in dataset:
        row["factor_holdout_bucket"] = factor_holdout_bucket_for_key(row, caps, axis)
    learned = train_language_model(dataset, profile, seed + 17)
    test_records = split_records(dataset, "test")
    results = [predict(row, learned) for row in test_records]
    matched_sparse_results = [matched_budget_sparse_read_result(row, caps) for row in test_records]
    train_buckets, validation_buckets, test_buckets = bucket_sets(dataset)
    field_floor = min(
        mean_metric(results, "color_success"),
        mean_metric(results, "shape_success"),
        mean_metric(results, "pos_success"),
        mean_metric(results, "vel_success"),
        mean_metric(results, "provenance_success"),
    )
    return {
        "parameter_count": float(learned["parameter_count"]),
        "joint": mean_metric(results, "joint_success"),
        "state": mean_metric(results, "state_success"),
        "action": mean_metric(results, "action_success"),
        "field_floor": float(field_floor),
        "bucket_clean": float(int(len(train_buckets & validation_buckets) == 0 and len(train_buckets & test_buckets) == 0 and len(validation_buckets & test_buckets) == 0)),
        "learned_bits": float(learned_code_bits(caps)),
        "matched_sparse_bits": mean_metric(matched_sparse_results, "bits_committed"),
        "matched_sparse_joint": mean_metric(matched_sparse_results, "joint_correct"),
        "prompt": record_prompt(test_records[0]) if test_records else "",
        "response": results[0]["response"] if results else "",
        "train_loss_start": float(learned["train_loss_start"]),
        "train_loss_final": float(learned["train_loss_final"]),
    }


def build_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    runs = []
    for axis_index, axis in enumerate(AXES):
        for seed_index in range(int(SEED_COUNT)):
            runs.append(axis_summary(profile, axis, seed + axis_index * 10_003 + seed_index * 1_009))
    joints = [float(row["joint"]) for row in runs]
    states = [float(row["state"]) for row in runs]
    actions = [float(row["action"]) for row in runs]
    field_floors = [float(row["field_floor"]) for row in runs]
    params = [float(row["parameter_count"]) for row in runs]
    learned_bits = [float(row["learned_bits"]) for row in runs]
    sparse_bits = [float(row["matched_sparse_bits"]) for row in runs]
    sparse_joint = [float(row["matched_sparse_joint"]) for row in runs]
    useful_density = [joints[index] / max(learned_bits[index], 1e-9) for index in range(len(runs))]
    sparse_density = [sparse_joint[index] / max(sparse_bits[index], 1e-9) for index in range(len(runs))]
    return {
        "language_grounded_local_model_authorized": 1.0,
        "language_grounded_full_model_authorized": 0.0,
        "language_grounded_paid_compute_authorized": 0.0,
        "language_grounded_arbitrary_chat_authorized": 0.0,
        "language_grounded_constrained_message_response_supported": 1.0,
        "language_grounded_axis_count": int(len(AXES)),
        "language_grounded_seed_count": int(SEED_COUNT),
        "language_grounded_run_count": int(len(runs)),
        "language_grounded_total_train_record_count": int(len(runs) * int(TRAIN_EPISODES)),
        "language_grounded_total_validation_record_count": int(len(runs) * int(VAL_EPISODES)),
        "language_grounded_total_test_record_count": int(len(runs) * int(TEST_EPISODES)),
        "language_grounded_parameter_count_max": float(max(params) if params else 0.0),
        "language_grounded_parameter_count_mean": float(np.mean(params)) if params else 0.0,
        "language_grounded_test_joint_success_min": float(min(joints) if joints else 0.0),
        "language_grounded_test_state_success_min": float(min(states) if states else 0.0),
        "language_grounded_test_action_success_min": float(min(actions) if actions else 0.0),
        "language_grounded_field_accuracy_floor": float(min(field_floors) if field_floors else 0.0),
        "language_grounded_matched_sparse_joint_success_max": float(max(sparse_joint) if sparse_joint else 0.0),
        "language_grounded_learned_committed_bits_max": float(max(learned_bits) if learned_bits else 0.0),
        "language_grounded_matched_sparse_bits_min": float(min(sparse_bits) if sparse_bits else 0.0),
        "language_grounded_useful_operation_success_per_committed_bit_min": float(min(useful_density) if useful_density else 0.0),
        "language_grounded_matched_sparse_operation_success_per_committed_bit_max": float(max(sparse_density) if sparse_density else 0.0),
        "language_grounded_useful_state_density_advantage_min": float(min([useful_density[index] - sparse_density[index] for index in range(len(useful_density))]) if useful_density else 0.0),
        "language_grounded_bucket_clean_rate": float(np.mean([float(row["bucket_clean"]) for row in runs])) if runs else 0.0,
        "language_grounded_engineering_pass": float(int(runs and min(joints) >= 0.95 and min(field_floors) >= 0.95 and max(params) < 10_000 and max(sparse_joint) == 0.0)),
        "language_grounded_example_prompt": str(runs[0]["prompt"]) if runs else "",
        "language_grounded_example_response": str(runs[0]["response"]) if runs else "",
        "language_grounded_train_loss_start_mean": float(np.mean([float(row["train_loss_start"]) for row in runs])) if runs else 0.0,
        "language_grounded_train_loss_final_mean": float(np.mean([float(row["train_loss_final"]) for row in runs])) if runs else 0.0,
    }


def build_parser_resistant_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    runs = []
    for axis_index, axis in enumerate(AXES):
        for seed_index in range(int(SEED_COUNT)):
            runs.append(parser_resistant_axis_summary(profile, axis, seed + axis_index * 10_003 + seed_index * 1_009))
    joints = [float(row["joint"]) for row in runs]
    states = [float(row["state"]) for row in runs]
    actions = [float(row["action"]) for row in runs]
    field_floors = [float(row["field_floor"]) for row in runs]
    zero_joints = [float(row["zero_joint"]) for row in runs]
    shuffle_joints = [float(row["shuffle_joint"]) for row in runs]
    rule_joints = [float(row["rule_joint"]) for row in runs]
    matched_sparse = [float(row["matched_sparse_joint"]) for row in runs]
    uncapped_sparse = [float(row["uncapped_sparse_joint"]) for row in runs]
    params = [float(row["parameter_count"]) for row in runs]
    learned_bits = [float(row["learned_bits"]) for row in runs]
    sparse_bits = [float(row["matched_sparse_bits"]) for row in runs]
    useful_density = [joints[index] / max(learned_bits[index], 1e-9) for index in range(len(runs))]
    sparse_density = [matched_sparse[index] / max(sparse_bits[index], 1e-9) for index in range(len(runs))]
    engineering_pass = float(int(runs and min(joints) >= 0.95 and min(field_floors) >= 0.95 and max(matched_sparse) == 0.0 and max(zero_joints) < min(joints) and max(shuffle_joints) < min(joints)))
    return {
        "parser_resistant_gate_evaluated": 1.0,
        "parser_resistant_local_model_authorized": 1.0,
        "parser_resistant_full_model_authorized": 0.0,
        "parser_resistant_paid_compute_authorized": 0.0,
        "parser_resistant_arbitrary_chat_authorized": 0.0,
        "parser_resistant_template_family_count": float(EVENT_TEMPLATE_COUNT),
        "parser_resistant_query_template_family_count": float(QUERY_TEMPLATE_COUNT),
        "parser_resistant_prefix_dependency_removed": 1.0,
        "parser_resistant_deterministic_parser_reported": 1.0,
        "parser_resistant_learned_text_encoder_reported": 1.0,
        "parser_resistant_local_state_ablation_reported": 1.0,
        "parser_resistant_axis_count": int(len(AXES)),
        "parser_resistant_seed_count": int(SEED_COUNT),
        "parser_resistant_run_count": int(len(runs)),
        "parser_resistant_total_train_record_count": int(len(runs) * int(TRAIN_EPISODES)),
        "parser_resistant_total_test_record_count": int(len(runs) * int(TEST_EPISODES)),
        "parser_resistant_parameter_count_max": float(max(params) if params else 0.0),
        "parser_resistant_test_joint_success_min": float(min(joints) if joints else 0.0),
        "parser_resistant_test_state_success_min": float(min(states) if states else 0.0),
        "parser_resistant_test_action_success_min": float(min(actions) if actions else 0.0),
        "parser_resistant_field_accuracy_floor": float(min(field_floors) if field_floors else 0.0),
        "parser_resistant_zero_state_joint_success_max": float(max(zero_joints) if zero_joints else 0.0),
        "parser_resistant_state_shuffle_joint_success_max": float(max(shuffle_joints) if shuffle_joints else 0.0),
        "parser_resistant_rule_extractor_joint_success_min": float(min(rule_joints) if rule_joints else 0.0),
        "parser_resistant_matched_sparse_joint_success_max": float(max(matched_sparse) if matched_sparse else 0.0),
        "parser_resistant_uncapped_sparse_joint_success_min": float(min(uncapped_sparse) if uncapped_sparse else 0.0),
        "parser_resistant_learned_committed_bits_max": float(max(learned_bits) if learned_bits else 0.0),
        "parser_resistant_matched_sparse_bits_min": float(min(sparse_bits) if sparse_bits else 0.0),
        "parser_resistant_parser_schema_cost_bits": float(max([float(row["parser_bits"]) for row in runs]) if runs else 0.0),
        "parser_resistant_useful_operation_success_per_committed_bit_min": float(min(useful_density) if useful_density else 0.0),
        "parser_resistant_matched_sparse_operation_success_per_committed_bit_max": float(max(sparse_density) if sparse_density else 0.0),
        "parser_resistant_useful_state_density_advantage_min": float(min([useful_density[index] - sparse_density[index] for index in range(len(useful_density))]) if useful_density else 0.0),
        "parser_resistant_engineering_pass": float(engineering_pass),
        "parser_resistant_claim_downgraded_to_structured_bridge": float(int(engineering_pass == 0.0)),
        "parser_resistant_example_prompt": str(runs[0]["prompt"]) if runs else "",
        "parser_resistant_train_loss_start_mean": float(np.mean([float(row["train_loss_start"]) for row in runs])) if runs else 0.0,
        "parser_resistant_train_loss_final_mean": float(np.mean([float(row["train_loss_final"]) for row in runs])) if runs else 0.0,
    }


def build_event_binding_foundation_summary(profile: str, seed: int = SEED) -> dict[str, Any]:
    runs = []
    for axis_index, axis in enumerate(AXES):
        for seed_index in range(int(SEED_COUNT)):
            runs.append(event_binding_axis_summary(profile, axis, seed + axis_index * 10_003 + seed_index * 1_009))
    joints = [float(row["joint"]) for row in runs]
    states = [float(row["state"]) for row in runs]
    actions = [float(row["action"]) for row in runs]
    field_floors = [float(row["field_floor"]) for row in runs]
    zero_joints = [float(row["zero_joint"]) for row in runs]
    shuffle_joints = [float(row["shuffle_joint"]) for row in runs]
    trainable_joints = [float(row["trainable_joint"]) for row in runs]
    trainable_states = [float(row["trainable_state"]) for row in runs]
    trainable_actions = [float(row["trainable_action"]) for row in runs]
    trainable_field_floors = [float(row["trainable_field_floor"]) for row in runs]
    trainable_zero_joints = [float(row["trainable_zero_joint"]) for row in runs]
    trainable_shuffle_joints = [float(row["trainable_shuffle_joint"]) for row in runs]
    trainable_parameter_counts = [float(row["trainable_parameter_count"]) for row in runs]
    trainable_loss_starts = [float(row["trainable_loss_start"]) for row in runs]
    trainable_loss_finals = [float(row["trainable_loss_final"]) for row in runs]
    matched_sparse = [float(row["matched_sparse_joint"]) for row in runs]
    uncapped_sparse = [float(row["uncapped_sparse_joint"]) for row in runs]
    rule_scores = [float(row["rule_cost_score"]) for row in runs]
    rule_bits = [float(row["rule_bits"]) for row in runs]
    accounted_bits = [float(row["accounted_bits"]) for row in runs]
    committed_bits = [float(row["learned_bits"]) for row in runs]
    sparse_bits = [float(row["matched_sparse_bits"]) for row in runs]
    useful_density = [joints[index] / max(accounted_bits[index], 1e-9) for index in range(len(runs))]
    trainable_useful_density = [trainable_joints[index] / max(accounted_bits[index], 1e-9) for index in range(len(runs))]
    sparse_density = [matched_sparse[index] / max(sparse_bits[index], 1e-9) for index in range(len(runs))]
    baseline_pass = float(int(runs and min(joints) >= 0.95 and min(field_floors) >= 0.95 and max(matched_sparse) == 0.0 and max(zero_joints) < min(joints) and max(shuffle_joints) < min(joints) and max(rule_scores) < 10_000))
    engineering_pass = float(int(runs and baseline_pass == 1.0 and min(trainable_joints) >= 0.95 and min(trainable_states) >= 0.95 and min(trainable_actions) >= 0.95 and min(trainable_field_floors) >= 0.95 and max(trainable_zero_joints) < min(trainable_joints) and max(trainable_shuffle_joints) < min(trainable_joints) and max(trainable_parameter_counts) < 10_000))
    return {
        "event_binding_foundation_evaluated": 1.0,
        "event_binding_parser_baseline_reported": 1.0,
        "event_binding_trainable_encoder_reported": 1.0,
        "event_binding_local_mechanism_authorized": 1.0,
        "event_binding_full_model_authorized": 0.0,
        "event_binding_paid_compute_authorized": 0.0,
        "event_binding_arbitrary_chat_authorized": 0.0,
        "event_binding_prefix_dependency_removed": 1.0,
        "event_binding_template_family_count": float(EVENT_TEMPLATE_COUNT),
        "event_binding_query_template_family_count": float(QUERY_TEMPLATE_COUNT),
        "event_binding_axis_count": int(len(AXES)),
        "event_binding_seed_count": int(SEED_COUNT),
        "event_binding_run_count": int(len(runs)),
        "event_binding_total_train_record_count": int(len(runs) * int(TRAIN_EPISODES)),
        "event_binding_total_validation_record_count": int(len(runs) * int(VAL_EPISODES)),
        "event_binding_total_test_record_count": int(len(runs) * int(TEST_EPISODES)),
        "event_binding_rule_cost_score_max": float(max(rule_scores) if rule_scores else 0.0),
        "event_binding_test_joint_success_min": float(min(joints) if joints else 0.0),
        "event_binding_test_state_success_min": float(min(states) if states else 0.0),
        "event_binding_test_action_success_min": float(min(actions) if actions else 0.0),
        "event_binding_field_accuracy_floor": float(min(field_floors) if field_floors else 0.0),
        "event_binding_trainable_segment_joint_success_min": float(min(trainable_joints) if trainable_joints else 0.0),
        "event_binding_trainable_segment_state_success_min": float(min(trainable_states) if trainable_states else 0.0),
        "event_binding_trainable_segment_action_success_min": float(min(trainable_actions) if trainable_actions else 0.0),
        "event_binding_trainable_segment_field_accuracy_floor": float(min(trainable_field_floors) if trainable_field_floors else 0.0),
        "event_binding_trainable_segment_zero_state_joint_success_max": float(max(trainable_zero_joints) if trainable_zero_joints else 0.0),
        "event_binding_trainable_segment_shuffle_joint_success_max": float(max(trainable_shuffle_joints) if trainable_shuffle_joints else 0.0),
        "event_binding_trainable_segment_parameter_count_max": float(max(trainable_parameter_counts) if trainable_parameter_counts else 0.0),
        "event_binding_trainable_loss_start_mean": float(np.mean(trainable_loss_starts)) if trainable_loss_starts else 0.0,
        "event_binding_trainable_loss_final_mean": float(np.mean(trainable_loss_finals)) if trainable_loss_finals else 0.0,
        "event_binding_zero_state_joint_success_max": float(max(zero_joints) if zero_joints else 0.0),
        "event_binding_state_shuffle_joint_success_max": float(max(shuffle_joints) if shuffle_joints else 0.0),
        "event_binding_matched_sparse_joint_success_max": float(max(matched_sparse) if matched_sparse else 0.0),
        "event_binding_uncapped_sparse_joint_success_min": float(min(uncapped_sparse) if uncapped_sparse else 0.0),
        "event_binding_committed_bits_max": float(max(committed_bits) if committed_bits else 0.0),
        "event_binding_rule_schema_cost_bits": float(max(rule_bits) if rule_bits else 0.0),
        "event_binding_accounted_bits_max": float(max(accounted_bits) if accounted_bits else 0.0),
        "event_binding_matched_sparse_bits_min": float(min(sparse_bits) if sparse_bits else 0.0),
        "event_binding_useful_operation_success_per_committed_bit_min": float(min(useful_density) if useful_density else 0.0),
        "event_binding_trainable_useful_operation_success_per_accounted_bit_min": float(min(trainable_useful_density) if trainable_useful_density else 0.0),
        "event_binding_matched_sparse_operation_success_per_committed_bit_max": float(max(sparse_density) if sparse_density else 0.0),
        "event_binding_useful_state_density_advantage_min": float(min([useful_density[index] - sparse_density[index] for index in range(len(useful_density))]) if useful_density else 0.0),
        "event_binding_trainable_useful_state_density_advantage_min": float(min([trainable_useful_density[index] - sparse_density[index] for index in range(len(trainable_useful_density))]) if trainable_useful_density else 0.0),
        "event_binding_parser_supported_foundation_pass": float(baseline_pass),
        "event_binding_engineering_pass": float(engineering_pass),
        "event_binding_claim_downgraded_to_parser_supported_foundation": float(1.0 - engineering_pass),
        "event_binding_example_prompt": str(runs[0]["prompt"]) if runs else "",
        "event_binding_example_response": str(runs[0]["response"]) if runs else "",
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    summary.update(build_parser_resistant_summary(profile))
    summary.update(build_event_binding_foundation_summary(profile))
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "language_grounded_state_density_mirror_metrics.json"
    record = build_run_record(
        simulation_name="language_grounded_state_density_mirror",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=utc_now_iso(),
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "train_episodes": int(TRAIN_EPISODES),
            "validation_episodes": int(VAL_EPISODES),
            "test_episodes": int(TEST_EPISODES),
            "epochs": int(EPOCHS),
            "seed_count": int(SEED_COUNT),
            "head_rank": int(HEAD_RANK),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["language_grounded_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        artifacts=[{"path": metrics_path.as_posix(), "type": "metrics"}],
        warnings=["local symbolic language bridge only; parser-resistant gate may demote the bridge; not arbitrary chat and not solved compression"],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
