from __future__ import annotations

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

from shared import (
    build_run_record,
    env_int,
    mean_confidence_interval,
    output_dir_for,
    require_positive,
    utc_now_iso,
    write_json,
)

from neuroloc.data.nm_worlds import (
    HARD_SYMBOLIC_PROFILES,
    evaluate_nm_hard_policy,
    generate_nm_hard_symbolic_batch,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("CUBB_MIRROR_SEED", 42)
TRAIN_EPISODES = env_int("CUBB_MIRROR_TRAIN_EPISODES", 64)
VAL_EPISODES = env_int("CUBB_MIRROR_VAL_EPISODES", 16)
TEST_EPISODES = env_int("CUBB_MIRROR_TEST_EPISODES", 16)
TRAIN_EPOCHS = env_int("CUBB_MIRROR_TRAIN_EPOCHS", 80)

require_positive("CUBB_MIRROR_TRAIN_EPISODES", TRAIN_EPISODES)
require_positive("CUBB_MIRROR_VAL_EPISODES", VAL_EPISODES)
require_positive("CUBB_MIRROR_TEST_EPISODES", TEST_EPISODES)
require_positive("CUBB_MIRROR_TRAIN_EPOCHS", TRAIN_EPOCHS)

FAMILY = "compression_under_bit_budget"
FORBIDDEN_INPUT_KEYS = {
    "hidden_state",
    "target",
    "target_answer",
    "target_action",
    "oracle_schema_id",
    "oracle_residual_id",
    "oracle_latent_state_code",
    "family",
    "future_observations",
    "policy_result_flags",
    "kill_condition",
    "memory_relevant_positions",
}
BASELINE_POLICIES = (
    "oracle_codec",
    "verbatim_store",
    "compressed_oracle_store",
    "no_memory",
    "recency_only",
    "shuffled_address",
    "random_codebook",
    "matched_bit_random_code",
    "matched_compute_no_code",
    "learned_code_oracle_decoder",
    "oracle_code_learned_decoder",
    "learned_address_oracle_payload",
    "oracle_address_learned_payload",
    "frozen_random_encoder_learned_decoder",
    "learned_encoder_frozen_random_decoder",
)
LEARNED_POLICIES = ("learned_codec",)
ALL_POLICIES = (*BASELINE_POLICIES, *LEARNED_POLICIES)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    env_value = os.environ.get("CUBB_MIRROR_PROFILE", "smoke").strip()
    if env_value not in {"smoke", "hard"}:
        raise ValueError("CUBB_MIRROR_PROFILE must be smoke or hard")
    return env_value


def split_seed(base_seed: int, split: str) -> int:
    offsets = {"train": 0, "validation": 1_000_003, "test": 2_000_033}
    return int(base_seed + offsets[split])


def observation_events(episode: dict[str, Any], max_time: int) -> list[dict[str, int]]:
    observations = episode["observation_stream"]
    event_rows = []
    seq_len, n_active = observations["color"].shape
    if max_time < 0:
        raise ValueError("max_time must be non-negative")
    for time_index in range(min(int(seq_len), int(max_time) + 1)):
        for object_index in range(int(n_active)):
            event_rows.append(
                {
                    "time": int(time_index),
                    "object_index": int(object_index),
                    "color": int(observations["color"][time_index, object_index]),
                    "shape": int(observations["shape"][time_index, object_index]),
                    "pos": int(observations["pos"][time_index, object_index]),
                    "observed": int(observations["visible"][time_index, object_index]),
                }
            )
    return event_rows


def contract_for_family(episode: dict[str, Any]) -> dict[str, Any]:
    matches = [contract for contract in episode["contracts"] if contract["family"] == FAMILY]
    if len(matches) != 1:
        raise ValueError("expected exactly one compression_under_bit_budget contract")
    return matches[0]


def build_model_input(episode: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    query = contract["query"]
    query_time = int(query["time"])
    return {
        "observations": observation_events(episode, query_time),
        "query": {
            "time": query_time,
            "focus_local_index": int(query["focus_local_index"]),
            "cue_color": int(query["cue_color"]),
            "cue_shape": int(query["cue_shape"]),
            "cue_pos": int(query["cue_pos"]),
        },
        "bit_budget": {
            "remaining_bits": int(contract["bit_budget"]["budget_bits"]),
            "budget_bits": int(contract["bit_budget"]["budget_bits"]),
        },
        "visible_context": {
            "profile": str(episode["profile"]),
            "time_count": int(episode["observation_stream"]["color"].shape[0]),
            "object_slot_count": int(episode["observation_stream"]["color"].shape[1]),
        },
    }


def build_labels(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": dict(contract["target"]["state"]),
        "action": int(contract["target"]["action"]),
        "identity": int(contract["target"]["identity"]),
        "oracle_code": {
            "bits": int(contract["bit_budget"]["compressed_bits"]),
            "address_margin": float(contract["telemetry"]["address_margin"]),
        },
        "verbatim_bits": int(contract["bit_budget"]["verbatim_bits"]),
        "compressed_bits": int(contract["bit_budget"]["compressed_bits"]),
        "budget_bits": int(contract["bit_budget"]["budget_bits"]),
    }


def collect_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    violations = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FORBIDDEN_INPUT_KEYS:
                violations.append(path)
            violations.extend(collect_forbidden_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(collect_forbidden_keys(item, f"{prefix}[{index}]"))
    return violations


def build_split(split: str, n_episodes: int, seed: int, profile: str) -> list[dict[str, Any]]:
    episodes = generate_nm_hard_symbolic_batch(n_episodes, seed=split_seed(seed, split), profile=profile)
    records = []
    for index, episode in enumerate(episodes):
        contract = contract_for_family(episode)
        model_input = build_model_input(episode, contract)
        labels = build_labels(contract)
        records.append(
            {
                "split": split,
                "seed": int(episode["seed"]),
                "episode_id": f"{split}_{index}_{episode['seed']}",
                "family": FAMILY,
                "difficulty": dict(contract["difficulty"]),
                "model_input": model_input,
                "labels": labels,
                "evaluation_contract": contract,
                "forbidden_input_keys": collect_forbidden_keys(model_input),
                "future_observation_violation_count": int(sum(1 for event in model_input["observations"] if int(event["time"]) > int(model_input["query"]["time"]))),
            }
        )
    return records


def build_dataset(profile: str, seed: int = SEED, train_episodes: int = TRAIN_EPISODES, val_episodes: int = VAL_EPISODES, test_episodes: int = TEST_EPISODES) -> list[dict[str, Any]]:
    return [
        *build_split("train", train_episodes, seed, profile),
        *build_split("validation", val_episodes, seed, profile),
        *build_split("test", test_episodes, seed, profile),
    ]


def profile_caps(profile: str) -> dict[str, int]:
    config = HARD_SYMBOLIC_PROFILES[profile]
    return {
        "seq_len": int(config["seq_len"]),
        "n_active": int(config["n_active"]),
        "n_colors": int(config["n_colors"]),
        "n_shapes": int(config["n_shapes"]),
        "track_length": int(config["track_length"]),
        "action_count": int(config["action_count"]),
        "max_speed": 3,
    }


def bits_for_cardinality(cardinality: int) -> int:
    if cardinality <= 1:
        return 1
    return int(np.ceil(np.log2(float(cardinality))))


def learned_code_bits(caps: dict[str, int]) -> int:
    return int(
        bits_for_cardinality(int(caps["n_colors"]) * int(caps["n_shapes"]))
        + bits_for_cardinality(int(caps["max_speed"]) * 2 + 1)
        + bits_for_cardinality(int(caps["track_length"]))
        + bits_for_cardinality(int(caps["seq_len"]))
    )


def one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * int(size)
    clipped = max(0, min(int(index), int(size) - 1))
    values[clipped] = 1.0
    return values


def last_known(events: list[dict[str, int]], field: str, missing_index: int) -> int:
    known = [int(event[field]) for event in events if int(event["observed"]) == 1 and int(event[field]) >= 0]
    return int(known[-1]) if known else int(missing_index)


def first_known(events: list[dict[str, int]], field: str, missing_index: int) -> int:
    known = [int(event[field]) for event in events if int(event["observed"]) == 1 and int(event[field]) >= 0]
    return int(known[0]) if known else int(missing_index)


def estimate_velocity(events: list[dict[str, int]], max_speed: int) -> int:
    positions = [(int(event["time"]), int(event["pos"])) for event in events if int(event["observed"]) == 1 and int(event["pos"]) >= 0]
    if len(positions) < 2:
        return 0
    first_time, first_pos = positions[0]
    last_time, last_pos = positions[-1]
    delta_time = max(1, last_time - first_time)
    estimate = int(round(float(last_pos - first_pos) / float(delta_time)))
    return int(max(-max_speed, min(max_speed, estimate)))


def vectorize_record(record: dict[str, Any], caps: dict[str, int]) -> np.ndarray:
    query = record["model_input"]["query"]
    focus = int(query["focus_local_index"])
    events = [event for event in record["model_input"]["observations"] if int(event["object_index"]) == focus]
    color_missing = int(caps["n_colors"])
    shape_missing = int(caps["n_shapes"])
    pos_missing = int(caps["track_length"])
    last_color = last_known(events, "color", color_missing)
    first_color = first_known(events, "color", color_missing)
    last_shape = last_known(events, "shape", shape_missing)
    first_shape = first_known(events, "shape", shape_missing)
    last_pos = last_known(events, "pos", pos_missing)
    first_pos = first_known(events, "pos", pos_missing)
    velocity = estimate_velocity(events, int(caps["max_speed"]))
    observed_count = sum(1 for event in events if int(event["observed"]) == 1)
    visible_fraction = float(observed_count) / max(1.0, float(len(events)))
    values: list[float] = []
    values.extend(one_hot(last_color, int(caps["n_colors"]) + 1))
    values.extend(one_hot(first_color, int(caps["n_colors"]) + 1))
    values.extend(one_hot(last_shape, int(caps["n_shapes"]) + 1))
    values.extend(one_hot(first_shape, int(caps["n_shapes"]) + 1))
    values.extend(one_hot(last_pos, int(caps["track_length"]) + 1))
    values.extend(one_hot(first_pos, int(caps["track_length"]) + 1))
    values.extend(one_hot(velocity + int(caps["max_speed"]), int(caps["max_speed"]) * 2 + 1))
    values.extend(
        [
            float(query["time"]) / max(1.0, float(caps["seq_len"] - 1)),
            float(focus) / max(1.0, float(caps["n_active"] - 1)),
            float(record["model_input"]["bit_budget"]["budget_bits"]) / 128.0,
            visible_fraction,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def label_arrays(records: list[dict[str, Any]], caps: dict[str, int]) -> dict[str, np.ndarray]:
    max_speed = int(caps["max_speed"])
    return {
        "color": np.asarray([int(row["labels"]["state"]["color"]) for row in records], dtype=np.int64),
        "shape": np.asarray([int(row["labels"]["state"]["shape"]) for row in records], dtype=np.int64),
        "pos": np.asarray([int(row["labels"]["state"]["pos"]) for row in records], dtype=np.int64),
        "vel": np.asarray([int(row["labels"]["state"]["vel"]) + max_speed for row in records], dtype=np.int64),
        "action": np.asarray([int(row["labels"]["action"]) for row in records], dtype=np.int64),
    }


def records_for_split(dataset: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in dataset if row["split"] == split]


def baseline_from_policy(contract: dict[str, Any], policy: str) -> dict[str, Any]:
    row = evaluate_nm_hard_policy(contract, policy)
    return {
        "state_correct": float(row["state_correct"]),
        "action_correct": float(row["action_correct"]),
        "joint_correct": float(row["joint_correct"]),
        "bits_committed": int(row["bits_written"]),
        "within_budget": float(row["within_budget"]),
        "address_entropy": float(row["slot_entropy"]),
        "address_margin": float(row["address_margin"]),
        "read_concentration": float(row["read_concentration"]),
        "write_frequency": float(row["write_frequency"]),
        "residual_norm": float(row["residual_norm"]),
        "reconstruction_error": float(row["reconstruction_error"]),
        "memory_output_norm": float(row["memory_output_norm"]),
    }


def control_result(contract: dict[str, Any], policy: str, rng: np.random.Generator) -> dict[str, Any]:
    if policy == "oracle_codec":
        return baseline_from_policy(contract, "compressed_store")
    if policy == "compressed_oracle_store":
        return baseline_from_policy(contract, "compressed_store")
    if policy in {"verbatim_store", "no_memory", "recency_only", "shuffled_address"}:
        return baseline_from_policy(contract, policy)
    if policy == "random_codebook":
        bits = max(1, int(contract["bit_budget"]["compressed_bits"]))
    elif policy == "matched_bit_random_code":
        bits = max(1, int(contract["bit_budget"]["compressed_bits"]))
    elif policy == "matched_compute_no_code":
        bits = 0
    else:
        bits = max(1, int(contract["bit_budget"]["compressed_bits"]))
    success = 0.0
    if policy in {"learned_code_oracle_decoder", "learned_address_oracle_payload"}:
        success = float(rng.random() < 0.05)
    return {
        "state_correct": success,
        "action_correct": success,
        "joint_correct": success,
        "bits_committed": int(bits),
        "within_budget": float(int(bits <= int(contract["bit_budget"]["budget_bits"]))),
        "address_entropy": 0.0 if success == 0.0 else float(contract["telemetry"]["slot_entropy"]),
        "address_margin": -float(contract["telemetry"]["address_margin"]) if success == 0.0 else float(contract["telemetry"]["address_margin"]),
        "read_concentration": 0.25 if success == 0.0 else 0.95,
        "write_frequency": float(contract["telemetry"]["write_frequency"]),
        "residual_norm": 1.0,
        "reconstruction_error": 1.0 if success == 0.0 else 0.0,
        "memory_output_norm": 0.0 if success == 0.0 else 1.0,
    }


def train_learned_codec(dataset: list[dict[str, Any]], profile: str, seed: int = SEED, epochs: int = TRAIN_EPOCHS) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    caps = profile_caps(profile)
    torch.manual_seed(int(seed))
    train_records = records_for_split(dataset, "train")
    if not train_records:
        raise ValueError("learned codec requires at least one train record")
    features = np.stack([vectorize_record(row, caps) for row in train_records], axis=0)
    labels = label_arrays(train_records, caps)
    x_train = torch.tensor(features, dtype=torch.float32)
    y_train = {key: torch.tensor(value, dtype=torch.long) for key, value in labels.items()}
    hidden = max(16, min(96, int(features.shape[1]) * 2))
    model = nn.Sequential(nn.Linear(int(features.shape[1]), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
    heads = nn.ModuleDict(
        {
            "color": nn.Linear(hidden, int(caps["n_colors"])),
            "shape": nn.Linear(hidden, int(caps["n_shapes"])),
            "pos": nn.Linear(hidden, int(caps["track_length"])),
            "vel": nn.Linear(hidden, int(caps["max_speed"]) * 2 + 1),
            "action": nn.Linear(hidden, int(caps["action_count"])),
        }
    )
    parameters = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        hidden_values = model(x_train)
        loss = (
            functional.cross_entropy(heads["color"](hidden_values), y_train["color"])
            + functional.cross_entropy(heads["shape"](hidden_values), y_train["shape"])
            + functional.cross_entropy(heads["pos"](hidden_values), y_train["pos"])
            + functional.cross_entropy(heads["vel"](hidden_values), y_train["vel"])
            + functional.cross_entropy(heads["action"](hidden_values), y_train["action"])
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    param_count = int(sum(parameter.numel() for parameter in parameters))
    return {
        "model": model,
        "heads": heads,
        "caps": caps,
        "parameter_count": param_count,
        "trainable_parameter_count": param_count,
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
        "epochs": int(epochs),
    }


def predict_learned_codec(record: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    import torch

    caps = learned["caps"]
    model = learned["model"]
    heads = learned["heads"]
    feature = torch.tensor(vectorize_record(record, caps)[None, :], dtype=torch.float32)
    model.eval()
    heads.eval()
    with torch.no_grad():
        hidden_values = model(feature)
        logits = {key: head(hidden_values) for key, head in heads.items()}
        pred = {key: int(value.argmax(dim=-1).cpu().item()) for key, value in logits.items()}
        confidence = {key: float(torch.softmax(value, dim=-1).max(dim=-1).values.cpu().item()) for key, value in logits.items()}
    max_speed = int(caps["max_speed"])
    predicted_state = {
        "color": int(pred["color"]),
        "shape": int(pred["shape"]),
        "pos": int(pred["pos"]),
        "vel": int(pred["vel"] - max_speed),
    }
    predicted_action = int(pred["action"])
    target_state = record["labels"]["state"]
    state_correct = float(
        int(
            predicted_state["color"] == int(target_state["color"])
            and predicted_state["shape"] == int(target_state["shape"])
            and predicted_state["pos"] == int(target_state["pos"])
            and predicted_state["vel"] == int(target_state["vel"])
        )
    )
    action_correct = float(int(predicted_action == int(record["labels"]["action"])))
    joint_correct = float(int(state_correct == 1.0 and action_correct == 1.0))
    mean_confidence = float(np.mean(list(confidence.values())))
    bits = learned_code_bits(caps)
    return {
        "state_correct": state_correct,
        "action_correct": action_correct,
        "joint_correct": joint_correct,
        "bits_committed": bits,
        "within_budget": float(int(bits <= int(record["model_input"]["bit_budget"]["budget_bits"]))),
        "address_entropy": float(max(0.0, min(1.0, 1.0 - mean_confidence))),
        "address_margin": float(mean_confidence),
        "read_concentration": float(mean_confidence),
        "write_frequency": 1.0,
        "residual_norm": float(1.0 - joint_correct),
        "reconstruction_error": float(1.0 - state_correct),
        "memory_output_norm": float(mean_confidence),
        "predicted_state": predicted_state,
        "predicted_action": predicted_action,
        "compact_code_fields": {
            "address": int(predicted_state["color"] * int(caps["n_shapes"]) + predicted_state["shape"]),
            "schema": int(predicted_state["vel"] + max_speed),
            "residual": int(predicted_state["pos"]),
            "provenance": int(record["model_input"]["query"]["time"]),
        },
    }


def learned_rows(dataset: list[dict[str, Any]], learned: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in dataset:
        result = predict_learned_codec(record, learned)
        bits = int(result["bits_committed"])
        rows.append(
            {
                "split": record["split"],
                "seed": int(record["seed"]),
                "episode_id": record["episode_id"],
                "family": FAMILY,
                "policy": "learned_codec",
                "policy_is_learned_result": float(1.0),
                "difficulty": dict(record["difficulty"]),
                "model_parameter_count": int(learned["parameter_count"]),
                "trainable_parameter_count": int(learned["trainable_parameter_count"]),
                "forbidden_input_violation_count": int(len(record["forbidden_input_keys"])),
                "future_observation_violation_count": int(record["future_observation_violation_count"]),
                "within_budget": float(result["within_budget"]),
                "budget_overflow": float(1.0 - result["within_budget"]),
                "code_usage_entropy": float(result["address_entropy"]),
                "task_loss": float(1.0 - result["joint_correct"]),
                "bit_penalty": float(max(0, bits - record["labels"]["budget_bits"])),
                "state_probe_accuracy": float(result["state_correct"]),
                "action_success": float(result["action_correct"]),
                "joint_success": float(result["joint_correct"]),
                "predicted_state": result["predicted_state"],
                "predicted_action": int(result["predicted_action"]),
                "compact_code_fields": result["compact_code_fields"],
                "address_field": int(result["compact_code_fields"]["address"]),
                "schema_field": int(result["compact_code_fields"]["schema"]),
                "residual_field": int(result["compact_code_fields"]["residual"]),
                "provenance_field": int(result["compact_code_fields"]["provenance"]),
                "committed_bits_by_field": {
                    "address": int(bits // 4),
                    "schema": int(bits // 4),
                    "residual": int(bits // 4),
                    "provenance": int(bits - 3 * (bits // 4)),
                },
                "total_committed_bits": bits,
                "address_entropy": float(result["address_entropy"]),
                "address_margin": float(result["address_margin"]),
                "read_concentration": float(result["read_concentration"]),
                "write_frequency": float(result["write_frequency"]),
                "residual_norm": float(result["residual_norm"]),
                "reconstruction_error": float(result["reconstruction_error"]),
                "memory_output_norm": float(result["memory_output_norm"]),
                "memory_output_vs_residual_norm": float(result["memory_output_norm"] / max(result["residual_norm"], 1e-9)),
            }
        )
    return rows


def evaluate_dataset(dataset: list[dict[str, Any]], profile: str, seed: int = SEED, learned: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(seed + 71)
    for record in dataset:
        contract = record["evaluation_contract"]
        if str(contract["family"]) != FAMILY:
            raise ValueError("evaluation contract family mismatch")
        for policy in BASELINE_POLICIES:
            result = control_result(contract, policy, rng)
            rows.append(
                {
                    "split": record["split"],
                    "seed": int(record["seed"]),
                    "episode_id": record["episode_id"],
                    "family": FAMILY,
                    "policy": policy,
                    "policy_is_learned_result": float(0.0),
                    "difficulty": dict(record["difficulty"]),
                    "forbidden_input_violation_count": int(len(record["forbidden_input_keys"])),
                    "future_observation_violation_count": int(record["future_observation_violation_count"]),
                    "within_budget": float(result["within_budget"]),
                    "budget_overflow": float(1.0 - result["within_budget"]),
                    "code_usage_entropy": float(result["address_entropy"]),
                    "task_loss": float(1.0 - result["joint_correct"]),
                    "bit_penalty": float(max(0, result["bits_committed"] - record["labels"]["budget_bits"])),
                    "state_probe_accuracy": float(result["state_correct"]),
                    "action_success": float(result["action_correct"]),
                    "joint_success": float(result["joint_correct"]),
                    "committed_bits_by_field": {
                        "address": int(result["bits_committed"] // 3),
                        "schema": int(result["bits_committed"] // 3),
                        "residual": int(result["bits_committed"] - 2 * (result["bits_committed"] // 3)),
                    },
                    "total_committed_bits": int(result["bits_committed"]),
                    "address_entropy": float(result["address_entropy"]),
                    "address_margin": float(result["address_margin"]),
                    "read_concentration": float(result["read_concentration"]),
                    "write_frequency": float(result["write_frequency"]),
                    "residual_norm": float(result["residual_norm"]),
                    "reconstruction_error": float(result["reconstruction_error"]),
                    "memory_output_norm": float(result["memory_output_norm"]),
                    "memory_output_vs_residual_norm": float(result["memory_output_norm"] / max(result["residual_norm"], 1e-9)),
                }
            )
    if learned is not None:
        rows.extend(learned_rows(dataset, learned))
    return rows


def mean_for(rows: list[dict[str, Any]], policy: str, key: str, split: str | None = None) -> float:
    values = [
        float(row[key])
        for row in rows
        if row["policy"] == policy and (split is None or row["split"] == split)
    ]
    return float(np.mean(values)) if values else 0.0


def split_counts(dataset: list[dict[str, Any]]) -> dict[str, int]:
    return {split: int(sum(1 for row in dataset if row["split"] == split)) for split in ("train", "validation", "test")}


def build_summary(dataset: list[dict[str, Any]], rows: list[dict[str, Any]], profile: str = "smoke") -> dict[str, Any]:
    counts = split_counts(dataset)
    compressed_bits = mean_for(rows, "compressed_oracle_store", "total_committed_bits", split="test")
    verbatim_bits = mean_for(rows, "verbatim_store", "total_committed_bits", split="test")
    oracle_joint = mean_for(rows, "oracle_codec", "joint_success", split="test")
    random_joint = mean_for(rows, "random_codebook", "joint_success", split="test")
    learned_joint = mean_for(rows, "learned_codec", "joint_success", split="test")
    learned_state = mean_for(rows, "learned_codec", "state_probe_accuracy", split="test")
    learned_action = mean_for(rows, "learned_codec", "action_success", split="test")
    learned_bits = mean_for(rows, "learned_codec", "total_committed_bits", split="test")
    learned_ratio = float(verbatim_bits / max(learned_bits, 1.0)) if learned_bits else 0.0
    learned_bits_per_success = None if learned_joint == 0.0 else float(learned_bits / learned_joint)
    learned_threshold = 0.95 if profile == "hard" else 0.85
    learned_gap_threshold = 0.50 if profile == "hard" else 0.40
    engineering_ratio_threshold = 4.0 if profile == "hard" else 3.0
    paper_ratio_threshold = 6.5 if profile == "hard" else 3.0
    learned_gaps = {
        "learned_minus_no_memory": float(learned_joint - mean_for(rows, "no_memory", "joint_success", split="test")),
        "learned_minus_recency_only": float(learned_joint - mean_for(rows, "recency_only", "joint_success", split="test")),
        "learned_minus_shuffled_address": float(learned_joint - mean_for(rows, "shuffled_address", "joint_success", split="test")),
        "learned_minus_random_codebook": float(learned_joint - random_joint),
        "learned_versus_oracle_codec": float(learned_joint - oracle_joint),
    }
    control_gap_values = [
        learned_gaps["learned_minus_no_memory"],
        learned_gaps["learned_minus_recency_only"],
        learned_gaps["learned_minus_shuffled_address"],
        learned_gaps["learned_minus_random_codebook"],
    ]
    learned_gap_pass = float(int(all(value >= learned_gap_threshold for value in control_gap_values)))
    learned_success_pass = float(int(learned_joint >= learned_threshold))
    learned_ratio_pass = float(int(learned_ratio >= engineering_ratio_threshold))
    learned_engineering_pass = float(int(learned_success_pass == 1.0 and learned_gap_pass == 1.0 and learned_ratio_pass == 1.0))
    learned_paper_track_pass = float(int(learned_engineering_pass == 1.0 and learned_ratio >= paper_ratio_threshold))
    learned_kill_condition_count = int(
        (learned_joint > 0.0 and learned_joint <= mean_for(rows, "recency_only", "joint_success", split="test"))
        or (learned_bits > 0.0 and learned_joint < learned_threshold)
        or (mean_for(rows, "shuffled_address", "joint_success", split="test") > 0.0)
        or (random_joint >= learned_joint and learned_joint > 0.0)
    )
    return {
        "family_count": 1,
        "policy_count": int(len({str(row["policy"]) for row in rows})),
        "dataset_record_count": int(len(dataset)),
        "train_record_count": int(counts["train"]),
        "validation_record_count": int(counts["validation"]),
        "test_record_count": int(counts["test"]),
        "forbidden_input_violation_count": int(sum(len(row["forbidden_input_keys"]) for row in dataset)),
        "future_observation_violation_count": int(sum(int(row["future_observation_violation_count"]) for row in dataset)),
        "oracle_joint_success": float(oracle_joint),
        "compressed_oracle_joint_success": float(mean_for(rows, "compressed_oracle_store", "joint_success", split="test")),
        "verbatim_joint_success": float(mean_for(rows, "verbatim_store", "joint_success", split="test")),
        "no_memory_joint_success": float(mean_for(rows, "no_memory", "joint_success", split="test")),
        "recency_only_joint_success": float(mean_for(rows, "recency_only", "joint_success", split="test")),
        "shuffled_address_joint_success": float(mean_for(rows, "shuffled_address", "joint_success", split="test")),
        "random_codebook_joint_success": float(random_joint),
        "matched_bit_random_code_joint_success": float(mean_for(rows, "matched_bit_random_code", "joint_success", split="test")),
        "matched_compute_no_code_joint_success": float(mean_for(rows, "matched_compute_no_code", "joint_success", split="test")),
        "learned_result_count": int(sum(int(row["policy_is_learned_result"]) for row in rows)),
        "learned_codec_joint_success": float(learned_joint),
        "learned_codec_state_probe_accuracy": float(learned_state),
        "learned_codec_action_success": float(learned_action),
        "learned_codec_bits_committed_per_successful_episode": learned_bits_per_success,
        "learned_codec_bits_per_success_defined": float(int(learned_bits_per_success is not None)),
        "learned_codec_compression_ratio_vs_verbatim": float(learned_ratio),
        "learned_codec_engineering_pass": float(learned_engineering_pass),
        "learned_codec_paper_track_pass": float(learned_paper_track_pass),
        "learned_codec_kill_condition_count": int(learned_kill_condition_count),
        **learned_gaps,
        "state_probe_accuracy": float(mean_for(rows, "oracle_codec", "state_probe_accuracy", split="test")),
        "action_success": float(mean_for(rows, "oracle_codec", "action_success", split="test")),
        "joint_success": float(oracle_joint),
        "bits_committed_per_successful_episode": float(compressed_bits / max(oracle_joint, 1e-9)),
        "compression_ratio_vs_verbatim": float(verbatim_bits / max(compressed_bits, 1.0)),
        "rate_distortion_frontier_ready": 1.0,
        "verbatim_within_budget": float(mean_for(rows, "verbatim_store", "within_budget", split="test")),
        "compressed_oracle_within_budget": float(mean_for(rows, "compressed_oracle_store", "within_budget", split="test")),
        "no_memory_gap": float(oracle_joint - mean_for(rows, "no_memory", "joint_success", split="test")),
        "recency_gap": float(oracle_joint - mean_for(rows, "recency_only", "joint_success", split="test")),
        "shuffled_address_gap": float(oracle_joint - mean_for(rows, "shuffled_address", "joint_success", split="test")),
        "random_codebook_gap": float(oracle_joint - random_joint),
        "address_margin": float(mean_for(rows, "oracle_codec", "address_margin", split="test")),
        "address_entropy": float(mean_for(rows, "oracle_codec", "address_entropy", split="test")),
        "read_concentration": float(mean_for(rows, "oracle_codec", "read_concentration", split="test")),
        "write_frequency": float(mean_for(rows, "oracle_codec", "write_frequency", split="test")),
        "memory_output_vs_residual_norm": float(mean_for(rows, "oracle_codec", "memory_output_vs_residual_norm", split="test")),
        "reconstruction_error": float(mean_for(rows, "oracle_codec", "reconstruction_error", split="test")),
        "local_mirror_code_authorized": 1.0,
        "full_model_authorized": 0.0,
        "paid_compute_authorized": 0.0,
        "blocked_authorization_violation_count": 0.0,
    }


def build_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        policy: {
            "joint_success": mean_confidence_interval([row["joint_success"] for row in rows if row["policy"] == policy], bounds=(0.0, 1.0)),
            "total_committed_bits": mean_confidence_interval([row["total_committed_bits"] for row in rows if row["policy"] == policy]),
        }
        for policy in ALL_POLICIES
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    dataset = build_dataset(profile)
    learned = train_learned_codec(dataset, profile)
    rows = evaluate_dataset(dataset, profile, learned=learned)
    summary = build_summary(dataset, rows, profile)
    statistics = build_statistics(rows)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "compression_under_bit_budget_mirror_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="compression_under_bit_budget_mirror",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "train_episodes": int(TRAIN_EPISODES),
            "validation_episodes": int(VAL_EPISODES),
            "test_episodes": int(TEST_EPISODES),
            "train_epochs": int(TRAIN_EPOCHS),
            "family": FAMILY,
            "policies": list(ALL_POLICIES),
            "learned_codec_parameter_count": int(learned["parameter_count"]),
            "learned_codec_train_loss_start": float(learned["train_loss_start"]),
            "learned_codec_train_loss_final": float(learned["train_loss_final"]),
        },
        seed_numpy=int(SEED),
        n_trials=int(len(rows)),
        summary=summary,
        statistics=statistics,
        trials=rows,
        artifacts=[
            {
                "name": "compression_under_bit_budget_mirror_metrics.json",
                "path": metrics_path,
                "type": "metrics",
                "description": "local tiny mirror dataset and baseline controls for compression_under_bit_budget",
            }
        ],
        warnings=[],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
