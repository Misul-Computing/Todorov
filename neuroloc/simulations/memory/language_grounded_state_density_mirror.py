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

from shared import build_run_record, env_int, output_dir_for, require_positive, utc_now_iso, write_json
from neuroloc.simulations.memory.compression_under_bit_budget_mirror import (
    build_factor_heldout_distributed_dataset,
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


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("LGSD_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("LGSD_PROFILE must be smoke or hard")
    return value


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


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
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
        warnings=["local symbolic language bridge only; not arbitrary chat and not solved compression"],
    )
    write_json(metrics_path, record)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
