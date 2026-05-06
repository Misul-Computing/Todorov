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
    "content_routed_sparse_read",
    "compressed_oracle_store",
    "no_memory",
    "recency_only",
    "shuffled_address",
    "random_codebook",
    "matched_bit_random_code",
    "matched_compute_no_code",
    "frozen_random_encoder_learned_decoder",
    "learned_encoder_frozen_random_decoder",
)
LEARNED_POLICIES = ("learned_codec",)
DIAGNOSTIC_POLICIES = (
    "learned_code_oracle_decoder",
    "oracle_code_learned_decoder",
    "learned_address_oracle_payload",
    "oracle_address_learned_payload",
    "provenance_exposed_learned_codec",
    "visible_source_codec",
    "visible_source_state_oracle_action_oracle_decoder",
    "source_observation_learned_action",
    "provenance_exposed_oracle_decoder",
    "learned_state_oracle_action_oracle_decoder",
    "oracle_state_learned_action_oracle_decoder",
)
ALL_POLICIES = (*BASELINE_POLICIES, *DIAGNOSTIC_POLICIES, *LEARNED_POLICIES)


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


def observation_events(episode: dict[str, Any], max_time: int, commit_time: int | None = None, commit_object: int | None = None) -> list[dict[str, int]]:
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
                    "commit_marker": int(commit_time is not None and commit_object is not None and time_index == int(commit_time) and object_index == int(commit_object)),
                    "commit_next_marker": int(commit_time is not None and commit_object is not None and time_index == int(commit_time) + 1 and object_index == int(commit_object)),
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
    commit_time = int(query["commit_time"])
    commit_object = int(query["commit_local_index"])
    return {
        "observations": observation_events(episode, query_time, commit_time, commit_object),
        "query": {
            "time": query_time,
            "focus_local_index": int(query["focus_local_index"]),
            "cue_color": int(query["cue_color"]),
            "cue_shape": int(query["cue_shape"]),
            "cue_pos": int(query["cue_pos"]),
            "commit_time": commit_time,
            "commit_local_index": commit_object,
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
        + bits_for_cardinality(int(caps["action_count"]))
        + bits_for_cardinality(int(caps["seq_len"]))
    )


def learned_bits_by_field(caps: dict[str, int]) -> dict[str, int]:
    return {
        "address": bits_for_cardinality(int(caps["n_colors"]) * int(caps["n_shapes"])),
        "schema": bits_for_cardinality(int(caps["max_speed"]) * 2 + 1),
        "residual": bits_for_cardinality(int(caps["track_length"])),
        "action": bits_for_cardinality(int(caps["action_count"])),
        "provenance": bits_for_cardinality(int(caps["seq_len"])),
    }


def sparse_read_record_bits(caps: dict[str, int]) -> int:
    return int(
        bits_for_cardinality(int(caps["seq_len"]))
        + bits_for_cardinality(int(caps["n_active"]))
        + bits_for_cardinality(int(caps["n_colors"]) + 1)
        + bits_for_cardinality(int(caps["n_shapes"]) + 1)
        + bits_for_cardinality(int(caps["track_length"]) + 1)
        + 1
        + 1
        + 1
    )


def sparse_read_bits_by_field(caps: dict[str, int], selected_count: int) -> dict[str, int]:
    selected = int(selected_count)
    return {
        "address": int(selected * (bits_for_cardinality(int(caps["seq_len"])) + bits_for_cardinality(int(caps["n_active"])))),
        "schema": int(selected * (bits_for_cardinality(int(caps["n_colors"]) + 1) + bits_for_cardinality(int(caps["n_shapes"]) + 1))),
        "residual": int(selected * (bits_for_cardinality(int(caps["track_length"]) + 1) + 3)),
    }


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


def marked_source_event(record: dict[str, Any]) -> dict[str, int]:
    matches = [event for event in record["model_input"]["observations"] if int(event.get("commit_marker", 0)) == 1]
    if not matches:
        return source_event_for_record(record)
    return dict(matches[0])


def marked_source_next_event(record: dict[str, Any]) -> dict[str, int] | None:
    matches = [event for event in record["model_input"]["observations"] if int(event.get("commit_next_marker", 0)) == 1]
    if not matches:
        return None
    return dict(matches[0])


def estimate_marked_source_velocity(record: dict[str, Any], max_speed: int) -> int:
    source = marked_source_event(record)
    next_event = marked_source_next_event(record)
    if next_event is None:
        return estimate_velocity(focus_events_for_record(record), max_speed)
    if int(source["observed"]) != 1 or int(next_event["observed"]) != 1:
        return 0
    if int(source["pos"]) < 0 or int(next_event["pos"]) < 0:
        return 0
    delta_time = max(1, int(next_event["time"]) - int(source["time"]))
    estimate = int(round(float(int(next_event["pos"]) - int(source["pos"])) / float(delta_time)))
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
    source_event = marked_source_event(record)
    source_velocity = estimate_marked_source_velocity(record, int(caps["max_speed"]))
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
    values.extend(one_hot(int(source_event["color"]) if int(source_event["color"]) >= 0 else int(caps["n_colors"]), int(caps["n_colors"]) + 1))
    values.extend(one_hot(int(source_event["shape"]) if int(source_event["shape"]) >= 0 else int(caps["n_shapes"]), int(caps["n_shapes"]) + 1))
    values.extend(one_hot(int(source_event["pos"]) if int(source_event["pos"]) >= 0 else int(caps["track_length"]), int(caps["track_length"]) + 1))
    values.extend(one_hot(source_velocity + int(caps["max_speed"]), int(caps["max_speed"]) * 2 + 1))
    values.extend(
        [
            float(query["time"]) / max(1.0, float(caps["seq_len"] - 1)),
            float(focus) / max(1.0, float(caps["n_active"] - 1)),
            float(query["commit_time"]) / max(1.0, float(caps["seq_len"] - 1)),
            float(query["commit_local_index"]) / max(1.0, float(caps["n_active"] - 1)),
            float(record["model_input"]["bit_budget"]["budget_bits"]) / 128.0,
            visible_fraction,
            float(source_event["observed"]),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def source_event_for_record(record: dict[str, Any]) -> dict[str, int]:
    relevant = record["evaluation_contract"]["memory_relevant_positions"][0]
    source_time = int(relevant["time"])
    source_object = int(relevant["object_index"])
    matches = [
        event
        for event in record["model_input"]["observations"]
        if int(event["time"]) == source_time and int(event["object_index"]) == source_object
    ]
    if not matches:
        return {
            "time": source_time,
            "object_index": source_object,
            "color": -1,
            "shape": -1,
            "pos": -1,
            "observed": 0,
        }
    return dict(matches[0])


def source_event_present(record: dict[str, Any]) -> float:
    relevant = record["evaluation_contract"]["memory_relevant_positions"][0]
    source_time = int(relevant["time"])
    source_object = int(relevant["object_index"])
    return float(
        int(
            any(
                int(event["time"]) == source_time and int(event["object_index"]) == source_object
                for event in record["model_input"]["observations"]
            )
        )
    )


def focus_events_for_record(record: dict[str, Any]) -> list[dict[str, int]]:
    focus = int(record["model_input"]["query"]["focus_local_index"])
    return [
        event
        for event in record["model_input"]["observations"]
        if int(event["object_index"]) == focus
    ]


def source_event_observed(record: dict[str, Any]) -> float:
    return float(int(int(source_event_for_record(record)["observed"]) == 1))


def source_event_complete(record: dict[str, Any]) -> float:
    event = source_event_for_record(record)
    return float(
        int(
            int(event["observed"]) == 1
            and int(event["color"]) >= 0
            and int(event["shape"]) >= 0
            and int(event["pos"]) >= 0
        )
    )


def source_observation_audit(record: dict[str, Any], caps: dict[str, int]) -> dict[str, float]:
    event = marked_source_event(record)
    next_event = marked_source_next_event(record)
    target_state = record["labels"]["state"]
    estimated_vel = estimate_marked_source_velocity(record, int(caps["max_speed"]))
    color_visible = float(int(int(event["observed"]) == 1 and int(event["color"]) >= 0))
    shape_visible = float(int(int(event["observed"]) == 1 and int(event["shape"]) >= 0))
    pos_visible = float(int(int(event["observed"]) == 1 and int(event["pos"]) >= 0))
    next_pos_visible = float(int(next_event is not None and int(next_event["observed"]) == 1 and int(next_event["pos"]) >= 0))
    vel_reconstructable = float(int(next_pos_visible == 1.0 and estimated_vel == int(target_state["vel"])))
    source_query_gap = int(record["model_input"]["query"]["time"]) - int(event["time"])
    source_state_reconstructable = float(
        int(
            color_visible == 1.0
            and shape_visible == 1.0
            and pos_visible == 1.0
            and vel_reconstructable == 1.0
            and int(event["color"]) == int(target_state["color"])
            and int(event["shape"]) == int(target_state["shape"])
            and int(event["pos"]) == int(target_state["pos"])
        )
    )
    return {
        "source_event_present": float(source_event_present(record)),
        "source_event_observed": float(source_event_observed(record)),
        "source_event_complete": float(source_event_complete(record)),
        "source_color_visible": color_visible,
        "source_shape_visible": shape_visible,
        "source_pos_visible": pos_visible,
        "source_vel_reconstructable": vel_reconstructable,
        "source_state_reconstructable": source_state_reconstructable,
        "source_required_fields_visible": float(int(color_visible == 1.0 and shape_visible == 1.0 and pos_visible == 1.0 and next_pos_visible == 1.0)),
        "source_query_gap": float(source_query_gap),
    }


def empty_source_observation_audit() -> dict[str, float]:
    return {
        "source_event_present": 0.0,
        "source_event_observed": 0.0,
        "source_event_complete": 0.0,
        "source_color_visible": 0.0,
        "source_shape_visible": 0.0,
        "source_pos_visible": 0.0,
        "source_vel_reconstructable": 0.0,
        "source_state_reconstructable": 0.0,
        "source_required_fields_visible": 0.0,
        "source_query_gap": 0.0,
    }


def source_observation_code_fields(record: dict[str, Any], caps: dict[str, int], action_value: int) -> dict[str, int]:
    event = marked_source_event(record)
    velocity = estimate_marked_source_velocity(record, int(caps["max_speed"]))
    color = int(event["color"])
    shape = int(event["shape"])
    address = color * int(caps["n_shapes"]) + shape if color >= 0 and shape >= 0 else -1
    return {
        "address": int(address),
        "schema": int(velocity + int(caps["max_speed"])),
        "residual": int(event["pos"]) if int(event["pos"]) >= 0 else -1,
        "action": int(action_value),
        "provenance": int(event["time"]),
    }


def source_signature_for_action(record: dict[str, Any], caps: dict[str, int]) -> tuple[int, ...]:
    event = marked_source_event(record)
    return (
        int(event["color"]),
        int(event["shape"]),
        int(event["pos"]),
        int(estimate_marked_source_velocity(record, int(caps["max_speed"]))),
        int(event["observed"]),
        int(source_event_complete(record)),
    )


def action_ambiguity_rate(dataset: list[dict[str, Any]], caps: dict[str, int], split: str) -> float:
    selected = [row for row in dataset if row["split"] == split]
    if not selected:
        return 0.0
    groups: dict[tuple[int, ...], set[int]] = {}
    for row in selected:
        signature = source_signature_for_action(row, caps)
        groups.setdefault(signature, set()).add(int(row["labels"]["action"]))
    ambiguous = sum(1 for row in selected if len(groups[source_signature_for_action(row, caps)]) > 1)
    return float(ambiguous) / max(1.0, float(len(selected)))


def visible_source_action(fields: dict[str, int], caps: dict[str, int]) -> int:
    state = state_from_code_fields(fields, caps)
    return int((int(state["color"]) * 7 + int(state["shape"]) * 5 + (int(state["vel"]) + 3) * 3) % int(caps["action_count"]))


def content_routed_sparse_read_result(record: dict[str, Any], caps: dict[str, int]) -> dict[str, Any]:
    query = record["model_input"]["query"]
    source = source_event_for_record(record)
    source_key = (int(source["time"]), int(source["object_index"]))
    selected = sorted(
        record["model_input"]["observations"],
        key=lambda event: (
            -int(event.get("commit_marker", 0)),
            -int(event.get("commit_next_marker", 0)),
            -int(int(event["observed"]) == 1),
            -int(int(event["color"]) == int(query["cue_color"]) and int(event["color"]) >= 0),
            -int(int(event["shape"]) == int(query["cue_shape"]) and int(event["shape"]) >= 0),
            -int(int(event["object_index"]) == int(query["commit_local_index"])),
            abs(int(event["time"]) - int(query["commit_time"])),
            int(event["time"]),
            int(event["object_index"]),
        ),
    )[:2]
    source_selected = any((int(event["time"]), int(event["object_index"])) == source_key for event in selected)
    next_selected = any(int(event.get("commit_next_marker", 0)) == 1 for event in selected)
    source_event = next((event for event in selected if int(event.get("commit_marker", 0)) == 1), None)
    next_event = next((event for event in selected if int(event.get("commit_next_marker", 0)) == 1), None)
    if source_event is None:
        source_event = selected[0] if selected else source
    velocity = 0
    if next_event is not None and int(source_event["observed"]) == 1 and int(next_event["observed"]) == 1 and int(source_event["pos"]) >= 0 and int(next_event["pos"]) >= 0:
        delta_time = max(1, int(next_event["time"]) - int(source_event["time"]))
        velocity = int(round(float(int(next_event["pos"]) - int(source_event["pos"])) / float(delta_time)))
        velocity = int(max(-int(caps["max_speed"]), min(int(caps["max_speed"]), velocity)))
    fields = {
        "address": int(int(source_event["color"]) * int(caps["n_shapes"]) + int(source_event["shape"])) if int(source_event["color"]) >= 0 and int(source_event["shape"]) >= 0 else -1,
        "schema": int(velocity + int(caps["max_speed"])),
        "residual": int(source_event["pos"]) if int(source_event["pos"]) >= 0 else -1,
        "action": 0,
        "provenance": int(source_event["time"]),
    }
    fields["action"] = visible_source_action(fields, caps)
    result = score_code_fields(fields, record, caps, confidence=float(int(source_selected and next_selected)))
    selected_count = int(len(selected))
    bits = int(selected_count * sparse_read_record_bits(caps))
    false_selected = sum(
        1
        for event in selected
        if (int(event["time"]), int(event["object_index"])) != source_key and int(event.get("commit_next_marker", 0)) != 1
    )
    result.update(
        {
            "bits_committed": bits,
            "within_budget": float(int(bits <= int(record["model_input"]["bit_budget"]["budget_bits"]))),
            "selected_record_count": float(selected_count),
            "source_selection_recall": float(int(source_selected)),
            "next_source_selection_recall": float(int(next_selected)),
            "false_source_selection_rate": float(false_selected / max(1.0, float(selected_count))),
            "sparse_read_record_bits": float(sparse_read_record_bits(caps)),
        }
    )
    return result


def vectorize_record_with_provenance(record: dict[str, Any], caps: dict[str, int]) -> np.ndarray:
    values = list(vectorize_record(record, caps))
    event = marked_source_event(record)
    values.extend(one_hot(int(event["time"]), int(caps["seq_len"])))
    values.extend(one_hot(int(event["object_index"]), int(caps["n_active"])))
    values.extend(one_hot(int(event["color"]) if int(event["color"]) >= 0 else int(caps["n_colors"]), int(caps["n_colors"]) + 1))
    values.extend(one_hot(int(event["shape"]) if int(event["shape"]) >= 0 else int(caps["n_shapes"]), int(caps["n_shapes"]) + 1))
    values.extend(one_hot(int(event["pos"]) if int(event["pos"]) >= 0 else int(caps["track_length"]), int(caps["track_length"]) + 1))
    values.append(float(event["observed"]))
    return np.asarray(values, dtype=np.float32)


def vectorize_oracle_code(record: dict[str, Any], caps: dict[str, int]) -> np.ndarray:
    state = record["labels"]["state"]
    max_speed = int(caps["max_speed"])
    values: list[float] = []
    values.extend(one_hot(int(state["color"]), int(caps["n_colors"])))
    values.extend(one_hot(int(state["shape"]), int(caps["n_shapes"])))
    values.extend(one_hot(int(state["pos"]), int(caps["track_length"])))
    values.extend(one_hot(int(state["vel"]) + max_speed, max_speed * 2 + 1))
    values.extend(one_hot(int(record["labels"]["action"]), int(caps["action_count"])))
    values.extend(one_hot(int(record["evaluation_contract"]["memory_relevant_positions"][0]["time"]), int(caps["seq_len"])))
    return np.asarray(values, dtype=np.float32)


def oracle_code_fields(record: dict[str, Any], caps: dict[str, int]) -> dict[str, int]:
    state = record["labels"]["state"]
    return {
        "address": int(int(state["color"]) * int(caps["n_shapes"]) + int(state["shape"])),
        "schema": int(int(state["vel"]) + int(caps["max_speed"])),
        "residual": int(state["pos"]),
        "action": int(record["labels"]["action"]),
        "provenance": int(record["evaluation_contract"]["memory_relevant_positions"][0]["time"]),
    }


def state_from_code_fields(fields: dict[str, int], caps: dict[str, int]) -> dict[str, int]:
    address = int(fields["address"])
    return {
        "color": int(address // int(caps["n_shapes"])),
        "shape": int(address % int(caps["n_shapes"])),
        "pos": int(fields["residual"]),
        "vel": int(fields["schema"] - int(caps["max_speed"])),
    }


def field_accuracies(fields: dict[str, int], record: dict[str, Any], caps: dict[str, int]) -> dict[str, float]:
    oracle = oracle_code_fields(record, caps)
    target_state = record["labels"]["state"]
    predicted_state = state_from_code_fields(fields, caps)
    return {
        "encoder_address_accuracy": float(int(int(fields["address"]) == int(oracle["address"]))),
        "encoder_payload_accuracy": float(
            int(
                int(fields["schema"]) == int(oracle["schema"])
                and int(fields["residual"]) == int(oracle["residual"])
                and int(fields["action"]) == int(oracle["action"])
            )
        ),
        "encoder_payload_color_accuracy": float(int(predicted_state["color"] == int(target_state["color"]))),
        "encoder_payload_shape_accuracy": float(int(predicted_state["shape"] == int(target_state["shape"]))),
        "encoder_payload_pos_accuracy": float(int(predicted_state["pos"] == int(target_state["pos"]))),
        "encoder_payload_vel_accuracy": float(int(predicted_state["vel"] == int(target_state["vel"]))),
        "encoder_action_accuracy": float(int(int(fields["action"]) == int(oracle["action"]))),
        "encoder_provenance_accuracy": float(int(int(fields["provenance"]) == int(oracle["provenance"]))),
    }


def score_code_fields(fields: dict[str, int], record: dict[str, Any], caps: dict[str, int], confidence: float = 1.0) -> dict[str, Any]:
    predicted_state = state_from_code_fields(fields, caps)
    predicted_action = int(fields["action"])
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
    bits = learned_code_bits(caps)
    mean_confidence = float(max(0.0, min(1.0, confidence)))
    result = {
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
        "compact_code_fields": dict(fields),
    }
    result.update(field_accuracies(fields, record, caps))
    return result


def label_arrays(records: list[dict[str, Any]], caps: dict[str, int]) -> dict[str, np.ndarray]:
    max_speed = int(caps["max_speed"])
    return {
        "color": np.asarray([int(row["labels"]["state"]["color"]) for row in records], dtype=np.int64),
        "shape": np.asarray([int(row["labels"]["state"]["shape"]) for row in records], dtype=np.int64),
        "pos": np.asarray([int(row["labels"]["state"]["pos"]) for row in records], dtype=np.int64),
        "vel": np.asarray([int(row["labels"]["state"]["vel"]) + max_speed for row in records], dtype=np.int64),
        "action": np.asarray([int(row["labels"]["action"]) for row in records], dtype=np.int64),
        "provenance": np.asarray([int(row["evaluation_contract"]["memory_relevant_positions"][0]["time"]) for row in records], dtype=np.int64),
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


def control_result(record: dict[str, Any], policy: str, rng: np.random.Generator) -> dict[str, Any]:
    contract = record["evaluation_contract"]
    if policy == "oracle_codec":
        return baseline_from_policy(contract, "compressed_store")
    if policy == "content_routed_sparse_read":
        return content_routed_sparse_read_result(record, profile_caps(str(record["model_input"]["visible_context"]["profile"])))
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
    provenance_features = np.stack([vectorize_record_with_provenance(row, caps) for row in train_records], axis=0)
    oracle_code_features = np.stack([vectorize_oracle_code(row, caps) for row in train_records], axis=0)
    labels = label_arrays(train_records, caps)
    x_train = torch.tensor(features, dtype=torch.float32)
    x_provenance_train = torch.tensor(provenance_features, dtype=torch.float32)
    x_oracle_code_train = torch.tensor(oracle_code_features, dtype=torch.float32)
    y_train = {key: torch.tensor(value, dtype=torch.long) for key, value in labels.items()}
    hidden = max(16, min(96, int(features.shape[1]) * 2))
    provenance_hidden = max(16, min(128, int(provenance_features.shape[1]) * 2))
    decoder_hidden = max(16, min(96, int(oracle_code_features.shape[1]) * 2))

    def make_stack(input_dim: int, hidden_dim: int) -> tuple[nn.Sequential, nn.ModuleDict]:
        stack = nn.Sequential(nn.Linear(int(input_dim), int(hidden_dim)), nn.Tanh(), nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh())
        stack_heads = nn.ModuleDict(
            {
                "color": nn.Linear(int(hidden_dim), int(caps["n_colors"])),
                "shape": nn.Linear(int(hidden_dim), int(caps["n_shapes"])),
                "pos": nn.Linear(int(hidden_dim), int(caps["track_length"])),
                "vel": nn.Linear(int(hidden_dim), int(caps["max_speed"]) * 2 + 1),
                "action": nn.Linear(int(hidden_dim), int(caps["action_count"])),
                "provenance": nn.Linear(int(hidden_dim), int(caps["seq_len"])),
            }
        )
        return stack, stack_heads

    model, heads = make_stack(int(features.shape[1]), hidden)
    provenance_model, provenance_heads = make_stack(int(provenance_features.shape[1]), provenance_hidden)
    decoder_model, decoder_heads = make_stack(int(oracle_code_features.shape[1]), decoder_hidden)
    parameters = list(model.parameters()) + list(heads.parameters())
    provenance_parameters = list(provenance_model.parameters()) + list(provenance_heads.parameters())
    decoder_parameters = list(decoder_model.parameters()) + list(decoder_heads.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    provenance_optimizer = torch.optim.Adam(provenance_parameters, lr=0.03)
    decoder_optimizer = torch.optim.Adam(decoder_parameters, lr=0.03)
    losses = []
    provenance_losses = []
    decoder_losses = []

    def loss_for(stack: nn.Sequential, stack_heads: nn.ModuleDict, x_value: Any) -> Any:
        hidden_values = stack(x_value)
        return (
            functional.cross_entropy(stack_heads["color"](hidden_values), y_train["color"])
            + functional.cross_entropy(stack_heads["shape"](hidden_values), y_train["shape"])
            + functional.cross_entropy(stack_heads["pos"](hidden_values), y_train["pos"])
            + functional.cross_entropy(stack_heads["vel"](hidden_values), y_train["vel"])
            + functional.cross_entropy(stack_heads["action"](hidden_values), y_train["action"])
            + functional.cross_entropy(stack_heads["provenance"](hidden_values), y_train["provenance"])
        )

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, heads, x_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        provenance_optimizer.zero_grad(set_to_none=True)
        provenance_loss = loss_for(provenance_model, provenance_heads, x_provenance_train)
        provenance_loss.backward()
        provenance_optimizer.step()
        provenance_losses.append(float(provenance_loss.detach().cpu().item()))

        decoder_optimizer.zero_grad(set_to_none=True)
        decoder_loss = loss_for(decoder_model, decoder_heads, x_oracle_code_train)
        decoder_loss.backward()
        decoder_optimizer.step()
        decoder_losses.append(float(decoder_loss.detach().cpu().item()))

    param_count = int(sum(parameter.numel() for parameter in parameters))
    provenance_param_count = int(sum(parameter.numel() for parameter in provenance_parameters))
    decoder_param_count = int(sum(parameter.numel() for parameter in decoder_parameters))
    return {
        "model": model,
        "heads": heads,
        "provenance_model": provenance_model,
        "provenance_heads": provenance_heads,
        "decoder_model": decoder_model,
        "decoder_heads": decoder_heads,
        "caps": caps,
        "parameter_count": param_count,
        "trainable_parameter_count": param_count,
        "provenance_parameter_count": provenance_param_count,
        "decoder_parameter_count": decoder_param_count,
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
        "provenance_train_loss_start": float(provenance_losses[0]) if provenance_losses else 0.0,
        "provenance_train_loss_final": float(provenance_losses[-1]) if provenance_losses else 0.0,
        "decoder_train_loss_start": float(decoder_losses[0]) if decoder_losses else 0.0,
        "decoder_train_loss_final": float(decoder_losses[-1]) if decoder_losses else 0.0,
        "epochs": int(epochs),
    }


def predict_fields_with_stack(learned: dict[str, Any], feature: np.ndarray, model_key: str, heads_key: str) -> tuple[dict[str, int], float]:
    import torch

    caps = learned["caps"]
    model = learned[model_key]
    heads = learned[heads_key]
    x_value = torch.tensor(feature[None, :], dtype=torch.float32)
    model.eval()
    heads.eval()
    with torch.no_grad():
        hidden_values = model(x_value)
        logits = {key: head(hidden_values) for key, head in heads.items()}
        pred = {key: int(value.argmax(dim=-1).cpu().item()) for key, value in logits.items()}
        confidence = {key: float(torch.softmax(value, dim=-1).max(dim=-1).values.cpu().item()) for key, value in logits.items()}
    max_speed = int(caps["max_speed"])
    fields = {
        "address": int(pred["color"] * int(caps["n_shapes"]) + pred["shape"]),
        "schema": int(pred["vel"]),
        "residual": int(pred["pos"]),
        "action": int(pred["action"]),
        "provenance": int(pred["provenance"]),
    }
    mean_confidence = float(np.mean(list(confidence.values())))
    return fields, mean_confidence


def vectorize_code_fields(fields: dict[str, int], caps: dict[str, int]) -> np.ndarray:
    values: list[float] = []
    address = int(fields["address"])
    values.extend(one_hot(int(address // int(caps["n_shapes"])), int(caps["n_colors"])))
    values.extend(one_hot(int(address % int(caps["n_shapes"])), int(caps["n_shapes"])))
    values.extend(one_hot(int(fields["residual"]), int(caps["track_length"])))
    values.extend(one_hot(int(fields["schema"]), int(caps["max_speed"]) * 2 + 1))
    values.extend(one_hot(int(fields["action"]), int(caps["action_count"])))
    values.extend(one_hot(int(fields["provenance"]), int(caps["seq_len"])))
    return np.asarray(values, dtype=np.float32)


def decode_fields_with_learned_decoder(record: dict[str, Any], learned: dict[str, Any], fields: dict[str, int], confidence: float) -> dict[str, Any]:
    decoded = predict_fields_with_stack(learned, vectorize_code_fields(fields, learned["caps"]), "decoder_model", "decoder_heads")[0]
    result = score_code_fields(decoded, record, learned["caps"], confidence=confidence)
    result["compact_code_fields"] = dict(fields)
    result.update(field_accuracies(fields, record, learned["caps"]))
    return result


def predict_learned_codec(record: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    fields, confidence = predict_fields_with_stack(learned, vectorize_record(record, learned["caps"]), "model", "heads")
    return decode_fields_with_learned_decoder(record, learned, fields, confidence)


def learned_code_oracle_decoder_result(record: dict[str, Any], learned: dict[str, Any], learned_result: dict[str, Any]) -> dict[str, Any]:
    result = dict(learned_result)
    result["joint_correct"] = 1.0
    result["state_correct"] = 1.0
    result["action_correct"] = 1.0
    result["residual_norm"] = 0.0
    result["reconstruction_error"] = 0.0
    result["predicted_state"] = dict(record["labels"]["state"])
    result["predicted_action"] = int(record["labels"]["action"])
    for key, value in field_accuracies(result["compact_code_fields"], record, learned["caps"]).items():
        if float(value) == 0.0:
            result["joint_correct"] = 0.0
            result["state_correct"] = 0.0
            result["action_correct"] = 0.0
            result["residual_norm"] = 1.0
            result["reconstruction_error"] = 1.0
            break
    return result


def row_from_result(record: dict[str, Any], policy: str, result: dict[str, Any], learned: dict[str, Any], learned_result: float, diagnostic_control: float, oracle_input_used: float) -> dict[str, Any]:
    bits = int(result["bits_committed"])
    fields = dict(result["compact_code_fields"])
    field_bits = learned_bits_by_field(learned["caps"])
    audit = source_observation_audit(record, learned["caps"]) if diagnostic_control == 1.0 else empty_source_observation_audit()
    row = {
        "split": record["split"],
        "seed": int(record["seed"]),
        "episode_id": record["episode_id"],
        "family": FAMILY,
        "policy": policy,
        "policy_is_learned_result": float(learned_result),
        "policy_is_diagnostic_control": float(diagnostic_control),
        "diagnostic_oracle_input_used": float(oracle_input_used),
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
        "compact_code_fields": fields,
        "address_field": int(fields["address"]),
        "schema_field": int(fields["schema"]),
        "residual_field": int(fields["residual"]),
        "action_field": int(fields["action"]),
        "provenance_field": int(fields["provenance"]),
        "committed_bits_by_field": field_bits,
        "total_committed_bits": bits,
        "address_entropy": float(result["address_entropy"]),
        "address_margin": float(result["address_margin"]),
        "read_concentration": float(result["read_concentration"]),
        "write_frequency": float(result["write_frequency"]),
        "residual_norm": float(result["residual_norm"]),
        "reconstruction_error": float(result["reconstruction_error"]),
        "memory_output_norm": float(result["memory_output_norm"]),
        "memory_output_vs_residual_norm": float(result["memory_output_norm"] / max(result["residual_norm"], 1e-9)),
        "encoder_address_accuracy": float(result["encoder_address_accuracy"]),
        "encoder_payload_accuracy": float(result["encoder_payload_accuracy"]),
        "encoder_payload_color_accuracy": float(result["encoder_payload_color_accuracy"]),
        "encoder_payload_shape_accuracy": float(result["encoder_payload_shape_accuracy"]),
        "encoder_payload_pos_accuracy": float(result["encoder_payload_pos_accuracy"]),
        "encoder_payload_vel_accuracy": float(result["encoder_payload_vel_accuracy"]),
        "encoder_action_accuracy": float(result["encoder_action_accuracy"]),
        "encoder_provenance_accuracy": float(result["encoder_provenance_accuracy"]),
        "selected_record_count": float(result.get("selected_record_count", 0.0)),
        "source_selection_recall": float(result.get("source_selection_recall", 0.0)),
        "next_source_selection_recall": float(result.get("next_source_selection_recall", 0.0)),
        "false_source_selection_rate": float(result.get("false_source_selection_rate", 0.0)),
        "sparse_read_record_bits": float(result.get("sparse_read_record_bits", 0.0)),
    }
    row.update(audit)
    return row


def learned_rows(dataset: list[dict[str, Any]], learned: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row_from_result(record, "learned_codec", predict_learned_codec(record, learned), learned, 1.0, 0.0, 0.0)
        for record in dataset
    ]


def diagnostic_result(record: dict[str, Any], learned: dict[str, Any], policy: str) -> dict[str, Any]:
    caps = learned["caps"]
    learned_result = predict_learned_codec(record, learned)
    learned_fields = dict(learned_result["compact_code_fields"])
    oracle_fields = oracle_code_fields(record, caps)
    if policy == "learned_code_oracle_decoder":
        return learned_code_oracle_decoder_result(record, learned, learned_result)
    if policy == "oracle_code_learned_decoder":
        return decode_fields_with_learned_decoder(record, learned, oracle_fields, confidence=1.0)
    if policy == "learned_address_oracle_payload":
        fields = dict(oracle_fields)
        fields["address"] = int(learned_fields["address"])
        fields["provenance"] = int(learned_fields["provenance"])
        return score_code_fields(fields, record, caps, confidence=float(learned_result["address_margin"]))
    if policy == "oracle_address_learned_payload":
        fields = dict(learned_fields)
        fields["address"] = int(oracle_fields["address"])
        fields["provenance"] = int(oracle_fields["provenance"])
        return score_code_fields(fields, record, caps, confidence=float(learned_result["address_margin"]))
    if policy == "provenance_exposed_learned_codec":
        fields, confidence = predict_fields_with_stack(learned, vectorize_record_with_provenance(record, caps), "provenance_model", "provenance_heads")
        return decode_fields_with_learned_decoder(record, learned, fields, confidence)
    if policy == "visible_source_codec":
        fields = source_observation_code_fields(record, caps, 0)
        fields["action"] = visible_source_action(fields, caps)
        return score_code_fields(fields, record, caps, confidence=float(source_event_complete(record)))
    if policy == "visible_source_state_oracle_action_oracle_decoder":
        return score_code_fields(source_observation_code_fields(record, caps, int(record["labels"]["action"])), record, caps, confidence=float(source_event_complete(record)))
    if policy == "source_observation_learned_action":
        return score_code_fields(source_observation_code_fields(record, caps, int(learned_fields["action"])), record, caps, confidence=float(source_event_complete(record)))
    if policy == "provenance_exposed_oracle_decoder":
        fields, confidence = predict_fields_with_stack(learned, vectorize_record_with_provenance(record, caps), "provenance_model", "provenance_heads")
        return score_code_fields(fields, record, caps, confidence)
    if policy == "learned_state_oracle_action_oracle_decoder":
        fields = dict(learned_fields)
        fields["action"] = int(oracle_fields["action"])
        return score_code_fields(fields, record, caps, confidence=float(learned_result["address_margin"]))
    if policy == "oracle_state_learned_action_oracle_decoder":
        fields = dict(oracle_fields)
        fields["action"] = int(learned_fields["action"])
        return score_code_fields(fields, record, caps, confidence=float(learned_result["address_margin"]))
    raise ValueError(f"unknown diagnostic policy: {policy}")


def diagnostic_rows(dataset: list[dict[str, Any]], learned: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in dataset:
        for policy in DIAGNOSTIC_POLICIES:
            oracle_input_used = 1.0 if policy in {"learned_code_oracle_decoder", "oracle_code_learned_decoder", "learned_address_oracle_payload", "oracle_address_learned_payload", "provenance_exposed_learned_codec", "visible_source_state_oracle_action_oracle_decoder", "source_observation_learned_action", "provenance_exposed_oracle_decoder", "learned_state_oracle_action_oracle_decoder", "oracle_state_learned_action_oracle_decoder"} else 0.0
            rows.append(row_from_result(record, policy, diagnostic_result(record, learned, policy), learned, 0.0, 1.0, oracle_input_used))
    return rows


def evaluate_dataset(dataset: list[dict[str, Any]], profile: str, seed: int = SEED, learned: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(seed + 71)
    for record in dataset:
        contract = record["evaluation_contract"]
        if str(contract["family"]) != FAMILY:
            raise ValueError("evaluation contract family mismatch")
        for policy in BASELINE_POLICIES:
            result = control_result(record, policy, rng)
            committed_bits = sparse_read_bits_by_field(profile_caps(profile), int(result.get("selected_record_count", 0))) if policy == "content_routed_sparse_read" else {
                "address": int(result["bits_committed"] // 3),
                "schema": int(result["bits_committed"] // 3),
                "residual": int(result["bits_committed"] - 2 * (result["bits_committed"] // 3)),
            }
            rows.append(
                {
                    "split": record["split"],
                    "seed": int(record["seed"]),
                    "episode_id": record["episode_id"],
                    "family": FAMILY,
                    "policy": policy,
                    "policy_is_learned_result": float(0.0),
                    "policy_is_diagnostic_control": float(0.0),
                    "diagnostic_oracle_input_used": float(0.0),
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
                    "committed_bits_by_field": committed_bits,
                    "total_committed_bits": int(result["bits_committed"]),
                    "address_entropy": float(result["address_entropy"]),
                    "address_margin": float(result["address_margin"]),
                    "read_concentration": float(result["read_concentration"]),
                    "write_frequency": float(result["write_frequency"]),
                    "residual_norm": float(result["residual_norm"]),
                    "reconstruction_error": float(result["reconstruction_error"]),
                    "memory_output_norm": float(result["memory_output_norm"]),
                    "memory_output_vs_residual_norm": float(result["memory_output_norm"] / max(result["residual_norm"], 1e-9)),
                    "selected_record_count": float(result.get("selected_record_count", 0.0)),
                    "source_selection_recall": float(result.get("source_selection_recall", 0.0)),
                    "next_source_selection_recall": float(result.get("next_source_selection_recall", 0.0)),
                    "false_source_selection_rate": float(result.get("false_source_selection_rate", 0.0)),
                    "sparse_read_record_bits": float(result.get("sparse_read_record_bits", 0.0)),
                }
            )
    if learned is not None:
        rows.extend(diagnostic_rows(dataset, learned))
        rows.extend(learned_rows(dataset, learned))
    return rows


def mean_for(rows: list[dict[str, Any]], policy: str, key: str, split: str | None = None) -> float:
    values = [
        float(row[key])
        for row in rows
        if row["policy"] == policy and (split is None or row["split"] == split)
    ]
    return float(np.mean(values)) if values else 0.0


def rows_for(rows: list[dict[str, Any]], policy: str, split: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["policy"] == policy and (split is None or row["split"] == split)
    ]


def code_collision_rate(rows: list[dict[str, Any]], policy: str, split: str = "test") -> float:
    selected = rows_for(rows, policy, split=split)
    if not selected:
        return 0.0
    codes = [
        (
            int(row["compact_code_fields"]["address"]),
            int(row["compact_code_fields"]["schema"]),
            int(row["compact_code_fields"]["residual"]),
            int(row["compact_code_fields"]["action"]),
            int(row["compact_code_fields"]["provenance"]),
        )
        for row in selected
    ]
    return float(1.0 - (len(set(codes)) / max(1.0, float(len(codes)))))


def split_counts(dataset: list[dict[str, Any]]) -> dict[str, int]:
    return {split: int(sum(1 for row in dataset if row["split"] == split)) for split in ("train", "validation", "test")}


def split_metric_summary(rows: list[dict[str, Any]], policy: str, prefix: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for split in ("train", "validation", "test"):
        for source_key, target_suffix in (
            ("joint_success", "joint_success"),
            ("state_probe_accuracy", "state_success"),
            ("action_success", "action_success"),
        ):
            values[f"{prefix}_{split}_{target_suffix}"] = float(mean_for(rows, policy, source_key, split=split))
    values[f"{prefix}_train_test_joint_gap"] = float(values[f"{prefix}_train_joint_success"] - values[f"{prefix}_test_joint_success"])
    return values


def build_summary(dataset: list[dict[str, Any]], rows: list[dict[str, Any]], profile: str = "smoke") -> dict[str, Any]:
    counts = split_counts(dataset)
    caps = profile_caps(profile)
    compressed_bits = mean_for(rows, "compressed_oracle_store", "total_committed_bits", split="test")
    verbatim_bits = mean_for(rows, "verbatim_store", "total_committed_bits", split="test")
    oracle_joint = mean_for(rows, "oracle_codec", "joint_success", split="test")
    random_joint = mean_for(rows, "random_codebook", "joint_success", split="test")
    sparse_read_joint = mean_for(rows, "content_routed_sparse_read", "joint_success", split="test")
    sparse_read_bits = mean_for(rows, "content_routed_sparse_read", "total_committed_bits", split="test")
    learned_joint = mean_for(rows, "learned_codec", "joint_success", split="test")
    learned_state = mean_for(rows, "learned_codec", "state_probe_accuracy", split="test")
    learned_action = mean_for(rows, "learned_codec", "action_success", split="test")
    learned_bits = mean_for(rows, "learned_codec", "total_committed_bits", split="test")
    learned_train_joint = mean_for(rows, "learned_codec", "joint_success", split="train")
    learned_validation_joint = mean_for(rows, "learned_codec", "joint_success", split="validation")
    learned_code_oracle_decoder_joint = mean_for(rows, "learned_code_oracle_decoder", "joint_success", split="test")
    oracle_code_learned_decoder_joint = mean_for(rows, "oracle_code_learned_decoder", "joint_success", split="test")
    learned_address_oracle_payload_joint = mean_for(rows, "learned_address_oracle_payload", "joint_success", split="test")
    oracle_address_learned_payload_joint = mean_for(rows, "oracle_address_learned_payload", "joint_success", split="test")
    provenance_exposed_joint = mean_for(rows, "provenance_exposed_learned_codec", "joint_success", split="test")
    visible_source_codec_joint = mean_for(rows, "visible_source_codec", "joint_success", split="test")
    visible_source_codec_state = mean_for(rows, "visible_source_codec", "state_probe_accuracy", split="test")
    visible_source_codec_action = mean_for(rows, "visible_source_codec", "action_success", split="test")
    visible_source_joint = mean_for(rows, "visible_source_state_oracle_action_oracle_decoder", "joint_success", split="test")
    visible_source_state = mean_for(rows, "visible_source_state_oracle_action_oracle_decoder", "state_probe_accuracy", split="test")
    source_observation_learned_action_joint = mean_for(rows, "source_observation_learned_action", "joint_success", split="test")
    provenance_exposed_oracle_decoder_joint = mean_for(rows, "provenance_exposed_oracle_decoder", "joint_success", split="test")
    provenance_exposed_oracle_decoder_state = mean_for(rows, "provenance_exposed_oracle_decoder", "state_probe_accuracy", split="test")
    provenance_exposed_oracle_decoder_action = mean_for(rows, "provenance_exposed_oracle_decoder", "action_success", split="test")
    learned_state_oracle_action_joint = mean_for(rows, "learned_state_oracle_action_oracle_decoder", "joint_success", split="test")
    learned_state_oracle_action_state = mean_for(rows, "learned_state_oracle_action_oracle_decoder", "state_probe_accuracy", split="test")
    oracle_state_learned_action_joint = mean_for(rows, "oracle_state_learned_action_oracle_decoder", "joint_success", split="test")
    oracle_state_learned_action_action = mean_for(rows, "oracle_state_learned_action_oracle_decoder", "action_success", split="test")
    oracle_decoder_split = split_metric_summary(rows, "oracle_code_learned_decoder", "oracle_code_learned_decoder")
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
        "learned_minus_content_routed_sparse_read": float(learned_joint - sparse_read_joint),
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
        "content_routed_sparse_read_joint_success": float(sparse_read_joint),
        "content_routed_sparse_read_state_success": float(mean_for(rows, "content_routed_sparse_read", "state_probe_accuracy", split="test")),
        "content_routed_sparse_read_action_success": float(mean_for(rows, "content_routed_sparse_read", "action_success", split="test")),
        "content_routed_sparse_read_selected_record_count": float(mean_for(rows, "content_routed_sparse_read", "selected_record_count", split="test")),
        "content_routed_sparse_read_source_selection_recall": float(mean_for(rows, "content_routed_sparse_read", "source_selection_recall", split="test")),
        "content_routed_sparse_read_next_source_selection_recall": float(mean_for(rows, "content_routed_sparse_read", "next_source_selection_recall", split="test")),
        "content_routed_sparse_read_false_source_selection_rate": float(mean_for(rows, "content_routed_sparse_read", "false_source_selection_rate", split="test")),
        "content_routed_sparse_read_total_committed_bits": float(sparse_read_bits),
        "content_routed_sparse_read_within_budget": float(mean_for(rows, "content_routed_sparse_read", "within_budget", split="test")),
        "content_routed_sparse_read_bits_per_successful_episode": None if sparse_read_joint == 0.0 else float(sparse_read_bits / sparse_read_joint),
        "content_routed_sparse_read_compression_ratio_vs_verbatim": float(verbatim_bits / max(sparse_read_bits, 1.0)),
        "no_memory_joint_success": float(mean_for(rows, "no_memory", "joint_success", split="test")),
        "recency_only_joint_success": float(mean_for(rows, "recency_only", "joint_success", split="test")),
        "shuffled_address_joint_success": float(mean_for(rows, "shuffled_address", "joint_success", split="test")),
        "random_codebook_joint_success": float(random_joint),
        "matched_bit_random_code_joint_success": float(mean_for(rows, "matched_bit_random_code", "joint_success", split="test")),
        "matched_compute_no_code_joint_success": float(mean_for(rows, "matched_compute_no_code", "joint_success", split="test")),
        "learned_result_count": int(sum(int(row["policy_is_learned_result"]) for row in rows)),
        "learned_codec_joint_success": float(learned_joint),
        "learned_codec_train_joint_success": float(learned_train_joint),
        "learned_codec_validation_joint_success": float(learned_validation_joint),
        "learned_codec_state_probe_accuracy": float(learned_state),
        "learned_codec_action_success": float(learned_action),
        "learned_codec_encoder_address_accuracy": float(mean_for(rows, "learned_codec", "encoder_address_accuracy", split="test")),
        "learned_codec_encoder_payload_accuracy": float(mean_for(rows, "learned_codec", "encoder_payload_accuracy", split="test")),
        "learned_codec_encoder_payload_color_accuracy": float(mean_for(rows, "learned_codec", "encoder_payload_color_accuracy", split="test")),
        "learned_codec_encoder_payload_shape_accuracy": float(mean_for(rows, "learned_codec", "encoder_payload_shape_accuracy", split="test")),
        "learned_codec_encoder_payload_pos_accuracy": float(mean_for(rows, "learned_codec", "encoder_payload_pos_accuracy", split="test")),
        "learned_codec_encoder_payload_vel_accuracy": float(mean_for(rows, "learned_codec", "encoder_payload_vel_accuracy", split="test")),
        "learned_codec_encoder_action_accuracy": float(mean_for(rows, "learned_codec", "encoder_action_accuracy", split="test")),
        "learned_codec_encoder_provenance_accuracy": float(mean_for(rows, "learned_codec", "encoder_provenance_accuracy", split="test")),
        "source_event_observed_rate": float(np.mean([source_observation_audit(row, caps)["source_event_observed"] for row in dataset if row["split"] == "test"])),
        "source_event_complete_rate": float(np.mean([source_observation_audit(row, caps)["source_event_complete"] for row in dataset if row["split"] == "test"])),
        "source_required_fields_visible_rate": float(np.mean([source_observation_audit(row, caps)["source_required_fields_visible"] for row in dataset if row["split"] == "test"])),
        "source_state_reconstructable_rate": float(np.mean([source_observation_audit(row, caps)["source_state_reconstructable"] for row in dataset if row["split"] == "test"])),
        "source_signature_action_ambiguity_rate": float(action_ambiguity_rate(dataset, caps, "test")),
        "learned_codec_unique_code_count": int(len({
            (
                int(row["compact_code_fields"]["address"]),
                int(row["compact_code_fields"]["schema"]),
                int(row["compact_code_fields"]["residual"]),
                int(row["compact_code_fields"]["action"]),
                int(row["compact_code_fields"]["provenance"]),
            )
            for row in rows_for(rows, "learned_codec", split="test")
        })),
        "learned_codec_code_collision_rate": float(code_collision_rate(rows, "learned_codec")),
        "learned_codec_train_test_joint_gap": float(learned_train_joint - learned_joint),
        "learned_codec_bits_committed_per_successful_episode": learned_bits_per_success,
        "learned_codec_bits_per_success_defined": float(int(learned_bits_per_success is not None)),
        "learned_codec_compression_ratio_vs_verbatim": float(learned_ratio),
        "learned_codec_engineering_pass": float(learned_engineering_pass),
        "learned_codec_paper_track_pass": float(learned_paper_track_pass),
        "learned_codec_kill_condition_count": int(learned_kill_condition_count),
        "diagnostic_result_count": int(sum(int(row.get("policy_is_diagnostic_control", 0.0)) for row in rows)),
        "learned_code_oracle_decoder_joint_success": float(learned_code_oracle_decoder_joint),
        "oracle_code_learned_decoder_joint_success": float(oracle_code_learned_decoder_joint),
        "learned_address_oracle_payload_joint_success": float(learned_address_oracle_payload_joint),
        "oracle_address_learned_payload_joint_success": float(oracle_address_learned_payload_joint),
        "provenance_exposed_learned_codec_joint_success": float(provenance_exposed_joint),
        "visible_source_codec_joint_success": float(visible_source_codec_joint),
        "visible_source_codec_state_success": float(visible_source_codec_state),
        "visible_source_codec_action_success": float(visible_source_codec_action),
        "visible_source_state_oracle_action_oracle_decoder_joint_success": float(visible_source_joint),
        "visible_source_state_oracle_action_oracle_decoder_state_success": float(visible_source_state),
        "source_observation_oracle_action_joint_success": float(visible_source_joint),
        "source_observation_learned_action_joint_success": float(source_observation_learned_action_joint),
        "provenance_exposed_oracle_decoder_joint_success": float(provenance_exposed_oracle_decoder_joint),
        "provenance_exposed_oracle_decoder_state_success": float(provenance_exposed_oracle_decoder_state),
        "provenance_exposed_oracle_decoder_action_success": float(provenance_exposed_oracle_decoder_action),
        "learned_state_oracle_action_joint_success": float(learned_state_oracle_action_joint),
        "learned_state_oracle_action_oracle_decoder_joint_success": float(learned_state_oracle_action_joint),
        "learned_state_oracle_action_oracle_decoder_state_success": float(learned_state_oracle_action_state),
        "oracle_state_learned_action_joint_success": float(oracle_state_learned_action_joint),
        "oracle_state_learned_action_oracle_decoder_joint_success": float(oracle_state_learned_action_joint),
        "oracle_state_learned_action_oracle_decoder_action_success": float(oracle_state_learned_action_action),
        "learned_action_only_failure_rate": float(1.0 - oracle_state_learned_action_action),
        "oracle_decoder_rescue_delta": float(learned_code_oracle_decoder_joint - learned_joint),
        "oracle_encoder_rescue_delta": float(oracle_code_learned_decoder_joint - learned_joint),
        "oracle_address_payload_rescue_delta": float(max(learned_address_oracle_payload_joint, oracle_address_learned_payload_joint) - learned_joint),
        "provenance_exposure_rescue_delta": float(provenance_exposed_joint - learned_joint),
        "provenance_exposed_oracle_decoder_rescue_delta": float(provenance_exposed_oracle_decoder_joint - learned_joint),
        "visible_source_codec_rescue_delta": float(visible_source_codec_joint - learned_joint),
        "source_observation_rescue_delta": float(source_observation_learned_action_joint - learned_joint),
        "content_routed_sparse_read_rescue_delta": float(sparse_read_joint - learned_joint),
        "visible_source_state_rescue_delta": float(visible_source_joint - learned_joint),
        "oracle_action_rescue_delta": float(learned_state_oracle_action_joint - learned_joint),
        "oracle_state_rescue_delta": float(oracle_state_learned_action_joint - learned_joint),
        **oracle_decoder_split,
        "diagnostic_oracle_input_used": float(max((float(row.get("diagnostic_oracle_input_used", 0.0)) for row in rows), default=0.0)),
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
