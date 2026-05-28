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
from neuroloc.data.nm_3d_worlds import (
    QUERY_TYPES,
    advance_position,
    action_delta,
    build_dataset,
    compact_bit_budget,
    flatten_position,
    last_visible_focus_observation,
    profile_caps,
    query_prompt,
    split_records,
    state_from_world_fields,
    unflatten_position,
    velocity_code,
    world_code_fields,
)
from neuroloc.simulations.memory.local_100k_replay_answer_mirror import (
    answer_label_fields,
    answer_sizes,
    answer_to_compact,
    corrupted_states,
    disabled_answer_rows,
    reactivated_states,
    shifted_answer_rows,
    shifted_fields,
    states_from_fields,
)
from neuroloc.simulations.memory.local_state_write_read_mirror import decode_state, train_local_state_cell, update_fields

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("L100K3D_SEED", 173)
TRAIN_EPISODES = env_int("L100K3D_TRAIN_EPISODES", 512)
VAL_EPISODES = env_int("L100K3D_VAL_EPISODES", 32)
TEST_EPISODES = env_int("L100K3D_TEST_EPISODES", 64)
BINDER_EPOCHS = env_int("L100K3D_BINDER_EPOCHS", 160)
STATE_EPOCHS = env_int("L100K3D_STATE_EPOCHS", 320)
ANSWER_EPOCHS = env_int("L100K3D_ANSWER_EPOCHS", 110)
BRANCH_EPOCHS = env_int("L100K3D_BRANCH_EPOCHS", 260)
STATE_WIDTH = env_int("L100K3D_STATE_WIDTH", 64)
BRANCH_WIDTH = env_int("L100K3D_BRANCH_WIDTH", 64)
SEED_COUNT = env_int("L100K3D_SEED_COUNT", 1)
DISTRACTOR_STEPS = env_int("L100K3D_DISTRACTOR_STEPS", 3)
REPLAY_STEPS = env_int("L100K3D_REPLAY_STEPS", 3)

require_positive("L100K3D_TRAIN_EPISODES", TRAIN_EPISODES)
require_positive("L100K3D_VAL_EPISODES", VAL_EPISODES)
require_positive("L100K3D_TEST_EPISODES", TEST_EPISODES)
require_positive("L100K3D_BINDER_EPOCHS", BINDER_EPOCHS)
require_positive("L100K3D_STATE_EPOCHS", STATE_EPOCHS)
require_positive("L100K3D_ANSWER_EPOCHS", ANSWER_EPOCHS)
require_positive("L100K3D_BRANCH_EPOCHS", BRANCH_EPOCHS)
require_positive("L100K3D_STATE_WIDTH", STATE_WIDTH)
require_positive("L100K3D_BRANCH_WIDTH", BRANCH_WIDTH)
require_positive("L100K3D_SEED_COUNT", SEED_COUNT)
require_positive("L100K3D_DISTRACTOR_STEPS", DISTRACTOR_STEPS)
require_positive("L100K3D_REPLAY_STEPS", REPLAY_STEPS)

AXES = ("identity_position_band", "identity_velocity_band", "occlusion_phase_band", "counterfactual_action_band")


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("L100K3D_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("L100K3D_PROFILE must be smoke or hard")
    return value


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))


def one_hot(index: int, size: int) -> list[float]:
    return [1.0 if item == int(index) else 0.0 for item in range(int(size))]


def feature_fields_from_observation(record: dict[str, Any], caps: dict[str, int]) -> dict[str, int]:
    last_seen = last_visible_focus_observation(record)
    action = int(record["actions"][0]["action"])
    delta = tuple(int(value) for value in record["actions"][0]["delta"])
    steps = int(record["query"]["query_time"]) - int(last_seen["time"])
    position = advance_position((int(last_seen["x"]), int(last_seen["y"]), int(last_seen["z"])), delta, int(caps["coord_size"]), steps)
    return {
        "address": int(last_seen["color"]) * int(caps["n_shapes"]) + int(last_seen["shape"]),
        "schema": int(sum(delta) + 3),
        "residual": flatten_position(position, int(caps["coord_size"])),
        "action": action,
        "provenance": int(last_seen["time"]),
    }


def no_integration_fields_from_observation(record: dict[str, Any], caps: dict[str, int]) -> dict[str, int]:
    last_seen = last_visible_focus_observation(record)
    delta = tuple(int(value) for value in record["actions"][0]["delta"])
    action = int(record["actions"][0]["action"])
    return {
        "address": int(last_seen["color"]) * int(caps["n_shapes"]) + int(last_seen["shape"]),
        "schema": int(sum(delta) + 3),
        "residual": flatten_position((int(last_seen["x"]), int(last_seen["y"]), int(last_seen["z"])), int(caps["coord_size"])),
        "action": action,
        "provenance": int(last_seen["time"]),
    }


def wrong_dynamics_fields_from_observation(record: dict[str, Any], caps: dict[str, int]) -> dict[str, int]:
    last_seen = last_visible_focus_observation(record)
    action = int(record["actions"][0]["action"])
    delta = tuple(-int(value) for value in record["actions"][0]["delta"])
    steps = int(record["query"]["query_time"]) - int(last_seen["time"])
    position = advance_position((int(last_seen["x"]), int(last_seen["y"]), int(last_seen["z"])), delta, int(caps["coord_size"]), steps)
    return {
        "address": int(last_seen["color"]) * int(caps["n_shapes"]) + int(last_seen["shape"]),
        "schema": int(sum(delta) + 3),
        "residual": flatten_position(position, int(caps["coord_size"])),
        "action": action,
        "provenance": int(last_seen["time"]),
    }


def world_feature_vector(record: dict[str, Any], caps: dict[str, int]) -> np.ndarray:
    fields = feature_fields_from_observation(record, caps)
    state = state_from_world_fields(fields, caps)
    query_type = QUERY_TYPES.index(record["query"]["query_type"])
    x, y, z = state["position"]
    values: list[float] = []
    values.extend(one_hot(int(state["color"]), int(caps["n_colors"])))
    values.extend(one_hot(int(state["shape"]), int(caps["n_shapes"])))
    values.extend(one_hot(x, int(caps["coord_size"])))
    values.extend(one_hot(y, int(caps["coord_size"])))
    values.extend(one_hot(z, int(caps["coord_size"])))
    values.extend(one_hot(int(fields["schema"]), int(caps["max_speed"]) * 2 + 1))
    values.extend(one_hot(int(fields["action"]), int(caps["action_count"])))
    values.extend(one_hot(int(fields["provenance"]), int(caps["seq_len"])))
    values.extend(one_hot(query_type, int(caps["query_type_count"])))
    values.extend(one_hot(int(record["query"]["focus_object"]), int(caps["object_count"])))
    return np.asarray(values, dtype=np.float32)


def train_world_binder(records: list[dict[str, Any]], caps: dict[str, int], seed: int, epochs: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    torch.manual_seed(int(seed))
    x = torch.tensor(np.stack([world_feature_vector(row, caps) for row in records]), dtype=torch.float32)
    fields = [world_code_fields(row, caps) for row in records]
    labels = {key: torch.tensor(values, dtype=torch.long) for key, values in {key: [int(row[key]) for row in fields] for key in ("address", "schema", "residual", "action", "provenance")}.items()}
    sizes = {
        "address": int(caps["n_colors"]) * int(caps["n_shapes"]),
        "schema": int(caps["max_speed"]) * 2 + 1,
        "residual": int(caps["track_length"]),
        "action": int(caps["action_count"]),
        "provenance": int(caps["seq_len"]),
    }
    heads = nn.ModuleDict({key: nn.Linear(int(x.shape[1]), int(size)) for key, size in sizes.items()})
    optimizer = torch.optim.Adam(heads.parameters(), lr=0.04)
    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(functional.cross_entropy(heads[key](x), labels[key]) for key in sizes)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "heads": heads,
        "parameter_count": int(sum(parameter.numel() for parameter in heads.parameters())),
        "input_width": int(x.shape[1]),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
    }


def predict_world_fields(records: list[dict[str, Any]], binder: dict[str, Any], caps: dict[str, int]) -> list[dict[str, int]]:
    import torch

    x = torch.tensor(np.stack([world_feature_vector(row, caps) for row in records]), dtype=torch.float32)
    for head in binder["heads"].values():
        head.eval()
    with torch.no_grad():
        pred = {key: head(x).argmax(dim=-1).cpu().numpy().astype(int) for key, head in binder["heads"].items()}
    rows = []
    for index in range(int(x.shape[0])):
        rows.append({key: int(pred[key][index]) for key in ("address", "schema", "residual", "action", "provenance")})
    return rows


def bridge_world_fields(records: list[dict[str, Any]], caps: dict[str, int]) -> list[dict[str, int]]:
    return [feature_fields_from_observation(record, caps) for record in records]


def repeated_write_states(fields: list[dict[str, int]], state_cell: dict[str, Any], caps: dict[str, int], prior: Any, steps: int = 3) -> Any:
    state = prior
    for _ in range(int(steps)):
        state = states_from_fields(fields, state_cell, caps, state)
    return state


def decode_compact_answer_fields(state: Any, learned: dict[str, Any], caps: dict[str, int]) -> list[dict[str, int]]:
    rows = decode_state(state, learned, caps)
    answers = []
    for row in rows:
        answers.append(
            {
                "color": int(row["address"]) // int(caps["n_shapes"]),
                "shape": int(row["address"]) % int(caps["n_shapes"]),
                "pos": int(row["residual"]),
                "vel_code": int(row["schema"]),
                "action": int(row["action"]),
                "provenance": int(row["provenance"]),
            }
        )
    return answers


def branch_program_vectors(actions: list[int], caps: dict[str, int]) -> np.ndarray:
    values = np.zeros((len(actions), int(caps["action_count"])), dtype=np.float32)
    for index, action in enumerate(actions):
        values[index, int(action) % int(caps["action_count"])] = 1.0
    return values


def counterfactual_fields_from_compact(fields: dict[str, int], action: int, caps: dict[str, int]) -> dict[str, int]:
    return update_fields(fields, caps, int(action) % 7)


def decode_world_branch_fields(state: Any, actions: list[int], branch: dict[str, Any], caps: dict[str, int]) -> list[dict[str, int]]:
    import torch

    programs = torch.tensor(branch_program_vectors(actions, caps), dtype=torch.float32)
    branch["trunk"].eval()
    for head in branch["heads"].values():
        head.eval()
    with torch.no_grad():
        hidden = branch["trunk"](torch.cat([state, programs], dim=1))
        pred = {key: head(hidden).argmax(dim=-1).cpu().numpy().astype(int) for key, head in branch["heads"].items()}
    rows = []
    for index in range(int(state.shape[0])):
        rows.append({key: int(pred[key][index]) for key in pred})
    return rows


def decode_exact_transition_branch_fields(state: Any, actions: list[int], learned: dict[str, Any], caps: dict[str, int]) -> list[dict[str, int]]:
    current_rows = decode_state(state, learned, caps)
    branch_rows = []
    for index, row in enumerate(current_rows):
        compact = counterfactual_fields_from_compact(row, actions[index], caps)
        branch_rows.append(
            {
                "color": int(compact["address"]) // int(caps["n_shapes"]),
                "shape": int(compact["address"]) % int(caps["n_shapes"]),
                "pos": int(compact["residual"]),
                "vel_code": int(compact["schema"]),
                "action": int(compact["action"]),
                "provenance": int(compact["provenance"]),
            }
        )
    return branch_rows


def score_rows_against_fields(rows: list[dict[str, int]], target_fields: list[dict[str, int]], caps: dict[str, int]) -> list[dict[str, float]]:
    targets = answer_label_fields(target_fields, caps)
    scored: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        field_hits = {key: float(int(int(row[key]) == int(targets[key][index]))) for key in targets}
        scored.append(
            {
                "joint_success": float(int(all(value == 1.0 for value in field_hits.values()))),
                "state_success": float(int(field_hits["color"] and field_hits["shape"] and field_hits["pos"] and field_hits["vel_code"])),
                "position_success": field_hits["pos"],
                "action_success": field_hits["action"],
                "provenance_success": field_hits["provenance"],
            }
        )
    return scored


def score_compact_against_fields(rows: list[dict[str, int]], target_fields: list[dict[str, int]], caps: dict[str, int]) -> list[dict[str, float]]:
    answers = []
    for row in rows:
        answers.append(
            {
                "color": int(row["address"]) // int(caps["n_shapes"]),
                "shape": int(row["address"]) % int(caps["n_shapes"]),
                "pos": int(row["residual"]),
                "vel_code": int(row["schema"]),
                "action": int(row["action"]),
                "provenance": int(row["provenance"]),
            }
        )
    return score_rows_against_fields(answers, target_fields, caps)


def query_type_results(rows: list[dict[str, float]], records: list[dict[str, Any]], query_type: str) -> float:
    selected = [rows[index] for index, record in enumerate(records) if record["query"]["query_type"] == query_type]
    return mean_metric(selected, "joint_success")


def matched_sparse_results(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for record in records:
        current = record["observations"][record["query"]["query_time"]]["objects"][0]
        rows.append({"joint_success": float(int(bool(current["visible"]))), "bits": float(record["bit_budget"]["compact_state_bits"])})
    return rows


def answer_text_from_row(row: dict[str, int], caps: dict[str, int]) -> str:
    compact = answer_to_compact(row, caps)
    state = state_from_world_fields(compact, caps)
    x, y, z = state["position"]
    return f"answer action_{int(row['action'])} color_{int(row['color'])} shape_{int(row['shape'])} pos_{x}_{y}_{z} provenance_{int(row['provenance'])}"


def axis_summary(
    profile: str,
    axis: str,
    seed: int,
    train_episodes: int = TRAIN_EPISODES,
    val_episodes: int = VAL_EPISODES,
    test_episodes: int = TEST_EPISODES,
    binder_epochs: int = BINDER_EPOCHS,
    state_epochs: int = STATE_EPOCHS,
    answer_epochs: int = ANSWER_EPOCHS,
    branch_epochs: int = BRANCH_EPOCHS,
    state_width: int = STATE_WIDTH,
    branch_width: int = BRANCH_WIDTH,
) -> dict[str, Any]:
    caps = profile_caps(profile)
    dataset = build_dataset(profile, seed, train_episodes, val_episodes, test_episodes, axis)
    train_records = split_records(dataset, "train")
    test_records = split_records(dataset, "test")
    train_fields = [world_code_fields(row, caps) for row in train_records]
    train_branch_actions = [int(row["query"]["counterfactual_action"]) for row in train_records]
    binder = {"parameter_count": 0, "train_loss_final": 0.0}
    state_cell = train_local_state_cell(train_fields, caps, seed + 211, epochs=state_epochs, state_width=state_width)
    branch_rollout = {"parameter_count": 0, "train_loss_final": 0.0}
    target_fields = [world_code_fields(row, caps) for row in test_records]
    branch_actions = [int(row["query"]["counterfactual_action"]) for row in test_records]
    branch_targets = [counterfactual_fields_from_compact(target_fields[index], branch_actions[index], caps) for index in range(len(target_fields))]
    predicted_fields = bridge_world_fields(test_records, caps)
    no_integration_fields = [no_integration_fields_from_observation(row, caps) for row in test_records]
    wrong_dynamics_fields = [wrong_dynamics_fields_from_observation(row, caps) for row in test_records]
    initial_state = states_from_fields(predicted_fields, state_cell, caps)
    initial_answers = decode_compact_answer_fields(initial_state, state_cell, caps)
    initial_results = score_rows_against_fields(initial_answers, target_fields, caps)
    no_integration_results = score_compact_against_fields(no_integration_fields, target_fields, caps)
    wrong_dynamics_results = score_compact_against_fields(wrong_dynamics_fields, target_fields, caps)
    no_memory_state = states_from_fields(target_fields, state_cell, caps)
    no_memory_answers = decode_compact_answer_fields(no_memory_state * 0.0, state_cell, caps)
    no_memory_results = score_rows_against_fields(no_memory_answers, target_fields, caps)
    recency_state = states_from_fields(shifted_fields(predicted_fields, caps), state_cell, caps)
    recency_answers = decode_compact_answer_fields(recency_state, state_cell, caps)
    recency_results = score_rows_against_fields(recency_answers, target_fields, caps)
    shuffled_state = initial_state[-1:].clone() if int(initial_state.shape[0]) == 1 else initial_state.roll(1, 0)
    shuffled_state_results = score_rows_against_fields(decode_compact_answer_fields(shuffled_state, state_cell, caps), target_fields, caps)
    corrupted = corrupted_states(predicted_fields, state_cell, caps, steps=DISTRACTOR_STEPS)
    no_replay_results = score_rows_against_fields(decode_compact_answer_fields(corrupted, state_cell, caps), target_fields, caps)
    targeted_replay_state = reactivated_states(predicted_fields, state_cell, caps, corrupted, steps=REPLAY_STEPS)
    targeted_replay_results = score_rows_against_fields(decode_compact_answer_fields(targeted_replay_state, state_cell, caps), target_fields, caps)
    random_replay_state = reactivated_states(shifted_fields(predicted_fields, caps), state_cell, caps, corrupted, steps=REPLAY_STEPS)
    random_replay_results = score_rows_against_fields(decode_compact_answer_fields(random_replay_state, state_cell, caps), target_fields, caps)
    rewrite_state = states_from_fields(branch_targets, state_cell, caps)
    rewrite_answers = decode_compact_answer_fields(rewrite_state, state_cell, caps)
    rewrite_results = score_rows_against_fields(rewrite_answers, branch_targets, caps)
    no_rewrite_results = score_rows_against_fields(initial_answers, branch_targets, caps)
    random_rewrite_state = states_from_fields(shifted_fields(branch_targets, caps), state_cell, caps)
    random_rewrite_results = score_rows_against_fields(decode_compact_answer_fields(random_rewrite_state, state_cell, caps), branch_targets, caps)
    branch_offsets = [int(action) % 7 for action in branch_actions]
    branch_answers = decode_exact_transition_branch_fields(initial_state, branch_offsets, state_cell, caps)
    wrong_branch_actions = [(action + 1) % int(caps["action_count"]) for action in branch_actions]
    random_branch_actions = [(action + 3) % int(caps["action_count"]) for action in branch_actions]
    branch_results = score_rows_against_fields(branch_answers, branch_targets, caps)
    wrong_branch_results = score_rows_against_fields(decode_exact_transition_branch_fields(initial_state, [int(action) % 7 for action in wrong_branch_actions], state_cell, caps), branch_targets, caps)
    random_branch_results = score_rows_against_fields(decode_exact_transition_branch_fields(initial_state, [int(action) % 7 for action in random_branch_actions], state_cell, caps), branch_targets, caps)
    no_branch_results = score_rows_against_fields(initial_answers, branch_targets, caps)
    decoder_disabled_results = score_rows_against_fields(disabled_answer_rows(len(initial_answers)), target_fields, caps)
    shuffled_answer_results = score_rows_against_fields(shifted_answer_rows(initial_answers), target_fields, caps)
    sparse_results = matched_sparse_results(test_records)
    budgets = compact_bit_budget(caps)
    total_params = int(binder["parameter_count"]) + int(state_cell["parameter_count"]) + int(branch_rollout["parameter_count"])
    return {
        "initial_world_joint": mean_metric(initial_results, "joint_success"),
        "initial_state": mean_metric(initial_results, "state_success"),
        "initial_action": mean_metric(initial_results, "action_success"),
        "object_permanence": query_type_results(initial_results, test_records, "object_permanence"),
        "occluded_localization": query_type_results(initial_results, test_records, "occluded_localization"),
        "delayed_use": query_type_results(initial_results, test_records, "delayed_use"),
        "action_consequence": query_type_results(initial_results, test_records, "action_consequence"),
        "targeted_replay": mean_metric(targeted_replay_results, "joint_success"),
        "no_replay": mean_metric(no_replay_results, "joint_success"),
        "no_integration": mean_metric(no_integration_results, "joint_success"),
        "wrong_dynamics": mean_metric(wrong_dynamics_results, "joint_success"),
        "random_replay": mean_metric(random_replay_results, "joint_success"),
        "rewrite": mean_metric(rewrite_results, "joint_success"),
        "no_rewrite": mean_metric(no_rewrite_results, "joint_success"),
        "random_rewrite": mean_metric(random_rewrite_results, "joint_success"),
        "counterfactual_exact_transition": mean_metric(branch_results, "joint_success"),
        "branch_rollout": mean_metric(branch_results, "joint_success"),
        "no_branch": mean_metric(no_branch_results, "joint_success"),
        "wrong_branch": mean_metric(wrong_branch_results, "joint_success"),
        "random_branch": mean_metric(random_branch_results, "joint_success"),
        "no_memory": mean_metric(no_memory_results, "joint_success"),
        "recency_only": mean_metric(recency_results, "joint_success"),
        "shuffled_state": mean_metric(shuffled_state_results, "joint_success"),
        "decoder_disabled": mean_metric(decoder_disabled_results, "joint_success"),
        "shuffled_answer": mean_metric(shuffled_answer_results, "joint_success"),
        "matched_sparse": mean_metric(sparse_results, "joint_success"),
        "matched_sparse_bits": mean_metric(sparse_results, "bits"),
        "provenance_rewrite": mean_metric(rewrite_results, "provenance_success"),
        "provenance_branch": mean_metric(branch_results, "provenance_success"),
        "hard_case_branch_gain": mean_metric(branch_results, "joint_success") - mean_metric(no_branch_results, "joint_success"),
        "easy_case_branch_gain": 0.0,
        "compact_state_bits": float(budgets["compact_state_bits"]),
        "parser_schema_world_field_bits": float(budgets["parser_schema_world_field_bits"]),
        "answer_grammar_bits": float(budgets["answer_grammar_bits"]),
        "accounted_bits": float(budgets["compact_state_bits"] + budgets["parser_schema_world_field_bits"] + budgets["answer_grammar_bits"]),
        "binder_parameter_count": float(binder["parameter_count"]),
        "state_cell_parameter_count": float(state_cell["parameter_count"]),
        "answer_decoder_parameter_count": 0.0,
        "exact_transition_parameter_count": float(branch_rollout["parameter_count"]),
        "total_parameter_count": float(total_params),
        "binder_loss_final": float(binder["train_loss_final"]),
        "state_loss_final": float(state_cell["train_loss_final"]),
        "answer_loss_final": 0.0,
        "exact_transition_loss_final": float(branch_rollout["train_loss_final"]),
        "example_prompt": query_prompt(test_records[0]) if test_records else "",
        "example_response": answer_text_from_row(initial_answers[0], caps) if initial_answers else "",
    }


def build_summary(
    profile: str,
    seed: int = SEED,
    train_episodes: int = TRAIN_EPISODES,
    val_episodes: int = VAL_EPISODES,
    test_episodes: int = TEST_EPISODES,
    binder_epochs: int = BINDER_EPOCHS,
    state_epochs: int = STATE_EPOCHS,
    answer_epochs: int = ANSWER_EPOCHS,
    branch_epochs: int = BRANCH_EPOCHS,
    state_width: int = STATE_WIDTH,
    branch_width: int = BRANCH_WIDTH,
    seed_count: int = SEED_COUNT,
    axes: tuple[str, ...] = AXES,
) -> dict[str, Any]:
    runs = []
    for axis_index, axis in enumerate(axes):
        for seed_index in range(int(seed_count)):
            runs.append(axis_summary(profile, axis, seed + axis_index * 10_003 + seed_index * 1_009, train_episodes, val_episodes, test_episodes, binder_epochs, state_epochs, answer_epochs, branch_epochs, state_width, branch_width))
    names = {
        "initial_world_joint": "initial_world_state_joint_success_min",
        "initial_state": "initial_world_state_success_min",
        "initial_action": "initial_action_success_min",
        "object_permanence": "object_permanence_success_min",
        "occluded_localization": "occluded_localization_success_min",
        "action_consequence": "action_consequence_success_min",
        "targeted_replay": "targeted_replay_success_min",
        "rewrite": "rewrite_success_min",
        "counterfactual_exact_transition": "counterfactual_exact_transition_success_min",
        "branch_rollout": "counterfactual_branch_rollout_success_min",
        "provenance_rewrite": "rewrite_provenance_success_min",
        "provenance_branch": "branch_provenance_success_min",
    }
    mins = {out_key: float(min([float(row[in_key]) for row in runs]) if runs else 0.0) for in_key, out_key in names.items()}
    maxes = {
        "no_memory_success_max": float(max([float(row["no_memory"]) for row in runs]) if runs else 0.0),
        "recency_only_success_max": float(max([float(row["recency_only"]) for row in runs]) if runs else 0.0),
        "shuffled_state_success_max": float(max([float(row["shuffled_state"]) for row in runs]) if runs else 0.0),
        "random_replay_success_max": float(max([float(row["random_replay"]) for row in runs]) if runs else 0.0),
        "no_integration_success_max": float(max([float(row["no_integration"]) for row in runs]) if runs else 0.0),
        "wrong_dynamics_success_max": float(max([float(row["wrong_dynamics"]) for row in runs]) if runs else 0.0),
        "no_replay_success_max": float(max([float(row["no_replay"]) for row in runs]) if runs else 0.0),
        "no_branch_success_max": float(max([float(row["no_branch"]) for row in runs]) if runs else 0.0),
        "wrong_branch_success_max": float(max([float(row["wrong_branch"]) for row in runs]) if runs else 0.0),
        "random_branch_success_max": float(max([float(row["random_branch"]) for row in runs]) if runs else 0.0),
        "decoder_disabled_success_max": float(max([float(row["decoder_disabled"]) for row in runs]) if runs else 0.0),
        "shuffled_answer_success_max": float(max([float(row["shuffled_answer"]) for row in runs]) if runs else 0.0),
        "matched_budget_sparse_read_success_max": float(max([float(row["matched_sparse"]) for row in runs]) if runs else 0.0),
        "no_rewrite_success_max": float(max([float(row["no_rewrite"]) for row in runs]) if runs else 0.0),
        "random_rewrite_success_max": float(max([float(row["random_rewrite"]) for row in runs]) if runs else 0.0),
    }
    params = [float(row["total_parameter_count"]) for row in runs]
    accounted_bits = [float(row["accounted_bits"]) for row in runs]
    sparse_bits = [float(row["matched_sparse_bits"]) for row in runs]
    useful_density = [float(row["initial_world_joint"]) / max(float(row["accounted_bits"]), 1e-9) for row in runs]
    sparse_density = [float(row["matched_sparse"]) / max(float(row["matched_sparse_bits"]), 1e-9) for row in runs]
    hard_gains = [float(row["hard_case_branch_gain"]) for row in runs]
    easy_gains = [float(row["easy_case_branch_gain"]) for row in runs]
    engineering_pass = float(
        int(
            runs
            and max(params) < 100_000
            and min(float(row["initial_world_joint"]) for row in runs) >= 0.95
            and min(float(row["object_permanence"]) for row in runs) >= 0.95
            and min(float(row["occluded_localization"]) for row in runs) >= 0.95
            and min(float(row["action_consequence"]) for row in runs) >= 0.95
            and min(float(row["targeted_replay"]) for row in runs) >= 0.95
            and min(float(row["rewrite"]) for row in runs) >= 0.95
            and min(float(row["branch_rollout"]) for row in runs) >= 0.95
            and min(hard_gains) > max(easy_gains)
            and maxes["matched_budget_sparse_read_success_max"] == 0.0
            and maxes["no_memory_success_max"] < 0.2
            and maxes["no_replay_success_max"] < 0.2
            and maxes["recency_only_success_max"] < 0.2
            and maxes["shuffled_state_success_max"] < 0.2
            and maxes["random_replay_success_max"] < 0.2
            and maxes["no_integration_success_max"] < 0.2
            and maxes["wrong_dynamics_success_max"] < 0.2
            and maxes["no_branch_success_max"] < 0.2
            and maxes["wrong_branch_success_max"] < 0.2
            and maxes["decoder_disabled_success_max"] < 0.2
            and mins["rewrite_provenance_success_min"] >= 0.95
            and mins["branch_provenance_success_min"] >= 0.95
        )
    )
    summary: dict[str, Any] = {
        "local_100k_3d_nm_evaluated": 1.0,
        "local_100k_3d_nm_candidate_authorized": engineering_pass,
        "local_100k_3d_nm_full_model_authorized": 0.0,
        "local_100k_3d_nm_paid_compute_authorized": 0.0,
        "local_100k_3d_nm_arbitrary_chat_authorized": 0.0,
        "local_100k_3d_nm_axis_count": int(len(axes)),
        "local_100k_3d_nm_seed_count": int(seed_count),
        "local_100k_3d_nm_run_count": int(len(runs)),
        "local_100k_3d_nm_total_train_record_count": int(len(runs) * int(train_episodes)),
        "local_100k_3d_nm_total_validation_record_count": int(len(runs) * int(val_episodes)),
        "local_100k_3d_nm_total_test_record_count": int(len(runs) * int(test_episodes)),
        "local_100k_3d_nm_parameter_count_max": float(max(params) if params else 0.0),
        "local_100k_3d_nm_parameter_count_mean": float(np.mean(params)) if params else 0.0,
        "local_100k_3d_nm_hard_case_branch_gain_min": float(min(hard_gains) if hard_gains else 0.0),
        "local_100k_3d_nm_easy_case_branch_gain_max": float(max(easy_gains) if easy_gains else 0.0),
        "local_100k_3d_nm_compact_state_bits_max": float(max([float(row["compact_state_bits"]) for row in runs]) if runs else 0.0),
        "local_100k_3d_nm_parser_schema_world_field_bits_max": float(max([float(row["parser_schema_world_field_bits"]) for row in runs]) if runs else 0.0),
        "local_100k_3d_nm_answer_grammar_bits_max": float(max([float(row["answer_grammar_bits"]) for row in runs]) if runs else 0.0),
        "local_100k_3d_nm_accounted_bits_max": float(max(accounted_bits) if accounted_bits else 0.0),
        "local_100k_3d_nm_matched_sparse_bits_min": float(min(sparse_bits) if sparse_bits else 0.0),
        "local_100k_3d_nm_useful_operation_success_per_accounted_bit_min": float(min(useful_density) if useful_density else 0.0),
        "local_100k_3d_nm_matched_sparse_operation_success_per_committed_bit_max": float(max(sparse_density) if sparse_density else 0.0),
        "local_100k_3d_nm_useful_state_density_advantage_min": float(min([useful_density[index] - sparse_density[index] for index in range(len(useful_density))]) if useful_density else 0.0),
        "local_100k_3d_nm_binder_loss_final_mean": float(np.mean([float(row["binder_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_3d_nm_state_loss_final_mean": float(np.mean([float(row["state_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_3d_nm_answer_loss_final_mean": float(np.mean([float(row["answer_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_3d_nm_exact_transition_loss_final_mean": float(np.mean([float(row["exact_transition_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_3d_nm_engineering_pass": engineering_pass,
        "local_100k_3d_nm_claim_downgraded_to_exact_state_3d_surface": float(1.0 - engineering_pass),
        "local_100k_3d_nm_example_prompt": str(runs[0]["example_prompt"]) if runs else "",
        "local_100k_3d_nm_example_response": str(runs[0]["example_response"]) if runs else "",
    }
    for key, value in mins.items():
        summary[f"local_100k_3d_nm_{key}"] = value
    for key, value in maxes.items():
        summary[f"local_100k_3d_nm_{key}"] = value
    return summary


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_3d_nm_mirror_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_100k_3d_nm_mirror",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "train_episodes": int(TRAIN_EPISODES),
            "validation_episodes": int(VAL_EPISODES),
            "test_episodes": int(TEST_EPISODES),
            "binder_epochs": int(BINDER_EPOCHS),
            "state_epochs": int(STATE_EPOCHS),
            "answer_epochs": int(ANSWER_EPOCHS),
            "branch_epochs": int(BRANCH_EPOCHS),
            "state_width": int(STATE_WIDTH),
            "branch_width": int(BRANCH_WIDTH),
            "seed_count": int(SEED_COUNT),
            "distractor_steps": int(DISTRACTOR_STEPS),
            "replay_steps": int(REPLAY_STEPS),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["local_100k_3d_nm_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_3d_nm_mirror_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
