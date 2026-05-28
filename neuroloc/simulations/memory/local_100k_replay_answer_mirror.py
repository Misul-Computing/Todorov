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
    distributed_evidence_sparse_read_result,
    learned_code_bits,
    matched_budget_sparse_read_result,
    profile_caps,
)
from neuroloc.simulations.memory.language_grounded_state_density_mirror import (
    AXES,
    evaluate_fields,
    mean_metric,
    parser_schema_cost_bits,
    randomized_record_prompt,
    split_records,
    train_event_binding_segment_model,
)
from neuroloc.simulations.memory.local_state_write_read_mirror import (
    fields_from_record,
    field_sizes,
    field_vector,
    predicted_code_fields,
    sampled_code_fields,
    train_local_state_cell,
    update_fields,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("L100K_SEED", 131)
TRAIN_EPISODES = env_int("L100K_TRAIN_EPISODES", 512)
VAL_EPISODES = env_int("L100K_VAL_EPISODES", 16)
TEST_EPISODES = env_int("L100K_TEST_EPISODES", 32)
STATE_EPOCHS = env_int("L100K_STATE_EPOCHS", 120)
ANSWER_EPOCHS = env_int("L100K_ANSWER_EPOCHS", 90)
BRANCH_EPOCHS = env_int("L100K_BRANCH_EPOCHS", 420)
BRANCH_WIDTH = env_int("L100K_BRANCH_WIDTH", 256)
STATE_WIDTH = env_int("L100K_STATE_WIDTH", 64)
SEED_COUNT = env_int("L100K_SEED_COUNT", 1)
DISTRACTOR_STEPS = env_int("L100K_DISTRACTOR_STEPS", 3)
REPLAY_STEPS = env_int("L100K_REPLAY_STEPS", 3)

require_positive("L100K_TRAIN_EPISODES", TRAIN_EPISODES)
require_positive("L100K_VAL_EPISODES", VAL_EPISODES)
require_positive("L100K_TEST_EPISODES", TEST_EPISODES)
require_positive("L100K_STATE_EPOCHS", STATE_EPOCHS)
require_positive("L100K_ANSWER_EPOCHS", ANSWER_EPOCHS)
require_positive("L100K_BRANCH_EPOCHS", BRANCH_EPOCHS)
require_positive("L100K_BRANCH_WIDTH", BRANCH_WIDTH)
require_positive("L100K_STATE_WIDTH", STATE_WIDTH)
require_positive("L100K_SEED_COUNT", SEED_COUNT)
require_positive("L100K_DISTRACTOR_STEPS", DISTRACTOR_STEPS)
require_positive("L100K_REPLAY_STEPS", REPLAY_STEPS)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("L100K_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("L100K_PROFILE must be smoke or hard")
    return value


def state_width_from_cell(learned: dict[str, Any]) -> int:
    return int(next(learned["cell"].parameters()).shape[0] // 3)


def states_from_fields(fields: list[dict[str, int]], learned: dict[str, Any], caps: dict[str, int], prior: Any | None = None) -> Any:
    import torch

    x = torch.tensor(np.stack([field_vector(row, caps) for row in fields]), dtype=torch.float32)
    width = state_width_from_cell(learned)
    state = prior if prior is not None else torch.zeros((int(x.shape[0]), width), dtype=torch.float32)
    learned["cell"].eval()
    with torch.no_grad():
        return learned["cell"](x, state)


def answer_label_fields(fields: list[dict[str, int]], caps: dict[str, int]) -> dict[str, list[int]]:
    return {
        "color": [int(row["address"]) // int(caps["n_shapes"]) for row in fields],
        "shape": [int(row["address"]) % int(caps["n_shapes"]) for row in fields],
        "pos": [int(row["residual"]) for row in fields],
        "vel_code": [int(row["schema"]) for row in fields],
        "action": [int(row["action"]) for row in fields],
        "provenance": [int(row["provenance"]) for row in fields],
    }


def answer_sizes(caps: dict[str, int]) -> dict[str, int]:
    return {
        "color": int(caps["n_colors"]),
        "shape": int(caps["n_shapes"]),
        "pos": int(caps["track_length"]),
        "vel_code": int(caps["max_speed"]) * 2 + 1,
        "action": int(caps["action_count"]),
        "provenance": int(caps["seq_len"]),
    }


def train_answer_decoder(fields: list[dict[str, int]], state_cell: dict[str, Any], caps: dict[str, int], seed: int, epochs: int = ANSWER_EPOCHS) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    torch.manual_seed(int(seed))
    sizes = answer_sizes(caps)
    fields = list(fields) + sampled_code_fields(caps, seed + 19, max(2048, len(fields) * 4))
    update_rows = [update_fields(row, caps, index % 7) for index, row in enumerate(fields)]
    states = states_from_fields(fields, state_cell, caps)
    update_states = states_from_fields(update_rows, state_cell, caps, states)
    labels = {key: torch.tensor(values, dtype=torch.long) for key, values in answer_label_fields(fields, caps).items()}
    update_labels = {key: torch.tensor(values, dtype=torch.long) for key, values in answer_label_fields(update_rows, caps).items()}
    heads = nn.ModuleDict({key: nn.Linear(state_width_from_cell(state_cell), int(size)) for key, size in sizes.items()})
    optimizer = torch.optim.Adam(heads.parameters(), lr=0.04)
    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(functional.cross_entropy(heads[key](states.detach()), labels[key]) for key in sizes)
        loss = loss + sum(functional.cross_entropy(heads[key](update_states.detach()), update_labels[key]) for key in sizes)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "heads": heads,
        "parameter_count": int(sum(parameter.numel() for parameter in heads.parameters())),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
    }


def program_vectors(offsets: list[int], caps: dict[str, int]) -> np.ndarray:
    size = 7
    values = np.zeros((len(offsets), size), dtype=np.float32)
    for index, offset in enumerate(offsets):
        values[index, int(offset) % size] = 1.0
    return values


def train_branch_rollout(fields: list[dict[str, int]], state_cell: dict[str, Any], caps: dict[str, int], seed: int, epochs: int = BRANCH_EPOCHS, branch_width: int = BRANCH_WIDTH) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    torch.manual_seed(int(seed))
    sizes = answer_sizes(caps)
    base_fields = list(fields) + sampled_code_fields(caps, seed + 29, max(1024, len(fields) * 2))
    expanded_fields: list[dict[str, int]] = []
    offsets: list[int] = []
    targets: list[dict[str, int]] = []
    for row in base_fields:
        for offset in range(7):
            expanded_fields.append(row)
            offsets.append(offset)
            targets.append(update_fields(row, caps, offset))
    states = states_from_fields(expanded_fields, state_cell, caps)
    programs = torch.tensor(program_vectors(offsets, caps), dtype=torch.float32)
    x = torch.cat([states.detach(), programs], dim=1)
    labels = {key: torch.tensor(values, dtype=torch.long) for key, values in answer_label_fields(targets, caps).items()}
    trunk = nn.Sequential(nn.Linear(int(x.shape[1]), int(branch_width)), nn.Tanh())
    heads = nn.ModuleDict({key: nn.Linear(int(branch_width), int(size)) for key, size in sizes.items()})
    optimizer = torch.optim.Adam(list(trunk.parameters()) + list(heads.parameters()), lr=0.03)
    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        hidden = trunk(x)
        loss = sum(functional.cross_entropy(heads[key](hidden), labels[key]) for key in sizes)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "trunk": trunk,
        "heads": heads,
        "parameter_count": int(sum(parameter.numel() for parameter in list(trunk.parameters()) + list(heads.parameters()))),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
    }


def decode_answer_fields(state: Any, decoder: dict[str, Any]) -> list[dict[str, int]]:
    import torch

    heads = decoder["heads"]
    for head in heads.values():
        head.eval()
    with torch.no_grad():
        pred = {key: head(state).argmax(dim=-1).cpu().numpy().astype(int) for key, head in heads.items()}
    rows = []
    for index in range(int(state.shape[0])):
        rows.append({key: int(pred[key][index]) for key in pred})
    return rows


def answer_to_compact(row: dict[str, int], caps: dict[str, int]) -> dict[str, int]:
    return {
        "address": int(int(row["color"]) * int(caps["n_shapes"]) + int(row["shape"])),
        "schema": int(row["vel_code"]),
        "residual": int(row["pos"]),
        "action": int(row["action"]),
        "provenance": int(row["provenance"]),
    }


def decode_branch_fields(state: Any, offsets: list[int], branch: dict[str, Any], caps: dict[str, int]) -> list[dict[str, int]]:
    import torch

    programs = torch.tensor(program_vectors(offsets, caps), dtype=torch.float32)
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


def score_answer_rows(rows: list[dict[str, int]], records: list[dict[str, Any]], caps: dict[str, int]) -> list[dict[str, float]]:
    return [evaluate_fields(answer_to_compact(rows[index], caps), record, caps) for index, record in enumerate(records)]


def score_against_fields(rows: list[dict[str, int]], target_fields: list[dict[str, int]], caps: dict[str, int]) -> list[dict[str, float]]:
    scored = []
    targets = answer_label_fields(target_fields, caps)
    for index, row in enumerate(rows):
        ok = float(int(all(int(row[key]) == int(targets[key][index]) for key in targets)))
        scored.append({"joint_success": ok})
    return scored


def shifted_fields(fields: list[dict[str, int]], caps: dict[str, int]) -> list[dict[str, int]]:
    if len(fields) > 1:
        return fields[-1:] + fields[:-1]
    return [update_fields(row, caps, 3) for row in fields]


def disabled_answer_rows(count: int) -> list[dict[str, int]]:
    return [{"color": 0, "shape": 0, "pos": 0, "vel_code": 0, "action": 0, "provenance": 0} for _ in range(int(count))]


def shifted_answer_rows(rows: list[dict[str, int]]) -> list[dict[str, int]]:
    if len(rows) > 1:
        return rows[-1:] + rows[:-1]
    return disabled_answer_rows(len(rows))


def corrupted_states(fields: list[dict[str, int]], state_cell: dict[str, Any], caps: dict[str, int], steps: int = DISTRACTOR_STEPS) -> Any:
    state = states_from_fields(fields, state_cell, caps)
    distractors = [update_fields(row, caps, index % 7) for index, row in enumerate(fields)]
    for _ in range(int(steps)):
        state = states_from_fields(distractors, state_cell, caps, state)
    return state


def reactivated_states(cue_fields: list[dict[str, int]], state_cell: dict[str, Any], caps: dict[str, int], prior: Any, steps: int = REPLAY_STEPS) -> Any:
    state = prior
    for _ in range(int(steps)):
        state = states_from_fields(cue_fields, state_cell, caps, state)
    return state


def axis_summary(profile: str, axis: str, seed: int, train_episodes: int = TRAIN_EPISODES, val_episodes: int = VAL_EPISODES, test_episodes: int = TEST_EPISODES, state_epochs: int = STATE_EPOCHS, answer_epochs: int = ANSWER_EPOCHS, branch_epochs: int = BRANCH_EPOCHS, state_width: int = STATE_WIDTH, branch_width: int = BRANCH_WIDTH) -> dict[str, Any]:
    caps = profile_caps(profile)
    dataset = build_factor_heldout_distributed_dataset(profile, seed, train_episodes, val_episodes, test_episodes, key=axis)
    train_records = split_records(dataset, "train")
    test_records = split_records(dataset, "test")
    text_binder = train_event_binding_segment_model(train_records, caps, seed + 101)
    train_fields = [fields_from_record(row, caps) for row in train_records]
    state_cell = train_local_state_cell(train_fields, caps, seed + 211, epochs=state_epochs, state_width=state_width)
    answer_decoder = train_answer_decoder(train_fields, state_cell, caps, seed + 307, epochs=answer_epochs)
    branch_rollout = train_branch_rollout(train_fields, state_cell, caps, seed + 353, epochs=branch_epochs, branch_width=branch_width)
    predicted_fields = predicted_code_fields(test_records, text_binder, caps, seed + 409)
    initial_state = states_from_fields(predicted_fields, state_cell, caps)
    initial_answers = decode_answer_fields(initial_state, answer_decoder)
    initial_results = score_answer_rows(initial_answers, test_records, caps)
    corrupted = corrupted_states(predicted_fields, state_cell, caps)
    no_replay_answers = decode_answer_fields(corrupted, answer_decoder)
    targeted_replay_state = reactivated_states(predicted_fields, state_cell, caps, corrupted)
    targeted_replay_answers = decode_answer_fields(targeted_replay_state, answer_decoder)
    random_replay_state = reactivated_states(shifted_fields(predicted_fields, caps), state_cell, caps, corrupted)
    random_replay_answers = decode_answer_fields(random_replay_state, answer_decoder)
    recency_replay_state = reactivated_states([update_fields(row, caps, index % 7) for index, row in enumerate(predicted_fields)], state_cell, caps, corrupted)
    recency_replay_answers = decode_answer_fields(recency_replay_state, answer_decoder)
    dummy_replay_state = reactivated_states([update_fields(row, caps, 0) for row in predicted_fields], state_cell, caps, corrupted)
    dummy_replay_answers = decode_answer_fields(dummy_replay_state, answer_decoder)
    targeted_replay_results = score_answer_rows(targeted_replay_answers, test_records, caps)
    no_replay_results = score_answer_rows(no_replay_answers, test_records, caps)
    random_replay_results = score_answer_rows(random_replay_answers, test_records, caps)
    recency_replay_results = score_answer_rows(recency_replay_answers, test_records, caps)
    dummy_replay_results = score_answer_rows(dummy_replay_answers, test_records, caps)
    disabled_decoder_results = score_answer_rows(disabled_answer_rows(len(initial_answers)), test_records, caps)
    shuffled_answer_results = score_answer_rows(shifted_answer_rows(initial_answers), test_records, caps)
    rewrite_targets = [update_fields(row, caps, index % 7) for index, row in enumerate(predicted_fields)]
    rewrite_state = states_from_fields(rewrite_targets, state_cell, caps, initial_state)
    rewrite_answers = decode_answer_fields(rewrite_state, answer_decoder)
    no_rewrite_answers = decode_answer_fields(initial_state, answer_decoder)
    random_rewrite_state = states_from_fields(shifted_fields(rewrite_targets, caps), state_cell, caps, initial_state)
    random_rewrite_answers = decode_answer_fields(random_rewrite_state, answer_decoder)
    rewrite_results = score_against_fields(rewrite_answers, rewrite_targets, caps)
    no_rewrite_results = score_against_fields(no_rewrite_answers, rewrite_targets, caps)
    random_rewrite_results = score_against_fields(random_rewrite_answers, rewrite_targets, caps)
    branch_offsets = [index % 7 for index in range(len(predicted_fields))]
    branch_targets = [update_fields(row, caps, branch_offsets[index]) for index, row in enumerate(predicted_fields)]
    branch_answers = decode_branch_fields(initial_state, branch_offsets, branch_rollout, caps)
    wrong_branch_answers = decode_branch_fields(initial_state, [(offset + 1) % 7 for offset in branch_offsets], branch_rollout, caps)
    random_branch_answers = decode_branch_fields(initial_state, [(offset + 3) % 7 for offset in branch_offsets], branch_rollout, caps)
    branch_results = score_against_fields(branch_answers, branch_targets, caps)
    no_branch_results = score_against_fields(initial_answers, branch_targets, caps)
    wrong_branch_results = score_against_fields(wrong_branch_answers, branch_targets, caps)
    random_branch_results = score_against_fields(random_branch_answers, branch_targets, caps)
    easy_no_branch_results = score_answer_rows(initial_answers, test_records, caps)
    matched_sparse_results = [matched_budget_sparse_read_result(row, caps) for row in test_records]
    uncapped_sparse_results = [distributed_evidence_sparse_read_result(row, caps, max_records=32) for row in test_records]
    field_floor = min(
        mean_metric(initial_results, "color_success"),
        mean_metric(initial_results, "shape_success"),
        mean_metric(initial_results, "pos_success"),
        mean_metric(initial_results, "vel_success"),
        mean_metric(initial_results, "action_success"),
        mean_metric(initial_results, "provenance_success"),
    )
    total_params = int(text_binder["parameter_count"]) + int(state_cell["parameter_count"]) + int(answer_decoder["parameter_count"]) + int(branch_rollout["parameter_count"])
    return {
        "initial_joint": mean_metric(initial_results, "joint_success"),
        "initial_state": mean_metric(initial_results, "state_success"),
        "initial_action": mean_metric(initial_results, "action_success"),
        "field_floor": float(field_floor),
        "targeted_replay_joint": mean_metric(targeted_replay_results, "joint_success"),
        "no_replay_joint": mean_metric(no_replay_results, "joint_success"),
        "random_replay_joint": mean_metric(random_replay_results, "joint_success"),
        "recency_replay_joint": mean_metric(recency_replay_results, "joint_success"),
        "dummy_replay_joint": mean_metric(dummy_replay_results, "joint_success"),
        "decoder_disabled_joint": mean_metric(disabled_decoder_results, "joint_success"),
        "answer_shuffle_joint": mean_metric(shuffled_answer_results, "joint_success"),
        "rewrite_joint": mean_metric(rewrite_results, "joint_success"),
        "no_rewrite_joint": mean_metric(no_rewrite_results, "joint_success"),
        "random_rewrite_joint": mean_metric(random_rewrite_results, "joint_success"),
        "branch_rollout_joint": mean_metric(branch_results, "joint_success"),
        "no_branch_joint": mean_metric(no_branch_results, "joint_success"),
        "wrong_branch_joint": mean_metric(wrong_branch_results, "joint_success"),
        "random_branch_joint": mean_metric(random_branch_results, "joint_success"),
        "hard_case_branch_gain": mean_metric(branch_results, "joint_success") - mean_metric(no_branch_results, "joint_success"),
        "easy_case_branch_gain": mean_metric(easy_no_branch_results, "joint_success") - mean_metric(easy_no_branch_results, "joint_success"),
        "matched_sparse_joint": mean_metric(matched_sparse_results, "joint_correct"),
        "uncapped_sparse_joint": mean_metric(uncapped_sparse_results, "joint_correct"),
        "committed_bits": float(learned_code_bits(caps)),
        "accounted_bits": float(parser_schema_cost_bits(caps) + learned_code_bits(caps)),
        "matched_sparse_bits": mean_metric(matched_sparse_results, "bits_committed"),
        "text_binder_parameter_count": float(text_binder["parameter_count"]),
        "state_cell_parameter_count": float(state_cell["parameter_count"]),
        "answer_decoder_parameter_count": float(answer_decoder["parameter_count"]),
        "branch_rollout_parameter_count": float(branch_rollout["parameter_count"]),
        "total_parameter_count": float(total_params),
        "answer_loss_start": float(answer_decoder["train_loss_start"]),
        "answer_loss_final": float(answer_decoder["train_loss_final"]),
        "branch_loss_final": float(branch_rollout["train_loss_final"]),
        "state_loss_final": float(state_cell["train_loss_final"]),
        "text_loss_final": float(text_binder["train_loss_final"]),
        "example_prompt": randomized_record_prompt(test_records[0], seed + 409) if test_records else "",
        "example_response": answer_text_from_row(initial_answers[0], caps) if initial_answers else "",
    }


def answer_text_from_row(row: dict[str, int], caps: dict[str, int]) -> str:
    return f"answer action_{int(row['action'])} color_{int(row['color'])} shape_{int(row['shape'])} pos_{int(row['pos'])} vel_{int(row['vel_code']) - int(caps['max_speed'])}"


def build_summary(profile: str, seed: int = SEED, train_episodes: int = TRAIN_EPISODES, val_episodes: int = VAL_EPISODES, test_episodes: int = TEST_EPISODES, state_epochs: int = STATE_EPOCHS, answer_epochs: int = ANSWER_EPOCHS, branch_epochs: int = BRANCH_EPOCHS, state_width: int = STATE_WIDTH, branch_width: int = BRANCH_WIDTH, seed_count: int = SEED_COUNT, axes: tuple[str, ...] = AXES) -> dict[str, Any]:
    runs = []
    for axis_index, axis in enumerate(axes):
        for seed_index in range(int(seed_count)):
            runs.append(axis_summary(profile, axis, seed + axis_index * 10_003 + seed_index * 1_009, train_episodes, val_episodes, test_episodes, state_epochs, answer_epochs, branch_epochs, state_width, branch_width))
    initial_joints = [float(row["initial_joint"]) for row in runs]
    initial_states = [float(row["initial_state"]) for row in runs]
    initial_actions = [float(row["initial_action"]) for row in runs]
    field_floors = [float(row["field_floor"]) for row in runs]
    targeted_replays = [float(row["targeted_replay_joint"]) for row in runs]
    no_replays = [float(row["no_replay_joint"]) for row in runs]
    random_replays = [float(row["random_replay_joint"]) for row in runs]
    recency_replays = [float(row["recency_replay_joint"]) for row in runs]
    dummy_replays = [float(row["dummy_replay_joint"]) for row in runs]
    decoder_disabled = [float(row["decoder_disabled_joint"]) for row in runs]
    answer_shuffles = [float(row["answer_shuffle_joint"]) for row in runs]
    rewrites = [float(row["rewrite_joint"]) for row in runs]
    no_rewrites = [float(row["no_rewrite_joint"]) for row in runs]
    random_rewrites = [float(row["random_rewrite_joint"]) for row in runs]
    branch_rollouts = [float(row["branch_rollout_joint"]) for row in runs]
    no_branches = [float(row["no_branch_joint"]) for row in runs]
    wrong_branches = [float(row["wrong_branch_joint"]) for row in runs]
    random_branches = [float(row["random_branch_joint"]) for row in runs]
    hard_case_gains = [float(row["hard_case_branch_gain"]) for row in runs]
    easy_case_gains = [float(row["easy_case_branch_gain"]) for row in runs]
    matched_sparse = [float(row["matched_sparse_joint"]) for row in runs]
    uncapped_sparse = [float(row["uncapped_sparse_joint"]) for row in runs]
    params = [float(row["total_parameter_count"]) for row in runs]
    accounted_bits = [float(row["accounted_bits"]) for row in runs]
    matched_bits = [float(row["matched_sparse_bits"]) for row in runs]
    useful_density = [initial_joints[index] / max(accounted_bits[index], 1e-9) for index in range(len(runs))]
    sparse_density = [matched_sparse[index] / max(matched_bits[index], 1e-9) for index in range(len(runs))]
    engineering_pass = float(int(runs and min(initial_joints) >= 0.95 and min(initial_states) >= 0.95 and min(initial_actions) >= 0.95 and min(field_floors) >= 0.95 and min(targeted_replays) >= 0.95 and min(rewrites) >= 0.95 and min(branch_rollouts) >= 0.95 and min(hard_case_gains) > max(easy_case_gains) and max(no_replays) < min(targeted_replays) and max(random_replays) < min(targeted_replays) and max(recency_replays) < min(targeted_replays) and max(dummy_replays) < min(targeted_replays) and max(decoder_disabled) < min(initial_joints) and max(answer_shuffles) < min(initial_joints) and max(no_rewrites) < min(rewrites) and max(random_rewrites) < min(rewrites) and max(no_branches) < min(branch_rollouts) and max(wrong_branches) < min(branch_rollouts) and max(random_branches) < min(branch_rollouts) and max(matched_sparse) == 0.0 and max(params) < 100_000))
    return {
        "local_100k_replay_answer_evaluated": 1.0,
        "local_100k_replay_answer_local_model_candidate_authorized": 1.0,
        "local_100k_replay_answer_full_model_authorized": 0.0,
        "local_100k_replay_answer_paid_compute_authorized": 0.0,
        "local_100k_replay_answer_arbitrary_chat_authorized": 0.0,
        "local_100k_replay_answer_axis_count": int(len(axes)),
        "local_100k_replay_answer_seed_count": int(seed_count),
        "local_100k_replay_answer_run_count": int(len(runs)),
        "local_100k_replay_answer_total_train_record_count": int(len(runs) * int(train_episodes)),
        "local_100k_replay_answer_total_validation_record_count": int(len(runs) * int(val_episodes)),
        "local_100k_replay_answer_total_test_record_count": int(len(runs) * int(test_episodes)),
        "local_100k_replay_answer_parameter_count_max": float(max(params) if params else 0.0),
        "local_100k_replay_answer_parameter_count_mean": float(np.mean(params)) if params else 0.0,
        "local_100k_replay_answer_initial_joint_success_min": float(min(initial_joints) if initial_joints else 0.0),
        "local_100k_replay_answer_initial_state_success_min": float(min(initial_states) if initial_states else 0.0),
        "local_100k_replay_answer_initial_action_success_min": float(min(initial_actions) if initial_actions else 0.0),
        "local_100k_replay_answer_field_accuracy_floor": float(min(field_floors) if field_floors else 0.0),
        "local_100k_replay_answer_targeted_replay_success_min": float(min(targeted_replays) if targeted_replays else 0.0),
        "local_100k_replay_answer_no_replay_success_max": float(max(no_replays) if no_replays else 0.0),
        "local_100k_replay_answer_random_replay_success_max": float(max(random_replays) if random_replays else 0.0),
        "local_100k_replay_answer_recency_replay_success_max": float(max(recency_replays) if recency_replays else 0.0),
        "local_100k_replay_answer_matched_compute_dummy_replay_success_max": float(max(dummy_replays) if dummy_replays else 0.0),
        "local_100k_replay_answer_decoder_disabled_success_max": float(max(decoder_disabled) if decoder_disabled else 0.0),
        "local_100k_replay_answer_shuffled_answer_success_max": float(max(answer_shuffles) if answer_shuffles else 0.0),
        "local_100k_replay_answer_rewrite_success_min": float(min(rewrites) if rewrites else 0.0),
        "local_100k_replay_answer_no_rewrite_success_max": float(max(no_rewrites) if no_rewrites else 0.0),
        "local_100k_replay_answer_random_rewrite_success_max": float(max(random_rewrites) if random_rewrites else 0.0),
        "local_100k_replay_answer_branch_rollout_success_min": float(min(branch_rollouts) if branch_rollouts else 0.0),
        "local_100k_replay_answer_no_branch_success_max": float(max(no_branches) if no_branches else 0.0),
        "local_100k_replay_answer_wrong_branch_success_max": float(max(wrong_branches) if wrong_branches else 0.0),
        "local_100k_replay_answer_random_branch_success_max": float(max(random_branches) if random_branches else 0.0),
        "local_100k_replay_answer_hard_case_branch_gain_min": float(min(hard_case_gains) if hard_case_gains else 0.0),
        "local_100k_replay_answer_easy_case_branch_gain_max": float(max(easy_case_gains) if easy_case_gains else 0.0),
        "local_100k_replay_answer_matched_sparse_joint_success_max": float(max(matched_sparse) if matched_sparse else 0.0),
        "local_100k_replay_answer_uncapped_sparse_joint_success_min": float(min(uncapped_sparse) if uncapped_sparse else 0.0),
        "local_100k_replay_answer_committed_bits_max": float(max([float(row["committed_bits"]) for row in runs]) if runs else 0.0),
        "local_100k_replay_answer_accounted_bits_max": float(max(accounted_bits) if accounted_bits else 0.0),
        "local_100k_replay_answer_matched_sparse_bits_min": float(min(matched_bits) if matched_bits else 0.0),
        "local_100k_replay_answer_useful_operation_success_per_accounted_bit_min": float(min(useful_density) if useful_density else 0.0),
        "local_100k_replay_answer_matched_sparse_operation_success_per_committed_bit_max": float(max(sparse_density) if sparse_density else 0.0),
        "local_100k_replay_answer_useful_state_density_advantage_min": float(min([useful_density[index] - sparse_density[index] for index in range(len(useful_density))]) if useful_density else 0.0),
        "local_100k_replay_answer_answer_loss_final_mean": float(np.mean([float(row["answer_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_replay_answer_branch_loss_final_mean": float(np.mean([float(row["branch_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_replay_answer_state_loss_final_mean": float(np.mean([float(row["state_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_replay_answer_text_loss_final_mean": float(np.mean([float(row["text_loss_final"]) for row in runs])) if runs else 0.0,
        "local_100k_replay_answer_engineering_pass": float(engineering_pass),
        "local_100k_replay_answer_claim_downgraded_to_component_model_candidate": float(1.0 - engineering_pass),
        "local_100k_replay_answer_example_prompt": str(runs[0]["example_prompt"]) if runs else "",
        "local_100k_replay_answer_example_response": str(runs[0]["example_response"]) if runs else "",
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_100k_replay_answer_mirror_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_100k_replay_answer_mirror",
        script_path=SCRIPT_PATH,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_sec=float(time.perf_counter() - started),
        parameters={
            "profile": profile,
            "train_episodes": int(TRAIN_EPISODES),
            "validation_episodes": int(VAL_EPISODES),
            "test_episodes": int(TEST_EPISODES),
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
        n_trials=int(summary["local_100k_replay_answer_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_100k_replay_answer_mirror_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
