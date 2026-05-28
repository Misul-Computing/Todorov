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
    code_fields_from_bound_events,
    event_binding_head_blocked_tokens,
    evaluate_fields,
    mean_metric,
    parser_schema_cost_bits,
    randomized_record_parts,
    randomized_record_prompt,
    segment_features,
    split_records,
    train_event_binding_segment_model,
)

SCRIPT_PATH = Path(__file__).resolve()
SEED = env_int("LSWR_SEED", 97)
TRAIN_EPISODES = env_int("LSWR_TRAIN_EPISODES", 512)
VAL_EPISODES = env_int("LSWR_VAL_EPISODES", 64)
TEST_EPISODES = env_int("LSWR_TEST_EPISODES", 64)
STATE_EPOCHS = env_int("LSWR_STATE_EPOCHS", 120)
STATE_WIDTH = env_int("LSWR_STATE_WIDTH", 64)
SEED_COUNT = env_int("LSWR_SEED_COUNT", 1)

require_positive("LSWR_TRAIN_EPISODES", TRAIN_EPISODES)
require_positive("LSWR_VAL_EPISODES", VAL_EPISODES)
require_positive("LSWR_TEST_EPISODES", TEST_EPISODES)
require_positive("LSWR_STATE_EPOCHS", STATE_EPOCHS)
require_positive("LSWR_STATE_WIDTH", STATE_WIDTH)
require_positive("LSWR_SEED_COUNT", SEED_COUNT)


def infer_profile() -> str:
    raw = str(sys.argv[1]).strip() if len(sys.argv) > 1 else ""
    if raw in {"smoke", "hard"}:
        return raw
    value = os.environ.get("LSWR_PROFILE", "smoke").strip()
    if value not in {"smoke", "hard"}:
        raise ValueError("LSWR_PROFILE must be smoke or hard")
    return value


def field_sizes(caps: dict[str, int]) -> dict[str, int]:
    return {
        "address": int(caps["n_colors"]) * int(caps["n_shapes"]),
        "schema": int(caps["max_speed"]) * 2 + 1,
        "residual": int(caps["track_length"]),
        "action": int(caps["action_count"]),
        "provenance": int(caps["seq_len"]),
    }


def field_vector(fields: dict[str, int], caps: dict[str, int]) -> np.ndarray:
    sizes = field_sizes(caps)
    values: list[float] = []
    for key in ("address", "schema", "residual", "action", "provenance"):
        size = int(sizes[key])
        index = max(0, min(size - 1, int(fields[key])))
        values.extend(1.0 if item == index else 0.0 for item in range(size))
    return np.asarray(values, dtype=np.float32)


def fields_from_record(record: dict[str, Any], caps: dict[str, int]) -> dict[str, int]:
    events = list(record["model_input"]["observations"])
    focus = int(record["model_input"]["query"]["focus_local_index"])
    return code_fields_from_bound_events(events, focus, caps)


def update_fields(fields: dict[str, int], caps: dict[str, int], offset: int) -> dict[str, int]:
    sizes = field_sizes(caps)
    return {
        "address": int((int(fields["address"]) + offset + 1) % int(sizes["address"])),
        "schema": int((int(fields["schema"]) + 2 * offset + 1) % int(sizes["schema"])),
        "residual": int((int(fields["residual"]) + 3 * offset + 1) % int(sizes["residual"])),
        "action": int((int(fields["action"]) + offset + 2) % int(sizes["action"])),
        "provenance": int((int(fields["provenance"]) + offset + 1) % int(sizes["provenance"])),
    }


def field_targets(fields: list[dict[str, int]], caps: dict[str, int]) -> dict[str, list[int]]:
    return {key: [int(row[key]) for row in fields] for key in field_sizes(caps)}


def sampled_code_fields(caps: dict[str, int], seed: int, count: int) -> list[dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    sizes = field_sizes(caps)
    return [{key: int(rng.integers(0, int(size))) for key, size in sizes.items()} for _ in range(int(count))]


def train_local_state_cell(fields: list[dict[str, int]], caps: dict[str, int], seed: int, epochs: int = STATE_EPOCHS, state_width: int = STATE_WIDTH) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    torch.manual_seed(int(seed))
    sizes = field_sizes(caps)
    fields = list(fields) + sampled_code_fields(caps, seed + 17, max(2048, len(fields) * 4))
    x = torch.tensor(np.stack([field_vector(row, caps) for row in fields]), dtype=torch.float32)
    y = {key: torch.tensor(values, dtype=torch.long) for key, values in field_targets(fields, caps).items()}
    update_rows = [update_fields(row, caps, index % 7) for index, row in enumerate(fields)]
    x_update = torch.tensor(np.stack([field_vector(row, caps) for row in update_rows]), dtype=torch.float32)
    y_update = {key: torch.tensor(values, dtype=torch.long) for key, values in field_targets(update_rows, caps).items()}
    width = int(state_width)
    cell = nn.GRUCell(int(x.shape[1]), width)
    heads = nn.ModuleDict({key: nn.Linear(width, int(size)) for key, size in sizes.items()})
    optimizer = torch.optim.Adam(list(cell.parameters()) + list(heads.parameters()), lr=0.03)
    losses = []
    zero = torch.zeros((int(x.shape[0]), width), dtype=torch.float32)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        state = cell(x, zero)
        loss = sum(functional.cross_entropy(heads[key](state), y[key]) for key in sizes)
        updated = cell(x_update, state)
        loss = loss + sum(functional.cross_entropy(heads[key](updated), y_update[key]) for key in sizes)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "cell": cell,
        "heads": heads,
        "parameter_count": int(sum(parameter.numel() for parameter in list(cell.parameters()) + list(heads.parameters()))),
        "train_loss_start": float(losses[0]) if losses else 0.0,
        "train_loss_final": float(losses[-1]) if losses else 0.0,
    }


def decode_state(state: Any, learned: dict[str, Any], caps: dict[str, int]) -> list[dict[str, int]]:
    heads = learned["heads"]
    sizes = field_sizes(caps)
    rows: list[dict[str, int]] = []
    for key in sizes:
        heads[key].eval()
    import torch

    with torch.no_grad():
        pred = {key: heads[key](state).argmax(dim=-1).cpu().numpy().astype(int) for key in sizes}
    for index in range(int(state.shape[0])):
        rows.append({key: int(pred[key][index]) for key in sizes})
    return rows


def write_read_fields(fields: list[dict[str, int]], learned: dict[str, Any], caps: dict[str, int], mode: str = "normal") -> list[dict[str, int]]:
    import torch

    x = torch.tensor(np.stack([field_vector(row, caps) for row in fields]), dtype=torch.float32)
    width = int(next(learned["cell"].parameters()).shape[0] // 3)
    zero = torch.zeros((int(x.shape[0]), width), dtype=torch.float32)
    learned["cell"].eval()
    with torch.no_grad():
        state = learned["cell"](x, zero)
        if mode == "zero":
            state = zero
        elif mode == "shuffle" and int(state.shape[0]) > 1:
            state = torch.cat([state[-1:], state[:-1]], dim=0)
    return decode_state(state, learned, caps)


def update_read_results(fields: list[dict[str, int]], learned: dict[str, Any], caps: dict[str, int], mode: str) -> list[dict[str, int]]:
    import torch

    update_rows = [update_fields(row, caps, index % 7) for index, row in enumerate(fields)]
    x = torch.tensor(np.stack([field_vector(row, caps) for row in fields]), dtype=torch.float32)
    x_update = torch.tensor(np.stack([field_vector(row, caps) for row in update_rows]), dtype=torch.float32)
    width = int(next(learned["cell"].parameters()).shape[0] // 3)
    zero = torch.zeros((int(x.shape[0]), width), dtype=torch.float32)
    learned["cell"].eval()
    with torch.no_grad():
        state = learned["cell"](x, zero)
        if mode == "normal":
            state = learned["cell"](x_update, state)
        elif mode == "no_update":
            state = state
        elif mode == "random_update":
            shifted = torch.cat([x_update[-1:], x_update[:-1]], dim=0) if int(x_update.shape[0]) > 1 else torch.flip(x_update, dims=[1])
            state = learned["cell"](shifted, state)
        elif mode == "zero":
            state = zero
        else:
            raise ValueError(f"unknown update mode {mode}")
    return decode_state(state, learned, caps)


def predicted_code_fields(records: list[dict[str, Any]], learned: dict[str, Any], caps: dict[str, int], seed: int) -> list[dict[str, int]]:
    import torch

    prompts = [randomized_record_prompt(record, seed + index * 104_729) for index, record in enumerate(records)]
    segment_groups: list[list[str]] = []
    query_texts: list[str] = []
    flat_segments: list[str] = []
    for prompt in prompts:
        parts = [segment.strip() for segment in prompt.split("|") if segment.strip()]
        segment_groups.append(parts[:-1])
        flat_segments.extend(parts[:-1])
        query_texts.append(parts[-1] if parts else "")
    inputs = {key: torch.tensor(segment_features(flat_segments, learned["vocab"], event_binding_head_blocked_tokens(key)), dtype=torch.float32) for key in ("event", "slot", "time", "color", "shape", "pos")}
    query_inputs = torch.tensor(segment_features(query_texts, learned["vocab"]), dtype=torch.float32)
    heads = learned["heads"]
    heads.eval()
    with torch.no_grad():
        pred = {key: heads[key](inputs[key]).argmax(dim=-1).cpu().numpy().astype(int) for key in ("event", "slot", "time", "color", "shape", "pos")} if flat_segments else {key: np.asarray([], dtype=int) for key in ("event", "slot", "time", "color", "shape", "pos")}
        query_pred = heads["query"](query_inputs).argmax(dim=-1).cpu().numpy().astype(int) if query_texts else np.asarray([], dtype=int)
    rows: list[dict[str, int]] = []
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
        rows.append(code_fields_from_bound_events(events, int(query_pred[record_index]), caps))
    return rows


def score_predicted_fields(fields: list[dict[str, int]], records: list[dict[str, Any]], caps: dict[str, int]) -> list[dict[str, Any]]:
    return [evaluate_fields(fields[index], record, caps) for index, record in enumerate(records)]


def update_score_rows(fields: list[dict[str, int]], predicted: list[dict[str, int]], caps: dict[str, int]) -> list[dict[str, float]]:
    targets = [update_fields(row, caps, index % 7) for index, row in enumerate(fields)]
    rows = []
    for index, row in enumerate(predicted):
        target = targets[index]
        correct = float(int(all(int(row[key]) == int(target[key]) for key in field_sizes(caps))))
        rows.append({"joint_correct": correct})
    return rows


def axis_summary(profile: str, axis: str, seed: int, train_episodes: int = TRAIN_EPISODES, val_episodes: int = VAL_EPISODES, test_episodes: int = TEST_EPISODES, epochs: int = STATE_EPOCHS, state_width: int = STATE_WIDTH) -> dict[str, Any]:
    caps = profile_caps(profile)
    dataset = build_factor_heldout_distributed_dataset(profile, seed, train_episodes, val_episodes, test_episodes, key=axis)
    train_records = split_records(dataset, "train")
    test_records = split_records(dataset, "test")
    text_binder = train_event_binding_segment_model(train_records, caps, seed + 101)
    state_fields = [fields_from_record(row, caps) for row in train_records]
    state_cell = train_local_state_cell(state_fields, caps, seed + 211, epochs=epochs, state_width=state_width)
    predicted_fields = predicted_code_fields(test_records, text_binder, caps, seed + 307)
    decoded_fields = write_read_fields(predicted_fields, state_cell, caps, "normal")
    zero_fields = write_read_fields(predicted_fields, state_cell, caps, "zero")
    shuffle_fields = write_read_fields(predicted_fields, state_cell, caps, "shuffle")
    update_fields_normal = update_read_results(predicted_fields, state_cell, caps, "normal")
    update_fields_no_write = update_read_results(predicted_fields, state_cell, caps, "no_update")
    update_fields_random = update_read_results(predicted_fields, state_cell, caps, "random_update")
    results = score_predicted_fields(decoded_fields, test_records, caps)
    zero_results = score_predicted_fields(zero_fields, test_records, caps)
    shuffle_results = score_predicted_fields(shuffle_fields, test_records, caps)
    update_results = update_score_rows(predicted_fields, update_fields_normal, caps)
    no_update_results = update_score_rows(predicted_fields, update_fields_no_write, caps)
    random_update_results = update_score_rows(predicted_fields, update_fields_random, caps)
    matched_sparse_results = [matched_budget_sparse_read_result(row, caps) for row in test_records]
    uncapped_sparse_results = [distributed_evidence_sparse_read_result(row, caps, max_records=32) for row in test_records]
    field_floor = min(
        mean_metric(results, "color_success"),
        mean_metric(results, "shape_success"),
        mean_metric(results, "pos_success"),
        mean_metric(results, "vel_success"),
        mean_metric(results, "action_success"),
        mean_metric(results, "provenance_success"),
    )
    committed_bits = float(learned_code_bits(caps))
    accounted_bits = float(parser_schema_cost_bits(caps) + learned_code_bits(caps))
    return {
        "joint": mean_metric(results, "joint_success"),
        "state": mean_metric(results, "state_success"),
        "action": mean_metric(results, "action_success"),
        "field_floor": float(field_floor),
        "zero_joint": mean_metric(zero_results, "joint_success"),
        "shuffle_joint": mean_metric(shuffle_results, "joint_success"),
        "update_joint": mean_metric(update_results, "joint_correct"),
        "no_update_joint": mean_metric(no_update_results, "joint_correct"),
        "random_update_joint": mean_metric(random_update_results, "joint_correct"),
        "matched_sparse_joint": mean_metric(matched_sparse_results, "joint_correct"),
        "uncapped_sparse_joint": mean_metric(uncapped_sparse_results, "joint_correct"),
        "committed_bits": float(committed_bits),
        "accounted_bits": float(accounted_bits),
        "matched_sparse_bits": mean_metric(matched_sparse_results, "bits_committed"),
        "state_cell_parameter_count": float(state_cell["parameter_count"]),
        "text_binder_parameter_count": float(text_binder["parameter_count"]),
        "total_parameter_count": float(int(state_cell["parameter_count"]) + int(text_binder["parameter_count"])),
        "state_loss_start": float(state_cell["train_loss_start"]),
        "state_loss_final": float(state_cell["train_loss_final"]),
        "text_loss_start": float(text_binder["train_loss_start"]),
        "text_loss_final": float(text_binder["train_loss_final"]),
        "prompt": randomized_record_prompt(test_records[0], seed + 307) if test_records else "",
    }


def build_summary(profile: str, seed: int = SEED, train_episodes: int = TRAIN_EPISODES, val_episodes: int = VAL_EPISODES, test_episodes: int = TEST_EPISODES, epochs: int = STATE_EPOCHS, state_width: int = STATE_WIDTH, seed_count: int = SEED_COUNT, axes: tuple[str, ...] = AXES) -> dict[str, Any]:
    runs = []
    for axis_index, axis in enumerate(axes):
        for seed_index in range(int(seed_count)):
            runs.append(axis_summary(profile, axis, seed + axis_index * 10_003 + seed_index * 1_009, train_episodes, val_episodes, test_episodes, epochs, state_width))
    joints = [float(row["joint"]) for row in runs]
    states = [float(row["state"]) for row in runs]
    actions = [float(row["action"]) for row in runs]
    field_floors = [float(row["field_floor"]) for row in runs]
    zero_joints = [float(row["zero_joint"]) for row in runs]
    shuffle_joints = [float(row["shuffle_joint"]) for row in runs]
    update_joints = [float(row["update_joint"]) for row in runs]
    no_update_joints = [float(row["no_update_joint"]) for row in runs]
    random_update_joints = [float(row["random_update_joint"]) for row in runs]
    matched_sparse = [float(row["matched_sparse_joint"]) for row in runs]
    uncapped_sparse = [float(row["uncapped_sparse_joint"]) for row in runs]
    params = [float(row["total_parameter_count"]) for row in runs]
    accounted_bits = [float(row["accounted_bits"]) for row in runs]
    matched_bits = [float(row["matched_sparse_bits"]) for row in runs]
    useful_density = [joints[index] / max(accounted_bits[index], 1e-9) for index in range(len(runs))]
    sparse_density = [matched_sparse[index] / max(matched_bits[index], 1e-9) for index in range(len(runs))]
    engineering_pass = float(int(runs and min(joints) >= 0.95 and min(field_floors) >= 0.95 and min(update_joints) >= 0.95 and max(params) < 100_000 and max(matched_sparse) == 0.0 and max(zero_joints) < min(joints) and max(shuffle_joints) < min(joints) and max(no_update_joints) < min(update_joints) and max(random_update_joints) < min(update_joints)))
    return {
        "local_state_write_read_evaluated": 1.0,
        "local_state_write_read_local_mechanism_authorized": 1.0,
        "local_state_write_read_full_model_authorized": 0.0,
        "local_state_write_read_paid_compute_authorized": 0.0,
        "local_state_write_read_arbitrary_chat_authorized": 0.0,
        "local_state_write_read_axis_count": int(len(axes)),
        "local_state_write_read_seed_count": int(seed_count),
        "local_state_write_read_run_count": int(len(runs)),
        "local_state_write_read_total_train_record_count": int(len(runs) * int(train_episodes)),
        "local_state_write_read_total_validation_record_count": int(len(runs) * int(val_episodes)),
        "local_state_write_read_total_test_record_count": int(len(runs) * int(test_episodes)),
        "local_state_write_read_state_width": int(state_width),
        "local_state_write_read_parameter_count_max": float(max(params) if params else 0.0),
        "local_state_write_read_parameter_count_mean": float(np.mean(params)) if params else 0.0,
        "local_state_write_read_joint_success_min": float(min(joints) if joints else 0.0),
        "local_state_write_read_state_success_min": float(min(states) if states else 0.0),
        "local_state_write_read_action_success_min": float(min(actions) if actions else 0.0),
        "local_state_write_read_field_accuracy_floor": float(min(field_floors) if field_floors else 0.0),
        "local_state_write_read_zero_state_joint_success_max": float(max(zero_joints) if zero_joints else 0.0),
        "local_state_write_read_state_shuffle_joint_success_max": float(max(shuffle_joints) if shuffle_joints else 0.0),
        "local_state_write_read_update_joint_success_min": float(min(update_joints) if update_joints else 0.0),
        "local_state_write_read_no_update_joint_success_max": float(max(no_update_joints) if no_update_joints else 0.0),
        "local_state_write_read_random_update_joint_success_max": float(max(random_update_joints) if random_update_joints else 0.0),
        "local_state_write_read_matched_sparse_joint_success_max": float(max(matched_sparse) if matched_sparse else 0.0),
        "local_state_write_read_uncapped_sparse_joint_success_min": float(min(uncapped_sparse) if uncapped_sparse else 0.0),
        "local_state_write_read_committed_bits_max": float(max([float(row["committed_bits"]) for row in runs]) if runs else 0.0),
        "local_state_write_read_accounted_bits_max": float(max(accounted_bits) if accounted_bits else 0.0),
        "local_state_write_read_matched_sparse_bits_min": float(min(matched_bits) if matched_bits else 0.0),
        "local_state_write_read_useful_operation_success_per_accounted_bit_min": float(min(useful_density) if useful_density else 0.0),
        "local_state_write_read_matched_sparse_operation_success_per_committed_bit_max": float(max(sparse_density) if sparse_density else 0.0),
        "local_state_write_read_useful_state_density_advantage_min": float(min([useful_density[index] - sparse_density[index] for index in range(len(useful_density))]) if useful_density else 0.0),
        "local_state_write_read_state_loss_start_mean": float(np.mean([float(row["state_loss_start"]) for row in runs])) if runs else 0.0,
        "local_state_write_read_state_loss_final_mean": float(np.mean([float(row["state_loss_final"]) for row in runs])) if runs else 0.0,
        "local_state_write_read_text_loss_start_mean": float(np.mean([float(row["text_loss_start"]) for row in runs])) if runs else 0.0,
        "local_state_write_read_text_loss_final_mean": float(np.mean([float(row["text_loss_final"]) for row in runs])) if runs else 0.0,
        "local_state_write_read_engineering_pass": float(engineering_pass),
        "local_state_write_read_claim_downgraded_to_component_mirror": float(1.0 - engineering_pass),
        "local_state_write_read_example_prompt": str(runs[0]["prompt"]) if runs else "",
    }


def main() -> int:
    profile = infer_profile()
    started = time.perf_counter()
    started_at = utc_now_iso()
    summary = build_summary(profile)
    output_dir = output_dir_for(SCRIPT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "local_state_write_read_mirror_metrics.json"
    finished_at = utc_now_iso()
    record = build_run_record(
        simulation_name="local_state_write_read_mirror",
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
            "state_width": int(STATE_WIDTH),
            "seed_count": int(SEED_COUNT),
        },
        seed_numpy=int(SEED),
        n_trials=int(summary["local_state_write_read_run_count"]),
        summary=summary,
        statistics={},
        trials=[],
        warnings=[],
        artifacts=[{"name": "local_state_write_read_mirror_metrics.json", "path": metrics_path}],
    )
    write_json(metrics_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
