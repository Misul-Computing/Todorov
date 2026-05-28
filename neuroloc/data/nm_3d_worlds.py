from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
)

COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("cube", "sphere", "cone", "cylinder")
QUERY_TYPES = (
    "object_permanence",
    "occluded_localization",
    "delayed_use",
    "action_consequence",
    "counterfactual",
)


@dataclass(frozen=True)
class WorldProfile:
    name: str
    seq_len: int
    object_count: int
    coord_size: int
    action_count: int
    occlusion_rate: float
    distractor_count: int
    train_count: int
    val_count: int
    test_count: int


NM_3D_PROFILES = {
    "smoke": WorldProfile(
        name="smoke",
        seq_len=12,
        object_count=4,
        coord_size=5,
        action_count=7,
        occlusion_rate=0.5,
        distractor_count=2,
        train_count=512,
        val_count=32,
        test_count=64,
    ),
    "hard": WorldProfile(
        name="hard",
        seq_len=18,
        object_count=6,
        coord_size=5,
        action_count=9,
        occlusion_rate=0.8,
        distractor_count=4,
        train_count=1024,
        val_count=64,
        test_count=96,
    ),
}


def profile_caps(profile: str) -> dict[str, int]:
    spec = NM_3D_PROFILES[profile]
    return {
        "profile": profile,
        "seq_len": spec.seq_len,
        "object_count": spec.object_count,
        "coord_size": spec.coord_size,
        "n_colors": len(COLORS),
        "n_shapes": len(SHAPES),
        "max_speed": 3,
        "track_length": spec.coord_size**3,
        "action_count": spec.action_count,
        "query_type_count": len(QUERY_TYPES),
    }


def action_delta(action: int) -> tuple[int, int, int]:
    deltas = (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, 0, 1),
    )
    return deltas[action % len(deltas)]


def flatten_position(position: tuple[int, int, int], coord_size: int) -> int:
    x, y, z = position
    return x + coord_size * y + coord_size * coord_size * z


def unflatten_position(value: int, coord_size: int) -> tuple[int, int, int]:
    z = value // (coord_size * coord_size)
    rem = value % (coord_size * coord_size)
    y = rem // coord_size
    x = rem % coord_size
    return x, y, z


def clamp_position(position: tuple[int, int, int], coord_size: int) -> tuple[int, int, int]:
    return tuple(max(0, min(coord_size - 1, value)) for value in position)


def advance_position(
    position: tuple[int, int, int],
    delta: tuple[int, int, int],
    coord_size: int,
    steps: int,
) -> tuple[int, int, int]:
    x, y, z = position
    dx, dy, dz = delta
    return ((x + dx * steps) % coord_size, (y + dy * steps) % coord_size, (z + dz * steps) % coord_size)


def velocity_code(delta: tuple[int, int, int]) -> int:
    dx, dy, dz = delta
    return max(0, min(6, dx + dy + dz + 3))


def bits_for_cardinality(cardinality: int) -> int:
    return max(1, math.ceil(math.log2(max(2, cardinality))))


def compact_bit_budget(caps: dict[str, int]) -> dict[str, int]:
    state_bits = (
        bits_for_cardinality(caps["n_colors"] * caps["n_shapes"])
        + bits_for_cardinality(caps["max_speed"] * 2 + 1)
        + bits_for_cardinality(caps["track_length"])
        + bits_for_cardinality(caps["action_count"])
        + bits_for_cardinality(caps["seq_len"])
    )
    schema_bits = (
        bits_for_cardinality(caps["coord_size"])
        * 3
        + bits_for_cardinality(caps["object_count"])
        + bits_for_cardinality(caps["query_type_count"])
    )
    return {
        "compact_state_bits": state_bits,
        "parser_schema_world_field_bits": schema_bits,
        "answer_grammar_bits": bits_for_cardinality(caps["track_length"]) + 6,
        "budget_bits": state_bits + schema_bits,
    }


def _axis_offset(axis: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(axis))


def _split_seed(seed: int, split: str, index: int, axis: str) -> int:
    offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
    return seed * 10_000_019 + offsets[split] + index * 97 + _axis_offset(axis) * 13


def _focus_color_shape(index: int, caps: dict[str, int]) -> tuple[int, int]:
    return index % caps["n_colors"], (index // caps["n_colors"]) % caps["n_shapes"]


def _query_type(index: int) -> str:
    return QUERY_TYPES[index % len(QUERY_TYPES)]


def _object_record(
    object_id: int,
    color: int,
    shape: int,
    position: tuple[int, int, int],
    delta: tuple[int, int, int],
    visible: bool,
    time: int,
) -> dict[str, Any]:
    x, y, z = position if visible else (-1, -1, -1)
    return {
        "object_id": object_id,
        "color": color,
        "shape": shape,
        "x": x,
        "y": y,
        "z": z,
        "vx": delta[0] if visible else 0,
        "vy": delta[1] if visible else 0,
        "vz": delta[2] if visible else 0,
        "visible": visible,
        "time": time,
    }


def generate_episode(profile: str, seed: int, split: str, index: int, axis: str) -> dict[str, Any]:
    spec = NM_3D_PROFILES[profile]
    caps = profile_caps(profile)
    axis_shift = _axis_offset(axis)
    rng = random.Random(_split_seed(seed, split, index, axis))
    focus_object = (index + axis_shift) % spec.object_count
    color, shape = _focus_color_shape(index + axis_shift, caps)
    action = 1 + ((index * 3 + axis_shift + rng.randrange(spec.action_count)) % max(1, spec.action_count - 1))
    branch_action = (action + 1 + index % max(1, spec.action_count - 1)) % spec.action_count
    action = action % spec.action_count
    branch_action = branch_action % spec.action_count
    delta = action_delta(action)
    branch_delta = action_delta(branch_action)
    provenance_time = max(1, spec.seq_len // 3 + (index + axis_shift) % max(1, spec.seq_len // 5))
    query_time = spec.seq_len - 1
    if (query_time - provenance_time) % spec.coord_size == 0 and provenance_time > 1:
        provenance_time -= 1
    max_start = spec.coord_size - 1
    start = (
        rng.randrange(spec.coord_size),
        rng.randrange(spec.coord_size),
        rng.randrange(spec.coord_size),
    )
    provenance_position = advance_position(start, delta, spec.coord_size, provenance_time)
    current_position = advance_position(provenance_position, delta, spec.coord_size, query_time - provenance_time)
    branch_position = advance_position(current_position, branch_delta, spec.coord_size, 1)
    if branch_position == current_position:
        branch_position = clamp_position((max_start - current_position[0], current_position[1], current_position[2]), spec.coord_size)
    query_type = _query_type(index)
    occlusion_start = provenance_time + 1
    force_occluded = True
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for time in range(spec.seq_len):
        visible = time <= provenance_time
        if not force_occluded and rng.random() > spec.occlusion_rate:
            visible = True
        position = advance_position(start, delta, spec.coord_size, min(time, query_time))
        objects = [
            _object_record(focus_object, color, shape, position, delta, visible, time)
        ]
        for distractor in range(spec.distractor_count):
            object_id = (focus_object + 1 + distractor) % spec.object_count
            dcolor, dshape = _focus_color_shape(index + distractor + 7, caps)
            ddelta = action_delta((action + distractor + 2) % spec.action_count)
            dstart = (
                (start[0] + distractor + 1) % spec.coord_size,
                (start[1] + 2 * distractor + 1) % spec.coord_size,
                (start[2] + 3 * distractor + 1) % spec.coord_size,
            )
            dpos = advance_position(dstart, ddelta, spec.coord_size, time)
            objects.append(_object_record(object_id, dcolor, dshape, dpos, ddelta, True, time))
        observations.append({"time": time, "objects": objects})
        actions.append(
            {
                "time": time,
                "action": action,
                "focus_object": focus_object,
                "delta": delta,
            }
        )
    current = {
        "object_id": focus_object,
        "color": color,
        "shape": shape,
        "position": current_position,
        "velocity": delta,
        "velocity_code": velocity_code(delta),
        "action": action,
        "time": query_time,
        "provenance_time": provenance_time,
    }
    branch = {
        "object_id": focus_object,
        "color": color,
        "shape": shape,
        "position": branch_position,
        "velocity": branch_delta,
        "velocity_code": velocity_code(branch_delta),
        "action": branch_action,
        "time": query_time,
        "provenance_time": provenance_time,
    }
    return {
        "episode_id": f"{profile}-{split}-{axis}-{index}",
        "profile": profile,
        "split": split,
        "axis": axis,
        "hidden_state": {"focus": current, "branch": branch},
        "observations": observations,
        "actions": actions,
        "query": {
            "query_type": query_type,
            "focus_object": focus_object,
            "counterfactual_action": branch_action,
            "query_time": query_time,
            "occlusion_start": occlusion_start,
        },
        "labels": {
            "current": current,
            "branch": branch,
            "answer_text": answer_text_from_state(current, profile),
            "branch_answer_text": answer_text_from_state(branch, profile),
        },
        "memory_relevant_positions": [provenance_position, current_position, branch_position],
        "distractors": spec.distractor_count,
        "bit_budget": compact_bit_budget(caps),
    }


def answer_text_from_state(state: dict[str, Any], profile: str) -> str:
    coord_size = NM_3D_PROFILES[profile].coord_size
    x, y, z = state["position"]
    if coord_size > len(NUMBER_WORDS):
        raise ValueError("number vocabulary does not cover profile coordinates")
    return f"{COLORS[state['color']]} {SHAPES[state['shape']]} at {NUMBER_WORDS[x]} {NUMBER_WORDS[y]} {NUMBER_WORDS[z]}"


def world_code_fields(record: dict[str, Any], caps: dict[str, int], target: str = "current") -> dict[str, int]:
    state = record["labels"][target]
    return {
        "address": state["color"] * caps["n_shapes"] + state["shape"],
        "schema": state["velocity_code"],
        "residual": flatten_position(state["position"], caps["coord_size"]),
        "action": state["action"],
        "provenance": state["provenance_time"],
    }


def state_from_world_fields(fields: dict[str, int], caps: dict[str, int]) -> dict[str, Any]:
    color = fields["address"] // caps["n_shapes"]
    shape = fields["address"] % caps["n_shapes"]
    position = unflatten_position(fields["residual"], caps["coord_size"])
    velocity_score = fields["schema"] - 3
    return {
        "color": color,
        "shape": shape,
        "position": position,
        "velocity_code": fields["schema"],
        "velocity_score": velocity_score,
        "action": fields["action"],
        "provenance_time": fields["provenance"],
    }


def split_records(dataset: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [record for record in dataset if record["split"] == split]


def build_dataset(
    profile: str,
    seed: int,
    train_episodes: int | None = None,
    val_episodes: int | None = None,
    test_episodes: int | None = None,
    axis: str = "identity_position_band",
) -> list[dict[str, Any]]:
    spec = NM_3D_PROFILES[profile]
    counts = {
        "train": train_episodes if train_episodes is not None else spec.train_count,
        "val": val_episodes if val_episodes is not None else spec.val_count,
        "test": test_episodes if test_episodes is not None else spec.test_count,
    }
    records: list[dict[str, Any]] = []
    for split, count in counts.items():
        for index in range(count):
            records.append(generate_episode(profile, seed, split, index, axis))
    return records


def query_prompt(record: dict[str, Any]) -> str:
    q = record["query"]
    focus = q["focus_object"]
    parts = [f"query {q['query_type']} object {focus}"]
    for frame in record["observations"]:
        for obj in frame["objects"]:
            if obj["object_id"] == focus and obj["visible"]:
                parts.append(
                    f"seen t {obj['time']} color {COLORS[obj['color']]} shape {SHAPES[obj['shape']]} pos {obj['x']} {obj['y']} {obj['z']}"
                )
            elif obj["object_id"] == focus:
                parts.append(f"hidden t {obj['time']}")
    return " | ".join(parts)


def current_observation_for_focus(record: dict[str, Any]) -> dict[str, Any]:
    focus = record["query"]["focus_object"]
    frame = record["observations"][record["query"]["query_time"]]
    for obj in frame["objects"]:
        if obj["object_id"] == focus:
            return obj
    raise KeyError(focus)


def last_visible_focus_observation(record: dict[str, Any]) -> dict[str, Any]:
    focus = record["query"]["focus_object"]
    for frame in reversed(record["observations"]):
        for obj in frame["objects"]:
            if obj["object_id"] == focus and obj["visible"]:
                return obj
    raise KeyError(focus)
