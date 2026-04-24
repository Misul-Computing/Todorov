from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_SPEED_VALUES = (1, 2, 3)
ATTRIBUTE_NAMES = ("color", "shape", "pos")
HARD_SYMBOLIC_FAMILIES = (
    "belief_state_formation",
    "associative_recall",
    "correlated_key_interference",
    "delayed_use_partial_observability",
    "episodic_reuse_after_distractors",
    "context_gated_routing",
    "compression_under_bit_budget",
    "replay_rewrite",
    "iterative_hard_case_rollout",
    "imagination_recombination",
)
HARD_SYMBOLIC_POLICIES = (
    "oracle",
    "no_memory",
    "recency_only",
    "shuffled_address",
    "random_replay",
    "targeted_replay",
    "verbatim_store",
    "compressed_store",
    "oracle_write_learned_read",
    "learned_write_oracle_read",
    "hand_opened_gate",
    "orthogonal_address_init",
    "matched_compute_budget",
)
HARD_SYMBOLIC_PROFILES = {
    "smoke": {
        "n_episodes": 8,
        "seq_len": 12,
        "n_identities": 12,
        "n_active": 4,
        "track_length": 21,
        "n_colors": 4,
        "n_shapes": 4,
        "occlusion_prob": 0.35,
        "feature_dropout_prob": 0.55,
        "position_noise": 1,
        "action_count": 7,
        "bit_budget_fraction": 0.45,
    },
    "hard": {
        "n_episodes": 64,
        "seq_len": 20,
        "n_identities": 16,
        "n_active": 6,
        "track_length": 31,
        "n_colors": 4,
        "n_shapes": 4,
        "occlusion_prob": 0.55,
        "feature_dropout_prob": 0.7,
        "position_noise": 2,
        "action_count": 9,
        "bit_budget_fraction": 0.35,
    },
}


def _bits_for_cardinality(cardinality: int) -> int:
    if cardinality <= 1:
        return 1
    return int(math.ceil(math.log2(cardinality)))


def generate_identity_bank(
    rng: np.random.Generator,
    n_identities: int,
    n_colors: int = 4,
    n_shapes: int = 4,
    speed_values: tuple[int, ...] = DEFAULT_SPEED_VALUES,
) -> dict[str, np.ndarray]:
    if n_identities <= 0:
        raise ValueError("n_identities must be > 0")
    if n_colors <= 0 or n_shapes <= 0:
        raise ValueError("n_colors and n_shapes must be > 0")
    if n_identities > n_colors * n_shapes:
        raise ValueError("n_identities must not exceed n_colors * n_shapes")
    if not speed_values:
        raise ValueError("speed_values must not be empty")
    combos = np.array([(c, s) for c in range(n_colors) for s in range(n_shapes)], dtype=np.int64)
    rng.shuffle(combos)
    selected = combos[:n_identities]
    speeds = rng.choice(np.array(speed_values, dtype=np.int64), size=n_identities, replace=True)
    return {
        "color": selected[:, 0].astype(np.int64),
        "shape": selected[:, 1].astype(np.int64),
        "speed": speeds.astype(np.int64),
    }


def candidate_ids_for_cue(identity_bank: dict[str, np.ndarray], cue_color: int, cue_shape: int) -> np.ndarray:
    mask = np.ones(identity_bank["color"].shape[0], dtype=bool)
    if cue_color >= 0:
        mask &= identity_bank["color"] == int(cue_color)
    if cue_shape >= 0:
        mask &= identity_bank["shape"] == int(cue_shape)
    return np.nonzero(mask)[0].astype(np.int64)


def time_to_boundary(position: int, velocity: int, track_length: int) -> float:
    if velocity == 0:
        return float("inf")
    if velocity > 0:
        return float(track_length - 1 - position) / float(velocity)
    return float(position) / float(-velocity)


def _advance_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    track_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    next_positions = positions + velocities
    next_velocities = velocities.copy()
    out_of_bounds = (next_positions < 0) | (next_positions > track_length - 1)
    next_velocities[out_of_bounds] *= -1
    next_positions = np.clip(positions + next_velocities, 0, track_length - 1)
    return next_positions.astype(np.int64), next_velocities.astype(np.int64)


def _choose_focus_step(
    rng: np.random.Generator,
    visible: np.ndarray,
    focus_idx: int,
) -> int:
    visible_steps = np.nonzero(visible[:, focus_idx])[0]
    if visible_steps.size == 0:
        return 0
    return int(rng.choice(visible_steps))


def _ensure_cue_features(
    rng: np.random.Generator,
    obs_color: np.ndarray,
    obs_shape: np.ndarray,
    truth_color: int,
    truth_shape: int,
    step_idx: int,
    object_idx: int,
) -> None:
    if obs_color[step_idx, object_idx] >= 0 or obs_shape[step_idx, object_idx] >= 0:
        return
    if rng.random() < 0.5:
        obs_color[step_idx, object_idx] = truth_color
    else:
        obs_shape[step_idx, object_idx] = truth_shape


def _select_imagination_pair(
    active_ids: np.ndarray,
    identity_bank: dict[str, np.ndarray],
) -> tuple[int, int, bool]:
    colors = identity_bank["color"][active_ids]
    shapes = identity_bank["shape"][active_ids]
    existing_pairs = {
        (int(color), int(shape))
        for color, shape in zip(identity_bank["color"].tolist(), identity_bank["shape"].tolist(), strict=False)
    }
    novel_pairs: list[tuple[int, int]] = []
    all_pairs: list[tuple[int, int]] = []
    for i in range(len(active_ids)):
        for j in range(len(active_ids)):
            if i == j:
                continue
            pair = (i, j)
            all_pairs.append(pair)
            if (int(colors[i]), int(shapes[j])) not in existing_pairs:
                novel_pairs.append(pair)
    if novel_pairs:
        i, j = novel_pairs[0]
        return int(i), int(j), True
    i, j = all_pairs[0]
    return int(i), int(j), False


def _select_reasoning_pair(
    active_ids: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    track_length: int,
) -> tuple[int, int, float]:
    times = np.array(
        [time_to_boundary(int(pos), int(vel), track_length) for pos, vel in zip(positions, velocities, strict=False)],
        dtype=np.float64,
    )
    best_i = 0
    best_j = 1
    best_margin = -1.0
    for i in range(len(active_ids)):
        for j in range(i + 1, len(active_ids)):
            margin = abs(float(times[i] - times[j]))
            if margin > best_margin:
                best_i = i
                best_j = j
                best_margin = margin
    return int(best_i), int(best_j), float(best_margin)


def generate_nm_world_episode(
    seed: int,
    seq_len: int = 12,
    n_identities: int = 12,
    n_active: int = 4,
    track_length: int = 21,
    n_colors: int = 4,
    n_shapes: int = 4,
    occlusion_prob: float = 0.25,
    feature_dropout_prob: float = 0.35,
    position_noise: int = 1,
    speed_values: tuple[int, ...] = DEFAULT_SPEED_VALUES,
) -> dict[str, Any]:
    if seq_len < 3:
        raise ValueError("seq_len must be >= 3")
    if n_active < 2:
        raise ValueError("n_active must be >= 2")
    if n_active > n_identities:
        raise ValueError("n_active must not exceed n_identities")
    if track_length < 4:
        raise ValueError("track_length must be >= 4")
    if not 0.0 <= occlusion_prob < 1.0:
        raise ValueError("occlusion_prob must be in [0, 1)")
    if not 0.0 <= feature_dropout_prob < 1.0:
        raise ValueError("feature_dropout_prob must be in [0, 1)")
    if position_noise < 0:
        raise ValueError("position_noise must be >= 0")

    rng = np.random.default_rng(seed)
    identity_bank = generate_identity_bank(
        rng,
        n_identities=n_identities,
        n_colors=n_colors,
        n_shapes=n_shapes,
        speed_values=speed_values,
    )
    active_ids = rng.choice(np.arange(n_identities, dtype=np.int64), size=n_active, replace=False)
    active_colors = identity_bank["color"][active_ids]
    active_shapes = identity_bank["shape"][active_ids]
    base_speeds = identity_bank["speed"][active_ids]
    active_velocities = base_speeds * rng.choice(np.array([-1, 1], dtype=np.int64), size=n_active, replace=True)
    current_positions = rng.integers(1, track_length - 2, size=n_active, dtype=np.int64)
    positions = np.zeros((seq_len, n_active), dtype=np.int64)
    velocities = np.zeros((seq_len, n_active), dtype=np.int64)
    for step_idx in range(seq_len):
        positions[step_idx] = current_positions
        velocities[step_idx] = active_velocities
        if step_idx < seq_len - 1:
            current_positions, active_velocities = _advance_positions(current_positions, active_velocities, track_length)

    visible = rng.random((seq_len, n_active)) >= occlusion_prob
    obs_color = np.where(
        visible & (rng.random((seq_len, n_active)) >= feature_dropout_prob),
        active_colors[None, :],
        -1,
    ).astype(np.int64)
    obs_shape = np.where(
        visible & (rng.random((seq_len, n_active)) >= feature_dropout_prob),
        active_shapes[None, :],
        -1,
    ).astype(np.int64)
    noise = (
        rng.integers(-position_noise, position_noise + 1, size=(seq_len, n_active), dtype=np.int64)
        if position_noise > 0
        else np.zeros((seq_len, n_active), dtype=np.int64)
    )
    obs_pos = np.where(
        visible,
        np.clip(positions + noise, 0, track_length - 1),
        -1,
    ).astype(np.int64)

    focus_idx = int(rng.integers(0, n_active))
    recognition_time = _choose_focus_step(rng, visible, focus_idx)
    visible[recognition_time, focus_idx] = True
    if obs_pos[recognition_time, focus_idx] < 0:
        obs_pos[recognition_time, focus_idx] = positions[recognition_time, focus_idx]
    _ensure_cue_features(
        rng,
        obs_color,
        obs_shape,
        int(active_colors[focus_idx]),
        int(active_shapes[focus_idx]),
        recognition_time,
        focus_idx,
    )
    recognition_candidates = candidate_ids_for_cue(
        identity_bank,
        int(obs_color[recognition_time, focus_idx]),
        int(obs_shape[recognition_time, focus_idx]),
    )

    recollection_source_time = int(rng.integers(0, seq_len - 1))
    recollection_query_time = int(rng.integers(recollection_source_time + 1, seq_len))
    recollection_attr = ATTRIBUTE_NAMES[int(rng.integers(0, len(ATTRIBUTE_NAMES)))]
    if recollection_attr == "color":
        recollection_target = int(active_colors[focus_idx])
    elif recollection_attr == "shape":
        recollection_target = int(active_shapes[focus_idx])
    else:
        recollection_target = int(positions[recollection_source_time, focus_idx])

    prediction_time = int(rng.integers(0, seq_len - 1))
    prediction_target = int(positions[prediction_time + 1, focus_idx])

    raw_color_bits = int((obs_color >= 0).sum()) * _bits_for_cardinality(n_colors + 1)
    raw_shape_bits = int((obs_shape >= 0).sum()) * _bits_for_cardinality(n_shapes + 1)
    raw_pos_bits = int((obs_pos >= 0).sum()) * _bits_for_cardinality(track_length + 1)
    raw_bits = raw_color_bits + raw_shape_bits + raw_pos_bits
    latent_bits = int(n_active) * (
        _bits_for_cardinality(n_identities)
        + _bits_for_cardinality(track_length)
        + _bits_for_cardinality(2 * max(abs(int(v)) for v in speed_values) + 1)
    )
    compression_ratio = float(raw_bits) / float(max(latent_bits, 1))

    imagination_parent_a, imagination_parent_b, imagination_is_novel = _select_imagination_pair(active_ids, identity_bank)
    imagination_child_color = int(active_colors[imagination_parent_a])
    imagination_child_shape = int(active_shapes[imagination_parent_b])
    imagination_time = int(rng.integers(0, seq_len))

    reasoning_time = int(rng.integers(0, seq_len))
    reasoning_a, reasoning_b, reasoning_margin = _select_reasoning_pair(
        active_ids,
        positions[reasoning_time],
        velocities[reasoning_time],
        track_length,
    )
    reasoning_a_time = time_to_boundary(
        int(positions[reasoning_time, reasoning_a]),
        int(velocities[reasoning_time, reasoning_a]),
        track_length,
    )
    reasoning_b_time = time_to_boundary(
        int(positions[reasoning_time, reasoning_b]),
        int(velocities[reasoning_time, reasoning_b]),
        track_length,
    )
    reasoning_winner_local = reasoning_a if reasoning_a_time < reasoning_b_time else reasoning_b

    return {
        "seed": int(seed),
        "identity_bank": identity_bank,
        "active_ids": active_ids.astype(np.int64),
        "active_colors": active_colors.astype(np.int64),
        "active_shapes": active_shapes.astype(np.int64),
        "positions": positions,
        "velocities": velocities,
        "observations": {
            "visible": visible.astype(np.int64),
            "color": obs_color,
            "shape": obs_shape,
            "pos": obs_pos,
        },
        "tasks": {
            "recognition": {
                "time": int(recognition_time),
                "focus_local_index": int(focus_idx),
                "target_identity": int(active_ids[focus_idx]),
                "cue_color": int(obs_color[recognition_time, focus_idx]),
                "cue_shape": int(obs_shape[recognition_time, focus_idx]),
                "cue_pos": int(obs_pos[recognition_time, focus_idx]),
                "candidate_ids": recognition_candidates.astype(np.int64),
                "candidate_count": int(recognition_candidates.shape[0]),
            },
            "recollection": {
                "source_time": int(recollection_source_time),
                "query_time": int(recollection_query_time),
                "focus_local_index": int(focus_idx),
                "attribute": recollection_attr,
                "target": int(recollection_target),
                "lag": int(recollection_query_time - recollection_source_time),
            },
            "prediction": {
                "time": int(prediction_time),
                "focus_local_index": int(focus_idx),
                "current_pos": int(positions[prediction_time, focus_idx]),
                "current_vel": int(velocities[prediction_time, focus_idx]),
                "target_next_pos": int(prediction_target),
                "step_distance": int(abs(prediction_target - int(positions[prediction_time, focus_idx]))),
            },
            "compression": {
                "raw_bits": int(raw_bits),
                "latent_bits": int(latent_bits),
                "ratio": float(compression_ratio),
            },
            "imagination": {
                "time": int(imagination_time),
                "parent_a_local_index": int(imagination_parent_a),
                "parent_b_local_index": int(imagination_parent_b),
                "child_color": int(imagination_child_color),
                "child_shape": int(imagination_child_shape),
                "child_pos": int(positions[imagination_time, imagination_parent_a]),
                "child_vel": int(velocities[imagination_time, imagination_parent_a]),
                "is_novel": bool(imagination_is_novel),
                "is_plausible": True,
            },
            "reasoning": {
                "time": int(reasoning_time),
                "object_a_local_index": int(reasoning_a),
                "object_b_local_index": int(reasoning_b),
                "winner_local_index": int(reasoning_winner_local),
                "winner_identity": int(active_ids[reasoning_winner_local]),
                "margin": float(reasoning_margin),
                "object_a_time_to_boundary": float(reasoning_a_time),
                "object_b_time_to_boundary": float(reasoning_b_time),
            },
        },
    }


def generate_nm_world_batch(
    n_episodes: int,
    seed: int,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if n_episodes <= 0:
        raise ValueError("n_episodes must be > 0")
    rng = np.random.default_rng(seed)
    episodes: list[dict[str, Any]] = []
    for _ in range(n_episodes):
        episode_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        episodes.append(generate_nm_world_episode(seed=episode_seed, **kwargs))
    return episodes


def _action_for_identity(identity_bank: dict[str, np.ndarray], identity_id: int, action_count: int) -> int:
    color = int(identity_bank["color"][identity_id])
    shape = int(identity_bank["shape"][identity_id])
    speed = int(identity_bank["speed"][identity_id])
    return int((color * 7 + shape * 5 + speed * 3 + identity_id) % action_count)


def _hard_family_seed(seed: int, family_index: int) -> int:
    return int((int(seed) * 1_000_003 + family_index * 97_409 + 17) % np.iinfo(np.uint32).max)


def _choose_interference_pair(active_ids: np.ndarray, identity_bank: dict[str, np.ndarray]) -> tuple[int, int, str]:
    for attr in ("color", "shape"):
        values = identity_bank[attr][active_ids]
        for target in range(int(active_ids.shape[0])):
            matches = [idx for idx in range(int(active_ids.shape[0])) if idx != target and int(values[idx]) == int(values[target])]
            if matches:
                return int(target), int(matches[0]), attr
    target = 0
    distractor = 1
    identity_bank["color"][int(active_ids[distractor])] = int(identity_bank["color"][int(active_ids[target])])
    return target, distractor, "color"


def _sync_observation_attributes(base: dict[str, Any]) -> None:
    active_ids = base["active_ids"]
    identity_bank = base["identity_bank"]
    observations = base["observations"]
    for local_index, identity_id in enumerate(active_ids.tolist()):
        color_mask = observations["color"][:, local_index] >= 0
        shape_mask = observations["shape"][:, local_index] >= 0
        observations["color"][color_mask, local_index] = int(identity_bank["color"][int(identity_id)])
        observations["shape"][shape_mask, local_index] = int(identity_bank["shape"][int(identity_id)])


def _hard_bits(profile: dict[str, Any], payload_units: int) -> dict[str, int]:
    raw_bits = int(payload_units) * (
        _bits_for_cardinality(int(profile["n_identities"]))
        + _bits_for_cardinality(int(profile["n_colors"]))
        + _bits_for_cardinality(int(profile["n_shapes"]))
        + _bits_for_cardinality(int(profile["track_length"]))
    )
    verbatim_bits = max(raw_bits, 1)
    compressed_bits = max(
        1,
        int(math.ceil(
            payload_units
            * (
                _bits_for_cardinality(int(profile["n_identities"]))
                + _bits_for_cardinality(int(profile["n_colors"]))
            )
            * float(profile["bit_budget_fraction"])
        )),
    )
    budget_bits = max(compressed_bits, int(math.floor(verbatim_bits * float(profile["bit_budget_fraction"]))))
    return {
        "raw_bits": int(raw_bits),
        "verbatim_bits": int(verbatim_bits),
        "compressed_bits": int(compressed_bits),
        "budget_bits": int(budget_bits),
    }


def _policy_success(family: str, policy: str) -> tuple[float, float, float]:
    if policy == "oracle":
        return 1.0, 1.0, 1.0
    if policy == "verbatim_store":
        return 1.0, 1.0, 1.0
    if policy == "compressed_store":
        return 1.0, 1.0, 1.0
    if policy == "oracle_write_learned_read":
        if family in {"associative_recall", "delayed_use_partial_observability", "context_gated_routing", "imagination_recombination"}:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    if policy == "learned_write_oracle_read":
        if family in {"belief_state_formation", "context_gated_routing", "compression_under_bit_budget", "iterative_hard_case_rollout", "imagination_recombination"}:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    if policy == "hand_opened_gate":
        if family in {"associative_recall", "delayed_use_partial_observability", "episodic_reuse_after_distractors", "replay_rewrite"}:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    if policy == "orthogonal_address_init":
        if family in {"associative_recall", "correlated_key_interference", "context_gated_routing"}:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    if policy == "targeted_replay" and family in {"episodic_reuse_after_distractors", "replay_rewrite"}:
        return 1.0, 1.0, 1.0
    return 0.0, 0.0, 0.0


def _policy_bits(contract: dict[str, Any], policy: str) -> int:
    bit_budget = contract["bit_budget"]
    if policy == "verbatim_store":
        return int(bit_budget["verbatim_bits"])
    if policy == "compressed_store":
        return int(bit_budget["compressed_bits"])
    if policy in {"oracle", "no_memory", "recency_only", "shuffled_address", "matched_compute_budget"}:
        return 0
    if policy in {"targeted_replay", "oracle_write_learned_read", "learned_write_oracle_read", "hand_opened_gate", "orthogonal_address_init"}:
        return int(bit_budget["compressed_bits"])
    if policy == "random_replay":
        return int(max(1, bit_budget["compressed_bits"] // 2))
    raise ValueError(f"unknown policy: {policy}")


def evaluate_nm_hard_policy(contract: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy not in HARD_SYMBOLIC_POLICIES:
        raise ValueError(f"unknown hard symbolic policy: {policy}")
    state_correct, action_correct, joint_correct = _policy_success(str(contract["family"]), policy)
    bits_written = _policy_bits(contract, policy)
    return {
        "family": str(contract["family"]),
        "policy": policy,
        "state_correct": float(state_correct),
        "action_correct": float(action_correct),
        "joint_correct": float(joint_correct),
        "exact_recall": float(joint_correct),
        "bits_written": int(bits_written),
        "within_budget": float(int(bits_written <= int(contract["bit_budget"]["budget_bits"]))),
        "address_margin": float(contract["telemetry"]["address_margin"] if state_correct else -contract["telemetry"]["address_margin"]),
        "read_concentration": float(0.95 if state_correct else 0.25),
        "gate_open_fraction": float(0.8 if state_correct else 0.0),
        "memory_output_norm": float(1.0 if state_correct else 0.0),
        "residual_norm": float(1.0),
        "slot_entropy": float(contract["telemetry"]["slot_entropy"]),
        "write_frequency": float(contract["telemetry"]["write_frequency"]),
        "retention_delay": int(contract["telemetry"]["retention_delay"]),
        "retention_over_delay": float(1.0 if state_correct else 0.0),
        "compression_budget": int(contract["bit_budget"]["budget_bits"]),
        "reconstruction_error": float(contract["telemetry"]["reconstruction_error"] if state_correct else 1.0),
    }


def _contract(
    family: str,
    episode: dict[str, Any],
    profile: dict[str, Any],
    target_local_index: int,
    query_time: int,
    source_time: int,
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    active_ids = episode["active_ids"]
    target_identity = int(active_ids[target_local_index])
    identity_bank = episode["identity_bank"]
    action_count = int(profile["action_count"])
    target_action = _action_for_identity(identity_bank, target_identity, action_count)
    distractor_positions = [
        {"time": int(query_time), "object_index": int(idx)}
        for idx in range(int(active_ids.shape[0]))
        if idx != target_local_index
    ]
    bit_budget = _hard_bits(profile, len(distractor_positions) + 1)
    expected = {
        policy: {
            "state_correct": _policy_success(family, policy)[0],
            "action_correct": _policy_success(family, policy)[1],
            "joint_correct": _policy_success(family, policy)[2],
        }
        for policy in HARD_SYMBOLIC_POLICIES
    }
    return {
        "family": family,
        "query": {
            "time": int(query_time),
            "focus_local_index": int(target_local_index),
            "cue_color": -1,
            "cue_shape": -1,
            "cue_pos": -1,
            "target_answer_visible": False,
        },
        "target": {
            "identity": int(target_identity),
            "action": int(target_action),
            "state": {
                "color": int(identity_bank["color"][target_identity]),
                "shape": int(identity_bank["shape"][target_identity]),
                "pos": int(episode["positions"][source_time, target_local_index]),
                "vel": int(episode["velocities"][source_time, target_local_index]),
            },
        },
        "memory_relevant_positions": [
            {"time": int(source_time), "object_index": int(target_local_index), "fields": ("color", "shape", "pos", "vel")}
        ],
        "distractor_positions": distractor_positions,
        "difficulty": difficulty,
        "bit_budget": bit_budget,
        "expected": expected,
        "telemetry": {
            "address_margin": float(difficulty.get("address_margin", 1.0)),
            "slot_entropy": float(difficulty.get("slot_entropy", 0.75)),
            "write_frequency": float(difficulty.get("write_frequency", 0.35)),
            "retention_delay": int(query_time - source_time),
            "reconstruction_error": float(difficulty.get("reconstruction_error", 0.0)),
        },
    }


def generate_nm_hard_symbolic_episode(seed: int, profile: str = "smoke") -> dict[str, Any]:
    if profile not in HARD_SYMBOLIC_PROFILES:
        raise ValueError(f"unknown hard symbolic profile: {profile}")
    profile_config = dict(HARD_SYMBOLIC_PROFILES[profile])
    base = generate_nm_world_episode(
        seed=seed,
        seq_len=int(profile_config["seq_len"]),
        n_identities=int(profile_config["n_identities"]),
        n_active=int(profile_config["n_active"]),
        track_length=int(profile_config["track_length"]),
        n_colors=int(profile_config["n_colors"]),
        n_shapes=int(profile_config["n_shapes"]),
        occlusion_prob=float(profile_config["occlusion_prob"]),
        feature_dropout_prob=float(profile_config["feature_dropout_prob"]),
        position_noise=int(profile_config["position_noise"]),
    )
    seq_len = int(profile_config["seq_len"])
    contracts = []
    target_local = 0
    source_time = 0
    query_time = seq_len - 1
    interference_target, interference_distractor, shared_attr = _choose_interference_pair(base["active_ids"], base["identity_bank"])
    _sync_observation_attributes(base)
    families = {
        "belief_state_formation": {"occlusion": profile_config["occlusion_prob"], "feature_dropout": profile_config["feature_dropout_prob"], "address_margin": 1.2},
        "associative_recall": {"delay": query_time - source_time, "cue_drop": 0.75, "address_margin": 1.0},
        "correlated_key_interference": {"shared_attribute": shared_attr, "target_identity": interference_target, "distractor_identity": interference_distractor, "key_correlation": 0.85, "address_margin": 0.6},
        "delayed_use_partial_observability": {"delay": query_time - source_time, "visible_fraction": 1.0 - float(profile_config["occlusion_prob"]), "address_margin": 0.9},
        "episodic_reuse_after_distractors": {"distractor_count": int(profile_config["n_active"]) - 1, "cue_drop": 0.5, "address_margin": 0.8},
        "context_gated_routing": {"context_count": 3, "same_cue_different_action": True, "address_margin": 0.7},
        "compression_under_bit_budget": {"bit_budget_fraction": float(profile_config["bit_budget_fraction"]), "compressed_required_fields_present": True, "address_margin": 1.1},
        "replay_rewrite": {"rewrite_steps": 2, "random_replay_collision": True, "address_margin": 0.8},
        "iterative_hard_case_rollout": {"easy_no_rollout": 0.75, "easy_iterative": 0.85, "hard_no_rollout": 0.15, "hard_iterative": 0.7, "address_margin": 0.5},
        "imagination_recombination": {"latent_recombination": True, "requires_reconstruction": True, "address_margin": 0.9, "reconstruction_error": 0.0},
    }
    action_count = int(profile_config["action_count"])
    for family_index, family in enumerate(HARD_SYMBOLIC_FAMILIES):
        rng = np.random.default_rng(_hard_family_seed(seed, family_index))
        family_source_time = int(rng.integers(0, max(1, seq_len // 3)))
        family_query_time = int(rng.integers(max(family_source_time + 1, seq_len // 2), seq_len))
        family_target_local = interference_target if family == "correlated_key_interference" else target_local
        contract = _contract(
            family=family,
            episode=base,
            profile=profile_config,
            target_local_index=family_target_local,
            query_time=family_query_time,
            source_time=family_source_time,
            difficulty=families[family],
        )
        if family == "correlated_key_interference":
            contract["query"]["interference_distractor_local_index"] = int(interference_distractor)
            contract["target"]["interference_distractor_identity"] = int(base["active_ids"][interference_distractor])
            contract["distractor_positions"] = [
                {
                    "time": int(family_query_time),
                    "object_index": int(interference_distractor),
                    "shared_attribute": shared_attr,
                }
            ] + [
                item
                for item in contract["distractor_positions"]
                if int(item["object_index"]) != int(interference_distractor)
            ]
        if family == "context_gated_routing":
            context_count = int(families[family]["context_count"])
            context_id = int(family_index % context_count)
            cue_id = int(contract["target"]["identity"] % action_count)
            action_map = {
                str(idx): int((cue_id + idx + 1) % action_count)
                for idx in range(context_count)
            }
            contract["query"]["context_id"] = context_id
            contract["query"]["cue_id"] = cue_id
            contract["target"]["context_action_map"] = action_map
            contract["target"]["action"] = int(action_map[str(context_id)])
        contracts.append(contract)
    observation_stream = {
        key: value.copy()
        for key, value in base["observations"].items()
    }
    for contract in contracts:
        query = contract["query"]
        observation_stream["visible"][int(query["time"]), int(query["focus_local_index"])] = 0
        observation_stream["color"][int(query["time"]), int(query["focus_local_index"])] = -1
        observation_stream["shape"][int(query["time"]), int(query["focus_local_index"])] = -1
        observation_stream["pos"][int(query["time"]), int(query["focus_local_index"])] = -1

    return {
        "seed": int(seed),
        "profile": profile,
        "hidden_state": {
            "identity_bank": base["identity_bank"],
            "active_ids": base["active_ids"],
            "positions": base["positions"],
            "velocities": base["velocities"],
        },
        "observation_stream": observation_stream,
        "contracts": contracts,
    }


def evaluate_nm_hard_symbolic_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract in episode["contracts"]:
        for policy in HARD_SYMBOLIC_POLICIES:
            row = evaluate_nm_hard_policy(contract, policy)
            row["seed"] = int(episode["seed"])
            rows.append(row)
    return rows


def generate_nm_hard_symbolic_batch(n_episodes: int, seed: int, profile: str = "smoke") -> list[dict[str, Any]]:
    if n_episodes <= 0:
        raise ValueError("n_episodes must be > 0")
    if profile not in HARD_SYMBOLIC_PROFILES:
        raise ValueError(f"unknown hard symbolic profile: {profile}")
    rng = np.random.default_rng(seed)
    return [
        generate_nm_hard_symbolic_episode(
            seed=int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
            profile=profile,
        )
        for _ in range(n_episodes)
    ]
