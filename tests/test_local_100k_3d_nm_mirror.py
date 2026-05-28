from neuroloc.data.nm_3d_worlds import (
    NM_3D_PROFILES,
    build_dataset,
    current_observation_for_focus,
    flatten_position,
    profile_caps,
    query_prompt,
    split_records,
    unflatten_position,
    world_code_fields,
)
from neuroloc.simulations.memory.local_100k_3d_nm_mirror import (
    AXES,
    build_summary,
    counterfactual_fields_from_compact,
    feature_fields_from_observation,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_nm_3d_world_generation_is_deterministic_by_seed() -> None:
    first = build_dataset("smoke", 11, 4, 2, 3, "axis")
    second = build_dataset("smoke", 11, 4, 2, 3, "axis")
    third = build_dataset("smoke", 12, 4, 2, 3, "axis")
    fourth = build_dataset("smoke", 11, 4, 2, 3, "other_axis")
    assert first == second
    assert first != third
    assert first != fourth


def test_hidden_state_and_observation_are_separated_under_occlusion() -> None:
    record = split_records(build_dataset("smoke", 13, 4, 1, 2, "axis"), "test")[0]
    current = current_observation_for_focus(record)
    hidden = record["labels"]["current"]
    assert current["visible"] is False
    assert (current["x"], current["y"], current["z"]) == (-1, -1, -1)
    assert hidden["position"] != (-1, -1, -1)
    assert query_prompt(record).find(str(hidden["position"])) == -1


def test_action_transition_and_counterfactual_targets_are_exact_and_distinct() -> None:
    caps = profile_caps("hard")
    record = split_records(build_dataset("hard", 17, 8, 2, 4, "axis"), "test")[1]
    fields = world_code_fields(record, caps)
    branch = counterfactual_fields_from_compact(fields, int(record["query"]["counterfactual_action"]), caps)
    assert branch != fields
    assert unflatten_position(fields["residual"], caps["coord_size"]) == record["labels"]["current"]["position"]
    assert 0 <= branch["residual"] < caps["track_length"]


def test_number_coordinate_vocabulary_covers_hard_profile() -> None:
    hard = NM_3D_PROFILES["hard"]
    for value in range(hard.coord_size):
        assert flatten_position((value, value, value), hard.coord_size) < hard.coord_size**3


def test_observation_bridge_recovers_state_without_current_observation_leakage() -> None:
    caps = profile_caps("smoke")
    record = split_records(build_dataset("smoke", 23, 16, 2, 4, "axis"), "test")[2]
    assert current_observation_for_focus(record)["visible"] is False
    assert feature_fields_from_observation(record, caps) == world_code_fields(record, caps)


def test_local_100k_3d_nm_smoke_gate_passes_single_axis() -> None:
    summary = build_summary("smoke", seed=31, train_episodes=192, val_episodes=8, test_episodes=25, binder_epochs=1, state_epochs=360, answer_epochs=1, branch_epochs=1, state_width=64, branch_width=1, seed_count=1, axes=(AXES[0],))
    assert summary["local_100k_3d_nm_evaluated"] == 1.0
    assert summary["local_100k_3d_nm_engineering_pass"] == 1.0
    assert summary["local_100k_3d_nm_candidate_authorized"] == 1.0
    assert summary["local_100k_3d_nm_full_model_authorized"] == 0.0
    assert summary["local_100k_3d_nm_paid_compute_authorized"] == 0.0
    assert summary["local_100k_3d_nm_arbitrary_chat_authorized"] == 0.0
    assert summary["local_100k_3d_nm_parameter_count_max"] < 100000
    assert summary["local_100k_3d_nm_initial_world_state_joint_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_object_permanence_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_occluded_localization_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_action_consequence_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_targeted_replay_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_rewrite_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_counterfactual_exact_transition_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_hard_case_branch_gain_min"] > summary["local_100k_3d_nm_easy_case_branch_gain_max"]
    assert summary["local_100k_3d_nm_matched_budget_sparse_read_success_max"] == 0.0
    assert summary["local_100k_3d_nm_no_memory_success_max"] == 0.0
    assert summary["local_100k_3d_nm_recency_only_success_max"] == 0.0
    assert summary["local_100k_3d_nm_no_replay_success_max"] == 0.0
    assert summary["local_100k_3d_nm_random_replay_success_max"] == 0.0
    assert summary["local_100k_3d_nm_no_integration_success_max"] == 0.0
    assert summary["local_100k_3d_nm_wrong_dynamics_success_max"] == 0.0
    assert summary["local_100k_3d_nm_no_branch_success_max"] == 0.0
    assert summary["local_100k_3d_nm_wrong_branch_success_max"] == 0.0
    assert summary["local_100k_3d_nm_decoder_disabled_success_max"] == 0.0
    assert summary["local_100k_3d_nm_rewrite_provenance_success_min"] >= 0.95
    assert summary["local_100k_3d_nm_branch_provenance_success_min"] >= 0.95


def test_local_100k_3d_nm_registry_entry() -> None:
    assert "local_100k_3d_nm_mirror" in SIMULATION_SPECS
    assert "local_100k_3d_nm_mirror" in SUITES["compression_mirror"]
    assert "local_100k_3d_nm_mirror" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_3d_nm_mirror"]
    assert spec.category == "compression_mirror"
    assert dict(spec.maximum_summary_values)["local_100k_3d_nm_parameter_count_max"] == 99999.0
    assert dict(spec.minimum_summary_values)["local_100k_3d_nm_engineering_pass"] == 1.0
