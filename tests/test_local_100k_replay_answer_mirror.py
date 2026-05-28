from neuroloc.simulations.memory.local_100k_replay_answer_mirror import (
    answer_text_from_row,
    build_summary,
    disabled_answer_rows,
    shifted_answer_rows,
)
from neuroloc.simulations.memory.compression_under_bit_budget_mirror import profile_caps
from neuroloc.simulations.memory.language_grounded_state_density_mirror import NUMBER_WORDS, word_at
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_local_100k_answer_rows_have_bounded_schema() -> None:
    caps = profile_caps("smoke")
    rows = disabled_answer_rows(2)
    assert len(rows) == 2
    assert set(rows[0]) == {"color", "shape", "pos", "vel_code", "action", "provenance"}
    text = answer_text_from_row({"color": 1, "shape": 2, "pos": 3, "vel_code": int(caps["max_speed"]) + 1, "action": 4, "provenance": 5}, caps)
    assert text == "answer action_4 color_1 shape_2 pos_3 vel_1"


def test_hard_profile_number_words_do_not_wrap_track_positions() -> None:
    assert len(NUMBER_WORDS) >= int(profile_caps("hard")["track_length"])
    assert word_at(NUMBER_WORDS, 30) == "thirty"


def test_local_100k_shuffled_answer_control_changes_rows() -> None:
    rows = [
        {"color": 1, "shape": 1, "pos": 1, "vel_code": 1, "action": 1, "provenance": 1},
        {"color": 2, "shape": 2, "pos": 2, "vel_code": 2, "action": 2, "provenance": 2},
    ]
    shifted = shifted_answer_rows(rows)
    assert shifted[0]["color"] == 2
    assert shifted[1]["color"] == 1


def test_local_100k_replay_answer_gate_passes_single_axis_smoke() -> None:
    summary = build_summary("smoke", seed=123, train_episodes=512, val_episodes=16, test_episodes=32, state_epochs=140, answer_epochs=120, state_width=80, seed_count=1, axes=("color_shape_pair_band",))
    assert summary["local_100k_replay_answer_evaluated"] == 1.0
    assert summary["local_100k_replay_answer_local_model_candidate_authorized"] == 1.0
    assert summary["local_100k_replay_answer_full_model_authorized"] == 0.0
    assert summary["local_100k_replay_answer_paid_compute_authorized"] == 0.0
    assert summary["local_100k_replay_answer_arbitrary_chat_authorized"] == 0.0
    assert summary["local_100k_replay_answer_parameter_count_max"] < 100000
    assert summary["local_100k_replay_answer_initial_joint_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_initial_state_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_initial_action_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_field_accuracy_floor"] >= 0.95
    assert summary["local_100k_replay_answer_targeted_replay_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_rewrite_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_branch_rollout_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_no_replay_success_max"] == 0.0
    assert summary["local_100k_replay_answer_recency_replay_success_max"] == 0.0
    assert summary["local_100k_replay_answer_matched_compute_dummy_replay_success_max"] == 0.0
    assert summary["local_100k_replay_answer_decoder_disabled_success_max"] == 0.0
    assert summary["local_100k_replay_answer_no_rewrite_success_max"] == 0.0
    assert summary["local_100k_replay_answer_random_rewrite_success_max"] == 0.0
    assert summary["local_100k_replay_answer_no_branch_success_max"] == 0.0
    assert summary["local_100k_replay_answer_wrong_branch_success_max"] == 0.0
    assert summary["local_100k_replay_answer_random_branch_success_max"] == 0.0
    assert summary["local_100k_replay_answer_hard_case_branch_gain_min"] > summary["local_100k_replay_answer_easy_case_branch_gain_max"]
    assert summary["local_100k_replay_answer_matched_sparse_joint_success_max"] == 0.0
    assert summary["local_100k_replay_answer_uncapped_sparse_joint_success_min"] >= 0.95
    assert summary["local_100k_replay_answer_accounted_bits_max"] == 56.0
    assert summary["local_100k_replay_answer_engineering_pass"] == 1.0


def test_local_100k_replay_answer_registry_entry() -> None:
    assert "local_100k_replay_answer_mirror" in SIMULATION_SPECS
    assert "local_100k_replay_answer_mirror" in SUITES["compression_mirror"]
    assert "local_100k_replay_answer_mirror" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_replay_answer_mirror"]
    assert spec.category == "compression_mirror"
    assert dict(spec.maximum_summary_values)["local_100k_replay_answer_parameter_count_max"] == 99999.0
    assert dict(spec.maximum_summary_values)["local_100k_replay_answer_full_model_authorized"] == 0.0
    assert dict(spec.minimum_summary_values)["local_100k_replay_answer_engineering_pass"] == 1.0
