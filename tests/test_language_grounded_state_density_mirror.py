from neuroloc.simulations.memory.language_grounded_state_density_mirror import (
    answer_text,
    build_summary,
    build_parser_resistant_summary,
    parse_prompt,
    randomized_record_prompt,
    record_prompt,
    tokenize_randomized_prompt,
    vectorize_message_fields,
)
from neuroloc.simulations.memory.compression_under_bit_budget_mirror import build_factor_heldout_distributed_dataset, profile_caps


def test_language_grounded_prompt_exposes_message_and_answer_shape() -> None:
    dataset = build_factor_heldout_distributed_dataset("smoke", seed=73, train_episodes=2, val_episodes=1, test_episodes=1)
    record = dataset[0]
    prompt = record_prompt(record)
    response = answer_text(record)
    events, focus = parse_prompt(prompt)
    assert prompt.startswith("observations ")
    assert " question action_for slot_" in prompt
    assert response.startswith("answer action_")
    assert events
    assert focus == int(record["model_input"]["query"]["focus_local_index"])


def test_language_grounded_vectorizer_ignores_label_contract_shortcuts() -> None:
    dataset = build_factor_heldout_distributed_dataset("smoke", seed=73, train_episodes=2, val_episodes=1, test_episodes=1)
    caps = profile_caps("smoke")
    record = dataset[0]
    fields = vectorize_message_fields(record_prompt(record), caps)
    assert set(fields) == {"color", "shape", "pos", "vel", "action", "provenance"}
    assert all(value.size > 0 for value in fields.values())


def test_language_grounded_state_density_clears_constrained_message_gate() -> None:
    summary = build_summary("smoke", seed=81)
    assert summary["language_grounded_local_model_authorized"] == 1.0
    assert summary["language_grounded_full_model_authorized"] == 0.0
    assert summary["language_grounded_paid_compute_authorized"] == 0.0
    assert summary["language_grounded_arbitrary_chat_authorized"] == 0.0
    assert summary["language_grounded_constrained_message_response_supported"] == 1.0
    assert summary["language_grounded_axis_count"] == 4
    assert summary["language_grounded_seed_count"] == 2
    assert summary["language_grounded_run_count"] == 8
    assert summary["language_grounded_total_train_record_count"] == 16384
    assert summary["language_grounded_total_validation_record_count"] == 768
    assert summary["language_grounded_total_test_record_count"] == 768
    assert summary["language_grounded_parameter_count_max"] < 10000
    assert summary["language_grounded_test_joint_success_min"] >= 0.95
    assert summary["language_grounded_test_state_success_min"] >= 0.95
    assert summary["language_grounded_test_action_success_min"] >= 0.95
    assert summary["language_grounded_field_accuracy_floor"] >= 0.95
    assert summary["language_grounded_matched_sparse_joint_success_max"] == 0.0
    assert summary["language_grounded_matched_sparse_bits_min"] == 20.0
    assert summary["language_grounded_useful_operation_success_per_committed_bit_min"] > summary["language_grounded_matched_sparse_operation_success_per_committed_bit_max"]
    assert summary["language_grounded_useful_state_density_advantage_min"] > 0.0
    assert summary["language_grounded_engineering_pass"] == 1.0
    assert str(summary["language_grounded_example_prompt"]).startswith("observations ")
    assert str(summary["language_grounded_example_response"]).startswith("answer action_")


def test_parser_resistant_prompt_removes_stable_prefix_dependency() -> None:
    dataset = build_factor_heldout_distributed_dataset("smoke", seed=73, train_episodes=2, val_episodes=1, test_episodes=1)
    prompt_a = randomized_record_prompt(dataset[0], seed=11)
    prompt_b = randomized_record_prompt(dataset[0], seed=11)
    prompt_c = randomized_record_prompt(dataset[0], seed=12)
    tokens = tokenize_randomized_prompt(prompt_a)
    assert prompt_a == prompt_b
    assert prompt_a != prompt_c
    assert tokens
    assert "time_" not in prompt_a
    assert "slot_" not in prompt_a
    assert "color_" not in prompt_a
    assert "shape_" not in prompt_a
    assert "pos_" not in prompt_a
    assert "question action_for" not in prompt_a


def test_parser_resistant_local_state_gate_reports_controls() -> None:
    summary = build_parser_resistant_summary("smoke", seed=91)
    assert summary["parser_resistant_gate_evaluated"] == 1.0
    assert summary["parser_resistant_local_model_authorized"] == 1.0
    assert summary["parser_resistant_full_model_authorized"] == 0.0
    assert summary["parser_resistant_paid_compute_authorized"] == 0.0
    assert summary["parser_resistant_arbitrary_chat_authorized"] == 0.0
    assert summary["parser_resistant_template_family_count"] >= 3
    assert summary["parser_resistant_prefix_dependency_removed"] == 1.0
    assert summary["parser_resistant_deterministic_parser_reported"] == 1.0
    assert summary["parser_resistant_learned_text_encoder_reported"] == 1.0
    assert summary["parser_resistant_local_state_ablation_reported"] == 1.0
    assert summary["parser_resistant_test_joint_success_min"] == 0.0
    assert summary["parser_resistant_test_state_success_min"] == 0.0
    assert summary["parser_resistant_matched_sparse_joint_success_max"] == 0.0
    assert summary["parser_resistant_uncapped_sparse_joint_success_min"] >= 0.95
    assert summary["parser_resistant_state_shuffle_joint_success_max"] <= summary["parser_resistant_test_joint_success_min"]
    assert summary["parser_resistant_zero_state_joint_success_max"] <= summary["parser_resistant_test_joint_success_min"]
    assert summary["parser_resistant_learned_committed_bits_max"] == 19.0
    assert summary["parser_resistant_parser_schema_cost_bits"] == 37.0
    assert summary["parser_resistant_engineering_pass"] == 0.0
    assert summary["parser_resistant_claim_downgraded_to_structured_bridge"] == 1.0
