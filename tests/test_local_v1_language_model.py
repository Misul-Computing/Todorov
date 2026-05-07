from neuroloc.simulations.memory.local_v1_language_model import (
    answer_v1_prompt,
    build_v1_dialogue_summary,
    build_v1_language_summary,
    run_v1_dialogue,
    train_v1_language_state_model,
)


SOURCE_TEXTS = [
    "cortex binds memory for action.",
    "synapse stores update signals for future recall.",
    "replay refreshes provenance after distraction.",
    "imagination branches physics into counterfactual state.",
    "compression keeps useful payload fields under budget.",
    "language queries route into bounded memory state.",
    "occlusion hides objects while local state preserves position.",
    "dreaming rehearses branches without external observation.",
    "attention selects records while memory preserves payload.",
    "neuron state carries eligibility before commitment.",
    "world dynamics update velocity after action.",
    "provenance names the record that supported the answer.",
    "decoder reconstructs fields from compact state.",
    "gate opens when surprise exceeds local expectation.",
    "schema separates address payload action and source.",
    "dataset teaches routing without next token training.",
]


def test_v1_state_first_dataset_training_clears_local_language_gate():
    summary = build_v1_language_summary("smoke", seed=17, source_texts=SOURCE_TEXTS)
    assert summary["v1_dataset_grounded"] == 1.0
    assert summary["v1_state_first_training_used"] == 1.0
    assert summary["v1_next_token_training_used"] == 0.0
    assert summary["v1_local_model_authorized"] == 1.0
    assert summary["v1_full_model_authorized"] == 0.0
    assert summary["v1_paid_compute_authorized"] == 0.0
    assert summary["v1_arbitrary_chat_authorized"] == 0.0
    assert summary["v1_trainable_parameter_count_max"] < 10_000
    assert summary["v1_test_joint_success_min"] >= 0.9
    assert summary["v1_test_state_success_min"] >= 0.9
    assert summary["v1_test_action_success_min"] >= 0.9
    assert summary["v1_provenance_success_min"] >= 0.9
    assert summary["v1_zero_state_joint_success_max"] == 0.0
    assert summary["v1_shuffled_state_joint_success_max"] < 0.4
    assert summary["v1_useful_operation_success_per_accounted_bit_min"] > 0.0
    assert summary["v1_interactive_response_supported"] == 1.0
    assert summary["v1_engineering_pass"] == 1.0


def test_v1_interactive_response_uses_dataset_memory_state():
    bundle = train_v1_language_state_model("smoke", seed=23, source_texts=SOURCE_TEXTS)
    response = answer_v1_prompt("what does cortex preserve for action?", bundle)
    zero_response = answer_v1_prompt("what does cortex preserve for action?", bundle, state_mode="zero")
    shuffled_response = answer_v1_prompt("what does cortex preserve for action?", bundle, state_mode="shuffled")
    unknown_response = answer_v1_prompt("compose a poem about oceans and castles", bundle)
    assert response.startswith("v1 answer:")
    assert "cortex" in response
    assert "memory" in response
    assert "source record_" in response
    assert zero_response == "v1 answer: no grounded memory state"
    assert shuffled_response != response
    assert unknown_response == "v1 answer: outside grounded memory scope"


def test_v1_dialogue_update_replay_and_branch_state_controls():
    bundle = train_v1_language_state_model("smoke", seed=29, source_texts=SOURCE_TEXTS)
    dialogue = run_v1_dialogue(
        [
            "what does cortex preserve for action?",
            "update cortex payload planning from study note",
            "tell me about synapse",
            "replay cortex",
            "imagine cortex branch rehearsal",
            "what does cortex preserve for action?",
        ],
        bundle,
    )
    zero_dialogue = run_v1_dialogue(
        [
            "update cortex payload planning from study note",
            "replay cortex",
            "imagine cortex branch rehearsal",
            "what does cortex preserve for action?",
        ],
        bundle,
        state_mode="zero",
    )
    random_dialogue = run_v1_dialogue(
        [
            "update cortex payload planning from study note",
            "replay cortex",
            "imagine cortex branch rehearsal",
            "what does cortex preserve for action?",
        ],
        bundle,
        replay_mode="random",
    )
    assert dialogue["update_success"] == 1.0
    assert dialogue["targeted_replay_success"] == 1.0
    assert dialogue["branch_state_success"] == 1.0
    assert dialogue["final_joint_success"] == 1.0
    assert "planning" in dialogue["final_response"]
    assert "branch rehearsal" in dialogue["branch_response"]
    assert zero_dialogue["final_joint_success"] == 0.0
    assert random_dialogue["targeted_replay_success"] == 0.0


def test_v1_dialogue_summary_reports_memory_update_replay_and_branch_gates():
    summary = build_v1_dialogue_summary("smoke", seed=31, source_texts=SOURCE_TEXTS)
    assert summary["v1_dialogue_gate_evaluated"] == 1.0
    assert summary["v1_memory_update_success_min"] >= 0.9
    assert summary["v1_targeted_replay_success_min"] >= 0.9
    assert summary["v1_random_replay_success_max"] == 0.0
    assert summary["v1_branch_state_success_min"] >= 0.9
    assert summary["v1_dialogue_final_joint_success_min"] >= 0.9
    assert summary["v1_dialogue_zero_state_joint_success_max"] == 0.0
    assert summary["v1_dialogue_engineering_pass"] == 1.0
