from neuroloc.simulations.memory.local_foundation_neural_model import (
    build_foundation_neural_model_summary,
    control_bundle,
    foundation_chat_once,
    run_architecture_script,
    train_foundation_neural_model,
)


SOURCE_TEXTS = [
    "cortex binds memory for action.",
    "synapse stores update signals for future recall.",
    "replay refreshes provenance after distraction.",
    "imagination branches physics into counterfactual state.",
    "compression keeps useful payload fields under budget.",
    "language routes questions into bounded memory state.",
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
    "context keeps short term dialogue state.",
    "consolidation moves useful traces into long term memory.",
]


def test_foundation_neural_model_exposes_memory_object_architecture_under_budget():
    bundle = train_foundation_neural_model("smoke", seed=11, source_texts=SOURCE_TEXTS)
    assert bundle["parameter_count"] < 10_000
    assert bundle["training_method"] == "state_first_memory_object_binding"
    assert bundle["codec_name"] == "schema_residual_identity_v0"
    assert bundle["next_token_training_used"] is False
    assert bundle["full_model_authorized"] is False
    assert bundle["paid_compute_authorized"] is False
    assert bundle["arbitrary_chat_authorized"] is False
    code = bundle["long_term_memory"][0]
    assert set(["address", "schema", "payload", "action", "source", "vector"]).issubset(code)
    assert len(bundle["episodic_memory"]) == len(bundle["long_term_memory"])


def test_foundation_neural_model_answers_updates_replays_branches_and_actions():
    bundle = train_foundation_neural_model("smoke", seed=13, source_texts=SOURCE_TEXTS)
    transcript = run_architecture_script(bundle)
    assert transcript["grounded"] == 1.0
    assert transcript["action"] == 1.0
    assert transcript["update"] == 1.0
    assert transcript["context"] == 1.0
    assert transcript["replay"] == 1.0
    assert transcript["branch"] == 1.0
    assert transcript["final"] == 1.0
    assert transcript["working"] == 1.0
    assert transcript["episodic"] == 1.0
    assert transcript["long_term"] == 1.0
    assert transcript["replay_buffer"] == 1.0
    assert transcript["branch_memory"] == 1.0


def test_foundation_neural_model_controls_force_state_and_codec_dependence():
    bundle = train_foundation_neural_model("smoke", seed=17, source_texts=SOURCE_TEXTS)
    normal = foundation_chat_once("what does cortex preserve", bundle)
    disabled_route = foundation_chat_once("what does cortex preserve", control_bundle(bundle, "disabled_route"))
    disabled_codec = foundation_chat_once("what does cortex preserve", control_bundle(bundle, "disabled_codec"))
    disabled_replay = foundation_chat_once("replay cortex", control_bundle(bundle, "disabled_replay_buffer"))
    disabled_branch = foundation_chat_once("imagine cortex branch rehearsal", control_bundle(bundle, "disabled_branch_memory"))
    unknown = foundation_chat_once("write a sonnet about rain", bundle)
    assert normal["accepted"] is True
    assert "cortex" in normal["text"]
    assert disabled_route["accepted"] is False
    assert disabled_codec["accepted"] is False
    assert disabled_replay["accepted"] is False
    assert disabled_branch["accepted"] is False
    assert unknown["accepted"] is False


def test_foundation_neural_model_summary_reports_full_foundation_gate():
    summary = build_foundation_neural_model_summary("smoke", seed=19, source_texts=SOURCE_TEXTS)
    assert summary["foundation_nm_gate_evaluated"] == 1.0
    assert summary["foundation_nm_engineering_pass"] == 1.0
    assert summary["foundation_nm_parameter_count_max"] < 10_000
    assert summary["foundation_nm_codec_boundary_used"] == 1.0
    assert summary["foundation_nm_simple_codec_used"] == 1.0
    assert summary["foundation_nm_extreme_compression_claimed"] == 0.0
    assert summary["foundation_nm_grounded_response_success_min"] >= 0.9
    assert summary["foundation_nm_unknown_refusal_success_min"] == 1.0
    assert summary["foundation_nm_route_disabled_grounded_success_max"] <= 0.1
    assert summary["foundation_nm_route_shuffled_grounded_success_max"] <= 0.4
    assert summary["foundation_nm_codec_disabled_grounded_success_max"] == 0.0
    assert summary["foundation_nm_replay_disabled_success_max"] == 0.0
    assert summary["foundation_nm_branch_disabled_success_max"] == 0.0
    assert summary["foundation_nm_action_success_min"] >= 0.9
    assert summary["foundation_nm_memory_update_success_min"] >= 0.9
    assert summary["foundation_nm_context_success_min"] >= 0.9
    assert summary["foundation_nm_targeted_replay_success_min"] >= 0.9
    assert summary["foundation_nm_branch_state_success_min"] >= 0.9
    assert summary["foundation_nm_final_dialogue_joint_success_min"] >= 0.9
    assert summary["foundation_nm_working_memory_used_min"] >= 0.9
    assert summary["foundation_nm_episodic_memory_used_min"] >= 0.9
    assert summary["foundation_nm_long_term_memory_used_min"] >= 0.9
    assert summary["foundation_nm_replay_buffer_used_min"] >= 0.9
    assert summary["foundation_nm_branch_memory_used_min"] >= 0.9
