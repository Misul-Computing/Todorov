from neuroloc.simulations.memory.local_10k_chat_model import (
    build_local_10k_chat_summary,
    chat_once,
    run_chat_script,
    train_local_10k_chat_model,
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


def test_local_10k_chat_model_trains_state_first_under_parameter_budget():
    bundle = train_local_10k_chat_model("smoke", seed=41, source_texts=SOURCE_TEXTS)
    assert bundle["parameter_count"] < 10_000
    assert bundle["state_dim"] >= 8
    assert bundle["record_count"] >= 16
    assert bundle["training_method"] == "state_first_local_binding"
    assert bundle["next_token_training_used"] is False
    assert bundle["full_model_authorized"] is False
    assert bundle["paid_compute_authorized"] is False
    assert bundle["arbitrary_chat_authorized"] is False


def test_local_10k_chat_model_supports_grounded_chat_and_refuses_unknowns():
    bundle = train_local_10k_chat_model("smoke", seed=43, source_texts=SOURCE_TEXTS)
    grounded = chat_once("what does cortex preserve for action?", bundle)
    unknown = chat_once("write a pirate poem about thunder", bundle)
    assert grounded["accepted"] is True
    assert "cortex" in grounded["text"]
    assert "memory" in grounded["text"]
    assert "source record_" in grounded["text"]
    assert unknown["accepted"] is False
    assert unknown["text"] == "v1 chat: outside grounded memory scope"


def test_local_10k_chat_model_runs_memory_update_replay_branch_and_context():
    bundle = train_local_10k_chat_model("smoke", seed=47, source_texts=SOURCE_TEXTS)
    transcript = run_chat_script(
        [
            "what does cortex preserve for action?",
            "update cortex payload planning from study note",
            "what changed?",
            "tell me about synapse",
            "replay cortex",
            "imagine cortex branch rehearsal",
            "what does cortex preserve for action?",
        ],
        bundle,
    )
    assert transcript["update_success"] == 1.0
    assert transcript["short_term_context_success"] == 1.0
    assert transcript["targeted_replay_success"] == 1.0
    assert transcript["branch_state_success"] == 1.0
    assert transcript["final_joint_success"] == 1.0
    assert "planning" in transcript["responses"][-1]["text"]
    assert "branch rehearsal" in transcript["branch_response"]


def test_local_10k_chat_summary_reports_integrated_v1_controls():
    summary = build_local_10k_chat_summary("smoke", seed=53, source_texts=SOURCE_TEXTS)
    assert summary["local_10k_chat_gate_evaluated"] == 1.0
    assert summary["local_10k_chat_engineering_pass"] == 1.0
    assert summary["local_10k_chat_parameter_count_max"] < 10_000
    assert summary["local_10k_chat_grounded_response_success_min"] >= 0.9
    assert summary["local_10k_chat_unknown_refusal_success_min"] == 1.0
    assert summary["local_10k_chat_route_disabled_grounded_success_max"] <= 0.1
    assert summary["local_10k_chat_route_shuffled_grounded_success_max"] <= 0.4
    assert summary["local_10k_chat_memory_update_success_min"] >= 0.9
    assert summary["local_10k_chat_targeted_replay_success_min"] >= 0.9
    assert summary["local_10k_chat_branch_state_success_min"] >= 0.9
    assert summary["local_10k_chat_short_term_context_success_min"] >= 0.9
    assert summary["local_10k_chat_next_token_training_used"] == 0.0
    assert summary["local_10k_chat_paid_compute_authorized"] == 0.0
