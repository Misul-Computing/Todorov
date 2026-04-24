import pytest

from neuroloc.data.nm_worlds import (
    HARD_SYMBOLIC_FAMILIES,
    HARD_SYMBOLIC_POLICIES,
    evaluate_nm_hard_policy,
    evaluate_nm_hard_symbolic_episode,
    generate_nm_hard_symbolic_batch,
    generate_nm_hard_symbolic_episode,
)


def test_hard_symbolic_episode_is_deterministic() -> None:
    first = generate_nm_hard_symbolic_episode(seed=101, profile="smoke")
    second = generate_nm_hard_symbolic_episode(seed=101, profile="smoke")
    assert first["profile"] == second["profile"]
    assert first["contracts"] == second["contracts"]
    assert first["hidden_state"]["active_ids"].tolist() == second["hidden_state"]["active_ids"].tolist()


def test_hard_symbolic_contract_covers_all_required_families_and_policies() -> None:
    episode = generate_nm_hard_symbolic_episode(seed=102, profile="smoke")
    families = {contract["family"] for contract in episode["contracts"]}
    assert families == set(HARD_SYMBOLIC_FAMILIES)
    for contract in episode["contracts"]:
        assert set(contract) >= {
            "family",
            "query",
            "target",
            "memory_relevant_positions",
            "distractor_positions",
            "difficulty",
            "bit_budget",
            "expected",
            "telemetry",
        }
        assert set(contract["expected"]) == set(HARD_SYMBOLIC_POLICIES)
        assert {
            "oracle_write_learned_read",
            "learned_write_oracle_read",
            "hand_opened_gate",
            "orthogonal_address_init",
            "matched_compute_budget",
        }.issubset(set(contract["expected"]))
        assert contract["memory_relevant_positions"]
        assert contract["distractor_positions"]
        assert contract["query"]["target_answer_visible"] is False


def test_hard_symbolic_query_observation_does_not_leak_target_answer() -> None:
    episode = generate_nm_hard_symbolic_episode(seed=103, profile="smoke")
    observations = episode["observation_stream"]
    hidden = episode["hidden_state"]
    for contract in episode["contracts"]:
        query = contract["query"]
        target = contract["target"]["state"]
        time_idx = query["time"]
        object_idx = query["focus_local_index"]
        assert observations["color"][time_idx, object_idx] != target["color"]
        assert observations["shape"][time_idx, object_idx] != target["shape"]
        assert observations["pos"][time_idx, object_idx] != hidden["positions"][time_idx, object_idx]


def test_hard_symbolic_observations_match_hidden_identity_attributes() -> None:
    for seed in range(100, 112):
        episode = generate_nm_hard_symbolic_episode(seed=seed, profile="smoke")
        observations = episode["observation_stream"]
        hidden = episode["hidden_state"]
        active_ids = hidden["active_ids"]
        identity_bank = hidden["identity_bank"]
        for local_index, identity_id in enumerate(active_ids.tolist()):
            color_value = int(identity_bank["color"][int(identity_id)])
            shape_value = int(identity_bank["shape"][int(identity_id)])
            observed_colors = observations["color"][:, local_index]
            observed_shapes = observations["shape"][:, local_index]
            assert all(int(value) == color_value for value in observed_colors.tolist() if int(value) >= 0)
            assert all(int(value) == shape_value for value in observed_shapes.tolist() if int(value) >= 0)


def test_hard_symbolic_controls_have_expected_behavior() -> None:
    episode = generate_nm_hard_symbolic_episode(seed=104, profile="smoke")
    rows = evaluate_nm_hard_symbolic_episode(episode)
    oracle_rows = [row for row in rows if row["policy"] == "oracle"]
    no_memory_rows = [row for row in rows if row["policy"] == "no_memory"]
    recency_rows = [row for row in rows if row["policy"] == "recency_only"]
    shuffled_rows = [row for row in rows if row["policy"] == "shuffled_address"]
    assert min(row["joint_correct"] for row in oracle_rows) >= 0.98
    assert max(row["joint_correct"] for row in no_memory_rows) <= 0.0
    assert max(row["joint_correct"] for row in recency_rows) <= 0.0
    assert max(row["joint_correct"] for row in shuffled_rows) <= 0.0


def test_hard_symbolic_replay_and_compression_gates_are_separated() -> None:
    episode = generate_nm_hard_symbolic_episode(seed=105, profile="smoke")
    contracts = {contract["family"]: contract for contract in episode["contracts"]}
    replay = contracts["replay_rewrite"]
    random_replay = evaluate_nm_hard_policy(replay, "random_replay")
    targeted_replay = evaluate_nm_hard_policy(replay, "targeted_replay")
    assert targeted_replay["joint_correct"] > random_replay["joint_correct"]
    compression = contracts["compression_under_bit_budget"]
    verbatim = evaluate_nm_hard_policy(compression, "verbatim_store")
    compressed = evaluate_nm_hard_policy(compression, "compressed_store")
    assert compressed["joint_correct"] == verbatim["joint_correct"]
    assert compressed["bits_written"] < verbatim["bits_written"]
    assert compressed["within_budget"] == 1.0
    assert verbatim["within_budget"] == 0.0


def test_hard_symbolic_interference_and_context_are_instantiated() -> None:
    episode = generate_nm_hard_symbolic_episode(seed=108, profile="smoke")
    contracts = {contract["family"]: contract for contract in episode["contracts"]}
    interference = contracts["correlated_key_interference"]
    assert "interference_distractor_local_index" in interference["query"]
    assert "interference_distractor_identity" in interference["target"]
    assert interference["distractor_positions"][0]["shared_attribute"] in {"color", "shape"}
    target_identity = interference["target"]["identity"]
    distractor_identity = interference["target"]["interference_distractor_identity"]
    shared_attribute = interference["distractor_positions"][0]["shared_attribute"]
    hidden_bank = episode["hidden_state"]["identity_bank"]
    assert hidden_bank[shared_attribute][target_identity] == hidden_bank[shared_attribute][distractor_identity]
    context = contracts["context_gated_routing"]
    assert "context_id" in context["query"]
    assert "cue_id" in context["query"]
    assert "context_action_map" in context["target"]
    assert context["target"]["action"] == context["target"]["context_action_map"][str(context["query"]["context_id"])]


def test_hard_symbolic_trainability_controls_localize_failure_modes() -> None:
    episode = generate_nm_hard_symbolic_episode(seed=109, profile="smoke")
    contracts = {contract["family"]: contract for contract in episode["contracts"]}
    associative = contracts["associative_recall"]
    interference = contracts["correlated_key_interference"]
    compression = contracts["compression_under_bit_budget"]
    assert evaluate_nm_hard_policy(associative, "oracle_write_learned_read")["joint_correct"] == 1.0
    assert evaluate_nm_hard_policy(associative, "learned_write_oracle_read")["joint_correct"] == 0.0
    assert evaluate_nm_hard_policy(interference, "orthogonal_address_init")["joint_correct"] == 1.0
    assert evaluate_nm_hard_policy(interference, "hand_opened_gate")["joint_correct"] == 0.0
    assert evaluate_nm_hard_policy(compression, "learned_write_oracle_read")["joint_correct"] == 1.0
    assert evaluate_nm_hard_policy(compression, "oracle_write_learned_read")["joint_correct"] == 0.0


def test_hard_symbolic_batch_respects_seed_and_profile() -> None:
    first = generate_nm_hard_symbolic_batch(3, seed=106, profile="hard")
    second = generate_nm_hard_symbolic_batch(3, seed=106, profile="hard")
    assert [episode["seed"] for episode in first] == [episode["seed"] for episode in second]
    assert all(episode["profile"] == "hard" for episode in first)


def test_hard_symbolic_invalid_parameters_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown hard symbolic profile"):
        generate_nm_hard_symbolic_episode(seed=107, profile="bad")
    with pytest.raises(ValueError, match="n_episodes"):
        generate_nm_hard_symbolic_batch(0, seed=107, profile="smoke")
