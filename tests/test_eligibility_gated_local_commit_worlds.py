import pytest

from neuroloc.data.nm_worlds import (
    ELIGIBILITY_COMMIT_FAMILIES,
    ELIGIBILITY_COMMIT_POLICIES,
    evaluate_eligibility_gated_local_commit_episode,
    evaluate_eligibility_gated_local_commit_policy,
    generate_eligibility_gated_local_commit_batch,
    generate_eligibility_gated_local_commit_episode,
)


def test_eligibility_commit_episode_is_deterministic() -> None:
    first = generate_eligibility_gated_local_commit_episode(seed=201, profile="smoke")
    second = generate_eligibility_gated_local_commit_episode(seed=201, profile="smoke")
    assert first["profile"] == second["profile"]
    first_contracts = [
        {key: value for key, value in contract.items() if key not in {"hidden_state", "observation_stream"}}
        for contract in first["contracts"]
    ]
    second_contracts = [
        {key: value for key, value in contract.items() if key not in {"hidden_state", "observation_stream"}}
        for contract in second["contracts"]
    ]
    assert first_contracts == second_contracts
    assert first["hidden_state"]["active_ids"].tolist() == second["hidden_state"]["active_ids"].tolist()


def test_eligibility_commit_contract_fields_and_policies_are_complete() -> None:
    episode = generate_eligibility_gated_local_commit_episode(seed=202, profile="smoke")
    families = {contract["family"] for contract in episode["contracts"]}
    assert families == set(ELIGIBILITY_COMMIT_FAMILIES)
    required = {
        "episode_id",
        "seed",
        "family",
        "profile",
        "hidden_state",
        "observation_stream",
        "query",
        "target",
        "candidate_events",
        "relevance_events",
        "commit_targets",
        "read_queries",
        "exposure_targets",
        "memory_relevant_positions",
        "distractor_positions",
        "negative_commit_positions",
        "trace_eligible_positions",
        "commit_positions",
        "exposure_positions",
        "difficulty",
        "bit_budget",
        "output_budget",
        "oracle_codes",
        "expected",
        "telemetry",
        "leakage_checks",
        "kill_conditions",
    }
    phase_keys = {
        "mark_correct",
        "commit_correct",
        "read_correct",
        "exposure_correct",
        "state_correct",
        "action_correct",
        "joint_correct",
        "bits_committed",
        "within_commit_budget",
        "within_exposure_budget",
    }
    for contract in episode["contracts"]:
        assert set(contract) >= required
        assert set(contract["expected"]) == set(ELIGIBILITY_COMMIT_POLICIES)
        for policy in ELIGIBILITY_COMMIT_POLICIES:
            assert set(contract["expected"][policy]) >= phase_keys
        assert contract["candidate_events"]
        assert contract["relevance_events"]
        assert contract["commit_targets"]
        assert contract["read_queries"]
        assert contract["exposure_targets"]


def test_eligibility_commit_query_and_relevance_do_not_leak_answer() -> None:
    episode = generate_eligibility_gated_local_commit_episode(seed=203, profile="smoke")
    observations = episode["observation_stream"]
    for contract in episode["contracts"]:
        query = contract["query"]
        target = contract["target"]["state"]
        time_idx = query["time"]
        object_idx = query["focus_local_index"]
        assert observations["color"][time_idx, object_idx] != target["color"]
        assert observations["shape"][time_idx, object_idx] != target["shape"]
        assert observations["pos"][time_idx, object_idx] != target["pos"]
        assert query["target_answer_visible"] is False
        assert query["target_identity_visible"] is False
        assert query["target_action_visible"] is False
        for relevance in contract["relevance_events"]:
            assert relevance["names_answer"] is False
            assert relevance["names_target_identity"] is False
            assert relevance["names_unique_candidate_index"] is False
        assert not any(contract["leakage_checks"].values())


def test_eligibility_commit_candidate_relevance_commit_read_exposure_splits_exist() -> None:
    episode = generate_eligibility_gated_local_commit_episode(seed=204, profile="hard")
    for contract in episode["contracts"]:
        target_candidate = contract["target"]["candidate_id"]
        target_state = contract["target"]["state"]
        observations = contract["observation_stream"]
        relevant_candidates = [
            candidate["candidate_id"]
            for candidate in contract["candidate_events"]
            if candidate["eventually_relevant"]
        ]
        negated = set(contract["relevance_events"][0]["negates_candidate_ids"])
        exposed = set(contract["exposure_targets"][0]["should_expose_commit_ids"])
        suppressed = set(contract["exposure_targets"][0]["must_not_expose_commit_ids"])
        assert relevant_candidates == [target_candidate]
        assert target_candidate not in negated
        assert target_candidate in exposed
        assert target_candidate not in suppressed
        assert len(contract["candidate_events"]) > contract["output_budget"]
        assert contract["commit_targets"][0]["forbidden_before_time"] == contract["relevance_events"][0]["time"]
        assert contract["read_queries"][0]["required_commit_id"] == target_candidate
        assert contract["commit_targets"][0]["source_time"] == contract["candidate_events"][target_candidate]["time"]
        assert contract["memory_relevant_positions"][0]["time"] == contract["candidate_events"][target_candidate]["time"]
        target_time = [
            candidate["time"]
            for candidate in contract["candidate_events"]
            if candidate["candidate_id"] == target_candidate
        ][0]
        distractor_times = [
            candidate["time"]
            for candidate in contract["candidate_events"]
            if candidate["candidate_id"] != target_candidate
        ]
        assert max(distractor_times) > target_time
        assert min(distractor_times) < target_time
        all_times = [candidate["time"] for candidate in contract["candidate_events"]]
        assert all_times.count(target_time) >= 2
        shared_distractors = [
            candidate
            for candidate in contract["candidate_events"]
            if candidate["candidate_id"] != target_candidate
            and (
                candidate["candidate_payload"]["color"] == target_state["color"]
                or candidate["candidate_payload"]["shape"] == target_state["shape"]
            )
        ]
        assert shared_distractors
        assert any(
            observations["color"][candidate["time"], candidate["object_index"]] == target_state["color"]
            for candidate in shared_distractors
        )


def test_eligibility_commit_recency_and_surface_distractors_hold_across_seeds() -> None:
    target_times = []
    for seed in range(240, 260):
        episode = generate_eligibility_gated_local_commit_episode(seed=seed, profile="smoke")
        for contract in episode["contracts"]:
            target_candidate = contract["target"]["candidate_id"]
            target_state = contract["target"]["state"]
            candidate_by_id = {
                candidate["candidate_id"]: candidate
                for candidate in contract["candidate_events"]
            }
            target_time = candidate_by_id[target_candidate]["time"]
            target_times.append(target_time)
            observations = contract["observation_stream"]
            assert any(
                candidate["time"] > target_time
                for candidate in contract["candidate_events"]
                if candidate["candidate_id"] != target_candidate
            )
            assert any(
                candidate["time"] < target_time
                for candidate in contract["candidate_events"]
                if candidate["candidate_id"] != target_candidate
            )
            assert sum(
                int(candidate["time"] == target_time)
                for candidate in contract["candidate_events"]
            ) >= 2
            assert any(
                observations["color"][candidate["time"], candidate["object_index"]] == target_state["color"]
                or observations["shape"][candidate["time"], candidate["object_index"]] == target_state["shape"]
                for candidate in contract["candidate_events"]
                if candidate["candidate_id"] != target_candidate
            )
    assert len(set(target_times)) > 1


def test_eligibility_commit_bounded_exposure_has_committed_distractors() -> None:
    episode = generate_eligibility_gated_local_commit_episode(seed=205, profile="smoke")
    for family in {"bounded_output_exposure", "crossed_commit_exposure_split"}:
        contract = next(contract for contract in episode["contracts"] if contract["family"] == family)
        target_candidate = contract["target"]["candidate_id"]
        competitor_ids = set(contract["exposure_targets"][0]["must_not_expose_commit_ids"])
        committed_ids = {
            commit["candidate_id"]
            for commit in contract["commit_targets"]
            if commit["should_commit"]
        }
        assert target_candidate in committed_ids
        assert competitor_ids
        assert competitor_ids <= committed_ids
        assert len(contract["commit_positions"]) > contract["output_budget"]


def test_eligibility_commit_controls_have_expected_behavior() -> None:
    episode = generate_eligibility_gated_local_commit_episode(seed=206, profile="smoke")
    rows = evaluate_eligibility_gated_local_commit_episode(episode)
    oracle_rows = [row for row in rows if row["policy"] == "oracle"]
    no_memory_rows = [row for row in rows if row["policy"] == "no_memory"]
    recency_rows = [row for row in rows if row["policy"] == "recency_only"]
    shuffled_rows = [row for row in rows if row["policy"] == "shuffled_address"]
    random_trace_rows = [row for row in rows if row["policy"] == "random_trace"]
    no_trace_rows = [row for row in rows if row["policy"] == "no_trace"]
    oracle_commit_oracle_exposure_rows = [row for row in rows if row["policy"] == "oracle_commit_oracle_exposure"]
    oracle_mark_no_commit_rows = [row for row in rows if row["policy"] == "oracle_mark_no_commit"]
    no_commit_oracle_exposure_rows = [row for row in rows if row["policy"] == "no_commit_oracle_exposure"]
    hand_opened_rows = [row for row in rows if row["policy"] == "hand_opened_exposure"]
    residual_rows = [row for row in rows if row["policy"] == "matched_residual_capacity"]
    compute_rows = [row for row in rows if row["policy"] == "matched_compute_budget"]
    assert min(row["joint_correct"] for row in oracle_rows) >= 0.98
    assert min(row["joint_correct"] for row in oracle_commit_oracle_exposure_rows) >= 0.98
    assert max(row["joint_correct"] for row in no_memory_rows) <= 0.0
    assert max(row["joint_correct"] for row in recency_rows) <= 0.0
    assert max(row["joint_correct"] for row in shuffled_rows) <= 0.0
    assert max(row["commit_f1"] for row in random_trace_rows) <= 0.25
    assert max(row["joint_correct"] for row in no_trace_rows) <= 0.0
    assert max(row["joint_correct"] for row in oracle_mark_no_commit_rows) <= 0.0
    assert max(row["joint_correct"] for row in no_commit_oracle_exposure_rows) <= 0.0
    assert max(row["joint_correct"] for row in hand_opened_rows) <= 0.0
    assert max(row["joint_correct"] for row in residual_rows) <= 0.0
    assert max(row["joint_correct"] for row in compute_rows) <= 0.0


def test_eligibility_commit_output_exposure_and_compression_controls_are_separated() -> None:
    episode = generate_eligibility_gated_local_commit_episode(seed=207, profile="smoke")
    contracts = {contract["family"]: contract for contract in episode["contracts"]}
    bounded = contracts["bounded_output_exposure"]
    fixed_closed = evaluate_eligibility_gated_local_commit_policy(bounded, "fixed_closed_exposure")
    fixed_open = evaluate_eligibility_gated_local_commit_policy(bounded, "fixed_open_exposure")
    oracle = evaluate_eligibility_gated_local_commit_policy(bounded, "oracle")
    assert oracle["joint_correct"] == 1.0
    assert fixed_closed["state_correct"] == 1.0
    assert fixed_closed["action_correct"] == 0.0
    assert fixed_open["within_exposure_budget"] == 0.0
    assert fixed_open["joint_correct"] == 0.0
    compression = contracts["commit_compression_frontier"]
    always_unlimited = evaluate_eligibility_gated_local_commit_policy(compression, "always_commit_unlimited")
    always_matched = evaluate_eligibility_gated_local_commit_policy(compression, "always_commit_matched_budget")
    oracle_compressed = evaluate_eligibility_gated_local_commit_policy(compression, "oracle")
    assert always_unlimited["joint_correct"] == 1.0
    assert always_unlimited["within_commit_budget"] == 0.0
    assert always_matched["joint_correct"] == 0.0
    assert oracle_compressed["joint_correct"] == 1.0
    assert oracle_compressed["bits_committed"] < always_unlimited["bits_committed"]
    assert oracle_compressed["bits_committed"] == (
        compression["bit_budget"]["eligibility_commit_bits"]
        * len([commit for commit in compression["commit_targets"] if commit["counts_toward_commit_budget"]])
    )


def test_eligibility_commit_batch_respects_seed_and_profile() -> None:
    first = generate_eligibility_gated_local_commit_batch(3, seed=208, profile="hard")
    second = generate_eligibility_gated_local_commit_batch(3, seed=208, profile="hard")
    assert [episode["seed"] for episode in first] == [episode["seed"] for episode in second]
    assert all(episode["profile"] == "hard" for episode in first)


def test_eligibility_commit_invalid_parameters_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown eligibility commit profile"):
        generate_eligibility_gated_local_commit_episode(seed=209, profile="bad")
    with pytest.raises(ValueError, match="n_episodes"):
        generate_eligibility_gated_local_commit_batch(0, seed=209, profile="smoke")
