import inspect

import torch

from neuroloc.data.nm_3d_worlds import build_dataset, profile_caps, split_records
from neuroloc.simulations.memory.local_100k_full_nm import (
    build_model,
    build_summary,
    binary_code_from_fields,
    field_sizes,
    train_full_nm,
    world_code_fields,
    world_feature_vector,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_full_nm_model_is_single_trainable_module() -> None:
    caps = profile_caps("smoke")
    record = split_records(build_dataset("smoke", 7, 4, 1, 2, "axis"), "train")[0]
    model = build_model(len(world_feature_vector(record, caps)), caps, 32, 24, 32)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "encoder")
    assert hasattr(model, "state_cell")
    assert hasattr(model, "branch_trunk")
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_full_nm_bottleneck_is_smaller_than_explicit_3d_state() -> None:
    caps = profile_caps("hard")
    record = split_records(build_dataset("hard", 9, 4, 1, 2, "axis"), "train")[0]
    fields = world_code_fields(record, caps)
    code = binary_code_from_fields(fields, caps, 24)
    assert len(code) == 24
    assert 24 + 20 < 51
    assert sum(field_sizes(caps).values()) > 24


def test_full_nm_training_source_keeps_target_fields_out_of_forward_path() -> None:
    source = inspect.getsource(build_model)
    assert "world_code_fields" not in source
    assert "labels" not in source
    assert "hidden_state" not in source
    training_source = inspect.getsource(train_full_nm)
    assert "current_fields = [feature_fields_from_observation" in training_source


def test_local_100k_full_nm_smoke_gate_passes_single_axis() -> None:
    summary = build_summary("smoke", seed=211, train_episodes=4096, val_episodes=8, test_episodes=10, epochs=1000, hidden_width=112, state_width=80, code_bits=24, seed_count=1, axes=("identity_position_band",))
    assert summary["local_100k_full_nm_engineering_pass"] == 1.0
    assert summary["local_100k_full_nm_single_trainable_module"] == 1.0
    assert summary["local_100k_full_nm_local_full_candidate_authorized"] == 1.0
    assert summary["local_100k_full_nm_full_model_authorized"] == 0.0
    assert summary["local_100k_full_nm_parameter_count_max"] < 100000
    assert summary["local_100k_full_nm_learned_latent_state_bits_max"] < 51
    assert summary["local_100k_full_nm_accounted_bits_max"] <= 44
    assert summary["local_100k_full_nm_initial_world_state_joint_success_min"] >= 0.95
    assert summary["local_100k_full_nm_targeted_replay_success_min"] >= 0.95
    assert summary["local_100k_full_nm_rewrite_success_min"] >= 0.95
    assert summary["local_100k_full_nm_learned_branch_transition_success_min"] >= 0.95
    assert summary["local_100k_full_nm_bounded_language_answer_success_min"] >= 0.95
    assert summary["local_100k_full_nm_no_memory_success_max"] == 0.0
    assert summary["local_100k_full_nm_code_disabled_success_max"] == 0.0
    assert summary["local_100k_full_nm_shuffled_code_success_max"] == 0.0
    assert summary["local_100k_full_nm_no_replay_success_max"] == 0.0
    assert summary["local_100k_full_nm_random_replay_success_max"] == 0.0
    assert summary["local_100k_full_nm_no_branch_success_max"] == 0.0
    assert summary["local_100k_full_nm_wrong_branch_success_max"] == 0.0


def test_local_100k_full_nm_registry_entry() -> None:
    assert "local_100k_full_nm" in SIMULATION_SPECS
    assert "local_100k_full_nm" in SUITES["compression_mirror"]
    assert "local_100k_full_nm" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_full_nm"]
    assert spec.category == "compression_mirror"
    assert dict(spec.maximum_summary_values)["local_100k_full_nm_parameter_count_max"] == 99999.0
    assert dict(spec.maximum_summary_values)["local_100k_full_nm_accounted_bits_max"] == 44.0
    assert dict(spec.maximum_summary_values)["local_100k_full_nm_full_model_authorized"] == 0.0
    assert dict(spec.minimum_summary_values)["local_100k_full_nm_engineering_pass"] == 1.0
