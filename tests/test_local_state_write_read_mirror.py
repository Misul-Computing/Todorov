from neuroloc.simulations.memory.local_state_write_read_mirror import (
    build_summary,
    field_sizes,
    field_vector,
    update_fields,
)
from neuroloc.simulations.memory.compression_under_bit_budget_mirror import profile_caps
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_local_state_field_code_is_fixed_width() -> None:
    caps = profile_caps("smoke")
    sizes = field_sizes(caps)
    fields = {"address": 3, "schema": 2, "residual": 5, "action": 1, "provenance": 4}
    vector = field_vector(fields, caps)
    assert set(sizes) == {"address", "schema", "residual", "action", "provenance"}
    assert int(vector.sum()) == 5
    assert int(vector.shape[0]) == sum(sizes.values())


def test_local_state_update_changes_every_compact_field() -> None:
    caps = profile_caps("smoke")
    fields = {"address": 3, "schema": 2, "residual": 5, "action": 1, "provenance": 4}
    updated = update_fields(fields, caps, 2)
    assert set(updated) == set(fields)
    assert any(updated[key] != fields[key] for key in fields)
    assert all(0 <= int(updated[key]) < int(field_sizes(caps)[key]) for key in updated)


def test_local_state_write_read_gate_passes_single_axis_smoke() -> None:
    summary = build_summary("smoke", seed=125, train_episodes=256, val_episodes=8, test_episodes=16, epochs=100, state_width=64, seed_count=1, axes=("color_shape_pair_band",))
    assert summary["local_state_write_read_evaluated"] == 1.0
    assert summary["local_state_write_read_local_mechanism_authorized"] == 1.0
    assert summary["local_state_write_read_full_model_authorized"] == 0.0
    assert summary["local_state_write_read_paid_compute_authorized"] == 0.0
    assert summary["local_state_write_read_arbitrary_chat_authorized"] == 0.0
    assert summary["local_state_write_read_axis_count"] == 1
    assert summary["local_state_write_read_seed_count"] == 1
    assert summary["local_state_write_read_run_count"] == 1
    assert summary["local_state_write_read_parameter_count_max"] < 100000
    assert summary["local_state_write_read_joint_success_min"] >= 0.95
    assert summary["local_state_write_read_state_success_min"] >= 0.95
    assert summary["local_state_write_read_action_success_min"] >= 0.95
    assert summary["local_state_write_read_field_accuracy_floor"] >= 0.95
    assert summary["local_state_write_read_update_joint_success_min"] >= 0.95
    assert summary["local_state_write_read_zero_state_joint_success_max"] == 0.0
    assert summary["local_state_write_read_no_update_joint_success_max"] == 0.0
    assert summary["local_state_write_read_random_update_joint_success_max"] == 0.0
    assert summary["local_state_write_read_matched_sparse_joint_success_max"] == 0.0
    assert summary["local_state_write_read_uncapped_sparse_joint_success_min"] >= 0.95
    assert summary["local_state_write_read_committed_bits_max"] == 19.0
    assert summary["local_state_write_read_accounted_bits_max"] == 56.0
    assert summary["local_state_write_read_engineering_pass"] == 1.0


def test_local_state_write_read_registry_entry() -> None:
    assert "local_state_write_read_mirror" in SIMULATION_SPECS
    assert "local_state_write_read_mirror" in SUITES["compression_mirror"]
    assert "local_state_write_read_mirror" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_state_write_read_mirror"]
    assert spec.category == "compression_mirror"
    assert dict(spec.maximum_summary_values)["local_state_write_read_parameter_count_max"] == 99999.0
    assert dict(spec.maximum_summary_values)["local_state_write_read_full_model_authorized"] == 0.0
    assert dict(spec.minimum_summary_values)["local_state_write_read_engineering_pass"] == 1.0
