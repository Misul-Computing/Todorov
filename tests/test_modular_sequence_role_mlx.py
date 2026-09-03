import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping

import numpy as np
import pytest

from neuroloc.simulations.memory import modular_sequence_role_cpu as runner
from scripts import qualify_modular_mlx as qualification
from src.model import modular_mlx_backend as backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = PROJECT_ROOT / "neuroloc" / "simulations" / "memory" / "modular_sequence_role_mlx.py"
QUALIFICATION_PATH = PROJECT_ROOT / "scripts" / "qualify_modular_mlx.py"


def _engine_literal(name: str):
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"), filename=str(ENGINE_PATH))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return ast.literal_eval(assignment.value)


def _metric_worst_fixture(prefix: str, values: tuple[float, float, float, float]) -> dict[str, Any]:
    names = ("max_abs", "relative_max", "normalized_l2", "cosine")
    records = {}
    for name, value in zip(names, values):
        expected_max_magnitude = values[0] / value if name == "relative_max" else 0.5
        expected_l2 = values[0] / value if name == "normalized_l2" else 1.0
        records[name] = {
            "tensor": f"{prefix}.weight",
            "worst_index": [0, 0],
            "worst_observed": 0.25,
            "worst_expected": 0.25 + values[0],
            "expected_max_magnitude": expected_max_magnitude,
            "expected_l2": expected_l2,
            "observed_l2": 1.0,
            "difference_l2": values[0],
            "cosine_denominator": expected_l2,
            "gradient_floor": 1e-8,
            "mismatch_count": 1,
            "sign_flip_count": 0,
            "value": value,
        }
    return records


def _complete_self_check() -> dict[str, Any]:
    def forward(role: str, stage: str, objective: str, model_seed: int, data_seed: int, batch_size: int, sequence_length: int, parameter_count: int, route_count: int) -> dict[str, Any]:
        return {
            "role": role,
            "stage": stage,
            "objective": objective,
            "model_seed": model_seed,
            "data_seed": data_seed,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mapped_parameter_count": parameter_count,
            "mapping_bijective": True,
            "mapping_sha256": "a" * 64,
            "mapping_transpose": False,
            "mapping_value_count": parameter_count,
            "mapping_value_max_abs": 0.0,
            "mapping_value_byte_exact": True,
            "mapping_source_value_sha256": "b" * 64,
            "mapping_destination_value_sha256": "b" * 64,
            "block_count": 8,
            "raw_route_count": route_count,
            "effective_route_count": route_count,
            "logits_max_abs": 5e-6,
            "hidden_max_abs": 5e-6,
            "sequence_delta_max_abs_by_block": [5e-6] * 8,
            "feature_delta_max_abs_by_block": [5e-6] * 8,
            "forward_relative_max": 5e-6,
            "forward_normalized_l2_max": 5e-7,
            "forward_cosine_min": 1.0,
            "forward_worst_tensor": "logits",
            "forward_worst_index": [0, 0, 0],
            "forward_worst_observed": 0.5,
            "forward_worst_expected": 0.500005,
            "forward_absolute_pass": True,
            "forward_scale_aware_pass": True,
            "forward_pass": True,
            "forward_scale_aware_absolute_tolerance": 5e-5,
            "forward_relative_tolerance": 5e-5,
            "forward_normalized_l2_tolerance": 5e-6,
            "forward_cosine_tolerance": 0.99999999999,
            "route_exact": True,
            "router_loss_max_abs": 5e-7,
            "pass": True,
        }

    selected_identity = ("selected", "joint", "task_plus_0.1_times_internal_router_plus_supervised_route", 3123, 4123, 2, 128, 119, 2)
    full_model = forward(*selected_identity)
    calibration_records = [
        full_model,
        forward("all_eligible", "forward_calibration", "full_forward_output_surface", 3124, 4124, 2, 128, 119, 2),
        forward("dense", "forward_calibration", "full_forward_output_surface", 3125, 4125, 2, 128, 116, 1),
        forward("rung_two", "forward_calibration", "full_forward_output_surface", 3126, 4126, 2, 512, 119, 2),
    ]
    held_out_records = [
        forward("selected", "joint", "task_plus_0.1_times_internal_router_plus_supervised_route", 8123, 9123, 2, 128, 119, 2),
        forward("all_eligible", "forward_calibration", "full_forward_output_surface", 8124, 9124, 3, 128, 119, 2),
    ]
    gradient = {
        "role": "selected",
        "model_seed": 3123,
        "data_seed": 4123,
        "batch_size": 2,
        "loss_max_abs": 5e-7,
        "component_loss_max_abs": 5e-7,
        "component_loss_errors": {
            "task_loss": 5e-7,
            "internal_router_loss": 5e-7,
            "supervised_route_loss": 5e-7,
        },
        "gradient_count": 116,
        "gradient_max_abs": 5e-6,
        "gradient_relative_max": 5e-5,
        "gradient_normalized_l2_max": 2e-5,
        "gradient_cosine_min": 1.0,
        "gradient_absolute_pass": True,
        "gradient_scale_aware_pass": True,
        "gradient_pass": True,
        "gradient_scale_aware_absolute_tolerance": 3e-5,
        "gradient_relative_tolerance": 1e-4,
        "gradient_normalized_l2_tolerance": 5e-5,
        "gradient_cosine_tolerance": 0.999999999,
        "grad_none_zero_exact": True,
        "pass": True,
    }
    raw_metrics = (5e-6, 5e-5, 2e-5, 1.0)
    clipped_metrics = (4e-6, 4e-5, 1e-5, 1.0)
    update_metrics = (3e-7, 3e-5, 1e-5, 1.0)
    actual = {
        "lanes": 5,
        "stage": "joint",
        "objective": "task_plus_0.1_times_internal_router_plus_supervised_route",
        "construction_seeds": [11, 23, 37, 53, 71],
        "data_seeds": [300011, 300023, 300037, 300053, 300071],
        "batch_size_per_lane": 2,
        "sequence_length": 128,
        "logical_update": 1,
        "learning_rates": {"block_4_router": 0.001, "other_trainable": 0.00025},
        "initial_optimizer_step": 0,
        "initial_first_and_second_moments_exact_zero": True,
        "mapping_value_count": 595,
        "mapping_value_max_abs": 0.0,
        "mapping_value_byte_exact": True,
        "mapping_source_value_sha256": "c" * 64,
        "mapping_destination_value_sha256": "c" * 64,
        "unique_parameter_hashes": 5,
        "codebook_grad_none_effect_exact": True,
        "finite": True,
        "torch_loss_max_abs": 5e-7,
        "torch_parameter_max_abs": 3e-7,
        "torch_first_moment_max_abs": 2e-7,
        "torch_second_moment_max_abs": 1e-7,
        "torch_route_exact": True,
        "five_lane_gradient_count": 580,
        "five_lane_gradient_max_abs": raw_metrics[0],
        "five_lane_gradient_relative_max": raw_metrics[1],
        "five_lane_gradient_normalized_l2_max": raw_metrics[2],
        "five_lane_gradient_cosine_min": raw_metrics[3],
        "five_lane_gradient_worst_tensor": "raw.weight",
        "five_lane_gradient_worst_index": [0, 0],
        "five_lane_gradient_worst_observed": 0.25,
        "five_lane_gradient_worst_expected": 0.250005,
        "five_lane_gradient_metric_worst": _metric_worst_fixture("raw", raw_metrics),
        "five_lane_gradient_absolute_pass": True,
        "five_lane_gradient_scale_aware_pass": True,
        "five_lane_gradient_pass": True,
        "five_lane_gradient_scale_aware_absolute_tolerance": 1.25e-4,
        "five_lane_gradient_relative_tolerance": 2.5e-4,
        "five_lane_gradient_normalized_l2_tolerance": 1.25e-4,
        "five_lane_gradient_cosine_tolerance": 0.99999999,
        "five_lane_grad_none_zero_exact": True,
        "five_lane_raw_gradient_sha256": "d" * 64,
        "five_lane_clipped_gradient_count": 580,
        "five_lane_clipped_gradient_max_abs": clipped_metrics[0],
        "five_lane_clipped_gradient_relative_max": clipped_metrics[1],
        "five_lane_clipped_gradient_normalized_l2_max": clipped_metrics[2],
        "five_lane_clipped_gradient_cosine_min": clipped_metrics[3],
        "five_lane_clipped_gradient_metric_worst": _metric_worst_fixture("clipped", clipped_metrics),
        "five_lane_clipped_gradient_pass": True,
        "parameter_update_max_abs": update_metrics[0],
        "parameter_update_relative_max": update_metrics[1],
        "parameter_update_normalized_l2_max": update_metrics[2],
        "parameter_update_cosine_min": update_metrics[3],
        "parameter_update_metric_worst": _metric_worst_fixture("update", update_metrics),
        "optimizer_gradient_sha256": "e" * 64,
        "torch_optimizer_gradient_sha256": "f" * 64,
        "mlx_optimizer_parameter_formula_max_abs": 2e-8,
        "mlx_optimizer_first_formula_max_abs": 2e-8,
        "mlx_optimizer_second_formula_max_abs": 2e-8,
        "torch_optimizer_parameter_formula_max_abs": 2e-8,
        "torch_optimizer_first_formula_max_abs": 2e-8,
        "torch_optimizer_second_formula_max_abs": 2e-8,
        "optimizer_formula_max_bound_ratio": 0.5,
        "optimizer_formula_bound_ratio_tolerance": 1.0,
        "optimizer_formula_pass": True,
        "optimizer_formula_worst_runtime": "MLX",
        "optimizer_formula_worst_surface": "parameter",
        "optimizer_formula_worst_tensor": "weight",
        "optimizer_formula_worst_lane": 0,
        "optimizer_formula_worst_index": [0, 0],
        "optimizer_formula_worst_abs": 2e-8,
        "optimizer_formula_worst_bound": 4e-8,
        "optimizer_formula_worst_bound_ratio": 0.5,
        "optimizer_formula_worst_observed": 0.2,
        "optimizer_formula_worst_expected": 0.20000002,
        "causal_parameter_residual_max_abs": 2e-8,
        "causal_first_moment_residual_max_abs": 1e-8,
        "causal_second_moment_residual_max_abs": 5e-9,
        "causal_residual_summary": {
            "parameter": {"max_abs": 2e-8, "max_bound": 4e-8, "max_bound_ratio": 0.5, "worst_excess": -2e-8},
            "first_moment": {"max_abs": 1e-8, "max_bound": 4e-8, "max_bound_ratio": 0.25, "worst_excess": -3e-8},
            "second_moment": {"max_abs": 5e-9, "max_bound": 2e-8, "max_bound_ratio": 0.25, "worst_excess": -1.5e-8},
        },
        "causal_residual_pass": True,
        "causal_residual_worst_surface": "parameter",
        "causal_residual_worst_tensor": "weight",
        "causal_residual_worst_lane": 0,
        "causal_residual_worst_index": [0, 0],
        "causal_residual_worst_abs": 2e-8,
        "causal_residual_worst_bound": 4e-8,
        "causal_residual_worst_bound_ratio": 0.5,
        "causal_residual_worst_excess": -2e-8,
        "end_to_end_worst_max_abs": 3e-7,
        "end_to_end_worst_lane": 0,
        "end_to_end_worst_tensor": "weight",
        "end_to_end_worst_index": [0, 0],
        "end_to_end_worst_observed": 0.2,
        "end_to_end_worst_expected": 0.2000003,
        "end_to_end_worst_mlx_clipped_gradient": 0.01,
        "end_to_end_worst_torch_clipped_gradient": 0.010005,
        "mlx_preclip_gradient_norms": [2.0] * 5,
        "mlx_postclip_gradient_norms": [1.0] * 5,
        "torch_preclip_gradient_norms": [2.0] * 5,
        "torch_postclip_gradient_norms": [1.0] * 5,
        "torch_parity_pass": True,
        "pass": True,
    }
    carried = {
        "lanes": 5,
        "tensor_count": 2,
        "first_update": 1,
        "tested_update": 2,
        "bias_correction": True,
        "nonzero_carried_first_and_second_moments": True,
        "distinct_second_gradient": True,
        "canonical_gradient_sha256": "1" * 64,
        "gradient_clip_identity": True,
        "formula_unit_roundoff": 2.0**-24,
        "formula_parameter_operation_budget": 32,
        "formula_first_moment_operation_budget": 6,
        "formula_second_moment_operation_budget": 8,
        "max_bound_ratio": 0.5,
        "worst_runtime": "MLX",
        "worst_surface": "parameter",
        "worst_tensor": "decayed",
        "worst_lane": 0,
        "worst_index": [0],
        "worst_abs": 2e-8,
        "worst_bound": 4e-8,
        "cross_runtime_max_abs": 3e-7,
        "pass": True,
    }
    return {
        "schema_version": backend.IPC_SCHEMA_VERSION,
        "mlx_version": backend.MLX_VERSION,
        "device": "Device(gpu, 0)",
        "contract": backend.backend_contract(),
        "full_model_parity": full_model,
        "all_role_forward_calibration": {"roles": ["selected", "all_eligible", "dense", "rung_two"], "fresh_process_required": True, "records": calibration_records, "pass": True},
        "held_out_forward_admission": {"thresholds_frozen_before_execution": True, "records": held_out_records, "pass": True},
        "full_gradient_parity": gradient,
        "adamw_parity": {"max_abs": 0.0, "tolerance": 1e-7, "pass": True},
        "carried_adamw_parity": carried,
        "vmap5": {"lanes": 5, "unique_parameter_hashes": 5, "lane_local_clipping": True, "pure_parameter_tree": True, "pass": True},
        "functional_forward": {"logits_max_abs": 0.0, "query_route_max_abs": 0.0, "key_route_max_abs": 0.0, "router_loss_max_abs": 0.0, "route_exact": True, "pass": True},
        "actual_model_vmap5": actual,
        "memory": {"active_memory_bytes": 1, "cache_memory_bytes": 1, "peak_memory_bytes": 1, "parent_rss_and_swap_required": True},
        "pass": True,
    }


def _hello(sequence: int = 0) -> dict:
    dependencies = backend.dependency_hashes()
    self_check = _complete_self_check()
    return {
        "kind": "hello",
        "sequence": sequence,
        "schema_version": backend.IPC_SCHEMA_VERSION,
        "mlx_version": backend.MLX_VERSION,
        "engine_sha256": dependencies["engine"],
        "dependency_sha256s": dependencies,
        "self_check": self_check,
        "self_check_sha256": __import__("hashlib").sha256(json.dumps(self_check, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "device": "Device(gpu, 0)",
    }


def test_backend_contract_fixes_runtime_workload_and_time_gates() -> None:
    contract = backend.backend_contract()
    assert contract["schema_version"] == "todorov.modular-mlx-backend.1"
    assert contract["mlx_version"] == "0.29.3"
    assert contract["python_path"] == "/Users/dttdrv/Projects/Transformerov/.venv/bin/python"
    assert contract["positions"] == 45_613_056
    assert contract["target_seconds"] == 600
    assert contract["hard_limit_seconds"] == 1200
    assert contract["rung_one_seeds"] == [11, 23, 37, 53, 71]
    assert contract["rung_two_seed"] == 83
    assert contract["sequential_stages"] == ["donor", "rung_two"]
    assert contract["vectorized_stages"] == ["router_only", "joint", "dense_base", "dense_continuation"]
    assert contract["vectorized_width"] == 5
    assert contract["torch_reference_authority"] is True
    assert contract["resume_claim_allowed"] is False


def test_initial_parity_validator_rejects_every_frozen_surface_and_each_sequence_delta() -> None:
    valid = _complete_self_check()
    evidence = backend.validate_initial_self_check(valid)
    assert evidence["pass"] is True
    assert evidence["worst_bound_ratio"] == pytest.approx(0.5)
    mutations = (
        ("full_model_parity", "logits_max_abs", 1.00001e-5),
        ("full_model_parity", "hidden_max_abs", 1.00001e-5),
        ("full_model_parity", "router_loss_max_abs", 1.00001e-6),
        ("full_model_parity", "mapping_bijective", False),
        ("full_model_parity", "route_exact", False),
        ("full_gradient_parity", "loss_max_abs", 1.00001e-6),
        ("full_gradient_parity", "component_loss_max_abs", 1.00001e-6),
        ("full_gradient_parity", "gradient_pass", False),
        ("full_gradient_parity", "grad_none_zero_exact", False),
        ("adamw_parity", "pass", False),
        ("carried_adamw_parity", "pass", False),
        ("vmap5", "pass", False),
        ("functional_forward", "pass", False),
        ("actual_model_vmap5", "torch_loss_max_abs", 1.00001e-6),
        ("actual_model_vmap5", "torch_route_exact", False),
        ("actual_model_vmap5", "torch_parity_pass", False),
        ("actual_model_vmap5", "pass", False),
    )
    for section, field, value in mutations:
        candidate = json.loads(json.dumps(valid))
        candidate[section][field] = value
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)


def test_initial_parity_validator_rejects_held_out_lane_permutation_and_route_flip() -> None:
    valid = _complete_self_check()
    permutations = (
        ("all_role_forward_calibration", 1, "model_seed", 3125),
        ("held_out_forward_admission", 0, "role", "all_eligible"),
        ("held_out_forward_admission", 1, "batch_size", 2),
        ("held_out_forward_admission", 1, "route_exact", False),
    )
    for section, index, field, value in permutations:
        candidate = json.loads(json.dumps(valid))
        candidate[section]["records"][index][field] = value
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)


def test_initial_parity_validator_rejects_gradient_cardinality_and_moment_corruption() -> None:
    valid = _complete_self_check()
    mutations = (
        ("actual_model_vmap5", "five_lane_gradient_count", 579),
        ("actual_model_vmap5", "five_lane_clipped_gradient_count", 581),
        ("actual_model_vmap5", "initial_first_and_second_moments_exact_zero", False),
        ("carried_adamw_parity", "nonzero_carried_first_and_second_moments", False),
        ("carried_adamw_parity", "tested_update", 1),
        ("carried_adamw_parity", "max_bound_ratio", 1.000001),
    )
    for section, field, value in mutations:
        candidate = json.loads(json.dumps(valid))
        candidate[section][field] = value
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)


def test_initial_parity_validator_binds_declared_full_gradient_tolerances() -> None:
    valid = _complete_self_check()
    mutations = (
        ("gradient_scale_aware_absolute_tolerance", 3.00001e-5),
        ("gradient_relative_tolerance", 1.00001e-4),
        ("gradient_normalized_l2_tolerance", 5.00001e-5),
        ("gradient_cosine_tolerance", 0.999999998),
    )
    for field, value in mutations:
        candidate = json.loads(json.dumps(valid))
        candidate["full_gradient_parity"][field] = value
        with pytest.raises(backend.MlxQualificationError, match="gradient parity tolerance"):
            backend.validate_initial_self_check(candidate)


def test_initial_parity_validator_rejects_each_raw_and_clipped_gradient_threshold_branch() -> None:
    valid = _complete_self_check()
    branches = (
        ("five_lane_gradient_max_abs", 1.25001e-4),
        ("five_lane_gradient_relative_max", 2.50001e-4),
        ("five_lane_gradient_normalized_l2_max", 1.25001e-4),
        ("five_lane_gradient_cosine_min", 0.999999989),
    )
    for field, value in branches:
        candidate = json.loads(json.dumps(valid))
        actual = candidate["actual_model_vmap5"]
        metrics = [1.1e-4, 5e-5, 2e-5, 1.0]
        metrics[{"five_lane_gradient_max_abs": 0, "five_lane_gradient_relative_max": 1, "five_lane_gradient_normalized_l2_max": 2, "five_lane_gradient_cosine_min": 3}[field]] = value
        actual["five_lane_gradient_max_abs"], actual["five_lane_gradient_relative_max"], actual["five_lane_gradient_normalized_l2_max"], actual["five_lane_gradient_cosine_min"] = metrics
        actual["five_lane_gradient_worst_expected"] = 0.25 + metrics[0]
        actual["five_lane_gradient_metric_worst"] = _metric_worst_fixture("raw", tuple(metrics))
        actual["five_lane_gradient_absolute_pass"] = False
        actual["five_lane_gradient_scale_aware_pass"] = True
        actual["five_lane_gradient_pass"] = True
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)
    clipped = (
        ("five_lane_clipped_gradient_max_abs", 1.25001e-4),
        ("five_lane_clipped_gradient_relative_max", 2.50001e-4),
        ("five_lane_clipped_gradient_normalized_l2_max", 1.25001e-4),
        ("five_lane_clipped_gradient_cosine_min", 0.999999989),
    )
    for field, value in clipped:
        candidate = json.loads(json.dumps(valid))
        actual = candidate["actual_model_vmap5"]
        metrics = [1.1e-4, 4e-5, 1e-5, 1.0]
        metrics[{"five_lane_clipped_gradient_max_abs": 0, "five_lane_clipped_gradient_relative_max": 1, "five_lane_clipped_gradient_normalized_l2_max": 2, "five_lane_clipped_gradient_cosine_min": 3}[field]] = value
        actual["five_lane_clipped_gradient_max_abs"], actual["five_lane_clipped_gradient_relative_max"], actual["five_lane_clipped_gradient_normalized_l2_max"], actual["five_lane_clipped_gradient_cosine_min"] = metrics
        actual["five_lane_clipped_gradient_metric_worst"] = _metric_worst_fixture("clipped", tuple(metrics))
        actual["five_lane_clipped_gradient_pass"] = False
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)


def test_initial_parity_validator_rejects_metric_worst_and_a_priori_residual_corruption() -> None:
    valid = _complete_self_check()
    corruptions = (
        ("five_lane_gradient_metric_worst", "max_abs", "value", 4e-6),
        ("five_lane_clipped_gradient_metric_worst", "relative_max", "value", 3e-5),
        ("parameter_update_metric_worst", "normalized_l2", "sign_flip_count", -1),
    )
    for surface, metric, field, value in corruptions:
        candidate = json.loads(json.dumps(valid))
        candidate["actual_model_vmap5"][surface][metric][field] = value
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)
    for surface in ("parameter", "first_moment", "second_moment"):
        candidate = json.loads(json.dumps(valid))
        candidate["actual_model_vmap5"]["causal_residual_summary"][surface]["max_bound_ratio"] = 1.000001
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)
    for field in ("sequence_delta_max_abs_by_block", "feature_delta_max_abs_by_block"):
        for index in range(8):
            candidate = json.loads(json.dumps(valid))
            candidate["full_model_parity"][field][index] = 1.00001e-5
            with pytest.raises(backend.MlxQualificationError):
                backend.validate_initial_self_check(candidate)
    for field in ("task_loss", "internal_router_loss", "supervised_route_loss"):
        candidate = json.loads(json.dumps(valid))
        candidate["full_gradient_parity"]["component_loss_errors"][field] = 1.00001e-6
        candidate["full_gradient_parity"]["component_loss_max_abs"] = 1.00001e-6
        with pytest.raises(backend.MlxQualificationError):
            backend.validate_initial_self_check(candidate)


def test_causal_residual_producer_keeps_selected_index_and_summary_excess_distinct() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "actual_model_vmap5_probe")
    body = ast.get_source_segment(source, function)
    assert body is not None
    assert '"max_excess": float(np.max(excess))' in body
    assert '"worst_excess": float(excess.reshape(-1)[residual_flat_index])' in body
    assert '"worst_residual": float(residual.reshape(-1)[residual_flat_index])' in body
    assert '"worst_bound": float(allowed.reshape(-1)[residual_flat_index])' in body
    assert '"worst_excess": max(record["max_excess"] for record in surface_records)' in body
    assert '"causal_residual_worst_abs": worst_causal_residual["worst_residual"]' in body
    assert '"causal_residual_worst_bound": worst_causal_residual["worst_bound"]' in body
    assert '"causal_residual_worst_excess": worst_causal_residual["worst_excess"]' in body
    assert '"worst_excess": float(np.max(excess))' not in body


def test_parameter_update_reduction_allows_only_float32_subtraction_roundoff() -> None:
    valid = _complete_self_check()
    accepted = json.loads(json.dumps(valid))
    accepted_actual = accepted["actual_model_vmap5"]
    accepted_actual["parameter_update_max_abs"] = 3.05e-7
    accepted_actual["parameter_update_metric_worst"] = _metric_worst_fixture("update", (3.05e-7, 3e-5, 1e-5, 1.0))
    assert backend.validate_initial_self_check(accepted)["pass"] is True
    rejected = json.loads(json.dumps(valid))
    rejected_actual = rejected["actual_model_vmap5"]
    rejected_actual["parameter_update_max_abs"] = 3.12e-7
    rejected_actual["parameter_update_metric_worst"] = _metric_worst_fixture("update", (3.12e-7, 3e-5, 1e-5, 1.0))
    with pytest.raises(backend.MlxQualificationError):
        backend.validate_initial_self_check(rejected)
    exact_binding = json.loads(json.dumps(valid))
    exact_binding["actual_model_vmap5"]["end_to_end_worst_max_abs"] = 3.0000001e-7
    with pytest.raises(backend.MlxQualificationError):
        backend.validate_initial_self_check(exact_binding)


def test_projection_uses_measured_components_and_refuses_over_hard_limit() -> None:
    measured = {
        "donor_step_seconds": 0.0264007,
        "selected_vmap5_step_seconds": 0.0742675,
        "dense_vmap5_step_seconds": 0.0616947,
        "rung_two_step_seconds": 0.0438553,
        "cold_child_start_seconds": 1.0,
        "cold_compile_seconds": 2.0,
        "durable_ledger_seconds": 3.0,
        "routing_evidence_seconds": 4.0,
        "evaluation_seconds": 12.5,
        "checkpoint_reload_seconds": 5.0,
        "packaging_seconds": 8.0,
        "resource_finalization_seconds": 3.25,
        "lifecycle_close_join_seconds": 1.25,
    }
    projection = backend.project_full_package(measured)
    expected = 5 * 1024 * measured["donor_step_seconds"]
    expected += 1280 * measured["selected_vmap5_step_seconds"]
    expected += 1536 * measured["dense_vmap5_step_seconds"]
    expected += 1536 * measured["rung_two_step_seconds"]
    expected += sum(
        measured[name]
        for name in (
            "cold_child_start_seconds",
            "cold_compile_seconds",
            "durable_ledger_seconds",
            "routing_evidence_seconds",
            "evaluation_seconds",
            "checkpoint_reload_seconds",
            "packaging_seconds",
            "resource_finalization_seconds",
            "lifecycle_close_join_seconds",
        )
    )
    assert projection["projected_seconds"] == pytest.approx(expected)
    assert projection["positions"] == 45_613_056
    assert projection["target_pass"] is True
    assert projection["hard_limit_pass"] is True
    assert projection["attempt_event_rows"] == 41_472
    assert projection["routing_evidence_rows"] == 588_240
    too_slow = dict(measured)
    too_slow["packaging_seconds"] = 900.0
    stopped = backend.project_full_package(too_slow)
    assert stopped["projected_seconds"] > 1200
    assert stopped["target_pass"] is False
    assert stopped["hard_limit_pass"] is False


def test_projection_rejects_missing_resource_finalization_measurement() -> None:
    measured = {
        "donor_step_seconds": 0.01,
        "selected_vmap5_step_seconds": 0.01,
        "dense_vmap5_step_seconds": 0.01,
        "rung_two_step_seconds": 0.01,
        "cold_child_start_seconds": 0.01,
        "cold_compile_seconds": 0.01,
        "durable_ledger_seconds": 0.01,
        "routing_evidence_seconds": 0.01,
        "evaluation_seconds": 0.01,
        "checkpoint_reload_seconds": 0.01,
        "packaging_seconds": 0.01,
        "lifecycle_close_join_seconds": 0.01,
    }
    with pytest.raises(backend.MlxBackendRefusal, match="component keys"):
        backend.project_full_package(measured)


def test_resource_finalization_benchmark_validator_binds_exact_conservative_schema() -> None:
    valid = {
        "component_seconds": 5.5,
        "actual_stop_seconds": 3.0,
        "final_active_jobs": [],
        "final_attempted_updates": 132,
        "final_expected_pids": [321],
        "final_sample_id": 5,
        "final_token_positions": 292_864,
        "interval_seconds": 5.0,
        "max_observed_sample_duration_seconds": 0.25,
        "sample_transaction_count": 6,
    }
    assert qualification.validate_resource_finalization_benchmark(valid) == valid
    extra = {**valid, "invented": True}
    with pytest.raises(backend.MlxQualificationError, match="schema"):
        qualification.validate_resource_finalization_benchmark(extra)
    undercharged = {**valid, "component_seconds": 5.25}
    with pytest.raises(backend.MlxQualificationError, match="bound"):
        qualification.validate_resource_finalization_benchmark(undercharged)


def test_shared_tail_detail_schema_recomputes_all_four_component_formulas() -> None:
    engine_sha256 = backend.dependency_hashes()["engine"]
    runtime = runner._import_runtime()
    source = runner.generate_rung_one_batch(123456, 2, runtime.torch)["required_source"] * 16
    raw = [[index % 15, (index + 1) % 15] for index in range(32)]
    source_exclusion_fixture = runner.generate_source_exclusion_routes(633456, raw, source, runtime.torch)
    source_exclusion_sha256 = runner.canonical_json_sha256(source_exclusion_fixture)

    def timing(families: tuple[str, ...]) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, int]]:
        return (
            {name: [10] for name in families},
            {name: [20, 30, 40] for name in families},
            {name: 40 for name in families},
        )

    evaluation_families = (
        "endpoint_replay.dense_base",
        "endpoint_replay.dense_continuation",
        "endpoint_replay.donor",
        "endpoint_replay.joint",
        "endpoint_replay.router_only",
        "endpoint_replay.rung_two",
        "route_acquisition",
        "rung_one_dense",
        "rung_one_routed.all_eligible_clone",
        "rung_one_routed.all_eligible_donor",
        "rung_one_routed.block4_local_only",
        "rung_one_routed.block4_routed_knockout",
        "rung_one_routed.carry_reset",
        "rung_one_routed.carry_shuffle",
        "rung_one_routed.intact",
        "rung_one_routed.matched_random_route",
        "rung_one_routed.recurrent_knockout",
        "rung_one_routed.required_source_excluded",
        "rung_one_routed.target_forced",
        "rung_two.intact",
        "rung_two.recurrent_knockout",
    )
    checkpoint_families = ("dense_vmap5_all_lanes", "donor_single", "joint_vmap5_all_lanes", "router_only_vmap5_all_lanes", "rung_two_single")
    routing_families = ("routing_evidence_block",)
    packaging_families = ("file_batch", "io_block")
    evaluation_timing = timing(evaluation_families)
    checkpoint_timing = timing(checkpoint_families)
    routing_timing = timing(routing_families)
    packaging_timing = timing(packaging_families)
    evaluation_total = 80 * 40 + 880 * 40 + 80 * 40 + 32 * 40 + 26 * 40
    checkpoint_total = 5 * 40 + 40 + 40 + 2 * 40 + 40
    routing_total = 136 * 40
    packaging_total_numerator = 66 * 40 * 32 + 260 * 40
    evaluation_hashes = qualification.expected_tail_fixture_sha256s("evaluation", source_exclusion_sha256=source_exclusion_sha256)
    checkpoint_hashes = qualification.expected_tail_fixture_sha256s("checkpoint_reload", engine_sha256=engine_sha256)
    routing_hashes = qualification.expected_tail_fixture_sha256s("routing_evidence", routing_block_sha256="c" * 64)
    packaging_hashes = qualification.expected_tail_fixture_sha256s("packaging")
    details = {
        "evaluation": {
            "fixture_sha256s": evaluation_hashes,
            "warmup_duration_ns": evaluation_timing[0],
            "timed_duration_ns": evaluation_timing[1],
            "selected_max_duration_ns": evaluation_timing[2],
            "counts": {"route_acquisition_calls": 80, "rung_one_routed_calls": 880, "rung_one_routed_conditions": 11, "rung_one_dense_calls": 80, "rung_two_calls": 32, "rung_two_conditions": 2, "endpoint_replay_calls": 26, "endpoint_replay_roles": 6},
            "byte_sizes": {"nonclaim_fixture_bytes": 438368},
            "scaling": {"route_acquisition_ns": 3200, "rung_one_routed_ns": 35200, "rung_one_dense_ns": 3200, "rung_two_ns": 1280, "endpoint_replay_ns": 1040, "total_ns": evaluation_total},
            "scratch_cleanup_pass": True,
            "component_seconds": evaluation_total / 1_000_000_000,
        },
        "checkpoint_reload": {
            "fixture_sha256s": checkpoint_hashes,
            "warmup_duration_ns": checkpoint_timing[0],
            "timed_duration_ns": checkpoint_timing[1],
            "selected_max_duration_ns": checkpoint_timing[2],
            "counts": {"donor_single_coefficient": 5, "router_only_vmap5_all_lanes_coefficient": 1, "joint_vmap5_all_lanes_coefficient": 1, "dense_vmap5_all_lanes_coefficient": 2, "rung_two_single_coefficient": 1, "trained_endpoint_files": 26},
            "byte_sizes": {"dense_vmap5_all_lanes": 34240660, "donor_single": 6856580, "joint_vmap5_all_lanes": 34367440, "router_only_vmap5_all_lanes": 11567740, "rung_two_single": 7053188, "projected_checkpoint_bytes": 155752588},
            "scaling": {"donor_single_ns": 200, "router_only_vmap5_all_lanes_ns": 40, "joint_vmap5_all_lanes_ns": 40, "dense_vmap5_all_lanes_ns": 80, "rung_two_single_ns": 40, "total_ns": checkpoint_total},
            "scratch_cleanup_pass": True,
            "component_seconds": checkpoint_total / 1_000_000_000,
        },
        "routing_evidence": {
            "fixture_sha256s": routing_hashes,
            "warmup_duration_ns": routing_timing[0],
            "timed_duration_ns": routing_timing[1],
            "selected_max_duration_ns": routing_timing[2],
            "counts": {"block_copies": 128, "block_rows": 4352, "claim_rows": 588240, "microtrace_rows": 34, "projected_rows": 591872, "scale_blocks": 136},
            "byte_sizes": {"block_uncompressed_bytes": 4096, "max_line_bytes": 128, "raw_gzip_bytes_max": 2048},
            "scaling": {"scale_blocks": 136, "selected_max_duration_ns": 40, "total_ns": routing_total},
            "scratch_cleanup_pass": True,
            "component_seconds": routing_total / 1_000_000_000,
        },
        "packaging": {
            "fixture_sha256s": packaging_hashes,
            "warmup_duration_ns": packaging_timing[0],
            "timed_duration_ns": packaging_timing[1],
            "selected_max_duration_ns": packaging_timing[2],
            "counts": {"attempt_rows": 41472, "bulk_fixed_paths": 55, "check_detail_count": 116, "completed_update_rows": 20736, "evaluation_rows": 327, "file_batch_size": 32, "fixed_clean_files": 144, "future_parity_details": 108, "future_pilot_tail_details": 6, "prediction_rows": 31744, "preflight_detail_count": 2, "projected_files": 260, "remaining_fixed_paths": 89, "routing_rows": 588240, "scaled_io_blocks": 66},
                "byte_sizes": {"attempt_max_line_bytes": 100, "check_detail_schema_bound_bytes": 1048576, "evaluation_max_line_bytes": 100, "io_block_bytes": 16777216, "nonbulk_schema_bound_bytes": 8388608, "prediction_max_line_bytes": 100, "preflight_detail_bytes": 200, "projected_bytes": 1092225416, "projected_checkpoint_bytes": 155752588, "routing_max_line_bytes": 100, "train_max_line_bytes": 100},
                "scaling": {"file_batch_divisor": 32, "file_batch_max_duration_ns": 40, "file_batch_scaled_duration_numerator_ns": 10400, "io_block_max_duration_ns": 40, "io_block_total_ns": 2640, "total_duration_numerator_ns": packaging_total_numerator},
            "scratch_cleanup_pass": True,
            "component_seconds": packaging_total_numerator / 32 / 1_000_000_000,
        },
    }
    expected_hashes = {name: detail["fixture_sha256s"] for name, detail in details.items()}
    for name, detail in details.items():
        assert qualification.validate_tail_benchmark_detail(name, detail, detail["component_seconds"], expected_hashes[name]) == detail
        serialized = json.loads(json.dumps(detail, sort_keys=True, separators=(",", ":")))
        assert qualification.validate_tail_benchmark_detail(name, serialized, detail["component_seconds"], expected_hashes[name]) == serialized
    for name, detail in details.items():
        candidate = json.loads(json.dumps(detail))
        candidate["component_seconds"] += 1e-9
        with pytest.raises(backend.MlxQualificationError):
            qualification.validate_tail_benchmark_detail(name, candidate, candidate["component_seconds"], expected_hashes[name])
    corrupted = json.loads(json.dumps(details["evaluation"]))
    corrupted["fixture_sha256s"]["random_routes_seed_500011"] = "d" * 64
    with pytest.raises(backend.MlxQualificationError, match="fixture hashes"):
        qualification.validate_tail_benchmark_detail("evaluation", corrupted, corrupted["component_seconds"], evaluation_hashes)
    underreported_evaluation = json.loads(json.dumps(details["evaluation"]))
    underreported_evaluation["byte_sizes"]["nonclaim_fixture_bytes"] -= 1
    with pytest.raises(backend.MlxQualificationError, match="evaluation detail inputs"):
        qualification.validate_tail_benchmark_detail("evaluation", underreported_evaluation, underreported_evaluation["component_seconds"], evaluation_hashes)
    underreported_checkpoint = json.loads(json.dumps(details["checkpoint_reload"]))
    underreported_checkpoint["byte_sizes"] = {
        "dense_vmap5_all_lanes": 1,
        "donor_single": 1,
        "joint_vmap5_all_lanes": 1,
        "router_only_vmap5_all_lanes": 1,
        "rung_two_single": 1,
        "projected_checkpoint_bytes": 10,
    }
    with pytest.raises(backend.MlxQualificationError, match="checkpoint detail inputs"):
        qualification.validate_tail_benchmark_detail("checkpoint_reload", underreported_checkpoint, underreported_checkpoint["component_seconds"], checkpoint_hashes)
    assert evaluation_hashes == {
        "random_routes_seed_500011": "18f568b628517fa8f77d9e6adc17c3c2ead62c46070487d416d6eee25953e54c",
        "rung_one_seed_123456": "98ff3b54f14306135eafe5a92da7abdf1111cd8690e511188bb5f0e44dcab2a9",
        "rung_two_seed_123456": "7fff37e20adc2241c217b3ed6dad6ec4d85e818d69a59fa5b8e3f5a48f2b8afe",
        "source_exclusion_seed_633456": source_exclusion_sha256,
    }
    assert packaging_hashes == {
        "empty_file": hashlib.sha256(b"").hexdigest(),
        "io_block": hashlib.sha256(qualification.packaging_block()).hexdigest(),
    }
    child_details = {"evaluation": details["evaluation"], "checkpoint_reload": details["checkpoint_reload"]}
    child_components = {"evaluation_seconds": details["evaluation"]["component_seconds"], "checkpoint_reload_seconds": details["checkpoint_reload"]["component_seconds"]}
    assert qualification.validate_child_tail_benchmarks(child_details, child_components, source_exclusion_fixture) == child_details
    arbitrary_hashes = json.loads(json.dumps(child_details))
    arbitrary_hashes["evaluation"]["fixture_sha256s"]["source_exclusion_seed_633456"] = "a" * 64
    with pytest.raises(backend.MlxQualificationError, match="fixture hashes"):
        qualification.validate_child_tail_benchmarks(arbitrary_hashes, child_components, source_exclusion_fixture)
    fixture_mutations = []
    missing_key = json.loads(json.dumps(source_exclusion_fixture))
    missing_key.pop("raw")
    fixture_mutations.append(missing_key)
    wrong_cardinality = json.loads(json.dumps(source_exclusion_fixture))
    wrong_cardinality["raw"].pop()
    fixture_mutations.append(wrong_cardinality)
    wrong_range = json.loads(json.dumps(source_exclusion_fixture))
    wrong_range["raw"][0][0] = 15
    fixture_mutations.append(wrong_range)
    wrong_source = json.loads(json.dumps(source_exclusion_fixture))
    wrong_source["source"][0] = (wrong_source["source"][0] + 1) % 15
    fixture_mutations.append(wrong_source)
    wrong_route = json.loads(json.dumps(source_exclusion_fixture))
    wrong_route["routes"][0] = [13, 14] if wrong_route["routes"][0] != [13, 14] else [11, 12]
    fixture_mutations.append(wrong_route)
    for candidate in fixture_mutations:
        with pytest.raises(backend.MlxQualificationError, match="source exclusion"):
            qualification.validate_child_tail_benchmarks(child_details, child_components, candidate)
    assert qualification.TAIL_DETAIL_KEYS == tuple(json.loads((PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json").read_text(encoding="utf-8"))["artifacts"]["schemas"]["pilot"]["tail_benchmark_assertion_detail_output_exact_keys"])


def test_tail_producers_bind_selected_clone_fixture_bytes_and_owned_cleanup() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef)}
    parent_source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    parent_tree = ast.parse(parent_source, filename=str(QUALIFICATION_PATH))
    parent_functions = {node.name: ast.get_source_segment(parent_source, node) or "" for node in parent_tree.body if isinstance(node, ast.FunctionDef)}
    evaluation = functions["pilot_evaluation_benchmark"]
    merge = functions["pilot_merge_tail_benchmarks"]
    selected_branch = evaluation[evaluation.index('if workload == "selected_vmap5":'):evaluation.index('elif workload == "donor":')]
    donor_branch = evaluation[evaluation.index('elif workload == "donor":'):evaluation.index('elif workload == "dense_vmap5":')]
    assert "all_eligible_clone" in selected_branch
    assert "all_eligible_clone" not in donor_branch
    assert "pilot_owned_scratch" in evaluation
    assert "pilot_remove_owned_scratch" in evaluation
    assert 'scratch_cleanup_pass": all(cleanup)' in evaluation
    assert 'source_exclusion_route_bytes' in evaluation
    assert 'fixture_bytes += source_exclusion_route_bytes' in merge
    assert 'scratch_cleanup_pass": all(evaluation_cleanup)' in merge
    assert "pilot evaluation fixture hash differs" in merge
    parent_fixture = parent_functions["validate_source_exclusion_fixture"]
    parent_validation = parent_functions["validate_child_tail_benchmarks"]
    assert "cpu.generate_rung_one_batch(123456, 2, runtime.torch)" in parent_fixture
    assert "cpu.generate_source_exclusion_routes(633456, raw, expected_source, runtime.torch)" in parent_fixture
    assert "pilot_torch_model_from_state(state, 0, \"selected\")" in evaluation
    assert 'expected_tail_fixture_sha256s("evaluation", source_exclusion_sha256=source_exclusion_sha256)' in parent_validation
    assert "source_exclusion_sha256=hashes.get" not in parent_validation
    assert '"source_exclusion_fixture": source_exclusion_fixture' in functions["pilot"]
    checkpoint = functions["pilot_checkpoint_reload_benchmark"]
    assert 'engine_sha256 = dependency_hashes()["engine"]' in checkpoint
    assert '"engine_sha256": engine_sha256' in checkpoint
    assert 'scratch_cleanup_pass": all(cleanup)' in checkpoint
    assert 'scratch_cleanup_pass": all(checkpoint_cleanup)' in merge
    owned_cleanup = functions["pilot_remove_owned_scratch"]
    assert checkpoint.index("pilot_remove_owned_scratch(scratch)") < checkpoint.index("duration_ns = time.perf_counter_ns() - started_ns")
    assert owned_cleanup.index("path.rmdir()") < owned_cleanup.index("pilot_fsync_directory(parent)")


def test_checkpoint_tensor_byte_lower_bounds_follow_current_model_and_optimizer_membership() -> None:
    runtime = runner._import_runtime()
    torch = runtime.torch
    model_module = runtime.model_module
    specifications = {
        "dense_vmap5_all_lanes": ("dense_base", "dense", 5),
        "donor_single": ("donor", "all_eligible", 1),
        "joint_vmap5_all_lanes": ("joint", "selected", 5),
        "router_only_vmap5_all_lanes": ("router_only", "selected", 5),
        "rung_two_single": ("rung_two", "rung_two", 1),
    }
    observed = {}
    for family, (stage, role, lanes) in specifications.items():
        configuration = model_module.rung_two_config() if role == "rung_two" else model_module.rung_one_config(role)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            model = model_module.ModularNeuralMachine(configuration)
        _, _, membership = runner._make_optimizer(model, stage, runtime)
        model_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in model.state_dict().values())
        moment_bytes = sum(2 * int(parameter.numel() * parameter.element_size()) + 4 for name, parameter in model.named_parameters() if membership[name]["requires_grad"])
        observed[family] = lanes * (model_bytes + moment_bytes)
    assert observed == qualification.TAIL_CHECKPOINT_TENSOR_BYTE_LOWER_BOUNDS


def test_packaging_projection_charges_all_six_pilot_tail_details(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qualification, "preflight_detail_count", lambda _: (2, [{"sha256": "a" * 64, "bytes": 1}, {"sha256": "b" * 64, "bytes": 1}]))
    monkeypatch.setattr(qualification, "packaging_line_bounds", lambda *_: {"attempt_max_line_bytes": 1, "train_max_line_bytes": 1, "routing_max_line_bytes": 1, "prediction_max_line_bytes": 1, "evaluation_max_line_bytes": 1})
    monkeypatch.setattr(qualification, "packaging_block", lambda: b"x")
    monkeypatch.setattr(qualification, "measure_packaging_block", lambda *_: (1_000_000_000, True))
    monkeypatch.setattr(qualification, "measure_packaging_files", lambda *_: (32_000_000_000, True))
    _, detail = qualification.measure_packaging(tmp_path, tmp_path, 1, 1)
    assert detail["counts"]["future_pilot_tail_details"] == 6
    assert detail["counts"]["projected_files"] == 260


def test_lifecycle_cleanup_timer_includes_parent_stderr_descriptor_close() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_mlx_resource_pilot")
    body = ast.get_source_segment(source, function)
    assert body is not None
    assert body.index("cleanup_started_ns = time.perf_counter_ns()") < body.index("stderr_handle.close()")
    assert body.index("stderr_handle.close()") < body.index("scratch_cleanup = cleanup_scratch(scratch)")
    assert body.index("scratch_cleanup = cleanup_scratch(scratch)") < body.index("scratch_cleanup_seconds =")


def test_child_command_and_environment_are_exact() -> None:
    command, environment = backend.child_invocation("self-check")
    assert command == [
        "/Users/dttdrv/Projects/Transformerov/.venv/bin/python",
        str(QUALIFICATION_PATH),
        "--child-mode",
        "self-check",
    ]
    assert environment == {
        "HOME": "/Users/dttdrv",
        "LANG": "C",
        "LC_ALL": "C",
        "LOGNAME": "dttdrv",
        "MLX_METAL_DEBUG": "0",
        "OMP_NUM_THREADS": "4",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": "/private/tmp/todorov-mlx-pycache",
        "TMPDIR": "/private/tmp",
        "USER": "dttdrv",
        "VECLIB_MAXIMUM_THREADS": "4",
    }


def test_pilot_child_contract_matches_preregistered_workloads_and_counter_totals() -> None:
    payload = json.loads((PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json").read_text(encoding="utf-8"))
    resource = payload["artifacts"]["schemas"]["resource_row"]
    assert _engine_literal("PILOT_SEED_BASE") == payload["pilot"]["seed_base"] == 9_999_983
    assert _engine_literal("PILOT_WORKLOADS") == (
        ("donor", "donor", "all_eligible", "one_MLX_lane", 1, 16, 128),
        ("selected_vmap5", "joint", "selected", "compiled_MLX_vmap_width_5", 5, 16, 128),
        ("dense_vmap5", "dense_base", "dense", "compiled_MLX_vmap_width_5", 5, 16, 128),
        ("rung_two", "rung_two", "rung_two", "one_MLX_lane", 1, 8, 512),
    )
    assert _engine_literal("PILOT_WARMUP_UPDATES") == tuple(payload["pilot"]["updates"]["warmup_updates"]) == (1, 2, 3)
    assert _engine_literal("PILOT_TIMED_UPDATES") == tuple(payload["pilot"]["updates"]["timed_updates"]) == (4, 5, 6, 7, 8, 9, 10, 11)
    assert _engine_literal("PILOT_FINAL_ATTEMPTED_UPDATES") == resource["pilot_complete_final_values"]["attempted_updates"] == 132
    assert _engine_literal("PILOT_FINAL_TOKEN_POSITIONS") == resource["pilot_complete_final_values"]["token_positions"] == 292_864


def test_parent_pilot_workload_protocol_is_single_sourced_and_matches_engine_and_preregistration() -> None:
    payload = json.loads((PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json").read_text(encoding="utf-8"))
    lanes = {"one_MLX_lane": 1, "compiled_MLX_vmap_width_5": 5}
    preregistered = tuple(
        (record["name"], record["execution"], lanes[record["execution"]], record["batch_size"], record["sequence_length"])
        for record in payload["pilot"]["workloads"]
    )
    engine = tuple(
        (name, execution, lane_count, batch_size, sequence_length)
        for name, _, _, execution, lane_count, batch_size, sequence_length in _engine_literal("PILOT_WORKLOADS")
    )
    protocol = qualification.PILOT_PROTOCOL
    parent = tuple((record.name, record.execution, record.lanes, record.batch_size, record.sequence_length) for record in protocol.workloads)
    updates_per_workload = len(payload["pilot"]["updates"]["warmup_updates"]) + len(payload["pilot"]["updates"]["timed_updates"])
    resource = payload["artifacts"]["schemas"]["resource_row"]["pilot_complete_final_values"]
    assert parent == engine == preregistered
    assert protocol.seed_base == payload["pilot"]["seed_base"] == _engine_literal("PILOT_SEED_BASE")
    assert protocol.seed_stride == _engine_literal("PILOT_SEED_STRIDE") == 100
    assert protocol.data_seed_offset == _engine_literal("PILOT_DATA_SEED_OFFSET") == 1
    assert protocol.route_seed_offset == _engine_literal("PILOT_ROUTE_SEED_OFFSET") == 2
    assert payload["pilot"]["seed_formulas"] == {
        "model_seed": f"{protocol.seed_base}+{protocol.seed_stride}*workload_ordinal",
        "data_seed": f"model_seed+{protocol.data_seed_offset}",
        "route_seed": f"model_seed+{protocol.route_seed_offset}",
    }
    assert protocol.warmup_updates == tuple(payload["pilot"]["updates"]["warmup_updates"]) == _engine_literal("PILOT_WARMUP_UPDATES")
    assert protocol.timed_updates == tuple(payload["pilot"]["updates"]["timed_updates"]) == _engine_literal("PILOT_TIMED_UPDATES")
    assert protocol.all_updates == protocol.warmup_updates + protocol.timed_updates
    assert protocol.updates_per_workload == updates_per_workload == 11
    assert protocol.workload_order == tuple(payload["pilot"]["workload_order"])
    assert protocol.final_attempted_updates == resource["attempted_updates"] == _engine_literal("PILOT_FINAL_ATTEMPTED_UPDATES")
    assert protocol.final_token_positions == resource["token_positions"] == _engine_literal("PILOT_FINAL_TOKEN_POSITIONS")
    assert tuple(protocol.model_seed(index) for index in range(len(protocol.workloads))) == tuple(protocol.seed_base + protocol.seed_stride * index for index in range(len(protocol.workloads)))
    assert tuple(protocol.data_seed(index) for index in range(len(protocol.workloads))) == tuple(protocol.model_seed(index) + protocol.data_seed_offset for index in range(len(protocol.workloads)))
    assert tuple(protocol.route_seed(index) for index in range(len(protocol.workloads))) == tuple(protocol.model_seed(index) + protocol.route_seed_offset for index in range(len(protocol.workloads)))
    assert tuple(protocol.prior_attempts(index) for index in range(len(protocol.workloads))) == (0, 11, 66, 121)
    assert tuple(
        protocol.expected_update(workload_ordinal, protocol.prior_attempts(workload_ordinal) + protocol.workloads[workload_ordinal].lanes * update_ordinal)
        for workload_ordinal in range(len(protocol.workloads))
        for update_ordinal in range(protocol.updates_per_workload)
    ) == protocol.all_updates * len(protocol.workloads)
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    function_nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    functions = {name: ast.get_source_segment(source, node) for name, node in function_nodes.items()}
    combined = "\n".join((functions["validate_pilot_workload_record"] or "", functions["run_mlx_resource_pilot"] or ""))
    numeric_constants = {node.value for name in ("validate_pilot_workload_record", "run_mlx_resource_pilot") for node in ast.walk(function_nodes[name]) if isinstance(node, ast.Constant) and type(node.value) is int}
    for required in ("PILOT_PROTOCOL.workloads", "PILOT_PROTOCOL.model_seed", "PILOT_PROTOCOL.data_seed", "PILOT_PROTOCOL.route_seed", "PILOT_PROTOCOL.expected_update", "PILOT_PROTOCOL.warmup_updates", "PILOT_PROTOCOL.all_updates", "PILOT_PROTOCOL.updates_per_workload", "PILOT_PROTOCOL.final_attempted_updates", "PILOT_PROTOCOL.final_token_positions", "PILOT_PROTOCOL.workload_order"):
        assert required in combined
    assert not {9_999_983, 132, 292_864, 2048, 4096} & numeric_constants
    for forbidden in ("(1, 5, 5, 1)", "(2048, 2048, 2048, 4096)", '["donor", "selected_vmap5", "dense_vmap5", "rung_two"]', "model_seed + 1", "model_seed + 2", "// specification.lanes + 1"):
        assert forbidden not in combined
    producer = ast.get_source_segment(ENGINE_PATH.read_text(encoding="utf-8"), next(node for node in ast.parse(ENGINE_PATH.read_text(encoding="utf-8"), filename=str(ENGINE_PATH)).body if isinstance(node, ast.FunctionDef) and node.name == "pilot_workload"))
    assert producer is not None
    for required in ("PILOT_SEED_STRIDE", "PILOT_DATA_SEED_OFFSET", "PILOT_ROUTE_SEED_OFFSET"):
        assert required in producer
    for forbidden in ("100 * ordinal", "model_seed + 1", "model_seed + 2"):
        assert forbidden not in producer


def test_pilot_child_protocol_is_ack_gated_and_child_writes_no_artifacts() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("pilot", "pilot_workload", "pilot_exchange", "compiled_pilot_step", "pilot_measured_components"):
        assert name in functions
    pilot = ast.get_source_segment(source, functions["pilot"])
    workload = ast.get_source_segment(source, functions["pilot_workload"])
    exchange = ast.get_source_segment(source, functions["pilot_exchange"])
    assert pilot is not None and workload is not None and exchange is not None
    assert '"kind": "pilot_hello"' in pilot
    assert '"kind": "pilot_complete"' in pilot
    assert '"kind": "pilot_update_ready"' in workload
    assert '"kind": "pilot_update_complete"' in workload
    assert '"kind": "pilot_workload_complete"' in workload
    assert "sys.stdin.readline()" in exchange
    assert '"kind": "pilot_update_start_committed"' in exchange
    for forbidden in ("write_canonical_json", "write_text", "write_bytes", ".mkdir(", ".open("):
        assert forbidden not in pilot
        assert forbidden not in workload
        assert forbidden not in exchange


def test_pilot_timing_includes_eval_finite_route_gradient_and_optimizer_audits() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    workload = ast.get_source_segment(source, functions["pilot_workload"])
    functional = ast.get_source_segment(source, functions["functional_forward"])
    routed = ast.get_source_segment(source, functions["functional_routed"])
    compiled = ast.get_source_segment(source, functions["compiled_pilot_step"])
    batch = ast.get_source_segment(source, functions["pilot_batch"])
    main = ast.get_source_segment(source, functions["main"])
    assert workload is not None and functional is not None and routed is not None and compiled is not None and batch is not None and main is not None
    started = workload.index("time.perf_counter_ns()")
    stopped = workload.index("elapsed_ns = time.perf_counter_ns()")
    for required in ("mx.eval(", "np.asarray(audit_status)", "routing_reduction(", "np.isfinite(np.asarray(gradient_norm))"):
        assert started < workload.index(required) < stopped
    assert "route_override" in functional
    assert "return result[0], result[1], result[2], result[3], result[6]" in compiled
    assert "routing_reduction(output[4]" in workload
    claim_compiled = ast.get_source_segment(source, functions["compiled_stage_step"])
    optimizer = ast.get_source_segment(source, functions["batched_optimizer_step"])
    assert claim_compiled is not None and optimizer is not None
    assert "lane_step" not in claim_compiled
    assert "lane_step" not in compiled
    assert source.count("vectorized_value_and_grad = mx.vmap(value_and_grad") == 2
    assert source.count("batched_optimizer_step(") == 4
    assert "axis=lane_reduction_axes(gradient)" in optimizer
    assert "audit_status = mx.stack" in optimizer
    assert "in_axes=(0, 0, 0, 0, 0, None)" in compiled
    assert "route_lanes" not in batch
    assert 'if arguments == ["pilot"]' in main


def test_pilot_child_measures_evaluation_and_checkpoint_reload_tail_components() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = (
        "pilot_evaluation_fixture",
        "pilot_evaluation_benchmark",
        "pilot_endpoint_replay_benchmark",
        "pilot_checkpoint_reload_benchmark",
        "pilot_merge_tail_benchmarks",
    )
    assert set(required) <= set(functions)
    checkpoint = "\n".join(ast.get_source_segment(source, functions[name]) or "" for name in ("pilot_checkpoint_reload_benchmark", "pilot_checkpoint_reload_once", "pilot_validate_checkpoint_readback", "pilot_fsync_directory"))
    evaluation = "\n".join(ast.get_source_segment(source, functions[name]) or "" for name in ("pilot_evaluation_benchmark", "pilot_evaluation_once"))
    evaluation_once = ast.get_source_segment(source, functions["pilot_evaluation_once"])
    endpoint = "\n".join(ast.get_source_segment(source, functions[name]) or "" for name in ("pilot_endpoint_replay_benchmark", "pilot_endpoint_replay_once"))
    merge = ast.get_source_segment(source, functions["pilot_merge_tail_benchmarks"])
    assert checkpoint is not None and evaluation is not None and evaluation_once is not None and endpoint is not None and merge is not None
    for required_text in ("torch.save", "os.fsync", "hashlib.sha256", "torch.load", "mx.eval"):
        assert required_text in checkpoint
    for required_text in ("torch.inference_mode", "recurrent_knockout", "block4_routed_knockout", "required_source_excluded"):
        assert required_text in evaluation
    assert "model = MlxTorchEvaluationAdapter" in evaluation
    assert "source = model(" in evaluation
    assert "_rung_two_source_prediction" in evaluation
    assert "from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu" in evaluation_once
    for required_text in ("mx.value_and_grad", "torch_total.backward()", "optimizer_first_moment_max_abs"):
        assert required_text in endpoint
    assert "80 * selected[\"route_acquisition\"]" in merge
    assert "880 * selected[\"rung_one_routed\"]" in merge
    assert "80 * selected[\"rung_one_dense\"]" in merge
    assert "32 * selected[\"rung_two\"]" in merge
    assert "26 * selected[\"endpoint_replay\"]" in merge
    assert "5 * selected[\"donor_single\"]" in merge
    assert "2 * selected[\"dense_vmap5_all_lanes\"]" in merge


@pytest.mark.skipif(os.environ.get("TODOROV_RUN_MLX_PILOT") != "1", reason="requires explicit Metal pilot execution")
def test_real_metal_pilot_executes_all_acknowledged_workloads_without_child_writes() -> None:
    command, environment = backend.child_invocation("pilot")
    with tempfile.TemporaryDirectory(prefix="todorov-pilot-run-", dir="/private/tmp") as run_text, tempfile.TemporaryDirectory(prefix="todorov-pilot-scratch-", dir="/private/tmp") as scratch_text:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**environment, "MODULAR_MLX_RUN_ROOT": run_text, "MODULAR_MLX_SCRATCH_ROOT": scratch_text},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        messages = []
        while True:
            raw = process.stdout.readline()
            if not raw:
                break
            message = json.loads(raw)
            messages.append(message)
            if message["kind"] == "pilot_update_ready":
                process.stdin.write(json.dumps({"ack": True, "kind": "pilot_update_start_committed", "workload": message["workload"], "logical_update": message["logical_update"]}, sort_keys=True, separators=(",", ":")) + "\n")
                process.stdin.flush()
            if message["kind"] in {"pilot_complete", "pilot_hard_abort"}:
                if message["kind"] == "pilot_complete":
                    process.stdin.write(json.dumps({"ack": True, "kind": "close_committed"}, sort_keys=True, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                break
        return_code = process.wait(timeout=30)
        stderr = process.stderr.read()
        assert return_code == 0, json.dumps({"terminal": messages[-1] if messages else None, "stderr": stderr}, sort_keys=True)
        assert [message["kind"] for message in messages].count("pilot_hello") == 1
        assert [message["kind"] for message in messages].count("pilot_workload_started") == 4
        assert [message["kind"] for message in messages].count("pilot_update_ready") == 44
        assert [message["kind"] for message in messages].count("pilot_update_complete") == 44
        assert [message["kind"] for message in messages].count("pilot_workload_complete") == 4
        assert messages[-1]["kind"] == "pilot_complete"
        assert messages[-1]["attempted_updates"] == 132
        assert messages[-1]["token_positions"] == 292_864
        workload_records = [message["record"] for message in messages if message["kind"] == "pilot_workload_complete"]
        assert [record["workload"] for record in workload_records] == [specification[0] for specification in _engine_literal("PILOT_WORKLOADS")]
        assert all(len(record["warmup_update_ns"]) == 3 and len(record["timed_update_ns"]) == 8 for record in workload_records)
        assert all(all(type(value) is int and value > 0 for value in (*record["warmup_update_ns"], *record["timed_update_ns"])) for record in workload_records)
        components = messages[-1]["measured_components"]
        assert set(components) == {"donor_step_seconds", "selected_vmap5_step_seconds", "dense_vmap5_step_seconds", "rung_two_step_seconds", "cold_compile_seconds", "evaluation_seconds", "checkpoint_reload_seconds"}
        for record in workload_records:
            assert components[f"{record['workload']}_step_seconds"] == pytest.approx(sum(record["timed_update_ns"]) / 8_000_000_000)
        assert components["cold_compile_seconds"] == pytest.approx(sum(record["warmup_update_ns"][0] for record in workload_records) / 1_000_000_000)
        tail = messages[-1]["tail_benchmarks"]
        assert set(tail) == {"evaluation", "checkpoint_reload"}
        assert tail["evaluation"]["component_seconds"] == components["evaluation_seconds"]
        assert tail["checkpoint_reload"]["component_seconds"] == components["checkpoint_reload_seconds"]
        assert tail["checkpoint_reload"]["byte_sizes"]["projected_checkpoint_bytes"] > 0
        assert tail["evaluation"]["scratch_cleanup_pass"] is True
        assert tail["checkpoint_reload"]["scratch_cleanup_pass"] is True
        assert list(Path(run_text).iterdir()) == []
        assert list(Path(scratch_text).iterdir()) == []
        print(json.dumps({"terminal": messages[-1], "workloads": workload_records}, sort_keys=True, separators=(",", ":")))


@pytest.mark.skipif(os.environ.get("TODOROV_RUN_MLX_SERVE_HELLO") != "1", reason="requires explicit Metal serve execution")
def test_real_metal_serve_hello_is_inline_and_run_root_stays_clean() -> None:
    command, environment = backend.child_invocation("serve")
    with tempfile.TemporaryDirectory(prefix="todorov-serve-run-", dir="/private/tmp") as run_text, tempfile.TemporaryDirectory(prefix="todorov-serve-scratch-", dir="/private/tmp") as scratch_text:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**environment, "MODULAR_MLX_RUN_ROOT": run_text, "MODULAR_MLX_SCRATCH_ROOT": scratch_text},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        hello = json.loads(process.stdout.readline())
        state = backend.protocol_state()
        assert backend.validate_child_message(hello, state) == "hello"
        assert hello["self_check"]["pass"] is True
        assert hello["self_check"]["device"] == "Device(gpu, 0)"
        assert list(Path(run_text).iterdir()) == []
        assert list(Path(scratch_text).iterdir()) == []
        process.terminate()
        process.wait(timeout=10)


def test_protocol_rejects_unknown_keys_and_invalid_transitions() -> None:
    state = backend.protocol_state()
    hello = _hello()
    assert backend.validate_child_message(hello, state) == "hello"
    with pytest.raises(backend.MlxProtocolError):
        backend.validate_child_message({**hello, "sequence": 1, "extra": True}, state)
    with pytest.raises(backend.MlxProtocolError):
        backend.validate_child_message(
            {
                "kind": "stage_complete",
                "sequence": 1,
                "stage": "donor",
                "construction_seeds": [11],
                "checkpoint_paths": ["rung1/11/checkpoints/donor_last.pt"],
                "checkpoint_sha256s": ["b" * 64],
                "optimizer_state_sha256s": ["c" * 64],
                "rng_state_sha256s": ["d" * 64],
            },
            state,
        )


def test_protocol_binds_started_and_completed_messages_to_validated_request() -> None:
    state = backend.protocol_state()
    backend.validate_child_message(_hello(), state)
    request = backend.stage_request("joint")
    backend.bind_stage_request(state, request)
    assert backend.validate_child_message(
        {
            "kind": "stage_started",
            "sequence": 1,
            "stage": "joint",
            "construction_seeds": [11, 23, 37, 53, 71],
        },
        state,
    ) == "stage_started"
    state["completed_updates"] = 512
    wrong = {
        "kind": "stage_complete",
        "sequence": 2,
        "stage": "joint",
        "construction_seeds": [11, 23, 37, 53, 71],
        "checkpoint_paths": [f"rung1/{seed}/checkpoints/final_last.pt" for seed in (11, 23, 37, 53, 71)],
        "checkpoint_sha256s": ["b" * 64] * 5,
        "optimizer_state_sha256s": ["c" * 64] * 5,
        "rng_state_sha256s": ["d" * 64] * 5,
    }
    wrong["checkpoint_paths"][3] = "rung1/53/checkpoints/router_last.pt"
    with pytest.raises(backend.MlxProtocolError, match="checkpoint paths"):
        backend.validate_child_message(wrong, state)


def test_protocol_requires_durable_update_pair_before_each_compute_completion() -> None:
    state = backend.protocol_state()
    backend.validate_child_message(_hello(), state)
    request = backend.stage_request("joint")
    backend.bind_stage_request(state, request)
    backend.validate_child_message({"kind": "stage_started", "sequence": 1, "stage": "joint", "construction_seeds": [11, 23, 37, 53, 71]}, state)
    hashes = [character * 64 for character in "abcde"]
    ready = {
        "kind": "update_ready",
        "sequence": 2,
        "stage": "joint",
        "construction_seeds": [11, 23, 37, 53, 71],
        "logical_update": 1,
        "batch_sha256s": hashes,
        "token_positions": [2048] * 5,
    }
    assert backend.validate_child_message(ready, state) == "update_ready"
    with pytest.raises(backend.MlxProtocolError):
        backend.validate_child_message({"kind": "closed", "sequence": 3, "status": "clean_complete"}, state)
    metrics = [{"total_loss": 1.0, "task_loss": 0.8, "internal_router_loss": 0.2, "supervised_route_loss": 0.3, "gradient_norm": 0.4, "clip_result": "unchanged", "raw_overflow_count": 0, "max_bucket_load": 2, "elapsed_seconds": 0.03, "finite": True} for _ in range(5)]
    complete = {
        "kind": "update_complete",
        "sequence": 3,
        "stage": "joint",
        "construction_seeds": [11, 23, 37, 53, 71],
        "logical_update": 1,
        "batch_sha256s": hashes,
        "metrics": metrics,
        "mx_eval_complete": True,
        "memory": {"active_memory_bytes": 1, "cache_memory_bytes": 2, "peak_memory_bytes": 3, "parent_rss_and_swap_required": True},
    }
    assert backend.validate_child_message(complete, state) == "update_complete"
    with pytest.raises(backend.MlxProtocolError, match="update count"):
        backend.validate_child_message(
            {
                "kind": "stage_complete",
                "sequence": 4,
                "stage": "joint",
                "construction_seeds": [11, 23, 37, 53, 71],
                "checkpoint_paths": request["checkpoint_outputs"],
                "checkpoint_sha256s": ["f" * 64] * 5,
                "optimizer_state_sha256s": ["1" * 64] * 5,
                "rng_state_sha256s": ["2" * 64] * 5,
            },
            state,
        )


def test_protocol_accepts_canonical_hard_abort_from_any_active_stage() -> None:
    state = backend.protocol_state()
    backend.validate_child_message(_hello(), state)
    request = backend.stage_request("joint")
    backend.bind_stage_request(state, request)
    backend.validate_child_message({"kind": "stage_started", "sequence": 1, "stage": "joint", "construction_seeds": [11, 23, 37, 53, 71]}, state)
    message = {
        "kind": "hard_abort",
        "sequence": 2,
        "reason": "artifact_inconsistency",
        "error_type": "MlxEngineError",
        "message": "injected transport failure",
        "memory": {"active_memory_bytes": 1, "cache_memory_bytes": 2, "peak_memory_bytes": 3, "parent_rss_and_swap_required": True},
    }
    assert backend.validate_child_message(message, state) == "hard_abort"
    assert state["closed"] is True
    assert state["aborted"] is True
    with pytest.raises(backend.MlxProtocolError, match="state"):
        backend.validate_child_message(message, state)


def test_protocol_requires_exact_evaluation_cardinalities_before_clean_close() -> None:
    state = backend.protocol_state()
    backend.validate_child_message(_hello(), state)
    rung_one = [{"seed": seed, "evaluation_rows": 65, "prediction_rows": 6144, "state_rows": 1980, "intervention_rows": 108, "forward_sequence": 4048} for seed in backend.RUNG_ONE_SEEDS]
    rung_two = {"evaluation_rows": 2, "prediction_rows": 1024, "state_rows": 230, "intervention_rows": 14, "gate_conditions": 2, "source_telemetry_max_error": 0.0}
    assert backend.validate_child_message({"kind": "evaluation_complete", "sequence": 1, "result": {"rung_one": rung_one, "rung_two": rung_two}}, state) == "evaluation_complete"
    assert backend.validate_child_message({"kind": "closed", "sequence": 2, "status": "clean_complete"}, state) == "closed"


def test_engine_source_is_separate_and_has_no_forbidden_code_text() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    assert any(isinstance(node, ast.ClassDef) and node.name == "MlxModularModel" for node in tree.body)
    assert "TODO" not in source
    assert "#" not in source
    assert '"""' not in source
    assert "'''" not in source
    assert "import torch" in source
    assert "import mlx.core as mx" in source
    assert "mx.vmap" in source
    assert "mx.compile" in source
    assert "mx.eval" in source


def test_stage_request_requires_independent_seed_state_and_exact_stage_shape() -> None:
    request = {
        "schema_version": "todorov.modular-mlx-ipc.1",
        "kind": "run_stage",
        "sequence": 7,
        "stage": "joint",
        "construction_seeds": [11, 23, 37, 53, 71],
        "data_generator_seeds": [300011, 300023, 300037, 300053, 300071],
        "updates": 512,
        "warmup_updates": 32,
        "batch_size": 16,
        "checkpoint_inputs": [f"rung1/{seed}/checkpoints/router_last.pt" for seed in (11, 23, 37, 53, 71)],
        "checkpoint_outputs": [f"rung1/{seed}/checkpoints/final_last.pt" for seed in (11, 23, 37, 53, 71)],
    }
    validated = backend.validate_stage_request(request)
    assert validated["stage"] == "joint"
    assert len(set(validated["construction_seeds"])) == 5
    assert len(set(validated["data_generator_seeds"])) == 5
    bad = json.loads(json.dumps(request))
    bad["data_generator_seeds"][4] = bad["data_generator_seeds"][0]
    with pytest.raises(backend.MlxProtocolError):
        backend.validate_stage_request(bad)


@pytest.mark.parametrize(
    ("stage", "construction_seeds", "data_seeds", "inputs", "outputs"),
    (
        ("donor", [37], [100037], [], ["rung1/37/checkpoints/donor_last.pt"]),
        ("router_only", [11, 23, 37, 53, 71], [200011, 200023, 200037, 200053, 200071], [f"rung1/{seed}/checkpoints/donor_last.pt" for seed in (11, 23, 37, 53, 71)], [f"rung1/{seed}/checkpoints/router_last.pt" for seed in (11, 23, 37, 53, 71)]),
        ("joint", [11, 23, 37, 53, 71], [300011, 300023, 300037, 300053, 300071], [f"rung1/{seed}/checkpoints/router_last.pt" for seed in (11, 23, 37, 53, 71)], [f"rung1/{seed}/checkpoints/final_last.pt" for seed in (11, 23, 37, 53, 71)]),
        ("dense_base", [11, 23, 37, 53, 71], [100011, 100023, 100037, 100053, 100071], [], [f"rung1/{seed}/checkpoints/dense_base_last.pt" for seed in (11, 23, 37, 53, 71)]),
        ("dense_continuation", [11, 23, 37, 53, 71], [300011, 300023, 300037, 300053, 300071], [f"rung1/{seed}/checkpoints/dense_base_last.pt" for seed in (11, 23, 37, 53, 71)], [f"rung1/{seed}/checkpoints/dense_last.pt" for seed in (11, 23, 37, 53, 71)]),
        ("rung_two", [83], [900083], [], ["rung2/83/checkpoints/final_last.pt"]),
    ),
)
def test_stage_request_has_exact_seed_and_checkpoint_identity(stage, construction_seeds, data_seeds, inputs, outputs) -> None:
    request = backend.stage_request(stage, construction_seeds if stage == "donor" else None)
    assert request["construction_seeds"] == construction_seeds
    assert request["data_generator_seeds"] == data_seeds
    assert request["checkpoint_inputs"] == inputs
    assert request["checkpoint_outputs"] == outputs
    bad = dict(request)
    bad["checkpoint_outputs"] = ["../escape.pt"] * len(outputs)
    with pytest.raises(backend.MlxProtocolError):
        backend.validate_stage_request(bad)


def test_parameter_mapping_and_optimizer_policy_are_explicit() -> None:
    assert backend.mapped_mlx_parameter_name("blocks.4.mix.source_mixer.attention.qkv.weight") == "blocks.4.mix.qkv.weight"
    assert backend.mapped_mlx_parameter_name("blocks.4.mix.source_mixer.attention.router.codebooks") == "blocks.4.mix.codebooks"
    assert backend.mapped_mlx_parameter_name("blocks.1.mix.q.weight") == "blocks.1.mix.q.weight"
    manifest = backend.validate_parameter_mapping(
        {
            "blocks.4.mix.source_mixer.attention.qkv.weight": {"shape": [192, 64], "dtype": "torch.float32"},
            "blocks.4.mix.source_mixer.attention.router.codebooks": {"shape": [2, 4, 8], "dtype": "torch.float32"},
        },
        {
            "blocks.4.mix.qkv.weight": {"shape": [192, 64], "dtype": "float32"},
            "blocks.4.mix.codebooks": {"shape": [2, 4, 8], "dtype": "float32"},
        },
    )
    assert manifest["transpose"] is False
    assert manifest["bijective"] is True
    assert backend.optimizer_parameter_policy("embed.weight", "joint") == {"trainable": True, "peak_lr": 0.00025, "weight_decay": 0.01}
    assert backend.optimizer_parameter_policy("blocks.4.mix.source_mixer.attention.router.codebooks", "joint") == {"trainable": True, "peak_lr": 0.001, "weight_decay": 0.0}
    assert backend.optimizer_parameter_policy("blocks.0.n1.weight", "joint") == {"trainable": True, "peak_lr": 0.00025, "weight_decay": 0.0}
    assert backend.optimizer_contract()["fresh_state_each_stage"] is True
    assert backend.optimizer_contract()["bias_correction"] is True


def test_mlx_optimizer_policy_matches_cpu_membership_for_every_stage_and_name() -> None:
    from src.model.modular_neural_machine import ModularNeuralMachine, rung_one_config, rung_two_config

    runtime = runner._import_runtime()
    roles = {"donor": "all_eligible", "router_only": "selected", "joint": "selected", "dense_base": "dense", "dense_continuation": "dense"}
    for stage in ("donor", "router_only", "joint", "dense_base", "dense_continuation", "rung_two"):
        model = ModularNeuralMachine(rung_two_config() if stage == "rung_two" else rung_one_config(roles[stage]))
        _, _, membership = runner._make_optimizer(model, stage, runtime)
        for name, record in membership.items():
            assert backend.optimizer_parameter_policy(name, stage) == {
                "trainable": record["requires_grad"],
                "peak_lr": record["peak_lr"],
                "weight_decay": record["weight_decay"],
            }


def _vmap_attempt_batch(event_sequence: int, event: str, logical_update: int = 1):
    records = []
    for index, seed in enumerate(backend.RUNG_ONE_SEEDS):
        metrics = None
        if event == "completed":
            metrics = {
                "learning_rates": [{"parameter_group": "base", "learning_rate": 0.001}],
                "component_losses": {"task_loss": 1.0, "internal_router_loss": 0.1, "supervised_route_loss": 0.2},
                "total_loss": 1.3,
                "gradient_norm": 0.5,
                "clip_result": "unchanged",
                "raw_overflow_count": 0,
                "max_bucket_load": 2,
                "elapsed_seconds": 0.1,
                "finite": True,
            }
        records.append(
            runner._attempt_event(
                "test-vmap-run",
                1,
                seed,
                event_sequence,
                event,
                "selected",
                "joint",
                logical_update,
                16,
                2048,
                f"{index + 1:064x}",
                metrics,
            )
        )
    return records


def test_vmap_attempt_ledger_commits_five_rows_with_one_write_and_fsync(tmp_path, monkeypatch) -> None:
    ledger = backend.AtomicVmapAttemptLedger(tmp_path / "vmap_attempts.jsonl", runner.validate_attempt_row)
    ledger.precreate()
    writes = []
    fsyncs = []
    real_write = backend.os.write
    real_fsync = backend.os.fsync
    monkeypatch.setattr(backend.os, "write", lambda descriptor, raw: writes.append(len(raw)) or real_write(descriptor, raw))
    monkeypatch.setattr(backend.os, "fsync", lambda descriptor: fsyncs.append(descriptor) or real_fsync(descriptor))
    started = ledger.append_batch(_vmap_attempt_batch(0, "started"))
    completed = ledger.append_batch(_vmap_attempt_batch(1, "completed"))
    rows = ledger.validate_prefix()
    ledger.close()
    assert started["row_count"] == 5
    assert completed["row_count"] == 5
    assert len(writes) == 2
    assert len(fsyncs) == 2
    assert [row["event"] for row in rows] == ["started"] * 5 + ["completed"] * 5
    assert [row["construction_seed"] for row in rows[:5]] == list(backend.RUNG_ONE_SEEDS)


@pytest.mark.parametrize("fault", ("before_write", "short_write", "before_fsync", "fsync", "readback"))
def test_vmap_attempt_ledger_fault_restores_exact_committed_prefix(tmp_path, fault) -> None:
    ledger = backend.AtomicVmapAttemptLedger(tmp_path / "vmap_attempts.jsonl", runner.validate_attempt_row)
    ledger.precreate()
    committed = _vmap_attempt_batch(0, "started")
    ledger.append_batch(committed)
    before = (tmp_path / "vmap_attempts.jsonl").read_bytes()
    with pytest.raises(backend.MlxBatchLedgerError, match="not committed"):
        ledger.append_batch(_vmap_attempt_batch(1, "completed"), fault=fault)
    after = (tmp_path / "vmap_attempts.jsonl").read_bytes()
    assert after == before
    assert ledger.validate_prefix() == committed
    ledger.close()


def test_vmap_attempt_ledger_rejects_lane_identity_reuse(tmp_path) -> None:
    ledger = backend.AtomicVmapAttemptLedger(tmp_path / "vmap_attempts.jsonl", runner.validate_attempt_row)
    ledger.precreate()
    rows = _vmap_attempt_batch(0, "started")
    rows[4] = dict(rows[0])
    with pytest.raises(backend.MlxBatchLedgerError, match="seed order"):
        ledger.append_batch(rows)
    ledger.close()


def test_evaluation_contract_closes_all_roles_interventions_and_artifacts() -> None:
    contract = backend.evaluation_contract()
    assert contract["rung_one_conditions"] == [
        "intact",
        "target_forced",
        "recurrent_knockout",
        "carry_reset",
        "carry_shuffle",
        "matched_random_route",
        "block4_routed_knockout",
        "block4_local_only",
        "required_source_excluded",
        "all_eligible_donor",
        "all_eligible_clone",
        "dense_causal",
    ]
    assert contract["rung_two_conditions"] == ["intact", "recurrent_knockout"]
    assert contract["rung_one_examples_per_seed"] == 512
    assert contract["rung_two_examples"] == 512
    assert contract["evaluation_batch_size"] == 32
    assert contract["routing_evidence_rows"] == 588_240
    assert contract["routing_compression"] == {"format": "gzip", "level": 9, "mtime": 0}
    assert contract["artifacts"] == ["evaluation", "predictions", "state_statistics", "intervention_deltas", "routing_evidence"]
    assert contract["torch_authoritative_gate_replay"] is True


def test_ordered_background_writer_preserves_order_and_bounds_pending_work() -> None:
    observed = []
    writer = backend.OrderedBackgroundWriter(observed.append, 3)
    for value in range(12):
        writer.submit(value)
        assert writer.pending_count <= 3
    writer.close()
    assert observed == list(range(12))


def test_modular_mlx_tests_use_no_scheduler_sleeps() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    sleep_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr == "sleep"
    ]
    assert sleep_calls == []


def test_ordered_background_writer_propagates_worker_failure() -> None:
    observed = []

    def consume(value):
        if value == 4:
            raise RuntimeError("injected routing writer failure")
        observed.append(value)

    writer = backend.OrderedBackgroundWriter(consume, 2)
    with pytest.raises(backend.MlxBackgroundWriterError, match="routing writer failed"):
        for value in range(8):
            writer.submit(value)
        writer.close()
    assert observed == [0, 1, 2, 3]


def test_ordered_background_writer_abort_quiesces_pending_work() -> None:
    observed = []
    entered = threading.Event()
    release = threading.Event()

    def consume(value):
        if value == 0:
            entered.set()
            assert release.wait(1.0)
        observed.append(value)

    writer = backend.OrderedBackgroundWriter(consume, 17)
    for value in range(16):
        writer.submit(value)
    assert entered.wait(1.0)
    cancelled = threading.Event()
    writer.pending[1].add_done_callback(lambda future: cancelled.set() if future.cancelled() else None)
    abort_thread = threading.Thread(target=writer.abort)
    abort_thread.start()
    assert cancelled.wait(1.0)
    release.set()
    abort_thread.join(1.0)
    assert not abort_thread.is_alive()
    assert writer.closed is True
    assert writer.pending_count == 0
    assert observed == [0]
    with pytest.raises(backend.MlxBackgroundWriterError, match="closed"):
        writer.submit(17)


def test_execute_stage_quiesces_training_route_writer_on_every_failure_path() -> None:
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_stage")
    guarded = [node for node in ast.walk(function) if isinstance(node, ast.Try) and node.finalbody]
    assert len(guarded) == 1
    calls = [node for statement in guarded[0].finalbody for node in ast.walk(statement) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "abort" for call in calls)


def test_gradient_audit_uses_one_lane_preserving_device_transfer_per_update() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    compiled = ast.get_source_segment(source, functions["compiled_stage_step"])
    optimizer = ast.get_source_segment(source, functions["batched_optimizer_step"])
    update = ast.get_source_segment(source, functions["update_stage_audits"])
    execute = ast.get_source_segment(source, functions["execute_stage"])
    assert compiled is not None and optimizer is not None and "batched_optimizer_step(" in compiled and "audit_status = mx.stack" in optimizer
    assert update is not None and ".item()" not in update
    assert execute is not None and execute.count("np.asarray(audit_status)") == 1


def test_gradient_audit_writes_only_canonical_endpoint_artifacts() -> None:
    engine_source = ENGINE_PATH.read_text(encoding="utf-8")
    qualification_source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    assert "gradient_parts" not in engine_source
    assert "gradient_parts" not in qualification_source
    assert "write_qualification_gradient_artifacts" in engine_source
    assert 'seed_root / "grad_audit.json"' in engine_source
    assert 'seed_root / "dense_grad_audit.json"' in engine_source


def test_trained_endpoint_parity_is_captured_before_stage_completion_and_closed_at_26_records() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "trained_endpoint_parity" in functions
    assert "validate_trained_endpoint_parity_record" in functions
    assert "validate_trained_endpoint_parity_records" in functions
    execute = ast.get_source_segment(source, functions["execute_stage"])
    serve = ast.get_source_segment(source, functions["serve"])
    assert execute is not None and serve is not None
    assert execute.index("endpoint_parity_records.extend(records)") < execute.index('"kind": "stage_complete"')
    assert "validate_trained_endpoint_parity_records(endpoint_parity_records, 26)" in serve
    assert "mlx_endpoint_parity.jsonl" not in source


def test_trained_endpoint_parity_covers_every_required_numeric_surface() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "trained_endpoint_parity")
    body = ast.get_source_segment(source, function)
    assert body is not None
    for required in (
        "checkpoint_sha256",
        "comparison_positions",
        "parameter_max_abs",
        "logits_max_abs",
        "hidden_max_abs",
        "sequence_delta_max_abs",
        "full_tensor_logits_max_abs",
        "full_tensor_hidden_max_abs",
        "full_tensor_sequence_delta_max_abs",
        "total_loss_max_abs",
        "gradient_max_abs",
        "optimizer_first_moment_max_abs",
        "optimizer_second_moment_max_abs",
        "raw_route_exact",
        "effective_route_exact",
        "address_route_exact",
    ):
        assert required in body
    assert '"comparison_tolerance": 1e-5' in body
    assert '"logit_loss_gradient_tolerance": 1e-5' in body
    assert '"optimizer_tolerance": 0.0' in body
    assert 'checkpoint_record["optimizer_state_sha256"]' in body
    evaluation_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_rung_two_qualification")
    evaluation_body = ast.get_source_segment(source, evaluation_function)
    assert evaluation_body is not None
    assert "checkpoint_record" not in evaluation_body


def test_gradient_comparison_evidence_uses_conjunctive_evidence_bound_fallback() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    statistics = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tensor_comparison_statistics")
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gradient_comparison_evidence")
    namespace = {"Any": Any, "MlxEngineError": RuntimeError, "math": __import__("math"), "np": np}
    exec(compile(ast.Module(body=[statistics, function], type_ignores=[]), str(ENGINE_PATH), "exec"), namespace)
    evidence = namespace["gradient_comparison_evidence"](
        [("blocks.2.mix.k.weight", np.array([0.24644289910793304, 0.24646177887916565, 0.24646177887916565, 0.24646177887916565]), np.array([0.24646177887916565] * 4))]
    )
    assert evidence["gradient_max_abs"] == pytest.approx(1.887977123260498e-5)
    assert evidence["gradient_absolute_pass"] is False
    assert evidence["gradient_relative_max"] < 1e-4
    assert evidence["gradient_normalized_l2_max"] < 5e-5
    assert evidence["gradient_cosine_min"] > 0.999999999
    assert evidence["gradient_scale_aware_pass"] is True
    assert evidence["gradient_pass"] is True
    assert evidence["gradient_worst_tensor"] == "blocks.2.mix.k.weight"
    assert evidence["gradient_worst_index"] == [0]
    opposite = namespace["gradient_comparison_evidence"](
        [("direction", np.array([-1.0]), np.array([1.0]))]
    )
    assert opposite["gradient_absolute_pass"] is False
    assert opposite["gradient_scale_aware_pass"] is False
    assert opposite["gradient_pass"] is False
    fallback_absolute = namespace["gradient_comparison_evidence"](
        [("fallback_absolute", np.array([0.999969, 1.0, 1.0, 1.0]), np.ones(4))]
    )
    assert fallback_absolute["gradient_relative_max"] < 1e-4
    assert fallback_absolute["gradient_normalized_l2_max"] < 5e-5
    assert fallback_absolute["gradient_cosine_min"] > 0.999999999
    assert fallback_absolute["gradient_scale_aware_pass"] is False
    relative = namespace["gradient_comparison_evidence"](
        [("relative", np.concatenate((np.array([0.09998]), np.full(99, 0.1))), np.full(100, 0.1))]
    )
    assert relative["gradient_max_abs"] < 3e-5
    assert relative["gradient_relative_max"] > 1e-4
    assert relative["gradient_normalized_l2_max"] < 5e-5
    assert relative["gradient_scale_aware_pass"] is False
    normalized = namespace["gradient_comparison_evidence"](
        [("normalized", np.array([0.999971, 0.009971, 0.009971, 0.009971]), np.array([1.0, 0.01, 0.01, 0.01]))]
    )
    assert normalized["gradient_max_abs"] <= 3e-5
    assert normalized["gradient_relative_max"] < 1e-4
    assert normalized["gradient_normalized_l2_max"] > 5e-5
    assert normalized["gradient_scale_aware_pass"] is False
    cosine = namespace["gradient_comparison_evidence"](
        [("sparse_large_error", np.array([1.0, 2.9e-5, 2.9e-5, 1.8e-5]), np.array([1.0, 0.0, 0.0, 0.0]))]
    )
    assert cosine["gradient_max_abs"] <= 3e-5
    assert cosine["gradient_relative_max"] < 1e-4
    assert cosine["gradient_normalized_l2_max"] < 5e-5
    assert cosine["gradient_cosine_min"] < 0.999999999
    assert cosine["gradient_scale_aware_pass"] is False


def test_trained_endpoint_parity_schema_records_absolute_and_scale_aware_gradient_evidence() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    functions = {node.name: ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef)}
    trained = functions["trained_endpoint_parity"]
    validator = functions["validate_trained_endpoint_parity_record"]
    for required in (
        "gradient_relative_max",
        "gradient_normalized_l2_max",
        "gradient_cosine_min",
        "gradient_worst_tensor",
        "gradient_worst_index",
        "gradient_worst_observed",
        "gradient_worst_expected",
        "gradient_absolute_pass",
        "gradient_scale_aware_pass",
        "gradient_pass",
        "gradient_relative_tolerance",
        "gradient_normalized_l2_tolerance",
        "gradient_cosine_tolerance",
        "gradient_scale_aware_absolute_tolerance",
        "loss_tolerance",
    ):
        assert required in trained
        assert required in validator
    assert '"gradient_relative_tolerance": 1e-4' in trained
    assert '"gradient_normalized_l2_tolerance": 5e-5' in trained
    assert '"gradient_cosine_tolerance": 0.999999999' in trained
    assert '"gradient_scale_aware_absolute_tolerance": 3e-5' in trained
    assert '"loss_tolerance": 1e-6' in trained
    assert 'gradient_absolute_pass or gradient_scale_aware_pass' in functions["gradient_comparison_evidence"]


def test_actual_vmap_probe_consumes_complete_compiled_step_result() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "actual_model_vmap5_probe")
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "runner"]
    assert len(assignments) == 1
    target = assignments[0].targets[0]
    assert isinstance(target, ast.Tuple)
    assert len(target.elts) == 8
    body = ast.get_source_segment(source, function)
    assert body is not None
    for required in (
        "cpu._training_forward_loss",
        "cpu._clip_gradient_norm_finite",
        "optimizer.step",
        "np.array(observed)[lane]",
        "clipped_gradients",
        "optimizer_gradient_sha256",
        "end_to_end_worst_tensor",
        "torch_loss_max_abs",
        "torch_parameter_max_abs",
        "torch_first_moment_max_abs",
        "torch_second_moment_max_abs",
        "torch_route_exact",
    ):
        assert required in body
    assert "np.array(observed[lane])" not in body
    self_check = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "self_check")
    self_check_body = ast.get_source_segment(source, self_check)
    assert self_check_body is not None
    assert 'full_gradient_parity("selected", 3123, 4123, 2)' in self_check_body
    assert '"full_gradient_parity": gradient' in self_check_body


def test_dense_functional_forward_exposes_block_zero_routes_for_compiled_stage_tuple() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "functional_forward")
    body = ast.get_source_segment(source, function)
    assert body is not None
    assert "elif query_route is None:" in body
    assert "query_route = current_query" in body
    assert "key_route = current_key" in body


def test_durable_gzip_prefix_fsyncs_reads_back_and_remains_appendable(tmp_path) -> None:
    stream = runner.CanonicalGzipStream(tmp_path / "routing.jsonl.gz")
    stream.open()
    stream.write({"value": 1})
    first = backend.durable_gzip_prefix(stream)
    assert first["committed_bytes"] == (tmp_path / "routing.jsonl.gz").stat().st_size
    assert first["sha256"] == runner.sha256_file(tmp_path / "routing.jsonl.gz")
    stream.write({"value": 2})
    second = backend.durable_gzip_prefix(stream)
    assert second["committed_bytes"] > first["committed_bytes"]
    stream.close()
    import gzip

    with gzip.open(tmp_path / "routing.jsonl.gz", "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == [{"value": 1}, {"value": 2}]


def test_serve_wires_training_routes_and_validates_full_cardinality_before_close() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    serve = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "serve")
    body = ast.get_source_segment(source, serve)
    assert body is not None
    assert "execute_stage(validated, run_root, exchange, child_sequence, selected_streams, selected_sequences, endpoint_parity_records)" in source
    assert "training_rows != 522240 or evaluation_rows != 66000 or routing_rows != 588240" in source
    assert "validate_closed_training_gzip(routing_streams[construction_seed].path" in source
    assert "routing_parts = {seed: [] for seed in RUNG_ONE_SEEDS}" in source
    assert '"self_check": preflight' in body
    assert '"self_check_sha256": hashlib.sha256' in body
    assert "MODULAR_MLX_SCRATCH_ROOT" in body
    assert 'scratch_root / "routing_parts"' in body
    assert 'run_root / "run" / "routing_parts"' not in body
    assert "mlx_self_check.json" not in body
    assert "write_canonical_json(preflight" not in body


def test_production_driver_delegates_to_canonical_parent_lifecycle_callbacks() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "run" not in functions
    assert {"preflight_mlx_probe", "run_mlx_resource_pilot", "run_mlx_claim", "main"} <= set(functions)
    main = ast.get_source_segment(source, functions["main"])
    assert main is not None
    for required in (
        "cpu.parse_cli",
        "cpu.validate_entry_environment",
        "cpu.validate_run_root",
        "cpu.load_prereg_payload",
        "cpu._import_runtime",
        "cpu.configure_torch",
        "cpu.execute_run",
        "resource_pilot_runner=run_mlx_resource_pilot",
        "claim_runner=run_mlx_claim",
        "trained_backend_probe=preflight_mlx_probe",
    ):
        assert required in main
    for forbidden in ("write_resource_artifact", "write_qualification_manifest", "write_abort_closure", "write_orphan_closure"):
        assert forbidden not in source
    assert "TODO" not in source
    assert "#" not in source


def test_claim_callback_returns_canonical_lifecycle_accounting_without_terminal_packaging() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_mlx_claim")
    body = ast.get_source_segment(source, function)
    assert body is not None
    for required in (
        "QualificationResourceSampler",
        "final_claim_guard",
        "_finalize_seed_resource_references",
        "validate_parent_ledger_accounting",
        '"accounting": accounting',
        '"resource_final_sample_id"',
        '"resource_sampling_end_monotonic_ns"',
    ):
        assert required in body
    for forbidden in ("completion.json", "summary.json", "SHA256SUMS", "qualification_manifest"):
        assert forbidden not in body


def test_pilot_and_claim_preserve_primary_failure_when_sampler_cleanup_fails(monkeypatch) -> None:
    events = []

    class Sampler:
        def stop(self):
            events.append("stop")
            raise backend.MlxResourceSamplerError("cleanup failure")

    primary = backend.MlxQualificationError("primary failure")
    monkeypatch.setattr(qualification, "terminate", lambda process: events.append(("terminate", process)))
    qualification.cleanup_after_primary_failure("child", Sampler(), primary)
    assert events == [("terminate", "child"), "stop"]
    assert len(primary.cleanup_failures) == 1
    assert str(primary.cleanup_failures[0]) == "cleanup failure"
    unwind_events = []

    def fail_cleanup():
        unwind_events.append("fail")
        raise OSError("stderr cleanup failure")

    qualification.perform_cleanup_operations((fail_cleanup, lambda: unwind_events.append("scratch")), primary)
    assert unwind_events == ["fail", "scratch"]
    assert [str(error) for error in primary.cleanup_failures] == ["cleanup failure", "stderr cleanup failure"]
    direct_events = []

    def fail_direct():
        direct_events.append("fail")
        raise OSError("direct cleanup failure")

    with pytest.raises(OSError, match="direct cleanup failure"):
        qualification.perform_cleanup_operations((fail_direct, lambda: direct_events.append("scratch")), None)
    assert direct_events == ["fail", "scratch"]
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    functions = {node.name: ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("run_mlx_resource_pilot", "run_mlx_claim"):
        assert "except BaseException as primary_error:" in functions[name]
        assert "cleanup_after_primary_failure(process, sampler, primary_error)" in functions[name]
        assert "unwind_primary = primary_error" in functions[name]
        assert "perform_cleanup_operations(tuple(cleanup_operations), unwind_primary)" in functions[name]
        assert "if scratch.exists()" not in functions[name]
        assert "cleanup_scratch(scratch)" in functions[name]
    assert "cleanup_transport(transport_primary)" in functions["run_mlx_claim"]
    assert functions["run_mlx_claim"].count("cleanup_scratch(scratch)") == 2


def test_cleanup_scratch_exists_and_removal_failures_preserve_primary_and_raise_without_one(tmp_path: Path, monkeypatch) -> None:
    class ExistsFailure:
        def exists(self):
            raise OSError("exists failure")

    primary = backend.MlxQualificationError("primary")
    qualification.perform_cleanup_operations((lambda: qualification.cleanup_scratch(ExistsFailure()),), primary)
    assert [str(error) for error in primary.cleanup_failures] == ["exists failure"]
    with pytest.raises(OSError, match="exists failure"):
        qualification.perform_cleanup_operations((lambda: qualification.cleanup_scratch(ExistsFailure()),), None)
    absent = tmp_path / "absent"
    assert qualification.cleanup_scratch(absent) is True
    observed = tmp_path / "observed"
    observed.mkdir()
    assert qualification.cleanup_scratch(observed) is True
    assert not observed.exists()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(qualification.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("removal failure")))
    removal_primary = backend.MlxQualificationError("primary")
    qualification.perform_cleanup_operations((lambda: qualification.cleanup_scratch(scratch),), removal_primary)
    assert [str(error) for error in removal_primary.cleanup_failures] == ["removal failure"]
    with pytest.raises(OSError, match="removal failure"):
        qualification.perform_cleanup_operations((lambda: qualification.cleanup_scratch(scratch),), None)


def test_pilot_and_claim_cleanup_scratch_when_child_invocation_fails(tmp_path: Path, monkeypatch) -> None:
    scratch_paths = []

    def make_scratch(*args, **kwargs):
        path = tmp_path / f"scratch-{len(scratch_paths)}"
        path.mkdir()
        scratch_paths.append(path)
        return str(path)

    class PilotWriter:
        def validate_committed_prefix(self):
            return [{"attempted_updates": 0, "token_positions": 0}]

    class ClaimAppend:
        acknowledged = True
        reason_code = None

    class ClaimWriter:
        def append(self, row):
            return ClaimAppend()

        def validate_committed_prefix(self):
            return []

    class Transition:
        outcome = "ready"
        swap_baseline_bytes = 0

        def __init__(self, writers):
            self.writers = writers

    monkeypatch.setattr(qualification.tempfile, "mkdtemp", make_scratch)
    monkeypatch.setattr(qualification.backend, "child_invocation", lambda mode: (_ for _ in ()).throw(backend.MlxQualificationError(f"{mode} invocation failure")))
    monkeypatch.setattr(qualification.cpu, "final_frozen_guard", lambda *args: None)
    with pytest.raises(backend.MlxQualificationError, match="pilot invocation failure"):
        qualification.run_mlx_resource_pilot(tmp_path, {}, None, None, Transition({"run/pilot_resources.jsonl": PilotWriter()}))
    assert not scratch_paths[0].exists()
    monkeypatch.setattr(qualification.cpu, "final_claim_guard", lambda *args: None)
    monkeypatch.setattr(qualification.cpu, "_resource_sample", lambda *args: {"sample_id": 0})
    monkeypatch.setattr(qualification.cpu, "claim_resource_observations", lambda rows: [])
    with pytest.raises(backend.MlxQualificationError, match="serve invocation failure"):
        qualification.run_mlx_claim(tmp_path, {}, None, None, Transition({"run/resources.jsonl": ClaimWriter()}), 0)
    assert not scratch_paths[1].exists()


def test_receive_enforces_frozen_line_bound_before_json_parse(monkeypatch) -> None:
    class Stdout:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def fileno(self):
            return self.descriptor

    class Process:
        def __init__(self, descriptor, chunks=()):
            self.stdout = Stdout(descriptor)
            self.chunks = list(chunks)

        def poll(self):
            return None

    processes = {}

    def read(descriptor, size):
        process = processes[descriptor]
        chunk = process.chunks.pop(0)
        process.chunks.insert(0, chunk[size:]) if len(chunk) > size else None
        return chunk[:size]

    monkeypatch.setattr(qualification.os, "read", read)
    monkeypatch.setattr(qualification.select, "select", lambda values, *args: ([values[0]], [], []) if processes[values[0].fileno()].chunks else ([], [], []))
    maximum = qualification.CHILD_MESSAGE_MAX_BYTES
    prefix = b'{"value":"'
    suffix = b'"}\n'
    boundary = prefix + b"a" * (maximum - len(prefix) - len(suffix)) + suffix
    boundary_process = Process(101)
    boundary_process._modular_mlx_stdout_remainder = boundary
    processes[101] = boundary_process
    assert qualification.receive(boundary_process)["value"] == "a" * (maximum - len(prefix) - len(suffix))
    loads_calls = []
    monkeypatch.setattr(qualification.json, "loads", lambda raw: loads_calls.append(raw) or {"value": "parsed"})
    oversized = Process(102)
    oversized._modular_mlx_stdout_remainder = b"a" * (maximum + 1)
    processes[102] = oversized
    with pytest.raises(backend.MlxQualificationError, match="exceeds"):
        qualification.receive(oversized)
    nonterminated = Process(103, (b'{"value":"short"}', b""))
    processes[103] = nonterminated
    with pytest.raises(backend.MlxQualificationError, match="terminated"):
        qualification.receive(nonterminated)
    assert loads_calls == []
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    assert source.count("process.stdout.readline(") == 0
    receive_source = ast.get_source_segment(source, next(node for node in ast.parse(source, filename=str(QUALIFICATION_PATH)).body if isinstance(node, ast.FunctionDef) and node.name == "receive")) or ""
    assert "os.read(" in receive_source
    assert "CHILD_MESSAGE_MAX_BYTES" in receive_source
    assert receive_source.index("len(line) > CHILD_MESSAGE_MAX_BYTES") < receive_source.index("json.loads(raw)")
    assert receive_source.index('line.endswith(b"\\n")') < receive_source.index("json.loads(raw)")


def test_receive_partial_line_honors_deadline_and_preserves_coalesced_remainder(monkeypatch) -> None:
    class Stdout:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def fileno(self):
            return self.descriptor

    class Process:
        def __init__(self, descriptor, chunks):
            self.stdout = Stdout(descriptor)
            self.chunks = list(chunks)

        def poll(self):
            return None

    processes = {}
    read_calls = []

    def read(descriptor, size):
        read_calls.append(descriptor)
        process = processes[descriptor]
        chunk = process.chunks.pop(0)
        return chunk

    def select(values, *args):
        return ([values[0]], [], []) if processes[values[0].fileno()].chunks else ([], [], [])

    monkeypatch.setattr(qualification.os, "read", read)
    monkeypatch.setattr(qualification.select, "select", select)
    deadline_calls = []

    def deadline(deadline_ns):
        deadline_calls.append(deadline_ns)
        if len(deadline_calls) == 2:
            raise backend.MlxBackendRefusal("deadline")
        return 1_000_000_000

    monkeypatch.setattr(qualification.backend, "enforce_deadline", deadline)
    partial = Process(201, (b'{"sequence":',))
    processes[201] = partial
    with pytest.raises(backend.MlxBackendRefusal, match="deadline"):
        qualification.receive(partial, deadline_ns=123)
    assert read_calls == [201]
    coalesced = Process(202, (b'{"sequence":1}\n{"sequence":2}\n',))
    processes[202] = coalesced
    assert qualification.receive(coalesced)["sequence"] == 1
    assert qualification.receive(coalesced)["sequence"] == 2
    assert read_calls == [201, 202]


def test_receive_checks_deadline_and_sampler_before_buffered_line_admission(monkeypatch) -> None:
    class Process:
        stdout = object()

        def __init__(self, remainder):
            self._modular_mlx_stdout_remainder = remainder

        def poll(self):
            return None

    expired = Process(b'{"sequence":1}\n')
    monkeypatch.setattr(qualification.backend, "enforce_deadline", lambda deadline_ns: (_ for _ in ()).throw(backend.MlxBackendRefusal("expired")))
    with pytest.raises(backend.MlxBackendRefusal, match="expired"):
        qualification.receive(expired, deadline_ns=123)
    assert expired._modular_mlx_stdout_remainder == b'{"sequence":1}\n'

    class FailedSampler:
        def raise_if_failed(self):
            raise backend.MlxResourceSamplerError("sampler failed")

    monkeypatch.setattr(qualification.backend, "enforce_deadline", lambda deadline_ns: 1_000_000_000)
    failed = Process(b'{"sequence":1}\n')
    with pytest.raises(backend.MlxResourceSamplerError, match="sampler failed"):
        qualification.receive(failed, sampler=FailedSampler(), deadline_ns=123)
    assert failed._modular_mlx_stdout_remainder == b'{"sequence":1}\n'
    guard_calls = []

    class LiveSampler:
        def raise_if_failed(self):
            guard_calls.append("sampler")

    monkeypatch.setattr(qualification.backend, "enforce_deadline", lambda deadline_ns: guard_calls.append(("deadline", deadline_ns)) or 1_000_000_000)
    coalesced = Process(b'{"sequence":1}\n{"sequence":2}\n')
    sampler = LiveSampler()
    assert qualification.receive(coalesced, sampler=sampler, deadline_ns=456)["sequence"] == 1
    assert qualification.receive(coalesced, sampler=sampler, deadline_ns=456)["sequence"] == 2
    assert guard_calls == [("deadline", 456), "sampler", ("deadline", 456), "sampler"]
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    receive_source = ast.get_source_segment(source, next(node for node in ast.parse(source, filename=str(QUALIFICATION_PATH)).body if isinstance(node, ast.FunctionDef) and node.name == "receive")) or ""
    assert receive_source.index("backend.enforce_deadline(deadline_ns)") < receive_source.index('buffered.find(b"\\n")')
    assert receive_source.index("sampler.raise_if_failed()") < receive_source.index('buffered.find(b"\\n")')


def test_pilot_transport_uses_one_fail_closed_acceptance_deadline_for_every_receive_and_close() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_mlx_resource_pilot")
    body = ast.get_source_segment(source, function) or ""
    assert "pilot_deadline_ns = child_start_ns + backend.HARD_LIMIT_SECONDS * 1_000_000_000" in body
    assert body.count("receive(process, sampler, pilot_deadline_ns)") == 2
    assert "receive(process, sampler)" not in body
    assert 'send_guarded_ack(process, {"ack": True, "kind": "pilot_update_start_committed"' in body
    assert ', pilot_deadline_ns, signals, "mlx_pilot_update_ack")' in body
    assert 'send(process, {"ack": True, "kind": "close_committed"}, pilot_deadline_ns)' in body
    assert "backend.enforce_deadline(pilot_deadline_ns)" in body
    assert "process.wait(timeout=min(30.0, remaining_close_seconds))" in body


def test_resource_sampler_records_exact_parent_child_rss_and_swap() -> None:
    process_calls = []
    swap_calls = []

    def process_sampler(pids):
        process_calls.append(tuple(pids))
        return [
            {"pid": 321, "ppid": 1, "rss_bytes": 4096, "cpu_time_us": 7000},
            {"pid": 654, "ppid": 321, "rss_bytes": 16384, "cpu_time_us": 9000},
        ]

    def swap_sampler():
        swap_calls.append(True)
        return 8192

    sampler = backend.QualificationResourceSampler("qualify", 321, 654, process_sampler, swap_sampler, 5.0)
    sampler.begin_stage("joint", [11])
    sampler.start()
    sampler.await_stage_sample("joint", [11], time.monotonic_ns() + 1_000_000_000)
    rows = sampler.stop()
    assert rows
    assert process_calls == [(321, 654)] * len(rows)
    assert swap_calls == [True] * len(rows)
    assert [row["sample_id"] for row in rows] == list(range(len(rows)))
    assert all(row["run_id"] == "qualify" and row["expected_pids"] == [321, 654] for row in rows)
    assert all(row["processes"] == [{"pid": 321, "ppid": 1, "rss_bytes": 4096, "cpu_time_us": 7000}, {"pid": 654, "ppid": 321, "rss_bytes": 16384, "cpu_time_us": 9000}] for row in rows)
    assert all(row["swap_used_bytes"] == 8192 for row in rows)
    assert all(runner.validate_resource_row(row) is None for row in rows)
    assert sampler.sample_transaction_count == len(process_calls)
    assert sampler.max_sample_transaction_seconds > 0


def test_attempt_batch_stops_on_first_unacknowledged_lane_and_retains_only_committed_rows() -> None:
    rows = [
        {"construction_seed": 11, "event_sequence": 0},
        {"construction_seed": 23, "event_sequence": 0},
        {"construction_seed": 37, "event_sequence": 0},
    ]
    acknowledged = runner.AppendResult(0, 10, True, True, None, "a" * 64)
    retained = runner.AppendResult(0, 10, False, True, "signal_or_interruption", "b" * 64)
    calls = []

    class Writer:
        def __init__(self, result):
            self.result = result

        def append(self, row, pending_signal=None):
            calls.append((row["construction_seed"], callable(pending_signal)))
            return self.result

    committed, failure = qualification.write_attempt_batch(
        rows,
        {11: Writer(acknowledged), 23: Writer(retained), 37: Writer(acknowledged)},
        lambda: False,
    )
    assert committed == rows[:2]
    assert failure == retained
    assert calls == [(11, True), (23, True)]


def test_attempt_batch_returns_uncommitted_append_failure_without_advancing_lanes() -> None:
    rows = [{"construction_seed": 11}, {"construction_seed": 23}]
    failure = runner.AppendResult(0, 0, False, False, "artifact_inconsistency", None)

    class Writer:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def append(self, row, pending_signal=None):
            self.calls += 1
            if not self.result.committed:
                raise runner.LedgerAppendError(self.result)
            return self.result

    first = Writer(failure)
    second = Writer(runner.AppendResult(0, 10, True, True, None, "a" * 64))
    committed, observed = qualification.write_attempt_batch(rows, {11: first, 23: second}, lambda: False)
    assert committed == []
    assert observed == failure
    assert first.calls == 1
    assert second.calls == 0


@pytest.mark.parametrize(("mode", "expected_calls"), (("before", []), ("after", ["send"])))
def test_guarded_ack_never_authorizes_before_pending_signal_and_aborts_after_boundary(monkeypatch, mode, expected_calls) -> None:
    calls = []

    class Signals:
        def commit_guarded(self, boundary):
            if mode == "before":
                return runner.GuardedTransitionResult(False, None, 15)
            boundary()
            return runner.GuardedTransitionResult(True, None, 15)

    monkeypatch.setattr(qualification, "send", lambda *args: calls.append("send"))
    with pytest.raises(runner.HardAbort, match="signal_or_interruption"):
        qualification.send_guarded_ack(object(), {"ack": True}, None, Signals(), "test_ack")
    assert calls == expected_calls


def test_engine_resource_sample_mapping_requires_exact_seeds_and_ordered_unique_ids() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_resource_sample_ids_by_seed")

    class MlxEngineError(Exception):
        pass

    namespace = {"Any": Any, "Mapping": Mapping, "RUNG_ONE_SEEDS": (11, 23, 37, 53, 71), "MlxEngineError": MlxEngineError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ENGINE_PATH), "exec"), namespace)
    validate = namespace["validate_resource_sample_ids_by_seed"]
    valid = {"11": [1], "23": [2], "37": [3], "53": [4], "71": [5], "83": [6, 8]}
    assert validate(valid) == {11: [1], 23: [2], 37: [3], 53: [4], 71: [5], 83: [6, 8]}
    with pytest.raises(MlxEngineError):
        validate([1, 2])
    with pytest.raises(MlxEngineError):
        validate({**valid, "83": [8, 6]})
    with pytest.raises(MlxEngineError):
        validate({key: value for key, value in valid.items() if key != "71"})


def test_resource_sampler_advances_from_durable_starts_and_clears_training_jobs_before_tail() -> None:
    sampler = backend.QualificationResourceSampler("qualify", 321, 654, lambda pids: [], lambda: 0, 5.0)
    sampler.begin_stage("joint", [23, 11])
    sampler.observe_started("joint", [23, 11], 1, 2, 4096)
    assert sampler.attempted_updates == 2
    assert sampler.token_positions == 4096
    assert sampler.active_jobs == [
        {"worker": "S11", "seed": 11, "stage": "joint", "logical_update": 1},
        {"worker": "S23", "seed": 23, "stage": "joint", "logical_update": 1},
    ]
    sampler.clear_active_jobs(20_736, backend.POSITIONS)
    assert sampler.active_jobs == []


def test_resource_sampler_reuses_durable_clean_final_row_without_new_interval_wait() -> None:
    prior = [{"sample_id": 0, "active_jobs": [], "attempted_updates": 20_736, "token_positions": backend.POSITIONS, "monotonic_ns": time.monotonic_ns()}]
    calls = []
    sampler = backend.QualificationResourceSampler("qualify", 321, 654, lambda pids: calls.append(tuple(pids)) or [], lambda: 0, 5.0, prior_rows=prior)
    sampler.clear_active_jobs(20_736, backend.POSITIONS)
    sampler.start()
    rows = sampler.stop(final_sample=True)
    assert rows == prior
    assert calls == []


def test_resource_sampler_terminal_barrier_discards_stale_zombie_sample_and_durably_appends_clean_parent_row() -> None:
    prior = [
        {"sample_id": sample_id, "active_jobs": [], "attempted_updates": 0, "token_positions": 0, "monotonic_ns": time.monotonic_ns() - (10 - sample_id) * 5_000_000_000}
        for sample_id in range(4)
    ] + [
        {
            "sample_id": 4,
            "active_jobs": [{"worker": "MLX", "seed": 10000183, "stage": "dense_vmap5", "logical_update": 11}],
            "attempted_updates": 121,
            "token_positions": 247_808,
            "monotonic_ns": time.monotonic_ns() - 6_000_000_000,
            "processes": [
                {"pid": 321, "ppid": 1, "rss_bytes": 408_141_824, "cpu_time_us": 10_000_000},
                {"pid": 654, "ppid": 321, "rss_bytes": 1_271_414_784, "cpu_time_us": 14_250_000},
            ],
        }
    ]
    writer_rows = []
    sample_ready = threading.Event()
    release_sample = threading.Event()
    process_calls = []

    class Result:
        acknowledged = True
        reason_code = None

    class Writer:
        def append(self, row):
            writer_rows.append(dict(row))
            return Result()

    def process_sampler(pids):
        process_calls.append(tuple(pids))
        if len(process_calls) == 1:
            sample_ready.set()
            assert release_sample.wait(1.0)
            return [
                {"pid": 321, "ppid": 1, "rss_bytes": 408_141_824, "cpu_time_us": 10_020_000},
                {"pid": 654, "ppid": 321, "rss_bytes": 0, "cpu_time_us": 0},
            ]
        return [
            {"pid": 321, "ppid": 1, "rss_bytes": 408_141_824, "cpu_time_us": 10_030_000},
        ]

    sampler = backend.QualificationResourceSampler(
        "qualify",
        321,
        654,
        process_sampler,
        lambda: 8192,
        5.0,
        writer=Writer(),
        prior_rows=prior,
        phase="pilot",
        final_attempted_updates=132,
        final_token_positions=292_864,
    )
    sampler.observe_pilot_progress(
        "rung_two",
        10_000_283,
        11,
        132,
        292_864,
        {"active_memory_bytes": 1, "cache_memory_bytes": 1, "parent_rss_and_swap_required": True, "peak_memory_bytes": 1},
    )
    sampler.start()
    assert sample_ready.wait(1.0)
    sampler.clear_active_jobs(132, 292_864)
    sampler.mark_child_exited()
    outcome = {}

    def stop_sampler():
        outcome["rows"] = sampler.stop(final_sample=True)

    stop_thread = threading.Thread(target=stop_sampler)
    stop_thread.start()
    assert sampler.stop_event.wait(1.0)
    release_sample.set()
    stop_thread.join(1.0)
    assert not stop_thread.is_alive()
    assert process_calls == [(321, 654), (321,)]
    assert outcome["rows"][:-1] == prior
    assert writer_rows == [outcome["rows"][-1]]
    assert outcome["rows"][-1]["sample_id"] == 5
    assert outcome["rows"][-1]["expected_pids"] == [321]
    assert outcome["rows"][-1]["processes"] == [{"pid": 321, "ppid": 1, "rss_bytes": 408_141_824, "cpu_time_us": 10_030_000}]
    assert outcome["rows"][-1]["active_jobs"] == []
    assert outcome["rows"][-1]["attempted_updates"] == 132
    assert outcome["rows"][-1]["token_positions"] == 292_864


def test_resource_sampler_terminal_barrier_waits_exact_positive_interval_remainder_without_scheduler_sleep(monkeypatch) -> None:
    prior = [
        {
            "sample_id": 0,
            "active_jobs": [{"worker": "MLX", "seed": 10_000_283, "stage": "rung_two", "logical_update": 11}],
            "attempted_updates": 132,
            "token_positions": 292_864,
            "monotonic_ns": 10_000_000_000,
        }
    ]
    process_calls = []
    writer_rows = []
    waits = []
    clock = {"monotonic_ns": 12_000_000_000}

    class Result:
        acknowledged = True
        reason_code = None

    class Writer:
        def append(self, row):
            writer_rows.append(dict(row))
            return Result()

    class JoinedThread:
        def join(self):
            return None

    class TerminalWaiter:
        def wait(self, seconds):
            waits.append(seconds)
            clock["monotonic_ns"] += int(seconds * 1_000_000_000)
            return False

    sampler = backend.QualificationResourceSampler(
        "qualify",
        321,
        654,
        lambda pids: process_calls.append(tuple(pids)) or [{"pid": 321, "ppid": 1, "rss_bytes": 4096, "cpu_time_us": 10_030_000}],
        lambda: 8192,
        5.0,
        writer=Writer(),
        prior_rows=prior,
        phase="pilot",
        final_attempted_updates=132,
        final_token_positions=292_864,
    )
    sampler.clear_active_jobs(132, 292_864)
    sampler.mark_child_exited()
    sampler.thread = JoinedThread()
    monkeypatch.setattr(backend.time, "monotonic_ns", lambda: clock["monotonic_ns"])
    monkeypatch.setattr(backend.threading, "Event", TerminalWaiter)
    rows = sampler.stop(final_sample=True, deadline_ns=16_000_000_000)
    assert waits == [3.0]
    assert process_calls == [(321,)]
    assert writer_rows == [rows[-1]]
    assert rows[-1]["monotonic_ns"] - prior[-1]["monotonic_ns"] == 5_000_000_000
    assert rows[-1]["expected_pids"] == [321]
    source = (PROJECT_ROOT / "src" / "model" / "modular_mlx_backend.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sampler_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QualificationResourceSampler")
    stop_function = next(node for node in sampler_class.body if isinstance(node, ast.FunctionDef) and node.name == "stop")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "time" and node.func.attr == "sleep" for node in ast.walk(stop_function))


def test_resource_sampler_terminal_barrier_rejects_wait_whose_earliest_sample_reaches_acceptance_deadline(monkeypatch) -> None:
    prior = [{"sample_id": 0, "active_jobs": [{"worker": "MLX", "seed": 1, "stage": "rung_two", "logical_update": 1}], "attempted_updates": 132, "token_positions": 292_864, "monotonic_ns": 10_000_000_000}]
    waits = []

    class JoinedThread:
        def join(self):
            return None

    class ForbiddenWaiter:
        def wait(self, seconds):
            waits.append(seconds)
            return False

    sampler = backend.QualificationResourceSampler("qualify", 321, 654, lambda pids: pytest.fail("terminal sample crossed impossible deadline"), lambda: 0, 5.0, prior_rows=prior, phase="pilot", final_attempted_updates=132, final_token_positions=292_864)
    sampler.clear_active_jobs(132, 292_864)
    sampler.mark_child_exited()
    sampler.thread = JoinedThread()
    monkeypatch.setattr(backend.time, "monotonic_ns", lambda: 12_000_000_000)
    monkeypatch.setattr(backend.threading, "Event", ForbiddenWaiter)
    with pytest.raises(backend.MlxDeadlineExceeded, match="terminal resource sample"):
        sampler.stop(final_sample=True, deadline_ns=14_000_000_000)
    assert waits == []


def test_seed_resource_mapping_excludes_baseline_evaluation_and_closure_rows() -> None:
    seeds = [11, 23, 37, 53, 71]
    rows = [
        {"sample_id": 0, "active_jobs": []},
        {"sample_id": 1, "active_jobs": [{"worker": f"S{seed}", "seed": seed, "stage": "joint", "logical_update": 1} for seed in seeds]},
        {"sample_id": 2, "active_jobs": [{"worker": "S83", "seed": 83, "stage": "rung_two", "logical_update": 1}]},
        {"sample_id": 3, "active_jobs": [{"worker": "MLX", "seed": None, "stage": "evaluation", "logical_update": None}]},
        {"sample_id": 4, "active_jobs": [{"worker": "MLX", "seed": None, "stage": "closure", "logical_update": None}]},
        {"sample_id": 5, "active_jobs": []},
    ]
    assert qualification.seed_resource_sample_ids(rows) == {11: [1], 23: [1], 37: [1], 53: [1], 71: [1], 83: [2]}


def test_clean_child_close_commits_terminal_package_before_ack_and_join(monkeypatch) -> None:
    events = []

    class Process:
        def wait(self, timeout):
            events.append(("join", timeout))
            return 0

    monkeypatch.setattr(qualification.backend, "enforce_deadline", lambda deadline_ns: 7_000_000_000)
    monkeypatch.setattr(qualification, "send", lambda process, message, deadline_ns: events.append((message["kind"], deadline_ns)))
    result = {"resource_rows": [{"sample_id": 0}]}
    qualification.commit_clean_child_close(Process(), 123, result, lambda observed: events.append(("terminal", observed is result)))
    assert events == [("terminal", True), ("close_committed", 123), ("join", 7.0)]


def test_clean_child_close_late_wait_return_fails_closed_after_remaining_time_cap(monkeypatch) -> None:
    events = []
    deadline_calls = []

    class Process:
        def wait(self, timeout):
            events.append(("join", timeout))
            return 0

    def deadline(deadline_ns):
        deadline_calls.append(deadline_ns)
        if len(deadline_calls) == 2:
            raise backend.MlxDeadlineExceeded("late wait return")
        return 2_000_000_000

    monkeypatch.setattr(qualification.backend, "enforce_deadline", deadline)
    monkeypatch.setattr(qualification, "send", lambda process, message, deadline_ns: events.append((message["kind"], deadline_ns)))
    with pytest.raises(backend.MlxDeadlineExceeded, match="late wait return"):
        qualification.commit_clean_child_close(Process(), 456, {}, lambda _: events.append("terminal"))
    assert events == ["terminal", ("close_committed", 456), ("join", 2.0)]
    assert deadline_calls == [456, 456]


def test_clean_child_close_never_acknowledges_when_terminal_finalizer_fails(monkeypatch) -> None:
    events = []

    class Process:
        def wait(self, timeout):
            events.append("join")
            return 0

    monkeypatch.setattr(qualification, "send", lambda *args: events.append("ack"))

    def fail(_):
        events.append("terminal")
        raise runner.ContractError("terminal failure")

    with pytest.raises(runner.ContractError):
        qualification.commit_clean_child_close(Process(), 123, {}, fail)
    assert events == ["terminal"]


def test_production_claim_orders_final_resource_terminal_package_ack_and_join() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    functions = {node.name: ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef)}
    claim = functions["run_mlx_claim"]
    pilot = functions["run_mlx_resource_pilot"]
    training = functions["run_training"]
    assert 'validate_resource_timeline(resource_rows, "claim", require_clean_final=True)' in claim
    assert pilot.index("sampler.clear_active_jobs(") < pilot.index("sampler.mark_child_exited()") < pilot.index("sampler.stop(final_sample=True, deadline_ns=pilot_deadline_ns)")
    assert pilot.index("sampler.mark_child_exited()") < pilot.index("measure_pilot_fixed_components(")
    assert claim.index("sampler.stop(final_sample=True, deadline_ns=deadline_ns)") < claim.index("commit_clean_child_close(")
    assert claim.index("sampler.clear_active_jobs(") < claim.index("sampler.stop(final_sample=True, deadline_ns=deadline_ns)")
    assert "await_stage_sample" not in training
    assert "begin_unattributed_phase" not in training
    assert training.index("sampler.clear_active_jobs(") < training.index('send(process, {"kind": "evaluate"')
    assert "resource_sample_ids_by_seed" in training


def test_clean_claim_transport_checks_deadline_after_join_stderr_and_cleanup() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    claim = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_mlx_claim")
    clean_transport = next(node for node in ast.walk(claim) if isinstance(node, ast.FunctionDef) and node.name == "clean_transport")
    segment = ast.get_source_segment(source, clean_transport) or ""
    commit_index = segment.index("commit_clean_child_close(")
    post_join_deadline_index = segment.index("backend.enforce_deadline(deadline_ns)", commit_index)
    stderr_fsync_index = segment.index("os.fsync(stderr_handle.fileno())")
    post_fsync_deadline_index = segment.index("backend.enforce_deadline(deadline_ns)", stderr_fsync_index)
    stderr_read_index = segment.index("stderr_path.read_text(")
    post_read_deadline_index = segment.index("backend.enforce_deadline(deadline_ns)", stderr_read_index)
    cleanup_index = segment.index("cleanup_transport(transport_primary)")
    clean_cleanup_branch_index = segment.index("if transport_primary is None:", cleanup_index)
    post_cleanup_deadline_index = segment.index("backend.enforce_deadline(deadline_ns)", clean_cleanup_branch_index)
    assert commit_index < post_join_deadline_index < stderr_fsync_index < post_fsync_deadline_index
    assert post_fsync_deadline_index < stderr_read_index < post_read_deadline_index < cleanup_index
    assert cleanup_index < clean_cleanup_branch_index < post_cleanup_deadline_index


def test_clean_pilot_transport_checks_deadline_after_join_stderr_and_cleanup() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    pilot = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_mlx_resource_pilot")
    segment = ast.get_source_segment(source, pilot) or ""
    join_index = segment.index("process.wait(timeout=min(30.0, remaining_close_seconds))")
    post_join_deadline_index = segment.index("backend.enforce_deadline(pilot_deadline_ns)", join_index)
    stderr_fsync_index = segment.index("os.fsync(stderr_handle.fileno())", post_join_deadline_index)
    post_fsync_deadline_index = segment.index("backend.enforce_deadline(pilot_deadline_ns)", stderr_fsync_index)
    stderr_stat_index = segment.index("stderr_path.stat().st_size", post_fsync_deadline_index)
    post_stat_deadline_index = segment.index("backend.enforce_deadline(pilot_deadline_ns)", stderr_stat_index)
    scratch_cleanup_index = segment.index("cleanup_scratch(scratch)", post_stat_deadline_index)
    post_cleanup_deadline_index = segment.index("backend.enforce_deadline(pilot_deadline_ns)", scratch_cleanup_index)
    assert join_index < post_join_deadline_index < stderr_fsync_index < post_fsync_deadline_index
    assert post_fsync_deadline_index < stderr_stat_index < post_stat_deadline_index
    assert post_stat_deadline_index < scratch_cleanup_index < post_cleanup_deadline_index


def test_production_pilot_charges_conservative_resource_finalization_and_governs_terminal_deadline() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    functions = {node.name: ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef)}
    pilot = functions["run_mlx_resource_pilot"]
    claim = functions["run_mlx_claim"]
    assert "resource_finalization_actual_seconds = (time.perf_counter_ns() - resource_finalization_started_ns) / 1_000_000_000" in pilot
    assert "resource_sample_max_seconds = sampler.max_sample_transaction_seconds" in pilot
    assert "resource_finalization_seconds = resource_interval_seconds + 2 * resource_sample_max_seconds" in pilot
    assert "resource_finalization_seconds = resource_finalization_actual_seconds" not in pilot
    assert "resource_finalization_actual_seconds > resource_finalization_seconds" in pilot
    assert "sampler.stop(final_sample=True, deadline_ns=pilot_deadline_ns)" in pilot
    assert "sampler.stop(final_sample=True, deadline_ns=deadline_ns)" in claim
    assert '"pilot_tail_resource_finalization_projection"' in pilot
    assert '"interval_seconds": resource_interval_seconds' in pilot
    assert '"max_observed_sample_duration_seconds": resource_sample_max_seconds' in pilot
    assert '"sample_transaction_count": resource_sample_transaction_count' in pilot
    pilot_write = 'cpu.write_canonical_json(run_root / "run" / "pilot.json", pilot_record)'
    assert pilot_write in pilot
    assert pilot.index(pilot_write) < pilot.rindex("backend.enforce_deadline(pilot_deadline_ns)")
    payload = json.loads((PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json").read_text(encoding="utf-8"))
    exact_keys = payload["artifacts"]["schemas"]["pilot"]["resource_finalization_assertion_actual_exact_keys"]
    assert exact_keys == ["component_seconds", "actual_stop_seconds", "final_active_jobs", "final_attempted_updates", "final_expected_pids", "final_sample_id", "final_token_positions", "interval_seconds", "max_observed_sample_duration_seconds", "sample_transaction_count"]
    assert payload["pilot"]["time_statistics"]["resource_finalization_benchmark"]["component_formula"] == "resource_finalization_seconds=5.0+2*maximum_observed_sample_transaction_seconds"


def test_claim_second_swap_sample_is_after_data_and_before_child_spawn() -> None:
    source = QUALIFICATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(QUALIFICATION_PATH))
    claim = next(ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_mlx_claim")
    payload = json.loads((PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json").read_text(encoding="utf-8"))
    assert claim.index("preworker_row = cpu._resource_sample(") < claim.index("subprocess.Popen(")
    assert claim.index("subprocess.Popen(") < claim.index("run_training(")
    assert claim.index("sampler.swap_baseline = transition.swap_baseline_bytes") < claim.index("sampler.start()")
    assert payload["pilot"]["resource_sampling"]["claim_second_sample"] == "after_claim_data_construction_immediately_before_MLX_child_spawn"


def test_parent_resource_sampler_propagates_failure() -> None:
    entered = threading.Event()

    def process_sampler(_):
        entered.set()
        raise RuntimeError("injected sampler failure")

    sampler = backend.QualificationResourceSampler("qualify", 321, 654, process_sampler, lambda: 0, 0.001)
    sampler.start()
    assert entered.wait(1.0)
    with pytest.raises(backend.MlxResourceSamplerError, match="resource sampler failed"):
        sampler.stop()
