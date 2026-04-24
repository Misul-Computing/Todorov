from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationSpec:
    simulation_id: str
    category: str
    script_relative_path: str
    metrics_filename: str
    artifact_filenames: tuple[str, ...]
    smoke_env: dict[str, str]
    hard_env: dict[str, str] | None = None
    required_summary_keys: tuple[str, ...] = ()
    minimum_summary_values: tuple[tuple[str, float], ...] = ()
    required_modules: tuple[str, ...] = ()

    def script_path(self, project_root: Path) -> Path:
        return project_root / self.script_relative_path


SIMULATION_SPECS: dict[str, SimulationSpec] = {
    "capacity_scaling": SimulationSpec(
        simulation_id="capacity_scaling",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/capacity_scaling.py",
        metrics_filename="capacity_scaling_metrics.json",
        artifact_filenames=("capacity_scaling.png", "capacity_scaling_metrics.json"),
        smoke_env={
            "CAPACITY_NETWORK_SIZES": "64,128",
            "CAPACITY_LOAD_FACTORS": "0.05,0.10,0.138",
            "CAPACITY_TRIALS": "2",
            "CAPACITY_MAX_ITERS": "12",
        },
        required_summary_keys=("binary_at_target_alpha", "ternary_at_target_alpha"),
    ),
    "rate_coded_spike": SimulationSpec(
        simulation_id="rate_coded_spike",
        category="compression",
        script_relative_path="neuroloc/simulations/prototypes/rate_coded_spike.py",
        metrics_filename="rate_coded_spike_metrics.json",
        artifact_filenames=("rate_coded_spike.png", "rate_coded_spike_metrics.json"),
        smoke_env={
            "RATE_SPIKE_N_NEURONS": "128",
            "RATE_SPIKE_N_PATTERNS": "5,10",
            "RATE_SPIKE_TRIALS": "2",
            "RATE_SPIKE_MAX_ITERS": "12",
        },
        required_summary_keys=("reference_cosine.binary", "reference_cosine.ternary", "reference_cosine.rate"),
    ),
    "hierarchical_ternary": SimulationSpec(
        simulation_id="hierarchical_ternary",
        category="compression",
        script_relative_path="neuroloc/simulations/sparse_coding/hierarchical_ternary.py",
        metrics_filename="hierarchical_ternary_metrics.json",
        artifact_filenames=("hierarchical_ternary.png", "hierarchical_ternary_metrics.json"),
        smoke_env={
            "HTERN_D_VALUES": "64,256",
            "HTERN_K_FRACTIONS": "0.05,0.20,0.41",
            "HTERN_N_SAMPLES": "64",
            "HTERN_TRIALS": "2",
        },
        required_summary_keys=(
            "standard_reference_mi",
            "standard_reference_cka",
            "selected_hierarchical_reference_mi",
            "selected_hierarchical_reference_cka",
        ),
    ),
    "asymmetric_outer_product_recall": SimulationSpec(
        simulation_id="asymmetric_outer_product_recall",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/asymmetric_outer_product_recall.py",
        metrics_filename="asymmetric_outer_product_recall_metrics.json",
        artifact_filenames=("asymmetric_outer_product_recall.png", "asymmetric_outer_product_recall_metrics.json"),
        smoke_env={
            "ASYM_TRIALS": "2",
            "ASYM_PATTERN_COUNTS": "8,16",
            "ASYM_QUERY_NOISE": "0.0,0.1",
            "ASYM_DECAYS": "0.4,0.8",
        },
        required_summary_keys=("best_exact_query_encoding", "selected_exact_query_raw_cosine"),
    ),
    "correction_field_capacity": SimulationSpec(
        simulation_id="correction_field_capacity",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/correction_field_capacity.py",
        metrics_filename="correction_field_capacity_metrics.json",
        artifact_filenames=(
            "correction_field_capacity_heatmaps.png",
            "correction_field_capacity_summary.png",
            "correction_field_capacity_metrics.json",
        ),
        smoke_env={
            "CF_HEAD_DIMS": "32,64",
            "CF_PATTERN_COUNTS": "4,8,16",
            "CF_PRED_QUALITIES": "0.0,0.9,0.99",
            "CF_DECAYS": "0.5,0.9",
            "CF_TRIALS": "2",
        },
        required_summary_keys=(
            "capacity_raw_vs_corrfield_by_quality",
            "cosine_gain_by_quality_and_patterns",
            "threshold_shift_by_decay",
        ),
    ),
    "correction_field_trained_prediction": SimulationSpec(
        simulation_id="correction_field_trained_prediction",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/correction_field_trained_prediction.py",
        metrics_filename="correction_field_trained_prediction_metrics.json",
        artifact_filenames=(
            "correction_field_trained_prediction_heatmaps.png",
            "correction_field_trained_prediction_summary.png",
            "correction_field_trained_prediction_metrics.json",
        ),
        smoke_env={
            "CFT_HEAD_DIMS": "32",
            "CFT_PATTERN_COUNTS": "4,8",
            "CFT_TEMPORAL_CORRELATIONS": "0.5,0.9",
            "CFT_DECAYS": "0.5,0.9",
            "CFT_TRIALS": "2",
            "CFT_TRAIN_SEQUENCES": "64",
            "CFT_TRAIN_SEQUENCE_LENGTH": "16",
            "CFT_TRAIN_ITERS": "25",
            "CFT_BATCH_SIZE": "32",
        },
        required_summary_keys=(
            "memory_substrate_capacity_delta_by_temporal_correlation",
            "reconstruction_capacity_gain",
            "observed_prediction_quality_by_correlation",
        ),
    ),
    "multi_resolution_head_split": SimulationSpec(
        simulation_id="multi_resolution_head_split",
        category="compression",
        script_relative_path="neuroloc/simulations/memory/multi_resolution_head_split.py",
        metrics_filename="multi_resolution_head_split_metrics.json",
        artifact_filenames=(
            "multi_resolution_head_split.png",
            "multi_resolution_head_split_metrics.json",
        ),
        smoke_env={
            "MRHS_STREAM_LENGTHS": "32,64",
            "MRHS_HEAD_DIMS": "32,64",
            "MRHS_TRIALS": "2",
            "MRHS_QUERIES_PER_CLASS": "4",
        },
        required_summary_keys=(
            "recall_by_class_and_configuration",
            "capacity_gain_split_over_uniform_by_class",
            "crossover_pattern_count_by_class",
        ),
    ),
    "wta_dynamics": SimulationSpec(
        simulation_id="wta_dynamics",
        category="spiking",
        script_relative_path="neuroloc/simulations/lateral_inhibition/wta_dynamics.py",
        metrics_filename="wta_dynamics_metrics.json",
        artifact_filenames=("wta_dynamics.png", "wta_dynamics_anchor.png", "wta_dynamics_metrics.json"),
        smoke_env={
            "WTA_N_NEURONS": "40",
            "WTA_TARGET_FRACTIONS": "0.10,0.41",
            "WTA_NOISE_LEVELS": "0.2,0.6",
            "WTA_SELECTION_TRIALS": "2",
            "WTA_CALIBRATION_TRIALS": "32",
            "WTA_SCALING_NETWORK_SIZES": "40,80",
            "WTA_SCALING_TARGET_FRACTIONS": "0.10,0.41",
            "WTA_SCALING_TRIALS": "2",
            "WTA_ANCHOR_DURATION_MS": "75",
        },
        required_summary_keys=("kwta_exact_support_recovery", "threshold_exact_support_recovery"),
        required_modules=("brian2",),
    ),
    "lif_fi_curve": SimulationSpec(
        simulation_id="lif_fi_curve",
        category="spiking",
        script_relative_path="neuroloc/simulations/single_neuron/lif_fi_curve.py",
        metrics_filename="lif_leak_validation_metrics.json",
        artifact_filenames=("lif_leak_validation.png", "lif_leak_validation_metrics.json"),
        smoke_env={
            "LIF_GAP_VALUES_MS": "4,16",
            "LIF_GAP_TRIALS": "2",
            "LIF_DRIFT_LENGTHS": "64,128",
            "LIF_DRIFT_TRIALS": "2",
        },
        required_summary_keys=(
            "main_comparison.selected_gap_retained_fraction.integrator_control",
            "main_comparison.selected_max_length_final_abs_mv.integrator_control",
            "spike_probe.selected_gap_second_spike.integrator_control",
        ),
        minimum_summary_values=(("spike_probe.selected_gap_second_spike.integrator_control", 0.01),),
        required_modules=("brian2",),
    ),
    "thinking_loop_prototype": SimulationSpec(
        simulation_id="thinking_loop_prototype",
        category="reasoning",
        script_relative_path="neuroloc/simulations/reasoning/thinking_loop_prototype.py",
        metrics_filename="thinking_loop_prototype_metrics.json",
        artifact_filenames=("thinking_loop_prototype.png", "thinking_loop_prototype_metrics.json"),
        smoke_env={
            "THINK_N_TRAIN": "256",
            "THINK_N_VAL": "128",
            "THINK_N_TEST": "128",
            "THINK_N_EPOCHS": "2",
            "THINK_BATCH_SIZE": "64",
            "THINK_D_MODEL": "24",
            "THINK_DEPTH": "2",
            "THINK_DEEP_DEPTH": "4",
            "THINK_N_LIST": "2,4",
            "THINK_N_TRAIN_CAP": "2",
            "THINK_K_LIST": "1,2",
            "THINK_N_SEEDS": "2",
            "THINK_TRAIN_K": "1",
            "THINK_CONV_K_MAX": "4",
        },
        required_summary_keys=(
            "accuracy_by_K_and_N",
            "per_step_gain_by_task_difficulty",
            "generalization_gap_thinking_vs_feedforward",
            "fixed_point_convergence_fraction",
        ),
        required_modules=("torch",),
    ),
    "slot_buffer_capacity": SimulationSpec(
        simulation_id="slot_buffer_capacity",
        category="phase1_nm",
        script_relative_path="neuroloc/simulations/memory/slot_buffer_capacity.py",
        metrics_filename="slot_buffer_capacity_metrics.json",
        artifact_filenames=("slot_buffer_capacity.png", "slot_buffer_capacity_metrics.json"),
        smoke_env={
            "SLOT_CAP_HEAD_DIMS": "32,64",
            "SLOT_CAP_SLOTS": "16,64",
            "SLOT_CAP_TEMPERATURES": "0.1,1.0",
            "SLOT_CAP_QUERY_NOISE": "0.0,0.1",
            "SLOT_CAP_TRIALS": "2",
        },
        required_summary_keys=(
            "pass_criterion_slot_cosine_mean",
            "pass_criterion_slot_exact_mean",
            "capacity_threshold_by_dim",
            "by_cell",
        ),
    ),
    "slot_surprise_writes": SimulationSpec(
        simulation_id="slot_surprise_writes",
        category="phase1_nm",
        script_relative_path="neuroloc/simulations/memory/slot_surprise_writes.py",
        metrics_filename="slot_surprise_writes_metrics.json",
        artifact_filenames=("slot_surprise_writes.png", "slot_surprise_writes_metrics.json"),
        smoke_env={
            "SLOT_SW_HEAD_DIMS": "64",
            "SLOT_SW_SLOTS": "32",
            "SLOT_SW_DISTANCES": "64,128",
            "SLOT_SW_PREDICTABLE": "0.8,0.9",
            "SLOT_SW_TAUS": "0.05,0.1",
            "SLOT_SW_TEMPERATURE": "0.1",
            "SLOT_SW_TRIALS": "2",
        },
        required_summary_keys=(
            "pass_criterion_threshold",
            "cells",
        ),
    ),
    "contextual_recall_world": SimulationSpec(
        simulation_id="contextual_recall_world",
        category="phase1_nm",
        script_relative_path="neuroloc/simulations/memory/contextual_recall_world.py",
        metrics_filename="contextual_recall_world_metrics.json",
        artifact_filenames=("contextual_recall_world.png", "contextual_recall_world_metrics.json"),
        smoke_env={
            "CTX_WORLD_EPISODES": "16",
            "CTX_WORLD_SEQ_LEN": "10",
            "CTX_WORLD_IDENTITIES": "8",
            "CTX_WORLD_ACTIVE": "3",
            "CTX_WORLD_TRACK_LENGTH": "17",
            "CTX_WORLD_OCCLUSION": "0.2",
            "CTX_WORLD_FEATURE_DROPOUT": "0.3",
        },
        required_summary_keys=(
            "recognition_mean_candidate_count",
            "recollection_mean_lag",
            "prediction_mean_step_distance",
            "compression_mean_ratio",
            "imagination_novelty_rate",
            "reasoning_mean_margin",
        ),
        minimum_summary_values=(("compression_mean_ratio", 1.0),),
    ),
    "slot_key_interference_sweep": SimulationSpec(
        simulation_id="slot_key_interference_sweep",
        category="phase1_nm",
        script_relative_path="neuroloc/simulations/memory/slot_key_interference_sweep.py",
        metrics_filename="slot_key_interference_sweep_metrics.json",
        artifact_filenames=(
            "slot_key_interference_sweep.png",
            "slot_key_interference_sweep_metrics.json",
        ),
        smoke_env={
            "SLOT_INT_HEAD_DIMS": "32,64",
            "SLOT_INT_SLOTS": "16,32",
            "SLOT_INT_CORRELATIONS": "0.0,0.8,0.95",
            "SLOT_INT_TEMPERATURES": "0.1",
            "SLOT_INT_QUERY_NOISE": "0.0",
            "SLOT_INT_TRIALS": "2",
        },
        required_summary_keys=(
            "slot_interference_slope_by_dim",
            "matrix_interference_slope_by_dim",
            "slot_minus_matrix_gain_at_high_correlation",
            "by_cell",
        ),
    ),
    "multi_association_recall": SimulationSpec(
        simulation_id="multi_association_recall",
        category="phase1_nm",
        script_relative_path="neuroloc/simulations/memory/multi_association_recall.py",
        metrics_filename="multi_association_recall_metrics.json",
        artifact_filenames=(
            "multi_association_recall.png",
            "multi_association_recall_metrics.json",
        ),
        smoke_env={
            "MAR_HEAD_DIMS": "32,64",
            "MAR_CUE_COUNTS": "8,12",
            "MAR_VALUES_PER_CUE": "2,3",
            "MAR_QUERY_NOISE": "0.0,0.1",
            "MAR_TEMPERATURE": "0.1",
            "MAR_TRIALS": "2",
        },
        required_summary_keys=(
            "slot_bundle_exact_rate",
            "slot_value_hit_rate",
            "matrix_bundle_exact_rate",
            "matrix_value_hit_rate",
            "slot_minus_matrix_exact_gain",
            "by_cell",
        ),
    ),
    "delayed_cue_world": SimulationSpec(
        simulation_id="delayed_cue_world",
        category="phase1_nm",
        script_relative_path="neuroloc/simulations/memory/delayed_cue_world.py",
        metrics_filename="delayed_cue_world_metrics.json",
        artifact_filenames=(
            "delayed_cue_world.png",
            "delayed_cue_world_metrics.json",
        ),
        smoke_env={
            "DCW_GOALS": "4",
            "DCW_CUE_COUNTS": "1,3",
            "DCW_NOISE_LEVELS": "0.0,0.2,0.4",
            "DCW_DELAY_STEPS": "6",
            "DCW_EPISODES": "48",
        },
        required_summary_keys=(
            "bayes_accuracy_by_noise",
            "last_cue_accuracy_by_noise",
            "memory_advantage_by_noise",
            "oracle_accuracy",
            "no_memory_accuracy",
            "by_cell",
        ),
    ),
    "episodic_separation_completion": SimulationSpec(
        simulation_id="episodic_separation_completion",
        category="biology_phase1",
        script_relative_path="neuroloc/simulations/memory/episodic_separation_completion.py",
        metrics_filename="episodic_separation_completion_metrics.json",
        artifact_filenames=(
            "episodic_separation_completion.png",
            "episodic_separation_completion_metrics.json",
        ),
        smoke_env={
            "ESC_TRIALS": "8",
            "ESC_FEATURE_DIM": "16",
            "ESC_ACTION_COUNT": "4",
            "ESC_OVERLAPS": "0.5,0.9",
            "ESC_CUE_DROPS": "0.0,0.5",
            "ESC_DISTRACTORS": "0,4",
        },
        required_summary_keys=(
            "state_probe_accuracy",
            "action_success",
            "joint_success",
            "separation_margin",
            "completion_accuracy_by_cue_drop",
            "novelty_detection_accuracy",
            "delayed_recall_after_distractors",
            "policy_metrics",
        ),
    ),
    "episodic_replay_reuse": SimulationSpec(
        simulation_id="episodic_replay_reuse",
        category="biology_phase1",
        script_relative_path="neuroloc/simulations/memory/episodic_replay_reuse.py",
        metrics_filename="episodic_replay_reuse_metrics.json",
        artifact_filenames=(
            "episodic_replay_reuse.png",
            "episodic_replay_reuse_metrics.json",
        ),
        smoke_env={
            "ERR_TRIALS": "8",
            "ERR_FEATURE_DIM": "16",
            "ERR_ACTION_COUNT": "4",
            "ERR_CUE_DROPS": "0.25,0.5",
            "ERR_DISTRACTORS": "0,4,8",
            "ERR_REPLAY_STEPS": "3",
            "ERR_DECAY": "0.9",
            "ERR_DISTRACTOR_OVERLAP": "0.75",
        },
        required_summary_keys=(
            "state_probe_accuracy",
            "action_success",
            "joint_success",
            "replay_reuse_gain",
            "targeted_vs_random_replay_gap",
            "distractor_decay_curve",
            "state_probe_accuracy_by_cue_drop",
            "policy_metrics",
        ),
    ),
    "contextual_gate_routing": SimulationSpec(
        simulation_id="contextual_gate_routing",
        category="biology_phase1",
        script_relative_path="neuroloc/simulations/memory/contextual_gate_routing.py",
        metrics_filename="contextual_gate_routing_metrics.json",
        artifact_filenames=(
            "contextual_gate_routing.png",
            "contextual_gate_routing_metrics.json",
        ),
        smoke_env={
            "CGR_TRIALS": "24",
            "CGR_CUES": "6",
            "CGR_CONTEXTS": "3",
            "CGR_ACTIONS": "4",
            "CGR_CONTEXTUAL_FRACTION": "0.5",
        },
        required_summary_keys=(
            "state_probe_accuracy",
            "action_success",
            "joint_success",
            "context_gate_dependency",
            "false_bind_rate_under_wrong_context",
            "action_success_by_trial_type",
            "joint_success_by_trial_type",
            "policy_metrics",
        ),
    ),
    "nm_hard_symbolic_test_material": SimulationSpec(
        simulation_id="nm_hard_symbolic_test_material",
        category="hard_symbolic_nm",
        script_relative_path="neuroloc/simulations/memory/nm_hard_symbolic_test_material.py",
        metrics_filename="nm_hard_symbolic_test_material_metrics.json",
        artifact_filenames=("nm_hard_symbolic_test_material_metrics.json",),
        smoke_env={
            "NM_HARD_SYMBOLIC_PROFILE": "smoke",
            "NM_HARD_SYMBOLIC_EPISODES": "4",
            "NM_HARD_SYMBOLIC_SEED": "42",
        },
        hard_env={
            "NM_HARD_SYMBOLIC_PROFILE": "hard",
            "NM_HARD_SYMBOLIC_EPISODES": "32",
            "NM_HARD_SYMBOLIC_SEED": "42",
        },
        required_summary_keys=(
            "family_count",
            "policy_count",
            "state_probe_accuracy",
            "action_success",
            "joint_success",
            "exact_recall",
            "degraded_cue_recall",
            "oracle_joint_success",
            "no_memory_joint_success",
            "recency_only_joint_success",
            "shuffled_address_joint_success",
            "targeted_replay_gain",
            "interference_slope",
            "reuse_advantage",
            "hard_case_rollout_gain",
            "hard_case_pre_rollout_gap",
            "hard_minus_easy_rollout_gain",
            "compression_bit_saving_fraction",
            "bits_written_per_successful_episode",
            "verbatim_within_budget",
            "compressed_within_budget",
            "gate_open_fraction",
            "memory_output_vs_residual_norm",
            "slot_address_entropy",
            "address_margin",
            "write_frequency",
            "read_concentration",
            "retention_over_delay",
            "compression_budget",
            "reconstruction_error",
            "leakage_violation_count",
            "contract_field_violation_count",
        ),
        minimum_summary_values=(
            ("oracle_joint_success", 0.98),
            ("targeted_replay_gain", 0.1),
            ("interference_slope", 0.1),
            ("reuse_advantage", 0.1),
            ("hard_case_rollout_gain", 0.1),
            ("hard_case_pre_rollout_gap", 0.1),
            ("hard_minus_easy_rollout_gain", 0.1),
            ("compression_bit_saving_fraction", 0.1),
            ("compressed_within_budget", 0.98),
            ("gate_open_fraction", 0.1),
            ("memory_output_vs_residual_norm", 0.1),
            ("slot_address_entropy", 0.1),
            ("address_margin", 0.1),
            ("write_frequency", 0.1),
            ("read_concentration", 0.1),
            ("retention_over_delay", 0.98),
        ),
    ),
}


SUITES: dict[str, tuple[str, ...]] = {
    "compression": (
        "capacity_scaling",
        "rate_coded_spike",
        "hierarchical_ternary",
        "asymmetric_outer_product_recall",
        "correction_field_capacity",
        "correction_field_trained_prediction",
        "multi_resolution_head_split",
    ),
    "spiking": (
        "wta_dynamics",
        "lif_fi_curve",
    ),
    "reasoning": (
        "thinking_loop_prototype",
    ),
    "biology_phase1": (
        "episodic_separation_completion",
        "episodic_replay_reuse",
        "contextual_gate_routing",
    ),
    "phase1_nm": (
        "slot_buffer_capacity",
        "slot_surprise_writes",
        "contextual_recall_world",
        "slot_key_interference_sweep",
        "multi_association_recall",
        "delayed_cue_world",
        "episodic_separation_completion",
        "episodic_replay_reuse",
        "contextual_gate_routing",
    ),
    "hard_symbolic_nm": (
        "nm_hard_symbolic_test_material",
    ),
    "precompute": (
        "capacity_scaling",
        "rate_coded_spike",
        "hierarchical_ternary",
        "asymmetric_outer_product_recall",
        "correction_field_capacity",
        "correction_field_trained_prediction",
        "multi_resolution_head_split",
        "wta_dynamics",
        "lif_fi_curve",
        "thinking_loop_prototype",
        "slot_buffer_capacity",
        "slot_surprise_writes",
        "contextual_recall_world",
        "slot_key_interference_sweep",
        "multi_association_recall",
        "delayed_cue_world",
        "episodic_separation_completion",
        "episodic_replay_reuse",
        "contextual_gate_routing",
        "nm_hard_symbolic_test_material",
    ),
}


def get_suite_specs(suite_name: str) -> list[SimulationSpec]:
    if suite_name not in SUITES:
        raise KeyError(f"unknown suite: {suite_name}")
    return [SIMULATION_SPECS[item] for item in SUITES[suite_name]]


def all_specs() -> list[SimulationSpec]:
    return list(SIMULATION_SPECS.values())


def to_jsonable(spec: SimulationSpec) -> dict[str, Any]:
    return {
        "simulation_id": spec.simulation_id,
        "category": spec.category,
        "script_relative_path": spec.script_relative_path,
        "metrics_filename": spec.metrics_filename,
        "artifact_filenames": list(spec.artifact_filenames),
        "smoke_env": dict(spec.smoke_env),
        "required_summary_keys": list(spec.required_summary_keys),
        "minimum_summary_values": [
            {"key": dotted_key, "minimum": minimum}
            for dotted_key, minimum in spec.minimum_summary_values
        ],
        "required_modules": list(spec.required_modules),
    }
