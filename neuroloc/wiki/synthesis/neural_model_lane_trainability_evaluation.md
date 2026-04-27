# neural model lane: trainability and evaluation

status: current (as of 2026-04-27).

## thesis

no mechanism is accepted because it is plausible. it must be localized by controls. the project needs tests that identify whether failure lives in learned writes, learned reads, gates, addresses, decoder use, gradient flow, interference, or the task itself.

the first executable model target remains a tiny trainable neural-model mirror, but only after the broad research gates, first-mechanism symbolic/oracle test material, and oracle compression bounds define what it must learn.

## ranked unknowns

1. which controls best isolate learned-write, learned-read, gate, address, and decoder failure.
2. what telemetry proves the intended path is actually used.
3. how to set confidence intervals and seed sweeps so small wins are not accepted as evidence.
4. which symbolic tasks are hard enough to block leakage but still trainable by a tiny mirror.
5. how to prevent passkey, next-token loss, or any single metric from becoming the whole evaluation.

## evidence base

- [[cellular_state_storage_gap_map]] identifies the first local-state mechanisms that need oracle/write/read/gate controls before implementation.
- [[neural_model_dossier_eligibility_gated_local_commit]] defines the first specific trace, commit, and output-exposure localization problem.
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]] defines the phase-localized symbolic contract that must precede the tiny mirror.
- [[neural_model_research_test_material_plan]] defines dossier, control, and metric requirements.
- [[phase1_evaluation_surface_for_neural_models]] defines the broader state/action evaluation frame.
- [[tests/hard_symbolic_nm_test_material]] documents the implemented hard symbolic package.
- [[substrate_requires_architectural_change]] records the failed paid substrate evidence and ranked paused interventions.
- [[neural_model_dossier_trainability]] defines trainability controls and telemetry.

## proof gates

- require oracle-write and learned-read, learned-write and oracle-read, no-memory, recency-only, shuffled-address, hand-opened gate, orthogonal-address initialization, and matched compute/parameter controls.
- require state_probe_accuracy, action_success, joint_success, recall, interference slope, reuse advantage, hard-case rollout gain, and bits per successful episode where relevant.
- require confidence intervals and deterministic seed sets.
- require telemetry: gate-open fraction, memory-output norm versus residual norm, address entropy, address margin, write frequency, read concentration, retention, compression budget, reconstruction error.
- require prosecutor-clean docs before any paid compute gate.

## side-paper candidates

- hard symbolic neural-model test material, if the package becomes a reusable benchmark with non-leaky controls.
- trainability failure localization for neural memory, if oracle splits explain which learned component fails.
- control methodology for memory and imagination mechanisms, if it prevents false positives across mechanisms.

## kill conditions

- a mechanism improves loss but not state/action/joint success.
- telemetry shows gates opening into noise or memory output unused by the residual stream.
- the effect appears only with oracle components.
- results do not beat no-memory, recency-only, or shuffled-address controls.
- confidence intervals overlap noise.
- passkey is the only persisted success metric.

## next action

implement symbolic/oracle tests for [[neural_model_symbolic_contract_eligibility_gated_local_commit]] before the tiny-mirror target, while keeping oracle compression bounds ahead of any full-model path.

## see also

- [[PROJECT_PLAN]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[tests/hard_symbolic_nm_test_material]]
- [[substrate_requires_architectural_change]]
- [[neural_model_dossier_trainability]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_project_operations]]
