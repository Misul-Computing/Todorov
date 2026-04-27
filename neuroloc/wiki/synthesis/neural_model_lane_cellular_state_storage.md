# neural model lane: cellular state storage

status: current (as of 2026-04-27).

## thesis

the biology-led entry point is local state storage. the project needs to translate cellular, molecular, membrane, subthreshold, compartment, eligibility, and gating phenomena into candidate mathematical operations before deciding which of them belong in a trainable neural model.

the claim is not that biology should be copied. the claim is conditional: a cellular mechanism is useful only if it becomes a compact operation with measurable retention, routing, compression, or trainability value under strict controls.

## ranked unknowns

1. which local state variables can store task-relevant information over useful delays without requiring unobservable labels.
2. whether polarity-separated state, membrane summaries, eligibility traces, and compartment state preserve different operations or merely duplicate capacity.
3. whether local state can reduce committed memory bits, or only stabilize writing and reading.
4. what gating rule prevents local state from becoming closed, noisy, or unused.
5. which biological details are computation-relevant and which are implementation baggage.

## evidence base

- [[cellular_state_storage_gap_map]] records the first ranked no-paid-compute synthesis for this lane.
- [[neural_model_dossier_eligibility_gated_local_commit]] records the first cellular mechanism dossier: delayed write permission plus bounded output exposure.
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]] records the symbolic/oracle contract for the first cellular mechanism.
- [[cellular_molecular_computational_primitives]] maps candidate biological primitives into computation-filtered operations.
- [[cellular_molecular_neurobiology_research]] records the source shelf for dendrites, receptors, glia, and biochemical control motifs.
- [[indexed_reconstruction_compression]] records the recent source check: complemented-neuron ternary spiking models and neuronspark-style membrane state support preserving subthreshold state around discrete events, but do not prove retrieval capacity for this project.
- [[neural_model_dossier_local_neuron_state]] defines the current mechanism-dossier contract for local state.

## proof gates

- write a cellular-state mechanism dossier for each candidate operation before implementation.
- define the preserved operation: retention, address separation, write gating, reconstruction, compression, or trainability stabilization.
- run symbolic or oracle checks that show the operation is needed by the task rather than supplied by leakage.
- require a tiny trainable mirror to show learned use above no-local-state and matched-parameter controls before full-model integration.
- log local-state norms, persistence, gate-open fraction, gradient flow, and contribution to state/action/joint success.

## side-paper candidates

- cellular local state as task-relative memory storage, if a learned mechanism beats matched controls.
- polarity-separated or membrane-summary state as an operation-preserving code, if it improves retention or compression without hidden oracle labels.
- local trace gates for trainable memory formation, if they localize learned-write failure.

## kill conditions

- the state persists but does not improve state/action/joint success.
- gains disappear when parameter and compute budgets are matched.
- the mechanism only works with oracle labels, hand-opened gates, or hand-placed addresses.
- telemetry shows closed gates, noise-scale outputs, or unused gradients.
- the operation is indistinguishable from a simpler learned residual state.

## next action

implement [[neural_model_symbolic_contract_eligibility_gated_local_commit]] as symbolic/oracle test material before any model code changes.

## see also

- [[PROJECT_PLAN]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
- [[cellular_molecular_computational_primitives]]
- [[cellular_molecular_neurobiology_research]]
- [[indexed_reconstruction_compression]]
- [[neural_model_dossier_local_neuron_state]]
