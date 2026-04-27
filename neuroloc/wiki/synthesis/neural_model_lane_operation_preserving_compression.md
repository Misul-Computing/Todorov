# neural model lane: operation-preserving compression

status: current (as of 2026-04-27).

## thesis

compression is accepted only when it preserves the operation the memory object must support. the target is not tensor reconstruction by itself. the target is lower committed bits while preserving routing, recall, reconstruction, replay rewrite, imagined-branch use, world-state update, and action success.

the useful claim is conditional: an operation-preserving compression stack may be novel if the project proves replaceable codecs across memory surfaces under oracle bounds, trainable controls, telemetry, and related-work separation.

## ranked unknowns

1. what oracle compression ratios are possible on the hard symbolic worlds.
2. which task families can support 10x, 100x, or higher useful compression without discarding task state.
3. which compression objects matter first: addresses, payloads, episodes, imagined branches, replay rewrites, or world state.
4. how to define task-relative rate-distortion when the preserved output is an action, not a reconstruction.
5. where the prior-art boundary lies between known vector quantization, latent compression, schema memory, replay, and the project's compound interface.

## evidence base

- [[cellular_state_storage_gap_map]] proposes ranked local-state candidates that may affect write frequency, active forgetting, and useful bits per episode.
- [[neural_model_dossier_eligibility_gated_local_commit]] defines the first candidate whose compression claim depends on fewer committed bits at equal task success.
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]] defines the first mechanism-specific bit fields and storage policies for oracle compression.
- [[tests/eligibility_gated_local_commit_test_material]] documents the implemented mechanism-specific symbolic/oracle surface for those counters.
- [[neural_model_compression_stack]] defines the current stack-wide compression contract.
- [[oracle_compression_analysis_plan]] defines the next oracle-bound analysis.
- [[indexed_reconstruction_compression]] defines compact handles, schema or residual codes, provenance, and reconstruction.
- [[neural_model_dossier_compression]] defines the mechanism dossier for compression claims.
- [[tests/hard_symbolic_nm_test_material]] provides the symbolic worlds that expose hidden state, controls, and bit-budget tasks.

## proof gates

- compute verbatim trace bits, latent-state bits, schema/residual bits, and imagined-branch program bits per family.
- report ratios only beside preserved operations and control behavior.
- reject any ratio that drops task-relevant state.
- compare against verbatim storage, no-memory, recency-only, shuffled-address, and family-specific controls.
- train a tiny mirror only after oracle ratios justify the family.

## side-paper candidates

- operation-preserving compression stack, if the oracle and learned results show a Pareto improvement over verbatim storage.
- task-relative rate-distortion for memory objects, if it cleanly predicts state/action success under bit budgets.
- replay and imagination codecs, if branches and rewrites are stored as compact programs rather than traces.

## kill conditions

- oracle ratios are weak on the constructed worlds.
- compression wins only because the evaluator supplied schema labels unavailable to a model.
- reconstruction improves while action success or joint success falls.
- compressed codes beat no-memory but fail against verbatim storage or shuffled-address controls.
- learned codecs do not approach the oracle direction in the tiny mirror.

## next action

run the oracle compression analysis with [[tests/eligibility_gated_local_commit_test_material]] as the first mechanism-specific input, especially for write frequency, bounded output exposure, and commit compression frontier.

## see also

- [[PROJECT_PLAN]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[neural_model_dossier_compression]]
- [[tests/hard_symbolic_nm_test_material]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
