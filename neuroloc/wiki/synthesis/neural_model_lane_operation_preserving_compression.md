# neural model lane: operation-preserving compression

status: current (as of 2026-05-06).

## thesis

compression is accepted only when it preserves the operation the memory object must support. the target is not tensor reconstruction by itself. the target is lower committed bits while preserving routing, recall, reconstruction, replay rewrite, imagined-branch use, world-state update, and action success.

the useful claim is conditional: an operation-preserving compression stack may be novel if the project proves replaceable codecs across memory surfaces under oracle bounds, trainable controls, telemetry, and related-work separation.

## ranked unknowns

1. what oracle compression ratios are possible on the hard symbolic worlds.
2. which task families can support 10x, 100x, or higher useful compression without discarding task state.
3. which compression objects matter first: addresses, payloads, episodes, imagined branches, replay rewrites, or world state.
4. how to define task-relative rate-distortion when the preserved output is an action, not a reconstruction.
5. where the prior-art boundary lies between known vector quantization, latent compression, schema memory, replay, and the project's compound interface.
6. when content-routed sparse read over verbatim context is already sufficient, and when compressed memory objects add useful capability beyond it.

## evidence base

- [[cellular_state_storage_gap_map]] proposes ranked local-state candidates that may affect write frequency, active forgetting, and useful bits per episode.
- [[neural_model_dossier_eligibility_gated_local_commit]] defines the first candidate whose compression claim depends on fewer committed bits at equal task success.
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]] defines the first mechanism-specific bit fields and storage policies for oracle compression.
- [[tests/eligibility_gated_local_commit_test_material]] documents the implemented mechanism-specific symbolic/oracle surface for those counters.
- [[neural_model_compression_stack]] defines the current stack-wide compression contract.
- [[oracle_compression_analysis_plan]] records the oracle-bound analysis sequence and the handoff from completed counters to the repaired source-observability contract and current learned-generalization repair.
- [[tests/oracle_compression_analysis_results]] documents the first oracle-bound result: clean controls, no leakage, eight strong families, and six weak families below 10x.
- [[oracle_compression_frontier_split]] separates accepted frontier families from weak frontier families and ranks the next narrow learned-codec proof-package candidates.
- [[neural_model_dossier_compression_under_bit_budget_codec]] defines the first narrow learned-codec proof package before implementation.
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]] defines the local implementation contract for the first tiny learned-codec mirror.
- [[tests/compression_under_bit_budget_mirror]] documents the first local dataset, guard, baseline, learned-codec, source-diagnostic, bit-accounting, and telemetry surface for that mirror. the source-observability contract is repaired, the visible-source codec solves the smoke task, and the learned-codec smoke result remains negative on held-out splits because address, payload color, payload position, velocity, action, and learned decoder generalization still fail.
- [[content_routed_sparse_read_prior]] records the 2026 subq / selective sparse attention prior-art boundary. it validates content-dependent routing and functional context as load-bearing problems, but it does not prove memory-object compression, replay compression, imagined-branch compression, or local-neuron storage. it adds a required baseline family for future compression claims.
- [[neural_model_related_work_pressure_matrix]] consolidates the six-lane prior-art pass and sharpens the operation-preserving compression boundary: task-relative compression, latent/context compression, vector/cache compression, and schema/provenance engineering are all prior-art-covered pieces; the remaining project edge is a learned memory-object code that preserves operations under exact controls.
- [[indexed_reconstruction_compression]] defines compact handles, schema or residual codes, provenance, and reconstruction.
- [[neural_model_dossier_compression]] defines the mechanism dossier for compression claims.
- [[tests/hard_symbolic_nm_test_material]] provides the symbolic worlds that expose hidden state, controls, and bit-budget tasks.

## proof gates

- compute verbatim trace bits, latent-state bits, schema/residual bits, and imagined-branch program bits per family.
- report ratios only beside preserved operations and control behavior.
- reject any ratio that drops task-relevant state.
- compare against verbatim storage, no-memory, recency-only, shuffled-address, and family-specific controls.
- when a verbatim memory field is available, compare against content-routed sparse read before making a strong compression claim.
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

## current result

the first oracle compression analysis and frontier split are implemented. useful compression is not uniform across the symbolic surfaces. hard-profile controls are clean and leakage-free, but direct associative recall, correlated-key interference, delayed relevance local commit, bounded output exposure, crossed commit/exposure split, and commit compression frontier remain weak under the current 10x threshold. the first learned-codec mirror for `compression_under_bit_budget` fails held-out operation preservation. the source-observability contract is now clean: source event observed rate, required fields visible rate, source state reconstructable rate, and visible-source-codec joint success are all 1.0. remaining failure is learned address, payload color, payload position, velocity, action, and decoder generalization.

the subq prior-art update does not change that result. it tightens the next comparison standard: future compression wins must beat or complement a content-routed sparse-read baseline where that baseline is fair.

## next action

repair the local learned address/payload/action and decoder-generalization path documented in [[tests/compression_under_bit_budget_mirror]] before any global mirror, full model path, or paid compute. do not treat the visible-source diagnostic as a learned-compression result.

after a clean local learned result, add the content-routed sparse-read baseline to the relevant symbolic/mirror surfaces before expanding the claim.

## see also

- [[PROJECT_PLAN]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[tests/compression_under_bit_budget_mirror]]
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
- [[content_routed_sparse_read_prior]]
- [[neural_model_related_work_pressure_matrix]]
