# neural model lane: memory, addressing, replay, and imagination

status: current (as of 2026-05-08).

## thesis

memory, replay, and imagination are one latent-state manipulation lane. memory writes store task state. addressing retrieves the right object. replay rewrites or consolidates prior state. imagination creates branch-local latent continuations and stores the useful branch object, not a full generated trace.

the claim is conditional: this lane matters only if targeted routing, replay rewrite, and imagined-branch use beat strict controls and preserve state/action/joint success.

## ranked unknowns

1. how learned write decisions avoid the closed-gate fixed point seen in prior slot runs.
2. what address structure prevents correlated-key interference and one-slot collapse.
3. whether replay can reduce bits while preserving provenance and task state.
4. whether imagination should be tested as latent rollout, recombination, reconstruction, or action-improvement.
5. whether targeted replay and imagined branches can improve hard cases more than easy cases.

## evidence base

- [[neural_model_dossier_memory_formation]] defines write decisions and gate fixed points.
- [[neural_model_dossier_addressing]] defines address margin, slot entropy, and shuffled-address controls.
- [[neural_model_dossier_replay_rewrite]] defines replay as rewrite and recompression.
- [[world_models_imagination_and_planning]] translates latent rollout and world-model work into the project frame.
- [[synthetic_shared_world_bridge]] defines the exact-state phase-2 bridge for embodied worlds.
- [[tests/local_v1_language_model]] documents the first constrained local v1 update, targeted replay, random-replay, and branch-state rehearsal gate. it is dataset-record state routing, not open-ended imagination.

## proof gates

- require oracle-write and learned-read, learned-write and oracle-read, hand-opened gate, and shuffled-address splits.
- require targeted replay to beat random replay and no replay.
- require imagined-branch programs to reconstruct outcomes or improve action success under uncertainty.
- require hard-case rollout gain to exceed easy-case gain.
- log write frequency, gate-open fraction, address margin, read concentration, replay target selection, branch uncertainty, and retained provenance.

## side-paper candidates

- replay rewrite as operation-preserving memory compression, if targeted replay lowers bits without corrupting state or provenance.
- imagined branches as compact latent programs, if branch objects improve action selection and survive reconstruction controls.
- trainability-localized addressing, if oracle splits isolate the learned-write or learned-read failure.

## kill conditions

- replay works only when the target is oracle supplied.
- random replay matches targeted replay.
- imagined branches improve easy cases only.
- branch storage loses uncertainty or provenance.
- address-dependent tasks pass under shuffled addresses.
- write/read telemetry shows the intended memory path is unused.

## next action

harden the local v1 update/replay/branch-state loop with longer grounded dialogue, provenance-preserving rewrites, factor-heldout query forms, larger local dataset ingestion, and fair sparse-read baselines before expanding to embodied 3d tasks.

## see also

- [[PROJECT_PLAN]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
- [[neural_model_dossier_memory_formation]]
- [[neural_model_dossier_addressing]]
- [[neural_model_dossier_replay_rewrite]]
- [[tests/local_v1_language_model]]
- [[world_models_imagination_and_planning]]
- [[synthetic_shared_world_bridge]]
