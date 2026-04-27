# neural model paper spine

status: current (as of 2026-04-27).

## master-plan role

this page is the flagship-paper spine inside the broader six-lane master plan in [[PROJECT_PLAN]]. it is not the whole project plan. the paper spine depends on the lane pages for proof obligations, especially cellular state storage, operation-preserving compression, memory/replay/imagination, 3d world physics, trainability/evaluation, and project operations.

## abstract-level thesis

the paper argument is not that the project has already solved memory compression. the argument is conditional:

```text
if a neural model can compress each memory surface by preserving the operations that surface must support, then memory, imagination, replay, and world-state storage can be evaluated as one replaceable compression stack instead of as unrelated tricks.
```

the useful novelty would be the compound proof, not any isolated component. vector quantization, sparse coding, latent world models, indexing, residual coding, and replay all have prior art. the project-specific claim becomes novel if proved only when those pieces are bound into one audited interface and shown to preserve state, action, reconstruction, replay, and imagined-branch operations per committed bit.

## paper claims

### claim 1: the empirical motivation is negative

existing paid runs are evidence about failed trainability and failed retrieval, not a working memory result. six paid runs produced 0 percent passkey at 256 tokens across matrix and slot substrates, broken and fixed retention regimes, and natural-text plus synthetic-cognition corpora. source: [[substrate_requires_architectural_change]].

proof state: established as project evidence, but not a general theorem about all memory architectures.

### claim 2: the active object is the neural model

the old architecture names and run sequence are historical evidence. the current object is a neural model with explicit mechanisms for local state, addressing, writing, reading, compression, reconstruction, replay, rollout, and trainability. source: [[neural_model_research_test_material_plan]].

proof state: canonical project framing, not a scientific result.

### claim 3: compression must be operation-preserving

the compression target is not tensor reconstruction. for each memory surface, define the operations that must remain correct, then minimize committed bits under those operations. source: [[neural_model_compression_stack]] and [[indexed_reconstruction_compression]].

proof state: theory proposal. not yet validated by oracle bounds or learned compression.

### claim 4: lane research and oracle compression bounds must come before training

if the research lanes cannot state their proof obligations, the project should not train a mirror. the first biology-led lane gap map is [[cellular_state_storage_gap_map]]. if the hard symbolic worlds do not admit strong oracle compression ratios, a trained model should not be expected to discover extreme compression there. source and next plan: [[neural_model_lane_cellular_state_storage]], [[oracle_compression_analysis_plan]], and [[tests/hard_symbolic_nm_test_material]].

proof state: partially bounded. the first cellular mechanism dossier is [[neural_model_dossier_eligibility_gated_local_commit]], the first mechanism-specific symbolic contract is [[neural_model_symbolic_contract_eligibility_gated_local_commit]], the first oracle compression result is [[tests/oracle_compression_analysis_results]], and the resulting family split is [[oracle_compression_frontier_split]]. the bound is mixed: controls and leakage are clean, eight families clear the current strong-oracle threshold, and six families remain below 10x.

### claim 5: learned compression must beat controls

after oracle bounds exist, a tiny trainable neural-model mirror must learn at least one non-oracle codec above no-memory, recency-only, shuffled-address, and verbatim-storage controls before any full model integration. source: [[neural_model_research_test_material_plan]].

proof state: scoped but not trained. the first family-specific proof package is [[neural_model_dossier_compression_under_bit_budget_codec]], and the first implementation contract is [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]. no trained mirror result exists yet.

## missing proof obligations

1. oracle compression bounds on each hard symbolic family.
2. a clear separation between verbatim trace bits, latent-state bits, schema/residual bits, and imagined-branch program bits.
3. at least one learned codec that approaches the oracle direction without oracle schema labels; the first scoped candidate is [[neural_model_dossier_compression_under_bit_budget_codec]] under [[neural_model_tiny_mirror_contract_compression_under_bit_budget]].
4. evidence that compressed storage preserves operations, not only reconstructions.
5. evidence that replay can rewrite memories into smaller forms without losing task-relevant state.
6. evidence that imagined branches can be stored as latent programs and reused for action.
7. telemetry proving the intended path is used: address margin, read concentration, write frequency, gate-open fraction, reconstruction error, and confidence intervals.
8. related-work separation showing which pieces are prior art and what remains novel if proved.
9. limitations for high-entropy data, unbounded external memory, provenance failure, decoder leakage, and trainability collapse.

## novelty side-track rule

building the neural model may expose mechanisms that deserve separate papers. that is expected. the rule is that a side-track becomes real research only when it is packaged as a proof obligation, not when it feels new.

every candidate novelty must state:

- the mathematical operation
- the prior-art boundary
- what operation it preserves or improves
- what oracle or symbolic bound should hold before training
- what trained test would prove it is usable
- what control would falsify it
- what telemetry proves the path is actually used
- what result kills the idea

until those fields exist, the candidate stays a note or open question. after those fields exist, it can become a side-track paper without derailing the main neural-model spine.

## section map

### problem

start from the empirical failure: language-model loss and retrieval-shaped synthetic data did not make the prior substrates learn useful retrieval. establish why another paid run is not the right next action.

load-bearing pages: [[substrate_requires_architectural_change]], [[phase1_evaluation_surface_for_neural_models]], [[tests/hard_symbolic_nm_test_material]].

### theory

define memory surfaces and their required operations. state the operation-preserving codec contract and rate-distortion objective.

load-bearing pages: [[neural_model_compression_stack]], [[indexed_reconstruction_compression]], [[neural_model_dossier_compression]], [[neural_model_dossier_reconstruction]], [[neural_model_dossier_replay_rewrite]].

### methods

use hard symbolic worlds with exact hidden state and deterministic controls. compute oracle bit accounts before training.

load-bearing pages: [[oracle_compression_analysis_plan]], [[tests/hard_symbolic_nm_test_material]], [[neural_model_research_test_material_plan]].

### experiments

the first executable model experiment is not a full model run. the cellular/local-state gap map, first cellular mechanism dossier, symbolic/oracle contract implementation, first oracle compression analysis, frontier split, `compression_under_bit_budget` proof package, and tiny mirror contract come first. because the first analysis has weak families, the next model experiment is local only: a tiny trainable mirror for the accepted `compression_under_bit_budget` family. full integration and paid compute remain blocked.

load-bearing pages: [[oracle_compression_analysis_plan]], [[phase1_evaluation_surface_for_neural_models]], [[synthetic_shared_world_bridge]].

### novelty

state only conditional novelty: the stack may be novel if it proves replaceable operation-preserving codecs across cache, address, payload, episode, imagination, replay-rewrite, and world-state surfaces under strict controls.

load-bearing pages: [[neural_model_compression_stack]], [[indexed_reconstruction_compression]].

### limitations

state that extreme lossless compression is impossible on arbitrary high-entropy data. large ratios are plausible only when the world has repeated structure, shared dynamics, reusable schemas, and compact latent state.

load-bearing pages: [[oracle_compression_analysis_plan]], [[synthetic_shared_world_bridge]], [[world_models_imagination_and_planning]].

## not proved yet

- the first oracle compression ratio table and frontier split exist, but they are mixed: eight strong families and six weak families on the hard profile.
- no learned codec has beaten verbatim storage under a strict bit budget.
- no tiny trainable mirror has learned the compression path.
- no evidence yet shows replay rewrite reducing bits while preserving state/action/joint success.
- imagined-branch program bits have an oracle bound on the current symbolic worlds, but no learned model has inferred or used that code.
- no evidence yet shows a swappable codec interface working across memory levels.
- no related-work section has fully separated prior art from the conditional new claim.
- the first candidate proof package exists for `compression_under_bit_budget`, but it has not produced a learned result.
- no paid compute is authorised by this paper spine.

## next research action

the first biology-led lane gap map is [[cellular_state_storage_gap_map]], the first cellular mechanism dossier is [[neural_model_dossier_eligibility_gated_local_commit]], the first mechanism-specific symbolic contract is [[neural_model_symbolic_contract_eligibility_gated_local_commit]], the first oracle compression result is [[tests/oracle_compression_analysis_results]], the frontier split is [[oracle_compression_frontier_split]], the first narrow learned-codec proof package is [[neural_model_dossier_compression_under_bit_budget_codec]], and the tiny mirror contract is [[neural_model_tiny_mirror_contract_compression_under_bit_budget]].

next, implement the tiny local mirror for `compression_under_bit_budget`. if the implementation cannot keep non-oracle inputs, controls, telemetry, confidence intervals, and acceptance thresholds clean, the project should not move the result into the paper spine.

## see also

- [[oracle_compression_analysis_plan]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_dossier_compression]]
- [[neural_model_dossier_reconstruction]]
- [[neural_model_dossier_replay_rewrite]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[synthetic_shared_world_bridge]]
- [[world_models_imagination_and_planning]]
- [[substrate_requires_architectural_change]]
- [[tests/hard_symbolic_nm_test_material]]
- [[PROJECT_PLAN]]
- [[INDEX]]
