# neural model paper spine

status: current (as of 2026-05-08).

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

proof state: partially bounded with one ordinary-split local learned sanity pass, one factor-heldout falsification, one factorized structured local repair, one 10k-scale local data-heavy robustness gate, one structured constrained-language bridge, one negative parser-resistant learned token-count gate, one positive parser-supported event-binding baseline, one typed trainable event-binding local-state pass, and one dataset-grounded local v1 state responder. the first oracle compression result and frontier split exist, but they are mixed across families. the first learned codec for the original `compression_under_bit_budget` source-pair task fails held-out operation preservation, and the first content-routed sparse-read baseline solves that repaired source-pair smoke task from two legal raw records while exceeding the compact-code budget. matched-budget sparse read fails, and the distributed-evidence probe is solved by four-record sparse read but fails under the matched compact budget. the shared nonlinear tiny model reaches 1.0 ordinary-split test joint/state/action success at 19 bits but falls to 0.03125 on held-out color-shape pair bands. after prosecutor review removed an evaluator-source shortcut, the 9,792-parameter factorized structured local codec reaches minimum test joint/state/action success 1.0 across four heldout axes and three smoke seeds at 19 bits, with useful operation success per committed bit 0.05263157894736842. the 3,847-parameter structured language-grounded bridge reaches the same useful operation density from generated text observations and text queries across four heldout axes and two smoke seeds. the parser-resistant learned token-count extension fails: a 1,779-parameter encoder with bounded local state reaches minimum test joint/state success 0.0 on randomized prompts while uncapped sparse read remains 1.0. the event-binding parser baseline then restores constrained randomized-message answering by preserving event segments as bounded local state: minimum test joint/state/action success 1.0, 19 state bits plus 37 parser/schema bits for 56 accounted bits, matched-budget sparse read 0.0 at 20 bits, zero-state joint max 0.0, shuffled-state joint max 0.020833333333333332, and parser-supported foundation pass 1.0. the typed trainable segment binder reaches minimum test joint/state/action success 0.9583333333333334 with 8,856 parameters and engineering pass 1.0 on the same randomized symbolic-message gate. the first local v1 language-state gate trains a 2,688-parameter state-first router over 32 dataset-derived memory records, reaches minimum joint/state/action/provenance success 1.0, collapses under zero-state and shuffled-state controls, and adds constrained memory update, targeted replay, random-replay failure, and branch-state rehearsal controls. the operation-preserving storage/compression stack remains unproved beyond local symbolic diagnostics.

2026-05-06 related-work boundary: subquadratic's public subq / selective sparse attention material strengthens the prior-art baseline for content-dependent sparse reads over long context. this validates routing and functional-context concerns, but it also means any strong compression claim must beat or complement content-routed sparse read over verbatim context where that baseline is well-defined. the broader multi-lane pressure matrix also shows that local state, differentiable memory, sparse routing, semantic compression, replay, world models, and oracle-style evaluations are all prior-art-covered pieces. source: [[content_routed_sparse_read_prior]] and [[neural_model_related_work_pressure_matrix]].

### claim 4: lane research and oracle compression bounds must come before training

if the research lanes cannot state their proof obligations, the project should not train a mirror. the first biology-led lane gap map is [[cellular_state_storage_gap_map]]. if the hard symbolic worlds do not admit strong oracle compression ratios, a trained model should not be expected to discover extreme compression there. source and next plan: [[neural_model_lane_cellular_state_storage]], [[oracle_compression_analysis_plan]], and [[tests/hard_symbolic_nm_test_material]].

proof state: partially bounded. the first cellular mechanism dossier is [[neural_model_dossier_eligibility_gated_local_commit]], the first mechanism-specific symbolic contract is [[neural_model_symbolic_contract_eligibility_gated_local_commit]], the first oracle compression result is [[tests/oracle_compression_analysis_results]], and the resulting family split is [[oracle_compression_frontier_split]]. the bound is mixed: controls and leakage are clean, eight families clear the current strong-oracle threshold, and six families remain below 10x.

### claim 5: learned compression must beat controls

after oracle bounds exist, a tiny trainable neural-model mirror must learn at least one non-oracle codec above no-memory, recency-only, shuffled-address, and verbatim-storage controls before any full model integration. source: [[neural_model_research_test_material_plan]].

proof state: locally trained and diagnostically localized, with one ordinary-split local pass, one recombination failure, one factorized local repair, one 10k-scale local data-heavy pass, one structured constrained-language bridge, one parser-resistant learned token-count negative result, one parser-supported event-binding baseline pass, one typed trainable event-binding pass, and one dataset-grounded local v1 state-first responder pass. the first family-specific proof package is [[neural_model_dossier_compression_under_bit_budget_codec]], the first implementation contract is [[neural_model_tiny_mirror_contract_compression_under_bit_budget]], the local compact-state surface is [[tests/compression_under_bit_budget_mirror]], the first message-response bridge is [[tests/language_grounded_state_density_mirror]], and the first local v1 responder card is [[tests/local_v1_language_model]]. the source-observability contract is repaired: the legal visible-source codec reaches 1.0 joint/state/action success on the smoke suite. the content-routed sparse-read baseline also reaches 1.0 by selecting two legal observation records, committing 40 bits, and compressing verbatim storage by 1.3x, but it is outside the compact-code budget. matched-budget sparse read commits 20 bits and fails. the distributed-evidence probe removes commit markers and splits the answer across four legal fragments; uncapped sparse read solves at 80 bits, while matched-budget sparse read fails at 20 bits. a tiny shared-trunk local model learns the ordinary split but fails held-out color-shape pair bands. a 9,792-parameter factorized structured local codec clears four heldout factor axes across three smoke seeds at minimum test joint success 1.0 and 19 committed bits. a 3,847-parameter structured language-grounded bridge clears four heldout axes across two smoke seeds from generated text observations and text queries. the parser-resistant learned token-count gate fails with minimum learned joint/state success 0.0. the event-binding parser baseline clears the randomized message surface by committing event segments into bounded local state and answering from that state. the typed trainable segment binder then clears the same gate with 8,856 parameters, minimum test joint/state/action success 0.9583333333333334, field floor 0.9583333333333334, and engineering pass 1.0. the local v1 language-state gate trains on dataset-derived memory records, answers from bounded state rather than next-token decoding, clears its smoke controls with 2,688 parameters, and includes constrained update/replay/branch-state dialogue controls. this is constrained local-state evidence, not arbitrary chat, not solved compression, and not a paper claim.

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
10. comparison against content-routed sparse read over verbatim context for families where sparse read is a fair baseline.

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

the first executable model experiment is not a full model run. the cellular/local-state gap map, first cellular mechanism dossier, symbolic/oracle contract implementation, first oracle compression analysis, frontier split, `compression_under_bit_budget` proof package, tiny mirror contract, local mirror baseline surface, first negative learned-codec smoke result, source-diagnostic pass, source-observability repair, content-routed sparse-read baseline, matched-budget sparse-read control, distributed-evidence probe, tiny distributed local learned model, factor-heldout falsification gate, factorized structured local codec, 10k-scale local data-heavy gate, structured constrained-language bridge, parser-resistant learned token-count negative gate, parser-supported event-binding baseline, typed trainable event-binding pass, and dataset-grounded local v1 state responder come first. the local v1 responder answers from bounded dataset memory state and has a first constrained update/replay/branch-state loop, but it still lacks longer dialogue, provenance-preserving rewrites, factor-heldout query forms, broad dataset ingestion, and arbitrary-language grounding. full integration and paid compute remain blocked.

load-bearing pages: [[oracle_compression_analysis_plan]], [[phase1_evaluation_surface_for_neural_models]], [[synthetic_shared_world_bridge]].

### novelty

state only conditional novelty: the stack may be novel if it proves replaceable operation-preserving codecs across cache, address, payload, episode, imagination, replay-rewrite, and world-state surfaces under strict controls.

load-bearing pages: [[neural_model_compression_stack]], [[indexed_reconstruction_compression]].

related-work boundary: [[content_routed_sparse_read_prior]].

### limitations

state that extreme lossless compression is impossible on arbitrary high-entropy data. large ratios are plausible only when the world has repeated structure, shared dynamics, reusable schemas, and compact latent state.

load-bearing pages: [[oracle_compression_analysis_plan]], [[synthetic_shared_world_bridge]], [[world_models_imagination_and_planning]].

## not proved yet

- the first oracle compression ratio table and frontier split exist, but they are mixed: eight strong families and six weak families on the hard profile.
- one narrow structured local codec beats matched-budget sparse read on a symbolic surface, but no learned codec has beaten all fair verbatim/content-routed baselines under strict bit accounting.
- the first tiny trainable mirror has failed the compression path on held-out smoke: train joint success 1.0, validation joint success 0.0, test joint success 0.0, measured learned ratio 2.74x after action is counted, engineering pass 0.0.
- no evidence yet shows replay rewrite reducing bits while preserving state/action/joint success.
- imagined-branch program bits have an oracle bound on the current symbolic worlds, but no learned model has inferred or used that code.
- no evidence yet shows a swappable codec interface working across memory levels.
- no related-work section has fully separated prior art from the conditional new claim.
- content-routed sparse read is now a required prior-art baseline where a verbatim memory field exists.
- the first implemented content-routed sparse-read baseline solves the repaired `compression_under_bit_budget` source-pair smoke task, so that mirror is demoted as compression evidence. the learned codec must beat or complement sparse read under useful-bit accounting on the distributed-evidence slice before this family supports a compression claim.
- the related-work pressure matrix exists, but its requirements have not yet been implemented into every symbolic/mirror family.
- the first candidate proof package for `compression_under_bit_budget` has produced a repaired source-observability result, a sparse-read demotion of the source-pair task, a distributed-evidence probe, a negative learned result on the original small-data source-pair codec, and a narrow positive tiny local result on distributed evidence. the next obligation is stronger local falsification rather than claim expansion.
- the first local v1 responder trains on dataset-derived records, answers from bounded state, and passes constrained update/replay/branch-state controls, but it is still a state router over extracted memory records, not a general chat model or an open-ended imagination system.
- no paid compute is authorised by this paper spine.

## next research action

the first biology-led lane gap map is [[cellular_state_storage_gap_map]], the first cellular mechanism dossier is [[neural_model_dossier_eligibility_gated_local_commit]], the first mechanism-specific symbolic contract is [[neural_model_symbolic_contract_eligibility_gated_local_commit]], the first oracle compression result is [[tests/oracle_compression_analysis_results]], the frontier split is [[oracle_compression_frontier_split]], the first narrow learned-codec proof package is [[neural_model_dossier_compression_under_bit_budget_codec]], the tiny mirror contract is [[neural_model_tiny_mirror_contract_compression_under_bit_budget]], and the first local mirror surface is [[tests/compression_under_bit_budget_mirror]].

next, harden the local v1 responder inside the useful-state-density local mirror. source state and action target are legally inferable from non-oracle input, but the source-pair task collapses to shallow sparse read. the ordinary distributed-evidence result passes, the shared-trunk factor-heldout result fails, the factorized structured local codec clears four heldout factor axes across three smoke seeds, the structured language bridge clears constrained generated messages across two smoke seeds, the parser-resistant learned token-count encoder fails randomized prompts with minimum joint/state success 0.0, the parser-supported event-binding baseline clears the same randomized surface by answering from bounded event state, the typed trainable segment binder clears the local foundation gate, and the first dataset-grounded local v1 state router answers from local memory records with 2,688 parameters plus constrained update/replay/branch-state controls. the next localization target is longer grounded dialogue, provenance-preserving rewrites, factor-heldout query forms, larger local dataset ingestion, and fair sparse-read baselines while preserving matched-budget sparse read, uncapped sparse read, zero-state ablation, shuffled-state ablation, random-replay controls, useful-state-density accounting, and arbitrary-chat denial. if that cannot survive, keep the contract demoted before any model path.

after a clean local learned-codec result, extend the same sparse-read comparison to the other relevant symbolic/mirror families before expanding the compression claim. this comparison is required by the 2026-05-06 subq related-work update and does not authorize paid compute.

## see also

- [[oracle_compression_analysis_plan]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[tests/compression_under_bit_budget_mirror]]
- [[tests/language_grounded_state_density_mirror]]
- [[tests/local_v1_language_model]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
- [[neural_model_compression_stack]]
- [[content_routed_sparse_read_prior]]
- [[neural_model_related_work_pressure_matrix]]
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
