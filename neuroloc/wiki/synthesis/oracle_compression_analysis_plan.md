# oracle compression analysis plan

status: current (as of 2026-04-27).

## purpose

this page defines the compression-lane oracle proof step for the neural-model paper spine. under the 2026-04-26 master plan, the first cellular/local-state gap map, [[neural_model_dossier_eligibility_gated_local_commit]], and [[neural_model_symbolic_contract_eligibility_gated_local_commit]], it ran alongside symbolic/oracle test material for the first mechanism. for future compression families, the same oracle gate remains required before a tiny model is trained to compress memory; for `compression_under_bit_budget`, the first narrow mirror and diagnostic pass have already been run and failed held-out operation preservation.

the analysis asks one question:

```text
what is the smallest task-sufficient code when the evaluator already knows the hidden latent structure?
```

if the oracle cannot compress a family strongly while preserving the required operations, training a model on that family will not produce the desired compression result.

## bit accounts

### verbatim trace bits

the verbatim baseline stores the observed trace directly. for each episode, count every visible field that would need to be retained to answer later queries without a shared world model:

- observation identity fields
- observed attributes
- context tags
- locations or relation fields
- temporal positions
- query-relevant actions
- distractor facts
- payload bytes or symbols

this is the upper-rate baseline. it is not automatically a good memory policy, because it may exceed the budget and increase interference.

### latent-state bits

the latent oracle stores only the ground-truth variables needed to preserve the task operations:

- active entity ids
- task-relevant attributes
- hidden location or state variables
- context route
- relation or binding variables
- delay-relevant state
- dynamics parameters when needed for rollout

this is the cleanest lower bound for structured worlds. it is available only because the symbolic generator exposes exact hidden state.

### schema/residual bits

the schema/residual oracle stores a reusable schema id plus only the fields not reconstructable from that schema:

- schema id
- address or entity handle
- residual attributes
- provenance id
- uncertainty flag when ambiguity remains
- rewrite marker when replay can promote repeated traces into a cheaper schema

this account is the most relevant one for learned memory. it does not require the model to store every observed field, but it does require the decoder or prior to carry reusable structure.

### imagined-branch program bits

for imagination and recombination, the oracle should not store a generated trace. it should store a compact branch program:

- source memory ids
- start-state code
- intervention code
- transition handle
- predicted outcome code
- residual surprises
- uncertainty
- branch provenance

this account tests whether imagined branches can be memory objects rather than unbounded rollouts.

## expected ratio targets by family

these are planning targets, not results.

| family | expected oracle ratio | reason | kill signal |
|---|---:|---|---|
| belief-state formation | 5x-50x | hidden state is smaller than repeated masked observations | latent code needs most of the trace |
| associative recall | 1x-20x | arbitrary payloads compress weakly unless schemas repeat | payload is high entropy and no schema helps |
| correlated-key interference | 1x-10x | this is mainly an address-separation test | compression hides the address failure |
| delayed use | 10x-100x | compact state can replace repeated observations across delay | no-memory or recency-only approaches oracle |
| episodic reuse | 20x-200x | repeated episode structure can be stored as schema plus residual | reuse disappears when trace detail is dropped |
| context-gated routing | 10x-100x | context/action maps can be compact handles | route requires full history |
| compression under bit budget | 10x-300x | constructed to expose schema/residual advantage | verbatim and compressed policies tie |
| replay/rewrite | 20x-300x | repeated traces can be promoted into smaller invariants | targeted replay fails to reduce bits |
| iterative hard-case rollout | 1x-30x | compression is mostly compute/program compression | extra compute helps easy cases only |
| imagination/recombination | 50x-600x | branch program can replace full imagined trace on structured worlds | branch outcome cannot be reconstructed from program plus residual |

no ratio is accepted without preserved task operations. a smaller code that loses the answer is not compression.

## operation checks

each family must state which operations the oracle code preserves:

- route to the right address
- reconstruct the answer or action-relevant state
- resist distractors
- preserve target/non-target separation
- support replay rewrite
- support imagined-branch reconstruction
- support rollout improvement on hard cases

the analysis should report ratios only beside the operations that remain correct.

## controls

the oracle analysis must preserve the existing hard symbolic controls:

- no-memory
- recency-only
- shuffled-address
- random replay
- targeted replay
- verbatim store
- compressed store

compression claims are compared against verbatim storage and against task controls, not only against no-memory.

## kill conditions

stop the compression path for a family if any of these occur:

- the oracle ratio is weak for the claim being made.
- the oracle win depends on labels or schema ids that no trainable model could infer from observations.
- the compressed code preserves reconstruction but loses action success.
- bits drop only because task-relevant state was discarded.
- imagined-branch compression cannot reconstruct outcome or uncertainty.
- replay rewrite reduces bits but corrupts provenance.
- shuffled-address succeeds on an address-dependent claim.

for the main paper direction, compression/imagination/replay families should show at least one strong oracle target before a trained mirror is attempted. if the strongest constructed worlds cannot support around 10x useful compression under oracle access, the project should revise the worlds or abandon the extreme-compression claim for that family.

## expected output

the eventual analysis should produce a machine-readable artifact with one record per task family:

```text
family
seed
difficulty
verbatim_trace_bits
latent_state_bits
schema_residual_bits
imagined_branch_program_bits
oracle_ratio_latent
oracle_ratio_schema_residual
oracle_ratio_branch_program
operations_preserved
control_results
kill_condition
```

the implemented result is [[tests/oracle_compression_analysis_results]].

hard-profile result summary:

- 448 contracts across 14 families and two symbolic surfaces
- operation preservation rate 1.0
- controls preservation rate 1.0
- leakage-free rate 1.0
- accepted rate 0.5714
- eight strong families
- six weak families below the 10x useful-compression threshold
- global tiny-mirror recommendation 0.0

the first result therefore changes the next step from "run the counters" to "split the frontier." accepted families can be considered for narrow learned-codec scoping later. weak families need task revision, codec revision, or demotion of the compression claim before training.

the split is now documented in [[oracle_compression_frontier_split]].

## sequencing

1. completed: confirm the relevant compression questions from [[cellular_state_storage_gap_map]], [[neural_model_dossier_eligibility_gated_local_commit]], and [[neural_model_symbolic_contract_eligibility_gated_local_commit]], especially write frequency, bounded output exposure, active forgetting, useful bits per episode, and memory/replay/imagination.
2. completed: define deterministic bit counters over the existing hard symbolic episode contracts.
3. completed: run the counters over smoke and hard seed sets.
4. completed: verify that oracle policies still solve the tasks after replacing traces with oracle codes.
5. completed: report per-family ratio ranges and failure cases in [[tests/oracle_compression_analysis_results]].
6. completed: update the paper spine with the resulting proof state.
7. completed: write the accepted-frontier and weak-frontier split before scoping any tiny trainable neural-model mirror.
8. completed: write the narrow learned-codec proof package for `compression_under_bit_budget`: [[neural_model_dossier_compression_under_bit_budget_codec]].
9. completed: write the tiny local mirror contract for that family before any code: [[neural_model_tiny_mirror_contract_compression_under_bit_budget]].
10. completed: implement the local baseline mirror surface for that family only: [[tests/compression_under_bit_budget_mirror]].
11. completed: add the first learned codec to that local surface; smoke result is negative, with train joint success 1.0 and held-out joint success 0.0.
12. completed: add diagnostic-only trainability split and provenance/source controls without changing the full model or using paid compute. the result localizes first toward payload/action/source-state inference and learned decoder generalization, not a proved learned-compression claim.
13. next: revise the local mirror around payload/action/source-state inference before any broader mirror, full-model path, or paid compute.

## see also

- [[neural_model_paper_spine]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[tests/compression_under_bit_budget_mirror]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[neural_model_research_test_material_plan]]
- [[tests/hard_symbolic_nm_test_material]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[substrate_requires_architectural_change]]
- [[synthetic_shared_world_bridge]]
- [[PROJECT_PLAN]]
- [[INDEX]]
