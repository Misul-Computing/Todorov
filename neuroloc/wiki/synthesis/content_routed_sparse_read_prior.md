# content-routed sparse read prior

status: current (as of 2026-05-06).

## role

this page records the prior-art implication of subquadratic's public subq / selective sparse attention material for the neural-model program. it is a related-work and planning boundary, not a project component name and not an implementation decision.

external source pages:

- [subq homepage](https://subq.ai/)
- [how ssa makes long context practical](https://subq.ai/how-ssa-makes-long-context-practical)
- [introducing subq](https://subq.ai/introducing-subq)

## external claim as published publicly

subquadratic publicly describes subq as a long-context llm built around a content-dependent sparse attention mechanism. their public technical page states that the mechanism selects relevant positions per query and computes attention over those positions, with the goal of preserving arbitrary-position retrieval while avoiding full all-pairs attention. their product pages claim 12m-token research context, a 1m-preview production model, strong long-context retrieval/coding benchmarks, and large prefill speedups versus dense attention at million-token scale.

this project should treat those claims as important public evidence, but not as a fully inspectable paper result until the promised model card or technical report is available. the current pages give architecture-level claims, benchmark claims, and training-process framing; they do not expose enough implementation detail to reproduce the mechanism.

## what it validates for this project

1. long-context memory is an architecture and routing problem, not only a corpus problem.
2. content-dependent routing is load-bearing for arbitrary-position retrieval.
3. fixed sparse patterns are weak as a mainline memory strategy because they decide where to look before the query content is known.
4. recurrent compressed state alone is risky when exact arbitrary facts must remain recoverable.
5. nominal context length is not the same as functional context; the relevant question is whether the model can retrieve and combine distributed evidence.
6. training must explicitly target long-context retrieval behavior; ordinary next-token training does not automatically teach the model to use distant evidence.

these points match the project's existing diagnosis after six paid retrieval failures and strengthen the current proof-gated method.

## what it does not validate

subq does not publicly prove any of the following for this project:

- learned compression of memory objects into schema, residual, provenance, or imagined-branch program codes.
- compression of replay rewrites or imagined branches.
- embodied 3d world-state compression.
- local neuron or cellular state storage.
- the trainability of this project's local mirrors.
- a 100x or 600x useful memory compression claim.
- any paid neural-model preset.

the public mechanism appears to reduce attention work by sparsifying content-routed reads over a very large context. that is not the same claim as operation-preserving memory-object compression.

## corners this lets us cut

the project no longer needs to spend time treating content-dependent sparse selection as speculative. it should become a required prior-art boundary and a baseline family.

the project can demote fixed-window, strided, dilated, or position-only sparse reads to controls or ablations rather than mainline candidates for arbitrary-position memory.

the project can adopt the nominal-versus-functional-context distinction directly. project-native wording should be:

```text
nominal memory: the amount of state or context accepted by the system.
functional memory: the amount of state or context that can still support the required operation under controls.
```

the project can also tighten the compression paper claim. a compressed memory-object stack is useful only if it beats or complements a content-routed sparse read over verbatim context on task success, committed bits, and compute.

## new baseline requirement

add a project-native baseline family:

```text
content_routed_sparse_read
```

mathematical description:

```text
given a query q_t and a memory field m_1...m_n, select a small set i(q_t) of candidate records by content, then run exact read operations over only that selected set.
```

this baseline should be described by operation, not by the external name. subq / ssa are related work for this baseline, not the project's term.

required metrics:

- selected record count.
- selection recall for the true source record.
- operation success after sparse read.
- compute per query.
- bits retained in the verbatim memory field.
- address margin.
- failure under shuffled selection.
- failure under position-only sparse selection.

required controls:

- dense verbatim read.
- content-routed sparse read.
- position-only sparse read.
- random sparse read.
- no-memory.
- recency-only.
- shuffled-address or shuffled-selection.
- compressed-code read.

## effect on the compression stack

the compression stack must now answer a sharper question:

```text
can operation-preserving compressed memory objects preserve the useful operations of content-routed sparse reads while lowering committed bits or enabling operations that raw context selection does not handle?
```

possible ways to beat or complement the baseline:

- fewer committed bits than retaining verbatim context.
- better replay/rewrite behavior because memories can be consolidated.
- compact imagined branches that do not exist in raw observed context.
- world-state handles that support occlusion and counterfactual physics queries.
- provenance-aware compression that keeps source trace identity auditable.
- lower interference under continual writes.

ways the project loses:

- if sparse verbatim read solves the task at acceptable compute and storage.
- if compressed codes reduce bits but lose action, state, or reconstruction success.
- if the compressed path needs oracle schema labels.
- if the decoder hides all information and provenance fails.
- if learned routing cannot beat content-routed sparse read controls.

## effect on current local work

this does not change the immediate code target. the `compression_under_bit_budget` local mirror still needs learned address, payload, velocity, action, and decoder generalization repair.

it does change the next proof-package standard. after the local mirror has a clean learned result, the next symbolic/mirror expansion should add a content-routed sparse-read baseline where the task family permits a verbatim memory field. a compression result should be reported against that baseline, not only against no-memory, recency-only, shuffled-address, and verbatim storage.

## open proof obligations

1. define a content-routed sparse-read baseline for the hard symbolic worlds without importing external architecture names as project-native terms.
2. decide which families can expose a fair sparse-verbatim-read baseline.
3. measure whether oracle compression still wins when sparse read over verbatim context is allowed.
4. require learned codecs to beat or complement sparse read before any strong compression claim.
5. update related-work sections when subquadratic publishes the promised technical report or model card.

## see also

- [[PROJECT_PLAN]]
- [[neural_model_paper_spine]]
- [[neural_model_compression_stack]]
- [[neural_model_lane_operation_preserving_compression]]
- [[oracle_compression_analysis_plan]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[tests/compression_under_bit_budget_mirror]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[attention_as_precision_and_routing]]
- [[indexed_reconstruction_compression]]
- [[tests/hard_symbolic_nm_test_material]]
