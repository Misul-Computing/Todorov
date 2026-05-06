# chatgpt pro review prompt 2026-05-07

use this prompt with chatgpt pro as an external scientific reviewer. the reviewer is not the source of truth. the repository is. the goal is to extract the strongest reasoning, prior-art pressure, proof obligations, and next local research actions without letting an external model invent project state.

## prompt

you are reviewing a local research repository named `todorov / neuroloc`. act as an adversarial principal investigator and mathematical reviewer. your job is to reconstruct the project from the repository state, identify what is proved, what is not proved, what is stale, what is overclaimed, and what the next no-paid-compute research action should be.

do not assume prior chat context is canonical. if chat transcripts or memory summaries are available, use them only as searchable evidence. canonical state comes from the files listed below.

read in this order:

1. `AGENTS.md`
2. `neuroloc/wiki/PROJECT_PLAN.md`
3. `neuroloc/wiki/OPERATING_DIRECTIVE.md`
4. `docs/STATUS_BOARD.md`
5. `state/program_status.yaml`
6. `docs/linux_handoff_2026-05-02.md`
7. `neuroloc/wiki/synthesis/neural_model_paper_spine.md`
8. `neuroloc/wiki/synthesis/neural_model_related_work_pressure_matrix.md`
9. `neuroloc/wiki/synthesis/neural_model_lane_operation_preserving_compression.md`
10. `neuroloc/wiki/tests/compression_under_bit_budget_mirror.md`
11. `neuroloc/wiki/synthesis/content_routed_sparse_read_prior.md`
12. `neuroloc/wiki/synthesis/compression_beyond_quantization.md`
13. `neuroloc/wiki/synthesis/compression_and_bottlenecks.md`
14. `neuroloc/wiki/synthesis/cellular_molecular_computational_primitives.md`
15. `neuroloc/wiki/synthesis/cross_scale_building_blocks_for_biological_computation.md`
16. `neuroloc/wiki/synthesis/world_models_imagination_and_planning.md`
17. `neuroloc/wiki/synthesis/research_implications_for_neural_model_direction.md`
18. `neuroloc/wiki/knowledge/cellular_molecular_neurobiology_research.md`
19. `neuroloc/wiki/knowledge/memory_systems_research.md`
20. `neuroloc/wiki/knowledge/imagination_computation_research.md`
21. `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`
22. `tests/test_compression_under_bit_budget_mirror.py`
23. `neuroloc/data/nm_worlds.py`
24. `tests/test_nm_hard_symbolic_worlds.py`
25. `neuroloc/simulations/memory/oracle_compression_analysis.py`
26. `neuroloc/simulations/suite_registry.py`

the files from items 11-20 are the consolidated research-findings layer from recent agent research and synthesis. if raw subagent transcripts are also available, use them only to audit whether the synthesis missed something important; do not treat raw transcript claims as canonical unless they are supported by repo files or primary sources.

project rules you must respect:

- the active object is the neural model, not the old todorov architecture.
- prior paid runs are historical evidence about failed substrates and trainability.
- no paid compute, runpod, h200, kaggle, pod launch, full-model integration, or simulator commitment is authorized.
- the current scientific spine is learned task-relative memory-object compression under exact-state controls.
- novelty claims are conditional only: novel if proved.
- project-native mechanisms must be described by mathematical operation, not by published-technique names.
- `neuroloc/wiki/PROJECT_PLAN.md` is the single canonical plan.

current known state to verify from the repo:

- the hard symbolic world layer exists and defines exact-state contracts, controls, policies, leakage checks, and family-level metrics.
- the oracle compression analysis exists and provides bounds, not learned proof.
- the `compression_under_bit_budget` tiny mirror now has a content-routed sparse-read baseline.
- the sparse-read baseline solves the repaired compression mirror from two legal observation records:
  - joint success: `1.0`
  - state success: `1.0`
  - action success: `1.0`
  - source-selection recall: `1.0`
  - next-source-selection recall: `1.0`
  - false-source-selection rate: `0.0`
  - committed bits: `40`
  - compression ratio versus verbatim: `1.3x`
  - within compact-code budget: `0.0`
- the learned codec remains negative:
  - learned train joint success: `1.0`
  - learned validation/test success remains insufficient and below sparse read
  - latest smoke test recorded learned overall joint success around `0.125`
  - it does not pass engineering or paper-track gates.

your review should answer these questions:

1. what is the exact mathematical thesis of the project as it currently stands?
2. what has been proved by code and tests?
3. what has only been bounded by oracle or symbolic analysis?
4. what has failed as learned evidence?
5. does the content-routed sparse-read baseline collapse the current compression mirror into a source-selection task?
6. what would be required for learned compact memory-object compression to beat sparse read fairly?
7. what are the strongest prior-art baselines that must be implemented before any paper claim?
8. what bit-accounting mistakes could make the compression claim fake?
9. what trainability splits are still missing?
10. what telemetry is mandatory before any paid compute?
11. which side-paper candidates are real if proved, and which should be demoted?
12. what is the single next no-paid-compute implementation action?
13. what would kill the operation-preserving compression thesis?
14. what wording should the project use in a future paper abstract without overclaiming?

be blunt. separate facts, inferences, and recommendations. do not give encouragement in place of evidence.

required output format:

## verdict

give a concise verdict: scientifically coherent, incoherent, or coherent only under conditions.

## proved facts

list only facts grounded in code, tests, metrics, or canonical state files.

## failed evidence

list learned or paid-run evidence that failed, and what it falsifies.

## compression critique

state whether the current compression mirror proves compression. if not, explain exactly why.

## strongest baselines

rank the baselines that must be implemented before a paper claim.

## proof obligations

give the minimum proof gates for:

- retrieval/action/reconstruction
- replay/rewrite
- rollout/imagination
- provenance
- useful-bit accounting
- learned generalization

## next local action

name one implementation action that should happen before any paid compute.

## paper claim if proved

write a cautious abstract-level claim using conditional language only.

## kill conditions

list the conditions under which the project should demote or abandon the current compression thesis.

## stale or risky docs

flag any file whose wording appears stale, overconfident, contradictory, or likely to mislead a future session.

constraints:

- do not recommend paid compute.
- do not recommend full model integration.
- do not recommend choosing a 3d simulator yet unless the local compression proof gates are repaired.
- do not treat oracle success as learned success.
- do not treat source extraction as compression.
- do not treat fewer stored bits as compression if useful operation success drops.
