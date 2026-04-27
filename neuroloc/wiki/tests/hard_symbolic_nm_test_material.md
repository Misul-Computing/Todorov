# hard symbolic neural-model test material

status: current (as of 2026-04-27).

## purpose

this page documents the first hard symbolic test-material package for the neural model. it explains what was written, what the tests prove, what they do not prove, and what must happen next.

the package was implemented in commit `97a9fc6`. the first oracle compression analysis over this package is documented in [[tests/oracle_compression_analysis_results]]. the symbolic package lives in:

- `neuroloc/data/nm_worlds.py`
- `neuroloc/simulations/memory/nm_hard_symbolic_test_material.py`
- `neuroloc/simulations/suite_registry.py`
- `tests/test_nm_hard_symbolic_worlds.py`
- `tests/test_hard_symbolic_nm_suite.py`

this is symbolic test material. it is not a trained model result. it is the gate that prevents the project from training against weak, leaky, or underspecified tasks.

## what was written

the generator now emits hard symbolic contracts over the latent neural-model worlds. each episode exposes:

- exact hidden state
- masked observation stream
- task query
- required answer or action
- memory-relevant positions
- distractor positions
- difficulty parameters
- bit budget
- expected behavior for every control policy
- telemetry fields needed by later trainable mirrors

the first package covers ten task families:

- belief-state formation under occlusion and partial features
- associative recall with delayed query
- correlated-key interference
- delayed use under partial observability
- episodic reuse after distractors
- context-gated routing
- compression under explicit bit budget
- replay and rewrite
- iterative hard-case rollout
- imagination/recombination as latent recombination plus reconstruction

the deterministic symbolic policies are:

- oracle
- no-memory
- recency-only
- shuffled-address
- random-replay
- targeted-replay
- verbatim-store
- compressed-store
- oracle-write / learned-read
- learned-write / oracle-read
- hand-opened gate
- orthogonal-address initialization
- matched compute budget

## what the tests prove

the tests prove that the test material itself is valid enough to become the next training target.

they prove deterministic generation: the same seed produces the same contracts and hidden state.

they prove field completeness: every family has hidden state, observation stream, query, target, memory positions, distractors, difficulty parameters, expected controls, and telemetry.

they prove query masking: the target answer is not directly visible in the query observation.

they prove latent/observed consistency: visible identity attributes in the observation stream agree with the hidden identity bank, including forced correlated-key interference cases.

they prove control separation: oracle succeeds, no-memory fails, recency-only fails, and shuffled-address fails on address-dependent claims.

they prove replay separation: targeted replay beats random replay on replay/rewrite cases.

they prove compression separation: compressed storage is evaluated against verbatim storage under the same task, and compressed storage must fit the budget while verbatim storage exceeds it.

they prove hard-case rollout structure: hard cases are harder before rollout, and the intended rollout gain is larger on hard cases than on easy cases.

they prove trainability-localization controls are represented: oracle-write / learned-read, learned-write / oracle-read, hand-opened gate, and orthogonal-address initialization produce separable expected outcomes instead of all being aliases for oracle.

they prove registry integration: `hard_symbolic_nm` runs through the suite runner with both smoke and hard profiles and emits the required summary metrics.

## what the tests do not prove

these tests do not prove that the neural model can learn the worlds.

they do not prove that any neuron, memory, gate, addressing rule, compression rule, replay rule, or iterative compute mechanism works in a trained network.

they do not prove that compression is novel. they only enforce that a later compression claim must beat verbatim storage under an explicit bit budget while preserving task-relevant state.

they do not prove biological plausibility. the biology-grounded claims remain in the mechanism dossiers and supporting synthesis pages.

they do not authorize full-model integration or paid compute.

the correct interpretation is narrower: the symbolic tasks are now strict enough to support oracle compression analysis and then a tiny trainable neural-model mirror.

## validation record

validation after prosecutor fixes:

- focused pytest: `23 passed, 1 warning`
- suite runner smoke profile: passed
- suite runner hard profile: passed
- repository collection: `290 tests collected`
- yaml parse: passed
- diff check: passed, with only windows line-ending warnings
- implementation review pass: no remaining findings before the documentation page was added

the warning is the known numpy-on-windows experimental-build warning.

## next step

the first lane artifact from the six-lane master plan is [[synthesis/cellular_state_storage_gap_map]], the first cellular mechanism dossier is [[synthesis/neural_model_dossier_eligibility_gated_local_commit]], the first mechanism-specific symbolic contract is [[synthesis/neural_model_symbolic_contract_eligibility_gated_local_commit]], and the implemented mechanism-specific symbolic/oracle package is [[tests/eligibility_gated_local_commit_test_material]]. the first oracle compression analysis is now [[tests/oracle_compression_analysis_results]]. it compares verbatim trace bits, latent-state bits, schema/residual bits, and imagined-branch program bits before any trained mirror is asked to learn compression.

after the delayed-commit symbolic/oracle package, oracle compression analysis, and canonical cpu/control gates are defined, the next trained target may be a tiny trainable neural-model mirror only for a family whose oracle ratios justify it.

that mirror must use the exact same episode contracts and expose the same metrics:

- `state_probe_accuracy`
- `action_success`
- `joint_success`
- exact recall
- degraded-cue recall
- interference slope
- reuse advantage
- hard-case rollout gain
- bits written per successful episode

it must also log the same telemetry:

- gate-open fraction
- memory-output norm versus residual norm
- slot/address entropy
- address margin
- write frequency
- read concentration
- retention over delay
- compression budget
- reconstruction error

the mirror is allowed to fail. a clean failure is useful if it localizes whether the failure is write-side, read-side, gate-side, address-side, compression-side, replay-side, or optimization-side.

## decision rule

no mechanism should enter the full model path from this package alone. a mechanism must first pass the tiny mirror with controls and telemetry. no result is accepted from passkey-style smoke tests alone.

## see also

- [[tests/index]]
- [[synthesis/cellular_state_storage_gap_map]]
- [[synthesis/neural_model_dossier_eligibility_gated_local_commit]]
- [[synthesis/neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[tests/oracle_compression_analysis_results]]
- [[synthesis/neural_model_paper_spine]]
- [[synthesis/oracle_compression_analysis_plan]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/neural_model_lane_trainability_evaluation]]
- [[synthesis/neural_model_research_test_material_plan]]
- [[synthesis/neural_model_compression_stack]]
- [[PROJECT_PLAN]]
