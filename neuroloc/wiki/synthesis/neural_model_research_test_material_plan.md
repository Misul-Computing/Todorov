# neural model research and test-material plan

status: current (as of 2026-04-25).

## purpose

the active scientific object is the neural model. the old todorov architecture and the paid runs remain evidence, but they are not the design identity. this phase exists to prevent another cycle where a mechanism is implemented before the project can say exactly what it should prove.

the phase is research-to-test-material, not architecture execution. no metric code, model code, paid compute, or intervention preset is accepted until the relevant mechanism dossier states the claim, the test material, the control, the telemetry, and the kill condition.

## phase rule

every candidate mechanism must pass through this sequence:

1. mechanism dossier
2. test-material definition
3. symbolic verification
4. oracle compression analysis where the mechanism claims compression or memory rewriting
5. tiny trainable neural-model mirror
6. one-mechanism cpu gate
7. full model integration only if the small mirror localizes the effect
8. paid compute only after cpu evidence, telemetry, and prosecutor-clean state updates

after the hard symbolic package, the first executable target is oracle compression analysis on the same latent worlds. the tiny trainable neural-model mirror follows that analysis. the full model path is not the first target.

## dossier contract

each mechanism dossier must answer:

- mathematical operation
- evidence basis
- failure mode targeted
- required test material
- success metrics
- falsifying controls
- telemetry
- kill condition

if any answer is missing, the test is not ready. a weak result is not allowed to become a launch reason.

## required mechanism dossiers

- local neuron state: polarity-separated state, membrane or subthreshold state, eligibility and surprise traces
- memory formation: write decisions, output-gate fixed points, learned versus oracle writes
- addressing: softmax margin, slot entropy, key correlation, shuffled-address controls
- interference: target-to-nontarget read ratio, overwrite slope, continual-write drift
- compression: compact handles, schema or residual codes, provenance, bits written per useful memory
- reconstruction: shared decoder, residual correction, semantic versus verbatim success
- replay and rewrite: whether retrieved memories can be recompressed without losing task state
- iterative rollout: whether extra internal compute improves hard cases more than easy cases
- trainability: gate init, auxiliary loss, oracle write/read, address orthogonality, gradient flow

## test-material contract

every test world must expose:

- exact hidden state
- observation stream
- required action or answer
- memory-relevant positions
- distractors
- difficulty parameters
- expected no-memory performance
- expected recency-only performance
- expected oracle performance

no-memory does not mean the neural model should have no memory in production. it means the candidate memory path is disabled while the rest of the model remains usable. this proves the task is not solved by local statistics, recency leakage, or accidental dataset shortcuts.

## required controls

- no-memory
- recency-only
- shuffled-address
- oracle-write / learned-read
- learned-write / oracle-read
- hand-opened gate
- orthogonal-address initialization
- matched compute and parameter budget

## required metrics

top-line metrics:

- `state_probe_accuracy`
- `action_success`
- `joint_success`
- exact recall
- degraded-cue recall
- interference slope
- reuse advantage
- hard-case rollout gain
- bits written per successful episode

telemetry:

- gate-open fraction
- memory-output norm versus residual norm
- slot or address entropy
- address margin
- write frequency
- read concentration
- retention over delay
- compression budget
- reconstruction error
- confidence intervals

## acceptance rule

a mechanism passes only if it beats the relevant controls, shows telemetry that the intended path is used, and improves state/action/joint success rather than only loss. compression claims must show a Pareto improvement over verbatim storage. rollout claims must show larger gains on hard cases than easy cases.

a mechanism fails if it only improves loss, only works with oracle components, loses task-relevant state while reducing bits, opens gates into noise, improves easy cases only, or cannot beat no-memory and recency-only baselines.

## see also

- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[tests/hard_symbolic_nm_test_material]]
- [[neural_model_compression_stack]]
- [[neural_model_dossier_local_neuron_state]]
- [[neural_model_dossier_memory_formation]]
- [[neural_model_dossier_addressing]]
- [[neural_model_dossier_interference]]
- [[neural_model_dossier_compression]]
- [[neural_model_dossier_reconstruction]]
- [[neural_model_dossier_replay_rewrite]]
- [[neural_model_dossier_iterative_rollout]]
- [[neural_model_dossier_trainability]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[indexed_reconstruction_compression]]
- [[PROJECT_PLAN]]
