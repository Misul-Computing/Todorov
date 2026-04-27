# neural model symbolic contract: eligibility-gated local commit

status: current (as of 2026-04-27).

## purpose

this page defines the symbolic/oracle test-material contract for [[neural_model_dossier_eligibility_gated_local_commit]]. it is the bridge between the mechanism dossier and the implemented test-material package in [[tests/eligibility_gated_local_commit_test_material]].

the contract is not model code and not a trained result. it states what the generator, evaluator, and test suite must expose before oracle compression analysis can decide whether a tiny trainable mirror is justified for this mechanism.

## claim under test

the mechanism is useful only if a system can:

- mark candidate information before it knows whether the information matters.
- commit the right candidate after delayed relevance is revealed.
- avoid committing tempting distractors.
- expose only the task-relevant committed item when output capacity is limited.
- use fewer committed bits than always-write or verbatim storage at equal task success.

the contract must localize four failure points:

1. trace formation failure.
2. commit selection failure.
3. address/read failure.
4. output-exposure failure.

## required episode fields

each episode must expose these fields exactly enough for symbolic controls and later neural telemetry:

- `episode_id`
- `seed`
- `family`
- `profile`
- `hidden_state`
- `observation_stream`
- `query`
- `target`
- `candidate_events`
- `relevance_events`
- `commit_targets`
- `read_queries`
- `exposure_targets`
- `memory_relevant_positions`
- `distractor_positions`
- `negative_commit_positions`
- `trace_eligible_positions`
- `commit_positions`
- `exposure_positions`
- `difficulty`
- `bit_budget`
- `output_budget`
- `oracle_codes`
- `expected`
- `telemetry`
- `leakage_checks`

`candidate_events` must include `candidate_id`, source time, local entity id, address key, payload, visible fields, hidden payload, provenance id, whether the event should mark eligibility, and whether the event is eventually relevant.

`relevance_events` must include time, context id, available fields, resolved candidate ids, negated candidate ids, and the rule used to resolve them. a relevance event must disambiguate by context or relation. it must not name the answer, the target identity, or a unique candidate index.

`commit_targets` must include candidate id, source time, commit time, whether the candidate should commit, payload fields, address, commit latency, and forbidden-before time. no candidate may be committed before relevance is available unless the family explicitly tests an oracle early-commit ablation.

`read_queries` must include query time, query address, required commit id, target state, target action, and `target_answer_visible=false`.

`exposure_targets` must include time, maximum output capacity, exposure budget, commit ids that should be exposed, commit ids that must not be exposed, expected output gate state, and `residual_only_answer_possible=false`.

`oracle_codes` must include:

- trace code.
- committed event code.
- address code.
- output-exposure code.
- verbatim trace bit count.
- eligibility-gated commit bit count.
- always-write bit count.
- task-relevant bit count.

## required families

### delayed relevance local commit

a source event appears early. several candidate events and distractors follow. only after a delay does a relevance event reveal which earlier candidate matters.

required property: the target answer is not present at query time, the relevant event is not the most recent compatible event, and at least one distractor shares a surface feature with the target.

pass condition: oracle trace plus oracle commit solves the task. no-trace, random-trace, no-memory, recency-only, and shuffled-address controls fail below ceiling.

kill condition: recency-only or always-commit matched budget matches oracle under the hard profile.

### bounded output exposure

several committed items exist, but output capacity is smaller than the number of committed items. the query requires exposing exactly one committed item.

required property: memory content exists before the answer step, but fixed closed exposure fails and fixed open exposure injects at least one competing item.

pass condition: oracle exposure solves the task. hand-opened exposure without selection does not count as a pass unless it selects the target and suppresses distractors.

kill condition: the answer can be decoded from residual or query fields without reading committed state.

### crossed commit and exposure split

the episode is evaluated under four symbolic splits:

- oracle commit plus oracle exposure.
- oracle commit plus learned exposure proxy.
- learned commit proxy plus oracle exposure.
- learned commit proxy plus learned exposure proxy.

the symbolic proxy is not a trained model. it is a deterministic degraded control that exposes which half of the contract is required.

pass condition: oracle commit plus oracle exposure succeeds; each degraded split fails when its missing component is necessary.

kill condition: all splits behave the same.

### commit compression frontier

the same hidden episode is evaluated with three storage policies:

- verbatim trace.
- always-write committed memory.
- eligibility-gated commit.

pass condition: eligibility-gated commit preserves `joint_success` while writing fewer bits than always-write and verbatim storage.

kill condition: bits fall only because task-relevant state is dropped, or always-write achieves the same success at the same bit and output budget.

## required controls

minimum controls:

- oracle.
- no-memory.
- recency-only.
- shuffled-address.
- no-trace.
- random-trace.
- always-commit unlimited.
- always-commit matched budget.
- oracle-mark / no-commit.
- no-commit / oracle-exposure.
- fixed closed exposure.
- fixed open exposure.
- hand-opened exposure.
- oracle-trace / learned-commit proxy.
- learned-trace proxy / oracle-commit.
- oracle-commit / learned-exposure proxy.
- learned-commit proxy / oracle-exposure.
- matched residual-capacity baseline.
- matched compute budget.

control ceilings under hard profile:

- oracle: `joint_success >= 0.98`.
- no-memory: `joint_success <= 0.15`.
- recency-only: `joint_success <= 0.25`.
- shuffled-address: `joint_success <= 0.15` on address-dependent families.
- random-trace: `commit_f1 <= 0.25`.
- always-commit unlimited: may solve recall but must exceed commit or exposure budget.
- always-commit matched budget: must fail under interference or output-capacity competition.
- oracle-mark / no-commit: must fail read, action, and joint success.
- no-commit / oracle-exposure: must fail because exposure cannot hallucinate absent memory.
- fixed closed exposure: `action_success <= 0.15`.
- fixed open exposure: must show an exposure noise penalty when distractors exceed output budget.

## required metrics

top-line:

- `mark_correct`
- `commit_correct`
- `read_correct`
- `exposure_correct`
- `state_probe_accuracy`
- `action_success`
- `joint_success`
- exact recall
- delayed-use success

trace and commit:

- `trace_precision`
- `trace_recall`
- `write_precision`
- `write_recall`
- `commit_f1`
- `commit_latency`
- false-commit rate
- negative-commit rejection rate
- writes per successful episode

exposure:

- output-capacity precision
- output-capacity recall
- exposure noise cost
- memory-output norm versus residual norm

compression:

- verbatim trace bits
- always-write bits
- eligibility-gated commit bits
- task-relevant bits
- bits committed per successful episode
- useful bits fraction

localization:

- oracle-trace / learned-commit gap
- learned-trace / oracle-commit gap
- oracle-commit / learned-exposure gap
- learned-commit / oracle-exposure gap

## leakage checks

the generator must fail loudly if any of these are true:

- the query observation directly contains the target payload.
- the late relevance event names the answer instead of disambiguating a prior event.
- target identity can be inferred from time, row order, or object index alone.
- the relevant event is always the most recent compatible event.
- distractors are easier than targets because their fields are more masked or less correlated.
- output exposure can be solved from query fields without reading committed memory.
- bit savings discard task-relevant state.
- randomized controls are not deterministic under seed.

target and distractor observation completeness must be matched. source time, relevance time, query time, object index, and candidate order must be balanced across seeds so the correct answer cannot be recovered from position or ordering.

## implementation status

the first code pass is implemented without replacing the existing `hard_symbolic_nm` surface.

implemented targets:

1. eligibility-gated local-commit contract generator in `neuroloc/data/nm_worlds.py`.
2. deterministic policy evaluators for the required controls.
3. smoke and hard profiles.
4. field-completeness, determinism, leakage, committed-distractor, and control-separation tests.
5. summary metrics for mark, commit, read, exposure, compression, localization, writes per successful episode, and memory-output norm versus residual norm.
6. `eligibility_commit` suite registration after focused tests passed.

no model code, full-model integration, paid preset, h200, kaggle, pod, or simulator work is authorized by this contract.

## next action

run oracle compression counters on this mechanism-specific surface and the broader hard symbolic worlds. only if oracle ratios are strong enough should the tiny trainable mirror be scoped for this mechanism.

## see also

- [[PROJECT_PLAN]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[neural_model_research_test_material_plan]]
- [[tests/hard_symbolic_nm_test_material]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_paper_spine]]
- [[cellular_state_storage_gap_map]]
- [[INDEX]]
