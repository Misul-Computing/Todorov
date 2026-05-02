# linux handoff 2026-05-02

status: operational handoff supplement. not canonical project state.

this file is for moving the todorov / neuroloc project from the current windows workstation to a linux continuation environment. it is deliberately detailed and redundant. its job is to save the next researcher from reconstructing the project from scattered session context, while still preserving the single-plan rule.

the canonical plan remains `neuroloc/wiki/PROJECT_PLAN.md`. when this handoff disagrees with that file, fix `neuroloc/wiki/PROJECT_PLAN.md` first, then update this handoff only if the linux migration note itself still matters.

## first-read order on linux

1. read `AGENTS.md`.
2. read `neuroloc/wiki/PROJECT_PLAN.md`.
3. read this file.
4. if editing wiki or state files, read `neuroloc/wiki/OPERATING_DIRECTIVE.md`.
5. if touching structured state, read `state/program_status.yaml`.
6. if touching human status summaries, read `docs/STATUS_BOARD.md`.
7. if touching curriculum, read `pdf_curriculum/index/curriculum_status.md` and the private curriculum protocol at `~/.claude/plans/compressed-dancing-haven.md` if present in the linux user account.

do not make future sessions read all subordinate files just to recover the project state. promote durable decisions to `neuroloc/wiki/PROJECT_PLAN.md`.

## repository state at handoff

windows source path:

```text
C:\Users\deyan\Projects\todorov
```

expected linux path:

```text
not chosen at handoff
```

branch at handoff:

```text
master
```

git status at handoff:

```text
## master...origin/master [ahead 19]
```

latest commit at handoff:

```text
7a351c1 repair compression mirror source contract
```

latest visible history at handoff:

```text
7a351c1 repair compression mirror source contract
76b27ff add compression mirror source diagnostics
f5c0c1d add compression mirror diagnostics
83b3b51 add compression mirror learned codec result
a3832f1 add compression mirror baseline surface
108bc83 add compression mirror contract
a41fd79 add compression codec proof package
b2d0979 split oracle compression frontier
```

windows git status produced permission warnings for local pytest/temp folders:

```text
.codex_pytest_tmp/
.tmp_pytest/pytest-of-deyan/
.tmp_pytest_local/
.tmp_pytest_run/
codex_tmp_a/
temp_ok/pytest-of-deyan/
```

these were local windows permission artifacts, not project state. linux should not inherit them unless the workspace is copied directly with permissions intact. if they appear on linux, remove only untracked temp directories after verifying they are not inside a needed artifact path.

the branch was ahead of `origin/master` by 19 commits at handoff. before any linux work, run:

```bash
git status --short --branch
git log --oneline -8
```

do not assume the remote contains the full handoff state unless those commits have been pushed or the repository was copied directly.

## one-plan rule

the project previously suffered from multiple plan files competing for authority. the current rule is:

```text
neuroloc/wiki/PROJECT_PLAN.md is the one canonical plan and session-start state file.
```

other files are subordinate:

- `state/program_status.yaml` is machine-readable state.
- `docs/STATUS_BOARD.md` is a human status board.
- `pdf_curriculum/index/curriculum_status.md` is a lightweight curriculum index.
- `~/.claude/plans/compressed-dancing-haven.md` is the detailed curriculum production protocol, not the project master plan.
- `reports/` contains read-only reports and recovery dossiers.
- this file is a linux migration supplement, not a plan.

when files disagree:

1. resolve the disagreement into `neuroloc/wiki/PROJECT_PLAN.md`.
2. sync subordinate files only if the current task touches that surface.
3. do not create another persistent plan file unless the user explicitly asks and the file declares itself subordinate.

## project identity

the active scientific object is the neural model.

the old todorov architecture, old component names, old phase sequencing, and old paid runs are historical evidence. they are not the live design identity.

the current project target is a biology-led, proof-gated neural world-memory model:

- local state
- memory
- operation-preserving compression
- replay
- imagination as latent rollout and recombination
- physics reasoning
- language answers
- discrete actions
- later embodied 3d worlds with exact hidden state

language is co-primary, not the only objective. the model is not a normal language model with ordinary language-model evals stapled on. the user has repeatedly emphasized that this is a neural model, and the evals must be adapted to that object.

early observation surfaces should include exact world and physics state plus world-grounded language. early outputs are language and actions. drawing or image generation is not an early target.

## current phase

current phase:

```text
neural model master-plan research phase
```

current no-paid executable surface:

```text
compression_under_bit_budget local mirror
```

current canonical question:

```text
after repairing source observability, can the compression_under_bit_budget mirror learn a compact code that preserves source state and action on held-out records without oracle payloads or oracle decoder exposure?
```

current answer:

```text
not yet. the problem specification is repaired, but the learned codec still fails held-out operation preservation.
```

what this means:

- the symbolic family now exposes the needed source facts legally before query time.
- a legal visible-source codec can solve state/action/joint success at 1.0.
- the learned field codec overfits train and fails validation/test.
- the failure is now a learned-generalization problem, not an illegal-input or hidden-target problem.

## non-negotiable operating rules

no paid compute.

do not start h200, runpod, kaggle, pod, or other paid execution. the user monitors this strictly. paid compute can return only after proof gates, cpu controls, telemetry, prosecutor-clean docs, and one explicit selected hypothesis are in place and the user authorizes the run.

no full model integration.

do not edit paid presets or full neural-model paths as a curiosity exercise. the current work is local proof material, not a scale run.

no simulator choice yet.

the 3d simulator is a research selection gate, not a decision already made.

no chapter 2 research without approval.

chapter 1 is review-ready. chapter 2 is outline-only. curriculum is preserved as a support lane, not the active research driver.

no published-technique names as project-native names.

published names may appear when quoting or discussing external literature. project-native components must be named by their mathematical operation. prefer terms like `matrix memory`, `slot memory`, `compressed attention`, `output gate`, `surprise ratio`, `operation-preserving codec`, `schema/residual code`, `compact handle`, `local commit`, and `visible-source codec`.

no comments in code.

the repo rule is strict: no inline comments, no block comments, no docstrings, no todo comments. avoid adding comments even when they seem helpful.

no emojis.

no ai attribution.

do not add `co-authored-by`, `written with ai`, or similar text. project authorship remains deyan todorov.

lowercase docs and commit style.

use lower-case headings and commit messages unless quoting exact source names or identifiers.

use `apply_patch` for manual edits.

do not create or edit files with shell heredocs or ad hoc python writers when `apply_patch` can do it.

use `rg`.

prefer `rg` and `rg --files` for search. fall back only if unavailable.

prosecutor findings are bug classes.

if a reviewer flags one instance, grep for analogous instances, fix the class, add a structural guard where possible, and rerun review. priorities set order, not selection. no finding is silently ignored.

## current code surface

main current source files:

```text
neuroloc/data/nm_worlds.py
neuroloc/simulations/memory/compression_under_bit_budget_mirror.py
neuroloc/simulations/suite_registry.py
tests/test_nm_hard_symbolic_worlds.py
tests/test_compression_under_bit_budget_mirror.py
tests/test_simulation_suite.py
```

current docs for this surface:

```text
neuroloc/wiki/synthesis/neural_model_dossier_compression_under_bit_budget_codec.md
neuroloc/wiki/synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget.md
neuroloc/wiki/tests/compression_under_bit_budget_mirror.md
neuroloc/wiki/synthesis/oracle_compression_frontier_split.md
neuroloc/wiki/tests/oracle_compression_analysis_results.md
```

current suite:

```text
compression_mirror
```

current hard symbolic suite:

```text
hard_symbolic_nm
```

current mechanism-specific symbolic suite:

```text
eligibility_commit
```

current oracle compression suite:

```text
oracle_compression
```

## latest completed work

latest commit:

```text
7a351c1 repair compression mirror source contract
```

problem before the fix:

the `compression_under_bit_budget` learned-codec surface was being asked to reconstruct and act on source state that was not legally visible in the query-time observations. diagnostics showed:

- source event observed rate was 0.25.
- required source fields visible rate was 0.0.
- source state reconstructable rate was 0.0.
- visible-source-state plus oracle action plus oracle decoder joint success was 0.0.

that meant the learned codec was being blamed for a broken task contract.

fixes made:

1. `neuroloc/data/nm_worlds.py` now marks the committed source event for `compression_under_bit_budget`.
2. it also marks an adjacent source event before query time so velocity can be inferred legally.
3. this source marking is scoped only to `compression_under_bit_budget`, not every hard-symbolic family.
4. source event time avoids bounce cases so one-step observed motion matches hidden velocity.
5. target action is derived from visible source state instead of hidden identity.
6. `compression_under_bit_budget_mirror.py` now logs source-observability diagnostics and visible-source-codec diagnostics.
7. suite registry gates now require source observability and legal visible-source-codec success.
8. docs and state were synchronized to the repaired result.

current repaired result:

```text
policy_count: 23
diagnostic_result_count: 264
source_event_observed_rate: 1.0
source_required_fields_visible_rate: 1.0
source_state_reconstructable_rate: 1.0
visible_source_codec_joint_success: 1.0
visible_source_codec_state_success: 1.0
visible_source_codec_action_success: 1.0
visible_source_state_oracle_action_oracle_decoder_joint_success: 1.0
learned_codec_train_joint_success: 1.0
learned_codec_validation_joint_success: 0.0
learned_codec_test_joint_success: 0.0
learned_compression_ratio_vs_verbatim: 2.736842105263158
learned_address_success: 0.25
learned_color_success: 0.25
learned_shape_success: 1.0
learned_position_success: 0.25
learned_velocity_success: 0.0
learned_action_success: 0.0
learned_provenance_success: 1.0
learned_address_oracle_payload_joint_success: 0.25
oracle_address_learned_payload_joint_success: 0.0
oracle_code_learned_decoder_train_joint_success: 1.0
oracle_code_learned_decoder_validation_joint_success: 0.0
oracle_code_learned_decoder_test_joint_success: 0.0
```

interpretation:

- the test is now valid enough to be worth debugging.
- visible source information is sufficient.
- the legal non-learned codec can solve the operation.
- the learned codec has not learned held-out address, payload, velocity, action, or decoder generalization.
- this is not a compression success claim.
- this is not a paper result yet.
- this does not authorize paid compute.

## validations at handoff

focused validation:

```powershell
python -m pytest tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q
```

result:

```text
31 passed
```

targeted registry validation:

```powershell
python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_nm_hard_symbolic_worlds.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q
```

result:

```text
33 passed
```

suite runner validation:

```powershell
python neuroloc\simulations\suite_runner.py --suite compression_mirror --profile smoke --output-root reports\validation\compression_mirror_contract_repair --write-summary
```

result:

```text
passed
```

collection:

```powershell
python -m pytest tests --collect-only -q
```

result:

```text
330 tests collected
```

additional checks that passed:

- python compile for touched python files.
- yaml parse for `state/program_status.yaml`.
- `git diff --check`.

windows warnings:

- known numpy-on-windows experimental build warning.
- pytest cache or temp permission warnings from local windows temp directories.

linux should rerun the same checks with bash paths:

```bash
python -m pytest tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q
python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_nm_hard_symbolic_worlds.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q
python neuroloc/simulations/suite_runner.py --suite compression_mirror --profile smoke --output-root reports/validation/compression_mirror_contract_repair --write-summary
python -m pytest tests --collect-only -q
git diff --check
```

remove generated validation artifacts before commit unless the task explicitly calls for retaining them:

```bash
rm -rf reports/validation/compression_mirror_contract_repair
```

## latest mistakes and fixes

source-pointer leakage:

the source pointer and source event marking initially risked leaking into every hard-symbolic family. the fix scoped source observation markers to `compression_under_bit_budget`.

hidden-velocity mismatch:

the first source repair still allowed bounce cases where visible adjacent positions did not match hidden velocity. seed 3 exposed this. the source time is now chosen away from bounce points.

hidden-target action:

the action target originally depended on hidden identity rather than legally visible source state. the target is now derived from visible source state.

stale metrics:

docs and state briefly carried metrics from before the non-bounce repair. they were resynchronized before commit.

generated artifacts:

`reports/validation/` output from suite runner was removed before commit. keep generated validation output out of commits unless it is an explicit report artifact.

windows temp permissions:

some local temp directories produced git and pytest cache warnings. they did not affect the final commit.

## current mathematical posture

the project is trying to build useful neural memory, not just a better passkey model.

the compression thesis is operation-preserving compression:

- preserve the operation the memory object must support.
- do not optimize only tensor reconstruction.
- do not count compression as useful if the compressed object loses task-relevant state.
- compare against verbatim storage and oracle upper bounds.
- require controls that prove the intended path is being used.

current compression claim status:

```text
oracle compression bounds: partially positive.
learned compression: not yet positive.
paper novelty: conditional only.
paid compute: blocked.
```

important oracle compression result:

```text
profile: hard
contract_count: 448
surface_count: 2
family_count: 14
operation_preservation_rate: 1.0
controls_preservation_rate: 1.0
leakage_free_rate: 1.0
accepted_rate: 0.5714
strong_oracle_family_count: 8
weak_oracle_family_count: 6
kill_condition_count: 192
min_best_oracle_ratio: 7.09
max_best_oracle_ratio: 39.0
hard_symbolic_schema_ratio_mean: 11.23
eligibility_commit_ratio_vs_always_write: 3.61
imagination_branch_ratio_mean: 39.0
trainable_mirror_recommended: 0.0
```

accepted oracle-frontier families:

- belief_state_formation
- delayed_use_partial_observability
- episodic_reuse_after_distractors
- context_gated_routing
- compression_under_bit_budget
- replay_rewrite
- iterative_hard_case_rollout
- imagination_recombination

weak oracle-frontier families:

- associative_recall
- correlated_key_interference
- delayed_relevance_local_commit
- bounded_output_exposure
- crossed_commit_exposure_split
- commit_compression_frontier

first narrow learned-codec candidate:

```text
compression_under_bit_budget
```

why this candidate was chosen:

- it directly tests operation-preserving compression.
- oracle analysis predicted a useful ratio.
- it has explicit bit budget structure.
- it can be tested locally without full model code.
- it can fail cleanly without wasting paid compute.

## calculations to preserve

outer-product memory capacity reference:

```text
capacity approx 0.14 * d_head
d_head = 64
0.14 * 64 = 8.96
```

interpretation:

the classical outer-product-style memory capacity is roughly 9 patterns per head at `d_head=64` under clean assumptions. this is one reason the old matrix-memory substrate was not expected to support large verbatim retrieval without a stronger addressing or compression mechanism.

broken retention calculation:

```text
alpha_log_mean = -0.5
alpha_eff approx 0.377
0.377^256 approx 10^-109
```

interpretation:

the state effectively rounds to zero before 256-token retrieval. this bug contaminated several early paid runs via inherited defaults.

fixed retention calculation:

```text
alpha_log_mean = 5.0
alpha_eff approx 0.9933
0.9933^256 approx 0.18
```

interpretation:

state evaporation is no longer the sole explanation after the retention-fixed slot run still produced 0/100 passkey at 256 and 1024.

wilson upper bound reference:

```text
0/100 successes gives a 95% wilson upper bound of about 3.7%
0/20 successes gives a 95% wilson upper bound of about 14%
```

interpretation:

20-trial retrieval evals were too weak for architectural claims. later runs moved to 100 trials.

learned compression ratio from current mirror:

```text
learned_compression_ratio_vs_verbatim = 2.736842105263158
```

interpretation:

this is not useful yet because held-out joint success is 0.0. fewer committed bits do not matter if the useful operation is not preserved.

contract thresholds from the tiny mirror contract:

```text
engineering_pass_ratio_vs_verbatim: 4.0
paper_track_pass_ratio_vs_verbatim: 6.5
hard_joint_success_threshold: 0.95
```

interpretation:

even a learned codec that reaches held-out success must also beat meaningful compression thresholds before it can support a paper-track claim.

## paid-run history in one place

six paid runs produced 0% passkey at 256:

1. `god_run`
2. `god_run_v2`
3. `run1_baseline_noerasure`
4. `run2_slot_memory`
5. `run2_slot_memory_retention_fixed`
6. `run3_cognition_phase1`

this spans:

- two substrates: matrix memory and slot memory.
- two retention regimes: broken inherited retention and explicit fixed retention.
- two corpora: fineweb-edu natural text and a synthetic cognition corpus designed to reward retrieval.

current diagnosis:

```text
architectural and trainability failure, not corpus-only failure.
```

important run summaries:

`god_run`:

- h200.
- 283m params.
- fineweb-edu byte-level.
- 4000 steps.
- best val bpb 1.3950.
- passkey 0/20 at 256, 1024, 4096.
- selective copy 0/20 at 256, 512, 1024, 2048.
- metric logging bug dropped many probe keys.
- state-structure probe showed high-dimensional noise, not content-addressable memory.

`god_run_v2`:

- h200.
- 283m params.
- fineweb-edu byte-level.
- 4000 steps.
- critical recurrent-path math fix applied.
- val bpb 1.4453.
- passkey 0/100 at 256, 1024, 4096.
- copy 0/100 at 256, 512, 1024, 2048.
- the math fix did not recover recall.

`run1_baseline_noerasure`:

- h200.
- 353m params.
- matrix memory with dense keys and values.
- no erasure.
- auxiliary features off.
- final val bpb 1.4499.
- passkey 0/100 at 256.
- copy 0/100 at every length.
- confirmed the matrix-memory substrate itself did not retrieve in this setting.

`run2_slot_memory`:

- h200.
- 355m params.
- slot memory substrate.
- final val bpb 1.5107.
- passkey 0/100 at 256, 1024, 4096.
- copy 0/100.
- invalid as a clean slot test because it inherited `alpha_log_mean=-0.5`.
- mistake documented at `neuroloc/wiki/mistakes/run2_slot_memory_decay_copy_paste.md`.

`run2_slot_memory_retention_fixed`:

- h200.
- 355m params.
- slot memory with `alpha_log_mean=5.0`.
- flash-linear-attention actually active after dependency fix.
- 4000 steps in about 72 minutes at about 33000 tokens/s.
- final val bpb 1.4777.
- partial eval: passkey 0/100 at 256 and 1024.
- ruled out state evaporation as the sole cause.

`run3_cognition_phase1`:

- h200.
- 355m params.
- synthetic cognition corpus.
- corpus mix: 50% passkey, 30% key-value recall, 20% copy.
- 4000 steps in about 72 minutes at about 32800 tokens/s.
- best val bpb 6.3519.
- loss plateaued at the alphabet prior from step 150.
- partial eval: passkey 0/100 at 256 and 1024.
- this was the cleanest corpus discriminant. it still failed.

## big mistake record

the project has several expensive mistakes that should not be repeated.

matmul state approximation bug:

- earlier chunked evaluation captured only the last timestep instead of accumulating full state.
- this made long-context evaluation silently wrong.
- fixed in run_008 with full accumulation.

shape mismatch in spatial feedforward path:

- a run added a geometric residual at the wrong dimension.
- fixed by applying the residual after the down projection.

confounded training budgets:

- one comparison trained variants for mismatched step counts.
- any comparison must match compute, steps, data, and eval.

vacuous equivariance gate:

- an old gate tested algebra table correctness rather than trained model behavior.
- do not accept a gate that only tests a helper identity while claiming a model property.

20-trial retrieval evals:

- too weak for architectural claims.
- use 100 trials or more for meaningful Wilson bounds.

step logger cherry-pick:

- `god_machine.py` logged only a small hardcoded subset of metrics.
- a 4000-step h200 run lost critical probe telemetry.
- fixed class-wide by merging full metric dictionaries and smoke-testing disk round-trip.

train/eval recurrence divergence:

- recurrent eval path and accelerated train path used different effective decay math.
- fixed by aligning effective alpha computation.

validation mutated running state:

- buffers were updated during validation.
- fixed with training-only mutation.

flash-linear-attention silent fall-through:

- a paid slot-memory launch ran the slow python recurrent path because the dependency was missing.
- no warning was emitted.
- 17 minutes of h200 were wasted.
- dependency was pinned and startup guard added.

retention copy-paste:

- slot memory inherited `alpha_log_mean=-0.5`.
- this repeated a known state-evaporation bug.
- structural guard now requires explicit retention for relevant presets.

source-observability failure:

- the first compression mirror asked the learned codec to reconstruct facts it could not legally see.
- fixed by visible source contract repair.

## current research spine

paper-spine page:

```text
neuroloc/wiki/synthesis/neural_model_paper_spine.md
```

oracle compression plan:

```text
neuroloc/wiki/synthesis/oracle_compression_analysis_plan.md
```

six lane pages:

```text
neuroloc/wiki/synthesis/neural_model_lane_cellular_state_storage.md
neuroloc/wiki/synthesis/neural_model_lane_operation_preserving_compression.md
neuroloc/wiki/synthesis/neural_model_lane_memory_replay_imagination.md
neuroloc/wiki/synthesis/neural_model_lane_3d_world_physics.md
neuroloc/wiki/synthesis/neural_model_lane_trainability_evaluation.md
neuroloc/wiki/synthesis/neural_model_lane_project_operations.md
```

first cellular gap map:

```text
neuroloc/wiki/synthesis/cellular_state_storage_gap_map.md
```

first cellular mechanism dossier:

```text
neuroloc/wiki/synthesis/neural_model_dossier_eligibility_gated_local_commit.md
```

first cellular symbolic contract:

```text
neuroloc/wiki/synthesis/neural_model_symbolic_contract_eligibility_gated_local_commit.md
```

first compression learned-codec dossier:

```text
neuroloc/wiki/synthesis/neural_model_dossier_compression_under_bit_budget_codec.md
```

first tiny mirror contract:

```text
neuroloc/wiki/synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget.md
```

current local result card:

```text
neuroloc/wiki/tests/compression_under_bit_budget_mirror.md
```

## user preferences and project culture

the user is stepping back to study criminal psychology. they expect the project director agent to continue autonomously, document closely, and write for the next person.

autonomy means:

- keep working through coherent phases.
- document decisions as they are made.
- preserve exact command outputs and metrics when they matter.
- do not stop at "what next?" when the next action is already known.
- do not ask for paid compute unless a major proof gate has been reached.
- do not spin up paid compute as exploration.

the user cares strongly about:

- scientific correctness.
- no bloat.
- professional wiki writing.
- proper citations when making literature claims.
- proving what is useful before claiming novelty.
- avoiding name cargo-culting from published architectures.
- mathematical descriptions over fashionable labels.
- extremely thorough tests for every metric.
- operation-specific evals for the neural model.
- close handoff writing because agents are not permanent.

the user is frustrated by:

- repeated session warmup due to multiple plan files.
- ordinary language-model evals being treated as enough.
- compression claims that only reduce bits while losing useful knowledge.
- vague references to imagination or reasoning without exact symbolic gates.
- paid runs launched before local proof material is clean.
- stale docs.
- prosecutor findings being ignored or fixed only as instances.

## what remains right now

current immediate no-paid task:

```text
repair learned address, payload, velocity, action, and decoder generalization in the repaired compression_under_bit_budget local mirror.
```

do this before:

- global tiny mirror.
- full model integration.
- simulator implementation.
- paid compute.
- new architecture preset work.

the next researcher should inspect:

```text
neuroloc/simulations/memory/compression_under_bit_budget_mirror.py
tests/test_compression_under_bit_budget_mirror.py
neuroloc/data/nm_worlds.py
neuroloc/wiki/synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget.md
neuroloc/wiki/tests/compression_under_bit_budget_mirror.md
```

likely next debugging questions:

1. is the learned encoder seeing the minimal sufficient visible source features?
2. is the split design making validation/test require compositional generalization not represented in train?
3. are address and payload learned by memorized categorical lookup rather than rule abstraction?
4. does the decoder need a constrained rule form before a neural or learned mirror is meaningful?
5. is velocity represented in a learnable legal form, or only inferable by a brittle difference rule?
6. is action success failing because action is derived after state reconstruction errors, or because the action decoder is separately wrong?
7. does the provenance signal make train easy while not helping held-out operation preservation?
8. should the local mirror first implement a transparent rule-learning baseline before adding neural training?

do not lower thresholds to make the result pass. if the family is too brittle, revise the proof package and record the negative result.

## suggested linux first work session

1. verify git and tests:

```bash
git status --short --branch
git log --oneline -8
python -m pytest tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q
python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_nm_hard_symbolic_worlds.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q
```

2. run the suite profile:

```bash
python neuroloc/simulations/suite_runner.py --suite compression_mirror --profile smoke --output-root reports/validation/linux_compression_mirror_smoke --write-summary
rm -rf reports/validation/linux_compression_mirror_smoke
```

3. read the tiny mirror contract:

```bash
sed -n '1,220p' neuroloc/wiki/synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget.md
```

4. inspect current learned failures:

```bash
python neuroloc/simulations/memory/compression_under_bit_budget_mirror.py --profile smoke
```

if the script has no direct cli in the linux checkout, use the suite runner instead and inspect the generated summary before deleting it.

5. write a small diagnosis before editing:

- what fields fail?
- which diagnostic branch localizes the failure?
- which metric should change if the edit works?
- what control would falsify the improvement?

6. implement narrowly.

7. update docs and state only if the scientific state changes.

## validation before any commit

minimum for compression-mirror work:

```bash
python -m pytest tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q
python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_nm_hard_symbolic_worlds.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q
python neuroloc/simulations/suite_runner.py --suite compression_mirror --profile smoke --output-root reports/validation/compression_mirror_next --write-summary
python -m pytest tests --collect-only -q
git diff --check
rm -rf reports/validation/compression_mirror_next
```

if touching `state/program_status.yaml`:

```bash
python - <<'PY'
import yaml
with open('state/program_status.yaml', 'r', encoding='utf-8') as f:
    yaml.safe_load(f)
print('yaml ok')
PY
```

if touching python files:

```bash
python -m py_compile neuroloc/data/nm_worlds.py neuroloc/simulations/memory/compression_under_bit_budget_mirror.py neuroloc/simulations/suite_registry.py
```

if touching wiki/state:

- follow `neuroloc/wiki/OPERATING_DIRECTIVE.md`.
- every touched wiki page needs a lifecycle banner as first non-heading line.
- every touched wiki page needs a `see also` section.
- reciprocal links must exist for load-bearing cross-links.
- `PROJECT_PLAN.md` update history is append-only.
- run a prosecutor/review pass where available.

## documentation policy

write docs for the next researcher, not for the current session transcript.

good docs:

- state the claim.
- state what is not proved.
- give the exact metric values.
- give the command that generated them.
- name the code surface.
- name the controls.
- state the kill condition.
- link to the canonical plan and relevant result card.

bad docs:

- "we solved compression."
- "tiny mirror next" without oracle or local-family qualifier.
- "paid run next" without proof gates.
- "novel" without "if proved" and without prior-art boundary.
- stale counts from older test collection.
- treating passkey as the only eval.

when a result changes:

1. update the result card.
2. update `PROJECT_PLAN.md` if the current question, method, hypothesis, or answer changes.
3. update `state/program_status.yaml` if machine-readable status changes.
4. update `docs/STATUS_BOARD.md` if a human status snapshot would otherwise mislead.

## linux setup notes

the repository was developed on windows, but the next phase will continue on linux. likely differences:

- path separators become `/`.
- case sensitivity may expose filename mistakes hidden on windows.
- line-ending warnings may disappear.
- pytest temp permission warnings should disappear.
- shell commands should use bash syntax.
- do not rely on powershell-specific paths in docs except as historical notes.

recommended initial python setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

if dependency installation fails, do not start redesigning the project. record the exact error and fix the environment first. for local symbolic tests, most of the current surface should be cpu-only.

do not install or launch h200/runpod/kaggle tooling unless the user explicitly authorizes paid compute after proof gates.

## what not to remove

do not delete old architecture files yet.

the archive policy says old architecture material will later move under `neuroloc/wiki/archive/` after a focused migration pass. that has not happened. old files remain evidence.

do not remove run cards.

do not remove mistake docs.

do not remove historical old architecture summaries just because the active identity changed.

do not remove curriculum files.

chapter 1 remains review-ready and chapter 2 remains outline-only. the curriculum is a support lane.

## old architecture status

historical design vocabulary still exists in the repo, including terms tied to the previous architecture. treat it as evidence, not active naming guidance.

old architecture summary:

- sequence model with matrix memory and compressed attention.
- earlier framing used a shared recurrence/compression/bilinear/rotation/quantization abstraction.
- prior phase results showed strong bpb ratios but failed explicit retrieval.
- this is why the project pivoted to neural-model research and proof-gated symbolic worlds.

do not reactivate old phase sequencing merely because it is documented.

if architecture backlog resumes, it resumes from:

- implemented biology phase-1 symbolic battery.
- remaining latent-world deliberation or iterative-rollout probe.
- model-side neural-model evaluation surface.
- ranked intervention list in `neuroloc/wiki/synthesis/substrate_requires_architectural_change.md`.

but it does not resume until the proof gates say it should.

## current side-paper posture

everything potentially paper-worthy is conditional.

candidate side-paper seeds:

- operation-preserving compression stack.
- hard symbolic neural-model test material.
- cellular or local state storage.
- replay and imagination compression.
- trainability failure and control methodology.
- learned compression mirror, only if it passes held-out operation preservation and meaningful compression thresholds.

each side-paper candidate needs:

- mathematical operation.
- prior-art boundary.
- preserved or improved operation.
- source or proof basis.
- oracle or symbolic bound.
- trainable test.
- controls.
- telemetry.
- kill condition.
- paper claim if proved.

do not write "novel" as a fact. use "novel if proved" unless and until the evidence exists.

## current view on compression

the user wants compression that matters at every memory level, including cache-like surfaces, episodic memory, imagination branches, replay rewrites, and world state.

the key standard is not "fewer bits." the standard is:

```text
fewer committed bits per successful preserved operation at equal or better task success.
```

if useful knowledge is lost, the compression failed.

if the operation is preserved only by an oracle, it is an oracle bound, not a learned result.

if a learned codec overfits train and fails validation/test, it is a trainability failure, not a compression success.

if a compression claim only improves loss but not state/action/joint success, it fails.

if it improves easy cases but not hard cases, it fails for reasoning or rollout claims.

if it cannot beat no-memory, recency-only, shuffled-address, or verbatim baselines as appropriate, it fails.

## current view on reasoning and imagination

reasoning and imagination are not later add-ons.

they are first-class symbolic gates now:

- iterative hard-case rollout tests whether extra internal compute helps hard cases more than easy cases.
- imagination/recombination is latent recombination plus reconstruction, not vague generation.
- replay/rewrite tests whether retrieved memories can be recompressed or reused without losing task state.

however, full architecture implementation of these mechanisms is later. the current phase is proof material and local mirrors.

## current controls

standard controls to preserve:

- no-memory.
- recency-only.
- shuffled-address.
- oracle-write / learned-read.
- learned-write / oracle-read.
- hand-opened gate.
- orthogonal-address initialization.
- matched compute and parameter budget.
- random replay.
- targeted replay.
- verbatim store.
- compressed store.
- learned-code / oracle-decoder.
- oracle-code / learned-decoder.
- learned-address / oracle-payload.
- oracle-address / learned-payload.
- visible-source codec.

controls are not optional. they are the scientific object.

## current metrics

top-line metrics:

- state_probe_accuracy.
- action_success.
- joint_success.
- exact recall.
- degraded-cue recall.
- interference slope.
- reuse advantage.
- hard-case rollout gain.
- bits written per successful episode.

telemetry:

- gate-open fraction.
- memory-output norm versus residual norm.
- slot or address entropy.
- address margin.
- write frequency.
- read concentration.
- retention over delay.
- compression budget.
- reconstruction error.
- confidence intervals.
- learned address success.
- learned payload success.
- learned decoder generalization.
- source observability.

## final warning to the linux successor

the project is not waiting for a paid pod.

the project is waiting for proof that a local neural-model mechanism preserves useful operations under hard controls. the current proof object is the repaired `compression_under_bit_budget` mirror. if that mirror cannot learn held-out operation-preserving compression, the correct response is to record the negative result, revise the proof package, or select a different narrow oracle-frontier family. the wrong response is to scale a broken local result.

write everything down. write it so the next person can continue without guessing.
