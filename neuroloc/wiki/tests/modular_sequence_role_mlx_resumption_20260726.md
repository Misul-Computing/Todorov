# Modular sequence-role MLX resumption correction 2026-07-26

status: current (as of 2026-07-26).

This is focused prelaunch and failed-pilot correction evidence, not a training
result.

## Scope

Deyan resumed the result-first local path on 2026-07-26. This record covers
the initial launch-contract corrections, the later parity and lifecycle
closures, and three governed attempts:

- Rejecting a periodic resource sample whose external reads began before stop
  but whose durable append had not yet committed.
- Synchronizing the second claim swap sample to the implemented point after
  claim-data construction and immediately before MLX child spawn.
- Carrying the original post-`proceed` claim swap baseline into the live
  sampler before it starts.
- Single-sourcing the parent pilot workload identities after a correct child
  message exposed duplicated stale parent aliases.
- Validating the preregistered tail timing registry as lists with exact
  cardinalities after the second retry reached its final attempted counters.
- Invalidating stale in-flight lifecycle generations and durably sampling the
  clean parent-only terminal state after the third attempt exposed zeroed
  exited-child telemetry.
- Restricting that zeroed-child exception to abort accounting for a
  prior-positive direct child, never a parent, unrelated process, changed
  parentage, or clean-final proof.

The model equations, data, claim seeds, schedules, losses, controls, token
positions, scientific gates, and runtime deadlines did not change. The initial
cross-runtime parity method later changed after the complete-surface review
proved the exact-only forward and optimizer requirements invalid; that
amendment is recorded below. The first retry stopped before
pilot update one. The second durable ledger reached the final attempted pilot
counters and then stopped before pilot publication because the parent
reinterpreted a correct child shape incorrectly. A separate direct diagnostic,
not the governed run ledger, established child completion behavior. No
training-start publication, claim model update, or optimizer update occurred.
The third attempt passed current preflight and its child executed all pilot
updates, but stale zeroed exited-child telemetry prevented terminal sampling
and hard-abort finalization. It published neither `run/pilot.json` nor
`ABORTED.json`.

## Sampler-stop race

Root cause: `QualificationResourceSampler._sample` captured shared state,
performed process and swap reads, and could append afterward without rechecking
whether stop had won. `stop` set the event and joined the thread, so it waited
for and accepted that late row.

The deterministic regression pauses the sampler after both external reads and
before writer append, starts stop on another thread, releases the sample, and
requires the returned rows and durable writer rows to exclude the in-flight
sample.

Red command:

```text
/Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_sequence_role_mlx.py::test_resource_sampler_stop_rejects_sample_in_flight_before_durable_append -q
```

Observed red result:

```text
FAILED
Left contains one more item: sample_id 1
1 failed
```

The minimal fix places stop signaling and the final append decision under the
same condition boundary. If stop wins, the in-flight sample returns before
writer append.

Observed green results:

```text
1 passed in 0.04s
5 passed, 60 deselected in 0.06s
63 passed, 2 skipped in 2.03s
```

The shared sampler class covers pilot and claim clean and error stop call sites.

## Second claim swap-sample order

The stale machine-readable value placed the second sample immediately before
child stage release. The implementation and run card place it after claim-data
construction and immediately before child spawn. That point isolates
parent-side construction swap growth relative to the original post-`proceed`
baseline and permits failure before creating a child. Moving it after child
startup would confound the measurement with self-check, cold compilation, and
child allocation.

Red command:

```text
/Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_sequence_role_mlx.py::test_claim_second_swap_sample_is_after_data_and_before_child_spawn -q
```

Observed red result:

```text
FAILED
expected after_claim_data_construction_immediately_before_MLX_child_spawn
observed immediately_before_MLX_child_stage_release
1 failed
```

The preregistration value is now
`after_claim_data_construction_immediately_before_MLX_child_spawn`. The
regression also requires the sample-zero construction to precede
`subprocess.Popen` and child spawn to precede `run_training`.

Observed green result:

```text
1 passed in 0.04s
```

The resulting canonical preregistration digest is
`49c87c1b097be24b8e27c441b4ddb796cbecde63435c4f106eff9fcb73fdf6b5`.
Before its propagation, the two existing identity tests failed with the new
digest. After updating the CPU authority, CPU test identity, and run card, the
three digest and ordering tests reported:

```text
3 passed in 0.42s
```

## Live claim swap baseline

Review found that `run_mlx_claim` constructed the live sampler with the durable
preworker row but did not assign `transition.swap_baseline_bytes`. The sampler
would therefore establish a new baseline from its first periodic post-spawn
sample instead of comparing all later samples with the original
post-`proceed` baseline.

The ordering regression was extended to require
`sampler.swap_baseline = transition.swap_baseline_bytes` before
`sampler.start()`.

Observed red result:

```text
FAILED
ValueError: substring not found
1 failed
```

The minimal production change assigns the transition baseline after sampler
construction and before start.

Observed green results:

```text
1 passed in 0.13s
64 passed, 2 skipped in 4.79s
```

## Governed attempt and pilot protocol correction

The four independent base attestations admitted
`mlx-m5pro-20260726-1`. Its immutable `run/preflight.json` contains `48`
serialized passing assertion records and zero serialized failures. Later
complete-surface review proved that the initial backend parity assertion
omitted required surfaces and admitted hidden and sequence-delta errors above
the frozen `1e-5` bound, so this is not a contract-valid `48 / 48` preflight
pass. The pilot hard-aborted after `1.216127125` seconds with zero attempted
and completed updates,
`training_start_state: not_started`, and no
`run/training_start_request.json`.

The child reached Metal, passed its self-check, and emitted the first
`pilot_workload_started` message with execution identity `one_MLX_lane`.
The engine and preregistration already agreed on `one_MLX_lane` and
`compiled_MLX_vmap_width_5`. Two parent validator tuples instead expected
`single_lane` and `compiled_vmap5`.

The focused regression was:

```text
/Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_sequence_role_mlx.py::test_parent_pilot_workload_protocol_is_single_sourced_and_matches_engine_and_preregistration -q
```

Observed red result:

```text
FAILED
AttributeError: module 'qualify_modular_mlx' has no attribute 'PILOT_PROTOCOL'
1 failed
```

An intermediate correction single-sourced workload records but left update
lanes, token charges, prior-attempt counts, seed arithmetic, workload
cardinality and order, and final totals duplicated. The strengthened structural
regression failed until those values also consumed the shared protocol.

Observed strengthened structural red summary:

```text
1 failed because the parent pilot paths did not yet consume
PILOT_PROTOCOL.updates_per_workload
```

The parent now defines one immutable `PILOT_PROTOCOL`. Every pilot validator,
sampler boundary, timeline check, assertion, completion check, and serialized
pilot identity derives its workload identities, seeds, update identities,
charges, prior counts, order, cardinality, and totals from that object.

A second review found that data and route seed offsets still used parallel
`model_seed + 1` and `model_seed + 2` arithmetic and that update-order
validation still assumed consecutive integers beginning at one. The
strengthened regression failed with:

```text
AttributeError: 'PilotProtocol' object has no attribute 'data_seed'
1 failed
```

The protocol now owns both offsets through `data_seed` and `route_seed` and
selects the next identity from its frozen `all_updates` tuple through
`expected_update`. The structural regression rejects the three former literal
derivations.

Third review found that the child producer still carried independent
`100 * ordinal`, `model_seed + 1`, and `model_seed + 2` arithmetic, while the
parent regression compared only the consumer with its own fields. The
producer-binding regression failed before correction with:

```text
StopIteration: PILOT_SEED_STRIDE was absent from the child engine
1 failed
```

The child now exposes `PILOT_SEED_STRIDE`,
`PILOT_DATA_SEED_OFFSET`, and `PILOT_ROUTE_SEED_OFFSET`, uses them for emitted
seeds, and the regression requires exact child-parent-preregistration equality
while rejecting the three former producer literals.

Observed green results:

```text
1 passed in 0.11s
65 passed, 2 skipped in 2.54s
```

Python compilation and `git diff --check` passed. A separate bug-class audit
compared every successful pilot child message and parent acknowledgement
against the preregistration and found zero additional direct mismatch. The
aborted run remains immutable and a new run ID is required.

## Second governed attempt and tail timing correction

Four refreshed base attestations admitted `mlx-m5pro-20260726-2`. Its
immutable preflight again serialized `48` passing records under the incomplete
checker. It records `hidden_max_abs` at `1.055002212524414e-5` and block-six
`sequence_delta_max_abs` at `1.1650845408439636e-5`, both above the frozen
`1e-5` limit. The durable pilot resource ledger reached `132` attempted pilot
updates and `292,864` attempted pilot token positions across donor, selected,
dense, and rung-two workloads. Its final row still has an active rung-two job,
so it does not establish governed completion. It recorded peak resident memory
`1,966,882,816` bytes and zero swap growth.

The parent hard-aborted after `25.653311541` seconds before publishing
`run/pilot.json`. `ABORTED.json` correctly records zero attempted claim work,
`training_start_state: not_started`, and `new_run_required: true`. No
`run/training_start_request.json` exists.

A direct invocation of the same pilot child completed all workloads, returned
`pilot_complete`, exited zero, and wrote no run artifact. This separate
diagnostic does not establish governed completion. Its evaluation and
checkpoint-reload tail records used one-element positive-integer lists for
each `warmup_duration_ns` value and three-element positive-integer lists for
each `timed_duration_ns` value. The machine-readable preregistration freezes
those exact shapes. `validate_child_tail_benchmarks` instead iterated each
warmup list as though it were a scalar integer, producing the generic parent
artifact-inconsistency abort after the child reported `pilot_complete`. The
active final parent resource row and absent `run/pilot.json` still prevent a
governed completion claim.

Focused red command:

```text
/Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_sequence_role_mlx.py::test_child_tail_timing_registry_accepts_preregistered_list_shapes_and_exact_maxima -q
```

Observed red result:

```text
FAILED
MlxQualificationError: pilot child tail warmup differs
1 failed
```

The correction covers the whole registry: warmup, timed, and selected-maximum
mappings must have identical nonempty key sets; each warmup list has exactly
one positive integer; each timed list has exactly three positive integers; and
every selected maximum equals the maximum of its matching timed list. The
producer audit additionally found that evaluation maxima were aggregated by
projection family rather than emitted for every timed subfamily. The emitted
schema now uses the exact per-subfamily maxima while the conservative scaling
calculation remains unchanged.

Observed green results:

```text
1 passed in 0.14s
66 passed, 2 skipped in 1.96s
```

Python 3.9 compilation over the changed engine, qualifier, and test completed
with exit code zero. `git diff --check` also passed.

Complete-surface review then proved this repair incomplete. The parent still
accepted fabricated one-family timing registries without the exact fixture
hashes, byte operands, scaling operands, or shared detail keys. The checkpoint
producer added a forbidden duplicate top-level projected-byte field, while
routing and packaging used different detail schemas from the preregistered
shared form. The parity and full shared-tail schema bug classes therefore
remain open until the next test-first correction and fresh review.

## Verification boundary

The first post-label-fix modular command was:

```text
env PYTHONPYCACHEPREFIX=/private/tmp/todorov-pycache /Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
812 passed, 2 skipped in 189.33s
```

Review then exposed the remaining duplicated protocol counters and identities,
so this result is intermediate rather than final unchanged-byte evidence. The
complete bug-class correction changed the qualifier and test bytes afterward.

The same complete modular command was rerun on the final post-bug-class code
and test bytes. Observed final result:

```text
812 passed, 2 skipped in 168.67s
```

Second review then exposed the seed-offset and consecutive-update derivations.
Those final qualifier and test changes make this result intermediate. A final
unchanged-byte modular run used the same complete command and reported:

```text
812 passed, 2 skipped in 170.27s
```

Third review then exposed the independent child seed arithmetic. The child and
test bytes changed again, making this result intermediate. Final unchanged-byte
modular verification used the same complete command and reported:

```text
812 passed, 2 skipped in 166.42s
```

The tail-registry correction then changed the engine, qualifier, and test
bytes, making the preceding result intermediate for the next run. The complete
modular command was rerun on the corrected unchanged code-and-test bytes and
reported:

```text
813 passed, 2 skipped in 166.70s
```

Complete-surface review exposed the remaining parity and tail-schema failures
after this run. Because implementation and test bytes must change again, this
result is intermediate rather than launch-authorizing verification.

Python compilation over the model, CPU authority, MLX engine, qualifier, and
focused tests completed with exit code zero before the live-baseline review
fix.

The first three-file modular suite reported:

```text
811 passed, 2 skipped in 188.84s
```

That run began before the live-baseline qualifier and regression bytes changed,
so it is intermediate evidence rather than final-byte closure. A second
independent reviewer run likewise began before those final edits and is also
intermediate.

After the live-baseline fix, current documentation correction, and evidence
record were present, Python compilation again completed with exit code zero.
The final code-and-test-byte modular command was:

```text
env PYTHONPYCACHEPREFIX=/private/tmp/todorov-pycache /Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
811 passed, 2 skipped in 188.97s
```

The prior complete-surface review and four attestations bind the bytes that
entered `mlx-m5pro-20260726-1`. The parent validator and test bytes then
changed, and the subsequent review and refreshed attestations admitted
`mlx-m5pro-20260726-2`. The tail-registry correction changed the engine,
qualifier, and test bytes again, so current complete-surface review and four
new attestations are required before the next run.

The repository-wide `tests/test_simulation_suite.py` collection was attempted
separately and stopped because the prior temporary Python 3.12 overlay no
longer contained SciPy. This environment failure is outside the modular claim
and is not reported as a model test failure or pass.

## Complete initial parity method correction

The incomplete initial checker used one selected forward and exact
cross-runtime optimizer values as if float32 MLX Metal and Torch CPU execution
were bit-identical. The correction exposes all eight sequence and feature
deltas, calibrates selected, all-eligible, dense, and rung-two forward roles,
and retains exact route, cardinality, parameter mapping, runtime, seed,
objective, and optimizer-state identities.

Eight fresh Metal processes reproduced these forward maxima:

```text
selected: max_abs 1.1650845408439636e-5, relative 1.0516556297804179e-5, normalized_l2 1.6761503885626105e-6, cosine 0.9999999999985957
all_eligible: max_abs 8.52346420288086e-6, relative 6.6861196903326005e-6, normalized_l2 1.7982210768037015e-6, cosine 0.9999999999983848
dense: max_abs 3.910064697265625e-5, relative 2.9676513375899136e-5, normalized_l2 3.2511606655190023e-6, cosine 0.9999999999947158
rung_two: max_abs 6.67572021484375e-6, relative 5.0465922597718385e-6, normalized_l2 1.6802652307470292e-6, cosine 0.9999999999985892
```

The five-lane raw-gradient repetitions reached maximum absolute error
`9.965896606445312e-5`, maximum relative error
`1.983217896775374e-4`, maximum normalized L2
`1.0272829456440252e-4`, and minimum cosine
`0.9999999947931381`. The limits were then frozen in the machine-readable
method. Two untouched admissions passed without threshold changes: selected
model seed `8123`, data seed `9123`, batch `2`; and all-eligible model seed
`8124`, data seed `9124`, altered batch `3`.

Independent research review proved that the first causal residual bound was a
triangle-inequality tautology. It was removed. The replacement compares each
runtime independently with a float64 semantic formula and an a priori
binary32 error bound, then permits the cross-runtime residual only within the
sum of those independently derived bounds. The compiled five-lane full-model
probe passed with optimizer formula ratio `0.24960512243002983`; the largest
observed causal residual ratio across focused fresh processes was
`0.2014384380252251`. A separate two-update
probe used a different second gradient and nonzero carried first and second
moments across five lanes and two decay families; it passed with maximum
formula-bound ratio `0.2521877097866519`.

The primary documentation supports tolerant, identity-bound comparison rather
than bitwise float equality. The official
[MLX compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
states that compiled and regular outputs agree up to numerical precision and
requires model and optimizer state to enter the compiled training graph. The
official
[MLX AdamW documentation](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html)
shows that bias correction defaults to false, so the project requires it
explicitly. The official
[PyTorch numerical-accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
states that mathematically identical floating-point operations are not
guaranteed bitwise-identical and that CPU and GPU results may differ even for
bitwise-identical inputs. These sources justify the comparison form; they do
not establish that the observed project thresholds pass.

The complete amended Metal self-check command was:

```text
PYTHONHASHSEED=0 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 /Users/dttdrv/Projects/Transformerov/.venv/bin/python
```

It imported `modular_sequence_role_mlx`, executed `self_check()`, and reported:

```text
pass true
device Device(gpu, 0)
held_out_forward_admission true
carried_adamw_parity true
actual_model_vmap5 true
optimizer_formula_max_bound_ratio 0.24960512243002983
causal_residual_worst_bound_ratio 0.20104967195972676
```

This is engine evidence only. Parent validator and regression closure, final
unchanged-byte tests, review, attestations, assertions, and the pilot remain
pending.

## Fixed-tail evidence closure

The tail producer and consumer now bind deterministic fixture hashes, charge
the omitted `65,536`-byte source-exclusion tensor for an exact `438,368`-byte
evaluation fixture, enforce current-model checkpoint tensor-byte lower bounds,
bind checkpoint metadata to the current engine digest, use selected-final
state for the all-eligible clone, and derive scratch cleanup from observed
owned-path lifecycle.

Exact red-green evidence:

```text
tail fixture and clone RED: 2 failed
tail fixture and clone GREEN: 2 passed
checkpoint cleanup RED: 1 failed
checkpoint cleanup GREEN: 1 passed
final tail-focused: 6 passed, 65 deselected in 0.92s
```

No Metal pilot or training execution occurred during that closure. The
machine-readable method digest after the parity and tail amendments is
`560ec821ca0cba93b828311ffa7de788a29d73a8e2d424ed35d6b750cc80598c`.

## Parent-validator integration closure

The non-Metal validator fixture suite first passed at `73 passed, 2 skipped`,
but the first real parent validation of a fresh Metal self-check rejected the
actual evidence. Diagnosis showed that the selected causal-residual record
combined residual and bound from the maximum-ratio index with excess from a
different maximum-excess index. The summary correctly needs the maximum excess
for pass or failure, while the selected worst record must keep residual, bound,
ratio, and excess from one index.

The producer now retains both quantities separately. A structural regression
binds selected residual, bound, ratio, and excess to the same index and binds
summary pass to the independently aggregated maximum excess. A second live
observation also showed that recomputing parameter-update deltas can differ
from direct final-parameter subtraction by float32 rounding. The parent now
allows only `rel_tol=1e-5` and `abs_tol=1e-8` for that relationship while
retaining exact end-to-end worst-parameter reduction.

Focused regressions reported:

```text
2 passed, 75 deselected
```

The full non-Metal MLX file then reported:

```text
75 passed, 2 skipped in 2.01s
```

Four fresh Metal processes each ran the complete engine self-check and parent
validator. All four returned code zero, engine pass true, validator pass true,
and parent worst-bound ratio `0.95367431640625`. Their observed causal-residual
ratios were `0.20104967195972676`, `0.20104967195972676`,
`0.19990534445846742`, and `0.2014384380252251`.

This closes the engine-to-parent self-check integration only. Final modular
verification and literal-zero complete review still bind the next launch
decision.

## Final modular verification after parity closure

The first complete three-file rerun exposed one stale duplicate self-check
fixture in the CPU preflight test:

```text
1 failed, 821 passed, 2 skipped in 165.08s
```

Production validation correctly rejected that old fixture. The CPU test now
reuses the canonical complete MLX self-check fixture instead of maintaining a
second copy, and a structural guard prevents the duplicate contract from
returning. The failing test and directly related checks passed.

The complete command was rerun on the corrected code and test bytes:

```text
PYTHONPYCACHEPREFIX=/private/tmp/todorov-pycache /Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
823 passed, 2 skipped in 167.81s
```

This was the complete modular code-and-test verification before the first
fresh review added two further regressions. It does not
substitute for the required literal-zero complete-surface review or create an
attestation.

## First-review closure verification

The first fresh complete-surface review found five runtime-evidence and
cleanup bug classes. The child now transports the post-update runtime
source-exclusion `raw`, `source`, and `routes` only in the transient pilot
completion message. The parent exact-validates the payload, fixes the source
to the frozen seed-`123456` fixture, regenerates the routes under seed
`633456`, computes the canonical digest itself, and binds that digest to the
tail detail. The backend now binds all four declared full-gradient
tolerances. Checkpoint-reload duration now includes owned scratch removal and
parent-directory synchronization. The exact-record sampler test uses a
deterministic condition barrier. Pilot and claim preserve the primary failure
across child termination, sampler stop, standard-error flush, synchronization,
close, and scratch cleanup.

The complete MLX file reported:

```text
77 passed, 2 skipped
```

The complete modular command was:

```text
/Users/dttdrv/Projects/Todorov/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
825 passed, 2 skipped in 230.18s
```

The complete command was repeated after the current documentation settled,
without changing any code or test byte:

```text
825 passed, 2 skipped in 215.47s
```

The repeated result is the current complete modular code-and-test
verification. It does not
substitute for the required literal-zero complete-surface review or create an
attestation.

## Second-review transport and lifecycle closure

The next adversarial pass found that scratch creation and child invocation
still occurred before the unwind boundary, existence checks could still mask
an active primary failure, child output used an unbounded blocking line read,
and the pilot receive loop had no absolute deadline. The correction moves
scratch creation and invocation inside the pilot and claim unwind boundaries,
executes every existence check and removal through primary-preserving cleanup,
and returns an observed cleanup boolean for lifecycle evidence.

Child output now uses select-driven bounded byte chunks, a strict three-MiB
per-line cap before decoding or JSON parsing, strict UTF-8 handling, a retained
per-process remainder for coalesced messages, and deadline rechecks between
partial chunks. One absolute non-resetting `1,200`-second pilot deadline covers
child start, every receive and acknowledgment, close, and the remaining-time-
capped join.

The complete MLX file reported:

```text
82 passed, 2 skipped
```

The complete modular command was rerun on the corrected code and test bytes:

```text
/Users/dttdrv/Projects/Todorov/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
830 passed, 2 skipped in 217.01s
```

This is the current complete modular code-and-test verification. It does not
substitute for the required literal-zero complete-surface review or create an
attestation.

## Buffered-message admission correction

The next review found that a complete message retained from a prior coalesced
read could be consumed before the next deadline and sampler checks. The
receive loop now runs both guards at the top of every iteration, before
inspecting or consuming any buffered newline. An expired deadline or failed
sampler leaves the message unconsumed; valid coalesced messages remain ordered
and are guarded independently.

The complete MLX file reported:

```text
83 passed, 2 skipped
```

The complete modular command was rerun on the corrected code and test bytes:

```text
/Users/dttdrv/Projects/Todorov/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
831 passed, 2 skipped in 219.63s
```

This is the sole current complete modular code-and-test verification. It does
not substitute for the required literal-zero complete-surface review or create
an attestation.

## Deterministic asynchronous-test correction

The next review found one remaining scheduler sleep in the resource-sampler
failure test. A class-wide search found a second sleep in the background-
writer abort test. The sampler test now waits for an explicit worker-entry
event set immediately before the injected failure. The writer test uses
explicit entry, queued-future cancellation, release, and join barriers. An AST
regression requires zero `time.sleep` calls across the MLX test file.

The complete MLX file reported:

```text
84 passed, 2 skipped
```

The complete modular command was rerun on the corrected test bytes:

```text
/Users/dttdrv/Projects/Todorov/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
832 passed, 2 skipped in 219.61s
```

This is the sole current complete modular code-and-test verification. It does
not substitute for the required literal-zero complete-surface review or create
an attestation.

## Third governed attempt and terminal sampler correction

Four literal-zero base attestations admitted
`mlx-m5pro-20260726-3`. Its current preflight passed. The MLX child executed
all `132` pilot updates across the donor, selected, dense, and rung-two
workloads and reported `292,864` attempted pilot positions before clean close
and exit.

The parent resource ledger contains six rows. Its final row retained the older
active rung-two job and both parent and child PIDs. The parent had advanced to
`10,020,000` microseconds of CPU time. The exited child, previously observed at
`14,250,000` microseconds and positive resident memory, appeared with zero CPU
time and zero resident memory. The stale row was durably appended after the
parent cleared active jobs and marked the child exited.

`stop(final_sample=True)` rejected the absence of a clean terminal row.
Hard-abort finalization then rejected the zeroed exited-child telemetry as
decreasing process CPU time. The run published neither `run/pilot.json` nor
`ABORTED.json`, has no training-start request, and began no claim model or
optimizer update. The run directory remains unchanged and requires a new run
ID.

The deterministic regression pauses the process read after it captured the
older lifecycle generation, clears jobs, marks the child exited, returns the
zeroed child observation, and requires the stale row to be discarded. Final
stop must then durably append a parent-only row with empty active jobs and the
exact final counters.

Every sampler progress, job-clear, and child-exit mutation now increments a
generation. A periodic sample can commit only if that generation remains
current. A previously positive child that becomes zero-resident and zero-CPU is
discarded. Final stop quiesces the periodic thread before it either reuses an
exact durable terminal row or takes a fresh state-matched sample after the
remaining five-second interval. Abort validation accepts the already persisted
zeroed child only as terminal disappearance and rejects PID reappearance.

Focused verification reported:

```text
84 passed, 2 skipped
77 passed, 599 deselected
```

The complete modular command was rerun on the corrected bytes:

```text
/Users/dttdrv/Projects/Todorov/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q
```

Observed result:

```text
834 passed, 2 skipped in 214.05s
```

This is execution-harness correction evidence, not a valid pilot projection or
a model result. Literal-zero review and four fresh attestations remain required
before attempt 4.

## Session stop and interrupted contract synchronization

Deyan stopped the session before attempt 4. No attempt-4 run root,
training-start request, claim model update, or optimizer update exists.

After the `844 passed, 2 skipped in 220.75s` pre-interruption modular
verification, final review found that production clean-claim terminal order
and the machine-readable contract still disagreed. The contract-sync agent
partially changed the payload and run card before being interrupted. The
modified payload hashes to
`8c7825f69fd27a7f3653c2e3bfab8673f3bb13d9f543fecd4a6aa9b97a4868ab`,
while launch constants and the run-card digest still contain
`fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553`.
No test or literal-zero review covers those stopped bytes. The active
attestations bind earlier bytes and cannot admit a future run.

The session remains stopped until Deyan explicitly resumes it. The complete
file-by-file recovery record is
`docs/MLX_SESSION_STOP_HANDOFF_2026-07-26.md`.

## Category check

Implemented operation: corrected resource cutoff, swap-sample ordering,
baseline propagation, and parent-child pilot identity validation in the
prelaunch execution harness.

Strongest evidence: immutable governed-run artifacts, direct child diagnostics,
deterministic red-green regressions, focused MLX tests, and the independent
complete-surface review that invalidated the incomplete closures.

What failed: the original stop accepted an in-flight row, the preregistration
named the wrong lifecycle point, the claim live sampler rebased swap after
spawn, the first governed retry rejected the correct first child workload
because of duplicated stale parent aliases, and the second retry rejected
preregistered list-shaped tail timing evidence after the child reported
`pilot_complete`. Its durable parent ledger still retained an active job and
no `run/pilot.json` was published, so governed pilot completion was not
established. Attempt 3 passed the current preflight and its child executed all
pilot updates, but a stale in-flight resource sample committed zeroed
exited-child telemetry; terminal sampling and abort finalization then failed
before either `run/pilot.json` or `ABORTED.json` could be published.

What is not proved: zero-finding complete-surface review, attestations,
contract-valid pretraining assertions, a valid full-package projection,
training success, routed exact recall, recurrent world tracking, a 3D world
model, imagination, dreaming, replay, language generation, or a working neural
machine.

Why this is not promoted: this is execution-harness correction evidence, not a
pilot or trained-model result.

## See also

- [[modular_sequence_role_cpu_run]]
- [[../PROJECT_PLAN]]
- [[../mistakes/modular_mlx_parent_child_pilot_label_drift]]
- [[../mistakes/modular_mlx_tail_timing_registry_shape_drift]]
- [[../mistakes/modular_mlx_exited_child_terminal_sample_race]]
- `docs/MLX_TRAINING_PAUSE_HANDOFF_2026-07-22.md`
