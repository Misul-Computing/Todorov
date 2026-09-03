# Modular MLX exited-child terminal sample race

status: historical context only. frozen as of 2026-07-26. do not edit.

Governed attempt `mlx-m5pro-20260726-3` passed the current preflight, and its
MLX child executed all `132` pilot updates and reported `292,864` attempted
pilot positions. The child then closed and exited successfully. An already
in-flight parent resource sample had captured the older active rung-two state
before that closure but performed its process read after exit. On macOS,
`ps` returned the exited child PID with zero resident bytes and zero CPU time.
The sampler committed that stale row after the parent had cleared active jobs
and marked the child exited.

The run therefore had no durable clean terminal resource row.
`stop(final_sample=True)` rejected its final state, and hard-abort finalization
then rejected the zeroed process row as a decrease from the child's previously
positive CPU time. The run published neither `run/pilot.json` nor
`ABORTED.json`. It created no training-start request and began no claim model
or optimizer update. The directory is preserved unchanged and a new run ID is
required.

The bug class was a missing lifecycle-generation boundary across asynchronous
sampling. Stop exclusion alone was insufficient because progress, job-clear,
and child-exit state could change while external process and swap reads were
in flight.

The shared sampler now increments a generation at every state mutation and
requires the generation to remain current before a periodic row can commit.
A previously positive child that appears with both zero resident bytes and
zero CPU time is discarded as an exited-process observation. Final stop first
quiesces the periodic thread, then reuses an exact durable clean row or takes a
fresh state-matched sample after the remaining five-second interval. The
legacy attempt-3 prefix remains readable for abort accounting by treating the
zeroed child only as terminal disappearance and rejecting any later
reappearance of the PID.

Deterministic regressions pause the old process sample at the exact race,
change the lifecycle state, expose zeroed child telemetry, and require only a
new parent-only terminal row to commit. Separate tests cover the pilot and
claim call order, legacy timeline validation, PID reappearance rejection, and
real hard-abort finalization.

The complete MLX file reports `84 passed, 2 skipped`. Focused CPU resource and
abort verification reports `77 passed`. Complete modular verification reports
`834 passed, 2 skipped` in `214.05` seconds.

## Category check

Implemented operation: terminal resource-sampling and abort-accounting
correction in the execution harness.

Strongest evidence: the immutable attempt-3 resource ledger, the observed
zeroed exited-child row, deterministic race regressions, focused tests, and the
complete modular test run.

What failed: terminal lifecycle state did not invalidate an in-flight resource
snapshot, and abort validation interpreted exited-child telemetry as live CPU
regression.

What is not proved: a valid pilot projection, training success, routed exact
recall, recurrent world tracking, imagination, replay, a 3D world model, or a
working neural machine.

Why this is not promoted: this is execution-harness failure and correction
evidence, not a pilot or trained-model result.

## See also

- [[../tests/modular_sequence_role_mlx_resumption_20260726]]
- [[../PROJECT_PLAN]]
- [[modular_mlx_tail_timing_registry_shape_drift]]
