# Modular sequence-role CPU base M5 Pro run 2026-07-22

status: historical context only. frozen as of 2026-07-22. do not edit.

The preserved run root is
`neuroloc/results/modular_sequence_role_cpu/base-m5pro-20260722-1`. It reached
and completed the fixed resource-pilot workloads, but it did not enter claim
training. The durable evidence records `88` attempted pilot updates and
`225,280` pilot token positions. No claim attempt ledger, checkpoint, completed
training update, or model claim exists.

The pilot selected `stop` because the original exact path projected roughly
seven hours. The exact `Tprojected` scalar is not recoverable: the stop record
was deleted after terminal closure rejected two valid pilot-generated check
detail artifacts, and no terminal `SHA256SUMS` was produced. The approximate
value is context only and must not be promoted to an exact measured result.

The immediate closure failure was an implementation defect. The closure
builder collected detail digests from `run/preflight.json` but not from
`run/pilot.json`, even though the pilot record referenced two details created
during the pilot. Exact closure therefore treated its own valid detail files as
extras, removed the provisional pilot record, and entered hard-abort handling.
The preserved run root is intentionally unchanged evidence of that failure.

Post-failure profiling found a second independent defect before claim training:
the crash-atomic JSONL writer read the complete committed prefix on every
successful append, and the claim parent parsed and validated the complete
attempt ledger after every attempt event. That would make claim-ledger work
quadratic in the number of events. Claim training never began, so this defect
did not produce a training result and is not offered as the cause of a measured
claim slowdown.

The corrected selected-attention component uses only gathered selected block
IDs and an exact causal mask. On the local CPU, five-iteration median forward
measurements were `2.84 ms` versus `119.59 ms` for the pinned public loop at
`B16/H4/T128/D16`, a `42.13x` component speedup, and `4.67 ms` versus
`239.31 ms` at `B8/H4/T512/D16`, a `51.28x` component speedup. Forward and
query, key, and value gradient parity tests pass. These are component
measurements, not an end-to-end training speedup.

A separate corrected-byte single-process diagnostic measured full optimizer
step medians of `0.5716 s`, `0.5782 s`, `0.5058 s`, and `0.4685 s` for pilot
workloads A, S, D, and H. Substitution into the frozen projection gives about
`11,214 s`, or `3.12 h`. This diagnostic omits the authoritative two-worker
maximum and complete pilot lifecycle, so it is not a launch estimate. It proves
only that the current corrected CPU path remains far above the new hard
twenty-minute end-to-end launch gate and ten-minute target.

No task, gate, seed, update, token, control, or evaluation was removed. The next
fresh run is permitted only after an unchanged-method pilot reports
`Tprojected <= 1,200 s`; `600 s` is the target. A projection above twenty
minutes is a stop, not a number to relabel or extrapolate away.

Category check: this is a failed pilot and exact-math execution repair record.
It does not establish learned memory, a 3D world model, replay, imagination,
dreaming, language generation, or a working neural-machine claim.

## See also

- [[modular_sequence_role_cpu_run]]
- [[../mistakes/modular_cpu_pilot_closure_and_quadratic_ledger]]
- [[../PROJECT_PLAN]]
