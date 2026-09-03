# Modular MLX parent-child pilot label drift

status: historical context only. frozen as of 2026-07-26. do not edit.

Governed attempt `mlx-m5pro-20260726-1` passed all `48` immutable preflight
assertions with zero failures. Its pilot then hard-aborted after `1.216127125`
seconds with zero attempted and completed updates, no training-start request,
and no claim model or optimizer work.

The child engine and machine-readable preregistration both used
`one_MLX_lane` and `compiled_MLX_vmap_width_5`. Two parent-side validator
tuples in `scripts/qualify_modular_mlx.py` instead expected the stale aliases
`single_lane` and `compiled_vmap5`. The child reached Metal, passed its
self-check, emitted the first `pilot_workload_started` record, and was then
rejected by the parent before update one.

The bug class is duplicated protocol identity and accounting. A wire identity
or counter copied into independent producer and consumer registries can drift
even when both sides are individually tested. The correction introduces one
immutable parent `PILOT_PROTOCOL`. Workload names, execution labels, lanes,
batch sizes, sequence lengths, seed base and stride, warmup and timed update
identities, model, data and route seed derivation, next-update selection,
per-lane token charges, prior-attempt counts, workload order and cardinality,
and final attempt and token totals are defined or derived there. Every parent
validator, sampler boundary, timeline check, assertion, completion check, and
serialized pilot identity consumes that protocol.

The child producer separately exposes the same seed stride and data and route
offsets as named constants and uses them for every emitted workload seed. The
regression binds those constants to the parent protocol and preregistered
formulas and rejects the former raw producer arithmetic.

The focused regression first failed because the parent had no shared protocol.
A strengthened structural form then failed while update cardinality still used
a parallel literal. Second review found two seed offsets and consecutive-update
arithmetic outside the protocol; the regression failed again until protocol
methods owned those derivations. Third review then found the corresponding
child arithmetic was still independent; a producer-binding regression failed
until named child constants were bound to parent and preregistration. The
focused regression passed only after both producer and consumer sides were
closed, and the complete MLX test file reported `65 passed, 2 skipped`. A
separate read-only comparison of every successful pilot message,
acknowledgement, identity, counter, workload parameter, and exact key registry
found zero additional parent-child-preregistration mismatch.

The aborted run is immutable and has `new_run_required: true`. Its generic
`artifact_inconsistency` reason is terminal evidence, not a model result. The
scientific workload, model equations, tasks, seeds, schedules, losses,
controls, gates, token positions, and runtime threshold did not change.

## Correction (2026-07-26)

The immutable preflight serialized `48` passing records, but later review
proved that classification contract-invalid. The initial backend parity gate
omitted mandatory surfaces and admitted hidden-state and sequence-delta errors
above the frozen `1e-5` limit. This does not change the parent-child execution
label diagnosis or the fact that the pilot stopped before update one.

## See also

- [[../tests/modular_sequence_role_mlx_resumption_20260726]]
- [[../tests/modular_sequence_role_cpu_run]]
- [[../PROJECT_PLAN]]
