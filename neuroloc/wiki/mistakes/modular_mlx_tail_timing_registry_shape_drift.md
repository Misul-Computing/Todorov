# Modular MLX tail timing registry shape drift

status: historical context only. frozen as of 2026-07-26. do not edit.

Governed attempt `mlx-m5pro-20260726-2` passed all `48` immutable preflight
assertions and ran the complete four-workload pilot. Its durable resource
ledger reached `132` attempted pilot updates and `292,864` pilot token
positions, with peak resident memory `1,966,882,816` bytes and zero swap
growth. The parent then hard-aborted after `25.653311541` seconds before
publishing `run/pilot.json`. No training-start request, claim model, claim
optimizer, or claim update existed.

A direct same-environment child diagnostic completed with return code zero and
emitted `pilot_complete`. The child correctly encoded every evaluation and
checkpoint-reload warmup duration as a one-element positive-integer list and
every timed registry as a three-element positive-integer list. This is the
shape frozen by the machine-readable preregistration. The parent validator
instead required each warmup registry value to be a scalar integer and rejected
the valid child record after the pilot work had completed.

The bug class is independent producer and consumer interpretation of a
machine-readable schema. Matching field names were insufficient because the
consumer restated list cardinality and scalar shape by hand. The same validator
also compared only the warmup and timed key sets, leaving the maximum-duration
registry outside that structural equality.

The correction validates the complete timing registry as one contract.
Warmup, timed, and maximum registries must be nonempty mappings with identical
key sets. Every warmup value must be an exact one-element positive-integer
list. Every timed value must be an exact three-element positive-integer list.
Every selected maximum must be a positive integer equal to the maximum of the
matching timed list. A deterministic non-Metal regression supplies the real
evaluation and checkpoint-reload shape and rejects scalar warmups, wrong
cardinalities, key drift, invalid integers, and inconsistent maxima.

The aborted run remains immutable and has `new_run_required: true`. Its
terminal attempted-work counters refer to claim work and therefore remain
zero; the separate durable pilot resource ledger preserves the completed pilot
work. The generic `artifact_inconsistency` terminal reason is execution-harness
evidence, not a model result. The scientific workload, model equations, tasks,
seeds, schedules, losses, controls, gates, token positions, and runtime
threshold did not change.

## Correction (2026-07-26)

The governed attempt's durable resource ledger proves `132` attempted pilot
updates and `292,864` attempted pilot positions. Its last row still has an
active rung-two job, so the original wording that the governed attempt
completed the four-workload pilot is not established. The separate direct
diagnostic completed the child protocol and supports the emitted-schema
diagnosis only.

The immutable preflight serialized `48` passing records, but later review
proved that classification contract-invalid. Hidden-state error was
`1.055002212524414e-5` and block-six sequence-delta error was
`1.1650845408439636e-5`, above the frozen `1e-5` limit, while mandatory
initial parity surfaces were absent from the gate.

The first timing repair closed list cardinality, key equality, and per-family
maxima only. Later review proved that the parent still accepted fabricated
partial timing registries and that producer details violated the
preregistered shared schema. The complete parity and tail-detail repairs are
tracked by [[../PROJECT_PLAN]] and
[[../tests/modular_sequence_role_mlx_resumption_20260726]].

## See also

- [[../tests/modular_sequence_role_mlx_resumption_20260726]]
- [[modular_mlx_parent_child_pilot_label_drift]]
- [[../tests/modular_sequence_role_cpu_run]]
- [[../PROJECT_PLAN]]
