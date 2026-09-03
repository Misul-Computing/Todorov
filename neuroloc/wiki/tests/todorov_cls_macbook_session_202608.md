# Todorov CLS MacBook session

status: historical context only. frozen as of 2026-08-13. do not edit.

This record captures the local complementary-learning-systems (CLS) trainer
session from 2026-08-12 through 2026-08-13. It was a local Apple M5 Pro
exploration using Metal and public FineWeb-Edu data. It was not a paid run,
not a release, and not a combined-model proof. The trainer, checkpoints,
datasets, and result JSON files remain in the local-only `train/` directory
and are excluded from the public repository push.

## Decision

Todorov's local trainer direction is one neural machine with two learning
timescales:

- The fast path uses a private gated-delta state that adapts inside a
  sequence.
- The slow path uses a versioned specialist trained against a frozen trunk and
  opened through a zero-initialized scalar gate.
- Attention remains the exact-recall control and candidate recall path. A
  routed selected-set implementation is a later owned component, not a reason
  to replace the first working trainer.

The first trainer is owned by Todorov. It uses byte-level FineWeb-Edu, MLX
Metal, and copied source implementations with recorded source hashes. It does
not live-import Transformerov or Monodratic. The initial model pattern is
`[attn, delta, delta, delta]` with sequence length `512`, attention window
`128`, width `768`, `12` layers, `12` heads, and head width `64`.

## Observed evidence

The local result registry at `train/results/STATUS.json` records these
observations:

- The chunkwise gated-delta kernel measured a maximum difference of
  `1.73e-6` against the recurrent path on Metal. The current test bound was
  `1e-5`, so this does not establish the draft `1e-6` target.
- The sequence-512 fit probe recorded a Metal peak of about `2.2 GB` at
  microbatch `1`.
- The hybrid language probe recorded `2.731` bits per byte against `2.799`
  for the dense control and `8.295` at step zero. A continuation from a 2,000
  step checkpoint recorded `2.268` for the hybrid and `2.348` for the dense
  control.
- The toy role probe recorded hybrid accuracy `0.953`, dense accuracy `0.141`,
  and delta-knockout accuracy `0.141`.
- The best isolated count snapshot recorded `0.938` with the PHI path disabled
  and `0.078` with both recall and delta paths knocked out. The corresponding
  PHI-on count result was `0.531`, which exposes interference from the frozen
  recall path.
- The joint 4.6M snapshot `coexist_lm2` recorded passkey `1.00`, count `0.891`,
  feel `1.00`, and imagination `0.984` on `64` evaluation cases. These are
  local snapshot results, not a general language or memory claim.
- The frozen Llama-3.2-1B 4-bit comparison recorded `0.894` bits per byte,
  while the first hybrid student snapshot recorded `2.731` and the continued
  snapshot recorded `2.268`. The student did not reach parity.
- The frozen-trunk specialist probe recorded an open-versus-closed result of
  `2.738` versus `2.787`. This is a small local lift, not proof of durable
  specialist learning across held-out workloads.
- A routed passkey probe at gap `120` recorded accuracy `1.00` and knockout
  accuracy `0.09375`. A harder top-16 gap-`300` probe recorded `0.0703125`
  against chance `0.0625`, showing that hard routing can starve the needle.

## Interpretation

The fit path works on the tested M5 Pro geometry. Small hybrid and role probes
show useful local behavior, and the frozen-trunk specialist path remains a
viable direction for further controlled tests. The evidence does not establish
that the 100M-class model learns long-range passkey recall, matches the 1B
comparison, or provides a general CLS, memory, imagination, or chat system.

The PHI-on and PHI-off differences show that frozen recall features can relay
or damage a downstream task. Specialist docking and coexistence therefore
require explicit frozen-path, knockout, and matched-control accounting. The
4.6M `coexist_lm2` snapshot remains the working object. Repeated 100M-class
gap-32 passkey attempts are not the next efficient probe.

## Publication boundary

This record is the public summary of the local session. The `train/` source,
data, checkpoints, and result registry are deliberately not published in the
same commit. The reported values are preserved as local evidence and must be
reproduced from a future published trainer before they support a public model
claim.

## See also

- `neuroloc/wiki/PROJECT_PLAN.md`
- `docs/STATUS_BOARD.md`
- `docs/EXPERIMENT_LOG.md`
- `README.md`
