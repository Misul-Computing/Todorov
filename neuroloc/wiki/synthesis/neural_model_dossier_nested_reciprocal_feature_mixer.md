# Neural model dossier: nested-reciprocal feature mixer

status: current (as of 2026-07-19).

## Classification

This is an unvalidated external-mechanism candidate for the Monodratic
token-local feature-mixing slot. It is not project novelty. It is not memory,
attention, recurrent state, compression, replay, or imagination.

The source is the 2026 paper
[CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766),
accepted at ICML 2026 according to the
[IBM publication record](https://research.ibm.com/publications/cofrgenet-continued-fraction-architectures-for-language-generation).
The paper calls its feed-forward replacement CFFN. This dossier uses a
project-native name for the mathematical operation and keeps the published
name only for attribution.

## Target failure

The current Transformerov host uses an expanded gated projection. Monodratic
full-model configurations use their gated feature block only when a nonzero
feature width is configured; mixer-only configurations omit that block. The
candidate therefore attaches only to a host that exposes the token-local
feature slot. It asks whether a non-expanded gate plus a small
reciprocal-ladder ensemble can preserve token-local feature transformation with
fewer learned values, without disturbing the separation between routed recall
and recurrent world tracking.

The candidate does not address a known failure of the routed mixer or recurrent
state. Parameter reduction alone is insufficient. It must preserve task quality
and module necessity at an honest measured cost.

## Mathematical operation

For one normalized token vector `x` of width `p`, the paper's figure depicts the
non-expanded gate

`h = (g_left x) * swish(g_right x)`.

The prose does not state this gate equation explicitly. Both branches in the
figure have dense input connections, but the drawing alone cannot settle
parameter sharing.

For ladder `j` with depth `d`, compute

`a^(j) = w^(j) h`,

where `w^(j)` has shape `[d, p]`. Define the continuants in reverse denominator
order:

`k_0 = 1`,

`k_1 = a_d`,

`k_r = a_(d-r+1) k_(r-1) + k_(r-2)` for `r = 2, ..., d`.

The fractional output is

`z_j = k_(d-1) / k_d`.

The ensemble output is

`y = u h + v z`,

where `u` has shape `[p, p]`, `v` has shape `[p, l]`, and `z` has length `l`.
The host then adds `y` to its residual stream.

The paper's collapsed ensemble equation contributes one full direct projection
`u`, `l d p` denominator values, and `l p` output values. Combined literally
with the two gate projections in the figure, this gives

`3 p^2 + l p (d + 1)`

learned values without biases. Table 1 instead reports

`2 p^2 + l p (d + 1)`.

The paper does not reconcile the figure, the collapsed equation, and the table.
One gate projection may be shared or absorbed, the direct path may be omitted
in this component, or the table may use a different depth convention. No
official implementation was linked from the paper or IBM pages as of
2026-07-14 to resolve this. A local implementation must not claim paper
fidelity until it states which interpretation it uses and reports both the
literal learned-value count and the table comparison. Any bias, normalization,
range state, or optimizer state must be reported separately.

## Pole handling and backward rule

The denominator used for division is

`sign(k_d) max(abs(k_d), epsilon)`

with `epsilon = 0.01` in the paper's experiments. The implementation contract
must define `sign(0)` explicitly because the written expression otherwise
leaves an exact-zero denominator ambiguous.

The published derivative away from clamp boundaries is

`dz / da_k = (-1)^k (k_(d-k) / k_d)^2`.

The paper computes the continuants and one reciprocal per ladder in a custom
backward operation. It records each ladder's training minimum and maximum and
clips that ladder's evaluation output to the recorded range. Clamp-gradient
behavior, initialization, range updates under distributed training, final
selected ladder count and depth, and the figure-equation-table parameter
discrepancy are not fully specified.

## Published evidence boundary

The paper reports competitive results at fewer learned values when replacing
feed-forward blocks in GPT-2 XL and Llama 3.2B hosts. The feed-forward-only
variants have 985M learned values against the 1.5B GPT host and 2.1B against the
3.2B Llama host. Reported task results are mixed rather than uniformly
superior. The paper's staged depth schedule materially improves perplexity.

The reported runs used 16 H100 devices for GPT pretraining and 128 H100 devices
for Llama pretraining. Those results do not establish CPU speed, Monodratic
compatibility, Todorov quality, matched operation count, or a useful setting at
the project's scale. Division remains a hardware concern, and the paper leaves
custom kernels as future work.

No official implementation or weights were linked from the paper or IBM pages
as of 2026-07-14. No community implementation is accepted as a reference by
this dossier.

## Staged training requirement

The paper starts with only the linear component trainable. Denominator depth
`i` becomes trainable after fraction `1 - 1 / 2^i` of the total steps, so it
receives only the final `1 / 2^i` fraction of updates. This schedule is part of
the mechanism under test, not an optional optimization.

When separate specialist training is tested, the depth schedule belongs inside
feature-specialist training. Docking starts only from a finished immutable
specialist and may update only its zero-born bridge and scalar gate.

## CPU test material

The mathematical gate comes before model training:

1. Compare literal nested fractions with the continuant forward result at
   depths 1, 3, 5, and 7 in float64 away from poles.
2. Compare the custom gradient with automatic differentiation and finite
   differences away from clamp boundaries.
3. Probe exact zeros, both sides of the clamp, large magnitudes, mixed
   precision, empty batches, state reload, and evaluation-range clipping.
4. Assert denominator shape `[batch, time, ladders]`, exactly one division per
   token and ladder in the continuant path, and the complete learned-value
   ledger for each explicit interpretation.
5. Prove the selected Transformerov full-sequence host path calls the selected
   feature module, preserves output shape, and reloads exactly. Single-step or
   paged parity becomes relevant only if that selected host later exposes those
   paths; legacy Todorov and full-model Monodratic stateful interfaces are not
   adopted by this dossier.

The first training gate reuses the existing small deterministic byte-language
harness with at least three seeds. Its arms are the deployed gated feature
mixer, a parameter-matched gated feature mixer, the diagram-and-equation
interpretation with staged depth release, a separately named table-count
interpretation if one can be specified without hidden sharing, the candidate
with its fractional path disabled, and a schedule-disabled diagnostic. The
routed mixer, recurrent mixer, data, tokens, steps, optimizer, and stopping rule
remain fixed.

Associative recall is a regression check, not the primary feature-mixer
measure. The primary measure is held-out byte loss. The paired recall and
world-tracking battery must also retain its module-knockout signatures.

## Controls

- Deployed expanded gated feature mixer
- Parameter-matched gated feature mixer
- Direct linear path with the fractional contribution disabled
- Literal nested-fraction reference
- Staged schedule disabled
- Ladder order shuffled after training
- Evaluation clipping disabled as a diagnostic
- Routed recall disabled
- Recurrent state disabled and shuffled
- Matched seeds, data, tokens, steps, optimizer, and stopping rule

## Telemetry

- Learned values and optimizer-state values by module, with the paper-table
  discrepancy shown
- Analytic multiply, add, and division counts
- Warmed CPU tokens per second and peak memory
- Active depth and gradient norm by depth
- Minimum absolute denominator and clamp rate
- Ladder training ranges and evaluation clip rate
- Direct-path and fractional-path output norms
- Finite output and gradient rates
- Held-out loss by seed
- Recall route hit rate and overflow
- Recurrent gate values and state-knockout effect
- Full-sequence and reload parity

## Promotion and kill conditions

The candidate is killed before training if the parameter-sharing contract
remains implicit or if forward parity, gradient parity, shape, serialization,
division accounting, or finite-value checks fail.

After training it is killed if any of these conditions holds:

- The fractional contribution collapses and the direct-path control ties it.
- Clamp or evaluation clipping remains a routine operating path rather than a
  rare guard.
- The parameter-matched gated control is at least as good on every task and no
  measured cost advantage remains.
- Held-out loss worsens by more than the predeclared tolerance across the three
  seeds without a meaningful size, memory, or wall-time benefit.
- Routed-recall or recurrent-state knockout signatures disappear.
- The result depends on an unreported fallback, unmatched schedule, different
  token budget, or uncharged state.

Promotion requires finite stable dynamics, correct accounting, a non-collapsed
fractional contribution, no loss of the recall and world-state roles, and a
predeclared quality-cost tradeoff that survives all seeds. A single good seed
or a smaller checkpoint is not sufficient.

## Cost gate

The mechanism is already published, and the paper's separate attention
replacement is outside this scope. Inventing a framework around the feature
mixer has no present value. A standalone feature module and the existing CPU
harness are sufficient for falsification once implementation is explicitly
authorized. Paid compute has no justified expected value before CPU validation
and remains unauthorized.

## Future prerequisite

If implementation is explicitly authorized later, first resolve the
figure-equation-table discrepancy from a primary implementation or an explicit
local contract. No code or training run is authorized by this dossier. Consult
[[PROJECT_PLAN]] for the current action and authorization state.

## See also

- [[PROJECT_PLAN]]
- [[Home]]
- [[INDEX]]
- [[modular_neural_model_stack]]
- [[modular_neural_model_methods_review_2026_07]]
