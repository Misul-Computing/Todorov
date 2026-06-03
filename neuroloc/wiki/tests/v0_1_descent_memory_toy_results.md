# run card: v0_1_descent_memory_toy (paid)

status: historical context only. frozen as of 2026-06-02. do not edit.

## one-line result

first v0.1 session. authorized by deyan as an architecture reactivation (overriding the
compression-lane / no-paid-compute canonical state) on a personal runpod h200. built a clean,
test-gated descent-memory codebase from scratch and ran a controlled toy comparison on
associative recall (mqar) and induction (passkey). **the descent memory was made numerically
stable but learns at chance on mqar; the linear delta control is also at chance; a 4-layer
attention model solves passkey/induction at exact_acc 1.000 (n=100, wilson 95% lower bound
0.963).** the recurrent-memory retrieval wall is not broken; attention is the working v0.1
retriever.

## headline numbers

- run name: `v0_1_descent_memory_toy`
- code: `v01/` on git branch `v0.1` (commits 2fc8114, c98b745)
- params: ~4,200,000 (toy: d_model 256, 4 layers)
- compute: runpod h200 143GB, torch 2.8.0+cu128, fp32
- tasks: mqar (vocab 64, 32 pairs / 8 queries, seq 80); passkey (seq 128, key_len 5)
- chance token accuracy on mqar: ~1.6%
- substrate variants: linear delta (control), descent memory mlp (treatment), attention (path c)

## training trajectory

- linear control (mqar, 3000 steps): state_norm grew to ~22 and stayed bounded; token_acc ~0.02
  (chance); exact_acc 0 throughout.
- descent memory (mqar): v1 dead mlp init (state_norm 0.000); v3 nlms eps=1e-4 diverged
  (state_norm 12 -> 3.0e6 -> 4.0e8); v4 stabilized (nlms eps=1.0 + bounded sigmoid inner lr +
  hard state-norm clamp 100): state_norm bounded ~11-15, no nan, token_acc 0.011-0.015 (chance),
  masked loss flat at the alphabet prior (~4.16).
- attention (mqar, 2000 steps): token_acc ~0.04 (~3x chance), exact_acc 0.
- attention (passkey, 3000 steps): exact_acc 0.910 @ step 200, 1.000 @ step 800, holding 1.000
  through step 3000.

## eval

- passkey/induction (attention): token_acc 1.000, exact_acc 1.000, n=100, wilson 95% ci
  (0.963, 1.000). checkpoint pulled to `v01/v0.1_passkey_model.pt` (50 MB).
- mqar (all three substrates): exact_acc 0; token_acc at or near chance.

## how this compares to the prior paid runs

| run | substrate | task | retrieval |
|---|---|---|---|
| god_run .. run3_cognition_phase1 (6 paid) | matrix / slot memory | fineweb / synthetic cognition | 0/100 passkey |
| v0_1 descent memory | test-time gradient mlp | mqar | chance (stable, not diverging) |
| v0_1 attention | softmax sdpa | passkey / induction | 100/100 exact |

## what the run validates

- a test-time gradient (descent) memory can be made numerically stable under an adversarial
  outer optimizer via nlms normalization + bounded inner lr + a hard state-norm clamp.
- softmax attention forms induction heads and solves passkey/induction to 100% at 4.2m params.
- the sanity-gate + telemetry methodology catches dead/diverging memory that loss alone hides
  (see the bug log).

## what the run does NOT validate

- it does not show the recurrent memory retrieving. descent memory sits at chance on mqar, like
  the linear control and like all 17 prior runs.
- it does not reach 100m scale or produce a byte-lm bpb number.
- mqar as formulated here (value one token after its key) is unsolvable without a token-shift;
  even softmax attention only reaches ~3x chance. this is an architecture-task interaction.

## diagnosis

stability is not learning. the descent memory is healthy (bounded state, open output gate,
gradients flow, overfits a single batch to ~0) yet does not learn the position-independent
retrieval algorithm on fresh mqar sequences. this is consistent with the substrate-requires-
architectural-change thesis: a numerically-fine substrate can still fail to be trained into
retrieval by sgd at this scale.

## cost

one runpod h200 session (toy scale; minutes of gpu per run). bulk of wall-clock was lost to
bug-fix cycles and intermittent runpod ssh rate-limiting, not to compute. no 100m run.

## see also

- `wiki/synthesis/descent_memory_intervention.md` — mechanism, stabilization, interpretation
- `wiki/mistakes/descent_memory_v0_1_bugs.md` — the seven bugs and the documentation failure
- `wiki/synthesis/substrate_requires_architectural_change.md` — candidate E (test-time gradient)
- `wiki/tests/run3_cognition_phase1_results.md` — prior paid run (sixth, 0/100 passkey)
- `wiki/PROJECT_PLAN.md` — canonical project state
- `wiki/OPERATING_DIRECTIVE.md` — documentation rules
