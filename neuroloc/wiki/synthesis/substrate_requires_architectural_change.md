# substrate requires architectural change: six paid runs, zero retrieval, one discriminating corpus

status: current (as of 2026-04-26).

## the observation

six consecutive paid h200 runs across two substrates, two retention
regimes, and two corpora have produced 0% passkey at 256 tokens:

| # | run | substrate | retention | corpus | val_bpb | passkey@256 |
|---|---|---|---|---|---|---|
| 1 | god_run | matrix | broken inherited | fineweb-edu | 1.3950 | 0/20 |
| 2 | god_run_v2 | matrix | broken inherited | fineweb-edu | 1.4453 | 0/100 |
| 3 | run1_baseline_noerasure | matrix | broken inherited | fineweb-edu | 1.4499 | 0/100 |
| 4 | run2_slot_memory (first) | slot | broken inherited | fineweb-edu | 1.5107 | 0/100 |
| 5 | run2_slot_memory_retention_fixed | slot | explicit 5.0 | fineweb-edu | 1.4777 | 0/100 |
| 6 | run3_cognition_phase1 | slot | explicit 5.0 | synthetic cognition | 6.3519 | 0/100 |

run 6 is the project's first and only paid test on a corpus where
retrieval is an explicit and majority component of the training loss.
the substrate still did not learn it.

## why this supersedes the training-objective hypothesis

the 2026-04-16 analysis (`training_objective_vs_architectural_goal.md`)
argued that the five prior runs failed because next-byte cross-entropy
on natural text does not reward routing information through memory,
so gradient descent never learns to use the substrate. the proposed
discriminant was a paid run on cognition-shaped data. that article
predicted:

> "option 3 is the cleanest discriminant. if phase one cannot produce
> any non-zero passkey accuracy on synthetic data, the substrate
> genuinely cannot be trained by sgd at this scale and the architecture
> needs deeper changes."

run 3 is phase 1. passkey accuracy is 0. the predicted branch fires:
the substrate cannot be trained by sgd at this configuration, and the
architecture needs deeper changes.

the prior article's reasoning structure remains correct — its three
proposed reformulations (synthetic cognition, mixed pretraining, two-
phase) and its bits-per-parameter quantitative bar are intact. what
this article adds is the empirical answer to the discriminant. it is
the worst-case branch. this article does not supersede the prior one;
it records the outcome of its proposed test.

## what exactly happened in run 3

4000 steps, 355M params, slot memory with retention fixed, FLA active,
synthetic cognition corpus with 50% passkey blocks / 30% kv recall /
20% copy. the passkey blocks literally contain `MARK_STORE_START +
passkey + MARK_STORE_END + filler + MARK_QUERY + passkey` at byte level.
the cross-entropy loss directly rewards predicting the right hex digit
at the position immediately after MARK_QUERY.

training loss dropped from 5.74 at step 0 to 4.51 at step 50 and then
stayed in the band [4.34, 4.42] for the next 3950 steps. val_bpb
hit its best (6.3519) at step 2150 and never improved past it.

the model is fitting the alphabet prior (hex digits + printable filler
+ markers, expected uniform over ~100 unique bytes, so val_bpb ~ log2(
values observed) ≈ 6.4 on the cognition corpus's byte distribution).
it is not fitting the retrieval pattern. the slot output gate is
believed to have stayed near its near-closed initialisation (gate_init=-4,
sigmoid(-4)≈0.018) throughout run 3, consistent with run 2's measured
mean gate of 0.018 at every step across its 4000 steps (see
`wiki/tests/run2_slot_memory_retention_fixed_results.md`), but run 3's
per-step gate telemetry was not persisted before the pod was stopped,
so this claim rests on inference from the flat loss curve and from the
unchanged preset rather than on a direct run-3 measurement. the loss
curve has no visible inflection where retrieval would start to be
learned, and the prototype keys cannot be reconstructed without the
missing telemetry.

## possible causes, ranked

1. **the slot substrate has an optimisation pathology under SGD in this
   architecture.** the output gate's initial sigmoid(-4)≈0.018 means
   slot output contributes 1.8% of the signal at init; gradient
   through a gate this closed is weak; the gradient to open it depends
   on the slot already producing useful output, which requires the
   prototype keys to already be useful, which requires them to already
   be open. the fixed point is the closed state. no explicit auxiliary
   signal pulls the system out of it. this is consistent with the mean
   gate staying at 0.018 through run 2's 4000 steps and the same
   plateau in run 3.

2. **the softmax addressing learns one-slot collapse at init.** when
   prototype keys are initialised randomly at std=0.02, softmax over
   per-token logits favours one slot per token deterministically within
   a few steps; all token writes go to that slot; the slot bank
   degenerates to a rank-1 accumulator equivalent to a plain linear-
   attention state. this would make slot memory equivalent to the
   matrix memory the project was trying to replace, which is already
   known to fail.

3. **the passkey 50% fraction may still be too dilute.** at block_seq_len
   512 each passkey block dedicates ~15 bytes to the actual retrieval
   target; the remaining ~497 bytes are filler that the attention layer
   can compress trivially. the loss signal on retrieval positions is
   ~3% of the total loss; if the substrate cannot exploit it, the
   optimiser spends 97% of its gradient on learning to predict filler,
   which has no retrieval content.

4. **the data format is pessimal for a byte-level model.** at every
   step where the next byte is a passkey byte, the correct prediction
   requires consulting the entire context; at every other step, the
   correct prediction is the alphabet prior. the model sees ~3% of
   tokens where memory matters and ~97% where it actively doesn't.
   gradient descent with this ratio may systematically fail to form
   the specialised pathway.

5. **4000 steps / 131M tokens is just too few for this architecture
   shape.** no evidence supports this — the plateau is flat from step
   150, not slowly descending. but it cannot be ruled out without a
   longer run.

## paused intervention candidates

these intervention candidates remain useful historical backlog structure, ordered by cost and risk. they are not authorised for code changes or paid execution in the current master-plan phase. the next active work is the six-lane research gap map, followed by oracle compression analysis over the hard symbolic worlds and then a tiny trainable mirror only if the oracle ratios justify it.

### A. open the output gate at init

set `slot_gate_init=0.0` (sigmoid(0)=0.5) rather than `-4` (sigmoid(-4)≈
0.018). this gives the slot output 50% of the contribution at step 0
instead of 2%. the hypothesis is that the initial gradient depends on
the slot having some voice in the residual stream. the v2b iteration
of the cpu gate C sim tried this at tiny scale and saw no effect — but
tiny scale was underspecified for retrieval anyway. this remains a
candidate mechanism to localise under the future tiny mirror and cpu
controls, not a launch plan.

### B. auxiliary retrieval loss on marker-following positions

add a loss term that weights positions immediately following
MARK_QUERY by 10× or 100×. the cognition corpus already has the
marker bytes that identify these positions; the weighting is a
mechanical change in the loss function. this directly attacks the
3%-of-tokens-that-matter dilution (cause 3 / 4 above). the v2 tiny-
scale iteration used the same weighting (10×) and val_bpb did not
improve meaningfully. the interaction with retention remains a
future controlled hypothesis, not a launch reason.

### C. orthogonal prototype key init

replace the `std=0.02` random init of prototype keys with an
orthogonal init (e.g., random QR decomposition scaled to match the
variance budget). the hypothesis is that one-slot collapse (cause 2)
is an init artifact: orthogonal keys are by construction not collinear
and the softmax at init assigns tokens across slots more uniformly.
this is a single-line code change.

### D. curriculum phase with hand-placed addresses

warm-start training on a task where the slot bank has the correct
(key, value) pair pre-placed and the model only needs to learn the
readout. if the model can learn the readout under this setup, the
pathway exists and the write side is the bottleneck. if it cannot,
the readout itself is the bottleneck. this splits the question that
SGD is failing to answer into two simpler questions.

### E. substrate replacement

replace the slot buffer with a different content-addressable
mechanism. candidates:
- titans-style fast-weight MLP (test-time meta-learned associations)
- larimar-style orthogonal recursive least squares
- an explicit differentiable key-value table with hard attention at
  training time

each is a substantial architectural commitment. rank them by cpu
simulation cost only after the oracle-compression and tiny-mirror
preparation lane has produced a selected hypothesis.

## paid-run status under the current master-plan phase

no paid run is authorised from this article. A-E are paused backlog
candidates, not the current work queue. another paid run on the current
slot substrate with only hyperparameter changes remains predicted to
be scientifically weak because it would not answer the paper-spine
question: what operation-preserving compression bounds exist before
training?

paid compute can return only after the current canonical gate is met:
the six-lane research gap map, oracle compression analysis, a tiny trainable mirror, cpu controls,
telemetry, prosecutor-clean docs, and one selected hypothesis.

note (2026-04-21 / 2026-04-25): the project's backlog method widened
after this article. when the architecture track resumes, A-E
candidates are no longer judged by passkey alone.
they must first clear the broader cpu battery in
`wiki/synthesis/phase1_evaluation_surface_for_neural_models.md`,
including explicit trainability controls and state/action metrics.
passkey at 256 remains necessary, but now as a smoke test inside that
larger battery. after the master-plan refactor, they also inherit the
six-lane research, oracle-compression, and tiny-mirror gates before any paid path.

## see also

- [[neural_model_paper_spine]] — paper-spine layer that treats this article as negative empirical motivation
- [[oracle_compression_analysis_plan]] — next no-paid proof step before trained compression or architecture work
- [[neural_model_lane_trainability_evaluation]] — lane that localizes learned-write, learned-read, gate, address, and decoder failure before paid work
- `wiki/tests/run3_cognition_phase1_results.md` — the run card for this observation
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the prior analysis whose predicted discriminant this run executed
- `wiki/synthesis/slot_memory_design.md` — the substrate design itself
- `wiki/synthesis/phase1_evaluation_surface_for_neural_models.md` — the current external-synthesis article for the cpu-first battery that should replace passkey-only thinking
- `wiki/synthesis/synthetic_shared_world_bridge.md` — the phase-2 bridge that keeps multimodality tied to one latent world instead of a bolt-on stack
- [[indexed_reconstruction_compression]] — current compact-handle plus reconstruction direction after the cellular, capacity, compression, and evaluation-gap pass
- `wiki/synthesis/linear_attention_retrieval_wall.md` — the original matrix-memory diagnosis (already marked superseded)
- `wiki/tests/run2_slot_memory_retention_fixed_results.md` — the prior paid run with same substrate + retention on fineweb
- `wiki/PROJECT_PLAN.md` — the canonical project state
- `neuroloc/data/cognition_corpus.py` — the corpus generator used in run 3
- `neuroloc/model/god_machine.py` — the architecture implementation
- `wiki/OPERATING_DIRECTIVE.md` — rules governing this article
