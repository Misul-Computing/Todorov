# feel tests: a sensory channel on the v0.1 toy

status: current (as of 2026-06-10). cpu-only exploratory results, not paid compute.

## what this is

a small cpu bench that tests whether giving the toy model a "sense" (touch) helps it
learn, with a built-in cheat-detector. it grew out of the affect / predictive-coding steer:
the question is not "remember harder" but "does a felt sense the system can locate and
accumulate help it learn at all."

## method

token-sequence worlds run through the existing v0.1 `SequenceModel` (linear matrix memory,
no attention layers). the agent sweeps a hidden strip of cells; each step pairs a position
token with a felt value (bump/flat) when touch is on, or a blank token when touch is off.

three arms per test, the honesty contract:

- blind: sense off (felt cells are blank).
- real: sense on, carrying the truth.
- fake: sense on, carrying random garbage decorrelated from the answers.

an honest test passes only if real wins while blind and fake both stay at chance. if fake
also scores high, the harness is leaking the answer and the result is void. every arm also
passes `causal_no_future_leak` (no peeking at future tokens).

## results

cpu, linear memory, strip length 6, ~100-130s per 3-arm run.

recall (what was at position p):

- blind  exact 0.14, fake exact 0.15, real exact 1.00.

count (how many bumps total, 0..6):

- blind  acc 0.29, fake acc 0.34, real acc 1.00.

both verified by the fake-touch control: the identical-but-garbage sense gives no lift, so
the gain comes from the felt content, not from extra tokens or a leak.

robustness (count under bit-flip sensor noise):

- clean 0.98, 10% noise 0.44, 20% 0.41, 35% 0.39, 50% 0.39 (blind 0.32). exact counting is
  inherently fragile to per-bit noise: one flipped cell changes the total, so the answer is
  wrong. the cliff is a property of the task, not the model. a noise-tolerant system needs an
  approximate or redundant representation, not exact senses.

transfer probe (does the count-body keep per-cell detail or just the total):

- per-cell recovery, chance 0.500: real-touch count-body state 0.717, sum-only ceiling 0.655,
  blind body 0.499. learning to count compressed the felt strip to roughly just the total, with
  only a sliver of residual per-cell detail. the representation is question-shaped, not
  world-shaped. blind at chance confirms the probe does not leak. this motivates self-supervised
  prediction and multi-task experience as the route to a reusable representation.

predictive-coding pretraining (feel-and-predict a structured world, then learn to count; avg 3 seeds):

- structured world: each cell sticks to the previous with prob 0.8, so the sensory stream is
  predictable. self-supervised pretraining reaches next-sensation accuracy 0.767 (chance 0.500).
- count accuracy after few finetune steps [25, 50, 100, 200]: pretrained 0.115 0.272 0.660 0.990;
  from-scratch 0.692 0.927 0.935 0.975. NEGATIVE TRANSFER: the feel-and-predict body learns
  counting much slower and only catches up by step 200.
- hypothesis: predicting the next sensation shapes a recency representation (track the current
  value), which is the opposite of the persistent accumulation counting needs. predictive coding
  is not a free lunch; alignment between what is predicted and what the task needs decides whether
  it helps. confirm by testing a local task (recall), where prediction and task align and
  pretraining should help.
- alignment test result (recall finetune, same protocol, avg 3 seeds; next-sensation accuracy
  reproduces at 0.767): pretrained 0.618 0.751 0.748 0.884 vs from-scratch 0.799 0.888 0.948
  0.986 at steps [25, 50, 100, 200]. NEGATIVE TRANSFER AGAIN, on the task picked to be
  prediction-aligned, and the pretrained body does not catch up by step 200. the alignment
  hypothesis is falsified: the slowdown is generic at this scale, not a prediction-vs-counting
  misalignment. feel-and-predict pretraining displaces this small body somewhere that slows every
  subsequent readout tested. self-supervised prediction at toy scale is a cost, not a free lunch,
  for both global-aggregate and local-binding questions.

hard test (can a frozen body answer a NEVER-trained question from its representation of the sweep):

- novel question 1, left-vs-right majority (linear/spatial), chance ~0.50: untrained 0.977,
  multitask+touch 0.926, count-only+touch 0.836, multitask+blind 0.504. clean result in the middle
  rows: varied experience preserves reusable spatial structure that narrow (count-only) experience
  discards. blind at chance confirms the felt content is what is used. but the untrained body wins:
  training erodes reusable information, shaping the representation toward the trained task and
  discarding the rest (no training discards nothing, narrow discards most, varied in between).
- novel question 2, parity (nonlinear/global), chance ~0.58: untrained 0.699, count-only 0.711,
  multitask 0.674, blind 0.578. no clear separation; experience did not unlock parity here. caveat:
  the probe reads the post-sweep state, but the count (hence parity) is only surfaced when the model
  is explicitly queried, so this likely understates recoverability. inconclusive, not a clean no.
- parity SETTLED (probe at the count-query position, i.e. with the COUNT token appended and the
  hidden state read where the trained question is asked; sweep-end numbers reproduce within ±0.04):
  multitask 1.000, count-only 1.000, blind 0.578 (chance), untrained 0.697. the earlier
  inconclusive parity was a probe-placement artifact. count-trained bodies carry perfectly
  linearly-decodable parity exactly where the question surfaces the count; the untrained body does
  not. left-vs-right at the query position stays untrained-dominated (0.988 vs 0.852/0.854), so
  querying reorganizes the global aggregate, not the spatial detail.
- answer to "does accumulated experience answer new questions" (revised): yes, with a location
  caveat. varied experience preserves more reusable spatial structure than narrow experience, and
  the global aggregate (count, parity) becomes perfectly decodable, but only at the position where
  the trained question surfaces it. training still trades away spatial detail the untrained body
  keeps. the engineering problem stands: training that keeps the world, not just the answer, and
  representations that surface it everywhere, not only at the query slot.

## capacity context (mqar pairs sweep, linear matrix memory vs attention)

not a feel test; substrate context for this bench and for the affect-gate experiment
(`neuroloc/wiki/tests/affect_gated_write_cpu_experiment.md`). mqar vocab 32, 4 queries,
d_model 256, 4 layers, batch 32, lr 3e-3, n=100 eval, single seed, `quick_capacity_test.py`.

- linear matrix memory, 400 steps: exact 0.470 (token 0.830) at 6 pairs; chance at 10, 14,
  18, 24 pairs. first above-chance mqar from a recurrent memory substrate in this project.
- undertraining control, 1200 steps: 8 pairs exact 0.520 (token 0.865), 10 pairs exact 0.270
  (token 0.710). the 400-step cliff at 10 pairs is an optimization cliff, not a hard storage
  wall: more pairs need disproportionately more steps. a trainability slope, not a capacity
  wall, at this size.
- attention reference (4 layers, rope): chance at every pair count at 400 steps, and still at
  chance at 2000 steps (6 and 10 pairs, token 0.080 / 0.055). the v0.1 attention retriever
  solved passkey and induction, never mqar; attention-on-mqar at this width/lr/step budget
  does not train. no contradiction with v0.1, but a useful negative: the "guaranteed floor"
  substrate is task- and budget-dependent.
- reading: at these step budgets the dominant constraint for both substrates is trainability
  (whether sgd finds the circuit), not representational capacity. this matches the affect-gate
  finding that a trainability intervention (write-gain stochasticity), not a capacity change,
  moves the descent substrate off chance.

## reading

a truthful sense turns an otherwise unsolvable task into a solved one. recall shows the model
can bind a felt value to its location and retrieve it. count shows the sense also feeds an
integration over the whole sweep (a running total), not just single-fact lookup. an empty or
garbage sense is correctly ignored.

## what this does NOT show

- no transfer yet: every answer was directly trained. it does not show that a sense helps
  answer questions it was never trained on, or learn other tasks faster.
- nothing about consciousness or agi. this is foundational plumbing plus a verified bench.
- single seed per task at toy scale on the linear memory only.

## next

- stochastic write gain (the affect-experiment anomaly, see
  `neuroloc/wiki/tests/affect_gated_write_cpu_experiment.md`): ignition statistics across
  seeds, and which property of the gain process (distribution shape, endogenous ramp-in,
  population coupling) drives ignition. needs its own pre-registration before more arms.
- pair-coherent and temporally correlated write noise as the next structure probes (the
  head-correlation probe came back null; temporal structure is the open candidate).
- training that keeps the world: objectives that surface aggregates at every position, not
  only at the query slot. the settled parity result shows trained information is organized
  at the asked position only.
- why feel-and-predict pretraining hurts both aligned and unaligned finetuning at this scale
  (representation displacement vs loss-landscape position): distinguish by probing what the
  pretrained body still linearly contains before finetuning.

## files

- `v01/data.py` — `make_touch_world` (recall), `make_touch_count` (count), `make_sweep`
  (feel-and-predict world), `_structured_strip` (p_stay-correlated strips).
- `v01/quick_touch_test.py` — 3-arm recall verifier.
- `v01/quick_count_test.py` — 3-arm count verifier.
- `v01/quick_robust_test.py` — count under bit-flip sensor noise.
- `v01/quick_probe_test.py` — per-cell linear probe on the count-body state.
- `v01/quick_pc_test.py` — predict-then-finetune transfer curves (`--task count|recall`).
- `v01/quick_hard_test.py` — never-trained-question probes (sweep end and count-query position).
- `v01/quick_capacity_test.py` — mqar exact_acc vs n_pairs (memory vs attention substrate).
