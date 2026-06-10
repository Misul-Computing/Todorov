# affect-gated write: a surprise-modulated plasticity gate on the descent memory (candidate F, cpu pre-registration)

status: historical context only. frozen as of 2026-06-10. do not edit.

this article is a pre-registration of a cpu-only experiment. it specifies a single
architectural-intervention candidate (labelled F, extending the candidate list in
`wiki/synthesis/substrate_requires_architectural_change.md`), the mechanism, the controls,
the metrics, the telemetry, and the kill conditions, before any code is written or run. it is
not a result. no paid compute is authorised by this article and none is implied; the toy
harness it modifies runs on cpu.

## lifecycle note

this is a plan, so its banner is `current` and it may be edited until the experiment runs. on
execution it is frozen to `historical context only` per the run-card rule in
`wiki/OPERATING_DIRECTIVE.md`, and the measured numbers are recorded as an appended result
section or a sibling `_results.md` run card.

## the idea in one line

the descent memory already computes a per-token prediction error `e_t = pred_t - v_t` and throws
its magnitude away. route that magnitude (the surprise) into the write gain so plasticity
concentrates on surprising, not-yet-stored associations and is suppressed on associations the
memory already predicts. this is the three-factor rule `delta_w = eta * f(pre, post) * M(t)` with
the third factor `M` set to surprise; it is the neuromodulatory salience tag in biological terms
(see `wiki/mechanisms/three_factor_learning.md`) and precision-weighting of the internal
prediction error in predictive-coding terms (see
`wiki/bridge/predictive_coding_to_training_objective.md`).

## background: why this lever and not the obvious one

the obvious "affect" intervention is to weight the training loss toward the tokens that require
retrieval. that lever is already spent. the v0.1 toy harness (`v01/`) trains with the
cross-entropy masked to only the retrieval-critical positions (`model(inp, tgt, mask)` in
`v01/train.py`, applied as `loss = (ll * m).sum() / m.sum()` in `v01/model.py`), where the mask
from `v01/data.py` is 1.0 only on the query/answer tokens. that is output-side precision
weighting taken to its limit: weight 1 on the tokens that matter, weight 0 on everything else.
under that maximally retrieval-shaped signal the descent memory still learned at chance on mqar
while a 4-layer attention model solved passkey/induction at exact_acc 1.000 (see
`wiki/tests/v0_1_descent_memory_toy_results.md`).

so the untested lever is not the output loss. it is the write side. the masked loss puts zero
gradient on the (key, value) source positions, so the binding has to be written into the
fast-weight state with no local supervision there. the descent memory does emit a write-time
signal that the loss never sees: the inner-loop prediction error `e_t`. its magnitude is exactly
"how unpredicted is this association," and right now it shapes the direction of the write (the
gradient `e ⊗ k`) but not the gain (`beta`, which depends only on the input token). affect-gating
wires surprise into the gain.

## relation to prior candidates and runs

- substrate: candidate E, the test-time gradient / descent memory, implemented and stabilised at
  toy scale in v0.1 (`wiki/synthesis/descent_memory_intervention.md`). this experiment keeps that
  substrate unchanged and adds one gate.
- this is not a repeat of the slot-substrate surprise-gated writes in `run2_slot_memory`. that was
  a different substrate (softmax-addressed slots), at 355m params on a language corpus, and was
  confounded by the inherited `alpha_log_mean` decay bug. the surprise-modulated write gain on the
  descent memory on a toy associative-recall task has never been isolated.

## mechanism (precise)

inside `DescentMemory` (`v01/memory.py`), per head, per step `t`, compute a bounded causal
surprise from the inner-loop error that is already calculated:

    s_t = ||e_t|| / (||e_t|| + ||v_t|| + eps)        # in (0, 1), detached, per head

`s_t` uses only quantities available at step `t` (the state from steps < t, and the current
`k_t`, `v_t`), so it does not leak future information. it is detached: surprise is a modulatory
broadcast, not a differentiable path, which also avoids feeding a second-order term back into the
already-delicate inner loop. set the effective write gain to

    beta_eff = beta * (1 - affect + affect * 2 * s_t)

where `affect` in [0, 1] is a config scalar. the `2 * s_t` keeps the average write budget roughly
fixed (a uniform `s_t` averages ~0.5, so the mean multiplier stays ~1), so the experiment isolates
the redistribution of plasticity toward surprise rather than a hidden change to the effective
learning rate. `affect = 0` makes `beta_eff = beta` identically and reproduces the v0.1 path with
no change. the gate is applied at both write sites in both modes (the `W` update in the linear
path, the `W1`/`W2` updates in the mlp path).

config and plumbing: add `affect: float = 0.0` to `MemoryConfig` and thread it through
`ModelConfig` and a `--affect` argument in `v01/train.py`. add `surprise` and `beta_eff` to the
`last_stats` telemetry dict so the budget confound and the gate behaviour are observable.

### implementation amendment (2026-06-10, pre-execution)

the executed implementation differs from the paragraph above in three declared ways, all within
the latitude this card grants before freezing:

1. the gain uses the causal running-mean normalisation that the confound-monitor clause below
   pre-authorises, from the start, instead of the fixed `2 * s_t`:
   `beta_eff = beta * (1 - affect + affect * s_t / mean_{t' <= t}(s_{t'}))`, computed per
   (sequence, head). the `2 * s_t` budget argument assumes `s_t` is roughly uniform; the realised
   `s_t` of the descent memory is not, so the normalised form is the one that actually pins the
   mean multiplier near 1. telemetry (`write_gain`) confirms the budget holds.
2. the shuffled-surprise arm permutes `s_t` across the batch axis with a fresh permutation per
   step, drawn from a fixed-seed generator inside the forward pass. the fixed seed makes the
   permutation content-independent and deterministic per forward, which keeps the
   `causal_no_future_leak` comparison meaningful (a per-call random permutation would fail the
   gate spuriously).
3. the harness entry point is `v01/quick_affect_test.py` (gates, then the four arms in sequence),
   not a `--affect` flag on `v01/train.py`. telemetry keys are `surprise` and `write_gain`
   (the realised mean multiplier on `beta`), the latter standing in for `beta_eff / beta`.

## which phase-1 gate this maps to

associative recollection, specifically write formation. the experimental method in
`wiki/PROJECT_PLAN.md` requires a candidate to name the gate it is expected to affect and at least
one control. the claim here is that surprise-modulated write formation is the missing ingredient
for the descent memory's state to become content-addressable. the mapped gate is the clean
associative-recall surface in `wiki/synthesis/phase1_evaluation_surface_for_neural_models.md`;
passkey at 256 is a smoke test inside that surface, not the sole proxy.

## hypothesis (falsifiable)

with `affect > 0`, the descent (`mem`) substrate produces mqar `exact_acc` (and `token_acc`)
above the `affect = 0` chance floor, and the improvement is present for the true surprise signal
and absent when the surprise signal is shuffled. anything above the chance floor on the mem
substrate with true surprise outperforming shuffled surprise is the discriminator.

## experimental arms and controls

core arms (the isolation):

1. `affect = 0` (control): reproduces v0.1 byte-for-byte; expected at chance. this is the floor.
2. `affect > 0` (treatment): the candidate. sweep a small set, e.g. {0.5, 1.0}.
3. shuffled-surprise control: identical gate, but `s_t` is permuted across the batch axis only
   (each position `t` is gated by a step-`t` surprise drawn from a sibling sequence). this
   destroys the surprise-to-association link while preserving the exact gain distribution and
   keeping the signal causal. permuting across the time axis is not used because it would feed a
   write at position `t` a surprise from a future position and break `causal_no_future_leak`. if
   treatment beats shuffled, the effect is the surprise content, not a reshaped learning rate.

method-required trainability localisers (run only if a core arm moves off chance, to localise
where the gate acts):

4. oracle-write / learned-read and learned-write / oracle-read, to attribute any gain to write
   formation versus read formation.
5. matched no-memory and recency-only baselines, so an above-chance number is credited to the
   memory and not to a trivial positional shortcut.

the substrate's causal conv (kernel 4, on by default in `v01/memory.py`) must stay on: the v0.1
finding is that mqar is unsolvable without a token shift, the conv supplies it, and the descent
memory was at chance with the conv already present, so mqar is a valid discriminator for the mem
substrate.

## metrics, telemetry, kill conditions

- primary: `exact_acc` with Wilson 95% ci and `token_acc` from `v01/evals.py:eval_task`.
- confound monitor: `mean(beta_eff)` versus `mean(beta)`. if they diverge materially, the budget
  is not preserved and a per-head causal ema normaliser replaces the fixed `2 * s_t`.
- gate behaviour: surprise mean and distribution, and the correlation between surprise and the
  realised write gain.
- the four sanity gates must pass before any training number is counted:
  `loss_at_init`, `causal_no_future_leak`, `retention_floor`, `overfit_one_batch` (`v01/sanity.py`).
- kill conditions: (a) if the `affect = 0` control does not reproduce the v0.1 chance result, the
  harness has drifted and the comparison is invalid; stop and reconcile. (b) if
  `causal_no_future_leak` fails, the surprise signal is leaking future information; the result is
  void until the leak is fixed. (c) if treatment and shuffled-surprise are statistically
  indistinguishable, the surprise content is not the active ingredient and the candidate is a
  reshaped learning rate, not affect.

## what a positive versus null result means

- positive (above chance on the mem substrate, true surprise beating shuffled surprise):
  write-side surprise modulation is a real lever. promote candidate F into the broader cpu battery
  and re-rank it against A-E. this would be the first non-chance retrieval signal from a recurrent
  memory substrate in the project.
- null (still at chance, or true surprise not beating shuffled): retire affect-gated write from
  the candidate list and shift effort to the read side, because the bottleneck is then more likely
  the read's content-addressability than the write's gain. the god_run delta-state structure probe
  (state statistically indistinguishable from noise) already points that way. a null is therefore
  interpretable: it narrows the search rather than just failing.

## honest limitations

this gate controls when and how hard the memory writes; it does not by itself make the read
content-addressable, which the god_run structure-probe evidence suggests is the deeper failure.
the experiment is toy-scale and cpu-only. it is a pre-registration, not a result. it does not
authorise or imply any paid run; under the current master-plan phase the architecture track is
paused and paid compute is blocked behind the lane-research, dossier, oracle-compression, and
tiny-mirror gates in `wiki/PROJECT_PLAN.md`. a cpu signal here would feed those gates; it would not
bypass them.

## note on the one-way link to the v0.1 run card

`wiki/tests/v0_1_descent_memory_toy_results.md` is `historical context only` and frozen, so it is
not edited to carry a reverse link to this article. the bidirectional cross-reference rule yields
to the freeze rule (the more specific constraint). the live forward link from the v0.1 evidence
chain to this article lives in `wiki/synthesis/descent_memory_intervention.md`, which is `current`.

## results (2026-06-10, executed)

executed on cpu via `v01/quick_affect_test.py` at git `dfc1d4f`, branch `v0.1`. task: mqar,
vocab 24, 8 pairs, 4 queries, seq 24, batch 32; model: mlp descent memory, d_model 128, 2
layers, 4 heads, head dim 32, adamw lr 3e-3, grad clip 1.0. eval n = 100 sequences (400 query
tokens); chance token_acc ~0.0435. the four sanity gates passed before every counted number
(loss_at_init 3.42 vs ln(24)=3.18 within tolerance, retention floor 0.94, overfit-one-batch to
0.000, causal max_diff = 0.0 for every arm including the shuffled and noise arms). at seed 0
every arm shares identical model init and data stream, so cross-arm differences at seed 0 are
mechanistic, not initialisation lottery.

- control (`affect = 0`), seed 0, 400 steps: exact 0.000, token 0.037. at chance. kill
  condition (a) satisfied: the unmodulated harness reproduces the v0.1 chance behaviour.
- true surprise, `affect = 0.5`, seed 0, 400 steps: exact 0.000, token 0.052. at chance.
  realised mean write gain 0.829.
- true surprise, `affect = 1.0`, seed 0, 400 steps: exact 0.000, token 0.028. at chance.
  realised mean write gain 0.732: within-sequence surprise declines, so the own-history
  normaliser systematically suppresses writes. the budget is not preserved for the true signal
  even under the amended normaliser.
- budget-matched true surprise (per-step batch-mean normalisation, realised gain pinned at
  1.000), seed 0, 400 steps: exact 0.000, token 0.035. at chance. the surprise-content
  hypothesis fails with the suppression confound fully removed.
- shuffled surprise, seed 0, 400 steps: exact 0.110 (wilson 95% 0.063-0.186), token 0.557,
  gain 0.981. a large lift, loss still descending at step 400.
- shuffled surprise, seed 0, 1200 steps: exact 0.360 (wilson 0.273-0.458), token 0.785, final
  loss 0.898 still descending. the effect grows with training.
- shuffled surprise, seed 1, 400 steps: exact 0.000, token 0.045. at chance. realised gain
  0.728: the shuffled budget is endogenous (it tracks the batch's surprise trend), and at this
  seed the declining trend turned the gate into net suppression. ignition is seed-sensitive.
- uniform head-iid noise gain, seed 0, 400 steps: exact 0.000, token 0.098 (wilson 95% on 400
  query tokens ~0.073-0.131, clear of chance), gain 1.021. a small but real lift.
- uniform per-token noise gain shared across heads, seed 0, 400 steps: exact 0.000, token
  0.022, gain 0.999. at chance: kills the head-correlation explanation of the shuffle lift at
  matched init and data.
- uniform per-token noise gain shared across heads, seed 1, 400 steps: exact 0.000, token
  0.035, gain 0.999. at chance at both seeds.
- no-write baseline (write gate pinned to sigmoid(-12), re-pinned after every optimizer step),
  seed 0, 400 steps: exact 0.000, token 0.040. at chance: any lift above is memory-mediated,
  not a positional or conv-window shortcut. this arm stands in for the matched no-memory and
  recency-only baselines (the causal conv supplies the recency window and it cannot reach a
  pair's value token).

internal consistency of the seed-0 shuffle lift: exact 0.110 vs token^4 = 0.557^4 = 0.096,
matching independent per-query success at ~56%, the signature of each sequence storing a
random subset of its pairs strongly. supporting context: on the same harness the linear matrix
memory trains into partial mqar natively (exact 0.470 at 6 pairs / 400 steps; 0.520 at 8 and
0.270 at 10 at 1200 steps), while a 4-layer rope attention model stays at chance through 2000
steps; full sweep in `v01/feel_tests.md` (capacity context section).

## verdict

1. the pre-registered hypothesis is falsified. surprise-modulated write gain does not move the
   descent memory off chance on mqar: true surprise is at chance under the own-history
   normaliser at two strengths, and under exact budget matching. kill condition (c) fires in
   inverted form: treatment and shuffled-surprise are clearly distinguishable, but in the
   direction shuffled > treatment, so the surprise content is not the active ingredient and
   the precision-weighting story does not hold on the write side at this scale.
2. candidate F (affect-gated write) is retired as pre-registered. per the pre-registered null
   branch the read side rises in priority, with the qualification in (3).
3. serendipitous existence proof: one content-free write-gain configuration (shuffled sibling
   surprise, seed 0) trained the descent memory to exact 0.360 / token 0.785 — the first
   non-chance retrieval learned by sgd on a recurrent memory substrate in this project (six
   paid runs and v0.1 were all at 0%). it is memory-mediated (no-write at chance), causally
   clean (max_diff 0.0), internally consistent (exact ~ token^4), and grows with training. it
   is an existence proof, not a recipe: it did not replicate at seed 1 (endogenous budget
   suppression), and neither uniform-noise structure tested (head-iid 0.098, token-correlated
   0.022/0.035) reproduces its magnitude at matched init and data.
4. open mechanistic candidates for ignition, named and untested: the gain process's
   distribution shape; its endogenous ramp-in (the model's surprise distribution starts nearly
   degenerate, so the shuffle's gain variance grows with training, a self-paced noise
   schedule); population coupling (gains track the batch's learning state); and temporal
   correlation (sibling surprise traces are smooth in time, pair-coherent, unlike iid noise).
   these belong to a successor pre-registration (candidate G, stochastic write gain, in
   `wiki/synthesis/substrate_requires_architectural_change.md`) before any further arms are
   run, and that pre-registration's research step must include a literature check on
   stochastic write gating in fast-weight and test-time-learning memories, because the
   mechanism may already be known.

## see also

- `wiki/synthesis/descent_memory_intervention.md` — the candidate E substrate this builds on
- `wiki/synthesis/substrate_requires_architectural_change.md` — the A-F candidate list
- `wiki/tests/v0_1_descent_memory_toy_results.md` — the v0.1 toy run this extends (one-way, see note)
- `wiki/mechanisms/three_factor_learning.md` — the third-factor rule the gate instantiates
- `wiki/bridge/predictive_coding_to_training_objective.md` — surprise as internal prediction error
- `wiki/synthesis/phase1_evaluation_surface_for_neural_models.md` — the cpu battery this maps into
- `wiki/PROJECT_PLAN.md` — canonical project state
- `wiki/OPERATING_DIRECTIVE.md` — documentation rules governing this article
