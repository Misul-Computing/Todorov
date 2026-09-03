# Modular neural model methods review, July 2026

status: current (as of 2026-07-19).

Fact-checked against the pinned sources below on 2026-07-19.

## Decision

The literature and local-source review does not change the first combined CPU
proof. The frozen Transformerov plus public Monodratic base remains the only
model in that proof. No reciprocal feature mixer, alternative router, output
gate, recurrence change, backend rewrite, optimizer change, specialist
protocol, or cache enters it.

Later work follows the ranked sequence below. Each item is an isolated
candidate, not an adopted model component. Architecture, execution backend,
optimizer, and training protocol are never changed in the same comparison.
Only independently passing candidates may enter a final integration run.
Every later stochastic training comparison uses at least three matched seeds
per arm and reports a predeclared 95% confidence interval; a point estimate
alone cannot promote a candidate.

This page records a research decision. It does not claim that any candidate is
implemented, fast, accurate, novel, or validated in this repository.

## Evidence boundary

The external evidence is primary literature, versioned framework
documentation, and pinned official source repositories linked inline.
Published scale results establish prior art and plausible mechanisms, not
compatibility with this model or this MacBook.

The `trainingnovel` evidence is a separate local artifact class. The audited
tree is `/Users/dttdrv/Projects/Laplace/trainingnovel`, occupies approximately
`4.0 GiB`, and is entirely untracked in the Laplace checkout. Its recorded
runtime is eager MLX `0.29.3` under Python `3.9.6`; no `mx.compile` use appears
in its Python source. These facts are local observations, not literature
claims, and the untracked tree has no immutable git provenance. The scoped
FLOP ledger is
`results/tmat_byte_lm_16m_flops.json`, SHA-256
`dc9defdfbc822c8461d2f4e36d1f8df44e7ac0a0946c0ff6121d0560c4c78364`.
The negative replacement record is
`results/tmat_byte_lm_16m_replacement_negative_seed0_attempt3.json`, SHA-256
`a467d6f17b2b6d4835b3a9cd1e7e3933869f5b941634c2308d32876d5c263e9d`.
That record explicitly says its checkpoint artifacts were not retained, so its
reported endpoint cannot be reloaded from the paths it names.

The reviewed paper versions are CoFrGeNet `2601.21766v4`,
Retrieval-Aware Distillation `2602.11374v1`, HOLA `2607.02303v1`,
Gated Attention `2505.06708v1`, HashAttention `2412.14468v2`, MoBA
`2502.13189v1`, SeerAttention `2410.13276v4`, Gated Delta Networks
`2412.06464v3`, Kimi Linear `2510.26692v2`, DeltaProduct `2502.10297v7`,
Native Sparse Attention `2502.11089v2`, and The Road Less Scheduled
`2405.15682v4`. The reviewed MLX source is release `v0.32.0`, tag commit
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`. The reviewed Schedule-Free
source is commit `d24878d3489bf8ede6148eb6390c9b272b9d93c4`. A later
preregistration must either retain these pins or record and review replacements.

## Ranked candidates

### 0. Preserve the frozen base

- **Fit:** The existing run card already isolates routed selected-set recall
  from recurrent world tracking and supplies the causal controls needed to
  decide whether the composition works.
- **Controls:** Use the fixed dense, local, random-route, source-excluded,
  target-forced, recurrent-knockout, carry-reset, and carry-shuffle arms.
- **Telemetry:** Preserve raw and effective routes, source hits, gradient
  classifications, recurrent gates, state statistics, resource samples, and
  exact artifact hashes.
- **Cost:** The fixed zero-cost M5 Pro pilot and claim package already define
  the budget.
- **Kill condition:** Any source-integrity, parity, firewall, accounting,
  resource, or role gate failure stops the base run unchanged.
- **Non-claim:** A base pass would establish only the two local task roles. It
  would not validate any later method on this page.

### 1. Apply exact-math M5 Pro optimizations

The first speed stage changes execution shape while preserving the frozen
operator, parameter values, masks, routes, losses, outputs, gradients, and
checkpoint meaning.

Candidate order:

1. Vectorize the public Monodratic Python loops. Require complete selected
   index, packed-mask, output, loss, and gradient parity. Drop the rewrite if
   the complete-package statistic below improves by less than `20%`.
2. Replace any explicit triangular inverse with
   [`torch.linalg.solve_triangular`](https://docs.pytorch.org/docs/2.8/generated/torch.linalg.solve_triangular.html).
   Drop the change if parity fails or the complete-package statistic improves
   by less than `10%`.
3. Fuse the query, key, value, write-strength, decay, and output projections
   where their inputs and semantics permit one concatenated projection. Test
   the two upper projections of the deployed gated feature block as a separate
   change. Require forward, gradient, optimizer-membership, reload, and
   checkpoint-conversion parity. Drop each fusion below a `10%` complete-
   package gain.
4. Sweep only process and thread topology using the
   [PyTorch 2.8 multiprocessing guidance](https://docs.pytorch.org/docs/2.8/notes/multiprocessing.html).
   Record the Apple
   [`ProcessInfo.thermalState`](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.property)
   beside wall time and retain a topology only when it improves the repeated
   complete-package statistic by at least `5%` without worse thermal
   throttling.

- **Fit:** These changes target the observed Python and small-operation costs
  without adding a mechanism or changing the scientific comparison.
- **Controls:** Compare each change separately with the frozen PyTorch CPU
  implementation on identical canonical batches, weights, routes, and
  checkpoints.
- **Telemetry:** Record cold and warmed latency, step time, tokens per second,
  peak resident memory, thermal state, process topology, output error,
  gradient error, and checkpoint hashes. The adoption statistic for every
  candidate is the projected complete-package wall time computed from the
  frozen `A`, `S`, `D`, and `H` workloads, never a microkernel or affected-
  operation latency. Run five alternating baseline/candidate blocks of ten
  warmed updates per workload after ten common warmups. For timing block `b`
  and arm `m`, let `tA_m_b`, `tS_m_b`, `tD_m_b`, and `tH_m_b` be the arithmetic
  mean seconds per complete update. Define `R_m_b = 1024*tA_m_b + 1280*tS_m_b
  + 1536*tD_m_b` and `P_m_b = 1.35*max(3*R_m_b, 2*R_m_b + 1536*tH_m_b) +
  2700`. The paired gain is `G_b = 1 - P_candidate_b/P_baseline_b`.
  Enumerate all `5^5 = 3125` ordered length-five resamples with replacement of
  the paired `G_b` values, take the median within each resample, sort those
  medians, and use zero-based elements `78` and `3046` as the deterministic 95%
  interval. A candidate's stated gain floor must be met by the lower endpoint.
  For MLX, the same construction is run separately against PyTorch CPU and
  eager MLX; the lower endpoints must be at least `1 - 1/1.5` and `1 - 1/1.25`,
  respectively. Every measured block must begin at the same recorded Apple
  thermal state. Microbenchmarks are diagnostic only.
- **Cost:** Each candidate is a bounded complete-workload benchmark plus parity
  test. Microbenchmarks may diagnose a bottleneck but cannot retain a change.
  No new dependency or training run is justified.
- **Kill condition:** In addition to the gain floors above, any hidden device
  fallback, graph-dependent result, changed mask, changed route, or changed
  numerical boundary kills the candidate.
- **Non-claim:** A faster equivalent execution path is an engineering result,
  not an architecture or training-method improvement.

### 2. Test only the CoFrGe feed-forward candidate

The [CoFrGeNet paper, version 4](https://arxiv.org/abs/2601.21766v4) supports a later test of
its feed-forward replacement in the token-local feature slot. Its separate
attention replacement is outside scope. The feature candidate fits the
stateless `[batch, time, width] -> delta` seam, but the paper's figure,
collapsed equation, and Table 1 leave an unresolved `3p^2` versus `2p^2`
learned-value discrepancy, and no official implementation was found.

- **Fit:** The candidate changes only token-local feature transformation and
  leaves routing, selected attention, cache behavior, recurrence, block order,
  host normalization, and residual ownership fixed.
- **Controls:** Start with float64 literal-fraction and continuant references;
  compare automatic differentiation, the published custom rule, and finite
  differences. Freeze `epsilon = 0.01` and the dyadic depth-release schedule.
  Training arms are the deployed gated feature block, a parameter-matched
  gated block, the explicit candidate interpretation, direct-linear-only,
  fractional-path knockout, and schedule-disabled diagnostics.
- **Telemetry:** Record learned and optimizer-state values, operation and
  division counts, active depth, per-depth gradients, minimum denominator,
  clamp and evaluation-clip rates, direct and fractional output norms,
  held-out loss, role knockouts, wall time, and peak memory.
- **Cost:** A float64 mathematical gate precedes one matched local run with at
  least three seeds and paired 95% intervals. Paid compute and a framework
  rewrite have no justified value.
- **Kill condition:** Stop for an implicit parameter-sharing choice, forward
  or gradient mismatch, nonfinite values, routine clamping, an inactive
  fractional path, lost recall or world-state roles, or no quality-efficiency
  advantage over the parameter-matched control.
- **Non-claim:** The mechanism is published prior art. A local positive would
  establish only a measured feature-slot tradeoff, not project novelty or
  paper-faithful reproduction of the unresolved specification.

### 3. Trim only after causal deletion evidence

[Retrieval-Aware Distillation for Transformer-SSM Hybrids, version 1](https://arxiv.org/abs/2602.11374v1)
supports using ablation to locate retrieval-critical computation before
removal. Its reported preservation of `2%` of attention heads and an `8x`
recurrent-state reduction are results for a different teacher, scale, task,
and training method; they are not targets for this model.

- **Fit:** The first and only candidate universe is the six recurrent sequence-
  mixer deltas at host blocks `1`, `2`, `3`, `5`, `6`, and `7`. A unit deletion
  runs that mixer completely and zeros only its exposed delta. Rank all six by
  the smallest mean absolute degradation across both frozen role metrics;
  ties select the lower block index. Use base seeds `11` and `23` as discovery
  data, freeze one selected block, then confirm it without reselection on
  untouched seeds `37`, `53`, and `71`. No routed head, feature branch,
  channel, parameter, or second deletion belongs to this candidate universe.
  Only a confirmation pass may authorize a separately preregistered smaller
  recurrent width.
- **Controls:** On discovery and confirmation data use intact, the selected
  deletion, one deterministic fixed comparator, and one generator-frozen
  random comparator. The fixed comparator is block `1` unless block `1` is the
  selected deletion, in which case it is block `2`. Remove the selected and
  fixed IDs from the ordered universe `[1, 2, 3, 5, 6, 7]`, seed a dedicated
  CPU Torch generator with `720260319`, draw exactly one
  `torch.randperm(len(remaining), generator=g)`, and use its first index to
  choose the random comparator. Thus all three deletion arms are distinct. A
  later retrained deletion comparison uses at least three new
  matched seeds per arm, byte-identical starts, the same tokens, optimizer, and
  stopping rule, plus a parameter-matched smaller-width control. Report 95%
  confidence intervals for every cross-seed difference.
- **Telemetry:** Record per-unit knockout deltas, ranking stability across
  seeds, route/source behavior, recurrent state and gate statistics, task
  quality, learned values, memory, and wall time.
- **Cost:** Evaluation-only discovery and untouched confirmation come first.
  At most one confirmed deletion and then one width candidate receive matched
  three-seed local training.
- **Kill condition:** Stop unless the same block is least harmful on each of
  discovery seeds `11` and `23`. On confirmation, stop if either frozen role
  gate regresses or if the selected deletion fails to beat both random and
  fixed deletions by the predeclared paired interval. Also stop when a later
  smaller model yields no measured memory or wall-time benefit.
- **Non-claim:** A local deletion does not reproduce the paper's distillation
  result or imply that its sparsity ratios transfer.

### 4. Audit the public router before replacing it

Static inspection raises a risk that public codebook values are disconnected
from the task gradient. The base run must settle this with parameter-level
evidence before any fallback is considered.

- **Fit:** Audit the deployed router's gradient connectivity, code occupancy,
  collisions, underfill, probe accuracy, and required-source hit rate without
  changing its route or loss.
- **Controls:** Preserve learned, target-forced, random, source-excluded,
  local-only, all-eligible, and dense controls. Include a deliberately frozen
  router parameter audit so absence of gradient cannot be mistaken for an
  instrumentation failure.
- **Telemetry:** Record every router parameter's optimizer membership and
  gradient state, code usage and collision distributions, raw and effective
  routes, overflow, underfill, probe accuracy, and conditional answer accuracy.
- **Cost:** The audit belongs inside the existing base evidence and adds no
  model or training stage.
- **Kill condition:** A disconnected value, hidden fallback, missing route
  record, or task success without the required routed source blocks a learned
  routing claim.
- **Non-claim:** Diagnosing a failed public router does not validate another
  sparse-attention method.

No fallback router is authorized by this review. [MoBA version
1](https://arxiv.org/abs/2502.13189v1), [SeerAttention version
4](https://arxiv.org/abs/2410.13276v4), and [HashAttention version
2](https://arxiv.org/abs/2412.14468v2) remain related work only. If the base
audit localizes a specific failure class, a new review must state that
observation, select one simplest matched-cost follow-up before any result is
seen, and freeze its controls and kill condition. No alternative may enter the
base result retroactively or be reported as validation of Monodratic.

### 5. Compile an MLX port only after CPU proof

[MLX release `v0.32.0`](https://github.com/ml-explore/mlx/releases/tag/v0.32.0)
at tag commit `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` is the pinned native
Apple-silicon backend candidate. Its [versioned compilation
contract](https://github.com/ml-explore/mlx/blob/v0.32.0/docs/src/usage/compile.rst)
requires pure compiled functions and explicit state. The current local source
uses eager MLX `0.29.3`; a compiled port is a new backend experiment, not an
optimization already present. Use a separate `v0.32.0` environment rather than
mutating the evidence environment.

- **Fit:** Port only the already proved operator graph and preserve the same
  parameter, route, mask, reset, loss, and checkpoint contracts. The public
  data-dependent route projection, Python index construction, scalar
  extraction, search, route intervention, and internal route-loss assembly
  remain eager in the first port. They materialize one immutable route tensor
  through an explicitly timed `mx.eval` boundary. Only pure fixed-shape tensor
  subgraphs after that boundary may be compiled. Each compiled function takes
  parameters, buffers, batch tensors, route tensors, and any random key as
  explicit inputs and returns all updated values; it mutates no module,
  optimizer, Python container, or global RNG. Optimizer update and checkpoint
  assembly remain eager. A graph-compatible router rewrite is a different,
  unauthorized experiment.
- **Controls:** Compare frozen PyTorch CPU, eager MLX, and compiled MLX on
  identical canonical tensors and matched training workloads.
- **Telemetry:** Record eager route/index time, every synchronization, compile
  time, compile count, recompilations, graph breaks, fallback operations, cold
  and warmed end-to-end step time, outputs, gradients, updates, explicit state
  trees, checkpoint conversion, peak unified memory, and thermal state.
- **Cost:** One reference port and one compiled path in an isolated local
  environment. No MLX architecture rewrite or sibling-tree mutation is
  justified.
- **Kill condition:** Drop the backend unless the complete training workload is
  at least `1.5x` faster than the proved PyTorch CPU reference and at least
  `1.25x` faster than eager MLX by the same paired 95% interval rule used for
  exact-math optimizations. Also drop it if parity fails, synchronization cost
  is omitted, fallback is hidden, state is implicit, or a fixed input shape
  causes repeated recompilation.
- **Non-claim:** Backend speed does not validate architecture quality, and an
  eager-versus-compiled result does not validate the local specialist protocol.

### 6. Consider one selected-attention output gate

[Gated Attention for Large Language Models, version 1](https://arxiv.org/abs/2505.06708v1)
supports a cheap later hypothesis: multiply each selected-attention head output
by a headwise sigmoid derived from the block's pre-normalized hidden state.
This is lower risk than replacing attention, but it still changes the
architecture and therefore comes after the base and feature tests.

- **Fit:** At each routed block, the source-form gate is
  `Y * sigmoid(XW)`: `Y` is the per-head selected-attention output after value
  aggregation and before head concatenation and the existing output
  projection, `X` is the same full-width pre-normalized hidden state supplied
  to that attention block, and the bias-free `W` maps `[B,T,D]` to `[B,T,H]`.
  This bias-free expression is the source-faithful arm. The primary local
  candidate is instead a paper-positioned near-identity adaptation that uses
  `Y * sigmoid(XW + b)`, initializes `W` to zero, and initializes every head
  bias to `log(99)`. Its initial mathematical multiplier is therefore `0.99`,
  subject only to the frozen floating-point representation. The added bias and
  special initialization are local choices not specified by the cited paper,
  so this adaptation is not called paper-faithful. Neither arm owns a route,
  cache, recurrent state, or sparsity loss.
- **Controls:** Compare the unchanged fixed-one path, parameter-free SiLU of
  the head output, the source-faithful bias-free `Y * sigmoid(XW)` arm, the
  paper-positioned local near-identity `Y * sigmoid(XW + b)` adaptation, an
  input-independent learned sigmoid scalar per head, a parameter-matched
  hidden-dependent affine multiplier without sigmoid, a parameter-matched
  non-sparse multiplier `0.5 + 0.5 * sigmoid(logit)`, and a frozen-permutation
  wrong-hidden sigmoid. All learned arms use matched seeds, tokens, optimizer
  budget, and initialization scale. The source-form arm has no bias inventory;
  that unavoidable parameter and initialization difference is reported rather
  than called matched.
- **Telemetry:** Record gate values and gradients by head and task phase,
  fractions below `0.1` and above `0.9`, effective-zero output fraction,
  saturation, output norms, source hits, held-out loss, and both role-knockout
  signatures.
- **Cost:** One small headwise projection and one isolated three-seed local
  comparison with paired 95% intervals.
- **Kill condition:** Stop for saturation, inactive gates, no multi-seed gain,
  lost route necessity, lost recurrent necessity, or no quality gain at honest
  parameter and wall-time cost.
- **Non-claim:** A positive source-form arm would not reproduce the paper's
  large-model scaling or attention-sink findings. A positive local
  near-identity adaptation would additionally not establish a paper-faithful
  reproduction because its bias and special initialization are local changes.

### 7. Run one optimizer A/B, not an optimizer search

The first optimizer candidate is Schedule-Free AdamW from
[The Road Less Scheduled, version 4](https://arxiv.org/abs/2405.15682v4),
using the official implementation pinned at
[commit `d24878d3489bf8ede6148eb6390c9b272b9d93c4`](https://github.com/facebookresearch/schedule_free/tree/d24878d3489bf8ede6148eb6390c9b272b9d93c4).
It is compared only with the frozen AdamW schedule. Cautious AdamW remains
unauthorized related work; a negative or nonfinite Schedule-Free result is
preserved and does not open a fallback arm.

- **Fit:** The candidate targets time to the already frozen gates without
  changing the model, backend, data, token budget, or evaluation.
- **Controls:** Use at least three matched seeds per arm from byte-identical
  initialization and data order; match update and token budgets, weight-decay
  classification, clipping, precision, evaluation cadence, and checkpoint
  cadence. Predeclare equal tuning effort and report paired 95% intervals.
  The Schedule-Free arm calls `optimizer.train()` whenever `model.train()` is
  entered and before the next gradient update. Before every evaluation or
  checkpoint it calls `optimizer.eval()` and then `model.eval()`, so metrics
  and persisted model values use the evaluation sequence. A checkpoint stores
  those evaluation-mode model values, the complete optimizer state, and an
  explicit `optimizer_mode: "eval"`. Strict resume reloads both, requires the
  saved optimizer groups to be in evaluation mode, verifies endpoint logits,
  and calls `optimizer.train()` before the next update. Transition, evaluation,
  checkpoint, and resume time is charged to the arm.
- **Telemetry:** Record time and tokens to each gate, loss and accuracy curves,
  gradient and update norms, nonfinite events, step time, memory, and final role
  results.
- **Cost:** One matched local A/B. No multi-optimizer sweep is authorized.
- **Kill condition:** Reject the candidate unless the lower bound of the paired
  95% interval shows at least `15%` less wall time to the complete gate set on
  all three seeds, with no final-quality or role-gate regression. Also reject a
  win that depends on unequal tuning, budget, evaluation sequence, or omitted
  transition cost.
- **Non-claim:** Lower step time alone is not faster training; the comparison is
  time to matched scientific gates.

### 8. Change recurrence only after a localized failure

The existing recurrence is already within the prior-art boundary of
[Gated Delta Networks, version 3](https://arxiv.org/abs/2412.06464v3). If the base passes,
it remains unchanged. If the base fails and the causal evidence localizes the
failure to recurrent state rather than routing, task construction, or output
use, test state width first. Only a width failure can justify an
expression-level local test of the finer-grained update in
[Kimi Linear, version 2](https://arxiv.org/abs/2510.26692v2); GPU-specific kernels are not
part of that test.

- **Fit:** Width directly tests finite-state capacity while preserving the
  recurrence family. The later expression-level candidate tests a finer update
  only if width cannot repair a diagnosed state failure.
- **Controls:** Compare the frozen recurrence, one smaller and one larger width,
  parameter-matched controls, reset and knockout arms, and the unchanged routed
  path with at least three matched seeds per arm and paired 95% intervals.
- **Telemetry:** Record carry-task accuracy, state and gate norms, retention,
  reset effects, numerical and gradient parity, parameters, operations,
  memory, and wall time.
- **Cost:** One bounded three-seed width comparison follows a diagnosed
  failure. A new recurrent expression requires its own dossier, run card, and
  separate three-seed comparison.
- **Kill condition:** Stop when width does not causally recover the failed role,
  when gains vanish against parameter matching, or when the Mac expression is
  not numerically stable and competitive without specialized GPU code.
- **Non-claim:** Neither the current recurrence nor a later finer-grained update
  is project novelty, and paper-scale kernel results do not transfer to M5 Pro.

### 9. Adopt only HOLA's surprise telemetry

[HOLA, version 1](https://arxiv.org/abs/2607.02303v1) ranks exact-cache writes
with a pre-write residual score. The project already assigns exact recall to
routed selected attention, so adding HOLA's exact cache would duplicate a role
and confound the base. The only current adoption candidate is a detached,
pre-clamp write-surprise diagnostic.

- **Fit:** In a detached token-sequential shadow of each recurrent layer, let
  `S_prev` be the state after any preceding reset and after the previous global-
  chunk clamp. Compute `e_t = v_t - g_t * (k_t^T S_prev)` before the token's
  state update and before any following clamp, then compute the headwise score
  `beta_t * ||e_t||_2`. This is a pre-clamp write-surprise proxy, not the norm
  of the final clamped state change. Reduce it online to count, mean, variance,
  extrema, and a fixed-size top-token heap; persist no full residual tensor.
  All shadow inputs are detached, the shadow owns no model state, and it cannot
  change writes, reads, routing, loss, outputs, or gradients.
- **Controls:** Compare the score with shuffled, delayed, wrong-source,
  recency-only, and random ranks on the same tokens and seeds.
- **Telemetry:** Record headwise and tokenwise score summaries, top-ranked token
  identity, enrichment for later-required events, correlation with the
  unclamped outer-product write norm, clamp incidence, downstream error, and
  predictiveness across at least three seeds.
- **Cost:** The sequential shadow can be material work even though its persisted
  result is scalar telemetry. Benchmark complete warmed steps with logging off
  and on, including reduction and synchronization.
- **Kill condition:** Output and gradient parity must be exact, and logging
  overhead must stay below `5%` by a paired 95% interval. Do not create a
  mechanism dossier unless the score also predicts later-required information
  beyond shuffled and recency controls on all three seeds.
- **Non-claim:** The telemetry is not a cache, hippocampal system, memory
  improvement, or reproduction of HOLA.

## Deferred local specialist protocol

The local `trainingnovel` tree is evidence for a future release-lifecycle
experiment, not evidence for faster training of the first combined model.
Its scoped planned-experiment ledger estimates one causal-proof lineage at
`214.1985116928` trillion linear-equivalent FLOPs versus `30.5362833408`
trillion for its end-to-end reference, or `7.014557380878309x`. Bridge plus
bounded repair is estimated at `13.510672267334006%` of that reference, but
the `7.0146x` scoped lineage is the relevant reported lower bound. It excludes
failed canonical-only surrogate attempts, provisional docking, calibration,
evaluation and checkpoint I/O, and failed witness objectives, so it is not a
complete lifecycle cost.

The strongest local replacement attempt is negative at matched update length:
the module-only path reaches `0.76953125` replacement accuracy versus
`0.9375` for end-to-end training. It updates `32.08%` of the trainable values
and uses an estimated `54.93%` of update FLOPs, but the quality mismatch blocks
an efficiency claim. The record says its checkpoint artifacts were not
retained, so the reported endpoints cannot be reloaded. The seeded Karkasov
projection remains a standalone scaffold and is not wired into the 16M
language model.

- **Fit:** Reconsider only when the project has at least three real, versioned
  releases that share one frozen interface. Then train immutable specialists,
  dock through identity-initialized bridges behind zero scalar gates, and
  verify frozen hashes before and after docking.
- **Controls:** For each of the three releases, compare ordinary end-to-end,
  module-only, bridge-only docking, and failed attempts with at least three
  matched seeds per arm. Charge every anchor, specialist, repair, optimizer
  state, artifact, operation, memory sample, and wall-clock second and report
  paired 95% intervals for capability and cumulative cost.
- **Telemetry:** Record immutable hashes, trainable membership, gate paths,
  capability retention, specialist quality, per-release and cumulative cost,
  peak memory, and wall time.
- **Cost:** The existing scoped `7.0146x` lower-bound estimate makes this a
  lifecycle experiment, not a first-run speed path. It remains deferred until
  three releases make cumulative reuse measurable.
- **Kill condition:** Reject the protocol for any frozen-hash drift, unmatched
  quality, hidden repair, or less than `1.2x` cumulative lifecycle advantage at
  matched capability after the third release.
- **Non-claim:** Local assembly witnesses, bridge integrity, and lower
  module-update FLOPs do not establish replacement quality, generality, or
  training speed.

## Explicit rejects for this program

- Reject the CoFrGe attention replacement; only its feed-forward candidate
  fits the isolated feature slot.
- Reject the HOLA exact cache; routed selected attention already owns the exact
  recall hypothesis.
- Reject the bundled
  [Native Sparse Attention, version 2](https://arxiv.org/abs/2502.11089v2) architecture and
  hardware path; it changes routing, attention, compression, and execution at
  once.
- Reject [DeltaProduct, version 7](https://arxiv.org/abs/2502.10297v7) unless a future
  diagnosed state-tracking failure survives the simpler recurrence-width gate.
- Reject CUDA, Triton, FlashAttention, Flash Linear Attention, and FP8 paths for
  the M5 Pro proof. They do not supply the required local execution boundary.
- Reject Muon, SOAP, and GaLore sweeps. The only optimizer experiment is one
  preregistered A/B.
- Reject hidden MPS fallback, a whole-repository Laplace or `trainingnovel`
  import, and an unprofiled Accelerate rewrite.
- Reject every simultaneous architecture, backend, optimizer, and specialist-
  protocol bundle. Such a result cannot identify which mechanism helped.

## Promotion boundary

Every later candidate requires its own concise mechanism contract, matched
controls, telemetry, cost ledger, kill condition, reviewed run card, zero-
finding review, assertions, and local resource pilot. A negative result is
preserved and excluded from integration. The final model may contain only the
unchanged base plus candidates that independently passed their own proof
boundaries.

## See also

- [[PROJECT_PLAN]]
- [[modular_neural_model_stack]]
- [[tests/modular_sequence_role_cpu_run]]
- [[neural_model_dossier_nested_reciprocal_feature_mixer]]
