# Modular neural model stack

status: current (as of 2026-07-26).

## Decision

Execution is stopped. Deyan stopped the 2026-07-26 session before attempt 4.
No attempt-4 run root or claim training exists. The final clean-claim contract
synchronization was interrupted with a payload-to-constant digest mismatch, so
the current bytes are unverified and the active attestations are stale. Do not
launch. The exact recovery boundary is
`docs/MLX_SESSION_STOP_HANDOFF_2026-07-26.md`.

Todorov owns the model identity and proof standard. Transformerov supplies the
latest tested host and recurrent world-state path. Monodratic supplies the
routed selected-set attention candidate that replaces an attention mixer and
must still prove exact recall after routing. The nested-reciprocal feature
mixer is an optional Monodratic feed-forward addition. Karkasov supplies only
deferred evidence for training and docking separately released specialists;
the current local audit rejects it as the trainer for the first model. Laplace
is outside this architecture. [[modular_neural_model_methods_review_2026_07]]
defines the ranked later optimization and intervention lane.

These sources are evidence, not packages to merge wholesale. Each imported
mechanism keeps its own validation boundary until the combined model passes
matched tests.

## Cost-before-build decision

The host, recurrent update, public routed mixer, reciprocal feature candidate,
and specialist-training scaffold already exist. The immediate deliverable is
still the minimal combined base implementation and its two-rung local proof
package, not a new host, router, registry, training framework, or
whole-repository merge. The lowest-risk path ports only the frozen source
boundary, reuses the public routing operation, and implements the reviewed task
firewall and controls in [[tests/modular_sequence_role_cpu_run]]. Exact-math
training through the compiled MLX backend is the frozen base execution
substrate, with Torch CPU as checkpoint, evaluation, gate, and artifact
authority. Reciprocal feature mixing, causal deletion, further exact-math
execution changes, and an optimizer comparison follow as separate local
interventions because simultaneous activation would erase causal attribution.
The audited `trainingnovel` lifecycle is deferred rather than activated.

The fixed package charges `45,613,056` token positions and has zero
external-compute cost. Its resource pilot is a run-or-stop gate, not a tuning
stage. No architectural novelty is claimed: each mechanism is prior work or
prior project work. The possible new result is narrower and empirical—whether
the fixed composition makes routed recall and recurrent state causally
necessary under matched controls. Paid compute has no justified expected value
before that result and remains unauthorized.

## Smallest architecture

The residual stream has shape `[batch, time, width]`. A host block owns
normalization, residual addition, and module order. A mixer consumes a
normalized residual stream and returns a same-shape delta. No mixer may add its
own hidden fallback path.

The host uses Transformerov's zero-indexed four-block cycle. Block `i` uses the
public routed mixer when `i mod 4 = 0`; the other three blocks use the recurrent
mixer. Depth must be a positive multiple of four. The first paired witness uses
eight blocks, because the routed block at index 4 is the first routed block that
can consume representations already transformed by recurrent blocks.

For block input `x`, the equations are fixed:

`u = x + sequence_mixer(sequence_norm(x))`

`next = u + feature_mixer(feature_norm(u))`

The resulting architecture is:

1. A routed selected-set attention layer uses the Monodratic public mixer in
   the existing attention-mixer slot and remains a candidate for exact recall
   until the route selects the required source.
2. A recurrent world-state layer performs the Transformerov bounded
   within-sequence update in its existing recurrent-mixer slot and returns a
   same-shape delta. The tested host exposes no external carry-state interface.
3. Every layer keeps one token-local feature-mixing slot after its sequence
   mixer.
4. The nested-reciprocal feature mixer may replace only that token-local slot.
5. The host owns embeddings, normalization, residual equations, output
   normalization, and the language head.

The first witness retains Transformerov's bias-free, four-times-width,
two-branch gated feature block, zero dropout, and untied embedding and output
matrices. Its feature operation is
`down((up_a(x) * sigmoid(up_a(x))) * up_b(x))`, with all three projections
bias-free and the two upper projections four times wider than the residual
stream. It uses source-native initialization. Transformerov's language-model
forward does not call `nextlat`, although its scale trainer invokes that module
as a separate objective. The target host excludes the objective and module from
the graph, optimizer, and parameter ledger because neither belongs to this
sequence-role hypothesis.

The recurrent mixer retains Transformerov's bias-bearing gate projections,
per-head output normalization, and source state-norm clamp. The host retains
its final normalization. Optimizer accounting separates matrix weights,
one-dimensional bias values, normalization scales, and route codebooks; the
bias-free feature projections do not imply that the recurrent gate projections
are bias-free.

Ordinary dense attention is a named scientific control, not a hidden production
fallback. Four surfaces remain distinct. The all-eligible donor is trained
task-only through the public selected-set implementation. The all-eligible clone
loads the jointly trained routed checkpoint and changes only the selection
limit. A separately trained dense baseline uses the full causal prefix and the
same host geometry. It receives the matched `1024 + 512` task-update schedule,
examples, shared-parameter optimizer settings, and evaluation budget frozen in
the run card, while its missing route parameters are reported rather than
disguised. The public dense-selected-mask calculation is
only a numerical oracle for the same selected IDs. It is an implementation
parity check, not a scientific arm.

The routed mixer remains stateless and rebuilds its route from the supplied
sequence. The recurrent mixer alone owns within-sequence state updates. The
feature mixer is token-local and owns neither recall nor recurrent state.
Externally carried decode state remains a future contract that requires
separate proof.

## Module contract

The minimum target contract for the Transformerov and Monodratic seam is
deliberately narrow. It is not the current legacy Todorov `src` interface,
whose modules return state and telemetry alongside their deltas:

- Input and output are finite `[batch, time, width]` tensors with the same dtype
  and device.
- Each module returns a delta; the host performs the residual addition.
- The first local proof uses float32 without automatic mixed precision. The MLX
  child trains through compiled Metal kernels, while Torch CPU remains the
  reference and evaluation authority. Token IDs and routing indices use each
  framework's 64-bit integer dtype. Each runtime keeps inputs, parameters, and
  transient state on its declared device.
- Width is divisible by each head count, the routed attention head width is
  even, routing width is divisible by its subspace count, and sequence length
  is exactly divisible by recurrent chunk length. The runner rejects invalid
  geometry; it never crops or pads implicitly.
- State ownership is explicit. The routed and feature mixers have no persistent
  state. The tested recurrent mixer updates state only within the supplied
  sequence and returns a delta, not external carried state. Any future
  decode-state interface requires a separate contract and proof.
- Disabling one module cannot silently activate another.
- Configuration, parameters, optimizer groups, serialization, and telemetry
  identify each module separately.
- Every release records configuration, seed, parameter count, and complete
  file hashes, plus a source revision when one exists.

This target contract is sufficient for replacement tests and zero-gate
docking. A registry, plugin system, general backend interface, or new
dependency is not justified.

## First CPU prototype method

### Frozen geometry

The first witness combines Karkasov's symbolic layout with its learned-routing
artifact's two-slot budget: sequence length 128, eight-token route blocks, ten
early candidate blocks, four randomly chosen rule-bearing blocks, a cue at
position 80, and a generic query at position 126. Host block 0 is local-only.
Every recurrent layer resets immediately before positions `8`, `16`, `24`,
`32`, `40`, `48`, `56`, `64`, `72`, and `80`, so no candidate-block state can
carry a rule mapping into the cue region. Host block 4 constructs and searches
the raw public route normally, but its effective remote IDs are `-1` outside
position `126`. This closes both direct recurrent rule carry and early remote
retrieval before the query.

At position `126`, the public prefix index has 15 remotely causal blocks: four
rule-bearing blocks and eleven non-rule blocks. The learned route has two slots
but may contain `-1` when public search underfills. Underfill is recorded and
never repaired by a dense, local, repeated-ID, canonical-ID, or other query
fallback. The model uses width 64, eight host blocks, four routed-attention
heads, four recurrent heads of width 16, and recurrent chunks of 32 tokens. The
boundary at position 96 lies between cue and query and is observable by the
state interventions.

The public router uses routing width 16, two subspaces, four codes per
subspace, four probes, bucket capacity 64, query chunk length 128, and one local
block. These are first-run constants, not a general configuration framework. A
change requires a separately reviewed hypothesis and repetition of every
control; the frozen claim run cannot change them.

### State and routing lifecycle

Each recurrent layer creates zero state for every forward pass and every batch.
Rung one uses a reset-aware implementation that preserves the frozen recurrent
equations and state-norm clamp between the fixed firewall boundaries. It must
match an independent reset-aware sequential reference and reproduce the frozen
chunkwise path when no reset mask is present. Rung two uses the unchanged
Transformerov chunkwise path with no firewall reset. State is not returned,
cached, serialized, or shared between layers or examples. The explicit
auxiliary audit path may return detached gate tensors and per-boundary state
norms for the current call. It creates no persistent telemetry scalar or
carried state.

Each routed layer rebuilds its packed route index from the current sequence and
discards the index after the call. Rung one records both the raw public route
and the effective query-only route. The ordinary model path returns only the
delta. A training or audit path may additionally expose routing loss and route
telemetry without changing residual ownership.

The trainer owns the routing curriculum and its weight. It first trains an
all-eligible donor on the task, copies every non-router value into the selected-
set model, trains only router parameters against required-source labels, and
then jointly trains the model with
`task_loss + 0.1 * (internal_router_loss + supervised_route_loss)`. This is a
reproduction of the public experiment's three stages, not a new training
method. In the combined host, the two route losses apply only to remote-enabled
block 4; block 0 stays local-only and does not request routing auxiliaries in
the donor, router-only, and joint stages. Its output-inactive router parameters
are excluded from the optimizer and reported in the inactive ledger rather than
counted as learned capacity.

The dense control receives a task-only stage matched to the donor's 1,024
updates and a second 512-update task-only stage matched to the routed model's
joint examples and shared-parameter learning rate. The complete two-rung
package charges `45,613,056` token positions. Exact schedules, generators,
optimizer groups, operation accounting, and resource limits live only in the
canonical run card.

Static inspection of the public route construction shows no differentiable
path from these losses through the discrete search to the product codebooks.
The first run must record `None`, zero, or nonzero gradient status for every
router parameter and report serialized codebook values separately. It must not
call those codebooks learned unless the gradient and update checks establish
that claim.

### Intervention semantics

Every causal control has one narrow meaning:

- Routed knockout runs block 4's routed mixer and then replaces only that
  block's returned sequence delta with zero. Block 0 remains intact.
- Local-only routing additionally sets block 4's number of selected remote
  blocks to zero; block 0 is already local-only.
- Target-forced routing changes only block 4 and uses the public one-block
  forced-input path with shape `[batch, time]`, 64-bit integer dtype, and `-1`
  as the inactive sentinel. It guarantees required-source inclusion in the
  two-slot tensor but preserves any remaining public underfill.
- Matched random routing changes only block 4. At the query it samples two of
  the 15 valid remote blocks uniformly without replacement, giving a target-
  inclusion prior of `2 / 15`; every other remote position is `-1`. The
  diagnostic override has shape `[batch, time, 1, 2]` and 64-bit integer dtype.
  Local blocks remain internally supplied. The run reports the realized
  target-inclusion rate. The override does not enter the production forward
  path.
- Required-source-excluded routing preserves every learned non-source slot in
  order, removes only the required source, and fills every vacancy from the
  frozen per-example non-source permutation. It therefore exposes exactly two
  distinct non-source IDs without replacing an already valid learned
  non-source choice.
- Recurrent knockout runs every recurrent mixer at block indices 1, 2, 3, 5,
  6, and 7, then replaces each returned sequence delta with zero.
- Carry reset replaces the state of every recurrent layer with zero at position
  96. Carry shuffle applies the deterministic cyclic permutation
  `state.roll(1, dims=0)` to every recurrent-layer state at that boundary and
  consumes no RNG.
- The later feature knockout replaces only the token-local feature delta with
  zero. It is not part of the first sequence-role run.

All intact and same-checkpoint intervention arms use the same examples, seeds,
trained values, and evaluation budget. A control may change only the named
intervention. Separately trained donor and dense baselines do not share trained
values and are reported separately.

### Accounting and artifacts

The run ledger separates active learned parameters, serialized values without
an observed gradient, inactive reference parameters, buffers, dynamic state,
route-index storage, temporary workspace, optimizer state, started attempts,
paired completed updates, token positions, peak memory, CPU time, and wall
time. It records the complete configuration, dataset and evaluation seeds,
file hashes, source revisions, per-stage steps, and reload parity. Reload parity covers
model output, selected blocks, and the absence of serialized recurrent state or
route-index state. One total parameter count cannot substitute for this ledger.

### Run-card and execution prerequisite

[[tests/modular_sequence_role_cpu_run]] and its working-tree
`neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json` payload form the
canonical base preregistration under review. They freeze the exact generators, vocabulary,
batches, seeds, stage budgets, optimizer, schedule, clipping, stopping,
interventions, sampling, artifact schema, accounting, and expected local resource
envelope. When a concise statement here differs from their exact execution
detail, repair the canonical plan first and then synchronize the linked
documents before proceeding.

Authorization is ordered. The linked contract documents previously reached a
zero-finding review, after which implementation exposed a self-certifying
review-artifact path. The replacement evidence-lifecycle design amendment and
the corrected batch-local carry-shuffle strata and payload digest have each
reached literal zero findings. The authorized scientific method and nine-file
source-and-test scope are frozen while the implementation bytes undergo complete review and
bug-class correction. The final unchanged
documentation, implementation, test, and complete surfaces must each reach
literal zero findings. Only then may the independent reviewer write one
run-id-free content-addressed attestation for each exact current scope. The
runner must select exactly one matching attestation per scope, verify its
current path-sorted target records and digest, and byte-copy it into the run; it
  may never author or mutate reviewer evidence. Every assertion and the fixed
  zero-cost resource pilot follow that closure. After a proceeding pilot, a
  durable `run/training_start_request.json` begins a bounded review transaction,
  not training. While the live plan stays at its launch bytes and no claim data,
  child stage runtime, training model, or optimizer exists, one exact fifth zero-finding
  attestation must bind the run-specific request and changed plan candidate.
  The runner precommits the request, candidate, review, and linkage, then one
  guarded exact-launch atomic replacement publishes the candidate and solely
  marks training start. Paid compute remains separately blocked.

## Implementation status

The authorized nine-file base implementation now exists as working-tree model
source, Torch CPU authority, compiled MLX engine and backend, qualifier, and
focused CPU and MLX tests. Complete-surface review on 2026-07-20
reopened its evidence schema, and the 2026-07-21 correction cycle also closed
the sampler-abort, process-identity retry, and training-start governance bug
classes without changing the scientific method or file scope. It remains a
scaffold and harness until the amended documentation, complete code-and-test
review, four fresh base review attestations, every pretraining assertion, and
a contract-valid resource pilot close in order. Three governed MLX attempts
have run without a valid pilot result. Attempt `mlx-m5pro-20260726-1` stopped
before pilot update one.
Attempt `mlx-m5pro-20260726-2` reached durable attempted counters of `132`
pilot updates and `292,864` positions, but its final parent resource row
retained an active job and no `run/pilot.json` was published. Attempt
`mlx-m5pro-20260726-3` passed the current preflight and its child executed all
pilot updates, but stale zeroed exited-child telemetry prevented both a clean
terminal resource row and hard-abort finalization. It published neither
`run/pilot.json` nor `ABORTED.json`. No attempt establishes a contract-valid
completed pilot, and no claim training has run. The terminal-sampler and abort-
timeline classes are repaired with deterministic regressions. The latest
complete modular verification reported `834 passed, 2 skipped` in `214.05`
seconds. Literal-zero complete-surface review, four fresh attestations, full
pretraining assertions, and attempt 4 remain ordered gates.
Committed `src/` outside this
working-tree addition remains the historical 6:1:1 hybrid and exposes carried
state, cache, and telemetry through a different interface. Committed `v01/` is
the descent-memory toy. Transformerov-like files under `v01/` remain unrelated
working-tree evidence. The selected Transformerov host carries recurrent state
only between internal chunks of one supplied sequence. The nested-reciprocal
feature mixer has no implementation in the base.

Legacy Todorov carried state and the full Monodratic model's stateful decode
interfaces are excluded from the target until a separate full-sequence versus
carried-state parity contract proves them.

## Evidence-contract amendment

The 2026-07-20 amendment changes evidence representation only. It leaves every
model tensor, task, gate, seed, schedule, loss, token budget, and the then-current
five-file implementation boundary unchanged. The 2026-07-22 execution-substrate
amendment adds the MLX engine, backend mapping, qualifier, and focused MLX test
to make a nine-file reviewed surface without changing the scientific method.

The 2026-07-21 lifecycle correction preserves the same boundary. Abort
finalization may synthesize a zero-counter final resource row only when the
primary cause is `resource_sampler_failure`; every other cause leaves an empty
timeline empty and invokes no sampler. A process-set retry carries typed old,
updated, and observed PID identities and is legal only when both newer sets are
proper subsets of the old set and the updated expected set is a subset of the observed
set. This admits clean exits discovered after a `ps` snapshot without hiding an
unexpected PID or a missing still-live PID.

After each committed pilot or claim sample-zero row, the parent repeats the
pending-signal and frozen-anchor guard immediately before spawn. Pilot and
claim children wait on a shared parent event, and a second guard plus guarded
event release follows spawn before any worker environment or runtime setup.
Signal or frozen-byte drift in either admission window therefore begins no
workload.

The launch plan is copied byte-for-byte into the immutable prepilot base. After
a proceeding pilot and the durable seven-file claim transition,
`run/training_start_request.json` creates the 12th anchor and a strict
1,800-second review-and-publication deadline. It is not a start. A changed
candidate must bind that request digest, and one run-specific zero-finding
`feature-dev:code-reviewer` attestation must target the path-sorted candidate
and request records. The copied candidate, review, and linkage complete 15
persisted proof paths while the live plan remains at launch. Once the linkage
is durable, later failure cannot roll those paths back. During the bounded
publication window, the single-coordinator process is the sole authorized plan
writer.
It blocks handled signals, acquires a nonblocking exclusive advisory lock on
the stable wiki-directory descriptor, rechecks the exact launch bytes and all 15
persisted proof anchors, performs one same-directory atomic replacement, latches
`started`, and fsyncs the same locked directory descriptor before unlock. The
cooperative lock provides no
protection from a noncooperating writer that ignores it; such a process is
outside the fault model. A busy or unsupported lock while the live plan remains
at the unchanged launch bytes is a `reviewed_ready` hard abort before start.
Live bytes matching neither launch nor candidate are preserved, permit zero
further work, and are an unrecoverable ambiguity or orphan with no terminal
result. Required post-replacement directory-fsync failure is also an
unrecoverable orphan. `RESULTS_PARENT`, exactly
`/Users/dttdrv/Projects/Todorov/neuroloc/results/modular_sequence_role_mlx`,
defines a coordinator private namespace for unique initialization staging
siblings, the published final run root, and disposable publication rehearsal
paths from successful exclusive creation or reservation through cleanup or
terminalization. Before activation, only the coordinator may mutate those
paths. After activation, the published final run root may be mutated only by
the coordinator and the validated single MLX child under frozen path assignments.
Arbitrary noncooperating path replacement, arbitrary noncooperating hard link
injection, arbitrary noncooperating symbolic link injection, or arbitrary
noncooperating mutation inside that private namespace is outside the fault
model. In-scope failures remain authorized process failure, storage or system
call failure, and the stated abrupt kill or power loss boundary. This boundary
does not weaken external protections: an external live project plan writer for
`PROJECT_PLAN.md` that ignores the cooperative advisory lock remains outside the fault
model, with exact drift preserved as unrecoverable, while external review
evidence remains outside `RESULTS_PARENT` and must be nonsymlink, content
addressed, and content bound.

An authorized child that crosses the orphan boundary sends an exact orphan IPC
message and reserves exit code `86` as the message-loss fallback. A child
transport-close failure cannot replace that code, and an already observable
code `86` remains an orphan even if the parent `join()` observation fails. Any
truncate, fsync, seek, read, or file-stat failure during JSONL rollback proof is
also unrecoverable. Abort finalization removes a coordinator-owned provisional
`ABORTED.json` after any preterminal failure, preserves an underlying orphan
unless that removal fails, and never publishes a false terminal checksum.
Successful replacement is the sole start commit, and the 1,200-second claim clock
is captured immediately afterward. If replacement return
or successful directory fsync reaches the request deadline, the full started
evidence remains, claim work stays at zero, and the run hard-aborts rather than
rewriting the state as not started.

The governed training-start transaction, including owned same-directory
candidate-temp creation, atomic live `PROJECT_PLAN.md` replacement, directory
fsync, and owned-temp cleanup, is the sole external write exception to the
published-run-root contract. Every disposable publication, ledger, and
terminal rehearsal path is removed after success or body failure; cleanup
failure is an unrecoverable orphan and cannot be replaced by the body error.

Persisted-state classification requires the live launch plan for
`not_started`, `awaiting_review`, and `reviewed_ready`, and the live candidate
for `started`. A pending signal after start aborts with the full started proof,
and no later read failure can downgrade it. The 1,800-second deadline ends at
durable publication. Claim data is then built and one MLX child starts in
command-wait state. A final pending-signal, 1,200-second claim, and frozen-anchor
guard immediately precedes the first `run_stage` request. The request is the
sole admission of training model or optimizer work. The linked plan bytes remain frozen for the rest of the run;
the post-terminal completion update does not create a second in-run barrier.

An unrecoverable orphan is never converted to a hard abort or terminalized.
After child start, the command stream remains closed to `run_stage` while the
single MLX child is quiesced and the parent transport is closed before
propagation. Before abort cleanup, the
runner verifies the exact persisted training-start state, review registry,
cleanup-tree identities, and absence of symbolic paths. Required cleanup,
`ABORTED.json`, or `SHA256SUMS` failure is an orphan, as is foreign replacement
of an exclusively created ledger, temporary artifact, or checksum. Initialization
staging uses the same device-and-inode ownership rule; after activation, an
`InitializationRefusal` maps to `artifact_inconsistency` unless the underlying
failure is already an orphan.

Each final aggregate intervention record now identifies both its evaluated
model and checkpoint and the model, checkpoint, and condition that supplied
its matched raw-delta baseline. The top-level selected-checkpoint shortcut is
removed. Selected-checkpoint conditions, including the local-only and
all-eligible clone configurations, use the selected intact raw delta as their
baseline. The separately trained donor and dense controls identify their own
unchanged condition and checkpoint as baseline. Rung two uses its own intact
endpoint. The aggregate `pre_delta_l2` is one final square root over the exact
cached per-batch float64 square sums keyed by baseline model, checkpoint,
condition, and block; no misleading batch index is added to the final record.

Every routed call summary now retains a compact position-ordered list of the
public search's raw and final effective canonical early-bypass IDs. Every
stored non-query effective list is all `-1`. Width fifteen explicitly includes
position `126`, whose effective list must equal every following query row in
the call. Query-example rows make the bypass-list field null and retain their
own position-`126` raw and effective IDs. Raw early-bypass evidence never
enters the query-only source-exclusion payload, and neither raw nor effective
bypass evidence may be rewritten after capture.

Each call summary also carries exact block-load and valid-posting histograms.
Their weighted sums reconstruct maximum bucket load, overflow, posting reads,
candidate blocks, search rows, and address probes. Search-row frequency is
anchored to one route group and the fixed 128-position geometry: zero rows at
width zero, `B * 104` at width two, and zero at width fifteen. Coherent
rescaling of a histogram and its counters is therefore rejected. The detached
audit calls the pinned public probe helper with exactly the current detached
query-route features, current detached codebooks from the same search, and
`probes=4`; only after that call may the returned addresses be combined with
the existing packed postings and `remote_limit`. Route IDs, stale codebooks,
or an omitted probe count are forbidden. The audit cannot feed a route, loss,
attention output, model state, or gradient and does not repeat packed-index
construction, search, or attention. The duplicate Monodratic source-symbol
registries must agree exactly except that the source-reference registry alone
also names the `MonodraticPHIMixer` class.

## Source-control boundary

The target implementation has a pinned reproducible source boundary. The
reference hashes and public revision below have been reverified locally and
must be checked again before any port or pilot begins:

- `/Users/dttdrv/Projects/Transformerov/scale/model.py` currently has SHA-256
  `6de04cac73a5f1d67cf2c9f5c51691658fcd06e4b62e1704528d51338643d904`.
- `/Users/dttdrv/Projects/Transformerov/scale/gated_delta.py` currently has
  SHA-256
  `e638ba2cb6c9861344befe25d21cce208fc391718f758f5a4338ed4936747bf2`.
- `/Users/dttdrv/Projects/Monodratic-public` is versioned at commit
  `0f9bf59ebdd032da46553d985bcf23348e1d5289`. Its public seam is
  `src/monodratic/core.py::MonodraticPHIMixer`.
- `/Users/dttdrv/Projects/Transformerov` and the private
  `/Users/dttdrv/Projects/Monodratic` tree have no Git metadata. Hashes can
  identify a local snapshot, but they do not make it a released dependency.
- `/Users/dttdrv/Projects/Laplace/trainingnovel` is wholly untracked in the
  Laplace worktree. Karkasov therefore remains protocol evidence rather than a
  source dependency.

The private and public Monodratic mixers expose different controls and
telemetry. Existing private integration results do not validate the public
release. The reviewed run card chooses the versioned public
`MonodraticPHIMixer` ABI. That reproducible choice still requires fresh raw
parity and combined-host proof.

## Repository rework map

The repository will be migrated in place. Historical code remains available
until the target host passes its replacement gates; it will not be refactored
into a registry or compatibility framework.

### Retain for direct reuse

- `v01/data.py` supplies the committed associative-recall, passkey,
  aggregate-counting, and sensor-world generators.
- `v01/evals.py` supplies exact and token-level evaluation plus Wilson
  intervals.
- `v01/sanity.py` supplies generic initialization, overfit, and causal checks.
  Its retention check is specific to the descent-memory toy and does not
  transfer.
- `neuroloc/data/nm_worlds.py` and `neuroloc/data/nm_3d_worlds.py` supply task
  and control contracts. Their oracle outcomes are specifications, not learned
  model evidence.
- `neuroloc/simulations/suite_runner.py`,
  `neuroloc/simulations/suite_registry.py`, and
  `tests/test_simulation_suite.py` remain possible sources of fail-closed
  execution and artifact-validation patterns. The current base harness exists
  independently and does not adopt that registry.
- `requirements.txt` remains unchanged. The first seam requires no new
  dependency beyond the existing Torch requirement.

### Adapt only when a gate requires it

- `config.py` encodes the historical 6:1:1 schedule and cannot describe the
  modular role schedule unchanged.
- `src/model/todorov.py` owns useful normalization and residual structure, but
  its public contract carries per-layer state, cache, offset, and telemetry.
  Shimming every new mixer into that dispatch would preserve the wrong
  abstraction. The working-tree base host therefore uses the narrow delta
  contract directly.
- `src/model/embedding.py` and `src/model/decode_head.py` remain outside the
  base. Legacy Todorov ties input and output weights while the frozen
  Transformerov reference does not; the working-tree host follows the frozen
  untied construction.
- `src/layers/swiglu.py` is not the fixed Transformerov feature control. It has
  a different ratio and rounding rule, returns telemetry, and can activate
  spikes or geometric self-interaction. The first combined gate keeps the
  Transformerov feature block unchanged.
- `src/training/loss.py`, `src/training/optimizer.py`, and
  `src/training/evaluator.py` have no tracked callers. They are considered only
  if a later gate names a concrete need; they are not a default training
  framework.

### Preserve as historical evidence

- `src/layers/kda.py`, `src/layers/mamba3.py`, `src/layers/mla.py`,
  `src/algebra/`, `src/spikes/`, and `src/utils/memory.py` describe and test the
  legacy architecture. They do not enter the first modular gate.
- Tracked `v01/` model, memory, training, quick-test, and report files preserve
  the descent-memory experiment and its negative results. Only the three
  harness files named above transfer directly.
- `docs/EXPERIMENT_LOG.md`, every tracked `neuroloc/wiki/tests/` record, every
  tracked `neuroloc/wiki/mistakes/` record, and `neuroloc/wiki/log.md` remain
  evidence. They are not cleanup targets.
- `docs/ARCHITECTURE.md`, `docs/MEMORY_ANALYSIS.md`, `docs/PHASE_GATES.md`,
  `docs/RUN_TRACKER.md`, `docs/SPIKE_HEALTH.md`,
  `docs/TRAINING_RECIPES.md`, `docs/linux_handoff_2026-05-02.md`,
  `neuroloc/spec/implementation_plan.md`, and `neuroloc/spec/next_gen.md` are
  archive candidates. They remain in place until the operating directive
  defines an archive-folder lifecycle and a link-preserving move is authorized.

### Deletion candidates after replacement proof

- `src/utils/convergence.py` and `src/utils/erf.py` have no tracked callers.
  The latter also assumes the legacy triple-return model.
- The unused training files above become deletion candidates if the modular
  harness does not adopt them.
- `neuroloc/neuroloc_guide.aux`, `neuroloc/neuroloc_guide.log`, and
  `neuroloc/neuroloc_guide.out` are generated build intermediates. Their
  removal is separate from preserving the source and final guide artifact.
No deletion is authorized by this map. Before any later deletion, verify
tracked callers, external consumers, evidence links, and replacement tests.

### Blocked and user-owned surfaces

Untracked files are excluded from the delivered architecture. This includes
the local `v01/transformerov.md`, `v01/bench.py`, `v01/gdn.py`,
`v01/chunk_delta.py`, other untracked `v01/` experiments, the untracked
Candidate G record, `.devin/`, and `.DS_Store` files. Four tracked `v01/` files
also have user modifications in the working tree: `memory.py`,
`quick_affect_test.py`, `quick_count_test.py`, and `quick_touch_test.py`. Their
committed revisions may be cited as historical evidence; their working-tree
changes are not part of this documentation milestone.

## Test and harness map

The working tree now contains the focused base host test, Torch CPU authority,
compiled MLX engine and backend, qualifier, paired local runner, and focused
CPU and MLX tests. The latest complete modular verification reported `834
passed, 2 skipped`, but complete-surface review still has findings, so these
remain scaffold and harness surfaces rather than a complete proof package:

- `tests/test_modular_neural_machine.py` exercises the target full-sequence
  contract, frozen-source adapters, recurrent reference behavior, routed
  intervention isolation, parameter accounting, and serialization boundary.
  The legacy `tests/test_model.py`, `tests/test_kda.py`,
  `tests/test_mamba3.py`, and `tests/test_mla.py` do not establish that target.
- `neuroloc/simulations/memory/modular_sequence_role_cpu.py` and
  `tests/test_modular_sequence_role_cpu.py` implement the frozen generators,
  required-source selection, posting-read, selected-count, overflow, matched
  control, resource, lifecycle, and artifact surfaces. Their direct test
  battery is not a pretraining-assertion, pilot, or claim result.
- `src/model/modular_mlx_backend.py`,
  `neuroloc/simulations/memory/modular_sequence_role_mlx.py`,
  `scripts/qualify_modular_mlx.py`, and
  `tests/test_modular_sequence_role_mlx.py` implement and test the compiled
  Metal training substrate and its exact Torch CPU authority boundary. They do
  not establish a trained result.
- The runner contains the frozen recurrent path, causal knockout, carry-reset,
  and carry-shuffle evaluation arms in the same host. None has yet produced a
  trained claim artifact.
- `hard_symbolic_nm` and `phase1_nm` provide task and control vocabulary. Their
  successful policies are oracle or analytic arms and do not prove a learned
  combined model.
- The deployed feature control lacks a focused shape, gradient, serialization,
  no-feature, and parameter-matched test. The reciprocal mixer has no tracked
  implementation or numerical test.
- Todorov has no tracked Karkasov docking test for exact zero-gate identity,
  trainable-parameter isolation, immutable hashes, incompatible widths, or
  reload.

The base host, paired sequence-role runners, qualifier, backend, and focused
tests are the nine-file working-tree implementation now awaiting literal-zero
review closure, fresh attestations, full pretraining assertions, and a new
governed pilot. The remaining later additions are one
focused reciprocal-mixer numerical test and one focused docking-invariant test,
each blocked behind its own future contract. The role runner is a narrow
preregistered harness, not a second general framework.

## Launch evidence integrity

The launch boundary reads each governed Python source and each selected review
file once through a nonsymbolic descriptor, requires a regular-file `fstat`,
and derives its digest, parse or compile input, and copied bytes from that same
snapshot. Governed module-cache entries are never reused. The four base review
attestations bind all nine reviewed paths before publication and throughout
the frozen lifecycle. Only the separately reviewed atomic training-start
transition may change `neuroloc/wiki/PROJECT_PLAN.md`; no other base target is
exempt.

Pilot and claim IPC are per-worker finite-state streams with exact schemas,
identities, ordering, completion rules, and failure messages. Evaluation
scalar nullability, elapsed-time boundary formulas, complete resource-sample
selection, population-object hashing, and summary pass equations are validated
as derivations rather than accepted as self-reported labels. These are
lifecycle and evidence corrections only; they do not change model arithmetic,
tasks, controls, schedules, seeds, thresholds, or scientific claims.

## Evidence carried forward

The pinned Transformerov reference files pass their current CPU recurrent
self-check against the sequential reference. Maximum errors are `5.36e-07`,
`6.85e-07`, and `1.31e-06` for chunk lengths `16`, `32`, and `64`, and every
arm is finite. Its prior 261M synthetic counting run reached 0.969 held-count
accuracy while an attention-only control stayed at chance. Its 36M
continuous-sensor run reached 1.0 beyond the attention relay while the blind
arm stayed near the prior. These are aggregate-tracking and sensor-carry
results, not language-quality or general world-model evidence. Recurrent
verbatim recall failed.

The pinned public Monodratic release has checksum-verifiable standing evidence.
Its three-seed result records `763 / 768` learned-route answers and `768 / 768`
donor and target-forced answers; `results/checksums.sha256` currently verifies
`mqar.json`, `scaling.json`, and `verification.json`. Its integration proof
covers gradients, optimizer updates, serialization, route equality, overflow,
and graph participation. Its attention operation is exact over gathered
candidates, while selection of the required source remains the routed-recall
question. It proves wiring, not learned division of labor, natural-language
recall, long-context quality, linear cost, or production-kernel speed.

These immutable sibling checks are standing prerequisite evidence and their
full training is not repeated. Local preflight must reverify every checksum and
source hash, rerun the Transformerov numerical self-check, and establish raw
public route, packed-index, and loss parity before the combined pilot. Any
failure blocks execution. None of this sibling evidence validates the combined
host.

Karkasov supplies a task-and-gate precedent, not target-module proof. In its
five-seed learned-routing artifact, the minimum full-route and full-answer
accuracies are both 0.958984375. The maximum state-disabled,
retrieval-disabled, and random-route accuracies are 0.228515625, 0.005859375,
and 0.283203125, and every recorded hash check passes. That witness uses MLX,
an `attention, recurrent, attention` schedule, and dense scores over all ten
candidate blocks before selecting two. It does not test the public Monodratic
index, the target four-block cycle, hard product-address lookup, or sublinear
routing. Its source tree is untracked, so only the recorded task geometry,
controls, thresholds, and artifact fields may be reused.

Karkasov's broader specialist-training phase remains incomplete because later
matched controls, bridge recovery, and replacement evidence are unfinished.
Its backward-compatible replacement result is negative at 0.7695 accuracy
against 0.9375 for end-to-end training. No model-quality or lifecycle advantage
transfers to Todorov.

The nested-reciprocal feature mixer is established only by the external
CoFrGeNet paper. No official implementation or weights were linked from the
paper or IBM pages as of 2026-07-14. It has no local Todorov or Monodratic
result.

## Adoption sequence

The lowest-risk sequence changes one scientific variable at a time:

1. Completed 2026-07-19: Synchronized the corrected batch-local carry-shuffle
   strata and payload digest across [[tests/modular_sequence_role_cpu_run]],
   this article, and the canonical plan, then obtained literal zero findings
   on the linked documentation surface. The replacement review-attestation
   design had already reached zero. Complete-surface review on 2026-07-20
   subsequently reopened the evidence schema defined above.
2. Completed 2026-07-19: Implemented only the minimal model-source port,
   combined host, CPU runner, and two focused test files named by
   [[PROJECT_PLAN]]. No sibling repository was merged and the reciprocal
   feature candidate remained unchanged.
3. Current action: Preserve governed attempts `mlx-m5pro-20260726-1`,
   `mlx-m5pro-20260726-2`, and `mlx-m5pro-20260726-3` unchanged. The third
   attempt passed current preflight and executed all pilot updates in its child,
   but terminal sampling and abort finalization failed on stale zeroed
   exited-child telemetry. The corrected sampler-generation and abort-timeline
   surfaces pass deterministic regressions and complete modular verification at
   `834 passed, 2 skipped` in `214.05` seconds. Fix every remaining complete-
   surface finding as a bug class and rerun
   `feature-dev:code-reviewer` until the documentation, implementation, test,
   and complete scopes each report literal zero findings. The independent
   reviewer then writes four fresh run-id-free content-addressed attestations
   outside every reviewed target scope. The runner neither authors nor mutates
   them.
4. Verify exactly one current attestation per scope, then run all pretraining
   assertions, including sibling checksum revalidation,
   the Transformerov numerical self-check, raw public routing parity,
   reset-aware rung-one parity, and unchanged rung-two chunkwise execution.
5. Run the fixed one-child compiled MLX, zero-cost M5 Pro resource pilot. Its
   target is 600 seconds and its hard complete-package gate is 1,200 seconds;
   it may decide only whether the frozen package can run.
6. Run the two-rung package through the compiled MLX training substrate only if
   every assertion and pilot gate
   passes. Write the durable run-specific request, obtain the exact fifth review
   of the request-bound plan candidate, precommit all 15 anchors, and atomically
   publish that candidate as the sole start before any claim data, child stage
   runtime, training model, or optimizer exists. Torch CPU remains checkpoint,
   evaluation, gate, artifact, and trained-endpoint replay authority. Then
   preserve either the positive or negative result with its full frozen
   evidence ledger.
7. Use [[modular_neural_model_methods_review_2026_07]] as the later-method
   evidence boundary; do not activate the rejected `trainingnovel` lifecycle.
8. Treat the compiled MLX engine and width-five lane vectorization as the
   frozen base substrate. Benchmark any further exact-math M5 execution change
   individually and promote only parity-preserving measured wins.
9. Test the nested-reciprocal feature mixer alone from its existing dossier,
   with no attention, routing, cache, recurrent-state, or base-task change.
10. Use causal knockout evidence for one-at-a-time deletion and recurrent-width
    tests; never import a published sparsity ratio as the local target.
11. Retain the compiled MLX substrate only while initial and every trained
    endpoint pass Torch CPU replay, then test at most one optimizer candidate
    with matched math, tokens, gates, and tuning effort.
12. Integrate only independently passing changes, rerun the complete matched
    role and feature-quality controls, and trim only edges defeated by explicit
    causal deletion tests.
13. Keep unreviewed dependencies and paid compute blocked. Paid work still
    requires a positive locally validated intervention and separate explicit authorization
    from Deyan.

The path not taken is a whole-repository rewrite or simultaneous activation of
all mechanisms. Both would erase the existing evidence boundaries and make a
failure uninterpretable.

## Combined CPU gate

The proof package has two rungs and uses separate checkpoints.

The first rung is the state-conditioned retrieval witness defined above. The
candidate-boundary resets prevent recurrence from carrying early rule mappings,
and query-only remote use prevents block 4 from retrieving a rule before
position `126`. A generic query can therefore identify the required remote rule
only through the cue at position `80`, reset-aware recurrent carry across the
boundary at position `96`, and the effective block-4 route at the query. The
learned route has two slots, records honest underfill, and has no query fallback.

The run card freezes five seeds and 512 held-out examples per seed. For every
seed, the intact effective route must hit the required source at least
`461 / 512`; intact answer count must be at least `461 / 512`; and
target-forced answer count must be at least `487 / 512`. Recurrent knockout and
carry reset must each reduce both original required-source hit and answer count
to at most `153 / 512`. The knockout still runs the exact recurrent mixer and
internal update, then discards only its returned delta.

Carry shuffle uses stratified foreign-state gates because the rolled condition
can equal the original condition. The rolled-condition source must be selected
at least `461 / 512` overall. On changed-condition rows, rolled-condition source
hit must be at least 0.90, while original-source hit and answer accuracy must
each be at most 0.30. Same-condition row counts and outcomes are diagnostic
only. Exact per-seed denominators and integer thresholds live in the run card.

Matched random-route answer count must be at most `153 / 512`. Block-4 routed
knockout, local-only routing, and the minimal required-source-excluded control
must each remain at or below `76 / 512`. The separately trained all-eligible
donor, same-checkpoint all-eligible clone, and separately trained dense causal
baseline must each reach at least `487 / 512`; the dense control uses the
matched `1024 + 512` task-update schedule. Selected-mask oracle parity must have
maximum absolute error at most `1e-5`, route overflow must be zero, and every
binary metric reports exact counts and Wilson 95% intervals. Any failed seed or
control kills the composition claim.

The second rung is the amended held-count query task. It uses the same width,
depth, heads, recurrent chunk length, and deployed feature mixer in a separate
checkpoint, sequence length 512, count region 64, query position 510, local-only
routing, and the unchanged Transformerov chunkwise recurrent path. The intact
model must reach at least `461 / 512`, recurrent-knockout success must be at
most `89 / 512`, and every recurrent gate statistic must be finite. This rung
does not reproduce the sibling project's published run and does not claim
routed recall, cross-task learning, or five-seed stability.

Data, trained values, evaluation examples, and seeds stay fixed across each
rung's interventions. The deployed Transformerov feature mixer stays fixed and
no reciprocal feature parameter enters the graph or optimizer. The complete
two-rung package charges `45,613,056` token positions. Exact generators,
schedules, controls, resource gates, and artifact fields live in
[[tests/modular_sequence_role_cpu_run]].

Telemetry includes raw and effective routes, required and rolled-condition
source hits, selected block IDs, underfill, addresses probed, posting reads,
candidate blocks, overflow, maximum bucket load, routing workspace,
per-recurrent-layer gate values, state norms at every relevant boundary,
intervention deltas, exact per-task counts, parameter and state accounting,
attempted updates, token positions, peak memory, CPU time, wall time, and reload
parity.

## Specialist training candidate

The Karkasov method enters only as this protocol:

1. Freeze a versioned host anchor with a fixed-width activation contract.
2. Train recall, world-state, and feature specialists separately with distinct
   seeds.
3. Publish immutable specialist artifacts and hashes.
4. Load them into a fresh assembly behind identity-initialized bridges and
   scalar gates initialized to zero.
5. Prove that zero gates preserve the anchor exactly.
6. Train only bridges and gates during docking.
7. Verify every frozen hash before and after docking.
8. Compare with matched end-to-end and matched specialist controls.
9. Charge failed specialists, docking, optimizer state, artifacts, operations,
   memory, and wall time across releases.

Engineering validity and scientific advantage are separate gates. Unchanged
hashes and finite metrics do not establish improved quality or lower lifecycle
cost.

## Non-claims

This document does not establish a trained Todorov model, a new recall
mechanism, a new recurrent mechanism, a new feature-mixing mechanism,
compression, imagination, replay, linear-time retrieval, reduced retained
key-value memory, or training superiority. It defines the smallest composition
whose claims can later be tested without category errors.

## Authorization boundary

Explicit direction authorizes the ordered zero-cost local sequence, not an
unreviewed launch. Complete-surface review reopened the evidence schema on
2026-07-20. Three governed attempts later ended without establishing a
contract-valid completed pilot. The scientific method and nine-file
source-and-test scope are frozen while the remaining complete-surface findings
undergo bug-class correction. The final unchanged documentation,
implementation, test, and complete surfaces
must each reach literal zero; the independent reviewer then writes four
run-id-free content-addressed attestations outside their target scopes. The
runner may verify and byte-copy exactly one current matching attestation per
scope but may never author or mutate one. Every pretraining assertion must then
pass, and the frozen resource pilot must approve the unchanged package. The
durable request begins only the bounded fifth-review transaction. Claim data,
worker runtime, model, and optimizer remain blocked while the exact
request-bound candidate receives its fifth zero-finding attestation and all 15
anchors are precommitted. One guarded atomic publication of that candidate is
the sole start and begins the 1,200-second claim clock; a final guarded
`run_stage` request then admits the one MLX child to training work. Compiled MLX
is the frozen base substrate, not a later intervention. Reciprocal feature
mixing, causal deletion, further exact-math execution changes, and one optimizer comparison are authorized only
after that result is preserved and only through separate reviewed run cards,
assertions, pilots, and matched controls. The `trainingnovel` specialist
lifecycle remains deferred. Final integration may contain only independently
passing changes. Unreviewed dependency changes, H200, Kaggle, pods, and every
other paid-compute path remain unauthorized.

[[PROJECT_PLAN]] is the canonical source for the current action and
authorization state.

## See also

- [[PROJECT_PLAN]]
- [[tests/modular_sequence_role_cpu_run]]
- [[Home]]
- [[INDEX]]
- [[concepts/start_here]]
- [[neural_model_dossier_nested_reciprocal_feature_mixer]]
- [[modular_neural_model_methods_review_2026_07]]
- [[substrate_requires_architectural_change]]
- `neuroloc/spec/blueprint.md` — Retained historical design intent and backlog
