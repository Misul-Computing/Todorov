# neural machine specification

see also: `neuroloc/wiki/PROJECT_PLAN.md` is the canonical persistent project state. that file holds the current run, current question, hypothesis under test, decision rules, prior runs index, open research questions, and project glossary. this blueprint covers the design intent and feature definitions; PROJECT_PLAN.md covers what is being run right now and what we have learned. update both when the project state changes.

## status banner (2026-04-17, updated after the sixth paid run)

six paid runs have produced 0% passkey at 256 tokens: god_run (2026-04-11), god_run_v2 (2026-04-12), run1_baseline_noerasure (2026-04-14), run2_slot_memory (2026-04-15, first launch with broken inherited retention), run2_slot_memory_retention_fixed (2026-04-15, second launch with `alpha_log_mean=5.0` explicit and FLA active), and run3_cognition_phase1 (2026-04-17, synthetic cognition corpus with 50% passkey / 30% kv recall / 20% copy). the sixth run trained cleanly to best val_bpb 6.3519 in 72 min on h200 on `neuroloc/data/cognition_corpus.py`, but val_bpb plateaued at the alphabet prior from step 150 and never descended. partial eval before user halted pod reported passkey 0/100 at both 256 and 1024. this run executed the discriminant proposed in `wiki/synthesis/training_objective_vs_architectural_goal.md`: a corpus where retrieval is explicit 50% of training. the article predicted that if phase one produced zero passkey on synthetic data, the substrate could not be trained by sgd at this configuration and the architecture needed deeper changes. phase one produced zero passkey. the architecture-cannot-be-trained-by-sgd branch fires. diagnosis is no longer "training objective mismatch" (the prior-banner hypothesis) but "this substrate does not learn retrieval under sgd with these initialisations and this loss, even when the loss directly rewards retrieval". analysis at `wiki/synthesis/substrate_requires_architectural_change.md` ranking five candidate architectural interventions: (A) output gate init 0 instead of -4, (B) auxiliary retrieval loss on marker-following positions, (C) orthogonal prototype key init, (D) warm start from hand-placed addresses, (E) substrate replacement (titans / larimar / differentiable kv table). run card at `neuroloc/wiki/tests/run3_cognition_phase1_results.md`.

new hard rule: every preset must explicitly override `alpha_log_mean`. enforced structurally by `_assert_preset_retention_safe` in `neuroloc/model/god_machine.py`, which raises at `_resolve_preset` for any preset with DELTA or SLOT layers that does not override the field. FLA hard rule: `god_machine.py` now asserts at preset resolution that any preset requesting FLA actually has the package installed (`_assert_fla_available_if_requested`), with an NM_FORCE_NO_FLA env escape hatch for the CPU smoke suite; this removes the run2-first-launch silent-fall-through failure mode. additional outstanding rule: no next paid run is authorised with only hyperparameter tweaks on the current slot substrate; see `wiki/synthesis/substrate_requires_architectural_change.md` for the allowed interventions and the cpu-simulation gate that must precede substrate-replacement runs.

## context

todorov proved that biological computation math outperforms standard
approaches at matched scale (33.7% better at 267M parameters). neuroloc
mapped 61 biological mechanisms, validated 5 bridge interventions, and
identified which principles are genuine vs superficial. this spec defines
what comes next: a unified neural machine that implements the validated
math as one operation, not a collection of named modules.

this is not a language model. it is a general-purpose neural computation
engine that recognizes, remembers, predicts, and imagines.

## design decisions (from user)

- input: mixed continuous signals (text, image, sensor, any modality)
- output: structured responses in the same signal space
- two operating modes: on-demand (process, respond, stop) and persistent
  (always-on with background processing between active tasks)
- one core operation with mode signals (not separate modules)
- prediction-error-driven processing (only process what surprises you)
- eidetic memory (everything stored, older things fade extremely slowly)
- novel compression to make eidetic storage feasible
- self-directed imagination (the machine decides when to simulate)
- automatic fast/slow path switching (machine detects its own uncertainty)
- first milestone: recognition + memory + prediction unified
- learning method: open research question (discover, don't prescribe)
- compression depth: open research question (find the best way)
- self-monitoring: separate inspectable system (medical telemetry)
- name: unnamed until it works

## the core operation

one mathematical operation, repeated. each step:

    z = Q(R(B(C(x), C(h))))

where:
- x is the current input signal
- h is the accumulated memory state
- C is compression: select the important parts, discard the rest.
  implemented as competitive selection (top-k by magnitude) producing
  sparse rate-coded values (survivors keep their continuous magnitude,
  the rest are zero). this is NOT ternary -- values are continuous.
- B is mixing: combine the compressed input with compressed memory.
  outer product (creates associations), element-wise product (gates),
  or structured bilinear map (geometric relationships)
- R is rotation: encode position, phase, and temporal order through
  rotation in the signal space
- Q is output shaping: compress the result back to a transmittable form

the same equation runs at every step. what changes is the MODE SIGNAL
that shifts which variant of each operator is active:

- **remember mode**: C selects what to store, B writes it to memory
  (outer product), R encodes when it happened, Q shapes the memory
  update. memory state h is updated.
- **retrieve mode**: C compresses the query, B matches it against
  memory (dot product retrieval), R aligns temporal position, Q
  shapes the retrieved content.
- **imagine mode**: C suppresses external input (gate closed), B
  recombines stored fragments (outer product on memory contents),
  R generates novel temporal sequences, Q evaluates plausibility.
- **predict mode**: C processes input normally, B compares input
  against prediction from previous step, R maintains sequence
  position, Q produces the prediction error (what was surprising).

the mode signal is not a hard switch. it is a continuous blend
determined by the machine's own state (how much external input vs
how much internal processing is needed at this moment).

## memory architecture

eidetic memory with slow fade:

the memory state h is a matrix per processing unit (like the current
outer-product state). UNLIKE current design:

- decay rate is near-zero for important memories (alpha close to 1.0)
- decay rate is activity-dependent (BCM-like: busy memories fade
  faster to prevent saturation, quiet memories persist). validated
  by pilot: gamma=0.3-0.5 stabilizes without degrading retrieval.
- targeted erasure: when writing a new value for an existing key, the
  machine can erase the old value first and then write the new one.
  this remains an explicit experimental branch, not the dense-key
  baseline default, because the current standalone overwrite sweep at
  `d_head=64` shows that erasure hurts the first useful retention knee.
- periodic consolidation: every N steps, compress the memory state
  into a summary snapshot and store it permanently. the fast state
  can then be partially cleared. old snapshots are addressable by
  the retrieval operation.

compression for eidetic storage:
- hierarchical ternary: top-k selection + ternary values = 0.37 bits/dim
- consolidated snapshots: rank-r SVD every 64 steps = 64x more compact
- content-addressable: hash-based deduplication, store by concept not token
- further compression methods are an open research question

## imagination

the machine can run internal simulations on demand.

mechanism: the same core operation runs, but with external input GATED
(suppressed). the compressed memory fragments are recombined through
the mixing operation to produce novel configurations that have never
been directly experienced.

control:
- imagination is OFF by default (suppression gate closed)
- the machine activates imagination when it detects high uncertainty
  (prediction errors are large and increasing) or when explicitly
  asked to "think about" something
- the suppression gate prevents uncontrolled generation (no intrusive
  thoughts). the gate is a learned binary control signal.
- during imagination, the machine evaluates each generated scenario
  against its learned model of plausibility. implausible scenarios
  are discarded. plausible ones inform the next action.

## fast and slow processing

the machine automatically detects when it needs more computation:

- **fast path**: one pass through the core operation. good for
  familiar inputs where predictions match reality (small errors).
  cheap. this is the default.
- **slow path**: multiple iterations of the core operation on the
  same input. each iteration refines the result. triggered when
  prediction errors exceed a threshold (the machine is surprised
  or uncertain). more expensive but more accurate.

the switching is automatic: the prediction error magnitude at each
step determines whether to continue iterating or commit to the output.
this is the biological equivalent of the ACC conflict monitor -- it
detects when more processing is needed and triggers it.

## processing units (not "neurons")

each processing unit is NOT a point. it has multiple independent
sub-operations (like dendritic compartments):

- K = 4-8 independent sub-gates per unit (block-diagonal)
- each sub-gate receives from a DIFFERENT source (some from input,
  some from memory, some from neighboring units)
- the sub-gates perform independent competitive selection (k-WTA)
  before their outputs are combined
- this gives each unit ~4-8x the computational capacity of a point
  unit at the same parameter count

the multi-compartment structure is validated by Beniaguev 2021
(single neuron = 5-8 layer deep network, depth from NMDA nonlinearity).

## connectivity

sparse, structured, not fully connected:

- each unit connects to ~1-20% of other units (not all)
- connections are structured: high local connectivity (nearby units
  connect densely), sparse long-range connections (distant units
  connect selectively through learned hubs)
- hub units (rich club) are a small fraction (~5-10%) that connect
  widely and mediate global integration
- this small-world topology is validated by the connectomics research
  (watts & strogatz 1998, van den heuvel & sporns 2011)

## self-monitoring (telemetry)

a separate, lightweight monitoring system observes the main computation:

- tracks: memory state norms, prediction error magnitude, spike
  patterns, imagination gate state, fast/slow mode, which sub-gates
  are active
- produces: a real-time dashboard of the machine's internal state
- enables: sampling data at any point for analysis
- does NOT interfere with the main computation
- this is the "medical system" -- see everything as it happens

planned for future: built-in self-awareness where the monitoring
signals feed back into the main computation (the machine reasons
about its own uncertainty).

## two operating modes

**on-demand mode**: receives input, processes it through enough
iterations to produce a confident output, emits the output, stops.
memory state is preserved between calls but no background processing
runs. like current AI chatbots.

**persistent mode**: always running. between active tasks, the machine
runs at reduced priority:
- consolidation: compresses and reorganizes memory
- imagination: generates and evaluates hypothetical scenarios
- maintenance: cleans up memory, removes redundant entries
- monitoring: self-checks, confidence calibration

the machine works in both modes. persistent is better but on-demand
is cheaper.

## what the first version must do

recognition: given an input signal (text, image, pattern), identify
what it is by matching against stored memory.

memory: store new experiences with full detail. retrieve them by
content (what was it?), by time (when did it happen?), or by
association (what else was happening?).

prediction: given a sequence of signals, evaluate the current state
against expected state transitions and expected consequences. use
prediction errors to drive learning and allocate processing.

these three capabilities emerge from the same core operation:
- recognition = retrieve mode (match input against memory)
- memory = remember mode (write input to memory state)
- prediction = predict mode (compare present state against expected state or action outcome)

## research questions -- ANSWERED by experiments (2026-04-09)

1. **learning rule**: ANSWERED. backprop for v1. one-shot hebbian gives
   90% of optimal for free (experiment 012). delta rule = gradient descent
   for linear layers. the unified learning hypothesis (forward-pass-as-
   learning) is an untested conjecture requiring prediction error and
   runtime-adaptive thresholds not yet in the architecture.

2. **compression level**: ANSWERED. rate-coded k-WTA at 10% = 0.56 bits/dim,
   CKA 0.71. at 20% = 0.92 bits/dim, CKA 0.81. the tradeoff is smooth
   and predictable (experiment 007). novel: no prior work on k-WTA +
   rate-coded activation compression with CKA validation.

3. **memory capacity**: ANSWERED IN PART. binary hopfield saturates at
  alpha=0.08-0.10 with 5% corruption. ternary has ~5x lower practical
  capacity (experiment 006). but the current asymmetric dense-key sweep at
  `d_head=64` changes the erasure claim: the focused overwrite test at the
  first useful retention knee (`decay=0.90`) shows that erasure hurts every
  tested encoding at 32 patterns, so overwrite is no longer a baseline
  assumption for the dense-key branch. recurrent state alone also cannot
  provide eidetic memory -- exact retrieval mode (MLA function) is still
  needed for long-range recall.

4. **imagination quality**: ANSWERED. recombined queries produce structured
   interpolation (cosine 0.93), not noise. random queries produce -0.02.
   novelty rate 7.5% (experiment 008). imagination via outer-product
   recombination is functional.

5. **connectivity**: ANSWERED. small-world beats random sparse at 50%
   density when connection counts are matched (experiment 009, corrected).
   at lower densities, random sparse shows higher overlap. use small-world
   BETWEEN processing stages, random sparse WITHIN each memory unit.

## exact retrieval as a mode (replacing separate MLA module)

the formal impossibility result (wen et al. ICLR 2025) proves pure recurrence
cannot solve associative recall perfectly beyond a retrieval horizon. the
solution is NOT a separate MLA module but a MODE of the same operation:

- when the mode signal indicates retrieval is needed and the recurrent state
  cannot provide a confident answer (high prediction error on retrieval),
  the operation switches to exact attention over consolidated snapshots
- consolidated snapshots are compressed summaries of past recurrent state,
  stored periodically (every N steps)
- this is functionally equivalent to MLA but implemented as a mode, not
  a separate layer type
- the retrieval mode uses dot-product attention over the snapshot bank,
  which is O(T/N) in the number of snapshots, not O(T) per token

## implementation sequencing (mandatory -- do NOT bundle)

each feature is validated in isolation before combination. this follows
the phase 5 sequencing protocol from CLAUDE.md. **the ordering below has
been revised on 2026-04-12 after god_run_v2's 0% passkey confirmed that
the prior ordering (k-WTA on both keys AND values) destroys verbatim
memory. see `neuroloc/output/god_run_v2/run_card.md` for the 8 external review findings.**

> **superseded 2026-04-17**: the run ordering below was the working plan
> as of god_run_v2 (2026-04-12). since then, `run1_baseline_noerasure`
> (2026-04-14), `run2_slot_memory` first launch (2026-04-15),
> `run2_slot_memory_retention_fixed` (2026-04-15), and
> `run3_cognition_phase1` (2026-04-17) were all executed, all returning
> 0/100 passkey at 256. the slot-memory design in
> `wiki/synthesis/slot_memory_design.md` superseded the dense-matrix
> sequencing below; the architectural-intervention analysis in
> `wiki/synthesis/substrate_requires_architectural_change.md` then
> supersedes that in turn. the current authoritative next-run protocol
> lives in `wiki/PROJECT_PLAN.md` (experimental method and decision
> rules) and the current ranked intervention list lives in
> `substrate_requires_architectural_change.md`. the remainder of this
> section is retained for evidence continuity and is not the plan of
> record for the next paid launch.

**current working rule (post-god_run_v2)**: keys stay dense in the
current run ordering. k-WTA, sparsification, and any form of
address-space truncation are deferred to the value side until isolation
runs say otherwise. a 20%-sparse key in a 64-dim head carries
~13 effective dimensions, which drops Hopfield interference-free capacity
from ~9 patterns to ~2 patterns per head. delta rule erasure with sparse
keys also reads only the 13 surviving dimensions of the state, leaving
"ghost" content in the zeroed dimensions that accumulates and interferes.
both failure modes are the strongest current review findings from
god_run_v2 and motivate the current run ordering.

**historical run block**: the run ordering below is preserved as the
pre-curriculum paid-run plan. it is not currently authorised; paid compute is
gated on funding plus a cpu-validated intervention.

**run 1 (baseline dense)**: use `god_machine.py` with preset
`run1_baseline_noerasure`: dense keys, dense values, and no delta
erasure. use the non-FLA path. no activity-adaptive decay, standard SwiGLU (not
multi-compartment), no imagination probe, no PC diagnostic head.
target: preserve the run_010 dense-memory quality regime while restoring
measurable retrieval on the isolated baseline path. **critical success metric: nonzero
passkey at 256 tokens.** before the full 4000-step launch, the official
surface now requires a 20-step fineweb-backed h200 benchmark that starts
from a fresh directory, does not use resume, and writes
`run1_baseline_noerasure_benchmark_manifest.json` in the shared output
root; the full launch requires that manifest plus `NM_AUTHORIZE_FULL_RUN=1`,
checks that the current device matches the benchmarked hardware class,
locks the official run to the canonical 4000-step config with no runtime
override drift, binds benchmark provenance to the recorded artifacts and
git working-tree fingerprint, and treats inconclusive eval outcomes as
hard failures instead of a completed gate. if this run produces 0% passkey, do one slower
static-retention ablation with erasure still off before claiming the
problem is deeper than the bundle.

**run 1a contingency (slower static retention, still no erasure)**: if run
1 still returns 0% passkey, raise the static retention initialization in
the same `god_machine.py` baseline path without adding activity-adaptive
decay or overwrite. the standalone decay sweep at `d_head=64` showed that
exact-query recall first reopens at 32 patterns around `decay=0.90`,
while the current `god_machine.py` baseline initialization starts much
lower. target: determine whether the dense-key baseline is failing because
overwrite was wrong, because static retention is still too low, or
because the base mechanism is genuinely insufficient.

**run 2 (value-side compression, dense keys)**: the leading candidate
for value compression is the correction-field design from
`wiki/synthesis/correction_field_memory.md`. it replaces raw value storage
with prediction-residual storage: values are projected from
`r_t = h_t - f(h_{t-1})` instead of `h_t`, and writes are modulated by
a surprise ratio. retrieval reconstructs as `prediction + correction`.
the design compounds predictive filtering and generative replacement
(mechanisms 1 and 2 from the six-mechanism list) plus prediction-residual
value storage in one mechanism, with ~1% parameter overhead (a
low-rank prediction head per layer). keys remain dense.

the correction-field design is validated by cpu simulation before entering
this slot, but the checked-in workspace currently contains only the prompt
for that harness (`simulations/prompts/correction_field_capacity.md`), not
the simulation implementation itself. if the eventual simulation does not
confirm a capacity increase, the fallback is rate-coded 50% selection on
values only. in either case, do not launch
run 2 until the checked-in config surface can express value-only
compression with separate key/value controls and a smoke-tested preset.
target: preserve run 1's retrieval accuracy while compressing value storage.

**run 3 (activity-adaptive decay with faster EMA)**: add activity-adaptive
decay on top of run 2. use momentum 0.95 instead of 0.99 (half-life ~14
steps vs ~69). initialize `running_state_norm` to 0.01 instead of 1.0 so
the correction is active from step 0 rather than kicking in late after the
state has already saturated. this is also a pending implementation branch,
not a launchable preset on the current checked-in surface: the bcm ema and
initial norm are still hardcoded, so run 3 requires config-threaded controls
plus a smoke-tested preset before launch. target: preserve or improve
retrieval over run 2 while testing whether adaptive retention helps without
reintroducing the bundle failure mode.

**run 4 (erasure ablation after dense-key validation)**: if the earlier
runs preserve retrieval, run an erasure ablation that compares the
no-erasure dense-key baseline against the overwrite-subtraction variant
under the same retrieval gate. target: determine whether erasure becomes
net helpful only after the dense-key baseline is already working.

**what is explicitly NOT added unless validated in a dedicated run:**
- imagination probe (external review flagged it as a likely gradient-bypass
  mechanism that may let the model lean away from content-addressable
  storage; gate rose from 0.12 to 0.22 during training)
- multi-compartment SwiGLU K=4 (external review flagged it as a likely
  contributor to effective width loss; it is also one source of the 283M
  vs larger-baseline parameter gap)
- PC diagnostic head (external review flagged it as likely loss drag with
  no demonstrated retrieval benefit in the bundled run)
- retrieval mode / consolidated snapshots (deferred to next_gen.md phase 6
  tiered architecture — not a bolt-on to the single-tier design)

each run that improves or maintains quality AND retrieval is KEPT.
each run that degrades is DROPPED. retrieval is a HARD gate: a run that
drops passkey to 0% is rejected regardless of BPB improvement.

## verification

each run is verified by:
1. prosecutor audit of the code before execution, with every finding fixed to zero
   (no cherry-picking; each finding is a bug class, grep analogous sites and fix them all)
2. eara ultra loop pre-push checks (5 subagents)
3. comparison against the baseline (run 1) on matched token budget
4. statistical significance (p < 0.01) for any claimed improvement
5. documentation update after each run
6. end-to-end telemetry round-trip: every metric computed in code must be written to the
   on-disk jsonl and verified by reading it back. in-memory assertions alone do not count
   (god_run lost 35 probe metrics because this guard was missing)

## empirical status

**god_run (2026-04-11)**: the first direct test of the 5-feature bundle. 283m params on
fineweb-edu, 131m tokens, 4000 steps on h200. bypassed the sequential-isolation protocol
below (activated all 5 features at once with always-on imagination probe and pc diagnostic
head). result: val_bpb 1.3950 (best in project history, 2.57x vs transformer baseline) but
retrieval FAILED at every tested length (passkey 0/20 and selective copy 0/20 at 256, 512,
1024, 2048, 4096). delta state structure probe shows the delta memory is statistically
indistinguishable from high-dimensional noise (mean_structure_ratio 0.981, mean_pairwise_cos
-0.003). perplexity-at-length decreases monotonically with context, proving the
compressed-attention path uses context correctly; the delta-rule memory is the channel
that failed.

interpretation at the time: the compressed-attention + mlp path fit the next-byte
distribution while the delta-rule memory never became content-addressable. this is the
lossy-mechanism failure mode predicted by
`wiki/synthesis/compression_beyond_quantization.md`. before the rerun, one confounder
remained: the BCM train/eval path divergence (prosecutor finding F1) meant the model was
trained under one dynamical rule and evaluated under a different one for everything beyond
the first chunk. that unresolved confound is why god_run_v2 was launched.

historical pre-rerun decision rule: if god_run_v2 had recovered passkey, F1+F2 would have
been treated as the primary culprit and the five-feature bundle would have stayed in play.
it did not: god_run_v2 kept passkey at 0/100, so the bundle-is-broken branch below is the
resolved project state.

**god_run_v2 (2026-04-12)**: the re-run. all 17 F1-F17 + 14 G1-G14 prosecutor findings
fixed, including the critical F1 `torch.sigmoid → torch.exp` math correction for the
log-space gate and F2 `running_state_norm` training-only gating. 283m params, same
pipeline, 4000 steps, 131m tokens, 59 min runtime. **result: val_bpb 1.4453 (+0.050 vs
v1), passkey 0/100 at 256/1024/4096 (Wilson 95% CI upper 3.7%), copy 0/100 at every
length. F1 math fix did NOT recover retrieval.** the tested 5-feature bundle failed, and
the strongest current review findings point to key-side k-WTA plus sparse-key erasure as
the leading contributors. decision rule hits the bundle-is-broken branch: fall back to run 1.

external review (2026-04-12) identified 8 candidate contributing mechanisms. the two
strongest are k-WTA on keys (destroys address space, drops per-head capacity from ~9
patterns to ~2) and delta erasure with sparse keys (ghost content accumulates in zeroed
dimensions). the six supporting review findings are listed in
`neuroloc/output/god_run_v2/run_card.md` and inform the
revised run ordering below.

see `neuroloc/output/god_run/run_card.md`, `neuroloc/output/god_run_v2/run_card.md`,
and `neuroloc/wiki/tests/god_run_findings.md` for full detail.

**run1_baseline_noerasure (2026-04-14)**: the "bundle off" baseline. 353m params on
fineweb-edu, 131m tokens, 4000 steps on h200, 42 min runtime after the FLA-no-erasure path
was added (otherwise the recurrent path projected 70h). all 5 bundle features removed: no
k-WTA, no delta erasure, no BCM alpha, no multi-compartment SwiGLU, no imagination probe,
no PC diagnostic head. result: **val_bpb 1.4499, passkey 0/100 at 256, copy 0/100 at every
length**. same outcome as god_run_v2 despite removing every bundle feature, which
falsified the "one of the bundle features is the killer" hypothesis at the substrate level:
the base matrix-memory substrate itself does not retrieve. inherited
`alpha_log_mean=-0.5` from Config defaults, putting this run in the state-evaporation
regime (alpha_eff=0.377 compounds to 10^-109 over 256 tokens). run card in
`neuroloc/output/run1_baseline_noerasure/`.

**run2_slot_memory first launch and retention_fixed (2026-04-15)**: pivot to a slot-memory
substrate — softmax addressing over 64 per-head prototype keys with surprise-gated
least-recently-used writes and an explicit output gate. FLA-accelerated via
`fused_recurrent_simple_gla`. 355m params. two paid launches:

- first launch inherited `alpha_log_mean=-0.5` from Config defaults, reproducing the
  state-evaporation bug documented four days earlier in
  `wiki/synthesis/linear_attention_retrieval_wall.md`. val_bpb 1.5107, passkey 0/100. the
  structural guard `_assert_preset_retention_safe` was added in commit `f0e1a9a` so the
  bug class is structurally closed. mistake doc:
  `wiki/mistakes/run2_slot_memory_decay_copy_paste.md`.

- second launch set `alpha_log_mean=5.0` explicitly (alpha_eff=0.9933, retention 0.18 at
  256 tokens). first launch attempt of this run lost 17 min to a silent fall-through to
  the python recurrent loop because `flash-linear-attention` was not pinned in
  `requirements.txt`; fix in commit `edcfe5d`, and structural guard
  `_assert_fla_available_if_requested` added 2026-04-17 to fail loudly on future
  FLA-requested / FLA-missing mismatches. the completed second launch trained cleanly:
  val_bpb 1.4777 (best), 72 min at 33k tok/s. partial eval before user halted pod:
  **passkey@256 = 0/100, passkey@1024 = 0/100** (Wilson 95% CI upper 3.7%). run card at
  `neuroloc/wiki/tests/run2_slot_memory_retention_fixed_results.md`.

with the retention fix and FLA actually active, the slot-substrate-under-SGD-on-fineweb
question was answered: the mechanism that retrieves cleanly on cpu gates A and B did not
retrieve after 4000 steps of LM training. the diagnosis reframed to the training objective
being the obstacle, analysed in `wiki/synthesis/training_objective_vs_architectural_goal.md`.

**run3_cognition_phase1 (2026-04-17)**: discriminant test of the training-objective
hypothesis. same slot substrate and retention as run2_slot_memory_retention_fixed, but
trained on a synthetic cognition corpus (`neuroloc/data/cognition_corpus.py`, 50% passkey
blocks / 30% kv recall / 20% copy, hex ASCII alphabet, distinct marker triples per task).
355m params, 131m tokens, 4000 steps in 72 min at 32,800 tok/s on h200. result: **val_bpb
6.3519, plateaued at the alphabet prior from step 150 and never descended. partial eval:
passkey@256 = 0/100, passkey@1024 = 0/100.** the training loss directly rewards predicting
the byte immediately after MARK_QUERY (the passkey target) in every passkey block — and
the substrate still did not learn it. this run executed the discriminant proposed in
`training_objective_vs_architectural_goal.md` between "the training objective is the
problem" and "the architecture cannot be trained"; the architecture-cannot-be-trained
branch fires.

post-run-3 analysis in `wiki/synthesis/substrate_requires_architectural_change.md`
catalogs five candidate architectural interventions ranked by cost: (A) open the output
gate at init, (B) auxiliary retrieval loss on marker-following positions, (C) orthogonal
prototype key init, (D) warm start from hand-placed addresses, (E) substrate replacement
(titans fast-weight, larimar, differentiable kv table with hard attention). A-C are
single-file changes that can be bundled in one paid run; D-E require the cpu-simulation
gate pattern that preceded slot memory. another paid run on the current substrate with
no architectural change is strictly predicted to produce 0% passkey and is not authorised.
run card: `wiki/tests/run3_cognition_phase1_results.md`.
