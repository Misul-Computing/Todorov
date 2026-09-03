# Status board

Canonical persistent project state lives in `neuroloc/wiki/PROJECT_PLAN.md`.
This status board is a subordinate human-readable mirror and historical run
summary; it does not compete with the project plan.

## Latest local session: CLS trainer (2026-08-12 to 2026-08-13)

The latest documented direction is a Todorov-owned complementary-learning-
systems trainer on an Apple M5 Pro using MLX Metal and byte-level FineWeb-Edu.
The fast path adapts a private gated-delta state inside a sequence. The slow
path is a versioned specialist trained against a frozen trunk and opened
through a zero-initialized scalar gate. The first model pattern is
`[attn, delta, delta, delta]` at sequence length `512`, attention window `128`,
width `768`, `12` layers, `12` heads, and head width `64`.

The local registry reports a `1.73e-6` Metal kernel difference under a `1e-5`
test bound, about `2.2 GB` peak memory for the sequence-512 fit probe, hybrid
language snapshots below their dense controls, a toy role result of `0.953`
versus `0.141` for the dense control and delta knockout, and a best isolated
count result of `0.938` with the PHI path disabled. A joint 4.6M snapshot
recorded passkey `1.00`, count `0.891`, feel `1.00`, and imagination `0.984`
on `64` cases. The frozen Llama-3.2-1B 4-bit comparison remained ahead at
`0.894` bits per byte versus `2.268` for the continued hybrid snapshot.

These values are local session snapshots, not a general model claim. The
`4.6M` joint snapshot remains the working object. PHI-on and PHI-off count
differences require explicit frozen-path and knockout controls. Repeated
100M-class gap-32 passkey attempts are not the next efficient probe.

The trainer, generated data, checkpoints, and result registry remain local and
are excluded from the public source push. The detailed evidence record is
`neuroloc/wiki/tests/todorov_cls_macbook_session_202608.md`.

## Preserved July MLX stop (2026-07-26)

Stopped state: Deyan stopped the session before attempt 4. No attempt-4 run
root or claim training exists. The final contract-order synchronization was
interrupted after the machine-readable payload changed but before its digest
constants and documentation settled. The payload currently hashes to
`8c7825f69fd27a7f3653c2e3bfab8673f3bb13d9f543fecd4a6aa9b97a4868ab`;
launch constants still bind
`fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553`.
The stopped bytes are unverified, active attestations are stale, and launch is
blocked. See `docs/MLX_SESSION_STOP_HANDOFF_2026-07-26.md`.

The preserved July workstream is the modular architecture contract in
`neuroloc/wiki/synthesis/modular_neural_model_stack.md`. Transformerov supplies
the tested host and recurrent world-state path. Monodratic supplies the routed
selected-set attention candidate and must still prove exact recall after
routing. `neuroloc/wiki/synthesis/neural_model_dossier_nested_reciprocal_feature_mixer.md`
defines an unvalidated token-local candidate. Karkasov supplies only a later
specialist-training and zero-gate docking protocol.

The July session resumed the result-first local path on 2026-07-26. Three prelaunch bug
classes were repaired with focused red-green tests: sampler stop no longer
accepts an in-flight row after stop wins, the second claim swap sample is
post-data and immediately pre-spawn, and the live claim sampler inherits the
original post-`proceed` swap baseline. Final code-and-test-byte verification
reported `811 passed, 2 skipped` in `188.97 s`.

Governed attempt `mlx-m5pro-20260726-1` serialized `48` passing preflight
records and then stopped after `1.216127125 s`, before update one, because
duplicated parent validators expected stale execution aliases. The parent
protocol is now fully single-sourced across every parent pilot identity and counter,
including seed offsets and next-update selection. The focused MLX file reports
`65 passed, 2 skipped`, and a complete pilot-message audit found no additional
mismatch. The modular `812 passed, 2 skipped` result in `168.67 s` preceded
the final reviewer-driven closure and is intermediate. Final unchanged
code-and-test-byte modular verification then reported `812 passed, 2 skipped`
in `170.27 s`, but third review exposed independent child seed arithmetic. The
child now uses named seed constants bound to parent and preregistration; final
unchanged code-and-test-byte modular verification reports `812 passed, 2
skipped` in `166.42 s`.

Four refreshed attestations admitted `mlx-m5pro-20260726-2`. It again
serialized `48` passing records, and its durable ledger reached `132`
attempted pilot updates and `292,864` attempted positions before the parent
rejected correct preregistered one-element warmup timing lists. Peak resident
memory was `1,966,882,816` bytes and swap growth was zero. Complete-surface
review then invalidated both `48 / 48` classifications because mandatory
parity surfaces were absent and hidden and sequence-delta errors exceeded the
uncalibrated `1e-5`-only bound. It also proved the tail validator still accepted
fabricated incomplete evidence. The prior focused MLX result at `66 passed, 2
skipped` and complete modular result at `813 passed, 2 skipped` in `166.70 s`
are therefore intermediate.

The tail producer and consumer now bind exact fixture hashes, the complete
`438,368`-byte evaluation fixture, current-model checkpoint lower bounds,
selected-final clone state, engine-bound metadata, and observed scratch
cleanup. Eight fresh Metal processes calibrated and froze complete forward and
raw-gradient limits; two untouched held-out forward cases passed. Independent
float64 semantic optimizer formulas and a priori bounds replaced the invalid
exact cross-runtime optimizer requirement, and a different-gradient second
update passed with nonzero carried moments. The full MLX regression file now
reports `84 passed, 2 skipped`. Four fresh complete engine-to-parent checks
passed on `Device(gpu, 0)` with parent worst-bound ratio
`0.95367431640625`. Complete modular verification reports `832 passed, 2
skipped` in `219.61 s`.

Four literal-zero attestations then admitted governed attempt
`mlx-m5pro-20260726-3`. Its current preflight passed, and its MLX child
executed all `132` pilot updates and reported `292,864` attempted positions.
An in-flight parent resource read captured the exited child with zero resident
memory and zero CPU time while retaining an older active rung-two job. That
stale row committed after lifecycle state changed. Terminal sampling failed,
and hard-abort finalization then rejected the zeroed process as decreasing CPU
time. The run published neither `run/pilot.json` nor `ABORTED.json`, created no
training-start request, and began no claim work.

The shared sampler was then changed to invalidate in-flight generations on every progress,
job-clear, and child-exit mutation, rejects zeroed telemetry for a previously
positive child, quiesces periodic sampling before a required final sample, and
durably writes a parent-only terminal row. Legacy abort validation treats the
attempt-3 zeroed child only as terminal disappearance and rejects PID
reappearance. The full MLX file then reported `84 passed, 2 skipped`, focused
CPU resource and abort verification reported `77 passed`, and the last settled
attempt-3 modular verification reported `834 passed, 2 skipped` in `214.05 s`.
A later pre-interruption modular surface reported `844 passed, 2 skipped` in
`220.75 s` before the stopped payload edit. Neither result certifies the
stopped bytes.

If Deyan explicitly resumes the July MLX lane, the next ordered gates are
literal-zero review of the stopped bytes, four fresh attestations, and the full
pretraining assertion package before a new one-child compiled-MLX pilot may
run. The pilot must project the unchanged complete package at no more than
`1,200 s`; `600 s` remains the target. No training-start publication, claim
model update, or optimizer update has occurred. Paid compute remains
unauthorized. The focused record is
`neuroloc/wiki/tests/modular_sequence_role_mlx_resumption_20260726.md`.

Everything below is a prior status snapshot retained as evidence. Candidate G,
the compression-first 100k line, and earlier paid-run proposals are not part of
the current documentation deliverable. Historical wording is preserved unless
a path or factual claim requires an explicit correction.

## Prior Candidate F result: affect-gated write closed negative on CPU (2026-06-10)

a cpu-only extension of the v0.1 track (no paid compute). the pre-registered candidate F — route the descent memory's inner-loop prediction-error magnitude (surprise) into the write gain — was executed on the v01/ mlp substrate on mqar and CLOSED NEGATIVE: true surprise is at chance under the own-history normaliser (affect 0.5 and 1.0) and under exact budget matching (realised write gain 1.000), so surprise content is inert on the write side. the falsification surfaced an anomaly: the shuffled-surprise control trained the descent memory to mqar exact 0.360 / token 0.785 (1200 steps, seed 0, n=100) — the first sgd-trained non-chance retrieval on a recurrent memory substrate in this project, verified memory-mediated (a pinned no-write baseline stays at chance) and causally clean (max_diff 0.0). it is an existence proof, not a recipe: ignition is seed-sensitive (chance at seed 1) and is not reproduced by uniform head-iid (token 0.098) or token-correlated (chance) write-gain noise at matched init/data. the anomaly is promoted to candidate G (stochastic write gain) in the backlog, gated behind its own cpu pre-registration and a literature check. supporting cpu context (bench `v01/feel_tests.md`): the linear matrix memory trains into partial mqar natively (exact 0.470 at 6 pairs / 400 steps; the cliff at 10 pairs is an optimization slope, not a hard capacity wall), a 4-layer attention reference stays at chance through 2000 steps, the feel-bench parity question settled positive at the count-query position, and feel-and-predict pretraining shows generic negative transfer. run card (frozen) `neuroloc/wiki/tests/affect_gated_write_cpu_experiment.md`; candidate F outcome and candidate G in `neuroloc/wiki/synthesis/substrate_requires_architectural_change.md`. the compression-lane phase below remains the prior documented default workstream.

## architecture-track v0.1 session complete (2026-06-02)

deyan reactivated the old todorov architecture track for a one-session v0.1 (user-authorized, overriding the no-paid-compute state) on a personal runpod h200. a clean, test-gated descent-memory codebase was built from scratch (git branch `v0.1`, `v01/`). controlled toy result: the recurrent descent memory (candidate E) was made numerically stable but learns at chance on mqar; the linear delta control is also at chance; a 4-layer attention model solves passkey/induction at exact_acc 1.000 (n=100, wilson 95% lower bound 0.963). seven implementation bugs were caught by the sanity gates and telemetry. no 100m run. run card `neuroloc/wiki/tests/v0_1_descent_memory_toy_results.md`; synthesis `neuroloc/wiki/synthesis/descent_memory_intervention.md`; bugs `neuroloc/wiki/mistakes/descent_memory_v0_1_bugs.md`. this is a user-authorized architecture side-session; the compression-lane phase below remains the prior documented workstream.

## compression-lane substrate status: neural model margin adapter/update product present, high-density knowledge compression unsolved (2026-05-13)

**the neural model is the live machine; this section records the compression-lane research substrate that feeds it.** the prior paid runs and failed substrates are the historical evidence that shaped the architecture program's division of labour. the canonical project surface is the single master plan at `neuroloc/wiki/PROJECT_PLAN.md`: a biology-led, proof-gated neural world-memory model program organized as a flagship paper plus gated side-paper candidates and six research lanes.

the six lanes are cellular state storage, operation-preserving compression, memory/addressing/replay/imagination, 3d world model and physics, trainability and evaluation, and project infrastructure/curriculum/paper operations. the first no-paid cellular/local-state gap map is `neuroloc/wiki/synthesis/cellular_state_storage_gap_map.md`, the first mechanism dossier is `neuroloc/wiki/synthesis/neural_model_dossier_eligibility_gated_local_commit.md`, the first mechanism-specific symbolic contract is `neuroloc/wiki/synthesis/neural_model_symbolic_contract_eligibility_gated_local_commit.md`, the implemented symbolic/oracle package is `neuroloc/wiki/tests/eligibility_gated_local_commit_test_material.md`, the first oracle compression result is `neuroloc/wiki/tests/oracle_compression_analysis_results.md`, the frontier split is `neuroloc/wiki/synthesis/oracle_compression_frontier_split.md`, the first narrow learned-codec proof package is `neuroloc/wiki/synthesis/neural_model_dossier_compression_under_bit_budget_codec.md`, the tiny local mirror contract is `neuroloc/wiki/synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget.md`, the first local compact-state mirror is `neuroloc/wiki/tests/compression_under_bit_budget_mirror.md`, the first constrained message-response bridge is `neuroloc/wiki/tests/language_grounded_state_density_mirror.md`, and the local v1 / local 10k / local foundation artifacts are now explicitly demoted in `neuroloc/wiki/mistakes/local_foundation_lookup_scaffold_category_error.md`. the current full local nm result remains `neuroloc/wiki/tests/local_100k_full_nm.md`: one trainable local module, 81,070 hard maximum parameters, 24 learned latent bits plus 20 fixed bridge/schema/answer bits, 44 accounted bits, registered smoke and hard suite pass, and full control collapse on the deterministic exact-state 3d bridge. the current bounded compression product is `neuroloc/wiki/tests/local_100k_margin_recompression_adapter.md`: 4,096 exact answers from a stable four-domain source-heldout block, compressed payload carried inside torch module state, transformer and recurrent/state-style host integration, paraphrase-stable answer success 1.0, false-hit rates 0.0, random-label twin success 0.0, controls collapse 1.0, trained recompression update success 1.0, update-controller-disabled success 0.0, 295,144 strict accounted bits, 299,240 paper-surface accounted bits, adapter strict multiplier 22.732738950163952x, paper-surface strict multiplier 22.421639537059313x, source-holdout overlap repair, and public baseline stack pass 1.0. the high-density target remains unsolved because the latest win is still token-signature routing rather than learned semantic recall or implicit base-weight storage, the executable same-block content-scan diagnostic remains slightly stronger, and strict 600x pass remains 0.0. the dedicated compression research corpus lives at `neuroloc/compression/`; it records in-repo attempts, modern related work, open-source baselines, theory limits, baseline stack, and the next non-row compression path. these are local evidence only, not broad full-model authorization, arbitrary chat, paid-scale trainability, or external-simulator transfer. the paper-spine pair remains active: `neuroloc/wiki/synthesis/neural_model_paper_spine.md` and `neuroloc/wiki/synthesis/oracle_compression_analysis_plan.md`.

every proposed local-neuron, memory, addressing, interference, compression, reconstruction, replay, rollout, world-model, and trainability mechanism must have a concise proof package before any model code or intervention preset is accepted. the first hard symbolic test-material package is implemented as the `hard_symbolic_nm` suite. it covers belief-state formation, associative recall, correlated-key interference, delayed use, episodic reuse, context-gated routing, compression under bit budget, replay/rewrite, iterative rollout, and imagination/recombination. the first mechanism-specific gate is implemented as the `eligibility_commit` suite, covering delayed relevance, local commit, committed-distractor exposure, crossed commit/exposure splits, and symbolic compression frontier controls. the first `oracle_compression` hard profile covers 448 contracts across 14 families with operation preservation 1.0, controls preservation 1.0, leakage-free rate 1.0, accepted rate 0.5714, eight strong families, six weak families below 10x, and no global tiny-mirror recommendation. this is not a full paid model path.

paid compute remains paused. it can return only after broad lane research, mechanism dossiers, dossier-driven test material, oracle compression analysis where relevant, a tiny trainable mirror clears cpu controls, telemetry is complete, canonical state is prosecutor-clean, and one selected hypothesis is explicitly approved in `neuroloc/wiki/PROJECT_PLAN.md`.

the six-paid-run history is preserved below as evidence. older phase summaries should be read as historical snapshots, not as the active program surface; some older phase-3 conclusions were later qualified as confounded or vacuous in the bug history and canonical state files.

## prior phase: neural machine research (neuroloc)

## status: run3_cognition_phase1 complete (2026-04-17). val_bpb 6.3519 (plateaued at the alphabet prior from step 150 and never descended), passkey 0/100 at 256 and 1024 (partial eval — pod stopped by user before further eval). SIX paid runs all returning 0% passkey at 256 across two substrates (matrix and slot), two retention regimes (broken inherited and fixed alpha_log_mean=5.0), and now two corpora (fineweb-edu natural text and synthetic cognition). this run executed the discriminant proposed in `neuroloc/wiki/synthesis/training_objective_vs_architectural_goal.md`: a corpus where retrieval is explicit 50% of training. the article predicted that if phase one produced zero passkey on synthetic data, the substrate could not be trained by sgd at this configuration and the architecture needed deeper changes. phase one produced zero passkey. the architecture-cannot-be-trained-by-sgd branch fires. diagnosis is no longer "training objective mismatch" (that was the prior-run hypothesis) but "this substrate does not learn retrieval under sgd with these initialisations and this loss, even when the loss directly rewards retrieval". analysis at `neuroloc/wiki/synthesis/substrate_requires_architectural_change.md`, which ranks five candidate architectural interventions: (A) output gate init 0 instead of -4 (gate sigmoid(-4)=0.018 appears to be a fixed point of SGD, mean gate stayed at 0.018 throughout run 2; run-3 gate telemetry was not persisted, so the same regime is inferred rather than observed), (B) auxiliary retrieval loss on marker-following positions at 10-100x weighting, (C) orthogonal prototype key init to break one-slot softmax collapse, (D) warm start from hand-placed key/value pairs, (E) substrate replacement. these are not accepted for paid compute and must not be bundled by default; the approved next work is the six-lane master-plan research phase and local proof gates. the current local compression mirror now includes source diagnostics, sparse-read baselines, matched-budget sparse read, distributed evidence, an ordinary-split tiny model sanity pass, a shared-trunk factor-heldout failure, and a 10k legal-input factorized structured gate across four factor axes. another paid run with no substrate changes is strictly predicted to produce 0% passkey and must not be authorised. run card at `neuroloc/wiki/tests/run3_cognition_phase1_results.md`.

## prior status: run2_slot_memory_retention_fixed complete (2026-04-15)

val_bpb 1.4777, passkey 0/100 at 256 and 1024 (partial eval — pod stopped by user before passkey@4096 / selective_copy / structure_probe completed). five paid runs all returning 0% passkey at 256 across two substrates (matrix and slot) and two retention regimes (broken inherited and fixed alpha_log_mean=5.0). diagnosis at that point: the language-modelling training objective on fineweb-edu does not exercise the memory substrate, so gradient descent collapses it to inactive (mean output gate stayed at 0.018 throughout the 4000-step run) and never learns to use it. cpu gates a and b prove the slot mechanism can retrieve when addresses are placed by hand. the proposed discriminant (synthetic cognition corpus) was run as run3_cognition_phase1 and produced the same 0%, superseding this hypothesis with the architectural-change one above. first launch attempt of run2 fell through to the python recurrent loop because `flash-linear-attention` was missing from the pod (now pinned in `requirements.txt` commit `edcfe5d`, mistake at `neuroloc/wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md`). run card at `neuroloc/wiki/tests/run2_slot_memory_retention_fixed_results.md`. analysis at `neuroloc/wiki/synthesis/training_objective_vs_architectural_goal.md` (the article was updated 2026-04-17 with the run3 result; its reasoning structure is correct, its predicted discriminant came back negative).

## prior status: run2_slot_memory first launch (2026-04-15)

val_bpb 1.5107, passkey 0/100 at every tested length. four paid runs at 0% passkey before the retention-fix relaunch. run 2 first-launch failure was a self-inflicted config bug — the slot preset inherited `alpha_log_mean=-0.5` from config defaults, reproducing the exact state-evaporation failure mode documented in `neuroloc/wiki/synthesis/linear_attention_retrieval_wall.md` four days earlier by the same author who built the slot preset. mistake documented in `neuroloc/wiki/mistakes/run2_slot_memory_decay_copy_paste.md`. fix committed as `7abb781`. structural guard `_assert_preset_retention_safe` enforces explicit `alpha_log_mean` at config resolution so the bug class is impossible to reintroduce by silence.

## god_run_v2: god_machine.py re-run with all 17 F + 14 G prosecutor fixes (2026-04-12)

h200, 283m params, 4000 steps on fineweb-edu byte-level, 131m tokens, 59 min runtime.
all 17 F1-F17 + 14 G1-G14 prosecutor findings fixed before launch. critical fix: F1
replaced `torch.sigmoid(log_alpha_eff)` with `torch.exp(log_alpha_eff)` so recurrent path
matches FLA's `exp(g)` convention for log-space gates (prior broken math gave ~28% wrong
alpha_eff at default init). F2 gated `running_state_norm` EMA on `self.training`. F3/F5
merged spike_stats and val_result into jsonl via loop (fixed the class-wide regression of
the step-logger bug). F10 low-rank factorized imag_filter/pc_head (d→64→d). F11 uint8
ByteDataset. 12 more findings covering smoke tests, resume correctness, load errors,
dead code, misleading metrics. cpu oracle verifies alpha_eff math passes before launch.

**training: final val_bpb 1.4453 (+0.050 vs v1's 1.3950), 0.404x vs transformer**

**eval: retrieval 0% at every tested length**

| task | result (100 trials) |
|---|---|
| passkey @256 | 0/100 (CI 3.7%) |
| passkey @1024 | 0/100 (CI 3.7%) |
| passkey @4096 | 0/100 (CI 3.7%) |
| copy @256 | 0/100 (CI 3.7%) |
| copy @512 | 0/100 (CI 3.7%) |
| copy @1024 | 0/100 (CI 3.7%) |
| copy @2048 | 0/100 (CI 3.7%) |

perplexity at length: 1.9254 → 1.8941 → 1.5373 → 1.4776 → 1.4192 (monotonic, attention
uses context). delta state structure probe: mean_structure_ratio 0.977, pairwise_cos
0.003 (statistically noise, not trained memory).

**verdict: F1 math fix was not the root cause. the tested 5-feature bundle still failed
verbatim memory.** external review (2026-04-12) identified 8 candidate contributing
mechanisms. the two strongest current review findings are: (1) k-WTA 20% on keys likely
destroys the address space (only ~13/64 key dims survive, Hopfield capacity drops from ~9
to ~2 patterns per head); (2) delta erasure with sparse keys likely leaves ghost content
in the zeroed dimensions. 6 supporting review findings: imagination probe may create a
gradient bypass that competes with the delta memory, BCM EMA half-life 69 steps may be too
slow, multi-compartment SwiGLU may reduce effective width, PC head may add loss drag,
FLA/recurrent numerical drift may compound during chunked eval, q/k normalization timing
differs between paths. preserved run card: `neuroloc/wiki/tests/god_run_v2_results.md`.

**historical path forward proposed on 2026-04-12, superseded by later runs and the 2026-04-27 master-plan gate**:

the bullets below are retained as historical context only. they do not authorize h200, pod, kaggle, paid compute, full-model integration, or model preset work.

- **run 1 baseline dense**: `god_machine.py` preset `run1_baseline_noerasure`; dense k/v,
  no delta erasure, non-FLA path, no BCM, no multi-compartment, no imagination, no PC head.
  prelaunch gate: short h200 timing/oom benchmark at full batch/seq on the recurrent path. the full launch now refuses to start without the recorded benchmark manifest in the same output root, rejects resumed benchmark replays, locks the official full run to the canonical 4000-step config, requires the current device to match the benchmarked hardware profile, binds the gate to the recorded benchmark artifacts and git working-tree fingerprint, and forbids FineWeb fallback under the canonical preset name. this is a local structural/provenance guard, not cryptographic attestation.
  hard retrieval gate after the full run: nonzero passkey at 256, persisted as `retrieval_gate`
  in the run results. inconclusive eval now fails closed instead of exiting zero.
- **if run 1 still returns 0% passkey**: run one slower static-retention ablation with erasure still off before declaring the base mechanism suspect. the standalone decay sweep at `d_head=64` only reopens exact-query 32-pattern recall around `decay=0.90`, while the current static init starts much lower.
- run 2: implement and smoke a value-only 50% k-wta preset before launch. current checked-in `god_machine.py` still sparsifies keys and values together when k-wta is enabled.
- run 3: after run 2, implement and smoke bcm controls for momentum 0.95 and init `running_state_norm=0.01` before launch. current checked-in code still hardcodes the old 0.99/1.0 behavior.
- run 4: reintroduce erasure only as an explicit post-baseline ablation after dense-key retrieval is validated
- not added until validated: imagination, multi-compartment, PC head, any k-WTA on keys

## god_run: god_machine.py first run (2026-04-11)

h200, 283m params (282,953,496), 4000 steps on fineweb-edu byte-level, 131,072,000 tokens, seed 42.
all 5 blueprint features active: k-wta 20% rate-coded compression, delta-rule erasure,
bcm-adaptive alpha (gamma=0.3), multi-compartment swiglu (k=4), compressed attention via sdpa.
plus always-on imagination probe (learned query into delta state with gated residual, now
low-rank factorized to ~131k params per layer) and per-layer predictive coding diagnostic head.

val_bpb 1.3950 (final). bpb_ratio 0.390x vs 3.58 transformer baseline at matched pipeline.
smooth monotonic decrease: 2.381 → 2.07 → 1.94 → 1.87 → 1.81 → 1.75 → 1.71 → 1.67 → 1.62 →
1.57 → 1.52 → 1.50 → 1.48 → 1.46 → 1.44 → 1.43 → 1.42 → 1.41 → 1.40 → 1.3950.
training loss final 0.9535 at step 3950. throughput ~45,500 tok/s steady. total 3166s (~53 min).
firing rate 0.200 exactly throughout (k-wta target met). no dead neurons.

**retrieval failed at every tested length (n=20 per cell, 95% wilson upper ~14%):**

passkey  @256: 0/20  passkey  @1024: 0/20  passkey  @4096: 0/20
copy     @256: 0/20  copy     @512:  0/20  copy     @1024: 0/20  copy @2048: 0/20

perplexity-at-length (monotonic decrease, attention path uses context):
bpb@256=1.9354  bpb@512=1.8437  bpb@1024=1.4909  bpb@2048=1.4110  bpb@4096=1.3751

delta state structure probe (closed-gate readout with novel keys, NOT image generation):
mean_structure_ratio=0.981, mean_pairwise_cos=-0.003, random_pairwise_cos=0.000.
state is near-orthogonal across 24 delta layers. this is NOT the structured-interpolation
signature (cosine ~0.93) that exp_008 reported. the delta memory accumulated high frobenius
norm but pairwise-orthogonal state that functions as noise, not content-addressable storage.

**diagnosis**: the compressed-attention+mlp path learned to fit the next-byte distribution
(bpb 1.395). the delta-rule memory state is noise. k-wta 20% + delta erasure + bcm alpha +
imagination probe combined so that verbatim retrieval was destroyed while statistical
distribution-fitting worked. this is exactly the lossy-mechanism failure mode that
`neuroloc/wiki/synthesis/compression_beyond_quantization.md` predicts: preserved statistical fit,
destroyed verbatim memory.

**17 prosecutor findings F1-F17 applied before re-run:**

- f1 (p0): bcm train/eval path divergence. recurrent path computed live-state alpha_eff
  per timestep; fla path used running_state_norm buffer. fixed by aligning recurrent path
  to use `_effective_log_alpha` which reads running_state_norm.
- f2 (p0): running_state_norm buffer mutated during eval. fixed with self.training gate.
- f3 (p1): history dict was a second cherry-pick site (class-level regression of the
  step-logger bug). fixed via setdefault loop that merges all spike_stats keys.
- f4 (p1): collect_god_metrics produced length-inconsistent per_layer arrays. fixed by
  pre-allocating [None]*n_layers lists.
- f5 (p1): other metrics_logger sites still cherry-picked val_result. fixed via loop merge.
- f6 (p1): passkey/copy 20 trials was too few (95% wilson upper 14% for 0/20). fixed to 100.
- f7 (p2): load_state_dict strict=False swallowed errors. fixed to raise on missing keys.
- f8 (p2): compartment guard was dead. fixed to check correct aux key name.
- f9 (p2): smoke-test required_god_keys was a hardcoded 9-key list. fixed to derive from aux.
- f10 (p2): imag_filter 1m params * 24 layers = 24m always-active dead weight. fixed to
  low-rank factorization (~131k params/layer). same for pc_head. added post-val warning
  if imag_ratio_mean < 0.02.
- f11 (p2): bytedataset materialized int64 (8x ram). fixed to keep uint8, cast per-slice.
- f12 (p2): no resume correctness test. added _test_resume_correctness() to smoke_test.
- f13 (p3): dead _parallel_no_erasure code. removed; all non-fla paths go through recurrent.
- f14 (p3): non-causal attn_entropy probe with T<=512 gate. removed.
- f15 (p3): hardcoded zero dead_pct/saturated_pct/per_layer_dead_count. removed.
- f16 (p3): dead imports (field), SEED constant, TernaryQuantizer, AdaptiveSpike. removed.
- f17 (p3): per-timestep state.norm O(H*D*D). fixed via f1 (alpha_eff hoisted).

local working-tree note: the reviewed `god_machine.py` currently contains a reintroduced, unused `TernaryQuantizer` / `AdaptiveSpike` pair near the top of the file. that is outside the selected run-1 launch surface and not part of the archived f16 fix record above.

**renamed**: run_imagination_test → run_delta_state_structure_probe. "imagination" was a
misleading metaphor for what is actually a state-structure probe; the byte-level text model
has no image generation capability.

**class-level telemetry gate**: smoke_test now writes collect_god_metrics output through a
real MetricsLogger to a tempfile, reads it back via json.loads, and asserts every enabled
feature's aux keys round-trip through disk. any future regression that drops metrics fails
smoke test before launch. required_god_keys is derived programmatically from aux.

the original local output directory was ephemeral. preserved run card:
`neuroloc/wiki/tests/god_run_results.md`.

**what happened next**: `god_run_v2` re-ran `god_machine.py` with all F1-F17 fixes and full
telemetry. the result stayed at 0% retrieval, so this archived branch is closed; the live
next-step plan is the no-erasure baseline summarized near the top of this file.

## phase 5: runs 010-011 complete

### run_011: full architecture (KDA+Mamba3+MLA, 6:1:1)

h200, 280m params, full architecture (18 KDA + 3 Mamba3 + 3 MLA).
bpb 2.592, ratio 0.722x (27.8% better than transformer baseline).
spike mi 1.246, cka 0.802. fr measurement bug from gradient
checkpointing (reported 0%, actual ~41%). 3/4 gates pass.
mamba3 sequential scan dominates training time (~10s/step overhead).
atmn too slow at 280m with gradient checkpointing (30+ min/step).

### run_010: KDA+MLA only (no Mamba3)

h200, 267m params, kda+mla only (no mamba3). fla disabled (chunk_kda NaN at
d_model=1024). mamba3 dropped (sequential scan ~15s/step at T=2048, not viable
without parallel scan kernel).

Within this incomplete KDA+MLA probe, bpb was 2.375 and the ratio was 0.663x
against its parameter-matched Transformer control. The run omitted Mamba3 and
does not validate the full historical hybrid or the current modular target.
Spike health was mi=1.168, cka=0.732, fr=40.8%.

| gate             | threshold                    | run_010 result                  | status |
|------------------|------------------------------|---------------------------------|--------|
| bpb_ratio        | < 1.0 (beat transformer)     | 0.663x (2.375 vs 3.583)        | pass   |
| spike_mi         | > 0.1                        | 1.168                           | pass   |
| spike_cka        | > 0.3                        | 0.732                           | pass   |
| spike_fr         | 30-60%                       | 40.8%                           | pass   |

next: fix fla chunk_kda NaN at d=1024, add mamba3 parallel scan, re-run phase
5 baseline with full 3:1 architecture.

## phase 3: complete -- all gates pass

## Phase 2: COMPLETE -- 2/3 gates PASS (selective copy deferred to Phase 5)

## Phase 1: COMPLETE -- ALL GATES PASS

| Gate             | Result   | Status |
|------------------|----------|--------|
| bpb_threshold    | 0.840x   | PASS   |
| spike_mi         | 1.275    | PASS   |
| spike_cka        | 0.913    | PASS   |
| spike_firing_rate| 42.0%    | PASS   |

## Phase 2 Summary (run_008)

- Progressive BPB: 3.31 (256) -> 3.11 (512) -> 2.94 (1024) -> 2.82 (2048)
- Perplexity stability: +4.0% from 256 to 4096 (PASSES <20% gate)
- Selective copy: 0% at all lengths (FAILS >60% gate -- deferred to Phase 5)
- BPB ratio vs transformer: 0.780x (Todorov better)

## Phase 2 Revised Gates (6M scale)

| Gate                | Threshold                            | run_008 Result                  | Status  |
|---------------------|--------------------------------------|---------------------------------|---------|
| selective_copy_1k   | >60% at 1024 token distance          | 0% at all lengths               | FAIL    |
| perplexity_stable   | <20% BPB increase from 256 to 4096   | +4.0% (3.962 -> 4.121)          | PASS    |
| mla_cache_linear    | Cache scales linearly with context   | Not yet measured                | PENDING |

## Phase 3: COMPLETE (run_009)

GP self-interaction enabled in SwiGLU (spatial_mode=True).
Training on mixed data: 50% WikiText-2 + 25% 3D shape + 25% n-body.

| Gate               | Threshold                              | run_009 Result                         | Status |
|--------------------|----------------------------------------|----------------------------------------|--------|
| spatial_classify   | outperform Transformer                 | GP 30.0% vs Transformer 25.0%         | PASS   |
| spatial_dynamics   | lower MAE than Transformer             | GP MAE=51.55 vs Transformer MAE=72.70  | PASS   |
| equivariance_test  | <5% error at 60-degree rotation        | error=1.34e-07 at 60 degrees           | PASS   |
| language_no_degrade| BPB not degraded >10% with GP enabled  | -18.8% (GP improves language)          | PASS   |

## run_009 Results Summary

Training (Kaggle T4, mixed data, GP enabled):

| Metric             | GP (Todorov)  | No-GP         | Transformer   |
|--------------------|---------------|---------------|---------------|
| Shape classify     | 30.0%         | --            | 25.0%         |
| N-body MAE         | 51.55         | --            | 72.70         |
| Equivariance err   | 1.34e-07      | --            | --            |
| Language BPB       | 3.009         | 3.707         | --            |
| Training time (s)  | 1,451         | 573           | 42            |
| Params             | 6,015,780     | 5,917,476     | 5,705,984     |

Per-class shape classification:

| Class        | GP     | Transformer |
|--------------|--------|-------------|
| sphere       | 80%    | 100%        |
| cube         | 0%     | 0%          |
| tetrahedron  | 16%    | 0%          |
| torus        | 24%    | 0%          |

Spike health: MI=1.311, CKA=0.907, FR=42.1%, dead=0%
Total wall clock: ~38 min on Kaggle T4

Bug found during implementation: src/layers/swiglu.py spatial_mode had latent
shape mismatch (GP output d_model added to hidden_dim tensor). Fixed in train.py:
GP residual applied after down projection.
