# Modular neural-machine local training pause handoff

Status: paused by Deyan Todorov on 2026-07-22 before a corrected MLX pilot or claim training began.

Tenth resume addendum and final session stop, 2026-07-26: Deyan stopped the session before attempt 4. No attempt-4 run root, training-start request, claim model update, or optimizer update exists. The post-attempt-3 sampler and deadline repairs reached `844 passed, 2 skipped` in `220.75` seconds on the pre-interruption modular surface. A final reviewer then found that production clean-claim order and the machine-readable contract disagreed. The contract-sync agent partially updated the payload and run card before being interrupted at Deyan's stop instruction. The modified payload currently hashes to `8c7825f69fd27a7f3653c2e3bfab8673f3bb13d9f543fecd4a6aa9b97a4868ab`, while launch constants and the run-card digest still contain `fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553`. The stopped bytes are unverified, the four active attestations are stale, and the runner must refuse launch. The complete final recovery record is `docs/MLX_SESSION_STOP_HANDOFF_2026-07-26.md`.

Resume addendum, 2026-07-26: Deyan resumed the result-first path. The sampler-stop in-flight-row race was reproduced with a deterministic failing test and repaired at the shared condition boundary. The preregistration was synchronized to the implemented post-claim-data, immediately-pre-spawn second swap sample, and its canonical digest became `49c87c1b097be24b8e27c441b4ddb796cbecde63435c4f106eff9fcb73fdf6b5`. Focused tests passed. Complete unchanged-surface verification, literal-zero review, new attestations, the corrected pilot, and training remain pending. The remainder of this document preserves the exact historical stop boundary rather than rewriting it as current status.

Second resume addendum, 2026-07-26: Two governed attempts occurred after the first addendum. Both serialized `48` passing preflight records, but later complete-surface review proved the parity gate incomplete and recorded hidden and sequence-delta errors above the frozen `1e-5` limit. Attempt one stopped before pilot update one on parent-child execution-label drift. Attempt two reached durable counters of `132` attempted pilot updates and `292,864` attempted positions before parent tail-schema validation aborted without publishing `run/pilot.json`; those counters do not establish completed governed work. The tail validator also remained permissive beyond the first list-shape repair. No training-start request or claim update occurred. The current recovery point is `neuroloc/wiki/PROJECT_PLAN.md`: complete both bug classes test-first, verify unchanged bytes, reach literal-zero review, issue fresh attestations, and only then attempt a new preflight and pilot.

Third resume addendum, 2026-07-26: The fixed-tail producer and validator now bind exact deterministic fixtures, the complete `438,368`-byte evaluation fixture, current-model checkpoint lower bounds, selected-final clone state, current engine identity, and observed scratch lifecycle. Eight fresh Metal processes calibrated and froze complete forward and five-lane raw-gradient limits; two untouched held-out forward cases passed those limits. Independent float64 optimizer formulas with a priori bounds replaced false exact cross-runtime equality, and a different-gradient second update passed with nonzero carried moments. The full MLX regression file reports `75 passed, 2 skipped`, four fresh complete engine-to-parent validations pass on `Device(gpu, 0)` with worst-bound ratio `0.95367431640625`, and complete modular verification reports `823 passed, 2 skipped` in `167.81` seconds. The machine-readable method digest is `560ec821ca0cba93b828311ffa7de788a29d73a8e2d424ed35d6b750cc80598c`. Literal-zero review, fresh attestations, assertions, and a new-run-ID pilot remain pending. No claim training has started.

Fourth resume addendum, 2026-07-26: The first fresh review found and closed runtime-bound source-exclusion self-certification, unbound gradient-tolerance metadata, checkpoint-cleanup timing exclusion, scheduler-dependent sampler testing, and cleanup masking of primary pilot or claim failures. The full MLX regression file now reports `77 passed, 2 skipped`, and complete modular verification reports `825 passed, 2 skipped` in `230.18` seconds. Literal-zero re-review, fresh attestations, assertions, and a new-run-ID pilot remain pending. No claim training has started.

Fifth resume addendum, 2026-07-26: Final unchanged code-and-test verification repeated at `825 passed, 2 skipped` in `215.47` seconds. The canonical preregistration digest remains `560ec821ca0cba93b828311ffa7de788a29d73a8e2d424ed35d6b750cc80598c`. Literal-zero re-review, fresh attestations, assertions, and a new-run-ID pilot remain pending. No claim training has started.

Sixth resume addendum, 2026-07-26: The second review cycle closed scratch creation and cleanup leakage, cleanup-time existence-error masking, unbounded or partial child transport reads, coalesced-message preservation, and the absent absolute pilot deadline. One non-resetting `1,200`-second pilot deadline now covers child start, every receive and acknowledgment, close, and join. The full MLX file reports `82 passed, 2 skipped`, and complete modular verification reports `830 passed, 2 skipped` in `217.01` seconds. Literal-zero re-review, fresh attestations, assertions, and a new-run-ID pilot remain pending. No claim training has started.

Seventh resume addendum, 2026-07-26: Buffered complete child messages now recheck the absolute deadline and resource sampler before admission, so coalesced output cannot acknowledge or start another update after either guard fails. The full MLX file reports `83 passed, 2 skipped`, and complete modular verification reports `831 passed, 2 skipped` in `219.63` seconds. Literal-zero re-review, fresh attestations, assertions, and a new-run-ID pilot remain pending. No claim training has started.

Eighth resume addendum, 2026-07-26: Every scheduler-dependent sleep is removed from the MLX test file. Resource-sampler failure and background-writer abort tests now use explicit worker-entry, cancellation, release, and join barriers, and an AST regression requires zero `time.sleep` calls. The full MLX file reports `84 passed, 2 skipped`, and complete modular verification reports `832 passed, 2 skipped` in `219.61` seconds. Literal-zero re-review, fresh attestations, assertions, and a new-run-ID pilot remain pending. No claim training has started.

Ninth resume addendum, 2026-07-26: Four literal-zero attestations admitted governed attempt `mlx-m5pro-20260726-3`. Its current preflight passed and its child executed all `132` pilot updates, but an in-flight resource read committed a zeroed exited-child row with stale active-job state. Terminal sampling failed and abort finalization then rejected decreasing process CPU, so neither `run/pilot.json` nor `ABORTED.json` was published and no claim training began. A lifecycle-generation barrier, zeroed-child rejection, durable parent-only terminal sample, and legacy abort-timeline rule now have deterministic regressions. The full MLX file reports `84 passed, 2 skipped`, focused CPU resource and abort verification reports `77 passed`, and complete modular verification reports `834 passed, 2 skipped` in `214.05` seconds. Literal-zero review and four fresh attestations are required before attempt 4.

This document is the exhaustive recovery record for the stopped Transformerov, Monodratic, MLX, and local-training work. It is evidence and a handoff, not a competing project plan. `neuroloc/wiki/PROJECT_PLAN.md` remains canonical. Any later disagreement must be resolved in the canonical plan first.

## Executive status

The requested working model has not been trained. The current repository contains a substantial, tested implementation of the first modular neural sequence-role proof base and a compiled MLX execution path, but the launch gate is not closed.

At the stop boundary:

- No corrected MLX resource pilot had started.
- No governed training-start publication had occurred.
- No claim-training model or optimizer work had begun.
- No review attestation existed under `neuroloc/results/modular_sequence_role_mlx_reviews`.
- All remaining collaboration agents were interrupted or had already completed.
- The last complete-surface review was interrupted and produced no authorization.
- Two unresolved launch blockers remained: a sampler-stop race and a preregistration/implementation handshake-order disagreement.
- The most recently discussed `590.761`-second value was invalid. It was an after-the-fact sum of measured components, not a reached full-package runtime, and it omitted mandatory lifecycle work.
- The repository remains a mixed dirty working tree with user and historical changes. Nothing was staged, committed, reset, deleted, or pushed as part of the stop.

The correct status is therefore: implementation and evidence work are advanced; training is not launched; the current bytes are not authorized for launch; the scientific deliverable remains incomplete.

## Stop actions and process boundary

The active review and acceleration agents were stopped when Deyan requested a pause to conserve limits. The agent tree at the stop boundary contained no running subagent other than the coordinator, and the coordinator performed documentation only after the stop.

An operating-system process-list query could not be completed inside the managed sandbox. The attempted `pgrep` inspection returned `sysmond service not found` and `Cannot get process list`. Therefore this record does not claim independent operating-system proof that no unrelated or previously detached process exists. What is established is narrower: no pilot or claim child was launched by the final correction cycle, no training-start transaction was published, and all visible collaboration agents were stopped.

On resume, inspect the operating-system process table before any new run. Do not assume this handoff proves host-wide process absence.

## Deliverable and claim boundary

The requested deliverable is a working neural model trained on this MacBook, not a plan, a router, a lookup table, a command-line responder, or a toy Python object.

The current first-stage deliverable is narrower than the final desired system. It is a modular neural sequence model that must prove two roles in one matched package:

1. Routed selected-set attention must support exact nonlocal recall after learned routing.
2. Recurrent neural state must support world-state tracking across time.

The first-stage result must survive causal knockouts, matched random and local controls, dense-attention controls, seed replication, confidence intervals, reload checks, route evidence, and complete artifact accounting. Passing this base gate would establish a trainable composition boundary. It would not by itself establish a 3D world model, imagination, dreaming, broad language ability, high-density knowledge compression, or general intelligence.

The intended progression after the base gate is:

1. Train and validate the unchanged routed-recall plus recurrent-tracking base.
2. Add a carried-state 3D latent object-slot world model as an actual neural mechanism, not a metadata wrapper.
3. Add learned latent branch rollout and recombination for imagination and dreaming, with dynamics, uncertainty, provenance, and controls.
4. Test the nested-reciprocal token-local feature mixer as one isolated intervention.
5. Retain only mechanisms that pass matched causal and efficiency gates.

The final system is therefore not intended to move back toward a Transformer. Dense attention remains a named control only. The design separates exact recall, recurrent world tracking, and token-local feature mixing so each claim can be proved or rejected independently.

## Concrete architecture currently implemented

The implementation is a neural network with learned parameters and differentiable internal dynamics. It is not a record table or scripted responder.

The fixed base geometry is:

- Model width: `64`.
- Blocks: `8`.
- Heads: `4`.
- Recurrent head width: `16`.
- Recurrent chunk length: `32`.
- Routed blocks: `0` and `4`.
- Recurrent blocks: `1`, `2`, `3`, `5`, `6`, and `7`.
- Sequence schedule: routed, recurrent, recurrent, recurrent, routed, recurrent, recurrent, recurrent.
- Rung-one vocabulary and sequence length: `128` and `128`.
- Rung-two vocabulary and sequence length: `256` and `512`.
- Rung-one query position: `126`.
- Rung-one carry position: `96`.
- Route block size: `8`.
- Local route blocks: `1`.
- Routing width: `16`.
- Routing subspaces: `2`.
- Routing codes per subspace: `4`.
- Routing probes: `4`.
- Routing bucket capacity: `64`.

Each block follows the host-owned normalization and residual contract:

```text
u = x + sequence_mixer(RMSNorm(x))
y = u + feature_mixer(RMSNorm(u))
```

The public Monodratic selected-set mixer supplies the routed sequence delta. Transformerov supplies the recurrent sequence mixer. The host retains normalization and residual addition. Every mixer presents the same `[batch, time, width] -> delta` seam. Disabling one module does not activate another, and dense attention is not a hidden fallback.

The recurrent mixer maintains a learned matrix-valued neural state per layer with shape `[batch, 4, 16, 16]`. Its operative recurrence is:

```text
S_t = alpha_t * (S_(t-1) - beta_t * k_t * (k_t^T * S_(t-1))) + beta_t * k_t * v_t^T
o_t = q_t^T * S_t
```

This state is updated within the sequence and returned as a sequence delta. The current proof base does not yet establish a general external carry-state interface.

The implemented feature path is the unchanged source feature mixer. The published nested-reciprocal CoFrGe-style component is documented but intentionally deferred. Adding it before the base composition is established would change more than one scientific variable.

The previously recorded parameter counts are:

- Rung one: `574,160` parameters.
- Rung two: `590,544` parameters.

These counts describe the small proof models, not a claim that scale alone is sufficient for the final desired system.

## Why this is not yet the requested 3D world model

The recurrent state is a real neural state and is used for time-dependent task computation. However, it has not yet been trained or validated as an explicit 3D world representation.

A valid 3D world-model addition must introduce and test at least:

- Latent object or scene slots with persistent identity.
- Learned writes from observations into those slots.
- Learned reads conditioned on the current query or action.
- Pose, geometry, relative position, occlusion, and persistence variables or an equally explicit learned alternative.
- Transition dynamics that predict how the world state changes under time and action.
- Reconstruction or prediction heads that make the latent state behaviorally testable.
- Partial-observation, occlusion, object-permanence, viewpoint-change, and counterfactual controls.
- Provenance and uncertainty for state updates and rollouts.
- Causal deletion tests showing that the proposed 3D state, rather than a shortcut, produces the gain.

Calling the current sequence-role scaffold a 3D world model would be an overclaim. It is the neural base intended to make that later addition technically and scientifically interpretable.

## What imagination and dreaming must mean here

Imagination cannot be a copied record with a branch label. Dreaming cannot be replaying an earlier row unchanged.

The later imagination mechanism must be a learned latent branch rollout or recombination process with:

- A learned world-state transition.
- Branch-local state rather than mutation of the factual base state.
- Multiple candidate futures or recombined scenes.
- Calibrated uncertainty and provenance.
- A decoder or decision surface that exposes consequences of the branch.
- Evidence that branch rollouts improve hard-case prediction, reconstruction, planning, or counterfactual action more than easy cases.
- No-branch, random-branch, shuffled-branch, wrong-dynamics, and matched-compute controls.

Replay must be delayed neural reactivation that changes later recall, interference, rewriting, compression, or action success. It must beat no-replay and random-replay controls under matched compute.

These requirements are preserved so a later implementation cannot satisfy the request with toy branch objects or hand-written templates.

## Frozen scientific workload

The full base package was deliberately fixed before acceleration work. The acceleration method is not allowed to earn the time target by silently shrinking the science.

The full package contains:

- `20,736` logical training updates.
- `41,472` attempt-event rows.
- `45,613,056` token positions.
- Rung-one construction and claim seeds `11`, `23`, `37`, `53`, and `71`.
- Rung-two construction seed `83`.
- `26` trained endpoint checkpoints and paired MLX/Torch endpoint replays.
- `124` registered gates.
- `588,240` route-evidence rows.
- `522,240` training route rows.
- `66,000` evaluation and acquisition route rows.
- Five rung-one selected-set lanes, five dense-control lanes, donor stages, router-only stages, joint stages, dense continuation, and the rung-two tracking task.
- Causal knockouts, local-only, random-route, forced-target, dense/all-eligible, carry reset, carry shuffle, source exclusion, route parity, checkpoint reload, and artifact-closure controls.

No seed, update, task, control, evaluation condition, evidence row, checkpoint, or gate may be removed to meet the runtime target. If the complete package projects above twenty minutes, the decision is to stop and improve the execution method, not to report an easier experiment.

## CPU implementation and preserved failed pilot

The first implementation used Torch CPU as both execution and authority. It established the model, generators, controls, evidence schemas, and lifecycle machinery, but was too slow for the required local turnaround.

The preserved pilot root is:

`neuroloc/results/modular_sequence_role_cpu/base-m5pro-20260722-1`

That pilot completed:

- `88` pilot updates.
- `225,280` pilot token positions.

It did not enter claim training.

The original path projected roughly seven hours. Its exact scalar was not retained because terminal pilot closure rejected its own otherwise valid artifacts: `run/pilot.json` referenced check-detail digests that the closure collector had omitted. The provisional pilot record was removed and no final terminal checksum was produced.

The same investigation found quadratic growing-prefix work in JSONL append validation and per-event validation. Both failures were documented as method and lifecycle defects rather than treated as scientific results.

The repair then:

- Collected all referenced pilot detail records during closure.
- Replaced growing-prefix validation with incremental crash-atomic attempt state.
- Preserved digest-backed rollback behavior.
- Replaced the public nested selected-attention loop with an exact gathered selected-ID implementation.
- Preserved causal, underfill, forward, gradient, and no-dense-fallback semantics.

The selected-attention component medians improved by:

- `42.13x` at `B16/H4/T128/D16`.
- `51.28x` at `B8/H4/T512/D16`.

These are component benchmarks only. A corrected full-step CPU diagnostic still projected about `3.12 h`. The CPU execution path was therefore not launchable under the twenty-minute hard gate.

## MLX execution path

MLX `0.29.3` with Metal was selected as the execution substrate on the M5 Pro. Torch `2.8.0` float32 CPU remains the authority for source tensors, deterministic initialization, checkpoint semantics, evaluation reductions, gates, and artifact validation.

The intended process layout is one parent and one MLX child:

- Donor seeds execute individually.
- The five rung-one router-only lanes execute through compiled width-five vectorization.
- The five rung-one joint lanes execute through compiled width-five vectorization.
- The five dense-base lanes execute through compiled width-five vectorization.
- The five dense-continuation lanes execute through compiled width-five vectorization.
- Rung two executes as one lane.

The MLX child is not allowed to define the scientific result independently. Complete outputs and required telemetry are transferred back to Torch CPU. Initial parity and all `26` trained endpoints must pass Torch replay before their applicable completion acknowledgment.

## MLX parity evidence achieved before the pause

The following narrow implementation evidence was obtained:

- MLX selected-attention logits matched Torch within approximately `5.275e-6` maximum absolute error.
- Selected route identities were exact.
- A `119`-tensor state mapping between Torch and MLX was bijective.
- AdamW update comparison had zero recorded error in the narrow check.
- Width-five vectorization preserved lane independence in the checked path.

The full trained-endpoint parity contract requires:

- Parameters exact.
- Raw, effective, and address routes exact.
- Optimizer parameter identity exact.
- Optimizer first and second moments exact.
- Optimizer step exact.
- Logits, hidden state, and sequence deltas within `1e-5`.
- Total and component loss within `1e-6`.
- Gradient parity passing either the primary absolute gate or the fixed scale-aware fallback.

The gradient gate is:

```text
primary: maximum absolute error <= 1e-5

or all fallback conditions:
maximum absolute error <= 3e-5
maximum per-tensor relative error <= 1e-4
maximum normalized L2 error <= 5e-5
minimum cosine >= 0.999999999
```

The fallback was introduced only after fixed-process diagnosis showed that an absolute-only threshold rejected numerically equivalent Metal gradients at large tensor scales. It is conjunctive and preserves exact optimizer state and route requirements.

The fixed endpoint study recorded a worst maximum absolute gradient error of `1.887977123260498e-5`, maximum normalized L2 error through `4.2887810898694334e-5`, minimum cosine `0.9999999993362688`, total/component loss error `4.76837158203125e-7`, exact routes, and exact optimizer moments and step.

Ephemeral study artifacts were:

- `/private/tmp/todorov-all-endpoint-gradient-study/report.json`
  - SHA-256: `7f2f359a0d9d70f9133804b78cad2e50060aa2da433884e960cf3a0e4998ec06`
- `/private/tmp/todorov-fixed-gradient-study.json`
  - SHA-256: `6fa91618156ce46626e2a556e0e7f27a082a7ea65bb894a085aa9556a4f0f881`

These paths are under `/private/tmp` and may disappear. Their absence on resume does not convert their recorded findings into final claim evidence. Reproduce the governed checks when required.

## Numerical research used for the parity decision

The parity decision was grounded in official numerical behavior documentation:

- MLX compilation documentation states that compiled and uncompiled functions are expected to agree to numerical precision rather than bitwise identity: <https://ml-explore.github.io/mlx/build/html/usage/compile.html>
- MLX function transforms document vectorization behavior: <https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html>
- PyTorch documents floating-point non-associativity and possible CPU/GPU differences: <https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html>

The inference from those sources was limited: backend arithmetic may differ at floating-point scale even when the implemented equations are equivalent. That does not justify a loose gate. The project therefore retained exact route, parameter, optimizer, and identity checks and used tightly bounded multi-metric evidence only for gradients.

## Measured timing components and why the total is invalid

The following component measurements were assembled during the MLX acceleration work:

| Component | Measured seconds | Full-package multiplier | Charged seconds |
| --- | ---: | ---: | ---: |
| Cold child start | `1.265085042` | `1` | `1.265085042` |
| Cold compile | `0.571052710` | `1` | `0.571052710` |
| Donor step | `0.039103994625` | `5,120` | `200.21245248` |
| Selected width-five step | `0.091444625125` | `1,280` | `117.04912016` |
| Dense width-five step | `0.07544702075` | `1,536` | `115.886623872` |
| Rung-two step | `0.053709244625` | `1,536` | `82.497399744` |
| Durable attempt ledger | `5.255333784` | `1` | `5.255333784` |
| Routing evidence | `9.485348288` | `1` | `9.485348288` |
| Evaluation and endpoint replay | `53.856777124` | `1` | `53.856777124` |
| Checkpoint save and reload | `3.051925794` | `1` | `3.051925794` |
| Packaging | `1.63014041978125` | `1` | `1.63014041978125` |

The arithmetic sum was `590.76125941778125` seconds. A displayed implementation value rounded one unit in the last place differently, ending in `...7813` rather than `...7812`.

This total is invalid as a launch projection and was never an achieved runtime. It omitted production-only behavior:

- Periodic resource-sampler lifecycle interaction.
- Real child `close_committed` acknowledgment.
- MLX runtime and Metal cache release on exit.
- Parent wait for clean exit.
- Child join.
- Child stderr flush and fsync.
- Parent stderr descriptor close.
- Actual owned scratch-tree cleanup.
- Verification that owned scratch was absent.

The private synthesis was assembled after the component work rather than emitted by the governed pilot:

- `/private/tmp/todorov-mlx-measured-projection-20260722.json`
- SHA-256: `9c73ff1563cf4a0282843434d7f711aa4bc6078835a171dc069fb4529d5a2083`

It must never be cited as the runtime reached by the model or as authorization to train.

## Projection-integrity correction that was in progress

The method was amended to account for the missing lifecycle work without reducing the scientific package:

- Training no longer waits up to five seconds at each stage solely to observe a periodic sample.
- The sampler runs concurrently during training.
- Exact nonempty per-seed resource coverage is checked after training and before evaluation.
- Active jobs are cleared for the fixed evaluation and closure tail.
- The real child close, exit, join, stderr, and owned-cleanup path becomes a twelfth measured projection component.
- Packaging charges a fifth content-addressed tail assertion for that component.
- The final resource row must already be durable, have no active jobs, and carry exact final-start counters.

No corrected pilot ran after this amendment. Consequently there is no current valid full-package projection.

## Unresolved blocker 1: sampler-stop race

The sampler-stop implementation does not yet prove the preregistered meaning of “reuse the latest already durable row.”

The current thread performs a resource sample outside the condition lock and appends it after the process and swap reads complete. The parent can set the stop event while that sample is already in flight. `stop()` then joins the thread and may accept the newly appended row as the final row.

That behavior contradicts the stated contract that stop must reuse a row that was already durable when stop began, without waiting for or taking another sample. It also means the supposedly uncharged final barrier can silently include the remainder of an in-flight sample.

Required repair:

1. Add a deterministic fault-injection test that blocks an in-flight sample between its external reads and durable append.
2. Begin sampler stop while that sample is blocked.
3. Prove the stop path either rejects the in-flight row or defines and charges a different explicit lifecycle contract.
4. Preserve the final exact counters, no-active-job identity, durability, writer acknowledgment, and failure propagation.
5. Search all pilot and claim stop call sites for the same bug class.
6. Re-run the focused lifecycle suite and the complete changed-surface suite.

No training or corrected pilot may launch before this is resolved and reviewed to zero findings.

## Unresolved blocker 2: handshake-order disagreement

The machine-readable preregistration currently says:

`claim_second_sample: immediately_before_MLX_child_stage_release`

The run card and implementation place the second swap sample before worker spawn, where it detects construction-time swap growth.

Those are different lifecycle points. The preregistration sentence must be synchronized with the actual intended method. This is a documentation/evidence-contract error, not permission to move the code casually.

Required repair:

1. Decide from the already established method whether the second swap sample belongs before child spawn.
2. Make the run card, preregistration, implementation, and tests state the same order.
3. Preserve the purpose of detecting construction-time growth relative to the original post-proceed baseline.
4. Recompute every payload and review digest affected by the change.
5. Re-review the complete surface to literal zero findings.

## Review and attestation state

The latest feature-development review was interrupted at the pause boundary. It produced no final zero-finding result and wrote no attestation.

The directory `neuroloc/results/modular_sequence_role_mlx_reviews` contained no files at the stop boundary.

The launch contract requires:

- Four base attestations over the final unchanged preregistration, implementation, tests, and complete surface.
- A fifth training-start attestation over the exact run-bound plan candidate and request record.
- Literal zero findings from the required `feature-dev:code-reviewer` review scope.
- Immutable target bytes matching every attestation.

Prior attestations or intermediate reviews bind superseded bytes and cannot authorize a run.

Because Deyan explicitly stopped work to conserve limits, the documentation changes in this pause record were not sent through another review cycle. That is an intentional stopped state, not a waiver. The canonical plan and handoff bytes changed and therefore require a fresh review on resume.

## Test evidence and its exact boundary

Intermediate test evidence accumulated during implementation:

- A focused complete-surface review suite reached `794 passed, 2 skipped` before later corrections.
- A later combined scoped suite reached `734 passed, 2 skipped` before the final three timing and documentation fixes.
- After those final three fixes, Python compilation was clean and the MLX-focused suite reached `62 passed, 2 skipped`.
- The combined suite and final re-review over the last bytes were active when the stop request interrupted them.

Therefore no complete-suite pass may be claimed for the final pre-pause claim bytes, and no test claim may be transferred to the documentation-modified bytes created by this handoff.

The project `.venv` could not collect the full repository suite. It used Python 3.9 and lacked dependencies including SciPy, Matplotlib, Brotli, and Zstandard, producing `65` collection errors. This was an environment failure, not a model test result.

The Python 3.12 overlay used for broader validation was:

```text
PYTHONPATH=/private/tmp/todorov-full-test-deps:/Users/dttdrv/Projects/Monodratic/.venv/lib/python3.12/site-packages /Users/dttdrv/Projects/Monodratic/.venv/bin/python -m pytest
```

On resume, use the same interpreter and overlay only after confirming the paths still exist. Record exact command lines and results in the appropriate test record.

## Last pre-pause digests

The last canonical payload digest after the final three timing and documentation corrections was:

`afb81ba36129e4565653e89ba44b4de077eda96c0fbfba24ff6b52f33b0ade9d`

That digest did not have a final zero-finding review and was already unlaunchable because of the two residual blockers. It is additionally superseded by the pause documentation changes to the canonical plan.

The last recorded scope digests were:

- Preregistration scope: `408abe0ff54ba686dae453024500b6fdf45e252cbce9acd9b625c8c1e3117fa2`
- Implementation scope: `d623e4b270dfb9eaa7c88d6b2955812ff351e88aa5554a8845430b1da609e6d3`
- Tests scope: `18908bced155d2d89ebd02108df91cf5630e122d40a84c0a22b72a559c0ad480`
- Complete scope: `8c1e46814814db0561c47b199b8f9c64c08a5d7ebce142eb60b86d5178af8b8b`

These values are historical diagnostics only. None authorizes launch.

The raw SHA-256 snapshot immediately before pause documentation was:

- `neuroloc/wiki/PROJECT_PLAN.md`: `d460c3f65e9a78eb4e8dfd4428ae2c1850c12d0a7c92e2f29402761555f82609`
- `neuroloc/wiki/synthesis/modular_neural_model_stack.md`: `5c042abbebc29e0369b03f9f2e9ba0b7869522fd98398c5840692d1e563f0516`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_run.md`: `0f3e57c6271e082061d6f2cd9be6eda11b1d6c6ee18f519d4f12eb5faf523258`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json`: `192a030712b5370a08b4ebeb17bccc14d679a255208e695a8517669bb928672f`
- `src/model/modular_sources.py`: `292383ff04ef5c03a044d8834ff033d8418ac5bfde6591dd017ef80f7cf36cb3`
- `src/model/modular_neural_machine.py`: `ebe8fd17af1204bca294fdbfcc8b081acb44b169aa5439573a06461ef23f3172`
- `src/model/modular_mlx_backend.py`: `ca92205bc296aac2adb816ab5f92c1982a9b39fe5ad33683f32a293355b3a3ab`
- `neuroloc/simulations/memory/modular_sequence_role_cpu.py`: `d282874a3bbcdb400b8ac76d973b158c21b07227bab0f94e7df3a9d70f7744fd`
- `neuroloc/simulations/memory/modular_sequence_role_mlx.py`: `87c6e95d9c31520239759b302ab72121b86d4be0e4f09da032de3c4568dd9369`
- `scripts/qualify_modular_mlx.py`: `8dd2353213bd10602ceee61ff59a85685f54cfe7fb4616d8691051d2d96be283`
- `tests/test_modular_neural_machine.py`: `ca0c046bdd5c94e2d07045bba1d0ec6da90a8fda67ff3148ea5de5dfa5bf80b1`
- `tests/test_modular_sequence_role_cpu.py`: `0d4be06d320a4a83c40c71765c57f7c18030d32169a6301a2f2b8d6172789e92`
- `tests/test_modular_sequence_role_mlx.py`: `6c3e1908e02f412f99863668fa056b82ac02d4ad56c206bfd9e4f6b819fa4a81`
- `neuroloc/simulations/shared.py`: `8f1df1bb2fcb692543b3c3db64c95eab54a946ee229e4c5b3921828d40d1fe01`
- `tests/test_simulation_suite.py`: `a140cfcb033a4690994deba964a0102293086ca31f39cf7bd12cd913da9fd811`

The `PROJECT_PLAN.md` hash above is necessarily superseded by this pause update. Recompute all raw and canonical digests after the two blockers are fixed and after the last documentation edit, never before.

## Separate support fix completed during the work

An unrelated but locally necessary repository-root discovery bug was fixed in `neuroloc/simulations/shared.py` and independently re-reviewed to zero findings before the pause.

The corrected rule requires:

- A regular-file `AGENTS.md` or legacy `CLAUDE.md` marker.
- A regular-file `requirements.txt` marker.
- Rejection of directory impostors using those names.
- Correct file-versus-directory fallback behavior.

The focused suite passed `12/12`, and the biology smoke path passed under the Python 3.12 overlay. This support result does not establish the modular-model claim.

## Separate unresolved repository findings

The following issues are outside the modular training claim and remain unfixed:

### God-machine run-name validation

On macOS, the run name `nested\\run` is accepted far enough to reach the H200 device gate while the corresponding test expects `invalid run name`. This is a separate input-validation bug and was not repaired during the stopped MLX work.

### Historical compression-test drift

Three historical compression tests currently disagree with their recorded expectations:

- Source MPH: return code `0` but `engineering_pass = 0`; relation count `2347`.
- External adapter: return code `2`, `engineering_pass = 0`, fact count `2928`, multiplier `40.344776`, PAQ margin `74328`.
- Knowledge pack: return code `2`, `engineering_pass = 0`, fact count `3875`, multiplier `36.597835`, PAQ margin `-23328`.

These do not establish or refute the modular neural-model claim. Do not mix their status into the training result.

## Working-tree ownership and scope boundary

The repository was already a mixed dirty working tree. Existing changes belong to Deyan unless proven otherwise. Do not clean, reset, overwrite, stage wholesale, or commit unrelated files.

Claim-relevant changed or untracked surfaces at the stop boundary include:

- `neuroloc/HANDOFF.md`
- `neuroloc/simulations/shared.py`
- `neuroloc/simulations/memory/modular_sequence_role_cpu.py`
- `neuroloc/simulations/memory/modular_sequence_role_mlx.py`
- `neuroloc/wiki/PROJECT_PLAN.md`
- `neuroloc/wiki/mistakes/modular_cpu_pilot_closure_and_quadratic_ledger.md`
- `neuroloc/wiki/synthesis/modular_neural_model_methods_review_2026_07.md`
- `neuroloc/wiki/synthesis/modular_neural_model_stack.md`
- `neuroloc/wiki/synthesis/neural_model_dossier_nested_reciprocal_feature_mixer.md`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_base_m5pro_20260722_1.md`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_run.md`
- `scripts/qualify_modular_mlx.py`
- `src/model/modular_mlx_backend.py`
- `src/model/modular_neural_machine.py`
- `src/model/modular_sources.py`
- `tests/test_modular_neural_machine.py`
- `tests/test_modular_sequence_role_cpu.py`
- `tests/test_modular_sequence_role_mlx.py`
- `tests/test_simulation_suite.py`
- `neuroloc/results/modular_sequence_role_cpu/`
- `neuroloc/results/modular_sequence_role_cpu_reviews/`

Many other modified and untracked files are present outside this claim. Use path-specific staging only if Deyan later requests a commit. Never use `git add -A` for this worktree.

## Exact resume order

Do not resume automatically. Wait for explicit authorization from Deyan.

When authorized, use this order:

1. Read `AGENTS.md`, `neuroloc/wiki/PROJECT_PLAN.md`, `neuroloc/wiki/OPERATING_DIRECTIVE.md`, this handoff, the stack contract, the run card, the preregistration payload, and the pilot mistake record.
2. Inspect host processes and confirm there is no surviving pilot, claim child, reviewer, or detached training process.
3. Inspect `git status --short` and preserve every unrelated change.
4. Verify that the claim files still match the last raw hashes above except for expected pause-documentation changes. Treat unexpected drift as a new review event.
5. Write a deterministic failing test for the sampler-stop in-flight-row race before changing implementation.
6. Repair the sampler lifecycle as a bug class across pilot and claim paths.
7. Synchronize the second-swap-sample lifecycle point in the preregistration, run card, code, and tests.
8. Run Python compilation and the focused sampler, protocol, projection, model, CPU, and MLX suites.
9. Run the complete scoped test package on the final unchanged bytes using the Python 3.12 environment.
10. Launch the required separate research, plan-compliance, self-critique, smoke-test, and `feature-dev:code-reviewer` agents only if the current protocol still requires them and Deyan accepts the limit cost.
11. Fix every finding as a bug class and repeat review until literal zero findings.
12. Generate the four base attestations from the final unchanged preregistration, implementation, tests, and complete surfaces.
13. Recompute raw hashes, scope digests, and the canonical payload digest once, after the last change.
14. Run the complete pretraining assertion package without claim training.
15. Run one corrected governed MLX resource pilot only if all prior gates pass.
16. Require the pilot-emitted full-package projection to include all twelve measured components and all five tail assertions.
17. If projected runtime exceeds `1,200` seconds, stop. Research and optimize the execution method without shrinking the frozen workload.
18. If projected runtime is at most `1,200` seconds and all memory, swap, parity, artifact, and lifecycle gates pass, prepare the run-bound training-start plan candidate and request.
19. Obtain the fifth zero-finding training-start attestation.
20. Perform the governed atomic `PROJECT_PLAN.md` publication transaction. That durable replacement is the sole training start.
21. Start claim work only after the child receives the authorized stage request and every anchor still matches.
22. Monitor the run, preserve either positive or negative results, close all artifacts, and update the canonical plan when the run starts and completes.

## Resume test command baseline

The focused files are:

```text
tests/test_modular_neural_machine.py
tests/test_modular_sequence_role_cpu.py
tests/test_modular_sequence_role_mlx.py
tests/test_simulation_suite.py
```

The complete repository command remains:

```text
pytest tests/ -q
```

Because the project `.venv` was unsuitable, a future session must first confirm a Python 3.12 interpreter with Torch, MLX, SciPy, Matplotlib, Brotli, Zstandard, and the repository's remaining test dependencies. Do not call environment collection failures model failures, and do not call a focused green suite complete closure.

## Actions forbidden on resume

- Do not launch training from the stale `afb81...` payload.
- Do not cite `590.761` seconds as measured or achieved runtime.
- Do not reuse an intermediate zero-finding review or attestation after bytes change.
- Do not reduce seeds, updates, controls, evaluations, route rows, checkpoints, gates, or token positions to pass the runtime gate.
- Do not fall back to a multi-hour run.
- Do not hide dense attention inside the selected-set path.
- Do not wholesale-merge Transformerov, Monodratic, Karkasov, CoFrGeNet, Laplace, or `trainingnovel`.
- Do not introduce the reciprocal feature mixer before the base gate.
- Do not call the sequence-role base a 3D world model.
- Do not call record copying imagination, dreaming, replay, memory, or compression.
- Do not claim success from a process, log line, checkpoint file, or partial test alone.
- Do not stage or reset the mixed worktree wholesale.
- Do not spend paid compute without a separate cost/value decision and explicit authorization.

## Cost and compute decision

The planned corrected pilot and training are local on the M5 Pro and have zero external-compute cost. They still consume local wall time, energy, machine responsiveness, and agent limits.

The hard runtime requirement remains:

- Target: at most `600` seconds for the complete fixed package.
- Absolute gate: at most `1,200` seconds.
- Any larger projection: stop and improve the method.

The path not being taken is a multi-hour local launch. It fails the explicit deliverable constraint and would consume time without proving the acceleration method.

The path also not being taken is a reduced toy workload. It could meet the clock while abandoning the actual scientific claim.

## Final recovery statement

The project stopped at a legitimate but incomplete boundary. The architecture exists as real neural code, the MLX fast path has strong narrow parity evidence, and much of the lifecycle and artifact machinery is implemented. The working model does not yet exist because no claim training occurred.

The shortest honest continuation is not more broad planning. It is two bounded corrections, final unchanged-byte testing and review, one corrected governed pilot, and then immediate training only if the complete package passes the twenty-minute gate. If it does not, the next work is execution-method research and optimization, not a longer run and not a smaller scientific experiment.
