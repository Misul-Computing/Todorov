# Modular MLX session stop handoff

Status: stopped by Deyan on 2026-07-26. No attempt-4 pilot or claim training was launched.

## Executive truth

Deyan stopped this session after repeated delay between the explicit instruction
to start local training and an actual training launch. The launch path is closed
for this session. No paid compute was used. No attempt-4 run root exists. No
training-start request was created. No claim model or optimizer update began.

The latest preserved governed run remains
`neuroloc/results/modular_sequence_role_mlx/mlx-m5pro-20260726-3`. Its preflight
passed the then-current complete assertion surface and its MLX child executed all
`132` pilot updates, accounting for `292,864` attempted pilot token positions.
The run did not publish `run/pilot.json` or `ABORTED.json`, did not create a
training-start request, and did not begin claim training. It remains an
immutable failed nonterminal pilot attempt.

The repository is not launch-ready at this stop point. An interrupted final
contract synchronization changed the preregistration payload without updating
the matching digest constants. A future qualifier invocation must refuse the
current surface before preflight. Do not launch by bypassing that refusal.

## User instruction and session boundary

The operative instruction at the stop point was:

- Document everything.
- Stop the session.
- Do not launch training.

All active implementation and reviewer agents were interrupted. The already
running modular verification process completed before the stop documentation
was written. No new training process was started.

## Actual model represented by this run

The attempted base composition is a neural modular sequence model, not a
Transformer fallback and not yet the complete requested 3D latent world model.
Its frozen base roles are:

- Routed selected-set attention for exact remote recall.
- Recurrent sequence-state mixing for world-state tracking.
- Token-local feed-forward mixing.
- A separately trained dense-attention control.
- Torch CPU authority for deterministic sources, checkpoint replay, reductions,
  gates, and artifact validation.
- MLX Metal execution for production forward, gradient, and optimizer work.

The nested-reciprocal feature mixer is not included in this base run. The
Karkasov specialist-docking protocol is not active. Laplace and `trainingnovel`
are not training backends for this attempt. The base run is intended to prove
routed exact recall and recurrent tracking in the same matched task package
before any later 3D latent-memory, learned imagination, dreaming, replay, or
feature-mixer claim.

No current artifact proves a learned 3D world model, imagination, dreaming,
replay, arbitrary chat, or a finished model. Attempt 3 is pilot execution
evidence only.

## Governed attempt history

### Attempt 1

Run ID: `mlx-m5pro-20260726-1`.

The run hard-aborted before pilot work because parent workload labels disagreed
with the child protocol. It published a terminal abort closure. No claim
training began.

### Attempt 2

Run ID: `mlx-m5pro-20260726-2`.

The child executed the pilot workload, but the tail timing registry and parent
validator disagreed. It published a terminal abort closure. No claim training
began.

### Attempt 3

Run ID: `mlx-m5pro-20260726-3`.

The current preflight passed. The child executed all `132` pilot updates and
reported `292,864` attempted positions. After clean child exit, an already
in-flight resource sample captured the child as zero-resident and zero-CPU while
retaining stale active rung-two job state. That stale sample committed after the
lifecycle generation changed. The required terminal sampler stop then rejected
the timeline, and abort finalization rejected the zeroed child as decreasing CPU
time.

Attempt 3 contains:

- `run/preflight.json`
- `run/pilot_resources.jsonl`
- `run/config_manifest.json`
- `run/environment.json`
- `run/prereg.json`
- `run/project_plan_launch.md`
- `run/source_manifest.json`
- Four copied reviewer attestations
- Forty-eight check-detail files
- The selected-attention oracle sentinel

Attempt 3 does not contain:

- `run/pilot.json`
- `run/training_start_request.json`
- `run/training_start_plan.json`
- Claim attempt ledgers
- Claim checkpoints
- `ABORTED.json`
- Terminal `SHA256SUMS`

The attempt is therefore neither a successful pilot nor a terminal abort. It
must remain unchanged and must never be resumed in place.

### Attempt 4

Reserved future run ID: `mlx-m5pro-20260726-4`.

The run root does not exist. No qualifier process was launched for this ID.

## Repairs completed after attempt 3

The terminal resource-sampler bug class was repaired test-first:

- Every progress, active-job clear, and child-exit mutation advances a sampler
  generation.
- A transaction captured under an older generation is discarded.
- A previously positive direct child that becomes zero-resident and zero-CPU is
  not accepted as a live process sample.
- Normal sampler stop first quiesces the periodic sampler.
- Stop may reuse an exact already durable clean row or wait only the remaining
  portion of the five-second witness interval before taking one terminal sample.
- The sampler records complete transaction duration through process sampling,
  swap sampling, writer append, fsync, and acknowledgement.
- The full-package projection adds a conservative resource-finalization
  component equal to the five-second interval plus twice the maximum observed
  sampler transaction.
- Packaging now accounts for thirteen measured components and six
  content-addressed tail assertions.
- The legacy attempt-3 zeroed-child prefix may be used only to finalize that
  historical abort shape. PID reappearance, parent disappearance, changed
  parentage, or a zeroed clean terminal row remains invalid.

The claim deadline bug class was also repaired:

- The `1,200`-second bound is a fail-closed acceptance deadline, not a claim that
  Python can preempt a blocked operating-system or storage call.
- Every governed terminal phase checks the deadline before and after return.
- A late-returning operating-system or storage call hard-aborts and cannot
  produce an accepted pilot.
- Child wait is capped to the lesser of thirty seconds and the remaining
  acceptance time.
- The deadline is rechecked after child wait, stderr flush and fsync, stderr
  read, and successful owned scratch cleanup.
- The clean child transport is finalized before terminal `SHA256SUMS` so a
  late child close cannot turn into a successful terminal claim.

The scientific workload was not reduced by these repairs. The model, tasks,
gates, construction seeds, data seeds, route seeds, update counts, token counts,
losses, controls, schedules, and FP32 method remain unchanged.

## Exact interrupted working-tree state

The final reviewer correctly found that code and preregistration prose disagreed
about clean-claim order. Production code now performs:

1. Completion and summary construction.
2. Child close acknowledgement.
3. Remaining-time-capped child join.
4. Stderr flush, fsync, and read.
5. Owned scratch cleanup.
6. Final acceptance guard.
7. Clean artifact-closure validation.
8. Terminal `SHA256SUMS` write and fsync.

The contract-sync agent began updating the machine-readable payload and run card
to that order. Deyan then stopped the session and the agent was interrupted.
The partial synchronization is visible in:

- `neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_run.md`
- `tests/test_modular_sequence_role_cpu.py`

The current canonical JSON digest computed directly from the modified payload
is:

`8c7825f69fd27a7f3653c2e3bfab8673f3bb13d9f543fecd4a6aa9b97a4868ab`

The following launch-bound constants still contain the prior digest:

- `neuroloc/simulations/memory/modular_sequence_role_cpu.py`:
  `fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553`
- `tests/test_modular_sequence_role_cpu.py`:
  `fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553`
- `neuroloc/wiki/tests/modular_sequence_role_cpu_run.md`:
  `fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553`

This mismatch is intentional evidence of an interrupted edit, not a digest to
paper over. Before changing any constant, a future session must review the
entire partially updated payload and run-card order, finish every analogous
description, and then compute the final digest from settled bytes.

The run card also still contains older references to twelve measured components
and five tail assertions in later sections even though the live payload has
thirteen components and six tail assertions. Earlier packaging formulas at the
canonical definition were changed from `Dpreflight + 5 + 108` to
`Dpreflight + 6 + 108`, but the whole run card was not re-reviewed after the
interrupted synchronization.

## Verification chronology

The last fully documented pre-session modular result was:

`834 passed, 2 skipped in 214.05s`

After the sampler and deadline repairs, focused checks reported:

- `5 passed in 0.70s` for commit-close, claim and pilot ordering, and the CPU
  terminal-order guard.
- `27 passed, 745 deselected in 0.87s` in the independent reviewer run.
- No code or machine-readable payload finding on the then-current
  `fc3c7130...` surface before the final clean-claim order mismatch was found.

A complete modular verification then passed:

`844 passed, 2 skipped in 220.75s`

That run covered:

- `tests/test_modular_neural_machine.py`
- `tests/test_modular_sequence_role_cpu.py`
- `tests/test_modular_sequence_role_mlx.py`

It used:

`PYTHONPYCACHEPREFIX=/private/tmp/todorov-pycache /Users/dttdrv/Projects/Todorov/.venv/bin/python -m pytest tests/test_modular_neural_machine.py tests/test_modular_sequence_role_cpu.py tests/test_modular_sequence_role_mlx.py -q`

The `844`-pass result preceded the interrupted payload and run-card order edit.
It does not certify the current stopped bytes.

One separate invocation of `.venv/bin/pytest tests/ -q` failed during collection
with `65` import errors because the Todorov Python 3.9 environment does not
contain repository-wide optional SciPy and Matplotlib dependencies. That
invocation changed no code and is not a modular regression result.

No focused or complete test was run after the interrupted payload edit. The
current working tree is therefore unverified.

## Review and attestation state

The implementation reviewer had reached zero code and payload findings before
discovering the final clean-claim order mismatch. After that finding, the
contract-sync agent changed the payload and was interrupted before completing
digest synchronization and focused tests. There is no final literal-zero review
of the stopped bytes.

The four files currently at
`neuroloc/results/modular_sequence_role_mlx_reviews/` attest to earlier bytes.
They are stale for any future attempt because the qualifier, tests,
machine-readable payload, and run card changed. They must not admit attempt 4.

A future governed launch requires exactly four fresh, run-ID-free,
content-addressed base attestations created by the required
`feature-dev:code-reviewer` after:

- The partial contract edit is completed.
- The canonical digest is synchronized.
- Focused and complete modular verification pass.
- Every documentation and implementation finding reaches literal zero.

The runner may copy those attestations but may not author its own reviewer
identity.

## Research completed without changing the launch

No defensible source supports claiming an arbitrary multi-hour-to-minutes
training factor for this exact workload. The current method remains MLX `0.29.3`
with unchanged FP32 science.

If a future governed pilot misses the `1,200`-second limit, the evidence-backed
contingency order is:

1. Benchmark MLX `0.32.0` in an isolated environment with the unchanged FP32
   package.
2. Retain and cache compiled stage closures by immutable signature so repeated
   stage execution does not recreate compiled functions.
3. Test MLX fast scaled dot-product attention only for the dense control, with
   exact masks and parity.
4. Profile the full training step before considering a custom routing or
   recurrence kernel.
5. Investigate input-copy removal only if the trace shows a material CPU gap.

Each comparison must preserve seeds, batches, updates, model, optimizer,
schedule, checks, and artifacts; separate cold compilation from warm execution;
and report end-to-end wall time, peak memory, route identity, parameter and
optimizer parity, and final gates. No speedup factor is established yet.

Primary sources used:

- MLX releases: `https://github.com/ml-explore/mlx/releases`
- MLX compilation guidance:
  `https://ml-explore.github.io/mlx/build/html/usage/compile.html`
- MLX fast operations:
  `https://ml-explore.github.io/mlx/build/html/python/fast.html`
- MLX Metal debugger:
  `https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html`
- MLX custom Metal kernels:
  `https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html`

These are contingency candidates, not validated interventions and not part of
the stopped frozen run.

## Exact safe recovery order

Do not begin by launching.

1. Read `AGENTS.md`, `neuroloc/wiki/PROJECT_PLAN.md`, this handoff, and
   `neuroloc/wiki/OPERATING_DIRECTIVE.md`.
2. Preserve all three governed run directories unchanged.
3. Inspect the interrupted diff in the payload, run card, CPU exact-order test,
   and digest constants.
4. Finish the clean-claim order contract class-wide.
5. Compute the canonical payload digest only after all payload bytes settle.
6. Update every digest constant and current run-card digest to that one value.
7. Replace all current twelve-component and five-tail descriptions with
   thirteen components and six assertions where they describe the live method.
8. Run the focused order, deadline, sampler, projection, and tracked-payload
   checks.
9. Run the complete modular command recorded above.
10. Synchronize `neuroloc/wiki/PROJECT_PLAN.md`, `docs/STATUS_BOARD.md`,
    `state/program_status.yaml`, `AGENTS.md`, `neuroloc/HANDOFF.md`, the current
    stack article, the resumption record, and the pause handoff.
11. Run the required complete-surface reviewer to literal zero.
12. Move superseded active attestations to the established historical
    directory and create exactly four fresh base attestations.
13. Re-run the complete assertion package.
14. Only after a new explicit user instruction to resume, launch attempt 4.

The future launch command, which was not executed in this session, is:

`/usr/bin/env PYTHONHASHSEED=0 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 /Users/dttdrv/Projects/Todorov/.venv/bin/python /Users/dttdrv/Projects/Todorov/scripts/qualify_modular_mlx.py --run-root /Users/dttdrv/Projects/Todorov/neuroloc/results/modular_sequence_role_mlx/mlx-m5pro-20260726-4`

It requires local Metal access. It authorizes only the governed pilot and, only
if every gate passes, the unchanged local claim run. It does not authorize paid
compute.

## Working-tree boundary

The repository was already broadly dirty before this final stop. Many modified
and untracked files are unrelated user work or prior project work. Do not reset,
clean, discard, stage, commit, or rewrite the whole tree. Limit future edits to
the exact modular MLX surface after inspecting ownership.

The three governed attempt directories and reviewer history are evidence. Do
not remove, rename, rewrite, or normalize them.

## Final category check

Implemented operation: a reviewed modular neural-model scaffold and governed
MLX pilot/claim harness with routed recall, recurrent tracking, controls,
artifact accounting, and fail-closed lifecycle handling.

Strongest current evidence: attempt 3 completed the MLX pilot workload and the
pre-interruption modular surface passed `844` tests with `2` skips.

What failed: no governed pilot produced an accepted full-package projection;
attempt 3 ended nonterminal after a terminal resource-sampling race; final
contract synchronization was interrupted; no claim training began.

What is not proved: a trained combined model, the same-task exact-recall and
world-tracking gate, a 3D latent world model, imagination, dreaming, replay,
feature-mixer benefit, specialist docking, a sub-twenty-minute full package, or
any acceleration factor.

Why this is not promoted: the stopped bytes are digest-inconsistent and
unverified, the active attestations are stale, attempt 4 does not exist, and no
training result exists.
