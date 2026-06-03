# mistake: v0.1 descent-memory session - seven bugs, and documentation dumped instead of integrated

status: historical context only. frozen as of 2026-06-02. do not edit.

## what happened

2026-06-02. the v0.1 descent-memory session hit seven distinct bugs during a single sitting and,
separately, produced its documentation as four standalone files in `v01/` instead of integrating
them into the neuroloc wiki per `wiki/OPERATING_DIRECTIVE.md`. the bugs were each caught by the
always-on checks, so no confounded or crashed result was reported as a finding. the
documentation failure is its own mistake and is the reason this doc exists; deyan caught it.

## the seven bugs

1. gate-bias / view layout mismatch. symptom: retention floor 3e-20 (state evaporates over 64
   tokens). cause: gate biases set grouped-by-gate-type but read with `view(B,T,H,3)` which
   interleaves by head, so the forget gate initialized near 0.5 instead of ~0.0025 - the exact
   state-evaporation class behind four prior paid runs. fix: read with `view(B,T,3,H)`. lesson:
   the retention sanity gate is non-optional; this bug passes every other check silently.

2. missing cross-token binding. symptom: overfit-one-batch plateaued at 0.44 loss (steps
   100/300/600 flat - stuck, not slow). cause: the memory computed key, value, and query all
   from the same token, so it could not bind token[p] (key) to token[p+1] (value). fix: a short
   causal depthwise conv (kernel 4) before the projections. lesson: an overfit that cannot reach
   ~0 is a representational gap, not a tuning problem.

3. dead mlp fast-weight init. symptom: telemetry `state_norm` exactly 0.000 while loss looked
   normal. cause: with W2 = 0 the first write gradient is identically zero, so the memory never
   activates; the swiglu path masked it in the overfit loss. fix: seed W1 from a learned
   parameter W1_init; W2 starts at zero. lesson: this is the "non-memory path masks memory
   failure" confound that fooled prior runs; the state_norm telemetry is what exposed it.

4. save_ckpt NameError. symptom: both training runs crashed at the first eval (step 250). cause:
   a `replace_all` during the torch.compile refactor renamed save_ckpt's `model` parameter to
   `core` but left `model.state_dict()` in the body. fix: `core.state_dict()`. lesson: never
   blind `replace_all` across a function signature.

5. nlms epsilon too small. symptom: the selftest overfit produced nan, aborting the run, even
   though a prior local check passed. cause: nlms eps=1e-4 blows up when hidden features are near
   zero; the prior local check used a different config and a fixed-batch overfit, so it missed
   it. fix: damped nlms eps=1.0. lesson: a verification that does not replicate the exact failing
   configuration gives false confidence.

6. test-time memory divergence under training. symptom: with eps=1.0 the single-batch overfit
   was stable, but real training (fresh batches) drove state_norm 12 -> 3e6 -> 4e8. cause: the
   unbounded softplus inner learning rate plus near-zero forget let the outer optimizer push the
   inner recurrence into a within-sequence runaway. fix: bound the inner lr (sigmoid in [0,1]) and
   add a hard state-norm clamp at 100; verified locally with a fresh-batch training loop. lesson:
   the verification had to switch from fixed-batch overfit to fresh-batch training to reproduce
   the failure; and a hard mathematical bound beats empirical hope under an adversarial outer
   optimizer.

7. slow, unrepresentative selftest. symptom: an attention run sat for minutes in selftest. cause:
   `run_selftest` hardcoded an all-memory probe model (slow sequential scan) even for an
   attention run. fix: build the selftest probe with the same layer kinds as the main model.
   lesson: a gate should test the architecture you ship, not a fixed surrogate.

## the documentation mistake

the session first wrote `README.md`, `TECHNICAL_REPORT.md`, `decision_log.md`, and
`decision_log_addendum.md` into `v01/`. none carried a banner, none were placed under
`wiki/tests/`, `wiki/mistakes/`, or `wiki/synthesis/`, none had bidirectional `see also` links,
and no canonical state file (`PROJECT_PLAN.md`, `program_status.yaml`, `STATUS_BOARD.md`) was
updated. per the source-of-truth hierarchy a standalone report in `v01/` is invisible to the
wiki graph: it will not be found, will not be cross-linked, and will not correct the stale claims
it contradicts (the curriculum/compression "active workstream" framing, the missing
`compressed-dancing-haven.md` plan reference). the fix was to author the proper run card, this
mistake doc, and a synthesis article, update the three state files and the results/bug-history in
`CLAUDE.md`, cross-link everything bidirectionally, and run the prosecutor to zero.

## why the checks caught the bugs but not the doc gap

the sanity gates and telemetry are wired into training; there is no equivalent automated gate
for "did you integrate the docs into the wiki." that gate is the human (deyan) and the prosecutor
on wiki changes, which was also skipped in the first pass.

## rule (self-imposed)

documentation is not done when content exists. it is done when it is integrated into the wiki
graph (banner, lifecycle state, bidirectional `see also`), the canonical state files agree on the
new current state, and the prosecutor returns zero findings.

## see also

- `wiki/tests/v0_1_descent_memory_toy_results.md` — the run card for this session
- `wiki/synthesis/descent_memory_intervention.md` — mechanism and interpretation
- `wiki/synthesis/substrate_requires_architectural_change.md` — candidate E context
- `wiki/OPERATING_DIRECTIVE.md` — the documentation rules that were skipped
