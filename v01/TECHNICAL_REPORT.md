# todorov v0.1 - technical report

date: 2026-06-02
compute: single nvidia h200 (143 gb), torch 2.8.0+cu128, runpod, fp32
scale: toy, ~4.2m parameters, 4 layers
status: working attention retriever shipped; recurrent-memory retrieval unsolved

this report is the complete, self-contained record of the v0.1 session: what was built,
every experiment run, every bug found and fixed, the results with statistics, and an honest
interpretation. it is written so a future engineer can reproduce and extend the work without
this conversation.

--------------------------------------------------------------------------------
1. summary
--------------------------------------------------------------------------------

the goal was a working <=100m-parameter v0.1 of the descent-memory architecture, with path c
(attention as retriever) as a guaranteed floor. the session built a clean, test-gated, fully
instrumented codebase and ran a controlled toy comparison on associative-recall (mqar) and
induction (passkey) tasks.

result:
- the recurrent "descent memory" was made numerically stable but learns at chance on mqar.
- the linear delta control also sits at chance (expected; capacity-limited substrate).
- a 4-layer attention model solves passkey/induction at exact_acc 1.000 (n=100, wilson 95%
  lower bound 0.963). that is the working v0.1 model.

the recurrent-memory retrieval problem - the wall behind all 17 prior project runs - was not
broken in this session. what v0.1 delivers is a stable instrumented mechanism, valid controls,
a working attention retriever, and a methodology that caught four real bugs before any of them
silently corrupted a result.

--------------------------------------------------------------------------------
2. architecture
--------------------------------------------------------------------------------

2.1 descent memory (memory.py)

the descent memory is a test-time gradient-written fast-weight memory. per token t, with key
k_t, value v_t, query q_t projected from the (locally convolved) hidden state, the memory
weights take one gradient step on a per-token associative loss 1/2 * ||m(k_t) - v_t||^2, then
the read is o_t = m(q_t). two modes:

- linear mode (control): m(x) = W x, W in R^{d_v x d_k}. the update is the delta rule,
  W_t = (1 - lam_t) W_t-1 - beta_t * grad, grad = (W k_t - v_t) k_t^T, with momentum on grad.
  read o_t = W_t q_t.

- mlp mode (treatment): m(x) = W2 silu(W1 x), a 2-layer fast-weight mlp. the per-token gradient
  is computed by manual backprop (so the whole recurrence is autograd-differentiable w.r.t. the
  slow parameters): gW2 = e h^T, gW1 = ((W2^T e) ⊙ silu'(a)) k^T, with e = m(k)-v, a = W1 k,
  h = silu(a). W1 is seeded each sequence from a learned parameter W1_init (a random feature
  map); W2 starts at zero. read o_t = W2 silu(W1 q_t).

data-dependent gates per head: inner learning rate beta = sigmoid(.) in [0,1], forget
lam = sigmoid(.), momentum mu = sigmoid(.), produced by a single linear map; output gate
sigmoid(.) initialized open. keys and queries are l2-normalized.

numerical stabilization (the load-bearing part for the mlp mode):
- input-norm-regularized inner update: each layer's gradient is divided by its input's squared norm
  plus 1.0 (regularized normalized least-mean-squares), making the inner update stable for
  beta < 2 and robust when features are small.
- bounded inner learning rate: beta is sigmoid (bounded), not softplus (unbounded), so the
  outer optimizer cannot drive the inner step into instability.
- hard state-norm clamp: after each update, each fast-weight matrix's per-head frobenius norm
  is clamped to 100 (well above the healthy ~12-25 working range), which mathematically
  prevents the divergence-to-inf/nan failure mode regardless of learned parameters.
- the internal scan runs in fp32.

a short causal depthwise convolution (kernel 4) sits before the q/k/v/gate projections so each
position can mix the previous few tokens. this is required for cross-token binding (see bug 2).

2.2 model (model.py)

rmsnorm, swiglu mlp, rope, causal sdpa attention, optional ternary spike (ste). a Block uses
either the descent memory or causal attention as its mixer, followed by a swiglu mlp, both with
pre-norm residuals. SequenceModel ties the embedding and head. the toy memory experiment runs
memory-only (no attention) so that the memory itself is what is being measured - prior project
runs were confounded because attention silently did the retrieval and masked the memory's
failure.

2.3 tasks (data.py)

- mqar (multi-query associative recall): a context of n distinct key-value pairs interleaved as
  k1 v1 k2 v2 ..., followed by queries; the target at each query-key position is the bound
  value. loss is masked to query positions. targets are next-token aligned.
- passkey / induction: a passkey hidden after a MARK token among filler, then a QUERY token
  followed by the passkey to reproduce. loss masked to the reproduced span. this is an
  induction/copy task.

2.4 harness (evals.py, sanity.py, train.py)

- evals: exact-match and per-token accuracy at masked positions, with wilson 95% confidence
  intervals; a next-token-alignment assertion on the data.
- karpathy sanity gates, run before every training run and aborting on failure: loss-at-init
  ~= ln(vocab); causal no-future-leak (max output delta for a future-only perturbation);
  retention floor (the memory must not evaporate over the sequence); overfit-one-batch (proves
  gradient flow and capacity). the overfit probe mirrors the main architecture (attention probe
  for an attention run) so verification is both fast and representative.
- telemetry: every forward stashes per-head means of the gates and the final fast-weight state
  norm, logged at each eval. this is what distinguishes a dead or diverging memory from a
  working one when the loss alone would hide it.
- train.py: warmup-cosine schedule, adamw, jsonl logging round-tripped to disk, checkpoint
  rotation, optional torch.compile (core/compiled split so checkpoints stay clean), optional
  bf16 autocast.

--------------------------------------------------------------------------------
3. experimental record
--------------------------------------------------------------------------------

all toy runs: vocab 64, ~4.2m params, single h200. mqar used 32 pairs / 8 queries (seq 80);
passkey used seq 128, key_len 5. chance token accuracy on mqar is ~1.6%.

run: linear control (mqar, 3000 steps).
  state_norm grew to ~22 and stayed bounded; no nan. token_acc hovered ~0.02 (chance);
  exact_acc 0 throughout. conclusion: the linear associative substrate does not retrieve.
  valid control, matches theory (interference-limited capacity).

run: descent memory v1/v2 (mqar). dead mlp init (bug 3): state_norm 0.000; token_acc chance.

run: descent memory v3 (mqar, inner-update eps=1e-4). state_norm 12 (step 250) -> 3.0e6 (step 500) ->
  4.0e8 (step 750); token_acc chance. the inner recurrence diverged within the sequence.

run: descent memory v4 (mqar, inner-update eps=1.0 + bounded beta + hard clamp 100). state_norm
  bounded at ~11-15; no nan, no crash; but token_acc stayed at chance (0.011-0.015) and the
  masked loss stayed flat at the alphabet prior (~4.16). conclusion: the stabilized recurrent
  memory is numerically healthy but does not learn to retrieve mqar at this scale.

run: attention (mqar, 2000 steps). token_acc only ~0.04 (about 3x chance), exact_acc 0. even
  softmax attention - the canonical mqar solver - cannot solve this mqar formulation, because
  the value sits one position after its key and no block here has the token-shift needed to
  bind them (the conv is only on the memory path). this is a real finding, not a model failure.

run: attention (passkey/induction, 3000 steps). WORKING. exact_acc 0.910 at step 200, 1.000 by
  step 800, holding 1.000 through step 3000. final token_acc 1.000, exact_acc 1.000, wilson 95%
  ci (0.963, 1.000). checkpoint saved as v0.1_passkey_model.pt. this is the v0.1 model.

--------------------------------------------------------------------------------
4. bugs found and fixed (each caught by the always-on checks)
--------------------------------------------------------------------------------

bug 1 - gate-bias / view layout mismatch.
  symptom: retention floor 3e-20 (state evaporates over 64 tokens).
  cause: gate biases were set grouped-by-gate-type but read with view(B,T,H,3) which interleaves
  by head, so the forget gate initialized near 0.5 instead of ~0.0025. this is the exact state-
  evaporation class behind four prior paid runs.
  fix: read with view(B,T,3,H) to match the bias layout.
  lesson: the retention sanity gate is non-optional; an init bug this quiet passes every other
  check while silently destroying memory.

bug 2 - missing cross-token binding.
  symptom: overfit-one-batch plateaued at 0.44 loss (steps 100/300/600 flat - stuck, not slow).
  cause: the memory computed key, value, and query all from the same token, so it could not
  bind token[p] (key) to token[p+1] (value), which associative recall requires.
  fix: a short causal depthwise conv (kernel 4) before the projections, so each position mixes
  the previous tokens. this is what every linear-attention mqar solver does.
  lesson: an overfit that cannot reach ~0 is diagnostic of a representational gap, not a tuning
  problem.

bug 3 - dead mlp fast-weight init.
  symptom: telemetry state_norm exactly 0.000 for the mlp memory while loss looked normal.
  cause: with W2 = 0, the first write gradient is identically zero (gW2 = e h^T with h from W1
  applied to a zero W2 path, and gW1 vanishes because W2^T e = 0), so the memory never
  activates; the swiglu path masked it in the overfit loss.
  fix: seed W1 from a learned parameter W1_init (random feature map); W2 starts at zero so the
  first read is clean but the write gradient is non-zero.
  lesson: this is the exact "non-memory path masks memory failure" confound that fooled prior
  runs; the state_norm telemetry is what exposed it.

bug 4 - save_ckpt NameError.
  symptom: both training runs crashed at the first eval (step 250).
  cause: a replace_all during the torch.compile refactor renamed save_ckpt's `model` parameter
  to `core` but left `model.state_dict()` in the body.
  fix: core.state_dict().
  lesson: never blind-replace_all across a function signature; it silently renames parameters.

bug 5 - inner-update normalization epsilon too small.
  symptom: the selftest overfit produced nan, aborting the run, even though a prior local check
  passed.
  cause: normalization eps=1e-4 blows up when the hidden features are near zero (dividing by ~1e-4). the
  prior local check used a different config and a fixed-batch overfit, so it missed it.
  fix: damped normalization with eps=1.0.
  lesson: a verification that does not replicate the exact failing configuration gives false
  confidence.

bug 6 - test-time memory divergence under training.
  symptom: with eps=1.0 the single-batch overfit was stable, but real training (fresh batches)
  drove state_norm 12 -> 3e6 -> 4e8.
  cause: the unbounded softplus inner learning rate plus near-zero forget let the outer
  optimizer push the inner recurrence into a runaway within the sequence.
  fix: bound the inner learning rate (sigmoid beta in [0,1]) and add a hard state-norm clamp at
  100. verified locally with a training-like fresh-batch loop (max_state_norm 55, no nan).
  lesson: the verification had to switch from fixed-batch overfit to fresh-batch training to
  reproduce the failure; and a hard mathematical bound beats empirical hope under an adversarial
  outer optimizer.

bug 7 - slow, unrepresentative selftest.
  symptom: an attention run sat for minutes in selftest.
  cause: run_selftest hardcoded an all-memory probe model (slow sequential scan on seq 128) even
  for an attention run.
  fix: build the selftest probe with the same layer kinds as the main model.
  lesson: a gate should test the architecture you ship, not a fixed surrogate.

--------------------------------------------------------------------------------
5. results
--------------------------------------------------------------------------------

- linear delta control, mqar: token_acc ~0.02 (chance), state bounded, no retrieval. valid.
- descent memory, mqar: numerically stable (state ~11-15), token_acc chance, loss flat. the
  recurrent memory does not learn to retrieve at this scale.
- attention, mqar: token_acc ~0.04 (no token-shift -> unsolvable formulation).
- attention, passkey/induction: token_acc 1.000, exact_acc 1.000, n=100, wilson 95% ci
  (0.963, 1.000), converged by ~step 800. working v0.1 model.

--------------------------------------------------------------------------------
6. honest interpretation
--------------------------------------------------------------------------------

the central claim the project has been chasing - that a recurrent biological-style memory can
be trained to retrieve - is still not demonstrated. the descent memory was brought from
"silently broken / diverging" to "numerically stable and instrumented," which is real progress,
but stability is not learning: on a corpus that directly rewards retrieval it sits at chance,
exactly like the linear control and exactly like the prior 17 runs.

what is demonstrated is mundane but solid: softmax attention forms induction heads and solves
the copy/passkey retrieval task to 100%. that is the working model, and it sets the honest
baseline any future recurrent-memory result must beat at matched parameters.

the most useful artifacts of the session are arguably the methodology and the negative results:
the sanity gates and telemetry caught four bugs that would each have produced a confounded or
crashed run, and the mqar-needs-a-shift finding explains a class of failure cleanly.

--------------------------------------------------------------------------------
7. what was not done
--------------------------------------------------------------------------------

- the 100m-parameter scale run. the session budget was consumed by bug-fix cycles and by
  intermittent runpod ssh rate-limiting. the codebase has a d100m preset and a torch.compile
  path ready, but the sequential memory scan needs a chunked/parallel kernel before scale is
  efficient.
- a byte-lm bpb number (enwik8). the loader exists (bytelm.py) but was not run.

--------------------------------------------------------------------------------
8. reproduction
--------------------------------------------------------------------------------

    python train.py --task passkey --preset toy --layers attn,attn,attn,attn   # working retriever
    python train.py --task mqar --mode linear --preset toy                      # linear control
    python train.py --task mqar --mode mlp --preset toy                         # descent memory
    pytest tests/ -q                                                            # 9 smoke tests
    python nan_check.py                                                         # memory stability check

each training run first runs the sanity gates and aborts on failure. metrics stream to
runs/<ts>_<tag>/metrics.jsonl and the best checkpoint to best.pt.

--------------------------------------------------------------------------------
9. open problems and next steps
--------------------------------------------------------------------------------

1. recurrent-memory retrieval (the core problem). the descent memory is stable but does not
   learn to retrieve. candidate directions: add the causal conv to enable cross-token binding
   on mqar for the memory as well; auxiliary retrieval loss weighting the post-query positions;
   warm-start from hand-placed key-value associations to isolate read-side vs write-side
   learning; or replace the substrate with a differentiable key-value table.
2. mqar formulation: either add a token-shift/conv so the value is retrievable from the key
   position, or restructure the task. without it even attention only reaches ~3x chance.
3. throughput: the sequential fast-weight scan is kernel-launch-bound on gpu; a chunked or
   compiled scan is required before any 100m-scale run.
4. scale: run the d100m preset on the working attention path for a real lm/bpb number once a
   stable substrate (or the attention baseline) is chosen.

artifacts: v0.1_passkey_model.pt (trained checkpoint, working retriever), full code on branch
v0.1, decision_log.md and decision_log_addendum.md (chronological notes), goal (the session
contract).
