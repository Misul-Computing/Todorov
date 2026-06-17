# todorov v0.1

a clean, test-gated, instrumented research codebase for the descent-memory architecture, plus a working attention retriever. built and trained in one session on a single h200.

## what this is

v0.1 is the first end-to-end implementation of the descent-memory idea (a test-time gradient-written fast-weight memory: per token, the memory weights take one gradient step on a per-token associative loss with data-dependent learning rate, momentum, and forget gate), together with the harness needed to evaluate it honestly: synthetic associative-recall (mqar) and induction (passkey) tasks, karpathy sanity gates, gate/state telemetry, and a matched linear control.

## results (toy scale, ~4.2m params, single h200, fp32)

- linear delta control, mqar: chance (token_acc ~0.02). the linear associative substrate does not retrieve. valid control.
- descent memory, mqar: numerically stabilized (input-norm-regularized inner update + bounded inner learning rate + hard state-norm clamp) but at chance. the recurrent memory does not learn to retrieve under sgd at this scale. this is consistent with the project's prior 17-run wall.
- attention (4 layers), passkey/induction: token_acc 1.000, exact_acc 1.000 (n=100, wilson 95% ci 0.963-1.000), converged by ~step 800 of 3000. attention retrieves. this is the working v0.1 model.

finding: a no-shift mqar formulation (value one position after its key) is not solvable by any of these blocks without a token-shift/short conv. even softmax attention, the canonical mqar solver, reaches only ~3x chance here. passkey/induction is the attention-native retrieval task and the working demonstration.

## engineering that caught real bugs before they corrupted results

every one of these was caught by the always-on checks, not in production:
- gate-bias/view layout mismatch -> forget gate initialized near 0.5 (state evaporation, the exact class behind four prior paid runs). caught by the retention gate.
- dead mlp fast-weight init (W2=0 gives zero first gradient, memory never activates). caught by state_norm=0 telemetry while the mlp path masked it in loss.
- save_ckpt NameError from an over-broad replace_all during the compile refactor. caught at first eval.
- test-time memory divergence (state_norm -> 4e8). caught by telemetry; fixed with input-norm regularization + bounded lr + hard clamp, verified locally with a training-like fresh-batch check (the fixed-batch overfit check missed it).

## honest status

the recurrent-memory retrieval problem is NOT solved here; it remains the architecture's core open problem. what v0.1 delivers: a numerically stable, instrumented memory mechanism, valid controls, a working attention retriever, and a methodology (sanity gates + telemetry) that blocks the silent-confound class that fooled prior runs.

## layout

memory.py model.py data.py evals.py sanity.py train.py bytelm.py tests/test_smoke.py decision_log.md decision_log_addendum.md goal

## run

    python train.py --task passkey --preset toy --layers attn,attn,attn,attn   # working attention retriever
    python train.py --task mqar --mode linear --preset toy                      # linear control
    python train.py --task mqar --mode mlp --preset toy                         # descent memory
    pytest tests/ -q
