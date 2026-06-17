todorov v0.1 - decision log addendum (toy experiment, control vs descent memory)

2026-06-02, mqar toy (memory-only model, vocab 64, 32 pairs, 8 queries, seq 80). chance token_acc ~1.6%.

control (linear delta memory): VALID, clean.
- 3000 steps, no nan, state_norm grew to ~22 and stayed bounded.
- token_acc hovered at chance (~0.02) the entire run; exact_acc 0.
- conclusion: the linear associative substrate does not solve mqar. matches theory (capacity-limited). this is a good control.

treatment (descent memory, mlp fast-weight). iterated through real bugs, each caught by the always-on checks:
- v1 dead-init: W2 initialized to 0 makes the first write gradient identically 0 (h = silu(W1*0) = 0, and W2^T e = 0), so the memory never activates. caught by telemetry state_norm = 0.000 exactly, while the swiglu path masked it by partially fitting the overfit batch. this is the exact "non-memory path masks memory failure" confound that fooled all 17 prior runs; telemetry exposed it. fix: seed W1 from a learned parameter W1_init (random feature map), W2 starts at 0.
- save_ckpt crash: a replace_all during the --compile refactor renamed save_ckpt's `model` param to `core` but left `model.state_dict()` in the body -> NameError at first eval (step 250). fix: core.state_dict(). lesson: do not blind replace_all across a function signature.
- v3 nan then divergence: with W1 alive, the test-time inner update diverged. nlms eps=1e-4 stopped the immediate nan in the single-batch overfit, but under real training (fresh batches) the state ran away: state_norm 12 (step250) -> 3e6 (step500) -> 4e8 (step750), token_acc at chance. root cause: inner learning rate is softplus (unbounded) and forget is near zero, so the outer optimizer drives the inner recurrence unstable; the divergence is within-sequence (end-of-forward state norm).
- v4 stabilization: bound the inner learning rate (beta = sigmoid in [0,1], stable for nlms beta<2) and add a hard state-norm clamp (each fast-weight matrix per-head frobenius norm capped at 100, well above the healthy 12-25 range so it does not restrict normal learning). verified locally with a TRAINING-like fresh-batch loop (the overfit/fixed-batch check missed the divergence; the verification must mimic the failing regime): max_state_norm 55, no nan, bounded. relaunched as v4.

key methodological lessons reinforced this session:
- telemetry (state_norm, gates) is what distinguishes a dead/diverging memory from a working one; loss alone hides it.
- a verification that does not replicate the failing configuration/regime gives false confidence (the cpu overfit passed while the gpu training diverged; the verify had to switch to fresh-batch training to be meaningful).
- hard mathematical bounds (clamp) beat empirical hope for stability under an adversarial outer optimizer.

open question at end of session window: does the stabilized descent memory LEARN to retrieve (generalize mqar), or is it stable-but-at-chance like the linear control. the overfit gate shows it can memorize a fixed batch; generalization to fresh sequences is the real test and is what the v4 eval trajectory measures.
