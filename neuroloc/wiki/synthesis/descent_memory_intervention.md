# descent memory: a test-time gradient intervention (toy result, 2026-06-02)

status: current (as of 2026-06-10).

this article records the first implementation and toy-scale test of a "descent memory" - the
test-time gradient / fast-weight family listed as candidate E in
`wiki/synthesis/substrate_requires_architectural_change.md`. it documents the mechanism, the
numerical stabilization required to make it trainable, the toy result (stable but at chance on
retrieval), and what that implies for the substrate question.

## mechanism

per token t, the memory holds fast weights theta_t and minimizes a per-token associative loss
1/2 * ||m_theta(k_t) - v_t||^2 by one gradient step, then reads o_t = m_theta(q_t):

- linear variant (the existing matrix / delta-rule memory): m(x) = W x; the update is the delta
  rule with data-dependent learning rate, momentum, and forget gate.
- nonlinear variant (the descent memory proper): m(x) = W2 silu(W1 x), a two-layer fast-weight
  mlp updated by manual backprop so the whole recurrence stays differentiable w.r.t. the slow
  parameters. W1 is seeded each sequence from a learned parameter W1_init (a random feature map);
  W2 starts at zero.

gates per head: inner learning rate, forget, momentum (all bounded sigmoids), plus an output gate
initialized open. keys and queries are l2-normalized. a short causal depthwise conv (kernel 4)
precedes the projections so a position can bind to the previous token (required for associative
recall).

## stabilization (the load-bearing finding)

a naive test-time gradient memory diverges under an adversarial outer optimizer: the optimizer
drives the unbounded inner learning rate and near-zero forget into a within-sequence runaway
(observed state_norm 12 -> 3e6 -> 4e8). three changes make it stable for any learned parameters:

1. input-norm-regularized inner update: divide each layer's gradient by its input's squared norm + 1.0
   (regularized normalized least-mean-squares); stable for inner lr < 2, robust when features are
   small.
2. bounded inner learning rate: sigmoid (in [0,1]), not softplus (unbounded).
3. hard state-norm clamp: cap each fast-weight matrix's per-head frobenius norm at 100, well
   above the healthy ~12-25 working range, so divergence is mathematically impossible.

with these, the memory is numerically healthy: bounded state, open gate, gradients flow, and it
overfits a single batch to ~0.

## toy result

toy scale, ~4.2m params, single h200, mqar (vocab 64, 32 pairs / 8 queries) and passkey
(seq 128, key_len 5). full numbers in the run card.

- linear delta control, mqar: chance (token_acc ~0.02). valid control.
- descent memory, mqar: numerically stable (state ~11-15) but token_acc at chance (0.011-0.015)
  and masked loss flat at the alphabet prior. it does not learn the retrieval algorithm.
- attention, passkey/induction: exact_acc 1.000 (n=100, wilson 95% ci 0.963-1.000). the working
  v0.1 retriever.

## the mqar-needs-a-shift finding

even softmax attention - the canonical mqar solver - reached only ~3x chance on this mqar
formulation, because the value sits one position after its key and the attention block has no
token-shift/conv. associative recall over this layout is unsolvable without a shift; induction /
passkey is the attention-native retrieval task and is what the working model solves.

## interpretation and relation to candidate E

stability is not learning. a substrate can be numerically fine and still fail to be trained into
retrieval by sgd at this scale - exactly the substrate-requires-architectural-change thesis. this
toy test of candidate E (test-time gradient memory) therefore does not lift the retrieval wall;
it narrows the problem to "the write/read loop does not become content-addressable under sgd,"
not "the memory is unstable." next directions: add the causal conv to the memory's binding path
for mqar; auxiliary retrieval loss on post-query positions; warm-start from hand-placed
key-value pairs to separate read-side vs write-side learning; or a differentiable key-value table
with hard attention at train time.

one write-side direction is pre-registered as candidate F: route the inner loop's own prediction
error (the surprise `||pred - v||`) into the write gain, a three-factor / affect gate, so plasticity
concentrates on surprising associations. mechanism, controls, and kill conditions in
`wiki/tests/affect_gated_write_cpu_experiment.md`.

executed 2026-06-10 and falsified: surprise content is inert on the write side (at chance under
own-history normalisation and under exact budget matching). the shuffled-surprise control,
however, ignited the first sgd-trained non-chance retrieval on this substrate (mqar exact 0.360 /
token 0.785 at 1200 steps, seed 0; seed-sensitive, not yet a recipe). the anomaly is promoted to
candidate G (stochastic write gain, backlog) in
`wiki/synthesis/substrate_requires_architectural_change.md`; full numbers in the frozen run card.

## see also

- `wiki/tests/v0_1_descent_memory_toy_results.md` — the run card with full numbers
- `wiki/mistakes/descent_memory_v0_1_bugs.md` — the seven bugs caught during implementation
- `wiki/synthesis/substrate_requires_architectural_change.md` — candidate E and the A-F list
- `wiki/tests/affect_gated_write_cpu_experiment.md` — candidate F, the affect-gated write pre-registration
- `wiki/synthesis/training_objective_vs_architectural_goal.md` — the prior corpus-pivot diagnosis
