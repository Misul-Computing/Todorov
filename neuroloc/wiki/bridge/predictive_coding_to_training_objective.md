# bridge: predictive coding to todorov training objective

status: current (as of 2026-04-16).

## the biological mechanism

[[predictive_coding]] (Rao & Ballard 1999) proposes that each cortical level maintains a generative model predicting the activity at the level below. only prediction errors propagate upward. the mathematical formulation:

    e_l = x_l - f(r_{l+1})
    dr_{l+1}/dt = -e_{l+1} + (df/dr_{l+1})^T * e_l
    dU_l/dt = eta * e_l * r_{l+1}^T

key properties: (1) errors are LOCAL to each level, (2) the learning rule is Hebbian (outer product of error and representation), (3) top-down predictions suppress expected input so only surprise propagates, (4) precision weighting ([[precision_weighting]]) modulates the gain of errors based on estimated reliability.

the [[free_energy_principle]] (Friston 2005, 2010) subsumes predictive coding: minimizing variational free energy under Gaussian assumptions yields the same update rules. the key addition: precision (inverse variance) of prediction errors is itself inferred, implementing attention as a natural consequence of Bayesian inference.

## the current todorov implementation

todorov trains with standard next-token prediction (cross-entropy loss). the loss is computed at the OUTPUT of the full model, not at each layer. there is no explicit prediction error computation between layers.

the architecture has:
- residual connections (x + f(x)) at every layer
- RMSNorm before each sublayer
- the residual stream carries the "prediction" forward
- each layer modifies the residual stream
- KDA state (S_t) accumulates associations over time
- no explicit top-down connections (purely feedforward at inference)

### the delta rule erasure as implicit precision

the KDA state update:

    S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T

the alpha decay ERASES old associations. the beta gate MODULATES write strength. together, they implement a form of "confidence management" over the recurrent state:

**alpha as temporal precision**: alpha determines how much old state persists. per-channel, learned. initialized at sigmoid(-2) ~ 0.12, meaning ~88% of state is erased each step. this is aggressive erasure. channels with high alpha (close to 1) act as long-term memory (high temporal precision: "old information is reliable"). channels with low alpha (close to 0) act as working registers (low temporal precision: "only the most recent input matters"). the distribution of alphas across channels determines the model's implicit timescale hierarchy.

is this erasure a form of prediction error? consider: when alpha is close to 0, old state is aggressively discarded, as if the model expects every new token to be "surprising" (requiring a fresh state). when alpha is close to 1, old state persists, as if the model expects continuity (new tokens are "predicted" by the existing state). in this reading, alpha encodes a prior belief about the temporal stability of the latent state, which is analogous to temporal precision in the free energy framework.

but this reading is strained. alpha is FIXED per channel -- it does not adapt to the actual prediction error at each timestep. a token that confirms the existing state and a token that contradicts it both encounter the same alpha. true predictive coding would adjust alpha dynamically based on the discrepancy between the incoming token and the predicted state.

**beta as input precision**: beta_t = sigmoid(beta_proj(x_t)) is input-dependent. high beta = "this input is important, write it to state." low beta = "this input is unimportant, ignore it." this is functionally closer to precision weighting: the model learns to assign different reliability to different inputs. but beta is computed from the input alone, not from a comparison between input and prediction. in predictive coding, precision is the reliability of the PREDICTION ERROR, not of the input itself.

### what the architecture is actually doing

the honest analysis: todorov is doing next-token prediction with recurrent state augmentation. the KDA state provides a compressed history. the MLA layers provide exact retrieval from recent context. the residual stream carries the evolving representation forward. the ternary spikes provide quantized information bottlenecks.

none of these components implement predictive coding in the technical sense. there are no per-layer prediction errors, no top-down predictions between layers, no precision-weighted error propagation, and no iterative settling dynamics.

Millidge et al. (2022) showed that backpropagation computes the same gradients as predictive coding at convergence. this means todorov's LEARNED parameters may approximate those of a predictive coding network. but the DYNAMICS at inference time are purely feedforward, not iterative/bidirectional.

## the proposed change

### option A: per-layer auxiliary prediction loss

add an auxiliary loss where each layer predicts the representation at the layer below:

    pred_l = g_l(h_l)
    aux_loss_l = ||h_{l-1}.detach() - pred_l||^2
    total_loss = cross_entropy_loss + lambda * sum_l(aux_loss_l)

where g_l is a lightweight linear projection from layer l's hidden state to the dimensionality of layer l-1's hidden state, and h_{l-1}.detach() stops gradients from flowing through the "target" representation.

this creates explicit per-layer prediction errors without changing the forward pass. the auxiliary loss encourages each layer to maintain a generative model of the layer below, which may improve representation quality.

**problem**: the prediction direction is WRONG. in biological predictive coding, higher levels predict LOWER levels (top-down). this auxiliary loss also predicts lower from higher, which is correct. but the "target" is the ACTUAL lower-level representation, not a precision-weighted error. and the auxiliary gradient flows BACKWARD through the prediction head g_l, not through a biologically plausible local update.

### option B: spike-as-surprise

modify the ternary spike mechanism to compare the current activation against a running prediction:

    prediction_t = ema_decay * prediction_{t-1} + (1 - ema_decay) * x_{t-1}
    error_t = x_t - prediction_t
    spike_t = ternary_quantize(error_t, threshold_t)

this makes spikes genuinely represent surprise: a neuron spikes when the current input differs from the exponential moving average prediction. neurons in a predictable context (repeated patterns) would spike less; neurons encountering novel input would spike more.

**expected consequence on firing rate**: the 41% firing rate would change. in highly predictable contexts (e.g., common collocations), the error signal would be small and fewer neurons would spike. in unpredictable contexts (e.g., rare words, topic changes), the error would be large and more neurons would spike. the firing rate would become context-dependent rather than approximately constant.

**problem 1**: the EMA prediction is trivial. it predicts "the next activation will be similar to the previous one." this captures only first-order temporal correlations. predictive coding uses a GENERATIVE MODEL that can capture complex hierarchical structure. an EMA is not a generative model.

**problem 2**: gradient flow through the EMA introduces a temporal dependency that complicates parallelized training. the prediction_t depends on prediction_{t-1}, which depends on prediction_{t-2}, etc. this is another recurrence that must be handled by BPTT.

**problem 3**: at training time, todorov processes all tokens in parallel (not sequentially). the EMA would need to be computed sequentially or approximated, adding computational overhead.

### option C: predictive coding loss on KDA state

the KDA state S_t is already a generative model: given a query q, it generates a value o = q^T * S_t. add a loss that evaluates whether S_t accurately predicts the NEXT token's key-value association:

    predicted_v = q_{t+1}^T * S_t
    actual_v = v_{t+1}
    pc_loss = ||actual_v - predicted_v||^2

this evaluates S_t's predictive quality: does the state accumulated from tokens 1..t correctly predict what will be needed at token t+1?

**problem**: this is circular. S_t was trained to produce representations that feed into the output projection for next-token prediction. adding a loss that measures whether S_t predicts the next token's VALUE is redundant with the next-token prediction loss itself. the KDA state is already optimized to carry information useful for predicting the next token.

## implementation spec

### option A (per-layer auxiliary prediction loss)

location: train.py, after the main transformer loop.

new components:
- L-1 linear projections g_l: nn.Linear(d_model, d_model) per layer
- auxiliary loss coefficient lambda (hyperparameter, suggest 0.01-0.1)

forward pass change: after each layer, store h_l. after the full forward pass, compute aux_loss_l = ||h_{l-1}.detach() - g_l(h_l)||^2 for each l.

parameter count increase: (L-1) * d_model^2. for L=24, d_model=1024: ~24M additional parameters (8% increase for a 300M model).

computational cost: L-1 additional matrix multiplies of shape (B, T, d_model) x (d_model, d_model) per forward pass. non-trivial but not dominant.

### option B (spike-as-surprise)

location: train.py, inside the TernarySpike class.

new components:
- per-spike running prediction buffer (shape: same as input activations)
- ema_decay parameter (suggest 0.9)

forward pass change: compute prediction from EMA, compute error = input - prediction, quantize error instead of input.

parameter count increase: zero new learnable parameters. one additional buffer per spike location.

computational cost: one additional subtraction per spike location. negligible.

training constraint: requires sequential token processing for the EMA update, or approximation via parallel scan.

### option C (predictive coding loss on KDA state)

location: train.py, after KDA state update.

new components:
- pc_loss coefficient (hyperparameter, suggest 0.01)

forward pass change: after computing S_t, evaluate q_{t+1}^T * S_t - v_{t+1} for each head.

parameter count increase: zero.

computational cost: one additional matrix-vector multiply per head per timestep. moderate.

training constraint: requires access to t+1 queries and values during loss computation at timestep t, which is available during teacher-forced training.

## expected impact

### option A
- positive: each layer learns an explicit generative model of the layer below. may improve representation quality and reduce the "dark knowledge" problem (layers learning redundant features)
- negative: the auxiliary loss may conflict with the main loss. the prediction heads g_l add parameters and computation. the benefit is unproven at scale
- risk: moderate. the auxiliary loss is well-understood (similar to deep supervision in computer vision)

### option B
- positive: spikes become genuinely interpretable as surprise signals. firing rate becomes context-dependent. may improve spike MI (currently 1.168 at 267M)
- negative: introduces sequential dependency that may slow training. the EMA prediction is too simple to capture complex patterns. firing rate may become unstable (too high in novel contexts, too low in predictable ones)
- risk: high. changes the fundamental semantics of the spike mechanism. may break the STE gradient flow that currently works

### option C
- positive: adds a predictive quality measure to the KDA state. might encourage better state utilization
- negative: likely redundant with the main loss. adds computational cost for little benefit
- risk: low but expected impact is also low

## risk assessment

| option | implementation difficulty | risk to existing performance | expected benefit | biological fidelity |
|---|---|---|---|---|
| A (per-layer aux loss) | moderate | moderate | unknown | moderate (correct direction, wrong dynamics) |
| B (spike-as-surprise) | low | high | potentially high | high (genuine prediction error) |
| C (KDA state prediction) | low | low | low | low (redundant with existing objective) |

## recommendation

none of these options should be implemented before phase 5 is complete. the current priority is validating the existing architecture at 300M scale (phase 5 baseline, 5a, 5b). adding predictive coding mechanisms is a phase 6+ investigation.

if predictive coding is pursued, option B (spike-as-surprise) is the most interesting from a research perspective: it would create a genuine connection between todorov's ternary spikes and biological surprise signals. but it is also the highest-risk change, and the sequential processing requirement may make it impractical at scale.

option A (per-layer auxiliary loss) is the safest to try first: it does not change the forward pass, adds a well-understood regularization term, and can be disabled by setting lambda=0.

option C should be deprioritized: it is likely redundant with the existing training objective.

the strongest argument AGAINST adding predictive coding: Millidge et al. (2022) showed that backpropagation already computes the same weight updates as predictive coding at convergence. todorov is already trained by backpropagation. adding explicit predictive coding dynamics may therefore be redundant -- the learned representations already approximate what predictive coding would produce. the potential benefit is in the FORWARD DYNAMICS (iterative inference, precision-weighted errors), not in the learning rule. but iterative inference is expensive and incompatible with efficient autoregressive generation.

## key references

- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience, 2(1), 79-87.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
- Millidge, B., Tschantz, A. & Buckley, C. L. (2022). Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation, 34(6), 1329-1368.
- Whittington, J. C. R. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network. Neural Computation, 29(5), 1229-1262.
- Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology, 76, 198-211.

## see also

- [[predictive_coding]]
- [[free_energy_principle]]
- [[precision_weighting]]
- [[predictive_coding_vs_next_token]]
- [[plasticity_to_matrix_memory_delta_rule]]
- [[sparse_coding_to_ternary_spikes]]
- `wiki/tests/affect_gated_write_cpu_experiment.md` — precision-weighting the internal prediction error: surprise gates the descent memory's write (candidate F)
