# Kimi Delta Attention (KDA): Channel-Wise Gated DeltaNet

status: current (as of 2026-04-17).

**naming note (2026-04-17)**: "KDA" / "Kimi Delta Attention" is an external paper name (Moonshot AI, October 2025). per the project's naming rule, the project's own component with the same mathematical form (asymmetric outer-product state with per-step decay and optional write-side erasure) is called **`matrix memory`** — see glossary in `wiki/PROJECT_PLAN.md`. this article is preserved as an explanation of the external paper the project drew from; it is NOT a description of the project's own current architecture.

Source paper: "Kimi Linear: An Expressive, Efficient Attention Architecture"
ArXiv ID: 2510.26692
Authors: Moonshot AI (Kimi team)
Released: October 2025


## 1. Core Innovation: Channel-Wise Alpha Gate

Standard Gated DeltaNet (GDN) uses a scalar alpha per attention head.
KDA replaces this with a vector alpha -- one independent forgetting rate
per feature dimension (channel). This is the defining difference.

Scalar gating (GDN):
    S_t = alpha_t * S_{t-1}  + beta_t * v_t * k_t^T
    where alpha_t is a single scalar per head

Channel-wise gating (KDA):
    S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1}  +  beta_t * k_t * v_t^T
    where alpha_t is a d_k-dimensional vector, Diag(alpha_t) is a diagonal matrix


## 2. Full Recurrence Equation

The KDA state update combines three operations:

    S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1}  +  beta_t * k_t * v_t^T

Term breakdown:
- Diag(alpha_t) * S_{t-1}:  Channel-wise decay (fine-grained forgetting)
- (I - beta_t * k_t * k_t^T): Householder-like reflection for targeted erasure
- beta_t * k_t * v_t^T: Rank-1 write of new key-value association

Output computation:
    o_t = S_t * q_t

State size: d_k x d_v per head (d_k = d_v = 128 in Kimi Linear).


## 3. Parameter Computation

    q, k  = L2Norm(Swish(ShortConv(W_q/k * x)))
    v     = Swish(ShortConv(W_v * x))
    alpha = f(W_up_alpha * W_down_alpha * x)    # low-rank projection
    beta  = Sigmoid(W_beta * x)                  # scalar per token

Key details:
- q, k are L2-normalized for training stability (critical for delta-rule models)
- alpha uses a low-rank factorization to keep parameter count manageable
- ShortConv is a short causal 1D convolution (typically kernel size 4)
- The output goes through normalization and gating before output projection


## 4. DPLR Form for Efficient Training

The state transition matrix M_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) has
the structure of a Diagonal Plus Low-Rank (DPLR) matrix.

    M_t = Diag(alpha_t) - beta_t * k_t * (Diag(alpha_t)^T * k_t)^T

This DPLR structure enables a specialized chunkwise parallel algorithm:
- The sequence is split into chunks of length C
- Within each chunk: parallel scan using the DPLR form
- Across chunks: recurrent propagation of the d_k x d_v state

The KDA DPLR variant is more constrained than general DPLR, which reduces
computation while remaining consistent with the delta rule formulation.


## 5. Comparison: Scalar vs Channel-Wise Gating

    +-------------------+-------------------+---------------------+
    | Property          | GDN (scalar)      | KDA (channel-wise)  |
    +-------------------+-------------------+---------------------+
    | Gate granularity   | 1 value/head      | d_k values/head     |
    | Memory control     | Coarse            | Fine-grained        |
    | State capacity     | Limited           | Better utilization  |
    | Param overhead     | Minimal           | Low-rank alpha proj |
    | DPLR structure     | Simpler           | Specialized variant |
    +-------------------+-------------------+---------------------+

The channel-wise gate allows each feature dimension to independently decide
its retention/forgetting rate. This enables the model to:
- Retain long-range information in some channels
- Quickly update short-range features in other channels
- Use the finite-state memory more effectively overall


## 6. Training Stability

Key techniques for stable KDA training:
- L2 normalization of q, k vectors (prevents explosion in delta rule updates)
- Low-rank parameterization of alpha gate (reduces overfitting risk)
- Short causal convolution on q, k, v (provides local context before recurrence)
- Sigmoid activation for beta (bounds the erasure/write strength in [0,1])


## 7. Kimi Linear Architecture Context

KDA is deployed in a hybrid architecture:
- 3:1 ratio: 3 KDA layers per 1 global MLA (Multi-Head Latent Attention) layer
- Model size: 48B total parameters, 3B activated (MoE)
- Pretrained on 5.7 trillion tokens
- Context length: up to 1 million tokens
- KV cache reduction: up to 75% compared to full attention
- Decoding throughput: up to 6x faster at long contexts


## 8. Implementation

- Open-source KDA Triton kernel available
- Integrated into vLLM for inference
- Repository: https://github.com/MoonshotAI/Kimi-Linear
- Model weights: moonshotai/Kimi-Linear-48B-A3B-Instruct on HuggingFace


## References

- Kimi Linear paper: arxiv 2510.26692
- Gated Delta Networks (GDN): arxiv 2412.06464
- DeltaNet (original delta rule for linear attention): arxiv 2406.06484


---

## 9. KDA State Quality at Long Context: Saturation, Monitoring, Failure Modes

Research date: 2026-03-22


### 9.1 The Fundamental State Capacity Problem

Linear attention compresses all past tokens into a fixed-size state
matrix S of dimension d_k x d_v per head. As sequence length grows,
this finite memory must encode an increasing amount of information.
The core tradeoff: "the less memory the model consumed during
inference, the worse it did on associative recall."

Source: https://hazyresearch.stanford.edu/blog/2024-03-03-based

Numerical evidence (Based, 355M params, Pile dataset):
- Pure linear attention perplexity: 2.29
- Mamba perplexity: 2.21
- Transformer perplexity: 1.87
Linear attention underperforms transformers on recall-intensive tasks
because of the finite state capacity.

Source: arxiv 2402.18668 (Based paper)


### 9.2 State Collapse: The "Stuffed Mamba" Phenomenon

The most systematic study of state saturation failure is "Stuffed Mamba"
(arxiv 2410.07145, COLM 2025), which identifies "state collapse" (SC).

Definition of state collapse:
When the context length exceeds the training length, RNN hidden states
exhibit "abnormal behavior": a few dominant outlier channels develop
exploding values, while other channels vanish after normalization.

Key finding: the failure is NOT inability to retain memory, but
inability to FORGET irrelevant past tokens. Models trained on short
sequences with large state sizes never learn to forget because the
state can hold everything seen during training.

Detection metrics for state collapse:
1. Perplexity on synthetic prompts exceeding 2x the training-length
   perplexity
2. Passkey retrieval accuracy dropping below 95%
3. Mean/variance explosions in hidden state distributions across layers

Source: arxiv 2410.07145


### 9.3 State Size vs Training Length Relationship

Stuffed Mamba establishes a linear relationship between the minimum
safe training length and state size:

    T_train = 5.172 * S - 4.469

Where:
- T_train = minimum training length (in tokens) to avoid state collapse
- S = state size (dimension of the recurrent state)

For KDA with d_k = d_v = 128, the state per head is 128 x 128 = 16,384
values. Applying the formula: T_train ~ 5.172 * 128 - 4.469 ~ 657.

This means that for a state dimension of 128, training on sequences of
at least ~657 tokens should be sufficient to learn forgetting behavior.
Training on 4K tokens (Phase 1) is well above this threshold.

For passkey retrieval, capacity scales EXPONENTIALLY with state size.
A 370M model with appropriate training achieved near-perfect 256K
passkey accuracy. This is encouraging for KDA.

Source: arxiv 2410.07145


### 9.4 Delta Rule State Transition Stability

DeltaNet's transition matrix M_t = (I - beta_t * k_t * k_t^T) has
specific spectral properties:

Eigenvalues of M_t:
- In the direction of k_t: eigenvalue = (1 - beta_t * ||k_t||^2)
- All directions perpendicular to k_t: eigenvalue = 1
- For stability: all eigenvalues must have magnitude <= 1
- This requires ||k_t||^2 <= 2 (given 0 <= beta_t <= 1)

L2 normalization of k ensures ||k_t||^2 = 1, guaranteeing that all
eigenvalues are in [0, 1]. This is why L2 normalization of q and k
is critical for training stability in delta-rule models.

Source: arxiv 2406.06484, arxiv 2412.06464


### 9.5 KDA Channel-Wise Gating as Saturation Defense

KDA's channel-wise alpha gate (alpha_t in [0,1]^d_k) provides a
natural defense against state saturation at long contexts:

1. Per-channel forgetting rates: different feature dimensions can
   independently control their retention/forgetting rates. Semantic
   dimensions (syntax, recency, topic cues) can persist longer, while
   noisy or irrelevant channels decay rapidly.

2. Adaptive memory management: alpha_t acts as weight decay (L2
   regularization) on the fast-weight state matrix. High-alpha
   channels retain long-range information; low-alpha channels clear
   quickly to absorb new information.

3. KDA outperforms scalar gating at long context: KDA achieves 84.3%
   on RULER 128K, versus 81.3% for full MLA attention. This shows
   the channel-wise gating provides effective state management even
   at 128K tokens.

Source: arxiv 2510.26692
Source: https://www.emergentmind.com/topics/kimi-delta-attention-kda


### 9.6 DeltaProduct: Multi-Step Householder for Better Extrapolation

DeltaProduct (arxiv 2502.10297) improves upon DeltaNet by using
n_h Householder reflections per token instead of one.

Key properties:
- Spectral norm guarantee: each generalized Householder transformation
  has norm <= 1, so their product also has norm <= 1. This provides
  automatic stability without explicit regularization.
- Effective rank: state transition matrices are diagonal plus rank-n_h,
  compared to DeltaNet's diagonal plus rank-1.
- Length extrapolation: at n_h = 3, performance degradation is minimal
  across sequence lengths up to 16,384 tokens.
- State forgetting speed: DeltaProduct can clear state n_h times faster
  than DeltaNet (~43 tokens vs ~128 tokens for n = 128).

This is relevant to KDA because KDA's channel-wise gating provides a
complementary mechanism (selective per-channel forgetting) whereas
DeltaProduct provides richer state transitions (multiple Householder
steps per token). A future architecture could combine both.

Source: arxiv 2502.10297


### 9.7 Gate Saturation in Linear Attention

ReGLA (arxiv 2502.01578, NAACL 2025) identifies gate saturation as a
separate failure mode in gated linear attention:

Problem: sigmoid gating activations approach 0 or 1, causing vanishing
gradients: grad(g) = g * (1 - g) -> 0 at extremes. Once saturated,
gates cannot escape extreme activation regions.

ReGLA's solution: a "refining gate" r_t that interpolates between two
bounds, ensuring higher gradients near saturation regions:

    F_t = ((1 - r_t) * g_t^2 + r_t * (1 - (1 - g_t)^2)) * 1^T

This keeps F_t in [0, 1] while maintaining effective gradients.

Implication for KDA: KDA uses sigmoid for beta and low-rank projection
for alpha. If alpha channels saturate (all approach 1.0 = full retain
or 0.0 = full forget), the state loses adaptive control. Monitoring
the distribution of alpha values across channels and layers is a
useful diagnostic.

Source: arxiv 2502.01578


### 9.8 Practical State Quality Monitoring for Todorov Phase 2

Based on the research above, the following diagnostics should be
implemented during context extension experiments:

1. State norm tracking: log the Frobenius norm of S per head per layer
   at regular intervals during inference. Explosive growth indicates
   state collapse onset.

2. Alpha distribution: histogram the channel-wise alpha values per
   layer. If most channels cluster near 0 or 1, gating is saturated
   and the model has lost adaptive control.

3. Effective rank of S: compute the singular values of S and report
   the effective rank (number of singular values above a threshold,
   e.g., > 1% of the largest). Declining effective rank at long
   context indicates information compression/loss.

4. Perplexity vs context length: plot perplexity at 4K, 8K, 16K, 32K,
   64K, 128K. Stable or slowly increasing perplexity is expected;
   sudden spikes indicate state collapse.

5. Passkey retrieval depth sweep: test at each context length with
   passkey at depths 0%, 25%, 50%, 75%, 100%. If accuracy drops at
   specific depths, it reveals which parts of the sequence the state
   cannot retain.

6. Per-layer analysis: compare KDA layer outputs vs MLA layer outputs
   at long context to identify which layers are bottlenecks.


### 9.9 Known Failure Modes Summary

    +-----------------------------+------------------------------------+
    | Failure Mode                | Cause                              |
    +-----------------------------+------------------------------------+
    | State collapse              | Overparameterized state + short    |
    |                             | training length; never learns to   |
    |                             | forget. Detection: perplexity      |
    |                             | spikes, state norm explosion.      |
    +-----------------------------+------------------------------------+
    | Memory collision            | Fixed state fills with superimposed|
    |                             | key-value pairs; retrieval becomes |
    |                             | noisy. Detection: declining        |
    |                             | effective rank, passkey failures   |
    |                             | at specific depths.                |
    +-----------------------------+------------------------------------+
    | Gate saturation             | Sigmoid gates approach 0/1,        |
    |                             | vanishing gradients prevent        |
    |                             | adaptation. Detection: bimodal     |
    |                             | alpha distribution clustered at    |
    |                             | extremes.                          |
    +-----------------------------+------------------------------------+
    | Low-rank state              | State matrix converges to low rank |
    |                             | as uniform attention weight        |
    |                             | distribution emerges. Detection:   |
    |                             | singular value analysis.           |
    +-----------------------------+------------------------------------+

Sources:
- arxiv 2410.07145 (Stuffed Mamba, state collapse)
- arxiv 2502.10297 (DeltaProduct, spectral stability)
- arxiv 2502.01578 (ReGLA, gate saturation)
- arxiv 2402.18668 (Based, recall-throughput tradeoff)
- arxiv 2510.26692 (Kimi Linear, KDA)
- arxiv 2411.12537 (Negative eigenvalues in linear RNNs)
- https://hazyresearch.stanford.edu/blog/2024-03-03-based
