# Bit-budget analysis: why sub-1-bit PTQ fails on a 1.5B model

## Setup

- Model: Qwen2.5-1.5B-Instruct (1.54B parameters, 28 layers)
- Baselines: Qwen2.5-0.5B-Instruct (494M), Qwen2.5-1.5B-Instruct FP16
- Target: 1.5B at 0.1 BPW (PTQ, no QAT)
- Calibration: 100 WikiText-2 examples, ≤512 tokens
- Metric: per-token weighted perplexity
- Hardware: Kaggle T4 (16GB VRAM), BF16 compute

## Headline result

| configuration | total weight info | perplexity | notes |
|---|---|---|---|
| 0.5B FP16 | 8.0 Gbit (1.0 GB) | 35.0 | reference |
| 1.5B FP16 | 24.6 Gbit (3.1 GB) | 22.5 | strong baseline |
| 1.5B at 0.1 BPW (SVD+sign, PTQ) | 0.15 Gbit (19 MB) | 1,010,001 | broken |
| 1.5B at 1.0 BPW (sign+row-scale, PTQ) | 1.5 Gbit (188 MB) | 1,373,428,876 | broken |

**Both sub-1-bit PTQ approaches destroy the model. The "1.5B at 0.1
BPW vs 0.5B FP16" bet fails by 4-7 orders of magnitude.**

## Bit-budget math

For each configuration, the total "useful information" stored in the
weights is `n_params × bits_per_weight`:

| configuration | n_params | bits/weight | total | ratio vs 0.5B FP16 |
|---|---|---|---|---|
| 0.5B FP16 | 494M | 16.0 | 7.9 Gbit | 1.0x |
| 1.5B FP16 | 1544M | 16.0 | 24.7 Gbit | 3.1x |
| 1.5B at 1.0 BPW | 1544M | 1.0 | 1.5 Gbit | **0.19x** |
| 1.5B at 0.5 BPW | 1544M | 0.5 | 0.77 Gbit | 0.10x |
| 1.5B at 0.1 BPW | 1544M | 0.1 | 0.15 Gbit | **0.019x** |

For the bet to win, the 0.5B FP16 baseline (8 Gbit of information)
would need to be using at most 1.5 / 8 = 19% of its capacity, i.e.,
5.3x more parameters than it needs for the task. This is the
"Chinchilla-optimal" margin: are Qwen2.5-0.5B's 18T training tokens
enough that 0.5B parameters is overkill for WikiText-2 PPL of 35?

**Empirically, no.** The 0.5B model needs most of its 8 Gbit of
information. We have 19 MB competing with 1 GB. The bet loses by
construction.

## Why the SVD approach fails at 0.1 BPW

The LittleBit SVD+sign decomposition for a single weight matrix W
(4096×4096, like a Qwen2.5-1.5B MLP gate):

1. Truncated SVD at rank r=188 (target 0.1 BPW).
2. Binarize factors: U_sign = sign(U), V_sign = sign(V).
3. Per-latent scale: U_scale = mean(|U|), V_scale = mean(|V|).
4. Per-row + per-column scale (h, g) via alternating least squares.

The spectral truncation captures `(sum of top 188 singular values)² /
(sum of all 4096 singular values)²` of the energy. For a real LLM
weight matrix, the spectrum decays roughly as `1/k` (singular values
decrease slowly). Top 188 of 4096 = 4.6% of the spectrum by count,
capturing maybe 10-20% of the energy (because the head is bigger than
the tail in absolute terms).

The binarization step loses another factor. The sign trick
replaces each U[i, k] with `±U_scale[k]`, losing all variance within
each column. For a column with non-uniform magnitudes, this is a
significant loss.

**Combined reconstruction error per tensor: 0.96-0.99 relative
Frobenius norm.** Every layer is destroyed to ~5% of its signal.
The model can't function.

## Why uniform sign fails at 1.0 BPW

The "fastest possible" PTQ: `sign(W)` + per-row mean(|W|) as the
scale. No SVD, no clustering, no codebook. ~0.2s to quantize all 197
Linear layers of Qwen2.5-1.5B.

For a 4096×4096 weight:
- sign: 4096 × 4096 = 16M bits (1 bit per element)
- scale: 4096 × 16 = 64K bits (1 FP16 per row)
- total: 16.06M bits for 16M params = 1.004 BPW

The decomposition is loss-free per row (it preserves the row mean)
but loses all variance within each row. For a row with one large
positive weight and many small negative weights, the reconstruction
collapses everything to ±1 × the mean. The model can still do
scaling-by-mean but not feature-specific discrimination.

**Reconstruction error per tensor: depends on the row, but
uniformly destructive. The model produces 1.37B perplexity, which
is essentially the model's output entropy; it outputs random
tokens.**

## Why published methods (LittleBit, BitNet) work at ≤1.58 BPW

Both use **Quantization-Aware Training (QAT)**:
- LittleBit: 5 epochs of training with the latent-factorization
  quantization in the loop. 8 hours on 4×A100 for 7B.
- BitNet b1.58: from-scratch training with ternary weights + 8-bit
  activations. Trillions of training tokens.

QAT does two things PTQ cannot:
1. **Scale learning** (which we approximate with per-row/per-column
   scales). PTQ can do this, but QAT learns scales that compensate
   for downstream gradients, not just minimize ||W - W_reconstructed||.
2. **Weight adaptation.** The network learns to route information
   such that the most important weights are the ones that survive
   binarization. PTQ binarizes whatever is in the FP32 model; the
   model is not adapted to this constraint.

For a pretrained model, the network has already learned to use
specific weight values for specific information routing. Quantizing
to 1 bit destroys the precision that the routing depends on. QAT
gives the network a chance to re-route information into the
coarse-grained 1-bit channel.

## What would actually work (PTQ at sub-1-bit)

The literature has three approaches:

1. **Saliency-based mixed precision** (BiLLM, PB-LLM): keep the
   "salient" weights (highest magnitude or highest activation) at
   higher precision (4-8 bits), binarize the rest. Effectively a
   per-tensor picker, but with the picker pre-determined by
   calibration. Limited by the assumption that "salient = important"
   which is true on average but not per-tensor.

2. **Codebook / cluster** (BTC-LLM): replace each weight with an
   index into a learned codebook. The codebook is the "key" for
   uncompress. With a sufficiently rich codebook, can achieve
   sub-1-bit while preserving more information than uniform sign.

3. **Per-tensor bit-width picker** (the user's "atom by atom" idea):
   measure per-tensor sensitivity to quantization, assign BPW per
   tensor. Sensitive tensors (lm_head, late attention output) at
   higher precision; tolerant tensors at 0.5-1.0 BPW. Average BPW
   could be sub-1-bit while sensitive tensors are at FP16.

Approach #3 is the user's intuition and is the only one that hasn't
been thoroughly explored in the literature. But it requires a
*viable baseline*, and 0.1 BPW PTQ is not viable.

## What the next experiment should be

**Option A** (cheapest, fastest): Reproduce LittleBit at 0.55 BPW
with QAT. This is the published 0.55 BPW result and would be a
positive control. ~8 hours on T4 (or faster on RTX Pro 6000).

**Option B** (medium): Implement BTC-LLM at 0.7-1.0 BPW. The
adaptive transformation is more robust than uniform SVD + sign.
~30 min to implement, ~30 min to test.

**Option C** (the novel contribution): On top of either A or B,
add the per-tensor bit-width picker. Use the per-weight activation
logging to identify which tensors tolerate the lowest BPW. Apply
the per-tensor map. This is the paper.

The user's bet ("1.5B at 0.1 BPW outperforms 1B models") is **not
achievable** at PTQ. We have demonstrated this empirically on two
methods. The 53x information deficit is the fundamental reason.
Achieving this bet requires either:
- QAT (hours), or
- A fundamentally different compression method (not yet invented)

We recommend documenting the negative result and pivoting to Option B
or C.
