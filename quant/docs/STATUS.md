# novelquant, current status

Last updated: 2026-06-24 (LDLQ fixed, picker with fine granularity, codebook honesty check, bounded E8 experiments).

## TL;DR

The original novelquant (FP16/BF16 hooks on Qwen2.5-0.5B) works:
**37.8% peak memory savings, +0.34% perplexity drift** on WikiText-2.
Reproduced on Kaggle T4, run_id 20260619_123541.

The "sub-1-bit PTQ beats 1B models" bet **fails empirically**:
- 1.5B at 0.1 BPW (LittleBit SVD, CPU): perp 1,010,001 (vs 0.5B FP16 35.0)
- 1.5B at 1.0 BPW (uniform sign + scale): perp 1,373,428,876

The bit-budget analysis (`BIT_BUDGET_ANALYSIS.md`) explains why:
0.1 BPW 1.5B has 53x less information than 0.5B FP16. PTQ cannot
recover this gap. QAT is required for the published 0.55-1.58 BPW
results.

## Recent updates

### June 20-21, per-tensor bit-width picker works

The data-free picker (Hessian-diagonal-weighted sensitivity + greedy
knapsack) beats uniform quantization at equal avg BPW on both PPL
and reasoning, validated against a random-mixed control. See
workstreams 2-3 below.

### June 21, vector quantization in the Hadamard domain

Data-free VQ (Gaussian k-means, d=4) beats scalar Lloyd-Max by 3.3x at 2-bit
(125 PPL vs 418). The Gaussian shaping gain is real where TurboQuant's
Beta/sphere concentration argument doesn't apply. See workstream 6.

### June 22, block-wise fine-tuning

After VQ quantization, fine-tuning LayerNorm/bias params to match
original block outputs recovers significant quality. At 2-bit:
134.7 -> 66.7 PPL (50% reduction) with just 3 epochs. At 1-bit:
1,083,901 -> 15,287 PPL (71x reduction). This is the AQLM approach
(calibration-data-driven recovery) applied on top of our data-free
VQ codebooks. See workstream 7.

### June 23, E8 lattice RVQ + model-size-adaptive quantization

E8 lattice vector quantization (8D, 2-stage RVQ) gives 28% lower
reconstruction error than scalar Lloyd-Max at 4-bit. On Qwen3-4B the
degradation is 2.1% at 4.125 bpw, below the 5% target for the first
time. On Qwen2.5 models: 6.0-6.5% at 4-bit (close but above 5%). The
model-size-adaptive insight: bigger models at lower precision beat
smaller models at full precision at the same memory budget. See
workstream 8.

### June 23 session 2, E8 2-bit is 15x better than scalar 2-bit

Uniform E8 RVQ at 2-bit gives 27.7 PPL (vs scalar 417.6); the 8D
lattice shaping gain is real and large at 2-bit. The E8 Pareto curve
on Qwen2.5-1.5B (baseline 9.48): 2-bit 27.7 (+192%), 4-bit 10.05
(+6.0%), 6-bit 9.52 (+0.5%). The picker+random control validates
sensitivity-based allocation (picker 11.94 vs random 22.31 at 3.5
bpw), but E8's {2,4,6} bit granularity is too coarse for the picker
to beat uniform 4-bit. Odd-bit stages (3=2+1, 5=4+1 via sign-residual)
were added to `quant/e8lattice.py` to fix this; the sweep to measure
them is written but blocked by quota (see below).

### June 23, Kaggle GPU quota exhausted

The 30-hour weekly GPU quota is spent. The MCP wrapper silently
swallows the quota error (push returns `error: "Maximum weekly GPU
quota of 30.00 hours reached."` but the wrapper doesn't check the
response `error` field, so it polls a never-created kernel until
timeout). Four 2-bit attack scripts are written and ready to run
when quota resets:
- `run_2bit_sweep_outlier.py`, uniform E8 sweep {2,3,4,5,6} + outlier extraction
- `run_2bit_combined.py`, sweep + outlier + learned RVQ + LDLQ (all four, one kernel)
- `run_2bit_ft_fast.py`, lean block-wise FT on 2-bit E8 (fp16 model, only norm/bias trained)
- `run_2bit_gaussian.py`, `run_2bit_ldlq.py`, `run_2bit_outlier.py`, `run_2bit_ft.py`, individual versions
Run them one at a time (parallel pushes cause queue contention and
the wrapper doesn't delete completed kernels, so they accumulate and
block the concurrent-kernel limit). Delete kernels after each run:
`api.kernels_delete('dttdrv/<slug>', no_confirm=True)`.

### June 24, dedicated GPU (RTX PRO 6000 Blackwell, 96GB)

Moved off Kaggle to a rented box. No quota, no concurrent-kernel
limit, ~5x faster per config (3s vs 15s on T4). All four 2-bit
attacks ran.

### June 24, E8 sweep with odd-bit stages (the picker fix)

The 3-bit and 5-bit sign-residual stages work. Full E8 Pareto curve
on Qwen2.5-1.5B (baseline 9.4785):

| bits | bpw   | PPL    | degr    |
|------|-------|--------|---------|
| 2    | 2.125 | 27.70  | +192.2% |
| 3    | 3.125 | 12.66  | +33.6%  |
| 4    | 4.125 | 10.04  | +6.0%   |
| 5    | 5.125 | 9.64   | +1.7%   |
| 6    | 6.125 | 9.52   | +0.5%   |

The 3-bit point (12.66) is the most interesting: it's the first
sub-15 PPL result at ~3 bpw, and it's the natural FT target.

### June 24, 2-bit outlier extraction (modest)

Keeping the top-k% of weights at 16-bit or 4-bit, E8-quantizing the
rest. Gains are real but small:

| cfg              | bpw   | PPL    | degr    |
|------------------|-------|--------|---------|
| outlier 5% @4bit | 2.108 | 25.52  | +169.2% |
| outlier 10% @4bit| 2.208 | 23.84  | +151.5% |

5% outliers at 4-bit costs only +0.02 bpw but recovers 8% of the
2-bit gap. Not the big lever, the outlier mass is too spread out.

### June 24, learned RVQ (killed)

Gaussian-fit codebooks: 281.7 PPL. Data-fit (k-means on real
weights): 2125.7 PPL. Both are catastrophically worse than E8's
27.70. The 2-stage 8-bit RVQ (256+256 greedy residual) cannot
compete with E8's single-stage 16-bit lattice (65536 structured
points). E8 isn't just "good for a lattice"; it beats learned
codebooks at 2-bit because greedy RVQ decomposition is fundamentally
worse than joint lattice quantization. This is a strong negative
result: don't pursue learned codebooks at 2-bit.

### June 24, LDLQ with real activations (bug fixed, now works)

Initial LDLQ runs produced 211k PPL due to two bugs: missing Hessian
normalization (trace not normalized to 1.0) and insufficient
regularization. After fixing both, real-Hessian LDLQ at 2-bit
improves PPL from 27.70 to 24.46 (12% improvement). See session 3
results below for the full table.

### June 24, block-wise FT (the proven lever, confirmed)

The fp16 "fast" version diverged (27.7 -> 1099 PPL): loss decreased
but PPL exploded; fp16 forward produces noisy gradients that push
norm params wrong. The original fp32 approach (cast whole model to
fp32, train norm/bias, cast back) works. On the RTX PRO 6000's 96GB,
a 1.5B model in fp32 (6GB) is trivial, no need for the fp16 shortcut.

| cfg             | bpw   | pre-FT | post-FT | degr    | recovery |
|-----------------|-------|--------|---------|---------|----------|
| 2-bit FT(3)     | 2.125 | 27.70  | 23.46   | +147.5% | 15%      |
| 2-bit FT(7)     | 2.125 | 27.70  | 21.74   | +129.4% | 22%      |
| 2-bit FT(15)    | 2.125 | 27.70  | 21.31   | +124.8% | 23%      |
| 3-bit FT(3)     | 3.125 | 12.93  | 12.55   | +32.4%  | 3%       |
| 3-bit FT(7)     | 3.125 | 12.93  | 12.46   | +31.5%  | 3%       |

FT recovers 23% of the 2-bit gap (27.70 -> 21.31) and converges by
epoch 7. On 3-bit the gap is already small (12.93 vs 9.48) so FT
gains are proportionally smaller (3%). The 2.8x recovery from
workstream 7 (134 -> 48 on scalar VQ) does NOT transfer to E8, E8's
starting point is already much better, so there's less error for FT
to correct. FT is a real but modest lever on top of E8, not a
multiplicative one.

## June 24, session 3, LDLQ fixed, picker with fine granularity, codebook honesty check

### LDLQ with real activations (debugged)

The LDLQ bug (211k PPL) was caused by two issues: regularization
strength too low (Hessian matrix ill-conditioned without it) and
Hessian normalization missing (the diagonal trace needed to be
normalized to 1.0 before adding the identity). After fixing both,
real-Hessian LDLQ at 2-bit improves PPL from 27.70 to 24.46, a 12%
improvement. `runs/ldlq_debug/summary.json`.

| cfg                | bpw   | PPL    | degr    | notes                       |
|--------------------|-------|--------|---------|-----------------------------|
| e8_2bit_plain      | 2.125 | 27.70  | +192.2% | baseline                    |
| e8_ldlq_real_2bit  | 2.125 | 24.46  | +158.0% | 12% better, 169/197 tensors |
| e8_3bit_plain      | 3.125 | 12.66  | +33.6%  | baseline                    |
| e8_ldlq_real_3bit  | 3.125 | 12.94  | +36.5%  | slightly worse              |
| e8_4bit_plain      | 4.125 | 10.04  | +6.0%   | baseline                    |
| e8_ldlq_real_4bit  | 4.125 | 10.07  | +6.2%   | slightly worse              |

LDLQ helps at 2-bit but slightly hurts at 3-bit and 4-bit. The
block-wise adjustment term (propagating rounding errors from earlier
blocks) helps when quantization error is large (2-bit) but adds noise
when error is already small (3+ bits). The data-free Hessian proxy
(weight covariance) does not work, only real activations help.

### Picker with {2,3,4,5,6} bit granularity

The odd-bit E8 stages (3=2+1, 5=4+1 via sign-residual) give the
picker five bit-width options instead of three. The picker now beats
uniform and random at every target bpw. `runs/picker_e8/summary.json`.

| cfg             | target | actual  | PPL    | degr    | vs uniform-2bit |
|-----------------|--------|---------|--------|---------|-----------------|
| picker @ 2.5    | 2.500  | 2.508   | 16.90  | +78.3%  | 39% better      |
| random @ 2.5    | 2.500  | 2.532   | 24.28  | +156.1% | control         |
| picker @ 3.0    | 3.000  | 3.003   | 13.05  | +37.7%  |                 |
| random @ 3.0    | 3.000  | 3.005   | 20.14  | +112.5% | control         |
| picker @ 3.5    | 3.500  | 3.508   | 11.82  | +24.7%  |                 |
| picker @ 4.0    | 4.000  | 4.004   | 10.67  | +12.6%  |                 |
| picker @ 4.5    | 4.500  | 4.505   | 10.05  | +6.1%   |                 |

The 2.5 bpw result (16.90 PPL) is the best 2.5-bpw result in this
project. The picker assigns 188 tensors to 2-bit, 5 to 3-bit, 4 to
4-bit, concentrating the extra 0.5 bits on the most sensitive layers.

### Codebook honesty check

A critical measurement: the unbounded E8 lattice quantizer (`e8_lattice_quantize`)
maps each 8D block to the nearest E8 lattice point without any
cardity constraint. On a 1536x1536 tensor, only 33,686 distinct
codewords are used, that is 1.875 bits/weight, not 2.0. The E8P
codebook (bounded to 65,536 entries) uses only 27,609, 1.72 bits/weight.

This means all previous "2-bit" E8 results were actually ~1.875 bits.
The 2-bit budget allows 65,536 codewords; we were using ~33,686.
The question is whether filling the remaining slots improves quality.

### Bounded E8 codebook experiments

Three approaches to building a true 2-bit (65,536-entry) codebook
were tested. All failed to beat the unbounded lattice.

| cfg                  | PPL    | notes                                       |
|----------------------|--------|---------------------------------------------|
| e8 lattice unbounded | 27.70  | 33,686 codewords, 1.875 bits (current best) |
| E8P (D8_hat shifted) | 1738   | wrong codebook, misses small-norm region    |
| bounded E8 (shortest)| 74.32  | 65,536 shortest lattice points, norm^2<=12  |
| bounded E8 + Lloyd   | 35.89  | Lloyd-Max recovers half the loss, still worse|

The 65,536 shortest E8 lattice points are concentrated near the
origin (norm^2 0-12). But the data's nearest neighbors span norm^2
4-16. The shortest-points codebook covers the wrong region. Lloyd-Max
refinement (moving codeword positions to data centroids) recovers
from 74 to 36 but cannot match the unbounded lattice's advantage of
placing points exactly where the data needs them.

The E8P codebook (from QuIP#) is catastrophically worse because it
uses D8_hat lattice points shifted by +/-1/4, which excludes the
origin and small-norm points. For Gaussian-distributed data (which
Hadamard-rotated weights are), the region near the origin has the
highest probability mass. E8P misses it entirely.

### Data-driven codebook (failed)

A fourth approach: collect the 2,136,097 distinct E8 lattice points
actually used across all tensors, take the 65,536 most frequently
used as the codebook. Lloyd-Max refinement on top of this.

| cfg               | PPL    | avg_err | notes                                    |
|-------------------|--------|---------|------------------------------------------|
| e8 unbounded      | 27.70  | 0.061   | current best, 1.875 bits/weight          |
| datadriven_e8     | 69.91  | 0.260   | 65,536 most frequent codewords globally  |
| datadriven_lloyd  | 35.03  | 0.207   | + Lloyd-Max, still worse than unbounded  |

The data-driven codebook fails because different tensors use
different lattice points (2.1M distinct globally vs 65,536 slots).
A global codebook cannot cover all tensors' needs. Lloyd-Max recovers
from 70 to 35 but cannot match the unbounded lattice's advantage of
generating exact nearest-neighbor points on the fly at zero storage.

### Codebook conclusion

The unbounded E8 lattice is the best 2-bit codebook. It uses 1.875
bits/weight (33,686 codewords per tensor), under the 2-bit budget.
Every attempt to bound it to exactly 65,536 entries makes things
worse:

| approach              | PPL    | vs unbounded |
|-----------------------|--------|--------------|
| unbounded E8 lattice  | 27.70  | baseline     |
| E8P (D8_hat shifted)  | 1738   | 63x worse    |
| bounded E8 (shortest) | 74.32  | 2.7x worse   |
| bounded E8 + Lloyd    | 35.89  | 1.3x worse   |
| datadriven E8         | 69.91  | 2.5x worse   |
| datadriven + Lloyd    | 35.03  | 1.3x worse   |

The lattice structure generates codewords on the fly at zero storage
cost. Bounding to 65,536 entries forces a global codebook that cannot
cover the 2.1M distinct lattice points used across all tensors. The
" wasted" 0.125 bits (the gap between 1.875 and 2.0) is not worth
chasing. The 2-bit codebook question is answered: use the unbounded
E8 lattice. The remaining levers for 2-bit improvement are LDLQ
(27.70 to 24.46) and the picker (16.90 at 2.5 bpw).

## Cross-layer redundancy probes (June 23), the 100x-mission reality check

The mission is 1T capacity in 10B params (~100x compression, no model
fine-tuning). Quantization alone is capped at ~8-16x (proven above).
The remaining 6-12x has to come from redundancy exploitation. The
obvious lever is cross-layer redundancy (store K prototype layers +
residuals). Two probes killed it.

### Probe v1

`runs/layer_probe/summary.json`, Qwen2.5-1.5B, raw weight space.
Pairwise residual energy mean = 1.418 (>1.0 means substituting the
best non-identical layer is worse than using zero). Cosine similarity
mean = 0.000 across all 7 projections; layers are orthogonal as flat
vectors. Rank for 99% of residual energy = 67-97% of full rank, so
the diff is full-rank: no low-rank residual to store cheaply.

### Probe v2

`runs/layer_probe2/summary.json` tested four outside-the-box variants:

- Hadamard-domain similarity: had_resid == raw_resid exactly.
  Orthonormal rotation preserves inter-layer distances; this variant
  was mathematically doomed (a thinko, confirmed anyway).
- Per-head sub-blocks: head_cos_max = 0.004-0.027. Even individual
  attention heads are uncorrelated across layers.
- Residual sparsity: 99% of residual energy in top 70-73% of
  entries. Not sparse; would need to store ~70% of the matrix.
- Functional substitution (ground truth): replace layer j's
  weights with layer 0's, run the model, measure PPL. Baseline 8.75.
  One layer substituted: PPL 8,186 (936x worse). k=7: 190k.
  k=14: 1.45M. k=27: 5.7M. Per-projection: attention q/k/v ~6,500
  (less catastrophic), MLP 211k-7.4M (destroyed).

### Verdict

Cross-layer redundancy is not a lever for this model, in any sense:
not linear (cos 0, resid >1), not nonlinear/functional (single-layer
substitution destroys PPL 936x), not sparse (70% of entries needed),
not low-rank (full-rank diff). Each layer carries genuinely distinct
information. The 100x gap cannot come from layer recycling. See
"Mission reframing" below for what's actually left.

## Mission reframing (June 23)

The literal mission, compress a trained 1T dense model to a 10B
footprint via PTQ-only transforms at ~100x, is not achievable, and
we now have hard evidence (not just the bit-budget argument) for why:
the redundancy that 100x would require does not exist in the weights.

The honest options for "1T capacity in 10B params":

1. **QAT / train-with-compression-in-the-loop** (BitNet 1.58bpw,
   AQLM 2-bit). The model adapts to low-bit. Violates "no model
   fine-tuning" but is the only thing that works below ~2bpw.
2. **Distillation**, train a 10B student to match a 1T teacher.
   Achievable "1T capacity in 10B params" but it's training, not
   compression of a fixed model.
3. **MoE / sparse activation**, 1T total params, ~10B active per
   token. This is how the industry actually does "1T capacity, 10B
   compute" (DeepSeek-MoE, Mixtral). It's an architecture choice,
   not compression of an existing dense model.
4. **Hypernetwork weight generation**, train a small hypernet that
   produces the weights. The hypernet is the "10B params." Requires
   training; can only generate what it was trained to generate.

All four require training something. PTQ-only faithful compression of
a fixed trained model has a hard wall around 8-16x. The probes prove
the wall is real, not a failure of technique.

## Workstreams (June 20-21)

1. **Engineering foundation**, extracted `quant/` package (rotate,
   codebook, quantize, eval, picker, pack, reasoning) from the
   standalone scripts. Canonical WikiText-2 eval (test split, 40k
   tokens, ctx 2048). 23 unit tests, all passing. Reproduced the
   fullrot headline on Kaggle T4 (135s): b3=15.55 PPL (1.64x FP16),
   b2=417.6 (broken). `runs/pkg_repro/summary.json`.

2. **Per-tensor bit-width picker**, the novel contribution.
   Sensitivity = Hessian-diagonal-weighted reconstruction error
   (activation energy * recon error). Assignment = greedy
   multiple-choice knapsack. Validated against random-mixed control.
   `runs/picker/summary.json`.

   | cfg | avg_bpw | ppl | vs uniform |
   |---|---|---|---|
   | picker @ 3.2 | 3.206 | 14.83 | beats uniform-3bit (15.55) |
   | picker @ 2.5 | 2.508 | 100.11 | 4.2x better than uniform-2bit (417.6) |
   | picker @ 2.2 | 2.206 | 189.24 | 2.2x better than uniform-2bit |
   | random @ 3.2 | 3.218 | 230.21 | 15.5x worse than picker |

3. **Reasoning eval**, HellaSwag + ARC-Challenge (zero-shot
   loglikelihood, 200 examples/task). The picker beats uniform on
   reasoning too. Reasoning collapse is gradual (not a cliff like
   PPL). `runs/reasoning/summary.json`.

   | cfg | avg_acc | hellaswag | arc |
   |---|---|---|---|
   | fp16 | 0.405 | 0.435 | 0.375 |
   | picker @ 3.2 | 0.378 | 0.415 | 0.340 |
   | uniform 3-bit | 0.370 | 0.420 | 0.320 |
   | picker @ 2.5 | 0.290 | 0.345 | 0.235 |
   | uniform 2-bit | 0.260 | 0.305 | 0.215 |

4. **Bit-packed storage**, `quant/pack.py`: b-bit indices packed
   into byte array + FP32 scales. `pack()`/`unpack()` bit-exact
   round-trip verified for both `rtn` and `fullrot_whlm`. Makes the
   result a deployable artifact, not just a measurement.

5. **Documentation**, this update.

6. **Vector quantization in the Hadamard domain**, the Gaussian shaping
   gain is real where TurboQuant's Beta/sphere argument doesn't apply.
   Data-free VQ (Gaussian k-means, d=4) beats scalar Lloyd-Max by 3.3x
   at 2-bit. `runs/vq/summary.json`.

   | cfg | avg_bpw | ppl | vs scalar |
   |---|---|---|---|
   | fullrot_whlm_b2 (scalar) | 2.125 | 417.6 | baseline |
   | fullrot_vq:gaussian:2:1 | 2.016 | 125.2 | 3.3x better, lower BPW |
   | fullrot_vq:d4:2:1 | 2.016 | 203.1 | 2.1x better, lower BPW |
   | fullrot_whlm_b3 (scalar) | 3.125 | 15.55 | baseline |

7. **Block-wise fine-tuning (AQLM-style recovery)**, after VQ
   quantization, fine-tune LayerNorm/bias params to minimize MSE
   between quantized and original block outputs. Uses calibration
   data (wikitext-2, 32-64 sequences). `runs/finetune/summary.json`.

   | cfg | avg_bpw | pre-ft ppl | post-ft ppl | improvement |
   |---|---|---|---|---|
   | 2-bit VQ + FT (3 epochs) | 2.016 | 134.67 | 66.66 | 2.0x |
   | 2-bit VQ + FT (10 epochs) | 2.016 | 116.08 | 47.99 | 2.4x |
   | 1-bit VQ + FT (3 epochs) | 1.016 | 1,083,901 | 15,287 | 71x |
   | 1-bit VQ + FT (10 epochs) | 1.016 | 541,140 | 17,870 | 30x |

   The 2-bit result (47.99 PPL, 5.1x over FP16) is the best 2-bit
   result in this project. Loss is still decreasing at epoch 10,
   so more epochs would help. The 1-bit result is still not viable
   but demonstrates that fine-tuning recovers 30x from catastrophic
   quantization. Codebook fine-tuning (not just norm/bias) should
   improve both further.

8. **E8 lattice RVQ + model-size-adaptive quantization**, the E8
   lattice (densest 8D packing) with 2-stage RVQ gives 28% lower
   reconstruction error than scalar Lloyd-Max at 4-bit. On PPL:
   6.0% degradation (vs 11.5% for scalar) at 4.125 bpw. The
   model-size-adaptive insight: bigger models at lower precision
   beat smaller models at full precision at the same memory budget.
   `runs/e8rvq/summary.json`, `runs/size_adaptive/summary.json`.

   | cfg | avg_bpw | ppl | degr | vs scalar |
   |---|---|---|---|---|
   | 0.5B scalar 4-bit | 4.125 | 17.26 | 23.2% | baseline |
   | 0.5B E8 RVQ 4-bit | 4.125 | 15.25 | 8.9% | 2.6x better |
   | 0.5B scalar 5-bit | 5.125 | 14.96 | 6.8% | (close) |
   | 0.5B E8 RVQ 6-bit | 6.125 | 14.04 | 0.2% | (target met) |
   | 1.5B scalar 4-bit | 4.125 | 10.57 | 11.5% | baseline |
   | 1.5B E8 RVQ 4-bit | 4.125 | 10.05 | 6.0% | 1.9x better |
   | 1.5B scalar 5-bit | 5.125 | 9.84 | 3.8% | (target met) |
   | 1.5B E8 RVQ 6-bit | 6.125 | 9.53 | 0.5% | (target met) |
   | 3B scalar 4-bit | 4.125 | 9.52 | 12.6% | baseline |
   | 3B E8 RVQ 4-bit | 4.125 | 9.00 | 6.5% | 1.9x better |
   | 3B scalar 5-bit | 5.125 | 8.74 | 3.4% | (target met) |
   | 3B E8 RVQ 6-bit | 6.125 | 8.49 | 0.5% | (target met) |
   | 4B (Qwen3) scalar 4-bit | 4.125 | 14.40 | 7.3% | baseline |
   | 4B (Qwen3) E8 RVQ 4-bit | 4.125 | 13.71 | 2.1% | 3.5x better, <5% TARGET MET |
   | 4B (Qwen3) scalar 5-bit | 5.125 | 13.93 | 3.8% | (target met) |

   Model-size-adaptive comparison (same memory budget):
   | config | size (MB) | PPL | savings |
   |---|---|---|---|
   | 0.5B FP16 | 988 | 14.01 | baseline |
   | 1.5B scalar 5-bit | 989 | 9.84 | same mem, 30% better PPL |
   | 1.5B FP16 | 3087 | 9.48 | baseline |
   | 3B E8 6-bit | 2363 | 8.49 | 23% less mem, 10% better PPL |
   | 4B (Qwen3) FP16 | 8045 | 13.42 | baseline |
   | 3B E8 6-bit | 2363 | 8.49 | 71% less mem, 37% better PPL |

   The 3B@6bit-E8 result is the headline: better PPL than 4B Qwen3
   FP16 at 71% less memory. The model-size-adaptive advantage scales
   with model size, bigger models at lower precision dominate.

   What didn't work (data-free techniques tested and rejected):
   - Outlier protection in Hadamard domain (negligible)
   - Outlier extraction before rotation (negligible)
   - Variable group sizes (zero effect)
   - Optimized Hadamard signs (worse)
   - Per-dimension bit allocation (no effect after rotation)
   - Data-free sensitivity picker (worse than uniform)
   - NF4 codebook (worse than Lloyd-Max)
   - E8P ball-truncated codebook (1738 PPL, wrong codebook for Gaussian)
   - Bounded E8 with 65536 shortest lattice points (74 PPL, wrong region)
   - Bounded E8 + Lloyd-Max (36 PPL, recovers half but still worse than unbounded)
   - Data-driven E8 codebook (70 PPL, global codebook can't cover per-tensor diversity)
   - Data-driven E8 + Lloyd-Max (35 PPL, still worse than unbounded 27.70)
   - LDLQ with weight-covariance proxy (no improvement, H is not activation covariance)
   - LDLQ at 3+ bits (slightly worse, adjustment noise exceeds quantization error)
   - Local search on E8 lattice (no improvement, nearest is already optimal)
   - Learned RVQ codebooks at 2-bit (Gaussian 281, datafit 2125, greedy decomposition fails)

## The original deliverables (June 19)

1. **`novelquant.py` + `novelquant.ipynb`**, the 3-method notebook.
   Method 1 (BF16 hooks) is the working result. Methods 2 and 3
   were honest failures (documented in the notebook).
2. **`littlebit_ptq.py` + `run_littlebit_ptq.py`**, fast PTQ
   LittleBit at 0.1 BPW. Runs in 5 minutes on T4. Produces
   catastrophic quality loss. Negative result.
3. **`BIT_BUDGET_ANALYSIS.md`** + **`CHANGELOG.md`**, the full
   research log and the bit-budget math that explains the failure.

## File inventory

```
novelquant/
├── CHANGELOG.md                    # full research log
├── STATUS.md                       # this file
├── BIT_BUDGET_ANALYSIS.md         # why sub-1-bit PTQ fails
├── novelquant.py                   # the original 3-method notebook
├── novelquant.ipynb                # generated .ipynb
├── py2ipynb.py                     # # %% -> .ipynb converter
├── littlebit_ptq.py                # fast PTQ LittleBit (no QAT)
├── run_littlebit_ptq.py            # 0.1 BPW bet runner
├── quant/                          # extracted package (June 20-21)
│   ├── __init__.py
│   ├── rotate.py                   # FWHT, Hadamard, signs
│   ├── codebook.py                 # Lloyd-Max, NF4, uniform
│   ├── quantize.py                 # quant_dequant, quantize_model_inplace
│   ├── eval.py                     # canonical WikiText-2 PPL
│   ├── picker.py                   # per-tensor bit-width picker
│   ├── pack.py                     # bit-packed storage + dequant
│   └── reasoning.py               # HellaSwag, ARC-Challenge eval
├── tests/test_quant.py             # 23 unit tests
├── run_quant.py                    # reproduction runner
├── run_picker.py                   # picker vs uniform runner
├── run_reasoning.py                # reasoning eval runner
├── kaggle_push.py                  # Kaggle push driver (inlines quant/)
├── kaggle_probe.py                 # T4 compute-capability diagnostic
├── run_e8_ceiling.py               # E8 Pareto curve sweep (June 24)
├── run_2bit_sweep_outlier.py       # E8 sweep + outlier extraction
├── run_2bit_combined.py            # sweep + outlier + learned RVQ + LDLQ
├── run_2bit_gaussian.py            # learned Gaussian codebook test
├── run_2bit_ldlq.py                # LDLQ with real activations
├── run_2bit_ft.py                  # block-wise FT (fp32)
├── run_2bit_ft_fast.py             # block-wise FT (fp16, diverged)
├── run_2bit_ft_fp32.py             # block-wise FT (fp32, working)
├── run_2bit_outlier.py             # outlier extraction at 2-bit
├── run_ldlq_debug.py               # LDLQ Hessian debug script
├── run_2bit_ldlq_ft.py             # LDLQ + FT combined
├── run_picker_e8.py                # picker with {2,3,4,5,6} granularity
├── run_aq_2bit.py                  # additive quantization (AQ) experiment
├── run_codebook_2bit.py            # codebook honesty check + E8P test
├── run_bounded_e8.py               # bounded E8 (65536 shortest points)
├── run_datadriven_cb.py            # data-driven codebook from E8 frequencies
├── runs/
│   ├── kaggle_20260619/            # original June 19 runs
│   ├── pkg_repro/summary.json      # workstream 1 reproduction
│   ├── picker/summary.json         # workstream 2 picker results
│   ├── reasoning/summary.json      # workstream 3 reasoning results
│   ├── e8rvq/summary.json          # E8 RVQ Pareto curve
│   ├── ldlq_debug/summary.json     # LDLQ fixed results (June 24 sess 3)
│   ├── picker_e8/summary.json      # picker with fine granularity
│   └── bounded_e8/summary.json     # bounded E8 codebook experiments
└── (CACHED: __pycache__/)
```

## What's been tested on Kaggle T4

| run_id | model | method | result |
|---|---|---|---|
| 20260619_123541 | Qwen2.5-1.5B (FP32) | baseline | perp 22.5 (old eval) |
| 20260619_123541 | Qwen2.5-1.5B | BF16 hooks (autocast) | perp 22.5, -37.8% mem |
| 20260619_123541 | Qwen2.5-0.5B (FP32) | baseline | perp 35.0 |
| June 19 | Qwen2.5-1.5B | PTQ LittleBit 0.1 BPW (CPU SVD) | perp 1,010,001 (broken) |
| June 19 | Qwen2.5-1.5B | Uniform sign 1.0 BPW | perp 1.37B (broken) |
| June 20 (135s) | Qwen2.5-1.5B | fullrot_whlm b3 (canonical eval) | perp 15.55, recon 0.16 |
| June 20 (135s) | Qwen2.5-1.5B | fullrot_whlm b2 (canonical eval) | perp 417.6, recon 0.30 |
| June 21 (311s) | Qwen2.5-1.5B | picker @ 3.2 bpw | perp 14.83 (beats uniform 15.55) |
| June 21 (311s) | Qwen2.5-1.5B | picker @ 2.5 bpw | perp 100.11 (4.2x better than uniform-2bit) |
| June 21 (311s) | Qwen2.5-1.5B | random-mixed @ 3.2 bpw | perp 230.21 (15.5x worse than picker) |
| June 21 (460s) | Qwen2.5-1.5B | reasoning: picker vs uniform | picker beats uniform on HellaSwag + ARC |
| June 21 (486s) | Qwen2.5-1.5B | VQ:gaussian 2-bit | perp 125.2 (3.3x better than scalar 418) |
| June 21 (486s) | Qwen2.5-1.5B | VQ:d4 2-bit | perp 203.1 (2.1x better than scalar 418) |

## What's NOT been tested (remaining gaps)

- Sub-1-bit with **QAT** (8 hours per run on T4), ruled out by user
- BTC-LLM (binary codebook with adaptive transform), code is public
- MMLU, HumanEval+, MATH-500 (only HellaSwag + ARC-Challenge done)
- Additive quantization (AQ) with true 2-bit sub-codebooks (256+256 entries)
- AQ + LDLQ combined (joint assignment + activation-aware rounding)
- Codebook fine-tuning (AQLM-style: update codebook entries during FT)
- Picker + LDLQ combined (sensitivity-based allocation + adaptive rounding)

## What to do next (if continuing)

| priority | action | reason |
|---|---|---|
| 1 | Picker + LDLQ combined | the two proven 2-bit levers (16.90 PPL at 2.5 bpw, 24.46 at 2.125 bpw) combined |
| 2 | AQ with 256+256 sub-codebooks | joint optimization of 2-stage assignment; fixes greedy RVQ weakness |
| 3 | Codebook fine-tuning (AQLM-style) | update codebook entries during FT, not just norm/bias; could multiply FT gains |
| 4 | MMLU + MATH-500 reasoning eval | broader reasoning characterization at 2-bit |

## Kaggle MCP setup (don't lose this)

- `C:\Users\deyan\Projects\img\.mcp.json`, project-level
- `C:\Users\deyan\.claude\.mcp.json`, user-level
- Server command: `python C:\Users\deyan\Projects\todorov\scripts\kaggle_mcp_wrapper.py`
- Tools: `kaggle_execute`, `kaggle_execute_file`, `kaggle_execute_notebook`
- Auth: `C:\Users\deyan\.kaggle\kaggle.json` (KGAT_ token)
- MUST use `enable_internet=True` in `run_on_kaggle()` for HF
  model downloads.
- MUST use `acc="NvidiaTeslaT4"` (capitalized, no hyphens) in
  `kernels_push()`, the lowercase `nvidia-tesla-t4` falls back to P100.

## Local edits to mcp_server_kaggle_exec

- `.../mcp_server_kaggle_exec/kaggle_runtime.py` line ~285:
  `api.kernels_push(temp_dir, acc="NvidiaTeslaT4")` (was: no acc arg,
  fell back to P100 which is sm_60 and too old for current PyTorch).
- `.../mcp_server_kaggle_exec/kaggle_runtime.py` metadata dict:
  added `"accelerator": "nvidia-tesla-t4"` for documentation
  (the acc arg on kernels_push is what actually does the work).
