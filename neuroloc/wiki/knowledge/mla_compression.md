# Multi-Head Latent Attention (MLA): Low-Rank KV Compression

status: current (as of 2026-04-17).

**naming note (2026-04-17)**: "MLA" / "Multi-Head Latent Attention" is an external paper name (DeepSeek, 2024). per the project's naming rule, the project's own component with the same mathematical form (low-rank projected dot-product attention with rotary positional encoding) is called **`compressed attention`** — see glossary in `wiki/PROJECT_PLAN.md`. this article is preserved as an explanation of the external paper the project drew from; it is NOT a description of the project's own current architecture.

Source papers:
- "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  ArXiv ID: 2405.04434
- "DeepSeek-V3 Technical Report"
  ArXiv ID: 2412.19437


## 1. Core Idea: Low-Rank KV Compression

Standard Multi-Head Attention (MHA) caches separate K and V vectors for each
head at every token position. MLA compresses all K and V information into a
single low-dimensional latent vector c_t per token.

    c_t^KV = W_DKV * h_t          (down-projection: d_model -> d_c)
    k_t    = W_UK  * c_t^KV       (up-projection: d_c -> n_h * d_h)
    v_t    = W_UV  * c_t^KV       (up-projection: d_c -> n_h * d_h)

Only c_t^KV (dimension d_c) is cached during inference, not the full K and V.


## 2. Key Dimensions and Compression Ratios

DeepSeek-V2/V3 hyperparameters:

    +-------------------+-------+
    | Parameter         | Value |
    +-------------------+-------+
    | n_h (num heads)   | 128   |
    | d_h (head dim)    | 128   |
    | d_c (compress dim)| 512   |
    | d_R (RoPE dim)    | 64    |
    +-------------------+-------+

Compression ratio calculation:
- Standard MHA KV cache per token: n_h * d_h * 2 = 128 * 128 * 2 = 32,768
- MLA KV cache per token: d_c + d_R = 512 + 64 = 576
- Compression factor: 32,768 / 576 ~ 57x (theoretical)
- Reported by DeepSeek: 93.3% KV cache reduction vs their 67B MHA model
- Actual storage: from 213.5 GB down to 7.6 GB for full-context KV cache


## 3. The RoPE Compatibility Problem

Rotary Position Embeddings (RoPE) cannot be applied to the compressed latent
because RoPE is applied between Q and K at attention time. If K is derived
from c_t^KV, then the up-projection W_UK cannot be absorbed into W_Q during
inference because the RoPE rotation matrix R_t lies between them:

    Attention score = q_t^T * R_t * W_UK * c_t^KV

Matrix multiplication is not commutative: R_t prevents absorption of W_UK.
This would force recomputation of K for all prefix tokens during generation.


## 4. Decoupled RoPE Solution

MLA introduces separate, dedicated RoPE projections:

    c_t^Q  = W_DQ  * h_t                    (query compression)
    q_t^C  = W_UQ  * c_t^Q                  (content query, no RoPE)
    q_t^R  = RoPE(W_QR * c_t^Q)             (RoPE query, dim d_R)

    k_t^C  = W_UK  * c_t^KV                 (content key, no RoPE)
    k_t^R  = RoPE(W_KR * h_t)               (RoPE key, dim d_R, shared)

Final Q and K are concatenations:
    q_t = [q_t^C ; q_t^R]
    k_t = [k_t^C ; k_t^R]

The RoPE component (d_R = 64) is small and cached separately alongside c_t^KV.
Total cache per token: d_c + d_R = 512 + 64 = 576 values.


## 5. Weight Absorption Trick for Inference

Since there is no nonlinearity between W_UQ, W_UK and the dot product:

    (q_t^C)^T * k_t^C = (W_UQ * c_t^Q)^T * (W_UK * c_t^KV)
                       = (c_t^Q)^T * (W_UQ^T * W_UK) * c_t^KV

The product W_UQ^T * W_UK can be precomputed and absorbed into a single matrix.
Similarly, W_UV can be absorbed into the output projection W_O.

This means at inference time:
- No need to explicitly compute full K and V vectors
- Attention operates directly on the compressed latents
- Saves both compute and memory for intermediate tensors

The absorption is only possible for the content (non-RoPE) components.
The RoPE components still require explicit computation due to the
position-dependent rotation matrices.


## 6. Comparison with MHA and GQA

    +-------------------+-----------+-----------+-----------+
    | Property          | MHA       | GQA       | MLA       |
    +-------------------+-----------+-----------+-----------+
    | KV cache/token    | 2*n_h*d_h | 2*n_g*d_h | d_c + d_R |
    | (DeepSeek dims)   | 32,768    | ~4,096    | 576       |
    | Compression ratio | 1x        | ~8x       | ~57x      |
    | Expressiveness    | Full      | Grouped   | Full*     |
    | RoPE compatible   | Native    | Native    | Decoupled |
    +-------------------+-----------+-----------+-----------+

    * MLA achieves expressiveness comparable to or exceeding MHA
      by learning the compression and decompression jointly.

    n_g = number of KV groups in GQA (typically n_h / group_size)
    For GQA with group_size=8: n_g = 128/8 = 16, cache = 2*16*128 = 4,096


## 7. Query Compression (Added in V2)

MLA also compresses queries with a similar low-rank scheme:

    c_t^Q = W_DQ * h_t            (d_model -> d_c')
    q_t   = W_UQ * c_t^Q          (d_c' -> n_h * d_h)

This reduces the activation memory during training, though it does not
affect inference KV cache size.


## 8. Performance Results

DeepSeek-V2 vs DeepSeek 67B (MHA baseline):
- 42.5% reduction in training cost
- 93.3% reduction in KV cache
- 5.76x improvement in maximum generation throughput
- Performance on benchmarks: equal or better than MHA


## 9. Adoption

MLA is used in:
- DeepSeek-V2 (May 2024)
- DeepSeek-V3 (December 2024)
- Kimi Linear (as the global attention component in hybrid layers)
- Multiple follow-up works: TransMLA (2502.07864), MHA2MLA-VLM (2601.11464)


## References

- DeepSeek-V2: arxiv 2405.04434
- DeepSeek-V3: arxiv 2412.19437
- TransMLA: arxiv 2502.07864
- Towards Economical Inference (MLA adaptation): arxiv 2502.14837
- MHA2MLA-VLM: arxiv 2601.11464
- Detailed explainer: https://planetbanatt.net/articles/mla.html
- Lior Sinai explainer: https://liorsinai.github.io/machine-learning/2025/02/22/mla.html


---

## 10. MLA Cache Memory at Long Context: 128K Scaling and Quantization

Research date: 2026-03-22


### 10.1 Per-Token Cache Memory Breakdown

MLA stores two latent vectors per token:

    +-------------------+-------+---------+----------+----------+
    | Component         | Dims  | FP16    | FP8      | Purpose  |
    +-------------------+-------+---------+----------+----------+
    | c_t^KV (NoPE)     | 512   | 1024 B  | 512 B    | Content  |
    | k_t^R  (RoPE)     | 64    | 128 B   | 128 B*   | Position |
    +-------------------+-------+---------+----------+----------+
    | Total per token   | 576   | 1152 B  | 640 B**  |          |
    +-------------------+-------+---------+----------+----------+

    * RoPE component is typically NOT quantized (kept in bf16)
      to preserve positional accuracy.
    ** FP8 NoPE (512 B) + bf16 RoPE (128 B) = 640 B per token.

For comparison, standard MHA (128 heads, d_h = 128):
- Per token: 128 * 128 * 2 (K + V) * 2 bytes = 65,536 B (FP16)
- MLA compression ratio vs MHA: 65,536 / 1,152 ~ 57x (FP16)

Source: https://mccormickml.com/2025/04/26/inner-workings-of-mla/
Source: arxiv 2405.04434


### 10.2 Total Cache Memory at 128K Context (DeepSeek-V3)

DeepSeek-V3 has 61 layers. Cache memory per layer at 128K tokens:

    FP16: 1,152 B * 128,000 = 147.5 MB per layer
    FP8:  640 B * 128,000 = 81.9 MB per layer

    Total 61 layers:
    FP16: 147.5 * 61 = 8.99 GB
    FP8:  81.9 * 61 = 5.0 GB

For comparison, standard MHA at 128K (61 layers, FP16):
    65,536 B * 128,000 * 61 = ~488 GB

    +-----------+----------+----------+
    | Precision | MLA      | MHA      |
    +-----------+----------+----------+
    | FP16      | ~9 GB    | ~488 GB  |
    | FP8       | ~5 GB    | ~244 GB  |
    +-----------+----------+----------+

The DeepSeek team reports: "from 213.5 GB down to 7.6 GB" for their
specific configuration. Minor discrepancies arise from how scale
factors and metadata are counted.

Sources:
- arxiv 2412.19437 (DeepSeek-V3 tech report)
- https://mccormickml.com/2025/02/12/the-inner-workings-of-deep-seek-v3/


### 10.3 vLLM FP8 MLA Cache Implementation

vLLM implements FP8 quantized MLA cache for DeepSeek models.

Per-token KV cache structure in vLLM (DeepSeek-V3.2-Exp):

    +---------------------------+--------+----------------------------+
    | Segment                   | Bytes  | Format                     |
    +---------------------------+--------+----------------------------+
    | NoPE latent (quantized)   | 512    | 512 x float8_e4m3          |
    | Scale factors             | 16     | 4 x float32 (1 per 128)   |
    | RoPE component            | 128    | 64 x bfloat16              |
    +---------------------------+--------+----------------------------+
    | Total                     | 656    |                            |
    +---------------------------+--------+----------------------------+

The NoPE latent is quantized to FP8 with block-wise scaling
(one float32 scale per 128 elements, 4 blocks total for 512 dims).
The RoPE component remains in bf16 for positional precision.

Memory capacity improvement: MLA + FP8 provides up to 9.6x more
KV cache capacity compared to standard attention in vLLM. On 8xH200,
token capacity expanded from 54,560 to 512,000 tokens.

Throughput: up to 3x throughput and 10x memory capacity improvement
with MLA + FP8 kernel optimizations in vLLM v0.7.1.

Sources:
- https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm
- https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html
- https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/


### 10.4 SGLang MLA + FP8 Implementation

SGLang implements FP8 MLA inference with weight absorption:

- W8A8 FP8 quantization for both weights and KV cache
- Batched Matrix Multiplication (BMM) for FP8 MLA with absorption
- Multiple attention backends: FlashAttention3, Flashinfer, FlashMLA,
  CutlassMLA, TRTLLM MLA (optimized for Blackwell), Triton

Source: https://docs.sglang.io/basic_usage/deepseek_v3.html


### 10.5 KV Cache Quantization Beyond FP8

Further cache compression options:

    +----------+-----------+-------------+------------------+
    | Format   | Bytes/val | vs FP16     | Accuracy loss    |
    +----------+-----------+-------------+------------------+
    | FP16     | 2.0       | baseline    | none             |
    | FP8      | 1.0       | 2x savings  | < 1%             |
    | INT8     | 1.0       | 2x savings  | < 1%             |
    | INT4     | 0.5       | 4x savings  | < 1%             |
    | NVFP4    | 0.5       | 4x savings  | < 1%             |
    +----------+-----------+-------------+------------------+

NVFP4 KV cache (NVIDIA, 2026):
- 50% reduction vs FP8 KV cache (effectively 4x vs FP16)
- Less than 1% accuracy loss on MMLU-PRO, MBPP, LiveCodeBench
- RULER 64K: 94.6% (vs 95.6% FP16, 95.5% FP8)
- Up to 3x improvement in time-to-first-token (TTFT) latency
- 20% higher cache hit rates compared to FP8

INT4 KV cache (HuggingFace quanto/HQQ backends):
- 2.5x memory savings vs FP16
- Negligible accuracy loss on tested benchmarks
- Supported formats: int2 (quanto), int4 (quanto + HQQ), int8 (HQQ)

KVQuant (arxiv 2401.18079):
- Enables up to 10M context on a single GPU via aggressive
  per-channel quantization of KV cache
- Uses per-channel quantization keys + per-token quantization values

Sources:
- https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- https://huggingface.co/blog/kv-cache-quantization
- arxiv 2601.04719 (GPU-accelerated INT8 KV cache)
- arxiv 2401.18079 (KVQuant)


### 10.6 MLA Cache Scaling for Small Models (Todorov Relevance)

For a smaller model (Todorov-scale), the MLA cache dimensions can be
adjusted proportionally:

    +-------------------+--------+---------+---------+
    | Config            | d_c    | d_R     | FP16/tok|
    +-------------------+--------+---------+---------+
    | DeepSeek-V3       | 512    | 64      | 1152 B  |
    | Reduced (4x)      | 128    | 64      | 384 B   |
    | Reduced (8x)      | 64     | 64      | 256 B   |
    +-------------------+--------+---------+---------+

For a Todorov model with d_c = 128, d_R = 64, at 128K context:
    FP16: 384 B * 128K = 49.2 MB per layer
    FP8:  256 B * 128K = 32.8 MB per layer (NoPE quantized)

With 6 MLA layers (assuming 24 total layers, 3:1 ratio):
    FP16: 49.2 * 6 = 295 MB total MLA cache
    FP8:  32.8 * 6 = 197 MB total MLA cache

This is well within the 16 GB T4 VRAM budget even at 128K context,
assuming model weights and KDA states leave sufficient headroom.

The key tradeoff: reducing d_c compresses the latent more aggressively,
which may reduce the expressiveness of the attention pattern. The
optimal d_c should be determined empirically by measuring perplexity
at each compression level.


### 10.7 Recommendations for Phase 2

1. Start with FP16 MLA cache at 4K-32K. Measure perplexity baseline.
2. Switch to FP8 NoPE + bf16 RoPE at 64K-128K if memory constrained.
3. Monitor perplexity delta between FP16 and FP8 cache at each length.
4. If FP8 degradation is < 1% perplexity, use FP8 throughout.
5. Track MLA cache memory as a function of context length and report
   the memory ceiling for each precision level on T4 (16 GB).
6. Consider INT4 cache only if FP8 is insufficient for 128K; the
   accuracy risk is higher for INT4 on small models.


### 10.8 Updated References

- vLLM quantized KV cache: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- NVFP4 KV cache: https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- vLLM DeepSeek-V3.2: https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html
- SGLang DeepSeek: https://docs.sglang.io/basic_usage/deepseek_v3.html
- KVQuant: arxiv 2401.18079
- GPU-accelerated INT8 cache: arxiv 2601.04719
- MLA inner workings: https://mccormickml.com/2025/04/26/inner-workings-of-mla/
- HF KV cache quantization: https://huggingface.co/blog/kv-cache-quantization
