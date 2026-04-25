# compression beyond quantization

status: current (as of 2026-04-25).

## the mistake worth naming

the neural network compression literature overwhelmingly treats memory compression as a quantization problem. turboquant (zandieh et al., google research, arxiv 2504.19874, to appear iclr 2026) computes the information-theoretic lower bound for vector quantization at bit-width b as mse ~ 1/4^b and gets within a constant factor ~2.7 of that bound; the paper reports near-lossless longbench-e performance at 3.5 bits per value (author-claimed ~4.5x reduction at zero degradation). kvtc (nvidia research, arxiv 2511.01815) adds pca decorrelation, dynamic programming bit allocation, and deflate entropy coding to reach author-claimed up to ~20x kv cache reduction at under 1 point accuracy drop across longbench, math-500, and ruler (specific numbers require direct table verification). recent structural kv compression work includes vqkv (arxiv 2603.16435, march 2026, author-reported ~5.7x on llama-3.1-8b with 98.6 percent longbench retention), commvq (apple ml, arxiv 2506.18879, author-reported ~8x at 2-bit), vqllm (kumar et al., neurips enlsp-iv 2024, arxiv 2410.15704, author-reported ~5.5x), and deltakv (arxiv 2602.08005, february 2026, author-reported ~3.45x via semantic residuals — one of the few papers that structurally compounds two biology-inspired mechanisms, predictive filtering plus content-addressable reference). deepseek-v2 low-rank attention (arxiv 2405.04434) uses low-rank projection, with the deepseek-v3 technical report (arxiv 2412.19437) reporting large kv cache reductions at 670b scale. these are strong engineering results, but they are all operating on the same wrong abstraction: **they assume memory is a fixed amount of content that must be stored, and the question is how many bits per value**.

the brain does not operate at this level. a human is exposed to roughly 10^17 bytes of raw sensory experience across an 80-year lifetime (30 frames per second of megapixel retinal input, compounded with auditory, tactile, and proprioceptive streams; this is a back-of-envelope estimate, not a measurement). bartol et al. 2015 measured ~4.7 bits per synapse in hippocampal ca1 via serial em reconstruction; extrapolating across the ~10^14 synapses in the human brain gives a synaptic substrate capacity of approximately 10^14 to 10^15 bits (roughly 10^13 bytes), most of which is used for non-content functions (timing, attention control, motor programs). the effective content memory actually retained is on the order of 10^5 to 10^7 bytes (this range comes from cognitive-science estimates of lifetime retrieval-eligible knowledge, not direct measurement; figures above 10^7 require assumptions about near-zero marginal cost for schema-consistent storage that are not directly measured). the resulting effective compression ratio for content is roughly 10^10 to 10^12, which is nine to eleven orders of magnitude beyond any published bit-quantization method.

the brain does not achieve this by quantizing better. the brain achieves this by **not storing the content at all**. it stores a learned model of the world and a set of sparse indices into that model. at retrieval time, the model reconstructs the content. the memory is the model parameters, not a separate cache. this article makes the case that todorov's memory design should move in the same direction, and enumerates the specific mechanisms.

role in the compression cluster:
- this file = the current project thesis and doctrine
- `knowledge/generative_memory_research.md` = the literature shelf and prior-art evidence
- `bridge/memory_compression_to_tiered_architecture.md` = the concrete five-tier architecture translation

## what the brain actually compresses

the brain's compression is not one mechanism. it is at least six mechanisms compounding. each mechanism alone is modest; compounded, they produce the observed gap.

### 1. predictive filtering (estimated 10^2 to 10^4 x)

cortex maintains a hierarchical generative model. each level predicts the activity of the level below. only prediction errors propagate upward (rao and ballard 1999; friston 2010). the consequence for memory: only information that deviates from the model's predictions is eligible for storage. predictable input is suppressed before it reaches long-term encoding.

the estimated compression factor here is derived, not measured: if a learned model predicts 99 percent of tokens correctly given context, then 99 percent of tokens carry no new information and need not be stored, giving a compression ratio around 10^2 for typical natural-language contexts. the upper bound 10^4 assumes a near-perfect model on highly redundant domains like technical writing and has not been directly measured in any published benchmark. treat these as order-of-magnitude estimates, not as established constants.

no current transformer architecture explicitly implements predictive filtering of memory writes. kv caches store all tokens, regardless of whether each token was predicted or surprising. titans (google 2025, arxiv 2501.00663) implements surprise-gated long-term memory updates via gradient magnitude, which is the closest published work. it is necessary but not sufficient.

### 2. generative replacement (estimated 10^3 to 10^4 x)

the brain does not store "what my kitchen looks like" as a pattern of retinal activations. it stores a sparse code that, when activated, causes the visual cortex to reconstruct a scene using its learned generative model. the generative model is permanent and shared across millions of retrievals. the code per memory is small (estimated a few hundred bits for a single scene reconstruction in hippocampus; figure from indexing-theory estimates, not directly measured).

the engineering analog is neural implicit representations (nerf, mildenhall et al. 2020; siren, sitzmann et al. 2020) and world models (dreamer v3, hafner et al., nature 2025, arxiv 2301.04104). both store entire signals in the weights of small mlps. retrieval is a forward pass at query coordinates. inrs compress 3d scenes from hundreds of megabytes to single megabytes (author-reported ratios 10^3-10^5). dreamer v3 compresses hours of rl environment observation into a learned world model plus discrete latent codes of ~160 bits per timestep.

no published architecture does this for autoregressive language memory. the analog for our case would be: store a learned decoder network, and for each "memory" store only a small latent code that the decoder reconstructs into a full hidden state at retrieval time. the memory is the decoder (~tens of mb, shared across all memories) plus the codes (~tens of bytes per memory). compression ratio depends on decoder quality.

### 3. schema-delta encoding (estimated 10^2 to 10^3 x)

schema theory (bartlett 1932; mcclelland, mcnaughton, o'reilly 1995) holds that repeated experiences (restaurants, morning routines, work meetings) are stored as schemas — reusable templates — with specific episodes encoded as small deviations from the template. when you remember "dinner at my parents' house last week," you reconstruct it by instantiating the "dinner at parents" schema and applying a small delta for what was specifically different this time. the compression factor here is derived from the intuition that the schema is stored once and reused across many episodes; direct measurement in biological memory is not available.

the engineering analog is delta encoding with a learned prior. each domain has a base model; each specific episode is a small residual. this is structurally identical to lora (hu et al. 2021), but applied to memory content instead of model adaptation. the compression ratio depends on the schema's coverage: tighter schemas give bigger deltas to larger coverage, so the choice of schema granularity is a design parameter.

no published memory architecture learns schemas in this structured way. the closest is compositional memory work (schema networks, kansky et al. 2017) which operates at the raw observation level, not the hidden-state level.

### 4. content-addressable indexing (estimated 10^2 to 10^4 x)

the hippocampus, per indexing theory (teyler and discenna 1986), does not store the content of episodic memories. it stores pointers — small indices — into cortical representations that encode the content. typical back-of-envelope estimates (from follow-on indexing-theory literature, not from direct measurement) put a cortical representation at roughly 10^6 bits and a hippocampal index at roughly 10^2 to 10^4 bits, implying a compression factor of roughly 10^2 to 10^4 per stored memory. these numbers are order-of-magnitude estimates and should not be taken as measured constants.

the engineering analog is a learned hash map whose keys are semantic addresses and whose values are regenerated at retrieval time by the main model. this is strictly smaller than a verbatim cache, and scales independently of hidden-state size.

memorizing transformers (wu et al. 2022) is often cited as the deep learning version of indexing, but it stores full kv pairs with the keys, which is a cache, not an index. larimar (das et al. 2024) is closer: it stores in a fixed-size memory matrix and retrieves by associative lookup. no published system implements the combination "small pointer + regenerated content" for language memory.

### 5. algorithmic / kolmogorov abstraction (variable, typically 10 to 10^2 x)

faces, objects, scenes, sentences, concepts are not stored at the pixel or phoneme level. they are stored as points on learned manifolds. a face is roughly 200-dimensional in its native embedding space, not 10^6 pixels. a sentence is roughly 1000-dimensional in contextual embedding, not 10^4 characters. the brain learns these manifolds during development and stores memories as coordinates on them.

the engineering analog is latent diffusion, vq-vae, and similar learned latent compression schemes. stable diffusion stores images as ~4000-dimensional latents that decode to 512x512 rgb images (approximately 250x compression before perceptual quality loss). codec models like encodec (défossez et al. 2022) achieve 50x-300x compression on audio via residual vector quantization.

no published memory system applies manifold compression to transformer hidden states at scale. the vq-vae literature is about input compression, not memory compression, and the difference matters: transformer hidden states are higher-dimensional and less natural-signal-like than images or audio.

### 6. reconsolidation + lossy reconstruction

every time a memory is retrieved, it is reconstructed and rewritten in the lossy-reconstructed form (nader, schafe, ledoux 2000; sevenster et al. 2014). the act of retrieval modifies the memory. this has two consequences: memories drift toward the current model's priors over time (gist memory, bad for verbatim accuracy, good for compression), and errors accumulate at boundaries where prediction and retrieval disagree (prediction error gates reconsolidation).

the engineering consequence is that there is no persistent verbatim store. every retrieval is a forward pass. what is "stored" is a sparse cue plus the current model state. old memories that have not been retrieved for a long time drift toward the model's generative prior and become indistinguishable from novel generations.

no published memory architecture implements reconsolidation as a standard operation. it is sometimes simulated via "memory distillation" (e.g., distilling old memories into a student model) but not as a per-retrieval update rule.

## the compounding argument

any single one of these mechanisms achieves modest compression, in the range of 10 to 10^4 x, roughly comparable to what bit-quantization methods achieve. the breakthrough is that they compound.

a back-of-envelope calculation for a lifetime of raw sensory experience (each multiplier here is an order-of-magnitude estimate, not a measured constant):

```
raw sensory input:                              10^17 bytes
after predictive filtering (~100x estimated):   10^15 bytes
after generative replacement (~1000x est.):     10^12 bytes
after schema-delta (~100x estimated):           10^10 bytes
after content-addressable indexing (~100x):     10^8 bytes
after manifold abstraction (~10x estimated):    10^7 bytes
effective stored information (biological):     10^5 to 10^7 bytes
```

this arithmetic gives a total compression ratio of roughly 10^10 to 10^12. the biological synaptic substrate has capacity of approximately 10^14 bits (~10^13 bytes), so only a small fraction of raw capacity is used for long-term content; the remainder serves timing, attention control, motor programs, and other non-content functions. the gap between the ~10^13-byte total substrate and the ~10^5-10^7-byte estimated content store is itself uncertain and reflects the difficulty of defining "content memory" precisely.

no published architecture compounds these mechanisms. the published state of the art is:
- predictive filtering: titans (partial, via surprise gradient)
- generative replacement: dreamer v3 (in rl, not language)
- schema-delta: nothing directly
- content-addressable indexing: larimar, memorizing transformers (partial)
- manifold abstraction: vq-vae, encodec (for inputs, not memory)
- reconsolidation: nothing directly

stacking any four of these mechanisms would represent a novel engineering contribution. stacking all six would be the closest engineering match to biological memory that has been attempted. the research opportunity is clear.

## why bit-quantization cannot close the gap

the shannon rate-distortion lower bound for a fixed source is a hard theoretical floor. no algorithm can compress below this bound while preserving the same information. turboquant is within constant factor 2.7 of the floor; therefore the maximum additional compression achievable by any quantization method over turboquant is under 3x. no bit-quantization approach will achieve 10^6 compression.

the way to beat the bound is to change the source. bit-quantization operates on a fixed random variable (the stored vector); the compression is lower-bounded by that variable's entropy. generative replacement operates on a conditional distribution: the stored content given the decoder and query. this conditional distribution has much lower entropy because most of the information is in the shared decoder, not the per-memory variable.

formally: if you store N memories with k bits each, quantization's floor is N * k bits. generative replacement stores the decoder (D bits, one-time) plus N * log(codebook_size) bits per memory. for large N, the decoder cost amortizes to zero and the per-memory cost is the log of the codebook size, which can be much smaller than k.

this is not speculation; it is the same argument that makes image compression with a learned prior (e.g., ballé et al. 2018) beat jpeg. the prior captures the structure of the source. for memory, the prior is the entire model.

## mapping to todorov

### current state

todorov's matrix memory state is a direct accumulator: `S_t = alpha * S_{t-1} + beta * k_t * v_t^T`. it stores outer products verbatim, subject to exponential decay. the compression ratio relative to raw kv storage is roughly constant: an outer-product memory of rank min(d_k, d_v) holds approximately `0.14 * min(d_k, d_v)` distinct associations at the hopfield interference-free threshold, rising to approximately `0.38 * min(d_k, d_v)` at the classical overcapacity boundary (hopfield 1982; amit, gutfreund, sompolinsky 1985; keeler 1988 for outer-product models). at `d_k = d_v = 64`, this gives roughly 9 patterns before interference begins and roughly 24 patterns at the overcapacity boundary where structured interpolation emerges (see kalaj et al., imagination_computation_research.md). larger sequences simply overwrite older content.

the current delta state is thus both:
- small (2 mb per layer at batch 16, 48 mb total)
- bottlenecked (approximately 9 patterns at interference-free capacity, approximately 24 at overcapacity)

it satisfies neither of the brain's six mechanisms at the level beyond "minor biological inspiration." the delta rule erasure (ye et al. gated delta networks 2024) is a weak form of content-addressable indexing, but only one of the six.

### what it must become

this synthesis does not own the full architecture sketch. the concrete five-tier proposal lives in `bridge/memory_compression_to_tiered_architecture.md`. at the synthesis level, the commitments are narrower and more important:

- recent memory may stay verbatim, but older memory should not remain a cache forever
- predictive filtering must decide what is worth storing, instead of writing every event equally
- retrieval should increasingly trade storage for reconstruction as memories age: latent codes, schema deltas, and finally generative priors
- continual learning must stay fixed-size; memory compression that depends on unbounded growth does not solve the core problem
- provenance and targeted erasure matter because a compressed continual learner must be auditable, not just efficient

the bridge article translates those commitments into one explicit five-tier candidate architecture. this article's role is only to say that some architecture of that family is required if the six-mechanism thesis is taken seriously.

## what is genuinely novel if we build this

nothing in this synthesis is theoretically new. each mechanism has biological precedent and engineering precedent in isolation. the novelty is in:

1. compounding the six mechanisms in a single architecture for autoregressive language memory. no published paper does more than two.

2. applying generative replacement to transformer hidden states at the scale of 300m+ parameter language models. all existing generative replacement work is either small-scale (hypernetworks) or different modality (dreamer v3 for rl, nerf for 3d scenes).

3. combining schema-delta encoding with content-addressable indexing in a single memory tier. neither has been done in the language setting.

4. provenance-logged continual learning with delta-rule erasure as the safety mechanism. no published work uses the delta rule's targeted-erase property as an explicit safety primitive.

5. the fixed-size continuous learning constraint as a hard architectural invariant. most continual learning work grows adapters or memory stores over time.

6. benchmarking against kolmogorov-style compression rather than per-token perplexity. this is a different evaluation target and requires a different methodology.

each of these is a potential research paper. stacked together, they are a new memory architecture that is mathematically grounded, biologically plausible, engineering-constrained, and testable at scale.

## the correction-field design (2026-04-12)

the correction-field memory design (`correction_field_memory.md`) is the project's first concrete implementation of the compound compression thesis. it stacks mechanisms 1 and 2 (predictive filtering and generative replacement) plus prediction-residual value storage in a single modification to the matrix memory's write path: value projections operate on prediction residuals instead of raw hidden states, and writes are modulated by a surprise ratio. retrieval reconstructs by adding the stored correction to the model's own prediction.

the design addresses the first of the three gaps listed below (how to train generative-replacement memory) by sidestepping it: the model's own forward computation is the generative decoder, and the prediction head is a lightweight linear map trained alongside the main model via standard backpropagation. the second gap (how to compute surprise) is addressed by a simple norm ratio of the prediction residual to the hidden state. the third gap (how to evaluate compression) remains open.

the design is no longer a live memory-side compression candidate for the current matrix memory. both the synthetic and trained-prediction cpu simulations returned `memory_capacity_gain = 0`, so residual values did not increase the state's retrievable pattern count. what survives is the reconstruction-side interpretation: prediction plus residual correction can still improve final reconstruction even when the memory substrate itself does not compress better. see `correction_field_memory.md` for the detailed evidence trail.

## what is not addressed

this synthesis does not address how to train a generative-replacement memory. dreamer v3 trains its world model via next-state prediction on environment observations; for autoregressive language, the analog is next-token prediction on hidden states, which is circular (the hidden states are already the output of next-token prediction). the training objective for a language-state decoder is an open research problem. *update 2026-04-12: the correction-field design partially addresses this by using the model's own forward computation as the decoder, with a lightweight prediction head trained jointly.*

this synthesis does not address how to compute the surprise signal in a compound memory architecture. titans uses gradient magnitude of the model's loss; this is expensive (requires a backward pass per token) and may not reflect "surprise" at the appropriate semantic level for long-term memory retention. *update 2026-04-12: the correction-field design uses a cheaper signal (norm ratio of prediction residual to hidden state) that does not require a backward pass.*

this synthesis does not address how to evaluate memory compression. current benchmarks (longbench, ruler, needle-in-haystack) measure retrieval accuracy on verbatim content, which penalizes lossy-reconstruction mechanisms even when they preserve the semantic content. new benchmarks that measure reconstruction quality at the semantic level are needed.

## see also

- [[correction_field_memory]] -- reconstruction-side correction-field evidence; residual storage did not increase matrix-memory capacity in synthetic or trained-prediction simulations
- [[generative_memory_research]] -- curated research library and prior-art evidence
- [[compression_and_bottlenecks]] -- biological compression overview
- [[compression_architecture]] -- earlier proposal shelf, superseded by the bridge article
- [[memory_systems_research]] -- episodic versus semantic memory background
- [[imagination_computation_research]] -- generative recombination background
- [[memory_consolidation]] -- biological consolidation background
- [[memory_compression_to_tiered_architecture]] -- current five-tier architecture translation
- [[indexed_reconstruction_compression]] -- current compact-handle plus reconstruction direction
- [[neural_model_compression_stack]] -- current operation-preserving compression-stack direction for every neural-model memory surface
