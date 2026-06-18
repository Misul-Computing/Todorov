# bridge: memory compression to tiered architecture

status: current (as of 2026-06-18).

this page is the architecture-translation layer for the compression cluster. it assumes the six-mechanism thesis argued in `synthesis/compression_beyond_quantization.md` and turns that thesis into one explicit five-tier candidate design. the literature shelf remains `knowledge/generative_memory_research.md`.

## the biological mechanism

the mammalian brain retains approximately 10^5 to 10^7 bytes of effective usable content memory across a lifetime (cognitive-science estimate, not a direct measurement; figures above 10^7 require assumptions about near-zero marginal cost for schema-consistent storage that are not directly measured), reconstructed from a raw sensory stream of approximately 10^17 bytes (back-of-envelope estimate over an 80-year lifetime). the implied effective compression ratio for content is roughly 10^10 to 10^12. bartol et al. 2015 measured approximately 4.7 bits per synapse via serial em reconstruction of hippocampal ca1 synapses specifically; extrapolating across the human brain's ~10^14 synapses gives a total substrate capacity of approximately 10^14 bits (~10^13 bytes), most of which serves non-content functions (timing, attention control, motor programs). this gap is not achieved by any single mechanism; it is the product of at least six compounding operations:

1. **predictive filtering**: only unexpected information reaches long-term memory. cortex maintains a hierarchical generative model; only prediction errors propagate upward (rao and ballard 1999; friston 2010).
2. **generative replacement**: memory is a pointer into a learned generative model. reconstruction happens at retrieval time.
3. **schema-delta encoding**: common patterns are stored once as schemas; specific episodes are small deviations (bartlett 1932; mcclelland et al. 1995).
4. **content-addressable indexing**: hippocampus stores pointers into cortical representations, not the representations themselves (teyler and discenna 1986).
5. **algorithmic / manifold abstraction**: memories are coordinates on learned manifolds, not raw feature patterns.
6. **reconsolidation**: retrieval is a rewrite operation; lossy reconstruction replaces the old memory (nader et al. 2000; sevenster et al. 2014).

none of todorov's existing memory mechanisms implement these at the level beyond "minor biological inspiration." the matrix memory state is a direct accumulator with exponential decay. the compressed-attention cache is a low-rank projection of kv. the residual stream passes activations unchanged between layers. there is no predictive filter, no generative decoder, no schema, no index, no manifold compression, and no reconsolidation.

the proposal in this document is an engineering architecture that implements the six biological mechanisms as a five-tier memory system with a provenance audit log, satisfying two hard constraints: total memory size must be bounded, and continuous learning must not grow model weights.

## the current todorov implementation

### matrix memory

```
S_t = diag(alpha) * S_{t-1} + beta_t * k_t * v_t^T
```

the matrix memory state is a rank-limited outer-product accumulator. it stores associations (k, v) and retrieves via `o_t = q_t^T * S_t`. the compression ratio relative to storing raw kv pairs is bounded by the interference capacity of the outer-product memory, approximately `0.14 * min(d_k, d_v)` distinct patterns before interference-free retrieval begins to fail, rising to approximately `0.38 * min(d_k, d_v)` at the overcapacity boundary (classical hopfield analysis: hopfield 1982; amit, gutfreund, sompolinsky 1985). at d_k = d_v = 64, this is approximately 9 patterns at the interference-free threshold and approximately 24 patterns at the overcapacity boundary. older patterns decay at rate `1 - alpha` per step.

this is a single-tier memory with fixed decay. there is no differentiation between "recent and verbatim" versus "old and compressed." the only compression is interference-based overwriting and a learned spike-mask projection from raw residual stream.

### compressed attention

```
c_kv = W_down @ x
k, v = W_up_k @ c_kv, W_up_v @ c_kv
```

compressed attention stores a compressed latent representation per token. the latent has dimensionality `d_c << d_model`, reducing kv storage by a factor of approximately `d_model / (d_c + d_R)` where d_R is the rope split. for d_model=1024, d_c=128, d_R=32, this is 6.4x compression. the cache still stores one entry per token, so it grows linearly with context length.

this is architectural low-rank compression, not structural replacement. it is roughly in the same family as more recent structural kv compression methods such as vqkv (arxiv 2603.16435, author-reported ~5.7x on llama-3.1-8b), commvq (apple ml, arxiv 2506.18879, author-reported ~8x at 2-bit), vqllm (kumar et al. 2024, arxiv 2410.15704, author-reported ~5.5x), and deltakv (arxiv 2602.08005, author-reported ~3.45x). all of these methods store the compressed kv verbatim as indices or residuals; none of them replace storage with computation. the published structural-compression frontier for language model memory is in the ~5-8x range from these methods, which is why a different abstraction is needed.

### residual stream

activations flow from layer to layer via addition (x = x + f(x)). the residual stream is not a memory store; it is a communication channel. no compression happens in the residual stream. it is mentioned here only to clarify what is outside the memory path.

### ternary quantized activations

k and v are quantized to {-1, 0, 1} before being multiplied into the state. this reduces storage bits per value from 16 (bf16) to approximately 1.5, a factor of ~10x. but the reduction is in representation bits, not in number of stored entries. all tokens are still stored.

### rate-coded sparsity

k and v are top-k filtered at 20 percent before state update. this reduces effective information density by 5x but does not reduce storage. stored entries still grow linearly with context length.

### summary

the current todorov architecture has approximately 10x-20x compression relative to a vanilla transformer kv cache via the combination of compressed-attention projection, low-bit or rate-coded sparsity, and matrix-memory interference limits. this is modest and operates entirely at the bit-packing or architectural-projection level. it does not implement any of the six brain-inspired compression mechanisms structurally.

## the proposed change

### five-tier memory architecture

replace the single delta state with a graded cascade of memory tiers, each with a different compression mechanism and a different retention policy. new memories enter at tier 0 and migrate downward as they age or as tier 0 fills. deletion is explicit via delta-rule erasure with provenance tracking.

```
tier 0  working memory                       verbatim              100% quality
tier 1  predictive residual                  surprise only           99% quality  
tier 2  latent codebook                      manifold coordinates    80% quality
tier 3  schema delta                         template + residual     50% quality
tier 4  generative only                      world model priors      20% quality
      (forgotten but reconstructible)
```

each tier has a specific algorithm, a specific size budget, and a specific retrieval operation.

#### tier 0 — working memory (verbatim)

- **substrate**: current matrix memory state, unmodified. an outer-product matrix of shape `(B, H, D_k, D_v)` per layer.
- **size budget**: fixed. at d_k = d_v = 64 and 24 delta layers, the state is approximately 48 mb per batch item.
- **retention policy**: exponential decay per step as currently implemented. no explicit eviction.
- **update rule**: `state = alpha * state * (I - beta * k * k^T) + beta * k * v^T` (current gated delta rule).
- **biological analog**: ca3 recurrent autoassociative memory, rank-limited, fast-write.
- **purpose**: immediately-useful memory for the current context window.
- **what's new**: nothing. this is the current matrix memory state.

#### tier 1 — predictive residual (surprise only)

- **substrate**: a fixed-size buffer of (token_position, surprise_vector) pairs. the surprise vector is `x_t - predict(x_{t-1}, S_{t-1})` where `predict` is a learned lightweight decoder from previous state to next hidden state.
- **size budget**: at most 1 percent of context_length entries per batch item. for seq_len=2048, this is ~20 entries.
- **retention policy**: entries are added when `||surprise_vector|| > threshold`, where the threshold is adaptive based on the moving average of recent surprise magnitudes. the buffer is fifo once full.
- **update rule**: append, not overwrite. when buffer is full, oldest entries migrate to tier 2.
- **biological analog**: free-energy-minimization cortex (friston); hippocampal novelty detection in ca1.
- **purpose**: preserve the information content of recent context in a fraction of the storage, selecting only what the prediction model failed to anticipate.
- **what's new**: 
  - a learned prediction head that takes (previous state, previous hidden) and predicts the next hidden state
  - an adaptive surprise threshold
  - a fifo buffer with tier-migration policy

#### tier 2 — latent codebook (manifold coordinates)

- **substrate**: a learned vq-vae-style codebook of `codebook_size` entries, each of dimension `d_latent`. stored memories are sequences of codebook indices.
- **size budget**: codebook is fixed (e.g., 8192 entries at d_latent=64 = 2 mb of codebook). per-memory storage is `log(codebook_size)` bits = 13 bits per latent token.
- **retention policy**: entries from tier 1 are encoded via a learned encoder into codebook indices and stored in this tier. the codebook itself is updated via standard vq-vae training (either offline or continually with ema updates as in neural discrete representations literature).
- **update rule**: encode new memory via nearest-codebook-entry; store the index.
- **retrieval rule**: decode via a learned decoder from codebook index back to hidden state.
- **biological analog**: cortical semantic representations; learned feature manifolds (face space, object space, etc.).
- **purpose**: compress middle-age memories to tiny per-entry storage (~13 bits per token) while preserving retrievable content quality via a shared learned decoder.
- **what's new**:
  - vq-vae applied to transformer hidden states (not images or audio) at scale
  - an encoder/decoder pair trained alongside the main model
  - migration from tier 1 on fifo overflow

#### tier 3 — schema delta (template + residual)

- **substrate**: a set of learned schemas (templates) per domain. each schema is a learned slow-vector of hidden-state dimensions. each stored memory is a (schema_id, small delta) pair.
- **size budget**: schema_count * d_hidden + per_memory_delta_size * memory_count. at schema_count=256 and d_hidden=1024, schemas are 1 mb total; each delta is a small residual at ~32 bytes per memory.
- **retention policy**: entries from tier 2 migrate to tier 3 when the schema match is strong (small delta) and the entry is old.
- **update rule**: find nearest schema (by dot product or learned similarity); compute delta as residual; store (schema_id, delta).
- **retrieval rule**: reconstruct as `schema + delta`.
- **biological analog**: bartlett-schema theory; cortical gist memory; semantic vs episodic distinction.
- **purpose**: compress very old memories where the "common case" is captured by a schema and only the unique deviation needs per-memory storage.
- **what's new**:
  - schemas as learned structures in a transformer memory system
  - a learned schema-similarity function
  - a policy for when to use schema delta vs latent codebook (tier 3 vs tier 2)

#### tier 4 — generative only (world model priors)

- **substrate**: nothing stored. the main language model's weights serve as the generative prior.
- **size budget**: zero per memory (the model itself already exists).
- **retention policy**: when an entry in tier 3 is older than max_age or its importance score drops below threshold, it is removed from tier 3 and not stored elsewhere. the content is implicitly reconstructible by the model's generative prior if queried with a sufficient cue.
- **retrieval rule**: at retrieval, if no matching entry is found in tiers 0-3, the model generates a response using its own priors. this is hallucination from the model's world knowledge.
- **biological analog**: remote memory reconstruction in cortex without hippocampal involvement (e.g., recalling childhood memories); confabulation.
- **purpose**: represent the infinite tail of forgotten memories whose content can still be approximately reconstructed by model priors.
- **what's new**:
  - explicit recognition that some memories should be "forgotten but reconstructible"
  - a retrieval policy that falls back to generative priors when stored tiers miss
  - no new stored state; this is a policy decision, not an architectural component

### provenance audit log

every memory write across all tiers emits an event to an append-only provenance log:

```
{
  timestamp: int,
  source_id: str or hash,
  tier: 0..4,
  memory_id: uuid,
  operation: write|migrate|erase,
  content_summary: hash,
}
```

at ~100 bytes per event and 10^7 writes per year of continuous training, the log grows by ~1 gb per year. it is bounded by disk, not by model memory. compression of old log regions is standard text compression.

the log enables:
- **audit trails**: for any memory, trace back when and from what source it was created.
- **poison scrubbing**: if a source is later determined to be malicious, iterate the log for all memories tagged with that source and apply delta-rule erasure across all tiers.
- **debugging**: when a memory produces a wrong retrieval, inspect when it was written and what source it came from.
- **continual learning safety**: bounded risk during continual learning because bad updates can be reversed via the log.

## implementation spec

### data flow

```
input token -> hidden state x_t
  |
  +-> tier 0 update: S_t = delta_rule(S_{t-1}, k(x_t), v(x_t))
  |
  +-> predict x_{t+1} from S_t and x_{1..t-1}
  |
  +-> compute surprise_t = x_{t+1} - predict
  |
  +-> if |surprise_t| > threshold:
        append (t, surprise_t, source_id) to tier 1 buffer
  |
  +-> on tier 1 overflow:
        for each evicted entry e:
          encode e into codebook index via vq-vae encoder
          append (codebook_idx, metadata) to tier 2
          log tier migration
  |
  +-> periodically (every K steps):
        for entries in tier 2 older than age_threshold:
          match against schema bank (find nearest schema)
          compute delta = entry - schema
          store (schema_id, delta) in tier 3
          log tier migration
  |
  +-> periodically (every M steps):
        for entries in tier 3 older than max_age and low-importance:
          apply delta-rule erasure for that entry's keys
          log tier-4 "forgotten" transition
```

### retrieval policy

```
given query q:
  try tier 0 retrieval: o_0 = q^T * S
  if confidence_0 > threshold: return o_0
  
  try tier 1: look up surprise_vector closest to q
  if confidence_1 > threshold: return predict_model(q) + surprise_vector
  
  try tier 2: quantize q into codebook space, find nearest codebook index
  if confidence_2 > threshold: return vq_vae_decoder(index)
  
  try tier 3: find matching schema, add stored delta
  if confidence_3 > threshold: return schema + delta
  
  fallback (tier 4): generate from world model priors
  return main_model.generate(q)
```

### safe continual learning

```
during training on new data:
  all memory writes go through provenance log with source_id = "training_source_x"
  tiers 0, 1 are updated freely
  tier 2 codebook updates with EMA (slow, bounded effect)
  tier 3 schemas update rarely and only when aligned with existing structure
  tier 4 world model updates via standard sgd on the fixed-size weights

during continual learning from user interaction:
  all writes tagged with source_id = "user_session_y"
  importance cap: sources tagged "user_session_*" cannot modify tier 3 schemas
  this limits the blast radius of any single user interaction
  
if a source is later flagged as malicious:
  scan provenance log for all entries with source_id = bad_source
  for each entry: apply delta-rule erasure for that memory's key in tier 0/1, 
                  mark the tier 2/3 entries as erased,
                  do not modify tier 4 world model
  log the scrub operation
  verify via replay: do the model's behaviors still reflect the bad source?
  if yes: escalate (may need retraining or further erasure)
```

### no-weight-growth verification

the architecture satisfies the no-weight-growth constraint at every tier:

- tier 0: fixed-size matrix, updates in place. already satisfied by matrix memory.
- tier 1: fixed-size buffer, fifo overwrite. satisfied by design.
- tier 2: fixed codebook count. new codebook entries are never added; existing entries are updated via ema. satisfied by design.
- tier 3: fixed schema count. new schemas are never added; existing schemas update slowly via online learning. satisfied by design.
- tier 4: world model weights are fixed-size and updated in place via delta rule or sgd. satisfied by design.

the provenance log is the only growing component, and it grows outside the model (on disk, auditable, compressible). this is not "model weight growth."

## parameter budget

for a 283m god_machine variant with this tiered memory:

```
main model parameters:                      ~283m (unchanged)
prediction head (tier 1 compute):            ~5m
vq-vae encoder:                              ~3m
vq-vae decoder:                              ~3m
vq-vae codebook (8192 x 64):                 ~2m
schema bank (256 x 1024):                    ~0.26m
schema similarity network:                   ~1m
provenance log (on disk, not in model):       0
total new parameters:                       ~14.3m
total model size:                          ~297m
percentage increase from baseline:           ~5%
```

the added parameters are modest relative to the main model. most of them are small encoders/decoders that amortize across all stored memories.

## expected impact

### positive

- **compression ratio**: each tier is ~10x-100x smaller than the previous (verbatim -> surprise -> codebook -> schema -> generative). compounded, the tail of old memory is effectively zero-cost, while the head of recent memory is verbatim. total effective capacity is 100x-1000x beyond the current matrix memory state for the same total memory footprint.
- **safety**: provenance logging + delta-rule erasure gives a concrete mechanism for "unlearning" specific sources without retraining. this is stricter than any published machine unlearning technique because delta-rule erasure is algebraically exact (up to key-overlap caveats, see below).
- **continuous learning viability**: the fixed-size constraint means the model can absorb new information without growing. combined with tier-protected schemas (only schemas get slow updates), the risk of catastrophic drift is bounded.
- **scientific contribution**: if implemented and validated, this is the first published memory architecture that compounds all six brain-inspired compression mechanisms in a single system.

### negative

- **training complexity**: each tier has its own loss, its own update rule, and its own convergence dynamics. training stability will be harder than a single-tier delta memory.
- **hyperparameter surface**: surprise thresholds, schema counts, codebook sizes, migration intervals, age thresholds, importance scores. many knobs to tune.
- **retrieval latency**: a query may need to traverse all 5 tiers to find its answer. the existing baseline matrix-memory retrieval is a single einsum. the tiered retrieval is multiple lookups. for generation at inference time this may dominate.
- **engineering complexity**: building, testing, and debugging a 5-tier system with provenance logging and adversarial-robust erasure is substantial work. months, not days.

### risks

- the **key overlap problem**: delta-rule erasure is algebraically exact for erasing the projection of state onto key k, but it does not erase information stored in directions orthogonal to k. if the same information is stored across multiple overlapping keys, erasing one key leaves residuals. this is a theoretical gap that has not been studied.
- the **surprise-gate ambiguity**: the "is this token surprising enough to store" decision is a learned threshold. bad calibration (too high or too low) defeats the purpose of the tier.
- the **schema drift**: if schemas update during continual learning, old memories (stored as schema + delta) may become inconsistent. this needs a re-encoding procedure or an "anchor schemas do not drift" policy.
- the **tier-migration thrashing**: memories may move between tiers frequently if the importance score oscillates. rate-limiting the migration is needed.
- the **world model hallucination**: tier 4 relies on the world model's priors for retrieval when no stored entry matches. if the world model has gaps, retrievals in tier 4 return plausible-but-wrong answers. this is the same failure mode as llm hallucination and has no clean fix.

## implementation sequencing

this is a phase 6+ research direction, not a run 2 modification to god_machine. the dependencies are:

1. **prerequisite**: god_machine run 1 completes successfully, validating the 5-feature single-tier design at 283m scale.
2. **prerequisite**: a prototype of a learned prediction head for hidden states, trained as an auxiliary objective. validate the prediction head's mse on held-out transformer states.
3. **prerequisite**: a prototype of vq-vae for hidden states, trained on a corpus of delta-state snapshots. validate reconstruction quality vs verbatim storage on retrieval tasks.
4. **milestone**: tier 0 + tier 1 combined (delta state + surprise buffer). validate no regression vs baseline.
5. **milestone**: add tier 2 (codebook). validate that tier 2 entries can be retrieved via the decoder with minimal quality loss.
6. **milestone**: add tier 3 (schemas). validate schema learning online without catastrophic drift.
7. **milestone**: add tier 4 and provenance log. validate safe continual learning on adversarial input.
8. **final**: full integrated 5-tier architecture, benchmarked end-to-end on language modeling, retrieval tasks (passkey, selective copy, perplexity-at-length), and a new adversarial-learning benchmark.

each milestone is a separate research paper's worth of work. the full architecture is a multi-year research program, not a single run.

## open research questions

1. **training objective for the tier 1 prediction head**: should it be mse on the next hidden state? cross-entropy via an auxiliary next-token prediction? how does the objective interact with the main next-token prediction loss?
2. **vq-vae training on hidden states**: is the hidden-state distribution well-suited to discrete codebook representation? what codebook size is optimal? how does training dynamics differ from image vq-vae?
3. **schema learning**: how are schemas discovered? clustering? online ema? explicitly supervised with domain labels?
4. **importance score computation**: what is the right measure of "which memories should we keep"? retrieval frequency? model surprise on retrieval? gradient-based attribution?
5. **benchmark design**: how do we evaluate a tiered memory system? standard perplexity does not distinguish between tiers. needle-in-haystack measures verbatim retrieval and penalizes lossy reconstruction. we need new benchmarks that measure "semantic faithfulness" at each tier.
6. **the safety proof**: under what assumptions is delta-rule erasure + provenance logging a reliable unlearning primitive? the key overlap problem needs formal analysis.

## risk assessment

| component | implementation difficulty | risk to existing performance | expected benefit | biological fidelity |
|---|---|---|---|---|
| tier 0 (current matrix memory) | already done | none | baseline | low |
| tier 1 (surprise buffer) | moderate | low | 10-100x compression | high |
| tier 2 (vq-vae codebook) | hard | moderate | 100-1000x compression | high |
| tier 3 (schema delta) | hard | moderate | 100x compression | high |
| tier 4 (generative only) | easy once tiers 0-3 work | low | infinite compression (zero storage) | high |
| provenance log | moderate | low | safety infrastructure | moderate (audit log is engineering, not biological) |
| write-side erasure for unlearning | easy (already in matrix memory) | low | adversarial robustness | moderate |

the estimated total implementation effort is approximately 3-6 months of focused engineering plus research, assuming the prerequisites (prediction head prototype, vq-vae prototype) succeed at small scale first.

## recommendation

do not treat this bridge as the next immediate implementation item. the active lane is the neural-machine architecture program (approach a). this page should be read as a phase-6+ architecture sketch rather than a launch recipe, to be picked up once the program reaches its compression tier.

1. first, resume from the implemented phase-1 battery and the remaining model-side neural-model evaluation gaps.
2. second, validate smaller compression components cpu-first. the correction-field result already showed that residual values do not raise matrix-memory capacity on their own, so a full tiered build should not be the first thing retried.
3. third, only if simpler components show signal, propose a dedicated tiered-memory variant with isolated ablations rather than folding it into an existing substrate run.
4. fourth, keep provenance logging and targeted erasure as first-class constraints from the start, not as bolt-ons.

## see also

- [[generative_memory_research]] -- curated research library and evidence layer
- [[compression_beyond_quantization]] -- current compression thesis
- [[compression_and_bottlenecks]] -- biological precedent
- [[compression_architecture]] -- earlier proposal shelf, superseded
- [[memory_systems_research]] -- episodic versus semantic memory
- [[matrix_memory_vs_hippocampus]] -- existing analysis of the matrix-memory gap versus hippocampus
- [[plasticity_to_matrix_memory_delta_rule]] -- matrix-memory write math
- [[predictive_coding_to_training_objective]] -- predictive-coding options for the project
- [[indexed_reconstruction_compression]] -- current compact-handle plus reconstruction synthesis

## key references

- rao, r. p. n., ballard, d. h. (1999). predictive coding in the visual cortex. nature neuroscience 2, 79-87.
- friston, k. (2010). the free-energy principle: a unified brain theory? nature reviews neuroscience 11, 127-138.
- mcclelland, j. l., mcnaughton, b. l., o'reilly, r. c. (1995). why there are complementary learning systems in the hippocampus and neocortex. psychological review 102, 419-457.
- teyler, t. j., discenna, p. (1986). the hippocampal memory indexing theory. behavioral neuroscience 100, 147-154.
- van den oord, a., vinyals, o., kavukcuoglu, k. (2017). neural discrete representation learning (vq-vae). arxiv 1711.00937.
- hafner, d. et al. (2023). mastering diverse domains through world models (dreamer v3). arxiv 2301.04104.
- das, p. et al. (2024). larimar: large language models with episodic memory control. icml 2024. arxiv 2403.11901.
- behrouz, a., zhong, p., mirrokni, v. (2025). titans: learning to memorize at test time. arxiv 2501.00663.
- nader, k., schafe, g. e., ledoux, j. e. (2000). fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. nature 406, 722-726.
- mildenhall, b., srinivasan, p. et al. (2020). nerf: representing scenes as neural radiance fields. eccv.
