# compression and bottlenecks

status: current (as of 2026-04-25).

## the biological principle

biological memory systems treat capacity limits as computational features, not engineering defects. this is the central insight of the compression story: bottlenecks are not obstacles to be overcome but architectural choices that enable specific computational properties.

the dentate gyrus enforces 2-5% population sparsity, reducing a rich entorhinal cortex input to a near-orthogonal code before it ever reaches the associative network in CA3 (see [[hippocampal_memory]]). this is not a bandwidth limitation -- it is pattern separation: the deliberate destruction of input similarity to prevent interference between stored memories. the combinatorial space is enormous (choosing 50,000 active cells from 1,000,000 gives a representation capacity that far exceeds the number of memories stored in a lifetime), yet only a tiny fraction of that space is used at any moment. the sparsity is enforced by specific biophysical mechanisms: granule cells have unusually negative resting potentials (~-75 mv vs ~-65 mv for pyramidal cells), strong basket cell inhibition implements winner-take-all competition, and mossy fiber "detonator synapses" ensure that only the winning pattern propagates to CA3.

the hippocampus itself is a compression bottleneck. the hippocampal indexing theory (teyler and discenna 1986) holds that the hippocampus stores compressed pointers to distributed cortical patterns, not the patterns themselves. the full memory lives in neocortex; the hippocampus holds a lossy index. this architecture separates the problem of storage (neocortex, high capacity, slow learning) from the problem of binding (hippocampus, low capacity, one-shot encoding). the capacity limit is what forces this separation -- without it, a single system could do both, and [[complementary_learning_systems]] would be unnecessary.

consolidation adds a temporal dimension to compression. sharp wave ripple replay during sleep replays recent hippocampal traces at 5-20x compression in time, gradually training neocortical circuits to represent the statistical gist of many episodes (see [[memory_consolidation]]). the cortical trace after consolidation is not a copy of the hippocampal trace -- it is an abstraction. episodic detail is discarded, shared structure is preserved. the multiple trace theory (nadel and moscovitch 1997) qualifies this: episodic memories may retain hippocampal involvement indefinitely, with consolidation transferring only the semantic gist. either way, the direction of information flow is clear: from high-fidelity, capacity-limited store to low-fidelity, high-capacity store.

the process is orchestrated by nested oscillations: cortical slow oscillations (0.5-1 hz) gate thalamic sleep spindles (10-16 hz), which in turn gate hippocampal sharp wave ripples (100-250 hz). the three-stage temporal coupling (klinzing, niethard and born 2019) is the mechanism for directed hippocampal-to-cortical information transfer. consolidation is not passive decay -- it is an active, structured compression pipeline.

reconsolidation adds a further layer of complexity. nader, schafe and ledoux (2000) showed that reactivating a consolidated memory returns it to a labile state, requiring another round of protein synthesis to re-stabilize. every act of retrieval is potentially an act of rewriting -- the compressed representation is not read-only, but read-write. this means the biological compression pipeline is bidirectional: information flows from hippocampus to cortex during consolidation, but also back from cortex to hippocampus during reconsolidation, where the memory can be updated, modified, or distorted. no artificial architecture implements this bidirectional compression.

three compression stages, then: sparse encoding (dg), compressed indexing (hippocampal trace), and lossy abstraction (consolidation to cortex). each stage destroys information. each destruction is useful.

## compression in todorov

todorov implements three compression mechanisms, each operating at a different level of the architecture.

**ternary spikes** quantize continuous activations to {-1, 0, +1}, reducing information from 32 bits per dimension (fp32) to 1.58 bits per dimension (log2(3)). at the validated firing rate of ~41%, roughly 59% of dimensions are zeroed entirely. this is the most aggressive information bottleneck in the architecture.

spike mutual information of 1.168 at 267m scale demonstrates that the bottleneck preserves task-relevant information despite a ~20x reduction in representational precision (see [[sparse_coding_to_ternary_spikes]]). cka of 0.732 at the same scale shows that the geometric structure of representations is largely preserved through the quantization.

these are strong empirical results: the bottleneck destroys most of the information content but retains what matters for next-token prediction. the natural question is why. the answer lies in the adaptive threshold: alpha * mean(|x|) adjusts the quantization boundary to the statistics of each input, ensuring that the ternary code captures the sign and relative magnitude of each dimension. this is not random compression -- it is signal-adaptive compression that discards magnitude information while preserving directional information. whether this is closer to rate coding (where magnitude matters) or spike coding (where timing and sign matter) is itself an interesting question.

but 41% firing is a fundamentally different regime from the brain's 2-5%. at 2-5% activity, two random binary codes overlap by roughly 0.04-0.25% of active units in expectation -- near-orthogonality is guaranteed by the statistics of sparse random vectors. at 41%, two random ternary codes share roughly 17% of active units. the dg achieves interference prevention through extreme sparsity; todorov's spikes achieve information compression through quantization. these are different computational functions. the biological analog for todorov's spikes is closer to action potential all-or-nothing transmission (a communication code that enables energy-efficient signal propagation) than to dg pattern separation (a storage code that prevents memory interference).

**mla's kv compression** projects d_model (1024) to d_c (128) before computing keys and values, an 8x compression of the per-token representation. this bottleneck forces the model to learn which information is worth preserving for future retrieval. every token passes through this 128-dimensional chokepoint, and what emerges on the other side determines what the attention mechanism can access.

this is the closest analog to hippocampal indexing: a compressed representation that can reinstate the full pattern (via the up-projection back to d_model) from a low-dimensional code. the per-token cache cost is d_c + d_R = 128 + 32 = 160 floats, compared to 1024 for a full kv cache -- a 6.4x memory savings. at context length 2048, this saves ~1.4 million floats per layer, making the difference between fitting in gpu memory and not fitting.

the compression is learned end-to-end: the down-projection and up-projection are trained jointly with the rest of the model, so the 128-dimensional latent learns to preserve exactly the information that downstream attention needs. this is a bottleneck in the variational autoencoder sense -- a low-dimensional representation that the model is forced to route all information through, creating an implicit prior on what information is worth attending to.

the differences from hippocampal indexing are substantial. hippocampal indices are formed rapidly and decay over days to months; mla's compressions are computed once and cached indefinitely within the context window. hippocampal indices are sparse and binary-like; mla's latents are dense and continuous. hippocampal indices point to distributed cortical patterns stored elsewhere; mla's latents ARE the compressed representation, with no separate store to point to. the index IS the memory.

**kda's rank-limited state** maintains a head_dim x head_dim (64 x 64) matrix per head, with 16 heads. the maximum rank of this state is 64 per head, meaning that at most 64 linearly independent associations can coexist. combined with exponential decay (alpha ~ 0.12, meaning ~88% loss per step), the effective capacity is further limited: after ~8 steps, early writes have decayed below 12% of their original magnitude.

this makes kda a severe bottleneck -- a working memory buffer that can hold only a handful of associations at useful strength. the total capacity across all heads is 16 * 64 = 1024 rank dimensions, but the exponential decay means the effective memory horizon is much shorter than the rank limit implies. information does not persist long enough to fill the available capacity.

the bottleneck is doubly constrained: by rank (spatial) and by decay (temporal). this double constraint has no clean biological analog. the hippocampus is capacity-limited (by interference, ~0.138N patterns for N neurons) but not time-limited -- a hippocampal memory can persist for weeks if not interfered with. working memory in prefrontal cortex is both capacity-limited (~4-7 items) and time-limited (seconds without rehearsal), making it a closer functional analog to kda than the hippocampus. but kda's decay is exponential and passive, while prefrontal working memory is maintained by active recurrent firing that can be disrupted by distraction or load.

## bottlenecks as computational features

the standard engineering instinct is to maximize capacity. biology does the opposite, and the reasons are instructive.

**interference prevention.** the dg's 2-5% sparsity ensures that two similar experiences activate nearly non-overlapping populations of granule cells. this is why the hippocampus can store ~34,000 patterns in an auto-associative network of ~250,000 neurons without catastrophic overwriting (see [[pattern_completion]]). without the bottleneck, similar inputs would produce overlapping codes, and storing one would corrupt the other.

kda's rank limitation serves a structurally similar function: with only 64 dimensions of rank per head, new associations cannot occupy arbitrary subspaces -- they are forced into a compressed representation that limits how much any single write can corrupt existing state. but the mechanism is different. dg achieves near-orthogonality by sparsification. kda achieves capacity limitation by projection into a fixed-rank manifold. the former prevents interference by construction; the latter limits the total amount of information that can interfere.

**forced selection.** a bottleneck forces the system to decide what matters. mla's 8x compression means the model cannot cache everything about a token -- it must learn to discard irrelevant features during training. the gradient signal from the language modeling objective shapes the compression: features that predict future tokens survive; features that do not are discarded. this is selection pressure operating through an information bottleneck, analogous to how the dg's sparsity forces the hippocampus to select which features of an experience to encode.

the critical distinction: biological selection operates at inference time (each experience is separately pattern-separated by the dg). todorov's selection is learned at training time (the projection matrices are fixed after training). mla cannot adapt its compression to a surprisingly important token at inference -- it applies the same learned projection to every input.

kda's write gating via beta_t offers a partial exception: the sigmoid gate determines how strongly each token writes to the state matrix, providing a token-level importance signal at inference time. but this modulates write magnitude, not write content. the compression applied to the key-value pair is the same regardless of beta_t; only the amplitude changes. biological encoding selection can change what is encoded (different features for emotional vs neutral stimuli, mediated by amygdala-hippocampal interactions), not just how strongly.

**efficient indexing.** a compressed representation is faster to search. mla attention over 128-dimensional latents is cheaper than attention over 1024-dimensional full representations. hippocampal indices are small relative to the cortical patterns they reference (~0.1% of cortical volume). in both cases, the compression creates a lookup structure that trades fidelity for speed.

the difference in optimization pressure: biological bottleneck sizes are tuned by evolution and development over millions of years. todorov's bottleneck sizes (d_c = 128, head_dim = 64, spike threshold alpha = 1.0) were set by engineering heuristics. the dg's 2-5% firing rate emerges from specific biophysical properties that are themselves under genetic control. todorov's 41% emerges from a hyperparameter. whether gradient descent would discover extreme sparsity (2-5%) if given the freedom to learn the threshold parameter is an open question -- and an experiment worth running.

a quantitative comparison makes the asymmetry concrete. the dg compresses ~10,000 entorhinal inputs per granule cell down to a binary fire/no-fire decision at 2-5% population activity: an information reduction of roughly 100-200x per neuron per timestep. todorov's ternary spikes compress 32-bit floats to 1.58-bit ternary values at 41% activity: a ~20x reduction per dimension. mla compresses 1024 continuous dimensions to 128 continuous dimensions: an 8x reduction that is lossless in principle. kda compresses a stream of rank-1 associations into a rank-64 matrix with exponential decay: a compression ratio that depends on sequence length but is effectively infinite for sequences longer than ~20 tokens (all early information is lost). the biological system compresses aggressively at the front end (dg) and preserves selectively through the pipeline. todorov compresses moderately at every stage and preserves nothing across sequences.

## the consolidation gap

the deepest asymmetry between biological compression and todorov's compression is consolidation -- the active transfer of information between memory systems across timescales.

in biology, the three compression stages are linked by a directed information flow: dg -> ca3 -> ca1 -> neocortex, mediated by replay during sleep. the hippocampal trace is temporary. the cortical trace is permanent. the bridge between them (sharp wave ripple replay) is itself a compression operation, replaying 10-second experiences in 50-100ms. the system is designed to lose the hippocampal index once the cortical trace is strong enough to stand alone.

todorov has no analog to this process. kda's state decays via alpha, but what is lost is simply lost -- it is not transferred to mla or any other store. mla's cache grows monotonically within a context window but is discarded entirely between sequences. there is no mechanism for the model to decide "this kda state contains information worth preserving" and distill it into a more permanent form. the two memory systems share a residual stream, but each layer's memory (kda state or mla cache) is private to that layer.

the bridge article (see [[memory_systems_to_matrix_memory_and_compressed_attention]]) considered three hypothetical consolidation mechanisms: periodic state distillation, experience replay during training, and kda-mla prediction loss. all were assessed as high-risk with uncertain benefit.

the fundamental reason: consolidation in biology solves a problem that todorov does not have at current scale. biological organisms experience events over seconds to years and must maintain memories across sleep cycles. todorov processes a finite context window in a single forward pass. there is no "overnight" between tokens. the timescale that motivates CLS does not exist in autoregressive inference.

whether this matters depends entirely on context length. at 2048 tokens, kda's ~8-step effective horizon covers a small fraction of the window, and mla covers the rest with quadratic cost. at 100k+ tokens, the gap between kda's decay horizon and mla's O(T^2) cost creates a regime where neither system serves well. this is precisely the regime where a consolidation mechanism -- distilling kda's transient state into periodic compressed summaries that mla can attend over -- would provide genuine architectural value. the consolidation gap is not a current problem. it is a scaling wall.

there is a subtler point. in biology, consolidation is not just about extending memory duration -- it is about changing the format of the memory. the hippocampal trace is episodic (bound to a specific time and place). the cortical trace is semantic (abstracted across many episodes). the compression that consolidation performs is not just lossy in the information-theoretic sense -- it is transformative, converting a specific memory into a general rule. todorov has no mechanism for this kind of format conversion. kda's state is always a matrix of key-value associations. mla's cache is always a sequence of per-token latents. neither can transform its contents into a different representational format. the model learns one format during training and uses it unchanged at inference.

the one process in todorov that resembles consolidation is training itself. over many gradient steps, the model's weights slowly incorporate the statistical structure of the training corpus -- exactly what the neocortex does via slow synaptic modification. but this operates across the training dataset, not within a single context window. during inference, there is no learning. the dissociation is absolute: training is slow consolidation across millions of sequences; inference is fast computation within a single sequence; and nothing bridges the two.

## the outer product: the one genuine compression

among todorov's mechanisms, the outer product write in kda is the one that most genuinely implements compression in the biological sense.

each token writes k_t * v_t^T to the state matrix -- a rank-1 update that encodes the association between a key and a value as a single direction in a 64x64 space.

this is hebbian storage: the same outer product rule used in classical hopfield networks (see [[pattern_completion]]) and hypothesized for ca3 recurrent connections (see [[hebbian_learning]]). the mathematical form is identical: W += eta * x * y^T in biology, S_t += beta_t * k_t * v_t^T in kda. the correspondence is not metaphorical -- it is the same equation with different variable names.

the compression happens because rank-1 updates are inherently lossy. a 64x64 matrix has 4096 parameters, but a rank-1 matrix is fully specified by 128 parameters (64 for k, 64 for v). each write compresses a token's information into these 128 degrees of freedom, projected onto the existing state. when multiple writes accumulate, they interfere: constructively when keys are similar and values reinforce, destructively when keys are similar but values differ. this interference IS compression -- the state cannot hold all associations at full fidelity, so it retains a lossy summary that reflects the statistics of recent associations weighted by recency (via alpha decay).

the delta rule modification (see [[plasticity_to_matrix_memory_delta_rule]]) makes this compression more intelligent. instead of blindly adding k_t * v_t^T, the delta rule subtracts the current prediction error, writing only what the state does not already know: delta_t = v_t - S_{t-1}^T * k_t. this is error-correcting hebbian learning, and it implements adaptive compression that prioritizes novel information over redundant information. the delta rule is implemented but not yet validated at scale.

this is the strongest mechanistic parallel to hippocampal memory: one-shot hebbian binding that produces a lossy, interference-limited associative store. both systems use rank-limited outer products. both forget through interference (or, in kda's case, through decay and rank saturation). both sacrifice fidelity for speed. the parallel is genuine -- it follows from the shared mathematics of outer-product association, not from an analogy imposed after the fact.

the write-gating via beta_t = sigmoid(beta_proj(x_t)) adds a further biological resonance. not all tokens write equally to memory -- the beta gate is a learned importance signal that modulates write strength. in hippocampal terms, this is analogous to the neuromodulatory gating of encoding: acetylcholine modulates the balance between encoding and retrieval in CA1 (hasselmo and schnell 1994), and emotional significance (via amygdala) biases which experiences are stored. the analogy is structural, not mechanistic -- beta_t is a simple sigmoid, not a neuromodulatory cascade -- but the computational function is the same: not everything that happens deserves to be remembered.

but the parallel has limits.

ca3 retrieval is nonlinear auto-associative: a partial cue triggers recurrent dynamics that converge to the nearest stored attractor, producing error correction. the attractor dynamics are the mechanism by which ca3 "decompresses" a partial cue into a full memory -- the stored pattern is reconstructed by the convergent dynamics, not by a single readout step.

kda retrieval is linear heteroassociative: o_t = q_t^T * S_t is a single matrix-vector product with no iteration, no convergence, no error correction. the compression may be similar, but the decompression is fundamentally different. ca3 can recover a stored pattern from a noisy cue; kda returns whatever the linear projection produces, noise and all.

## challenges and counter-arguments

**the sparsity regime mismatch undermines the pattern separation analogy.** todorov's 41% firing rate is 8-20x higher than the dg's 2-5%. this is not a quantitative difference -- it is a qualitative one. at 2-5% sparsity, two random inputs activate almost entirely non-overlapping populations. at 41%, substantial overlap is expected. the function that extreme sparsity serves in biology (interference prevention through near-orthogonality) cannot be served at 41%. todorov's spikes may serve a different function (information compression, gradient-friendly quantization), but the pattern separation story does not transfer. any claim that "ternary spikes implement biological pattern separation" requires demonstrating near-orthogonality of spike patterns for similar inputs -- a test that has not been conducted.

**mla's compression is not a biological bottleneck -- it is a parameter efficiency trick.** the d_model -> d_c projection in mla was introduced by deepseek to reduce kv cache size during inference, not to implement an information bottleneck in the information-theoretic sense. the compression is lossless in principle: a sufficiently expressive up-projection can reconstruct the full representation from the 128-dimensional latent. whether information is actually lost depends on the learned projections, not on the architecture. comparing this to the dg's pattern separation (which is inherently destructive by design) conflates engineering convenience with computational principle. the dg cannot reconstruct the entorhinal input from its sparse output; mla can reconstruct d_model from d_c, by design.

**the "bottleneck as feature" argument proves too much.** any capacity limit can be retroactively justified as beneficial. a model with 1 hidden dimension is a bottleneck that forces extreme compression -- but no one would call this a feature. the argument needs quantitative grounding: what is the optimal bottleneck size for a given task? is kda's 64x64 state too small, too large, or just right? is mla's 8x compression optimal? the biological analog (dg's 2-5%) was tuned by millions of years of evolution for specific ecological demands. todorov's bottleneck sizes were set by engineering heuristics. claiming that both are "features" without evidence of optimality is an analog without teeth.

**the consolidation gap may not matter -- or it may be fatal.** the bridge article argues that consolidation is unnecessary at current context lengths. but this is an argument from current engineering constraints, not from principle. if todorov were to scale to the context lengths where humans operate (entire conversations, days of experience, lifetimes of accumulated knowledge), the absence of consolidation would become a hard architectural limit. the honest question is not "does todorov need consolidation at 2048 tokens?" (it does not) but "is the absence of consolidation a fundamental limitation of the design or an irrelevant biological detail?" this cannot be answered without scaling experiments that do not yet exist.

**the ramsauer inversion complicates the compression narrative.** ramsauer et al. (2021) showed that softmax attention IS the update rule of a modern hopfield network with exponential storage capacity ~2^{d/2}. this means mla -- todorov's "compressed" memory system -- has astronomically higher associative capacity per query than kda. for d_c = 128, the theoretical capacity is ~2^64 ~ 10^19 patterns. kda's capacity is 64 per head. the gap is not quantitative -- it is civilizational.

the compression bottleneck (d_c = 128) limits the dimensionality of stored representations, not the number of retrievable patterns. kda, the "uncompressed" recurrent state, is actually the more severely capacity-limited system, with linear (rank-limited) storage that scales as O(head_dim).

the biological narrative (compression = limited capacity = pattern separation) maps onto kda, not mla. the architecture concentrates 75% of its depth on the capacity-limited system and uses the high-capacity system sparingly -- a ratio that matches the cortical/hippocampal volume ratio, but only under the inverted mapping where mla is hippocampal (see [[matrix_memory_vs_hippocampus]]). this inversion is uncomfortable: the "compressed" system has more capacity than the "uncompressed" one. the lesson is that compression and capacity are not opposites. compression can increase effective capacity by reducing interference, exactly as the dg's extreme sparsity increases hippocampal storage by reducing inter-pattern overlap.

## the honest assessment

todorov has three genuine compression mechanisms: ternary spikes (quantization), mla latent projection (dimensionality reduction), and kda rank-limited state (capacity-limited association). all three demonstrably work -- the 0.663x bpb ratio at 267m scale proves that the model performs well despite (or because of) these bottlenecks.

the biological parallel is real but narrow.

the outer product write in kda genuinely shares mathematical structure with hebbian ca3 storage. both produce rank-limited, interference-prone associative memories. this is the strongest claim: not that kda "is" the hippocampus, but that both implement the same mathematical operation (outer-product association) and both face the same fundamental trade-off (capacity vs interference).

the weaker claims should be stated honestly.

ternary spikes are information bottlenecks, but they do not implement pattern separation in any regime comparable to the dg. the 41% firing rate produces a qualitatively different code from the 2-5% code that biology uses for interference prevention.

mla's compression is a dimensionality reduction, but it was designed for parameter efficiency, not for the information-theoretic function that biological bottlenecks serve. the compression is invertible by design, unlike the dg's inherently destructive sparsification.

the consolidation gap is real and becomes more significant at longer context lengths. no inter-system transfer exists. training serves as a slow consolidation process across the dataset, but nothing operates within a single inference pass.

the three biological compression stages (sparse encoding, compressed indexing, lossy abstraction) have at most one genuine analog in todorov (the outer product write as lossy associative compression), and even that analog operates without the attractor dynamics that give ca3 its error-correcting power. the strongest version of the compression story is not "todorov implements biological compression" but rather "todorov and biology converge on the same mathematical primitive (outer-product association) for the same reason (it is the cheapest way to write associative memories), and both pay the same price (rank-limited capacity and interference-based information loss)."

what remains is a research program, not a conclusion.

three experiments could sharpen this assessment. first, sweep the spike threshold alpha from 0.1 to 5.0 and measure bpb and spike mi: if extreme sparsity (2-5%) emerges as optimal, the biological parallel gains force; if 41% is optimal, the systems solve different problems with the same primitive. second, measure the cosine similarity between ternary spike patterns for semantically similar vs dissimilar inputs: if similar inputs produce near-orthogonal spike patterns, pattern separation is real despite the high firing rate; if they produce correlated patterns (as expected at 41% density), the compression is gradient-shaped, not interference-preventing. this is the critical experiment: it distinguishes between "spikes implement pattern separation" and "spikes implement information compression," two claims with very different biological implications. third, implement a consolidation probe: periodically snapshot kda state and measure how much information about tokens beyond the decay horizon is recoverable from that state vs from mla's cache. this quantifies the consolidation gap in bits, not in analogies. if kda retains essentially zero information beyond ~20 tokens while mla retains everything, the case for a consolidation mechanism at long context becomes empirically grounded.

## see also

- [[hippocampal_memory]]
- [[pattern_completion]]
- [[complementary_learning_systems]]
- [[memory_consolidation]]
- [[matrix_memory_vs_hippocampus]]
- [[memory_systems_to_matrix_memory_and_compressed_attention]]
- [[sparse_coding_to_ternary_spikes]]
- [[plasticity_to_matrix_memory_delta_rule]]
- [[compression_beyond_quantization]]
