# imagination computation research

status: current (as of 2026-04-25).

curated peer-reviewed research on computational mechanisms of imagination, novelty, and generative recombination. the central question for the neural machine: can the kda outer-product state and recurrent dynamics support novel generation -- producing outputs that are structured interpolations or recombinations of stored patterns rather than direct retrievals? the evidence suggests that outer-product memory has intrinsic generative properties above a critical load, and that compositional generalization requires specific inductive biases.

## novelty and feature superposition

### novelty as superposition in latent space

pham, T. et al. (2024). novelty emerges from feature superposition in neural networks. *neurips workshop on creativity and generative ai*.

key finding: novel outputs in neural networks arise from the superposition of learned features in latent representations. when a network is asked to generate or classify inputs outside its training distribution, the latent representation is a linear combination of features from multiple training categories, producing outputs that are novel (not matching any single training example) but structured (composed of recognizable elements). this is not a failure mode but a feature of distributed representation: the latent space supports interpolation and extrapolation by construction.

relevance to neural machine: todorov's recurrent state accumulates information via outer-product updates between ternary spike keys and continuous value vectors. the state matrix is inherently a superposition of all past key-value associations, weighted by decay. when a novel query arrives (a key pattern not seen during training), the state will return a structured interpolation of stored values -- exactly the mechanism pham et al. describe. the ternary spike quantization constrains queries to a discrete lattice, which may either help (queries snap to meaningful prototypes) or hurt (queries cannot reach the precise interpolation point needed).

confidence: medium. workshop paper with limited experimental scale. the theoretical argument (superposition as novelty) is plausible but the empirical validation is on simple generative tasks. caveat: the relationship between latent superposition and meaningful novelty (as opposed to noise) depends on the structure of the latent space.

## compositional generalization

### meta-learning for compositionality

lake, B. M. & baroni, M. (2023). human-like systematic generalization through a meta-learning neural network. *nature*, 623, 115-121.

key finding: standard neural networks fail at systematic compositional generalization: they cannot reliably apply known rules to novel combinations of known elements (e.g., learning "dax" means "jump twice" and "fep" means "while spinning" and correctly inferring "dax fep" means "jump twice while spinning"). lake and baroni showed that meta-learning for compositionality (mlc) -- training a network to be compositional by exposing it to a distribution of compositional tasks during meta-training -- enables human-like systematic generalization. the key insight is that compositionality is not a default property of neural networks but requires an explicit inductive bias, either architectural or procedural.

relevance to neural machine: todorov does not have an explicit compositional inductive bias. the crbr framework composes operations (compress, bilinear, rotate, quantize) in a fixed order, but this is architectural composition, not semantic composition over inputs. if the neural machine is to generate novel combinations of learned concepts (a key aspect of imagination), it may need a meta-learning or curriculum strategy that explicitly trains for compositionality. the ternary spike discretization might help by forcing inputs into a combinatorial code (each spike pattern is a combination of binary features), but this is speculative.

confidence: high. published in nature with human comparison experiments. the mlc method is well-validated. caveat: mlc requires a distribution of compositional tasks for meta-training; it is unclear how to construct this for autoregressive language modeling.

## world models and latent imagination

### dreamerv3: imagining in latent space

hafner, D. et al. (2023). mastering diverse domains through world models. *arxiv preprint*, subsequently published in *nature* (2025).

key finding: dreamerv3 learns a world model that predicts future latent states from actions and uses this model to "imagine" future trajectories entirely in latent space, without rendering actual observations. policy optimization occurs on imagined trajectories, enabling learning from orders of magnitude fewer real interactions than model-free methods. the world model uses a recurrent state space model (rssm) with discrete latent variables (32 categorical variables with 32 classes each), providing a compressed representation that is both expressive and efficient for imagination.

relevance to neural machine: dreamerv3's discrete latent imagination is architecturally parallel to what todorov could achieve with its recurrent state. the kda state matrix accumulates a compressed representation of the input history; running the kda recurrence forward without new inputs (or with synthetic/null inputs) would generate imagined future states. the ternary spike discretization is analogous to dreamerv3's categorical latents -- both impose a discrete bottleneck that regularizes the latent space. the key difference is that dreamerv3 explicitly trains the world model for prediction, while todorov's state is trained only for next-token prediction and may not have sufficient predictive structure for multi-step imagination.

confidence: high. dreamerv3 achieves state-of-the-art across 150+ benchmark domains. the nature publication confirms rigor. caveat: world models for language are much harder than for simulated environments because language dynamics are less regular.

## quality metrics for generation

### cmmd replaces fid

jayasumana, S. et al. (2024). rethinking fid: towards a better evaluation metric for image generation. *cvpr*.

key finding: the frechet inception distance (fid) has systematic biases: it assumes gaussian feature distributions (which real distributions violate), it is sensitive to sample size, and it can rank generative models incorrectly when distributions are non-gaussian. the authors propose cmmd (clip maximum mean discrepancy) as a replacement. cmmd uses a kernel-based two-sample test in clip feature space, is distribution-free, and has better statistical properties (lower variance, fewer samples needed). cmmd correlates better with human quality judgments than fid across multiple generative model comparisons.

relevance to neural machine: if todorov develops generative capabilities (imagination, creative recombination), quality metrics will be needed to evaluate the outputs. cmmd's distribution-free property is important because ternary spike representations are highly non-gaussian (they are discrete and sparse), making fid-like metrics unreliable. for language generation, cmmd's approach could be adapted using language model embeddings instead of clip features.

confidence: high. published at cvpr with extensive comparisons. caveat: cmmd is designed for images; language generation quality metrics (perplexity, bleu, human evaluation) have their own challenges.

### vendi score for diversity

friedman, D. & dieng, A. B. (2023). the vendi score: a diversity evaluation metric using kernel-based entropy.

key finding: the vendi score measures the effective number of distinct modes in a set of generated samples using the exponential of the von neumann entropy of the kernel similarity matrix. unlike fid (which measures distributional distance to a reference) or inception score (which measures confidence), the vendi score directly measures diversity without requiring a reference distribution. it detects mode collapse (low vendi score = few distinct patterns) and mode invention (high vendi score = many distinct patterns).

relevance to neural machine: diversity is a key property of imagination -- generating many variations of a concept rather than repeating the same pattern. the vendi score could measure whether todorov's recurrent state, when probed with varied queries, produces diverse versus repetitive outputs. for the kda state specifically, a low vendi score across query variations would indicate that the state has collapsed to a low-rank representation, while a high vendi score would indicate rich, distributed storage.

confidence: medium-high. theoretically well-motivated with clear implementation. caveat: the kernel choice affects the score, and there is no universal kernel for language representations.

## controlled generation

### logit arithmetic for controlled output

dekoninck, J. et al. (2024). controlled text generation via language model arithmetic. *international conference on learning representations (ICLR)*.

key finding: controlled generation can be achieved by simple arithmetic operations on the output logits of multiple language models or model states. to generate text with property A but not property B, compute: logits_output = logits_base + alpha * logits_A - beta * logits_B. this "language model arithmetic" enables fine-grained control over generation without retraining, fine-tuning, or modifying the model architecture. the method works because logit space is approximately linear for semantic properties.

relevance to neural machine: logit arithmetic could be applied to todorov's output layer to control the properties of generated text. more fundamentally, the linearity of logit space suggests that the recurrent state (which feeds the output layer) may also support linear arithmetic for controlling generation. if state_A represents "formal language" and state_B represents "informal language," then alpha * state_A + beta * state_B might produce text with intermediate formality. this is a concrete mechanism for imagination via state interpolation.

confidence: medium-high. demonstrated on standard language models with clear improvements in controllability. caveat: the linearity assumption breaks down for complex, multi-dimensional properties; the method is more effective for simple stylistic controls than for semantic content.

## outer-product memory and generation

### structured interpolation above critical load

kalaj, G. et al. (2024). associative memory with outer-product states and novel queries.

key finding: when an outer-product associative memory (the same mathematical object as todorov's kda state) is loaded above its critical capacity and queried with a novel key (not matching any stored key), it produces a structured interpolation of stored values rather than random noise or a single nearest-neighbor retrieval. the interpolation weights depend on the similarity between the novel query and stored keys, producing outputs that blend multiple memories in a structured way. below critical load, queries retrieve exact matches; above critical load, the memory transitions to a generative regime where novel queries produce novel (but structured) outputs.

relevance to neural machine: this is the most directly relevant finding for todorov's imagination question. the kda state is an outer-product memory, and it operates above critical load for any sequence longer than the state capacity. the prediction is that querying the kda state with novel ternary spike patterns (patterns not seen during the current context) will produce structured blends of stored associations -- exactly what imagination requires. the ternary spike quantization constrains novel queries to the discrete lattice of spike patterns, which may improve the quality of interpolation by ensuring queries are always "close" to some stored pattern.

confidence: medium. theoretical analysis with simulation validation on synthetic data. caveat: the result assumes random stored patterns and random novel queries; natural language patterns have correlations that may change the interpolation behavior.

### memorization-to-generalization phase transition

related to kalaj et al. and the broader statistical mechanics of associative memory.

key finding: associative memories undergo a phase transition from memorization (exact retrieval of stored patterns) to generalization (structured interpolation of stored patterns) as the load factor (number of stored patterns / capacity) increases past a critical threshold. below the threshold, the memory acts as a lookup table; above it, the memory acts as a generative model that interpolates between stored patterns based on query similarity. this phase transition is sharp in the thermodynamic limit and gradual in finite-size systems.

relevance to neural machine: the kda state's behavior depends on where it operates relative to this phase transition. for short sequences (few stored associations), the state is in the memorization regime and retrieval is exact. for long sequences (many stored associations), the state transitions to the generalization regime and queries produce interpolated outputs. the 3:1 architecture with mla layers may serve as a correction mechanism: when the kda state is in the generalization regime and interpolation is inaccurate, the mla layer can provide exact retrieval from its exponential-capacity cache. this division of labor -- kda for fast generative interpolation, mla for precise retrieval when needed -- is a concrete architectural hypothesis for how imagination and memory coexist.

confidence: medium. established result in statistical mechanics of neural networks. the application to modern recurrent architectures is by analogy. caveat: the phase transition assumes a specific energy landscape; the delta-rule update (error-correcting) may shift the critical point compared to the Hebbian (accumulative) update.

## see also

- [[pattern_completion]]
- [[free_energy_principle]]
- [[imagination_research]]
- [[sparse_distributed_representations]]
- [[delta_rule_theory]]
- [[kda_channel_gating]]
- [[compression_architecture]]
- [[sleep_and_dreaming_research]]
- [[compression_beyond_quantization]]

## relevance to the neural machine

### validated connections
- kda outer-product state produces structured interpolation above critical load -- this is a concrete mechanism for imagination
- ternary spike patterns form a discrete lattice of queries, potentially improving interpolation quality
- dreamerv3's discrete latent imagination is architecturally parallel to running kda recurrence with synthetic inputs
- logit arithmetic demonstrates that linear operations on model outputs/states can control generation

### challenged assumptions
- todorov has no explicit compositional inductive bias (mlc shows this requires specific training or architecture)
- the kda state is trained for next-token prediction, not for multi-step imagination -- predictive quality over imagined trajectories is unknown
- quality metrics (cmmd, vendi score) are designed for images, not language; adapting them requires domain-specific work

### open questions
- can the kda recurrence be run forward with null inputs to generate imagined future states, and are those states meaningful?
- does the memorization-to-generalization phase transition occur at a predictable sequence length for todorov's specific state dimensions?
- could meta-learning for compositionality (mlc) be integrated into todorov's training curriculum?
- what is the diversity (vendi score equivalent) of kda state outputs across varied novel queries?
