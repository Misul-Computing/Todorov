# modern memory and compression pressure

status: current (as of 2026-05-12).

## role

this file records the modern public pressure on neuroloc compression. the point is not to copy a paper name. the point is to learn what each line of work already proves, what it does not prove, and what baseline it imposes.

## test-time neural memory

### titans

source: [titans](https://arxiv.org/abs/2501.00663).

mechanism:

titans adds a neural long-term memory module updated at test time, while attention remains the short-term accurate dependency surface.

what it proves:

test-time-updated neural memory is a serious big-lab direction. reported results improve long-context modeling and recall-style tasks versus transformers and linear recurrent baselines.

what it does not prove:

it does not prove exact arbitrary key-value retrieval under strict state accounting. it does not prove a 600x density cell.

neuroloc pressure:

any future local memory must compare against surprise-gated test-time update baselines and must prove the learned memory path, not nearby attention or retained context, is doing the work.

### miras

source: [miras](https://arxiv.org/abs/2504.13173).

mechanism:

miras frames sequence models as associative memories specified by memory architecture, attentional-bias objective, retention gate, and learning algorithm.

what it proves:

the design space can be named precisely. attention, recurrent state, and test-time memory are not separate magic categories; they are different choices over memory, objective, retention, and update.

what it does not prove:

it is a framework, not a high-density exact storage result.

neuroloc pressure:

each neuroloc candidate should state its memory object, key-value map, update objective, retention rule, and learning rule in this style.

### atlas

source: [atlas](https://arxiv.org/abs/2505.23735).

mechanism:

atlas optimizes long-term memory using current and past tokens instead of only the newest token, attacking limited capacity, myopic online update, and weak memory management.

what it proves:

history-aware memory optimization is a serious baseline for long context and recall-intensive tasks.

what it does not prove:

if a method revisits past tokens, the retained context or training material may itself be storage. that must be charged before calling it compression.

neuroloc pressure:

include a history-aware memory optimizer baseline for any test-time memory proposal.

## sparse trainable memory

### product-key memory

sources: [large memory layers with product keys](https://arxiv.org/abs/1907.05242), [lucidrains product-key-memory](https://github.com/lucidrains/product-key-memory).

mechanism:

product-key memory factorizes a large key space into subkeys, allowing efficient sparse lookup into a large value table.

what it proves:

large sparse trainable value storage can improve capacity and factual behavior with lower compute than dense expansion.

what it does not prove:

it mostly buys storage. the value table is the memory and must be charged.

neuroloc pressure:

this is the mandatory learned-address table baseline. if a proposed cell is worse than product-key memory at matched storage and exact retrieval, it is not useful.

### memory layers at scale

sources: [memory layers at scale](https://arxiv.org/abs/2412.09764), [facebookresearch memory repo](https://github.com/facebookresearch/memory).

mechanism:

scaled trainable sparse key-value memory layers add many memory parameters without proportional flops.

what it proves:

sparse memory layers remain useful at contemporary scale and are especially relevant for factual tasks.

what it does not prove:

capacity expansion is not strict compression. exact per-fact recovery and random-label controls are not the same as aggregate benchmark gains.

neuroloc pressure:

future claims need a sparse trainable memory-layer baseline, not only no-memory and recency-only controls.

## external retrieval and context memory

### memorizing transformers

source: [memorizing transformers](https://arxiv.org/abs/2203.08913).

mechanism:

a transformer performs approximate nearest-neighbor lookup into a non-differentiable memory of recent key-value pairs.

pressure:

if a target fact appears in context, an explicit recent key-value store is a hard baseline. charge the datastore, embeddings, values, and index.

### retro

source: [retro](https://arxiv.org/abs/2112.04426).

mechanism:

a model retrieves neighboring chunks from a massive external corpus and conditions generation through cross-attention.

pressure:

retrieval separates storage from reasoning. a dense internal memory must justify itself against explicit corpus retrieval.

### compressive transformer

source: [compressive transformer](https://arxiv.org/abs/1911.05507).

mechanism:

recent memories are retained, older memories are compressed to lower-resolution states.

pressure:

lower perplexity from compressed activations is not exact retrieval. neuroloc needs operation-preserving tests.

### infini-attention

source: [infini-attention](https://arxiv.org/abs/2404.07143).

mechanism:

local attention and long-term linear compressive memory are combined in one transformer block.

pressure:

bounded streaming memory is a baseline for long-context use. it does not imply infinite exact recall.

## recurrent and linear-state alternatives

### mamba

sources: [mamba](https://arxiv.org/abs/2312.00752), [state-spaces mamba repo](https://github.com/state-spaces/mamba).

mechanism:

selective state-space recurrence uses input-dependent parameters to propagate or forget information.

pressure:

any fixed recurrent state must beat selective recurrent baselines on exact delayed retrieval, not just sequence modeling.

### rwkv

sources: [rwkv paper](https://arxiv.org/abs/2305.13048), [rwkv organization](https://github.com/rwkv).

mechanism:

rwkv mixes recurrent inference with transformer-like training and weighted key-value state.

pressure:

constant-memory recurrent inference is not constant-loss exact memory. test interference, sequence length, and rare-key recall.

### retnet

source: [retnet](https://arxiv.org/abs/2307.08621).

mechanism:

retention supports parallel, recurrent, and chunkwise recurrent computation.

pressure:

retention is a compressed summary. exact retrieval tasks must show what survives.

### xlstm

sources: [xlstm](https://arxiv.org/abs/2405.04517), [xlstm repo](https://github.com/nx-ai/xlstm).

mechanism:

modernized gated recurrent memory with scalar and matrix memory forms.

pressure:

gated matrix memory is close to neuroloc's memory language, so exact capacity and bit accounting are mandatory.

### gated delta-style fast weights

sources: [gated deltanet](https://arxiv.org/abs/2412.06464), [nvlabs gateddeltanet](https://github.com/NVlabs/GatedDeltaNet).

mechanism:

matrix-valued recurrent state receives gated delta-style updates.

pressure:

this is a strong learned update baseline for associative write/read. exact storage still faces interference and entropy limits.

## discrete bottlenecks and neural codecs

### vector quantization

source: [vq-vae](https://arxiv.org/abs/1711.00937).

mechanism:

an encoder maps inputs to discrete codebook entries and a decoder reconstructs from those entries.

pressure:

discrete latents are the natural bridge to a real learned compression cell. exact byte retrieval requires exact reconstruction metrics and charged codebooks.

### residual vector quantization and perceptual codecs

sources: [soundstream](https://arxiv.org/abs/2107.03312), [encodec docs](https://facebookresearch.github.io/audiocraft/docs/ENCODEC.html).

mechanism:

stacked quantizers encode successive residuals for neural audio compression.

pressure:

residual quantization is useful, but exact knowledge cannot tolerate plausible reconstruction errors. per-fact residual streams must not become hidden tables.

### language modeling as compression

sources: [language modeling is compression](https://arxiv.org/abs/2309.10668), [deepmind repo](https://github.com/google-deepmind/language_modeling_is_compression).

mechanism:

a predictor plus arithmetic coder becomes a lossless compressor when it assigns probabilities to the sequence.

pressure:

this is the cleanest foundation for the next neuroloc candidate. better modeling means shorter ideal code length. the arithmetic coder does not beat entropy by itself.

### factual knowledge capacity

source: [physics of language models part 3.3](https://arxiv.org/abs/2404.05405).

mechanism:

the paper measures factual knowledge storage in transformers on controlled tuples.

pressure:

the ordinary factual-knowledge comparison bar around `2` bits per parameter is not a license to ignore state bits. the neuroloc 600x target must name the denominator and charge state.

## neuro-symbolic and embodied pressure

### tufts neuro-symbolic robotics

sources: [tufts hrilab](https://hrilab.tufts.edu/), [diarc project](https://hrilab.tufts.edu/projects/diarcrepo), [open-world novelty framework](https://hrilab.tufts.edu/publications/goeletal24aij.pdf), [few-shot neuro-symbolic imitation learning](https://hrilab.tufts.edu/publications/lorangetal25rss.pdf).

mechanism:

explicit symbolic planning, abstraction, language grounding, and learned low-level control are combined for robot tasks.

pressure:

if a task has explicit state and rules, symbolic state plus planner is a strong baseline. neuroloc must distinguish learned memory from state bookkeeping.

### tribe and tribe v2

sources: [tribe](https://arxiv.org/abs/2507.22229), [tribev2 repo](https://github.com/facebookresearch/tribev2).

mechanism:

tribe and tribe v2 are brain-encoding models that map text, audio, and video representations to fmri responses.

pressure:

they are useful for biological-alignment framing and multimodal brain-response evaluation. they do not prove memory compression or exact knowledge storage.

## synthesis

the field already covers most easy stories:

- sparse memory buys capacity.
- retrieval buys factual access.
- recurrent state buys streaming efficiency.
- neural codecs buy lossy or probabilistic compression.
- symbolic planners buy exact state manipulation.

neuroloc's remaining edge must be narrower and harder: a learned memory object that preserves exact operations with fewer charged bits than explicit storage, sparse read, product-key memory, and standard codecs, while random-label controls collapse.

