# neural model related-work pressure matrix

status: current (as of 2026-05-07).

## role

this page consolidates the 2026-05-06 multi-lane research pass after the subq / selective sparse attention update. it records what external work already covers, what can translate into the neural-model program, which baselines become mandatory, and what remains a defensible project claim if proved.

this is a research synthesis and planning page. it does not authorize paid compute, full model integration, simulator choice, or architecture preset work.

## executive conclusion

the literature covers many of the pieces:

- local cellular state and eligibility traces.
- fast weights and test-time memory.
- differentiable memory and content-addressed reads.
- dynamic sparse attention and content-routed long-context reads.
- semantic, latent, and task-relative compression.
- recurrent context compression and compressed text context.
- replay, consolidation, and latent imagination.
- world models and embodied simulators.
- long-context evaluation and oracle-style failure localization.

the literature does not appear to already prove the stronger neural-model claim:

```text
a learned compact memory-object code that preserves retrieval, action, reconstruction, replay/rewrite, rollout, and provenance operations under legal source visibility and held-out exact-world controls, while beating verbatim sparse-read and latent-token baselines on useful bits per successful operation.
```

that is the narrow paper edge. it is conditional. it remains unproved.

## what this changes

the project should not argue from novelty by component. nearly every component has prior art. the claim must be compound and proof-gated.

the next local code target is now sharpened by the implemented sparse-read baseline, the external demotion decision, and the first tiny distributed local learned result. the original source-pair `compression_under_bit_budget` mirror is a source-selection diagnostic, not compression evidence. matched-budget sparse read and a distributed-evidence probe are implemented. a 25,975-parameter local model now passes the distributed-evidence slice at 19 bits, so the next target is stronger local falsification rather than claim expansion.

the next research standard has changed in code as well as prose. the first content-routed sparse-read baseline over verbatim records is now implemented for `compression_under_bit_budget`; it solves the repaired source-pair smoke task from two legal records but exceeds the compact-code budget. matched-budget sparse read fails, and the distributed-evidence probe now requires four legal fragments for uncapped sparse read while still failing under the matched budget.

the simulator lane remains a selection gate. do not choose a simulator yet. run a small exact-state contract spike across candidate simulators before promoting one.

## lane 1: cellular and local state storage

### prior art pressure

local state is not new. the relevant prior art includes:

- short-term synaptic facilitation/depression as activity-silent working memory: Mongillo, Barak, and Tsodyks, [synaptic theory of working memory](https://doi.org/10.1126/science.1150769).
- timing and eligibility traces: Markram et al., [regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs](https://doi.org/10.1126/science.275.5297.213); Izhikevich, [solving the distal reward problem through linkage of STDP and dopamine signaling](https://izhikevich.org/publications/dastdp.htm); Bellec et al., [a solution to the learning dilemma for recurrent networks of spiking neurons](https://www.nature.com/articles/s41467-020-17236-y).
- synaptic tagging and capture: Frey and Morris, [synaptic tagging and long-term potentiation](https://pubmed.ncbi.nlm.nih.gov/9020359/); Redondo and Morris, [making memories last](https://www.nature.com/articles/nrn2963).
- dendritic and compartmental computation: Gidon et al., [dendritic action potentials and computation in human layer 2/3 cortical neurons](https://doi.org/10.1126/science.aax6239); Beniaguev, Segev, and London, [single cortical neurons as deep artificial neural networks](https://doi.org/10.1016/j.neuron.2021.07.002); Chavlis and Poirazi, [dendritic artificial neural networks](https://www.nature.com/articles/s41467-025-56297-9).
- metaplasticity and adaptive thresholds: Bienenstock, Cooper, and Munro style thresholding reviewed in [bcM theory at 30](https://www.nature.com/articles/nrn3353); Abraham and Bear, [metaplasticity](https://pubmed.ncbi.nlm.nih.gov/8658594/).
- fast weights and learned plasticity: Ba et al., [using fast weights to attend to the recent past](https://arxiv.org/abs/1610.06258); Schlag, Irie, and Schmidhuber, [linear transformers are secretly fast weight programmers](https://arxiv.org/abs/2102.11174); Miconi et al., [differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a.html); Behrouz et al., [titans](https://arxiv.org/abs/2501.00663).
- recent spiking language / ternary-state work: NeuronSpark, [0.9b spiking language model](https://arxiv.org/abs/2603.16148); complemented ternary spiking neurons, [complemented neurons and membrane potential aggregation](https://arxiv.org/abs/2601.15598).

### translation

the strongest project translation is a tagged eligibility buffer with sparse consolidation:

```text
tag_t = f(local_surprise_t, address_margin_t, context_gate_t)
elig_t = decay * elig_{t-1} + tag_t * local_payload_t
commit_t = consolidate_t * elig_t
read_t = query_gate_t * address_match(commit_state, query_t)
```

this separates mark, commit, retain, and read. it maps directly to the current failures: learned write, output-gate closure, address/payload/action localization, and decoder generalization.

### proof gate

minimum controls:

- oracle tag.
- learned tag.
- oracle consolidation.
- learned consolidation.
- oracle read.
- learned read.
- no-memory.
- recency-only.
- dense-write under matched budget.
- shuffled-address.
- content-routed sparse read where a verbatim memory field exists.

required telemetry:

- tag sparsity.
- commit sparsity.
- useful commit precision and recall.
- address entropy.
- read concentration.
- target-to-distractor read ratio.
- memory-output norm.
- decoder error by field.
- bits written per successful episode.

kill conditions:

- learned commit stays closed while oracle commit works.
- dense writes match sparse tags at equal bit budget.
- learned state cannot generalize to held-out source combinations.
- memory read path is unused or opens into noise.
- result improves loss but not exact state/action.

## lane 2: operation-preserving compression

### prior art pressure

the pieces are heavily covered:

- information bottleneck and task-relative compression: Tishby, Pereira, and Bialek, [information bottleneck method](https://www.ee.columbia.edu/~dpwe/papers/TishPB99-infobneck.pdf); [semantic compression with side information](https://arxiv.org/abs/2208.06094); [predictive rate-distortion](https://www.mdpi.com/1099-4300/21/7/640).
- semantic memory compression: [optimal forgetting as semantic compression of episodic memories](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008367); [compression in visual working memory](https://dash.harvard.edu/entities/publication/a9b32c42-474e-4d98-83cb-9213f6bccce7).
- behavior-preserving abstraction: Abel et al., [near optimal behavior via approximate state abstraction](https://proceedings.mlr.press/v48/abel16.html); [reward-predictive representations](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008317).
- latent/context compression: Rae et al., [compressive transformer](https://arxiv.org/abs/1911.05507); [in-context autoencoder](https://www.microsoft.com/en-us/research/publication/in-context-autoencoder-for-context-compression-in-a-large-language-model/); [recurrent context compression](https://arxiv.org/abs/2406.06110); Microsoft, [memento](https://www.microsoft.com/en-us/research/articles/memento-teaching-llms-to-manage-their-own-context/).
- cache and retrieval-preserving vector compression: [product quantization](https://ieeexplore.ieee.org/document/5432202), [DiskANN](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html), [RaBitQ](https://arxiv.org/abs/2405.12497), [TurboQuant](https://arxiv.org/abs/2504.19874).
- engineering analogies for schema/provenance: [protocol buffers](https://protobuf.dev/overview/), [apache parquet](https://parquet.apache.org/), [PROX provenance](https://pmc.ncbi.nlm.nih.gov/articles/PMC5001561/), and [provenance semirings](https://repository.upenn.edu/entities/publication/f1141264-46ee-4d61-b5ea-4ee75fb8d1be).

### translation

define compression by operations:

```text
retrieve(q, c)
act(q, c)
reconstruct_fields(q, c)
rewrite(c, evidence)
rollout(c, actions)
audit(c)
```

the candidate code shape remains:

```text
code = schema_id + address + residual_payload + operation_flags + provenance_id
```

the decoder must be evaluated through required-field recovery, action agreement, retrieval under query shifts, replay/rewrite consistency, rollout consistency, and provenance.

### mandatory baseline stack

- full verbatim source.
- visible-source hand codec.
- oracle minimal-field codec.
- content-routed sparse read over verbatim records.
- latent-token compressor.
- vector-quantized cache or retrieval geometry baseline.
- no-memory.
- recency-only.
- shuffled-address.
- shuffled-provenance.
- oracle-address / learned-payload.
- learned-address / oracle-payload.
- oracle-code / learned-decoder.
- learned-code / oracle-decoder.

### kill conditions

- compressed code beats no-memory but loses to sparse verbatim read on success or useful bits.
- bits fall while action, state, or provenance fails.
- decoder hides information and provenance becomes unauditable.
- learned code only works with oracle schema labels.
- held-out schema generalization fails.

## lane 3: memory, addressing, replay, and imagination

### prior art pressure

memory and replay are broad prior art:

- external differentiable memory: Graves et al., [neural turing machines](https://arxiv.org/abs/1410.5401); Weston et al., [memory networks](https://arxiv.org/abs/1410.3916); Graves et al., [differentiable neural computer](https://www.nature.com/articles/nature20101).
- associative retrieval: Kanerva, [sparse distributed memory](https://books.google.com/books?hl=en&id=I9tCr21-s-AC); Ramsauer et al., [modern associative retrieval](https://arxiv.org/abs/2008.02217).
- long-memory llm systems: Rae et al., [compressive transformer](https://arxiv.org/abs/1911.05507); Roy et al., [routing transformer](https://aclanthology.org/2021.tacl-1.4/); Wu et al., [memorizing transformers](https://arxiv.org/abs/2203.08913); Borgeaud et al., [RETRO](https://arxiv.org/abs/2112.04426); Wang et al., [LongMem](https://arxiv.org/abs/2306.07174); Zhong et al., [MemoryBank](https://arxiv.org/abs/2305.10250); Das et al., [Larimar](https://arxiv.org/abs/2403.11901).
- world-model imagination: Ha and Schmidhuber, [world models](https://arxiv.org/abs/1803.10122); Weber et al., [imagination-augmented agents](https://arxiv.org/abs/1707.06203); Wayne et al., [MERLIN](https://arxiv.org/abs/1803.10760); Schrittwieser et al., [MuZero](https://www.nature.com/articles/s41586-020-03051-4); Hafner et al., [DreamerV3](https://arxiv.org/abs/2301.04104).
- replay and consolidation: McClelland, McNaughton, and O'Reilly, [complementary learning systems](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf); Wilson and McNaughton, [hippocampal replay](https://pubmed.ncbi.nlm.nih.gov/8036517/); Nader, Schafe, and LeDoux, [reconsolidation](https://www.nature.com/articles/35021052); Mattar and Daw, [prioritized memory access](https://www.nature.com/articles/s41593-018-0232-z).

### translation

split every memory claim into operations:

1. write selection.
2. address formation.
3. read routing.
4. post-read use.
5. replay rewrite.
6. latent branch compression.
7. consolidation across tiers.

the project should not accept "memory improved" as a result. each operation must have an oracle split and telemetry.

### required telemetry

```text
write_precision
write_recall
gate_open_fraction_relevant
gate_open_fraction_distractor
memory_output_norm_ratio
address_margin_mean
address_margin_p05
slot_entropy
write_collision_count
read_concentration
target_selection_recall
selected_record_count
interference_slope
retention_over_delay
bits_committed_per_episode
state_probe_accuracy
action_success
joint_success
reconstruction_error
provenance_accuracy
replay_selection_accuracy
rewrite_bits_delta
rewrite_state_delta
branch_uncertainty_ece
hard_case_rollout_gain
easy_case_rollout_gain
answer_flip_rate_by_iteration
```

### kill conditions

- replay does not beat random replay under matched compute.
- branch rollout helps easy and hard cases equally.
- address formation collapses under correlated keys.
- post-read use fails even when oracle read succeeds.
- learned write fails while oracle write succeeds.
- content-routed sparse read over verbatim memory is stronger than the compressed path.

## lane 4: 3d world model and physics

### candidate pressure

the simulator is not chosen. the research pass ranks candidates for a small exact-state contract spike:

1. Kubric + PyBullet + Blender: [paper](https://arxiv.org/abs/2203.03570), [docs](https://kubric.readthedocs.io/en/latest/).
2. ThreeDWorld / TDW: [site](https://www.threedworld.org/), [paper](https://arxiv.org/abs/2007.04954).
3. AI2-THOR / ProcTHOR: [metadata docs](https://ai2thor.allenai.org/ithor/documentation/environment-state/), [ProcTHOR](https://arxiv.org/abs/2206.06994).
4. Habitat 2.0 / ReplicaCAD: [paper](https://arxiv.org/abs/2106.14405), [docs](https://aihabitat.org/docs/habitat-lab/habitat2.html).
5. MiniWorld: [docs](https://miniworld.farama.org/index.html), [paper](https://arxiv.org/abs/2306.13831).
6. CLEVRER as an external counterfactual-language benchmark: [paper](https://arxiv.org/abs/1910.01442).
7. BEHAVIOR / iGibson / OmniGibson: [BEHAVIOR concepts](https://behavior.stanford.edu/getting_started/important_concepts.html), [iGibson 2.0](https://arxiv.org/abs/2108.03272).
8. MineDojo / Minecraft: [site](https://minedojo.org/), [paper](https://arxiv.org/abs/2206.08853).
9. PHYRE / Virtual Tools as cheap 2d control baselines: [PHYRE](https://phyre.ai/), [paper](https://papers.neurips.cc/paper/8752-phyre-a-new-benchmark-for-physical-reasoning.pdf).

### next proof

run a no-paid local selection spike across Kubric, TDW, AI2-THOR, and MiniWorld only when the project is ready to touch simulator code. each candidate must export the same tiny contract:

```text
seed
world_id
object_list
full_hidden_state_per_tick
observation_per_tick
language_query
discrete_action_set
answer_or_action_target
counterfactual_branch_spec
oracle_answer
no_memory_baseline
recency_baseline
replay_hash
```

survival condition:

- repeated runs produce byte-identical or tolerance-bounded contract hashes.
- hidden state is exact and not leaked into observations.
- at least three families work: occluded object persistence, delayed dynamics, and counterfactual branch answer.
- local setup is cheap enough for development.

kill conditions:

- determinism fails.
- metadata omits needed velocity/contact/object state.
- language templates leak answers.
- setup or rendering is too heavy for local proof work.
- task success depends on control skill rather than memory/physics state.

## lane 5: trainability and evaluation

### prior art pressure

evaluation work already shows that context length and memory length are not enough:

- Weston et al., [bAbI tasks](https://arxiv.org/abs/1502.05698), for isolated reasoning skills.
- Liu et al., [lost in the middle](https://arxiv.org/abs/2307.03172), for position sensitivity.
- Hsieh et al., [RULER](https://arxiv.org/abs/2404.06654), for configurable long-context retrieval and reasoning.
- [MRCR](https://arxiv.org/abs/2409.12640) and the [OpenAI MRCR dataset](https://huggingface.co/datasets/openai/mrcr), for duplicate-needle disambiguation.
- [SCBench](https://arxiv.org/abs/2412.10319), for cache generation, compression, retrieval, and loading.
- Jozefowicz et al., [empirical exploration of recurrent network architectures](https://proceedings.mlr.press/v37/jozefowicz15.html), for gate-bias trainability.
- Tallec and Ollivier, [chrono initialization](https://arxiv.org/abs/1804.11188), for timescale-aware gate initialization.
- Agresti and Coull, [intervals for binomial proportions](https://www.tandfonline.com/doi/abs/10.1080/00031305.1998.10480550), for near-zero binary success intervals.
- NIST, [confidence intervals for information retrieval measures](https://www.nist.gov/publications/computing-confidence-intervals-common-ir-measures), for bootstrap/IR metric reporting.

### failure-localization ladder

every hard symbolic or mirror task should eventually expose:

1. oracle-write / oracle-read.
2. oracle-write / learned-read.
3. learned-write / oracle-read.
4. learned-address / oracle-payload.
5. oracle-address / learned-payload.
6. oracle-code / learned-decoder on held-out worlds.
7. visible-source / learned-code forbidden-input guard.
8. learned-code / shuffled-decoder negative control.

failure interpretation:

- oracle-write / oracle-read fails: generator/evaluator/decoder contract bug.
- oracle succeeds and no-memory succeeds: task is leaky or too easy.
- oracle-write / learned-read fails: read/address failure.
- learned-write / oracle-read fails: write/payload formation failure.
- oracle-address / learned-payload fails: representation/value failure.
- learned-address / oracle-payload fails: address/routing failure.
- oracle-code / learned-decoder train succeeds but held-out fails: decoder memorization.
- all ladders pass but end-to-end fails: optimization or gating failure.
- learned beats weak controls but loses to sparse read or knn: useful engineering result, not a strong compression claim.

### required sweeps

- distance: 16, 64, 128, 256, 512, 1024 where cheap.
- position: beginning, early-middle, center, late-middle, end.
- duplicate needles: 2, 4, 8 similar source events.
- distractor density.
- correlated-key interference.
- repeated writes with changed payload.

### statistical requirement

binary success must report Wilson intervals, including zero-success cases. aggregate metrics should report bootstrap or equivalent confidence intervals when used for claims.

## lane 6: content-routed sparse read prior

this lane is already promoted to [[content_routed_sparse_read_prior]]. the important boundary is:

```text
content-routed sparse read is the strongest shallow baseline: it reduces read compute while preserving raw context. the neural-model claim begins only where raw-context sparse read is insufficient or less bit-efficient under equal operation success.
```

related sources include:

- SubQ, [how selective sparse attention makes long context practical](https://subq.ai/how-ssa-makes-long-context-practical).
- Roy et al., [routing transformer](https://arxiv.org/abs/2003.05997).
- Mohtashami and Jaggi, [landmark attention](https://arxiv.org/abs/2305.16300).
- Gupta et al., [top-k attention](https://aclanthology.org/2021.sustainlp-1.5.pdf).
- Liu et al., [retrievalattention](https://arxiv.org/abs/2409.10516).
- Tang et al., [quest](https://arxiv.org/abs/2406.10774).
- Jiang et al., [minference](https://arxiv.org/abs/2407.02490).
- Child et al., [sparse transformer](https://arxiv.org/abs/1904.10509).
- Beltagy et al., [longformer](https://arxiv.org/abs/2004.05150).
- Zaheer et al., [bigbird](https://arxiv.org/abs/2007.14062).

## immediate next steps

1. harden the current tiny distributed local learned result for the `compression_under_bit_budget` slice against factor-held-out splits, multiple seeds, hard-profile/local larger sweeps, and less hand-shaped feature extraction.
2. extend the sparse-read baseline to other relevant symbolic/mirror families only after the local proof-package decision is clear.
3. add the oracle ladder and fieldwise failure-localization tree to every future mirror result.
4. define the tagged eligibility buffer with sparse consolidation as the next cellular/local-state proof package candidate, not as model code.
5. prepare a simulator-selection contract page before touching simulator implementation.
6. do not start paid compute, full model integration, simulator choice, or architecture preset work.

## see also

- [[PROJECT_PLAN]]
- [[content_routed_sparse_read_prior]]
- [[neural_model_compression_stack]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[tests/compression_under_bit_budget_mirror]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[synthetic_shared_world_bridge]]
