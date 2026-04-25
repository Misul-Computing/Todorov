# neural model compression stack

status: current (as of 2026-04-25).

## claim

the neural model needs compression at every memory level, not one kv-cache-style trick. the useful abstraction is a replaceable compression stack: each memory surface defines what operations must be preserved, then any codec that preserves those operations at lower rate can be swapped in.

the core claim is:

```text
compress memory by preserving required operations, not by preserving tensors.
```

this makes compression task-relative. a code is acceptable only if the model can still route, read, reconstruct, update, replay, imagine, or act with it.

## why this is not just quantization

kv-cache compression is the obvious shallow case. turboquant compresses high-dimensional key/value vectors while preserving geometric quantities used by attention, especially inner products. that is valuable because attention needs dot-product structure more than exact fp16 vectors.

the neural model needs the same discipline at more levels:

- transient state vectors need stable updates
- addresses need target separability and routing margins
- memory payloads need reconstructable task state
- episodes need reusable entity, context, and event structure
- imagined branches need provenance, uncertainty, and branch-local latent state
- replay needs rewrite targets that preserve invariants while discarding trace detail

so the target is broader than vector quantization:

```text
given a memory surface s and operation set omega_s,
find the smallest code c such that operations in omega_s remain correct.
```

## mathematical contract

for each memory surface `s`, define a codec:

```text
codec_s = (encode_s, decode_s, operate_s, residual_s, budget_s, uncertainty_s)
```

where:

- `encode_s(x, context) -> code`
- `decode_s(code, query, context) -> reconstruction`
- `operate_s(code, operation, query, context) -> result`
- `residual_s(x, reconstruction) -> error`
- `budget_s(code) -> committed_bits`
- `uncertainty_s(code, query, context) -> confidence`

the codec is judged by operation-preserving rate-distortion:

```text
loss_s = d_task_s + lambda * rate_s + mu * interference_s + rho * rewrite_cost_s
```

where:

- `d_task_s` is failure on the required operations, not generic mse
- `rate_s` is committed bits
- `interference_s` measures target/non-target collision and overwrite damage
- `rewrite_cost_s` measures whether compression requires excessive replay or decode work

a compression claim is valid only if it improves the Pareto frontier against verbatim storage and matched no-memory / recency-only / shuffled-address controls.

## memory object

a compressed memory record should carry enough structure to be audited and rewritten:

```text
memory_i = (
  address_i,
  code_i,
  schema_i,
  residual_i,
  provenance_i,
  uncertainty_i,
  timescale_i,
  supported_operations_i
)
```

the point is not that every field must exist in every implementation. the point is that the interface can represent all memory levels without forcing them to share the same codec.

## compression levels

### level 1: geometric cache compression

purpose: preserve dot products, distances, ranking, and value reconstruction for attention-like reads.

candidate family: rotation plus scalar/vector quantization, residual correction, low-rank projection, product codes.

success metric:

- attention-rank preservation
- inner-product error
- value reconstruction error
- downstream state/action success under fixed cache budget

novelty level: low by itself. useful as an engineering layer, not the main research claim.

### level 2: address compression

purpose: preserve routing, target identity, address margin, and target/non-target separation.

candidate family: compact learned addresses, orthogonalized codes, hashed entity/context addresses, entropy-regularized slot codes.

success metric:

- address margin
- slot entropy
- shuffled-address failure
- target-to-nontarget read ratio
- interference slope

novelty level: moderate if integrated with trainability controls and learned writes. not enough alone.

### level 3: payload compression

purpose: store what cannot be reconstructed by a shared decoder or schema prior.

candidate family: schema plus residual, compact handles, residual quantization, field-factorized value codes.

success metric:

- bits written per successful episode
- reconstruction error
- semantic and action-relevant retention
- success at fixed bit budget

novelty level: high if trained end to end inside the neural model rather than supplied by oracle labels.

### level 4: episodic compression

purpose: convert observation sequences into entity, context, event, and relation summaries.

candidate family: entity-state summaries, event graphs, context-bound traces, surprise-filtered episode writes.

success metric:

- delayed use under partial observability
- episodic reuse after distractors
- degraded-cue recall
- provenance-correct reconstruction

novelty level: high if it remains fixed-size and does not grow an external unbounded database.

### level 5: imagination compression

purpose: store imagined branches as compact latent programs, not generated traces.

an imagined branch should store:

```text
branch = (
  source_memory_ids,
  start_state_code,
  intervention_code,
  transition_handle,
  predicted_outcome_code,
  residual_surprises,
  uncertainty,
  branch_provenance
)
```

the model should not store every step of an imagined rollout. it should store the rule, intervention, start state, outcome, and surprises that made the branch worth keeping.

success metric:

- reconstruction of branch outcome
- action success after imagined branch use
- hard-case rollout gain larger than easy-case gain
- branch uncertainty calibrated to failure
- replay/rewrite can compress the branch further without losing task state

novelty level: very high if learned. this is where compression, reasoning, and imagination become the same mechanism.

### level 6: replay-rewrite compression

purpose: retrieve memories and rewrite them into smaller invariant forms.

candidate family: targeted replay, residual decay, schema promotion, repeated-episode consolidation, branch pruning.

success metric:

- targeted replay beats random replay
- rewrite lowers bits while preserving state/action/joint success
- old memories become cheaper without becoming hallucinated
- provenance remains auditable

novelty level: high if the rewrite is differentiable or locally trainable and has a kill condition.

### level 7: world-state compression

purpose: store the minimal latent state that explains many observations and imagined continuations.

candidate family: object-state models, causal state codes, shared generative decoder, compact world handles.

success metric:

- compression ratio relative to raw observation traces
- prediction under occlusion
- reconstruction under query
- transfer to new views of the same latent world
- failure under adversarial relevance shifts is detected by uncertainty

novelty level: highest. this is the only level where extreme compression ratios, including 100x or higher on structured worlds, are plausible.

## is it novel?

not at the component level.

the pieces already exist separately: vector quantization, rate-distortion theory, sparse coding, predictive coding, learned latent compression, world models, indexing theory, schema memory, replay, and residual coding.

the potentially novel claim is the compound system:

```text
a neural memory architecture where every memory surface is compressed by a swappable operation-preserving codec, including imagined branches and replay rewrites, and where success is measured by state/action/reconstruction per committed bit under strict controls.
```

that is not the same as kv-cache compression. kv-cache compression preserves one operation family. the proposed stack preserves different operation families at different memory levels and treats compression as a first-class trainable substrate.

the novelty is conditional. it becomes real only if the project proves:

- a trained tiny mirror can learn at least one non-oracle codec
- compressed memory beats verbatim storage on a Pareto frontier
- the win holds under delayed use, distractors, interference, and context shifts
- imagined branches can be stored as compact latent programs and later used
- replay can rewrite memories into smaller forms without losing task state
- the interface lets codecs be replaced without changing the rest of the model

without those proofs, it is only a good research thesis.

## is it useful?

yes, if it works. it attacks the actual bottleneck:

- memory cannot grow linearly with experience
- verbatim traces interfere
- current substrates do not learn useful writes
- imagination cannot store full rollouts
- next-token loss does not force the model to build reusable state

a useful compression stack would give:

- more recoverable task state per committed bit
- lower overwrite pressure
- better delayed recall
- compact imagined branches
- replay that improves memory instead of merely rehearsing it
- a path to fixed-size continual learning

the failure mode is also clear:

- the codec may become lossy summarization
- the model may hide information in the decoder and fail provenance
- compact codes may collapse under interference
- branch compression may turn imagination into hallucinated prior replay
- the trainable mirror may learn only with oracle schema labels

those failures are acceptable only if the tests expose them early.

## expected compression ratios

there is no universal ratio.

for arbitrary high-entropy data, extreme lossless compression is impossible. for structured latent worlds, large useful compression is possible because the per-memory code stores only conditional information given a shared decoder and world model.

expected order:

- geometric cache compression: 4x to 8x is realistic, based on current vector-quantization literature
- address compression: unknown; must be measured by margin and interference
- payload schema/residual compression: 10x to 100x is plausible on structured worlds
- episode-to-state compression: 50x to 500x is plausible when repeated observations share entities, dynamics, and context
- imagined-branch compression: 100x or more is plausible if the branch is generated by compact dynamics and only surprises are stored
- arbitrary content: no extreme guarantee

the right near-term target is not to promise 600x. it is to construct worlds where the oracle latent state has a known compression ratio and then measure whether a tiny model can approach that oracle.

## required next proof

before paid compute, build an oracle compression analysis over the existing hard symbolic worlds:

1. compute verbatim trace bits
2. compute ground-truth latent state bits
3. compute schema/residual bits
4. compute imagined-branch program bits
5. report oracle compression ratios per family
6. identify which families can theoretically support 10x, 100x, or higher compression
7. train the tiny mirror only after the oracle ratios are known

if the oracle cannot produce large compression on a constructed world, a neural model will not discover it by paid training.

## decision rule

do not use paid compute to search for this. use paid compute only after:

- the oracle compression ratios are known
- a tiny trainable mirror learns at least one codec above controls
- telemetry shows the intended compressed path is used
- replay and imagination compression have explicit failure cases
- prosecutor review has zero findings

## see also

- [[compression_beyond_quantization]]
- [[indexed_reconstruction_compression]]
- [[neural_model_dossier_compression]]
- [[neural_model_research_test_material_plan]]
- [[tests/hard_symbolic_nm_test_material]]
- [[PROJECT_PLAN]]

## references

- [zandieh et al. 2025, turboquant: online vector quantization with near-optimal distortion rate](https://huggingface.co/papers/2504.19874)
- [shannon 1959, coding theorems for a discrete source with a fidelity criterion](https://ieeexplore.ieee.org/document/1057159)
- [rao and ballard 1999, predictive coding in visual cortex](https://www.nature.com/articles/nn0199_79)
- [olshausen and field 1996, sparse coding of natural images](https://www.nature.com/articles/381607a0)
- [teyler and discenna 1986, hippocampal memory indexing theory](https://pubmed.ncbi.nlm.nih.gov/3008780/)
- [mcclelland, mcnaughton, o'reilly 1995, complementary learning systems](https://doi.org/10.1037/0033-295X.102.3.419)
- [van den oord, vinyals, kavukcuoglu 2017, neural discrete representation learning](https://arxiv.org/abs/1711.00937)
- [balle et al. 2018, variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436)
- [hafner et al. 2023, mastering diverse domains through world models](https://arxiv.org/abs/2301.04104)
