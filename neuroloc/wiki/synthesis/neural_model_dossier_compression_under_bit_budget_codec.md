# neural model dossier: compression under bit budget codec

status: current (as of 2026-04-27).

## claim

the first learned compression target is a narrow codec for the `compression_under_bit_budget` symbolic family. the claim is conditional:

```text
a learned codec is useful only if it stores fewer task-committed bits than verbatim storage while preserving the operations needed for state, action, and joint success.
```

this is not a claim that the project has learned compression. it is the proof package that must exist before a tiny trainable mirror is implemented.

## source decision

the source decision is [[oracle_compression_frontier_split]]. `compression_under_bit_budget` is ranked first because:

- it clears the current oracle threshold at 13.0x.
- it directly tests a compression claim rather than using compression as a secondary property.
- it already exposes verbatim, compressed, no-memory, recency-only, and shuffled-address controls.
- it can kill the idea cleanly if stored bits drop while task success drops.

## mathematical operation

the candidate learned operation is:

```text
c_t = E_\theta(o_t, q_t, h_{t-1}, b_t)
m_t = W_\theta(c_t, m_{t-1})
y_t = D_\theta(q_t, m_t, h_t)
```

where:

- `o_t` is the observation stream.
- `q_t` is the task query.
- `h_t` is the current local hidden state available to the tiny mirror.
- `b_t` is the remaining bit budget.
- `c_t` is a compact code containing address, schema, residual, and provenance fields.
- `m_t` is the bounded memory state.
- `y_t` is the answer or action-relevant reconstruction.

the operation is accepted only if the learned code preserves the task operation. a smaller code that loses the answer is not compression.

## non-oracle input rule

the learned mirror may not receive:

- hidden entity ids unless they are observable through the episode contract
- oracle schema labels
- oracle residual labels
- target answer fields at query time
- future observations
- family labels as a shortcut feature
- direct kill-condition labels

allowed inputs:

- observation stream
- query fields
- current time
- visible object attributes
- visible context tags
- memory-relevant positions only if represented as observable event markers
- explicit bit budget

## preserved operation

the codec must preserve:

- route to the stored task state
- reconstruct the action-relevant answer
- preserve joint success
- stay under the compressed budget
- keep provenance of the stored source
- resist no-memory, recency-only, and shuffled-address controls

the primary preserved operation is not pixel or tensor reconstruction. it is task-relative reconstruction under a bit budget.

## test material

the first tiny mirror should use the existing hard symbolic episode contract, narrowed to the `compression_under_bit_budget` family.

required episode fields:

- exact hidden state
- observation stream
- query
- target state or action
- memory-relevant positions
- distractor positions
- bit budget
- verbatim trace bits
- oracle latent-state bits
- oracle schema/residual bits
- expected no-memory, recency-only, shuffled-address, verbatim-store, and compressed-store behavior

the mirror must emit one record per episode with predicted answer or action, committed code fields, bits committed, reconstruction error, and telemetry.

## controls

required controls:

- oracle codec
- learned codec
- verbatim store
- no-memory
- recency-only
- shuffled-address
- random codebook
- matched-bit random code
- matched-parameter no-code baseline
- learned decoder with oracle code
- learned code with oracle decoder

the oracle-code and oracle-decoder splits localize whether failure is encoder-side, decoder-side, address-side, or optimization-side.

## metrics

top-line:

- `state_probe_accuracy`
- `action_success`
- `joint_success`
- bits committed per successful episode
- compression ratio versus verbatim storage
- rate-distortion frontier at fixed success

control gaps:

- learned codec minus no-memory
- learned codec minus recency-only
- learned codec minus shuffled-address
- learned codec minus random codebook
- learned codec versus verbatim at matched success

telemetry:

- address entropy
- address margin
- codebook usage entropy
- write frequency
- read concentration
- residual norm
- reconstruction error
- memory-output norm versus residual norm
- bit-budget usage
- confidence intervals

## pass condition

the learned codec passes the narrow proof package only if:

- oracle codec passes near-perfectly.
- learned codec beats no-memory, recency-only, shuffled-address, and random-code controls.
- learned codec preserves `joint_success` at or above the selected threshold.
- learned codec uses fewer committed bits than verbatim storage at matched success.
- oracle-code / learned-decoder and learned-code / oracle-decoder splits localize remaining errors.
- telemetry shows the compressed path is used rather than bypassed.
- confidence intervals rule out noise.

## kill condition

kill the learned-codec path for this family if any of these occur:

- learned bits drop but `joint_success` drops with them.
- learned codec improves reconstruction but not action or joint success.
- learned codec beats no-memory but not recency-only.
- shuffled-address succeeds.
- random codebook matches learned code.
- oracle-code / learned-decoder succeeds but learned-code / oracle-decoder fails, and encoder-side fixes do not improve it.
- learned-code / oracle-decoder succeeds but oracle-code / learned-decoder fails, and decoder-side fixes do not improve it.
- telemetry shows unused memory output, collapsed address entropy, or budget use that does not correlate with success.

## paper claim if proved

if the tiny mirror passes, the paper claim is narrow:

```text
on a controlled symbolic world, a learned task-relative codec can preserve state/action/joint success under a lower committed-bit budget than verbatim storage, with controls showing that the win is not recency, address shuffling, or random-code leakage.
```

this would still not prove general compression, biological novelty, full-model usefulness, or paid-scale viability. it would justify the next cpu implementation step.

## next implementation boundary

the next executable work is not the full model. it is the tiny local mirror defined in [[neural_model_tiny_mirror_contract_compression_under_bit_budget]], which restricts implementation to this one family and requires deterministic splits, forbidden-input guards, baselines, losses, telemetry schema, confidence intervals, failure-localization splits, and exact acceptance thresholds.

the tiny mirror must not touch:

- `god_machine.py`
- paid presets
- h200, kaggle, pod paths
- simulator selection
- full model integration

## see also

- [[PROJECT_PLAN]]
- [[tests/compression_under_bit_budget_mirror]]
- [[neural_model_paper_spine]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[oracle_compression_frontier_split]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_dossier_compression]]
- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[tests/hard_symbolic_nm_test_material]]
