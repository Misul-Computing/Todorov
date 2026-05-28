# local 100k shared-predictor exact codec

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a tested exact-codec product. no high-density knowledge-compression breakthrough is authorized.

## artifact tested

- simulation: `local_100k_shared_predictor_exact_codec`
- hard output root: `codex_local_output/suite_l100k_shared_predictor_exact_codec_hard`

## what was done

the simulation tests the first shared-predictor exact-codec product after the learned unknown-structure residual-row defeat. it stores one block value stream instead of per-fact value slices. that removes the earlier per-fact residual-value table shape from the value payload.

the result is still not a high-density breakthrough. exact retrieval remains routed through charged key and provenance machinery, the strict accounting is larger than the useful payload, and the product loses to stronger charged codec and minimal-perfect-hash payload baselines. the random-label density control collapses, which is the intended guard against formula or hidden-table compression, but collapse alone is not enough to authorize the 600x claim.

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- parameter count: `5`
- exact retrieval success: `1.0`
- product pass: `1.0`
- controls collapse: `1.0`
- payload bits: `388864`
- random-label payload bits: `1048944`
- random-label density control collapse: `1.0`
- committed state bits: `1017438`
- strict accounted bits: `1541726`
- strict multiplier: `4.352614012398447`
- committed-only multiplier: `6.5953490749058`
- beats charged codec baseline: `0.0`
- beats mph payload baseline: `0.0`
- strict 600x pass: `0.0`
- breakthrough authorized: `0.0`

## verification commands

- `python neuroloc\simulations\suite_runner.py --simulation local_100k_shared_predictor_exact_codec --profile hard --output-root codex_local_output\suite_l100k_shared_predictor_exact_codec_hard --timeout-sec 1200`

## category check

implemented operation: exact retrieval through a shared-predictor exact codec over one block value stream, with charged key and provenance routing.

strongest baseline: the charged corpus-codec baseline and the minimal-perfect-hash payload baseline.

what passed: exact retrieval is `1.0`, product pass is `1.0`, controls collapse is `1.0`, and the random-label density control collapses at `1.0`. the value payload is no longer represented as per-fact value slices.

what failed: strict multiplier is only `4.352614012398447`, committed-only multiplier is `6.5953490749058`, beats charged codec baseline is `0.0`, beats mph payload baseline is `0.0`, strict 600x pass is `0.0`, and breakthrough authorization is `0.0`.

what is not proved: high-density knowledge compression, a 600x neuron-cell, broad neural-model completion, arbitrary chat, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is not promoted: the value stream is product-shaped rather than per-fact-value-row-shaped, but charged key and provenance routing remain. under strict accounting it is still weaker than the stronger baselines, so it is a tested exact-codec product and not a high-density breakthrough.

## concerns

- the result improves the artifact shape by removing per-fact value slices, but it does not remove all charged per-fact routing pressure.
- the strict multiplier remains far below the 600x target.
- the stronger charged codec and minimal-perfect-hash payload baselines still block promotion.
- the page records the hard metrics only; broader plan or status-file synchronization is intentionally out of scope for this documentation-only ownership slice.

## verdict

accepted as a hard exact-codec product result and negative breakthrough result. the next candidate must beat the charged codec and minimal-perfect-hash payload baselines under the same strict accounting before any high-density claim can be reconsidered.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_learned_unknown_structure_density_cell]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[mistakes/learned_unknown_structure_residual_table_defeat]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
