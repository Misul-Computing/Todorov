# local 100k high density cell strict 600x not met

status: historical context only. frozen as of 2026-05-09. do not edit.

## summary

the first high-density neuron-cell proof cleared the params-only 600x density target but did not clear the same target under strict params-plus-committed-state accounting.

## what happened

the hard profile stored and retrieved `4096` exact associative facts with `8` trainable parameters and `114688` useful retrievable bits. this gives `14336.0` params-only useful bits per parameter, or `5734.4x` the ordinary `2.5` bits-per-parameter bar.

after charging `118816` committed state bits, strict density fell to `15.427495291902071` useful bits per parameter-equivalent, or `6.1709981167608285x` the ordinary bar. this is useful, but it is not 600x.

## why it matters

exact independent associative facts have entropy. moving that entropy from trainable weights into runtime state is not a compression breakthrough unless the runtime state is charged, amortized by a shared decoder, or reduced through a stronger generative or schema mechanism.

## prevention

future high-density claims must report both params-only and strict density. a strict 600x claim cannot pass unless committed state, schema, decoder, and model costs are included or explicitly amortized, and the candidate still beats sparse-read, product-key, bounded-recurrent-state, and verbatim baselines at equal success.

## see also

- [[tests/local_100k_high_density_cell]]
- [[synthesis/high_density_neuron_cell_related_work_pressure_matrix]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[PROJECT_PLAN]]
