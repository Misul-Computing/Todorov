# learned unknown-structure residual table defeat

status: current (as of 2026-05-12).

## summary

the learned unknown-structure density cell failed the high-density knowledge-compression target because exact retrieval depended on a charged per-fact residual/key table and lost to the prior charged corpus-codec baseline.

## what happened

`local_100k_learned_unknown_structure_density_cell` used non-generated source-heldout corpus chunks and opaque associative keys. it learned a byte-phrase dictionary from training chunks, then encoded held-out test chunks as dictionary tokens plus residual bytes. hard validation retrieved `4096` held-out facts exactly. cross-label random-value scoring was `0.0`, but a separately built random-label twin also retrieved at `1.0`, proving the mechanism can store random labels by table when allowed.

the exact path was still table-shaped. the cell stored a residual token stream behind each key, plus assignment, key-fingerprint, full query-key, decoder, manifest, dictionary, and training-supervision costs. hard strict multiplier reached only `3.0525410753623334x`, while the selected standard-codec comparison reached `5.029465628030584x` and the prior charged corpus-codec baseline stayed at `13.941917871967359x`.

## why it matters

this is not a breakthrough. it is a clean negative result. exact associative retrieval over unknown-structure text is possible, but the obvious learned dictionary plus residual-record method does not create a high-density neuron-cell. the hidden danger was treating learned dictionary use as compression while ignoring the residual table that actually carries the held-out entropy.

## prevention

future high-density unknown-structure attempts must fail closed unless all of the following are true:

- no per-fact residual, value, provenance, or hidden committed rows are required for exact retrieval.
- every discovered structure, training example, dictionary, key assignment, decoder, manifest, and side channel is charged.
- the method beats the charged corpus-codec baseline, sparse-read, product-key, and verbatim baselines at equal exact-retrieval success.
- random-label twin storage must fail for a promoted compressor; if it succeeds by building another residual table, the result is a defeat.
- shuffled-key/value/provenance, write/read/decoder-disabled, and wrong-code controls collapse.
- params-only density is not used as a substitute for strict params-plus-state density.

## see also

- [[tests/local_100k_learned_unknown_structure_density_cell]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[tests/local_100k_unstructured_density_cell]]
- [[tests/local_100k_high_density_cell]]
- [[mistakes/unstructured_exact_600x_entropy_wall]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[PROJECT_PLAN]]
