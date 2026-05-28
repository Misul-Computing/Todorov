# local 100k learned unknown-structure density cell

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a registered hard-defeat result. no 600x high-density knowledge-compression breakthrough is authorized.

## artifact tested

- `neuroloc/simulations/memory/local_100k_learned_unknown_structure_density_cell.py`
- `tests/test_local_100k_learned_unknown_structure_density_cell.py`
- `neuroloc/simulations/suite_registry.py`
- hard artifact: `codex_local_output/suite_l100k_learned_unknown_structure_density_cell_hard/local_100k_learned_unknown_structure_density_cell/local_100k_learned_unknown_structure_density_cell_metrics.json`

## what was done

the simulation moves beyond the previous offset-key standard-codec probe by using source-heldout non-generated local corpus chunks with opaque associative keys. training chunks come from stable project knowledge and synthesis files. test chunks come from separate stable knowledge files.

the cell learns a byte-phrase dictionary from training chunks, encodes held-out test values through dictionary tokens plus residual bytes, stores records behind opaque keys, and decodes exact values plus provenance. the accounting charges dictionary bits, residual payload bits, residual index bits, associative assignment bits, key fingerprints, full query-key storage, decoder bits, manifest bits, and training supervision bits.

the result is deliberately not promoted because exact retrieval still requires per-fact residual/key records. that is a charged residual table, not a solved high-density unknown-structure neuron-cell.

## related-work pressure used

- titans and atlas require comparison against test-time neural memory rather than no-memory only: [titans](https://arxiv.org/abs/2501.00663), [atlas](https://arxiv.org/abs/2505.23735).
- miras frames the design space as associative memory architecture, attentional bias, retention gate, and learning algorithm: [miras](https://arxiv.org/abs/2504.13173).
- memory layers and product-key memory require a sparse key-value lookup baseline: [memory layers at scale](https://arxiv.org/abs/2412.09764), [large memory layers with product keys](https://arxiv.org/abs/1907.05242), [product-key memory repo](https://github.com/lucidrains/product-key-memory).
- hdc/vsa libraries make distributed associative-memory baselines cheap enough to track later: [torchhd paper](https://jmlr.org/papers/v24/23-0300.html), [torchhd repo](https://github.com/hyperdimensional-computing/torchhd).

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- source holdout used: `1.0`
- train/test key overlap: `0.0`
- trainable parameter count: `6`
- dictionary entries: `1024`
- useful retrievable bits: `1048576`
- committed state bits: `1674075`
- training supervision bits: `524288`
- strict accounted bits: `2198363`
- dictionary bits: `30920`
- residual payload bits: `691273`
- residual index bits: `65536`
- associative assignment bits: `43250.04688993525`
- key fingerprint bits: `262144`
- full query-key bits: `524288`
- decoder bits: `32768`
- manifest bits: `23896`
- strict density: `7.631352688405833`
- strict multiplier versus 2.5 bits per parameter: `3.0525410753623334`
- committed-only multiplier: `4.008483243348499`
- selected standard-codec multiplier on the same selected chunks: `5.029465628030584`
- prior charged corpus-codec baseline multiplier: `13.941917871967359`
- exact retrieval success: `1.0`
- heldout exact retrieval success: `1.0`
- random-label twin storage success: `1.0`
- random-label cross-label success: `0.0`
- random-label control collapse: `0.0`
- controls collapse: `1.0`
- beats charged codec baseline: `0.0`
- beats all reported baselines: `0.0`
- no per-fact committed rows: `0.0`
- learned cell pass: `0.0`
- strict 600x pass: `0.0`
- strict breakthrough authorized: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_learned_unknown_structure_density_cell.py -q` passed: 9 passed, 1 warning.
- `python -m pytest tests/test_local_100k_learned_unknown_structure_density_cell.py tests/test_simulation_suite.py::test_suite_registry_contract -q` passed: 9 passed, 1 warning.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_learned_unknown_structure_density_cell --profile smoke --timeout-sec 300` passed.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_learned_unknown_structure_density_cell --profile hard --output-root codex_local_output\suite_l100k_learned_unknown_structure_density_cell_hard --timeout-sec 1200` passed.
- `python -m py_compile neuroloc\simulations\memory\local_100k_learned_unknown_structure_density_cell.py tests\test_local_100k_learned_unknown_structure_density_cell.py neuroloc\simulations\suite_registry.py` passed.

## category check

implemented operation: exact source-heldout associative retrieval of non-generated local corpus chunks through a learned dictionary plus charged residual code.

strongest baseline: the prior charged standard corpus-codec probe at `13.941917871967359x`, plus random-label twin storage, no-memory, recency-only, shuffled-key/value/provenance, write/read/decoder/dictionary/residual-disabled controls, sparse-read, product-key, and verbatim table baselines.

what passed: exact retrieval is `1.0`, heldout retrieval is `1.0`, cross-label random-value scoring is `0.0`, source holdout is clean, key overlap is `0.0`, and disabled or shuffled controls collapse.

what failed: strict multiplier is only `3.0525410753623334x`, selected standard-codec multiplier is `5.029465628030584x`, and the prior charged corpus-codec baseline remains stronger at `13.941917871967359x`. a separately built random-label twin succeeds at `1.0`, proving the mechanism can store random labels by table when allowed. the exact path also depends on per-fact residual/key records, so `no_per_fact_committed_rows` is `0.0`.

what is not proved: high-density knowledge compression, a 600x neuron-cell, broad nm completion, arbitrary chat, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is useful: it kills the most obvious learned-dictionary residual path under honest accounting. learning phrase structure from local project text helps exact reconstruction, but once residual rows, assignment, keys, decoder, manifest, and training data are charged, it loses to the stronger codec baseline and remains table-shaped.

## verdict

accepted as a hard defeat and baseline-strengthening result. the next candidate must remove the per-fact residual table or prove an amortized non-row associative mechanism that beats the charged corpus-codec baseline under the same random-label and disabled-path controls.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[tests/local_100k_unstructured_density_cell]]
- [[tests/local_100k_high_density_cell]]
- [[tests/local_100k_schema_density_cell]]
- [[mistakes/learned_unknown_structure_residual_table_defeat]]
- [[mistakes/unstructured_exact_600x_entropy_wall]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/high_density_neuron_cell_related_work_pressure_matrix]]
