# local 100k source-block codec

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a tested source/offset exact-codec product. it beats the prior charged corpus-codec strict multiplier on this source-block target. no high-density knowledge-compression breakthrough is authorized.

## artifact tested

- simulation: `local_100k_source_block_codec`
- hard output root: `codex_local_output/suite_l100k_source_block_codec_hard`
- metrics artifact: `codex_local_output/suite_l100k_source_block_codec_hard/local_100k_source_block_codec/local_100k_source_block_codec_metrics.json`

## what was done

the simulation compresses the source-heldout test corpus as one source block and answers exact chunk queries by source id and offset. it is stronger than the shared-predictor opaque-key codec for the source/offset target because it does not charge a stored associative assignment row for every queried fact.

this is not arbitrary associative knowledge memory. the query itself supplies source and offset. the result is a real compression product for exact source-block retrieval, not a proof that unknown opaque keys can retrieve arbitrary values at high density.

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- parameter count: `3`
- source block bytes: `173730`
- useful retrievable bits: `1048576`
- block payload bits: `408256`
- source-offset routing bits: `20`
- committed state bits: `476958`
- strict accounted bits: `476958`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- reads from compressed block: `1.0`
- raw source block retained: `0.0`
- raw source block bits charged: `0.0`
- strict multiplier: `14.06876726917481`
- beats charged codec baseline: `1.0`
- product pass: `1.0`
- strict 600x pass: `0.0`
- breakthrough authorization: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_source_block_codec.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- `python -m pytest tests/test_simulation_suite.py::test_suite_registry_contract -q`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_source_block_codec --profile smoke --timeout-sec 300`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_source_block_codec --profile hard --output-root codex_local_output\suite_l100k_source_block_codec_hard --timeout-sec 1200`
- `python -m py_compile neuroloc\simulations\memory\local_100k_source_block_codec.py tests\test_local_100k_source_block_codec.py neuroloc\simulations\suite_registry.py`

## category check

implemented operation: exact source-block retrieval from one compressed source stream using source id and offset query fields.

strongest baseline: the prior charged corpus-codec baseline at strict multiplier `13.941917871967359`.

what passed: exact retrieval is `1.0`, random-label twin success is `0.0`, controls collapse is `1.0`, reads from compressed block is `1.0`, raw source block retained is `0.0`, strict multiplier is `14.06876726917481`, and the product beats the prior charged corpus-codec baseline.

what failed: strict 600x pass remains `0.0`, breakthrough authorization remains `0.0`, and the artifact is not an opaque-key associative memory.

what is not proved: high-density knowledge compression, arbitrary unknown-key associative retrieval, a 600x neuron-cell, broad neural-model completion, arbitrary chat, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is not promoted further: the compression is real for source/offset retrieval, but source and offset are a privileged query surface. the next target must preserve the density win while moving toward opaque-key or learned generative retrieval without reintroducing stored assignment rows.

## verdict

accepted as the first local compression product in this line that beats the prior charged corpus-codec strict multiplier on a bounded exact-retrieval target. it becomes the new source/offset product baseline, not the final high-density knowledge-compression solution.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_shared_predictor_exact_codec]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[tests/local_100k_learned_unknown_structure_density_cell]]
- [[mistakes/source_block_codec_raw_cache_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
