# local 100k content-addressed source codec

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a tested content-addressed exact-codec product. it removes visible source/offset query fields and beats the previous source-block strict multiplier by a small margin. no high-density knowledge-compression breakthrough is authorized.

## artifact tested

- simulation: `local_100k_content_addressed_source_codec`
- hard output root: `codex_local_output/suite_l100k_content_addressed_source_codec_hard`
- metrics artifact: `codex_local_output/suite_l100k_content_addressed_source_codec_hard/local_100k_content_addressed_source_codec/local_100k_content_addressed_source_codec_metrics.json`

## what was done

the simulation compresses the source-heldout test corpus as one source block. each query uses a bounded opaque content digest handle derived from the target content window, not source id and offset fields. the read path decompresses the charged block stream, scans candidate windows, computes the same digest, and returns the matching chunk and provenance.

this is not arbitrary associative knowledge memory. the key is a privileged content handle derived from the answer window. the result is a real step away from source/offset routing and assignment rows, not proof that arbitrary unknown keys can retrieve arbitrary values.

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- parameter count: `3`
- source block bytes: `173730`
- candidate scan count: `5427`
- useful retrievable bits: `1048576`
- block payload bits: `408256`
- content digest bits: `16`
- source-offset bits: `0`
- key assignment bits: `0`
- committed state bits: `476954`
- strict accounted bits: `476954`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- selected digest collision count: `0.0`
- ambiguous match count: `0.0`
- reads from compressed block: `1.0`
- raw source block retained: `0.0`
- strict multiplier: `14.06888524576417`
- source-block baseline multiplier: `14.06876726917481`
- beats source-block baseline: `1.0`
- strict 600x pass: `0.0`
- breakthrough authorization: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_content_addressed_source_codec.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_content_addressed_source_codec --profile smoke --timeout-sec 300`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_content_addressed_source_codec --profile hard --output-root codex_local_output\suite_l100k_content_addressed_source_codec_hard --timeout-sec 1200`
- `python -m py_compile neuroloc\simulations\memory\local_100k_content_addressed_source_codec.py tests\test_local_100k_content_addressed_source_codec.py neuroloc\simulations\suite_registry.py`

## category check

implemented operation: exact content-addressed source-block retrieval from one compressed source stream using an opaque bounded digest handle.

strongest baseline: the previous source/offset product at strict multiplier `14.06876726917481`.

what passed: exact retrieval is `1.0`, random-label twin success is `0.0`, controls collapse is `1.0`, selected digest collision count is `0.0`, source-offset routing is `0.0`, key assignment bits are `0.0`, reads from compressed block is `1.0`, raw source block retained is `0.0`, strict multiplier is `14.06888524576417`, and the product beats the previous source-block baseline.

what failed: strict 600x pass remains `0.0`, breakthrough authorization remains `0.0`, and the artifact is not arbitrary opaque-key associative memory.

what is not proved: high-density knowledge compression, semantic recall, arbitrary unknown-key associative retrieval, a 600x neuron-cell, broad neural-model completion, arbitrary chat, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is not promoted further: the key is opaque at the surface but is derived from the target content window. the next target must move from content-derived handles toward learned semantic or generative retrieval without adding assignment rows, residual tables, or raw caches.

## verdict

accepted as the next local compression product. it preserves exact retrieval, removes source/offset query fields, avoids assignment rows, and slightly improves the strict multiplier over the previous source-block product. it becomes the new content-addressed product baseline, not the final high-density knowledge-compression solution.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_block_codec]]
- [[tests/local_100k_shared_predictor_exact_codec]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[mistakes/source_block_codec_raw_cache_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
