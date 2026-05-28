# local 100k llm-facing semantic qa codec

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a tested bounded llm-facing exact qa codec product. it replaces the prior content-derived digest handle with a runtime natural-language question surface over token-signature evidence, keeps one charged compressed source block, and slightly beats the content-addressed source-codec strict multiplier.

this is not a chat model, not learned semantic memory, not arbitrary associative knowledge compression, and not a high-density breakthrough.

## artifact tested

- simulation: `local_100k_llm_semantic_qa_codec`
- hard output root: `codex_local_output/suite_l100k_llm_semantic_qa_hard`
- metrics artifact: `codex_local_output/suite_l100k_llm_semantic_qa_hard/local_100k_llm_semantic_qa_codec/local_100k_llm_semantic_qa_codec_metrics.json`

## what was done

the simulation builds bounded natural-language questions from source-heldout corpus anchors, but the stored test facts do not expose source ids, offsets, content digests, answer digests, or assignment keys. each runtime question is normalized into a token-signature handle. the read path decompresses the charged source block stream, scans candidate anchor windows, computes the same token-signature handle, and returns the following exact 32-byte answer plus provenance.

the committed state is the compressed source block, codec selector, decoder and fixed parser budget, and manifest. the runtime question is input, not stored state. there are no per-fact value rows, no assignment rows, no raw decoded source cache, and no stored question-to-answer table.

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- parameter count: `3`
- source block bytes: `173730`
- candidate scan count: `5418`
- useful retrievable bits: `1048576`
- block payload bits: `408256`
- semantic runtime handle bits: `32`
- semantic question handle bits charged: `0`
- content digest bits: `0`
- source-offset bits: `0`
- key assignment bits: `0`
- fixed parser bits: `65536`
- fixed parser charged through decoder bits: `1.0`
- committed state bits: `476938`
- strict accounted bits: `476938`
- exact answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- selected semantic collision count: `0.0`
- ambiguous match count: `0.0`
- unanswerable question success: `0.0`
- overlap distractor question success: `0.0`
- reads from compressed block: `1.0`
- raw source block retained: `0.0`
- strict multiplier: `14.06935717190861`
- content-addressed baseline multiplier: `14.06888524576417`
- beats content-addressed baseline: `1.0`
- mph payload baseline beaten: `0.0`
- strict 600x pass: `0.0`
- arbitrary chat authorization: `0.0`
- breakthrough authorization: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_llm_semantic_qa_codec.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_llm_semantic_qa_codec --profile smoke --timeout-sec 300`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_llm_semantic_qa_codec --profile hard --output-root codex_local_output\suite_l100k_llm_semantic_qa_hard --timeout-sec 1200`
- `python -m py_compile neuroloc\simulations\memory\local_100k_llm_semantic_qa_codec.py neuroloc\simulations\suite_registry.py tests\test_local_100k_llm_semantic_qa_codec.py`
- `git diff --check -- neuroloc\simulations\memory\local_100k_llm_semantic_qa_codec.py neuroloc\simulations\suite_registry.py tests\test_local_100k_llm_semantic_qa_codec.py`

## category check

implemented operation: exact bounded natural-language qa retrieval from one compressed source stream using a runtime token-signature question handle.

strongest baseline: the prior content-addressed source-codec product at strict multiplier `14.06888524576417`, plus the minimal-perfect-hash payload line at `14.070360120095941`.

what passed: exact answer success is `1.0`, random-label twin success is `0.0`, controls collapse is `1.0`, selected semantic collision count is `0.0`, source-offset routing is `0.0`, content-digest key target is `0.0`, assignment row count is `0.0`, per-fact value row count is `0.0`, raw source block retained is `0.0`, strict multiplier is `14.06935717190861`, and the product beats the current content-addressed baseline.

what failed or remains weaker: strict 600x pass remains `0.0`, breakthrough authorization remains `0.0`, learned semantic retrieval authorization remains `0.0`, and the minimal-perfect-hash payload baseline remains slightly stronger than this three-gate product.

what is not proved: high-density knowledge compression, learned semantic recall, arbitrary opaque-key associative retrieval, a chat model, a full neural model, a 600x neuron-cell, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is not promoted further: the handle is a lexical token signature over evidence terms, not learned semantic understanding. the next target must move from lexical token signatures toward learned paraphrase-stable or generative retrieval while beating the minimal-perfect-hash payload line and preserving the no-row, no-cache, no-source-offset controls.

## verdict

accepted as a presentable bounded llm-facing compression product. it gives a natural-language query surface over charged compressed state, improves slightly over the content-addressed product, and records the next hard baseline plainly: minimal-perfect-hash payload still wins by a small margin.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_content_addressed_source_codec]]
- [[tests/local_100k_source_block_codec]]
- [[tests/local_100k_shared_predictor_exact_codec]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
