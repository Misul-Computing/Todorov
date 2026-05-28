# local 100k source-subtoken qa adapter

status: current (as of 2026-05-13).

## summary

`local_100k_source_subtoken_qa_adapter` supersedes `local_100k_source_structure_qa_adapter` as the strongest bounded exact qa adapter product.

it carries a source-subtoken compressed payload inside torch module state, answers bounded source-heldout questions exactly, passes transformer and recurrent host probes, preserves trained recompression update, rejects false-hit controls, and improves paper-surface strict multiplier from `23.688602733536655x` to `23.847532408460314x`.

this is a bounded exact qa adapter and update product, not learned semantic recall, arbitrary chat, full nm behavior, implicit base-weight storage, strict 600x proof, or broad public breakthrough authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_qa_adapter.py`
- tests: `tests/test_local_100k_source_subtoken_qa_adapter.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_qa_hard/local_100k_source_subtoken_qa_adapter/local_100k_source_subtoken_qa_adapter_metrics.json`

the adapter uses the same bounded qa surface as the source-structure qa adapter but replaces the count/body payload with count-delta, body-subtoken, and dictionary streams from the source-subtoken codec.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_qa_adapter --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_qa_hard --timeout-sec 1200
```

result:

- suite result: pass.
- exact answer success: `1.0`.
- heldout exact answer success: `1.0`.
- paraphrase-stable answer success: `1.0`.
- controls collapse: `1.0`.
- transformer surface pass: `1.0`.
- recurrent surface pass: `1.0`.
- trainable recompression update success: `1.0`.
- block payload bits: `244440`.
- committed state bits: `277248`.
- paper-surface accounted bits: `281344`.
- useful retrievable bits: `1048576`.
- adapter strict multiplier: `24.19976921301639x`.
- paper-surface strict multiplier: `23.847532408460314x`.
- raw content scan beaten: `1.0`.
- raw undercharged mph beaten: `1.0`.
- same-subtoken content scan multiplier: `24.20535549399815x`.
- same-subtoken content scan beaten: `0.0`.

## controls

- random-label twin success: `0.0`.
- no-memory success: `0.0`.
- read-disabled success: `0.0`.
- decoder-disabled success: `0.0`.
- parser-disabled success: `0.0`.
- adapter-disabled success: `0.0`.
- code-disabled success: `0.0`.
- update-controller-disabled success: `0.0`.
- wrong, unanswerable, partial-overlap, and marker-injection false-hit controls: `0.0`.
- external payload store used: `0.0`.
- stored manifest used: `0.0`.
- per-fact value row count: `0.0`.
- assignment row count: `0.0`.
- raw source block retained: `0.0`.
- hidden fact value row detected: `0.0`.
- hidden raw source prefix detected: `0.0`.
- formula or schema labels present: `0.0`.
- seed oracle authorized: `0.0`.

## category check

implemented operation:

bounded exact qa from a compressed source-code adapter payload, with model-state packaging, paraphrase-stable query surface, host integration, and trained decode-edit-recompress update.

strongest baseline:

raw content scan, raw undercharged mph, and fair same-subtoken content scan with the same payload and decoder budget.

what passed:

the adapter beats raw content scan and raw undercharged mph diagnostics, improves the previous source-structure qa paper-surface multiplier from `23.688602733536655x` to `23.847532408460314x`, and preserves the update and host probes.

what failed or remains limited:

the same-subtoken content scan remains slightly stronger at `24.20535549399815x`. static breakthrough authorization remains `0.0`. the result is still a bounded exact adapter, not learned semantic recall, not arbitrary chat, and not a broad compression breakthrough.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_qa_adapter.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_qa_adapter --profile smoke --output-root codex_local_output\suite_l100k_source_subtoken_qa_smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_qa_adapter --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_qa_hard --timeout-sec 1200
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_qa_adapter.py tests\test_local_100k_source_subtoken_qa_adapter.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests and registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- py compile: passed.

## decision

promote as the current strongest bounded exact qa adapter product. do not promote as high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_structure_qa_adapter]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
