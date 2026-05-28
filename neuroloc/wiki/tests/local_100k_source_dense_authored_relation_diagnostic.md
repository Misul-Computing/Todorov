# local 100k source-dense authored relation diagnostic

status: current (as of 2026-05-14).

## summary

`local_100k_source_dense_authored_relation_diagnostic` tests whether compressed source payload can amortize many exact source-authored relations better than an honest per-relation mph/index.

the diagnostic extracts only authored source relations from the reloaded source payload: definition parent links, statement-to-enclosing-signature links, and control-statement-to-enclosing-signature links. it does not use generated aliases, fixed stride rules, source ids, ordinal labels, hidden offsets, answer-contained signature queries, or schema-generated answers.

the result is useful but not promoted. it beats the honest relation mph/index by a large margin, but it loses to the paq8px public context-mixing baseline and the fair unlimited relation-aware scanner solves the same relation surface.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_dense_authored_relation_diagnostic.py`
- tests: `tests/test_local_100k_source_dense_authored_relation_diagnostic.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_dense_authored_relation_hard/local_100k_source_dense_authored_relation_diagnostic/local_100k_source_dense_authored_relation_diagnostic_metrics.json`

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_dense_authored_relation_diagnostic --profile hard --output-root codex_local_output\suite_l100k_source_dense_authored_relation_hard --timeout-sec 600
```

result:

- suite result: pass.
- relation fact count: `3741`.
- definition parent relation count: `314`.
- definition signature relation count: `0`.
- statement enclosing relation count: `2808`.
- control statement enclosing relation count: `619`.
- exact relation answer success: `1.0`.
- selected relation accounted bits: `437680`.
- honest mph relation index bits: `3366080`.
- paq8px level 2 relation accounted bits: `261144`.
- margin over honest mph relation index bits: `2928400`.
- margin over paq8px level 2 relation bits: `-176536`.
- useful retrievable bits: `2639616`.
- strict multiplier: `38.59793090842625`.
- random-label twin success: `0.0`.
- shuffled-value success: `0.0`.
- relation-decoder-disabled success: `0.0`.
- wrong-query hit rate: `0.0`.
- controls collapse: `1.0`.
- state-dict reload reconstruction success: `1.0`.
- state-dict raw source block retained: `0.0`.
- dense relation amortizes honest mph index: `1.0`.
- public context mixing not beaten: `1.0`.
- source dense authored relation product authorized: `0.0`.
- static relation breakthrough authorized: `0.0`.
- strict breakthrough authorized: `0.0`.

## controls

- answers must be produced after reloading the global-stream source payload from torch `state_dict`.
- target relations are extracted from authored source text only.
- generated aliases, fixed strides, formula labels, schema labels, and hidden offsets are forbidden.
- random-label twins, shuffled values, decoder-disabled queries, and wrong queries must fail.
- train and target source path/hash overlap must be zero.
- a fair unlimited relation-aware scanner must be reported.
- honest relation mph/index accounting must be reported.
- paq8px public context-mixing pressure must be reported and must block promotion when it is cheaper.

## category check

implemented operation:

dense bounded relation answering from source-authored definition and statement structure reconstructed from the charged global-stream source-code payload.

strongest baseline:

the paq8px v214 level 2 relation payload line at `261144` accounted bits, plus the fair unlimited relation-aware scanner at exact success `1.0`.

what passed:

the diagnostic answers `3741` source-authored relation queries exactly, beats the honest relation mph/index by `2928400` bits, preserves exact `state_dict` reconstruction, and keeps random-label/control gates passing.

what failed or remains limited:

the result loses to paq8px level 2 by `176536` bits and does not beat a fair unlimited relation-aware scanner. it is therefore a diagnostic, not a product, not learned semantic retrieval, not a static compression breakthrough, and not broad knowledge compression.

additional correction:

the first version counted a `definition_signature` relation whose question contained the exact answer. that subset was removed after [[mistakes/dense_relation_signature_query_leakage]].

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_dense_authored_relation_diagnostic.py -q
python -m pytest tests\test_local_100k_source_dense_authored_relation_diagnostic.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_dense_authored_relation_diagnostic --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_dense_authored_relation_diagnostic --profile hard --output-root codex_local_output\suite_l100k_source_dense_authored_relation_hard --timeout-sec 600
python -m py_compile neuroloc\simulations\memory\local_100k_source_dense_authored_relation_diagnostic.py tests\test_local_100k_source_dense_authored_relation_diagnostic.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests: `4 passed, 1 warning`.
- focused tests plus registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- python compile check: pass.

## decision

keep as a diagnostic and design warning. the dense relation surface proves source-payload amortization over honest per-relation indexing, but paq8px public context mixing blocks product promotion.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_authored_relation_diagnostic]]
- [[tests/paq8px_public_context_mixing_baseline_audit]]
- [[mistakes/public_context_mixing_baseline_missing]]
- [[mistakes/dense_relation_signature_query_leakage]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
