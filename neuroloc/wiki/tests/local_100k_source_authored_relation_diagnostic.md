# local 100k source-authored relation diagnostic

status: current (as of 2026-05-14).

## summary

`local_100k_source_authored_relation_diagnostic` tests the safe version of the relation idea after the fixed-stride mistake.

it uses only source-authored relation text extracted from disjoint frozen source blocks after torch `state_dict` reload: unique python definition signatures and unique import statements. it does not generate aliases, source ids, offsets, fixed strides, ordinal labels, or schema labels.

the diagnostic passes exact relation answering and read-work telemetry, but it is not promoted as a product or breakthrough because a fair unlimited relation-aware scanner solves the same task and an honest relation mph/index baseline is cheaper in bits.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_authored_relation_diagnostic.py`
- tests: `tests/test_local_100k_source_authored_relation_diagnostic.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_authored_relation_hard/local_100k_source_authored_relation_diagnostic/local_100k_source_authored_relation_diagnostic_metrics.json`

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_authored_relation_diagnostic --profile hard --output-root codex_local_output\suite_l100k_source_authored_relation_hard --timeout-sec 600
```

result:

- suite result: pass.
- relation fact count: `337`.
- definition relation count: `303`.
- import relation count: `34`.
- exact relation answer success: `1.0`.
- state-dict reload reconstruction success: `1.0`.
- source train-test path overlap count: `0.0`.
- source train-test hash overlap count: `0.0`.
- selected relation accounted bits: `437680`.
- raw relation content-scan bits: `457272`.
- undercharged relation mph bits: `457288`.
- honest mph relation index bits: `305016`.
- margin over raw relation content scan bits: `19592`.
- margin over undercharged relation mph bits: `19608`.
- margin over honest mph relation index bits: `-132664`.
- relation-aware unlimited scanner success: `1.0`.
- relation-aware unlimited scanner not beaten: `1.0`.
- read-limited scanner success: `0.0`.
- read-work gain over unlimited scan: `337.0`.
- random-label twin success: `0.0`.
- shuffled-value success: `0.0`.
- relation-decoder-disabled success: `0.0`.
- wrong-query hit rate: `0.0`.
- generated alias labels present: `0.0`.
- fixed stride relation used: `0.0`.
- formula or schema labels present: `0.0`.
- work-bounded relation diagnostic candidate: `1.0`.
- source-authored relation product authorized: `0.0`.
- static relation breakthrough authorized: `0.0`.
- strict breakthrough authorized: `0.0`.

## controls

- answers must be produced after reloading the global-stream codec payload from torch `state_dict`.
- target relations are extracted from authored source text only.
- generated aliases, fixed strides, formula labels, schema labels, and hidden offsets are forbidden.
- random-label twins, shuffled values, decoder-disabled queries, and wrong queries must fail.
- train and target source path/hash overlap must be zero.
- a fair unlimited relation-aware scanner must be reported, not hidden.
- an honest relation mph/index baseline must be reported and must block product promotion when it is cheaper.

## category check

implemented operation:

bounded relation answering from source-authored definition and import lines reconstructed from the charged global-stream source-code payload.

strongest baseline:

the honest relation mph/index baseline at `305016` bits, plus the fair unlimited relation-aware scanner at exact success `1.0`.

what passed:

the diagnostic answers `337` source-authored relation queries exactly, uses no train-overlapping target source files, beats raw relation content scan and the undercharged mph diagnostic by the same global-stream margin, and shows a read-work advantage over a no-index unlimited scan.

what failed or remains limited:

the result loses to an honest relation mph/index by `132664` bits and does not beat a fair unlimited relation-aware scanner. it is therefore a diagnostic, not a product, not learned semantic retrieval, not a static compression breakthrough, and not broad knowledge compression.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_authored_relation_diagnostic.py -q
python -m pytest tests\test_local_100k_source_authored_relation_diagnostic.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_authored_relation_diagnostic --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_authored_relation_diagnostic --profile hard --output-root codex_local_output\suite_l100k_source_authored_relation_hard --timeout-sec 600
python -m pytest tests\test_local_100k_source_authored_relation_diagnostic.py tests\test_local_100k_source_subtoken_disjoint_retrieval_codec.py tests\test_local_100k_source_subtoken_global_stream_corpus_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python -m py_compile neuroloc\simulations\memory\local_100k_source_authored_relation_diagnostic.py tests\test_local_100k_source_authored_relation_diagnostic.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests: `4 passed, 1 warning`.
- focused tests plus registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- broader regression bundle: `14 passed, 1 warning`.
- python compile check: pass.

## decision

keep as a diagnostic and design warning. do not promote to source-authored relation product, learned semantic retrieval, static breakthrough, high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_disjoint_retrieval_codec]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[mistakes/source_native_relation_stride_rule_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
