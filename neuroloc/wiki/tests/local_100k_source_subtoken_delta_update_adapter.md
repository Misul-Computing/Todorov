# local 100k source-subtoken delta-update adapter

status: current (as of 2026-05-14).

## summary

`local_100k_source_subtoken_delta_update_adapter` tests a narrow model-state update product: a source-subtoken qa adapter already carries a compressed base source payload, then a charged varint-delta-offset patch stream stores arbitrary replacement bytes for changed facts.

the product claim is delta-only. it is not a total static compression breakthrough. hard validation shows the patch stream can update exact held-out answers from model state while beating full recompression of the updated adapter and an undercharged mph update table. the matched delta-patch content-scan diagnostic is equal-cost and is not beaten. total base-plus-delta static bits still lose to full recompression on hard, so static compression authorization remains `0.0`.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_delta_update_adapter.py`
- tests: `tests/test_local_100k_source_subtoken_delta_update_adapter.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_delta_update_hard/local_100k_source_subtoken_delta_update_adapter/local_100k_source_subtoken_delta_update_adapter_metrics.json`

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_delta_update_adapter --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_delta_update_hard --timeout-sec 600
```

result:

- suite result: pass.
- fact count: `4096`.
- update fact count: `512`.
- unchanged fact count: `3584`.
- base payload bits: `244440`.
- delta patch bits: `138104`.
- total updated adapter bits: `382544`.
- full recompress updated bits: `380552`.
- same-block content scan update bits: `380552`.
- undercharged mph update bits: `188416`.
- matched delta-patch content scan bits: `138104`.
- margin over full recompress bits: `242448`.
- margin over same-block content scan update bits: `242448`.
- margin over undercharged mph update bits: `50312`.
- margin over matched delta-patch content scan bits: `0`.
- total static margin over full recompress bits: `-1992`.
- exact updated answer success: `1.0`.
- unchanged answer success: `1.0`.
- state-dict reload success: `1.0`.
- state-dict preload success: `0.0`.
- patch payload in state dict: `1.0`.
- patch header in state dict: `1.0`.
- random patch control success: `0.0`.
- patch disabled success: `0.0`.
- shuffled patch success: `0.0`.
- controls collapse: `1.0`.
- same-block content scan update beaten: `1.0`.
- undercharged mph update beaten: `1.0`.
- matched delta-patch content scan beaten: `0.0`.
- model state patch payload used: `1.0`.
- external payload store used: `0.0`.
- stored manifest used: `0.0`.
- per-fact value row count: `0.0`.
- assignment row count: `0.0`.
- raw source block retained: `0.0`.
- source subtoken delta update product authorized: `1.0`.
- source subtoken total static compression authorized: `0.0`.
- static compression breakthrough authorized: `0.0`.
- strict breakthrough authorized: `0.0`.
- broad breakthrough authorized: `0.0`.
- full nm authorized: `0.0`.
- broad chat authorized: `0.0`.

## controls

- the patch stream stores varint-delta offsets plus full replacement bytes, not a generated mask or value formula.
- updated answers and unchanged answers must both remain exact after patch application.
- state dict preload with a corrupted patch must fail before reload and pass after reload.
- random patch, patch-disabled, and shuffled-patch controls must fail.
- matched delta-patch content scan is reported and must not be treated as beaten.
- the base adapter payload and delta patch payload must be in torch module state.
- no external payload store, stored manifest, per-fact value row table, assignment row table, or raw source block is allowed.
- total static compression remains unauthorized unless base-plus-delta beats full updated recompression.

## category check

implemented operation:

delta update of an existing compressed model-state qa adapter.

strongest baseline:

full recompression of the updated source-subtoken adapter as the same-block content-scan update baseline, a matched delta-patch content scan, plus an undercharged mph update table.

what passed:

the hard profile updates `512` exact held-out answers with `138104` patch bits, beats full recompression by `242448` bits on the delta-update surface, beats undercharged mph update storage by `50312` bits, preserves unchanged answers, reloads from `state_dict`, and collapses random, disabled, and shuffled patch controls.

what failed or remains limited:

the matched delta-patch content-scan diagnostic is equal at `138104` bits, so no claim is made that the patch representation beats a same-interface delta patch. base-plus-delta total static bits are `1992` bits worse than full recompression on hard. this is therefore a delta-update product, not a total static compression breakthrough, not broad high-density knowledge compression, not arbitrary chat, and not a full nm.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_delta_update_adapter.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_delta_update_adapter --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_delta_update_adapter --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_delta_update_hard --timeout-sec 600
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_delta_update_adapter.py tests\test_local_100k_source_subtoken_delta_update_adapter.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests plus registry contract: `6 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- python compile check: pass.

## decision

promote only the narrow delta-update claim. this is a useful local model-state compression product for incremental updates, but it does not solve total static knowledge compression or the paq8px source-code byte-compression blocker.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_dense_authored_relation_diagnostic]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
