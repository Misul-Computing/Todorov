# local 100k source relation mph codec

status: current (as of 2026-05-14).

## summary

`local_100k_source_relation_mph_codec` is a narrow exact source-authored relation-index product. it does not reconstruct the source file. it stores a charged minimal-perfect-hash-style router, 17-bit fingerprints, compressed value/provenance id streams, and compressed value/provenance dictionaries inside torch module state.

the target relation surface is the repaired dense authored relation set: definition-parent, statement-enclosing-signature, and control-statement-enclosing-signature queries. the answer is not present in the query, generated aliases are not used, fixed stride rules are not used, and the full question table is not stored.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_relation_mph_codec.py`
- tests: `tests/test_local_100k_source_relation_mph_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_relation_mph_hard/local_100k_source_relation_mph_codec/local_100k_source_relation_mph_codec_metrics.json`

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_relation_mph_codec --profile hard --output-root codex_local_output\suite_l100k_source_relation_mph_hard --timeout-sec 600
```

result:

- suite result: pass.
- relation fact count: `3741`.
- definition-parent relation count: `314`.
- statement-enclosing relation count: `2808`.
- control-statement-enclosing relation count: `619`.
- selected relation accounted bits: `248784`.
- paq8px level 2 relation accounted bits: `261144`.
- margin over paq8px level 2 relation bits: `12360`.
- raw-source paq content-scan bits: `413888`.
- margin over raw-source paq content scan bits: `165104`.
- undercharged mph relation bits: `2771261`.
- honest mph relation index bits: `3366080`.
- margin over undercharged mph relation bits: `2522477`.
- margin over honest mph relation index bits: `3117296`.
- useful retrievable bits: `2639616`.
- strict multiplier: `67.90445687825584`.
- exact relation answer success: `1.0`.
- random-label twin success: `0.0`.
- random-label rebuild exact success: `1.0`.
- random-label rebuild selected relation accounted bits: `934984`.
- random-label rebuild density control collapse: `1.0`.
- decoder-disabled success: `0.0`.
- shuffled-fingerprint success: `0.0`.
- wrong-query variant count: `7482`.
- wrong-query hit rate: `0.0`.
- controls collapse: `1.0`.
- paq8px relation recomputed payload bits: `252952`.
- paq8px relation recomputed accounted bits: `261144`.
- paq8px relation recomputed archive bytes: `31619`.
- paq8px relation recomputed matches constant: `1.0`.
- state-dict reload success: `1.0`.
- state-dict exact reload answer success: `1.0`.
- stored question substring hit count: `0.0`.
- raw source block substring hit count: `0.0`.
- header raw bits within charged budget: `1.0`.
- model-state relation payload used: `1.0`.
- external payload store used: `0.0`.
- raw source block retained: `0.0`.
- full question table stored: `0.0`.
- paq8px baseline external constant used: `1.0`.
- paq8px baseline recomputed in run: `1.0`.
- self-contained paq8px baseline win authorized: `1.0`.
- source relation mph codec product authorized: `1.0`.
- source relation index product candidate: `1.0`.
- source relation static breakthrough candidate: `0.0`.
- broad breakthrough authorized: `0.0`.
- strict 600x authorized: `0.0`.
- broad knowledge authorized: `0.0`.
- full nm authorized: `0.0`.
- arbitrary chat authorized: `0.0`.

## controls

- cross-scored random-label twin must fail.
- separately rebuilt random-label codec must store exactly but cost more than the selected relation codec.
- decoder-disabled path must fail.
- shuffled-fingerprint path must fail.
- wrong query variants with injected prefix, injected suffix, and relation-token substitution must not hit.
- torch `state_dict` reload must preserve exact answers.
- header tensor raw bits must be no larger than the charged router header.
- serialized module state must not contain the full question table or whole raw source blocks.
- no raw source block, external payload store, generated alias labels, fixed-stride relation, formula labels, or full question table is allowed.

## category check

implemented operation:

exact keyed retrieval over a source-authored relation index.

strongest baselines:

paq8px level 2 on the relation surface, paq8px level 2 raw-source content scan, undercharged mph relation storage, and honest mph relation indexing.

what passed:

the hard profile answers all `3741` relation queries exactly from module state, recomputes the paq8px level 2 relation baseline in-run, beats that paq8px relation baseline by `12360` bits, beats raw-source paq content scan by `165104` bits, beats undercharged and honest mph diagnostics by large margins, reloads exact answers from `state_dict`, and collapses random-label, disabled, shuffled-fingerprint, and wrong-query controls.

what failed or remains limited:

this is a narrow keyed relation-index codec, not broad high-density knowledge compression. it is not arbitrary chat, not a full nm, not 600x, not paid-scale trainability, and not a general source-code byte compressor. the paq8px relation baseline is recomputed inside this simulation; the wider raw-source paq content-scan line remains imported from the paq8px audit card. the static breakthrough flag remains `0.0`; only the narrow source-relation index product candidate is authorized.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_relation_mph_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_relation_mph_codec --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_relation_mph_codec --profile hard --output-root codex_local_output\suite_l100k_source_relation_mph_hard --timeout-sec 600
python -m py_compile neuroloc\simulations\memory\local_100k_source_relation_mph_codec.py tests\test_local_100k_source_relation_mph_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests plus registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- python compile check: pass.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_dense_authored_relation_diagnostic]]
- [[tests/paq8px_public_context_mixing_baseline_audit]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
