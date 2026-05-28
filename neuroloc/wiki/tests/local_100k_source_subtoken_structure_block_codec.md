# local 100k source-subtoken-structure block codec

status: current (as of 2026-05-13).

## summary

`local_100k_source_subtoken_structure_block_codec` supersedes `local_100k_source_token_structure_block_codec` as the current narrow local source-code byte-compression product.

it keeps the train-learned indentation split and signed-delta count stream, but changes the body stream from whole-identifier substitution to reversible longest-match subtoken substitution. the target dictionary is fully charged through compressed dictionary payload bits plus a `896` bit subtoken-structure header.

this is a source-code byte-compression product, not high-density knowledge compression, learned semantic recall, arbitrary chat, full nm behavior, strict 600x proof, or broad public breakthrough authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_structure_block_codec.py`
- tests: `tests/test_local_100k_source_subtoken_structure_block_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_structure_hard/local_100k_source_subtoken_structure_block_codec/local_100k_source_subtoken_structure_block_codec_metrics.json`

the hard profile uses the same four held-out target files and six disjoint train files as the source-token-structure codec. payloads and selectors are carried through a reloadable torch module state dict.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_structure_block_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_structure_hard --timeout-sec 1200
```

result:

- suite result: pass.
- exact reconstruction success: `1.0`.
- model-state restore success: `1.0`.
- model-state reload success: `1.0`.
- target block bytes: `99761`.
- useful retrievable bits: `798088`.
- best standard payload bits: `128816`.
- prior source-token payload bits: `123088`.
- learned count-delta payload bits: `4096`.
- learned body-subtoken payload bits: `110728`.
- learned dictionary payload bits: `5232`.
- learned subtoken-structure header bits: `896`.
- learned payload bits: `120952`.
- best standard strict bits: `161648`.
- prior source-token strict bits: `155920`.
- learned strict bits: `153784`.
- payload improvement over best standard: `0.061048316979257236`.
- strict improvement over best standard: `0.048648916163515785`.
- strict-improvement delta over source-token: `0.013213896862318122`.
- payload-improvement delta over source-token: `0.01735343819056285`.
- adapter strict multiplier: `33.21387920719971x`.
- random-label payload improvement: `-0.0013331194996291321`.
- random-label payload incompressible: `1.0`.
- path overlap count: `0.0`.
- hash overlap count: `0.0`.

## controls

- decoder-disabled exact reconstruction: `0.0`.
- wrong-indent-unit exact reconstruction: `0.0`.
- token-dictionary-disabled exact reconstruction: `0.0`.
- shuffled body payload exact reconstruction: `0.0`.
- shuffled count payload exact reconstruction: `0.0`.
- shuffled dictionary payload exact reconstruction: `0.0`.
- codec state has raw target block: `0.0`.
- codec state has uncompressed count stream: `0.0`.
- codec state has uncompressed body stream: `0.0`.
- codec state has restored block: `0.0`.
- compressed count payload retained: `1.0`.
- compressed body payload retained: `1.0`.
- compressed dictionary payload retained: `1.0`.
- state-dict count payload used: `1.0`.
- state-dict body payload used: `1.0`.
- state-dict dictionary payload used: `1.0`.
- state-dict header used: `1.0`.
- target-charged dictionary accounted: `1.0`.
- train-free dictionary bits: `0.0`.
- formula or schema labels present: `0.0`.
- seed oracle authorized: `0.0`.
- controls collapse: `1.0`.

## category check

implemented operation:

lossless held-out source-code byte compression with count-delta structure coding and charged longest-match identifier-subtoken substitution.

strongest baseline:

best same-block standard codec sweep across zlib, bz2, lzma, brotli, and zstd. the hard best standard baseline is `128816` payload bits. the prior source-token-structure product is the internal baseline at `123088` payload bits.

what passed:

the new codec reconstructs exactly, reloads from torch module state, improves charged payload bits from `123088` to `120952`, improves strict charged bits from `155920` to `153784`, and keeps random-label payload gain negative.

what failed or remains limited:

this is not arbitrary opaque-key knowledge compression, learned semantic retrieval, model-weight implicit storage, high-density associative memory, 600x density, arbitrary chat, full nm behavior, or broad public compression authorization. the dictionary is target-adaptive and fully charged, which is honest for a codec but not evidence of learned knowledge stored in ordinary neural weights.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_structure_block_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_structure_block_codec --profile smoke --output-root codex_local_output\suite_l100k_source_subtoken_structure_smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_structure_block_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_structure_hard --timeout-sec 1200
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_structure_block_codec.py tests\test_local_100k_source_subtoken_structure_block_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests and registry contract: `3 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- py compile: passed.

## decision

promote as the current narrow local source-code block codec product. do not promote as high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_token_structure_block_codec]]
- [[tests/local_100k_source_structure_block_codec]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
