# local 100k source-token-structure block codec

status: historical context only. frozen as of 2026-05-13. do not edit.

## summary

`local_100k_source_token_structure_block_codec` supersedes the source-structure block codec as the current narrow local source-code byte-compression product.

it keeps the train-learned indentation split, changes the indentation-count plane to a signed-delta stream, and adds a fully charged target identifier dictionary for whole-word body substitution. the target dictionary is not free model state: its compressed payload and a `896` bit token-structure header are charged in the denominator.

this is a source-code byte-compression product, not high-density knowledge compression, semantic memory, arbitrary chat, full nm behavior, or strict public breakthrough authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_token_structure_block_codec.py`
- tests: `tests/test_local_100k_source_token_structure_block_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_token_structure_hard/local_100k_source_token_structure_block_codec/local_100k_source_token_structure_block_codec_metrics.json`

the hard profile uses the same four held-out target files and six disjoint train files as the prior source-structure block codec. the codec stores compressed count-delta payload, compressed token-substituted body payload, compressed token dictionary payload, codec selectors, line count, token count, indent unit, and the charged header. the payloads and selectors are carried through a reloadable torch module state dict.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_token_structure_block_codec --profile hard --output-root codex_local_output\suite_l100k_source_token_structure_hard --timeout-sec 1200
```

result:

- suite result: pass.
- exact reconstruction success: `1.0`.
- model-state restore success: `1.0`.
- model-state reload success: `1.0`.
- compressed-stream read success: `1.0`.
- target file count: `4`.
- train file count: `6`.
- target block bytes: `99761`.
- target line count: `2333`.
- count stream bytes: `2333`.
- count-delta stream bytes: `2333`.
- body stream bytes: `77453`.
- substituted body stream bytes: `55308`.
- dictionary stream bytes: `1780`.
- charged token count: `120`.
- useful retrievable bits: `798088`.
- best standard payload bits: `128816`.
- prior source-structure payload bits: `124200`.
- learned count-delta payload bits: `4096`.
- learned body-token payload bits: `112864`.
- learned dictionary payload bits: `5232`.
- learned token-structure header bits: `896`.
- learned payload bits: `123088`.
- best standard strict bits: `161648`.
- prior source-structure strict bits: `157032`.
- learned strict bits: `155920`.
- payload improvement over best standard: `0.04446652589740405`.
- strict improvement over best standard: `0.03543501930119766`.
- paper improvement over best standard: `0.03455932039772179`.
- source-structure strict-improvement baseline: `0.028555874492724932`.
- strict-improvement delta over source-structure: `0.00687914480847273`.
- payload-improvement delta over source-structure: `0.00895330112721417`.
- adapter strict multiplier: `32.75887121600821x`.
- best standard strict multiplier: `31.598059982183507x`.
- random-label payload improvement: `-0.0012328849507848366`.
- random-label payload incompressible: `1.0`.
- path overlap count: `0.0`.
- hash overlap count: `0.0`.
- sliding `64` byte ngram overlap count: `3104.0`.

## controls

- decoder-disabled exact reconstruction: `0.0`.
- wrong-indent-unit exact reconstruction: `0.0`.
- token-dictionary-disabled exact reconstruction: `0.0`.
- shuffled body payload exact reconstruction: `0.0`.
- shuffled count payload exact reconstruction: `0.0`.
- shuffled dictionary payload exact reconstruction: `0.0`.
- per-fact value row count: `0.0`.
- assignment row count: `0.0`.
- hidden fact value row detected: `0.0`.
- hidden raw source prefix detected: `0.0`.
- raw source block retained: `0.0`.
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

lossless held-out source-code byte compression with count-delta structure coding and charged whole-word identifier substitution.

strongest baseline:

best same-block standard codec sweep across zlib, bz2, lzma, brotli, and zstd. the hard best standard baseline is `128816` payload bits. the prior source-structure product is the internal baseline at `124200` payload bits.

what passed:

the new codec reconstructs exactly, reloads from torch module state, improves charged payload bits from `124200` to `123088`, improves strict charged bits from `157032` to `155920`, and keeps random-label payload gain negative.

what failed or remains limited:

this is not arbitrary opaque-key knowledge compression, learned semantic retrieval, model-weight implicit storage, high-density associative memory, 600x density, arbitrary chat, full nm behavior, or broad public compression authorization. the dictionary is target-adaptive and fully charged, which is honest for a codec but not evidence of learned knowledge stored in ordinary neural weights.

why this supersedes the source-structure product:

the previous source-structure product improved strict charged bits by `0.028555874492724932`. this token-structure product improves strict charged bits by `0.03543501930119766`, an additional `0.00687914480847273` absolute strict margin.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_token_structure_block_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_token_structure_block_codec --profile smoke --output-root codex_local_output\suite_l100k_source_token_structure_smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_token_structure_block_codec --profile hard --output-root codex_local_output\suite_l100k_source_token_structure_hard --timeout-sec 1200
python -m py_compile neuroloc\simulations\memory\local_100k_source_token_structure_block_codec.py tests\test_local_100k_source_token_structure_block_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests and registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- py compile: passed.

## decision

promote as the current narrow local source-code block codec product. do not promote as high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## correction (2026-05-13)

this run card is frozen evidence. later work superseded it with [[tests/local_100k_source_subtoken_structure_block_codec]], which lowers charged payload bits from `123088` to `120952` and raises strict improvement over the best same-block standard codec from `0.03543501930119766` to `0.048648916163515785`. the source-token-structure codec is no longer the current narrow source-code block codec product.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_source_structure_block_codec]]
- [[tests/local_100k_indent_token_block_codec]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
