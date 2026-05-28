# local 100k indent-token block codec

status: current

## claim

`local_100k_indent_token_block_codec` is a narrow local lossless source-code block codec candidate. it learns a single indentation token from disjoint training source files, replaces the token in held-out target source blocks, compresses the transformed block with the best standard codec from a fair in-repo sweep, and exactly restores the original bytes from the charged compressed stream.

this is not a high-density knowledge-compression result, not semantic memory, not arbitrary chat, not a full nm, and not a static qa retrieval wrapper. the authorized positive claim is only:

`source_block_codec_product_authorized = 1.0` for the local held-out source-code byte-compression surface.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_indent_token_block_codec.py`
- tests: `tests/test_local_100k_indent_token_block_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_indent_token_hard/local_100k_indent_token_block_codec/local_100k_indent_token_block_codec_metrics.json`

the hard profile uses four held-out target files and six disjoint train files. the learned token has length `4` bytes and occurs `5578` times in the held-out target block. token accounting charges the token bytes plus a `144` bit token-map header, plus the shared decoder, model header, and paper-surface contract where applicable.

## hard metrics

- exact reconstruction success: `1.0`
- compressed-stream read success: `1.0`
- target file count: `4`
- train file count: `6`
- target block bytes: `99761`
- transformed block bytes: `83027`
- useful retrievable bits: `798088`
- best standard payload bits: `128816`
- learned payload bits: `125552`
- best standard strict bits: `161648`
- learned strict bits: `158384`
- payload improvement over best standard: `0.025338467271146442`
- strict improvement over best standard: `0.020192022171632188`
- paper-surface improvement over best standard: `0.019693020561830293`
- adapter strict multiplier: `32.249237296696634x`
- best standard strict multiplier: `31.598059982183507x`
- random-label payload improvement: `-0.008129021911272377`
- random-label payload incompressible: `1.0`
- path overlap count: `0.0`
- hash overlap count: `0.0`
- sliding `64` byte ngram overlap count: `3104.0`

## controls

- decoder-disabled exact reconstruction: `0.0`
- wrong-token exact reconstruction: `0.0`
- shuffled-payload exact reconstruction: `0.0`
- wrong-source-split exact reconstruction: `0.0`
- token-map-disabled strict improvement: `0.0`
- per-fact value row count: `0.0`
- assignment row count: `0.0`
- hidden fact value row detected: `0.0`
- hidden raw source prefix detected: `0.0`
- raw source block retained: `0.0`
- codec state has raw target block: `0.0`
- codec state has transformed block: `0.0`
- codec state has restored block: `0.0`
- formula or schema labels present: `0.0`
- seed oracle authorized: `0.0`
- controls collapse: `1.0`

## category check

implemented operation:

lossless held-out source-code byte compression with a train-learned indentation token and exact byte restoration from a charged compressed stream.

strongest baseline:

best same-block standard codec sweep across zlib, bz2, lzma, brotli, and zstd. the best hard baseline is the brotli family, and the learned token transform still improves strict charged bits by `2.0192022171632188%`.

what failed or remains limited:

the result does not prove arbitrary opaque-key knowledge compression, learned semantic retrieval, model-weight implicit storage, high-density associative memory, 600x density, arbitrary chat, full nm behavior, or a broad source-code breakthrough. the paper-surface denominator improvement is `1.9693020561830293%`, so the promoted gate is the strict codec denominator, not the wider paper-surface denominator. sliding `64` byte ngram overlap is not zero because source files share ordinary code fragments; only path and whole-file hash holdout are zero.

why this is not a retrieval-wrapper repeat:

the artifact reconstructs a held-out byte block from a compressed stream. it does not map questions to answer rows, store per-fact payload rows, use content signatures as answer handles, or depend on a same-interface content scan being weaker than the model. the static retrieval wrapper authorization remains `0.0`.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_indent_token_block_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_indent_token_block_codec --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_indent_token_block_codec --profile hard --output-root codex_local_output\suite_l100k_indent_token_hard --timeout-sec 1200
python -m py_compile neuroloc\simulations\memory\local_100k_indent_token_block_codec.py tests\test_local_100k_indent_token_block_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests and registry contract: `6 passed, 1 warning`
- smoke suite: `1/1 passed`
- hard suite: `1/1 passed`
- py compile: passed

## decision

promote as the current narrow local source-code block codec product. do not promote as high-density knowledge compression, general neural compression, chat, or full nm.
