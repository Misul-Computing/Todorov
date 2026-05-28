# local 100k source-structure block codec

status: current

## claim

`local_100k_source_structure_block_codec` is the current narrow local source-code block codec product. it learns a four-space indentation unit from disjoint train files, separates held-out source bytes into an indentation-count plane and a body plane, compresses both planes with a fair same-block standard-codec sweep, and restores the original bytes exactly from the charged compressed streams.

this is not a high-density knowledge-compression result, not semantic memory, not arbitrary chat, not a full nm, and not a static qa retrieval wrapper. the authorized positive claim is only:

`source_block_codec_product_authorized = 1.0` for the local held-out source-code byte-compression surface.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_structure_block_codec.py`
- tests: `tests/test_local_100k_source_structure_block_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_structure_hard/local_100k_source_structure_block_codec/local_100k_source_structure_block_codec_metrics.json`

the hard profile uses four held-out target files and six disjoint train files. the learned structural rule is a four-space indentation unit. the codec stores only the compressed count payload, compressed body payload, line count, codec selectors, indent unit, and a charged `256` bit structure header. the compressed payloads and codec selectors are also carried through a reloadable torch module state dict.

## hard metrics

- exact reconstruction success: `1.0`
- model-state restore success: `1.0`
- model-state reload success: `1.0`
- model-state payload used: `1.0`
- external payload store used: `0.0`
- compressed-stream read success: `1.0`
- target file count: `4`
- train file count: `6`
- target block bytes: `99761`
- target line count: `2333`
- count stream bytes: `2333`
- body stream bytes: `77453`
- useful retrievable bits: `798088`
- best standard payload bits: `128816`
- learned count payload bits: `4416`
- learned body payload bits: `119528`
- learned structure header bits: `256`
- learned payload bits: `124200`
- best standard strict bits: `161648`
- learned strict bits: `157032`
- payload improvement over best standard: `0.03583405788100857`
- strict improvement over best standard: `0.028555874492724932`
- paper-surface improvement over best standard: `0.027850178588666858`
- indent-token strict-improvement baseline: `0.020192022171632188`
- strict-improvement delta over indent-token product: `0.008363852321092744`
- adapter strict multiplier: `32.52689388150186x`
- best standard strict multiplier: `31.598059982183507x`
- random-label payload improvement: `-0.0005412665637591965`
- random-label payload incompressible: `1.0`
- path overlap count: `0.0`
- hash overlap count: `0.0`
- sliding `64` byte ngram overlap count: `3104.0`

## controls

- decoder-disabled exact reconstruction: `0.0`
- wrong-indent-unit exact reconstruction: `0.0`
- shuffled body payload exact reconstruction: `0.0`
- shuffled count payload exact reconstruction: `0.0`
- per-fact value row count: `0.0`
- assignment row count: `0.0`
- hidden fact value row detected: `0.0`
- hidden raw source prefix detected: `0.0`
- raw source block retained: `0.0`
- codec state has raw target block: `0.0`
- codec state has uncompressed count stream: `0.0`
- codec state has uncompressed body stream: `0.0`
- codec state has restored block: `0.0`
- compressed count payload retained: `1.0`
- compressed body payload retained: `1.0`
- state-dict count payload used: `1.0`
- state-dict body payload used: `1.0`
- state-dict codec selectors used: `1.0`
- formula or schema labels present: `0.0`
- seed oracle authorized: `0.0`
- controls collapse: `1.0`

## category check

implemented operation:

lossless held-out source-code byte compression with an explicitly charged source-structure transform and exact byte restoration from compressed count/body streams.

strongest baseline:

best same-block standard codec sweep across zlib, bz2, lzma, brotli, and zstd. the hard best standard baseline is the brotli family at `128816` payload bits. the learned structure split reaches `124200` payload bits after header charges.

what failed or remains limited:

this does not prove arbitrary opaque-key knowledge compression, learned semantic retrieval, model-weight implicit storage, high-density associative memory, 600x density, arbitrary chat, full nm behavior, or a broad public compression breakthrough. sliding `64` byte ngram overlap is not zero because source files share ordinary code fragments; only path and whole-file hash holdout are zero.

why this supersedes the indent-token product:

the earlier indent-token product improved strict charged bits by `0.020192022171632188`. this source-structure split improves strict charged bits by `0.028555874492724932`, an additional `0.008363852321092744` absolute strict margin, while keeping the same broad category limits.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_structure_block_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_structure_block_codec --profile smoke --output-root codex_local_output\suite_l100k_source_structure_smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_structure_block_codec --profile hard --output-root codex_local_output\suite_l100k_source_structure_hard --timeout-sec 1200
python -m py_compile neuroloc\simulations\memory\local_100k_source_structure_block_codec.py tests\test_local_100k_source_structure_block_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests and registry contract: `7 passed, 1 warning`
- smoke suite: `1/1 passed`
- hard suite: `1/1 passed`
- py compile: passed

## decision

promote as the current narrow local source-code block codec product. do not promote as high-density knowledge compression, general neural compression, chat, full nm, or broad breakthrough.
