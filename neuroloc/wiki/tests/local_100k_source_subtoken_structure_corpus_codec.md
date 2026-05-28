# local 100k source-subtoken-structure corpus codec

status: superseded by tests/local_100k_source_subtoken_shared_dictionary_corpus_codec. retained for evidence continuity.

## summary

`local_100k_source_subtoken_structure_corpus_codec` was the broader frozen-corpus validation of the source-subtoken-structure codec.

it tests five fixed source-code blocks, checks normalized block hashes, charges selector bits and standard-fallback codec headers, requires exact reconstruction, requires random-label payload incompressibility, and gates product authorization on aggregate improvement over the same-block standard codec sweep.

this is a superseded source-code corpus compression product, not a static retrieval breakthrough, high-density knowledge compression, arbitrary chat, full nm behavior, strict 600x proof, or paid-compute authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_structure_corpus_codec.py`
- tests: `tests/test_local_100k_source_subtoken_structure_corpus_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_structure_corpus_hard/local_100k_source_subtoken_structure_corpus_codec/local_100k_source_subtoken_structure_corpus_codec_metrics.json`

the frozen hard manifest contains five blocks: held-out hard source files, `src/` library files, non-`local_100k` memory simulations, selected existing `local_100k` simulations, and selected tests. each block is truncated at `250000` bytes and hash-checked before promotion.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_structure_corpus_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_structure_corpus_hard --timeout-sec 1200
```

result:

- suite result: pass.
- block count: `5`.
- exact reconstruction success minimum: `1.0`.
- frozen manifest hash success minimum: `1.0`.
- aggregate standard payload bits: `849752`.
- aggregate selected payload bits: `812688`.
- aggregate payload improvement: `0.043617431909545375`.
- subtoken-structure selected block count: `5`.
- standard fallback selected block count: `0`.
- selector bits per block: `16`.
- standard codec header bits per block: `16`.
- random-label payload control required: `1.0`.
- random-label payload incompressible minimum: `1.0`.
- random-label payload improvement max: `-0.0006119877602447951`.
- controls collapse: `1.0`.

## controls

- all selected blocks reconstruct exactly from selected compressed streams.
- frozen block hashes must match the manifest before product authorization.
- the random-label byte twin must be incompressible relative to the same-block standard codec.
- disabled decoder and shuffled count/body/dictionary payload controls collapse through the block codec control surface.
- no raw target block, uncompressed count stream, uncompressed body stream, or restored block is retained as charged codec state.
- selector bits are charged for every block.
- standard codec header bits are charged for any future standard fallback.
- broad nm, chat, knowledge, full nm, external simulator, paid-compute, static breakthrough, and strict breakthrough authorization stay `0.0`.

## category check

implemented operation:

lossless source-code corpus byte compression over a fixed local manifest using source-structure splitting plus charged longest-match identifier-subtoken substitution.

strongest baseline:

best same-block standard codec sweep across zlib, bz2, lzma, brotli, and zstd on every block. the aggregate hard standard payload is `849752` bits.

what passed:

the corpus codec reconstructs all five blocks exactly, reduces aggregate selected payload bits to `812688`, and improves aggregate payload over the best same-block standard sweep by `0.043617431909545375` while preserving random-label incompressibility and disabled-control collapse.

what failed or remains limited:

the result is still a source-code codec. it does not prove arbitrary opaque-key knowledge compression, learned semantic recall, base-weight implicit storage, broad model compression, strict 600x neuron density, arbitrary chat, or full nm behavior.

why this supersedes the token corpus diagnostic:

the older `local_100k_source_token_structure_corpus_codec` used live repository discovery and had a weak control gate, so it is demoted to a diagnostic. this corpus product freezes the manifest, checks hashes, uses the seed for block controls, charges fallback headers, and requires random-label/control collapse.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_structure_corpus_codec.py tests\test_local_100k_source_subtoken_structure_block_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_structure_corpus_codec --profile smoke --output-root codex_local_output\suite_l100k_source_subtoken_structure_corpus_smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_structure_corpus_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_structure_corpus_hard --timeout-sec 1200
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_structure_block_codec.py neuroloc\simulations\memory\local_100k_source_subtoken_structure_corpus_codec.py tests\test_local_100k_source_subtoken_structure_block_codec.py tests\test_local_100k_source_subtoken_structure_corpus_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests and registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- py compile: passed.

## decision

superseded by [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec]], which lowers aggregate selected payload bits from `812688` to `803400`. do not promote either result as high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_source_token_structure_block_codec]]
- [[tests/local_100k_source_structure_block_codec]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
