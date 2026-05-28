# local 100k source-subtoken global-stream corpus codec

status: current (as of 2026-05-14).

## summary

`local_100k_source_subtoken_global_stream_corpus_codec` supersedes `local_100k_source_subtoken_shared_dictionary_corpus_codec` as the current broader local source-code corpus codec product.

it keeps the same frozen five-block source-code manifest and a charged shared identifier-subtoken dictionary, but removes per-block local dictionaries and compresses one global count stream plus one global body stream across the whole corpus. the first `120` dictionary tokens use one-byte codes; the remaining shared tokens use an escaped varint code. block reconstruction is framed by a charged compressed length stream and a conservative `2048` bit global header. the charged payload is also carried by a single torch module state surface and reloads from `state_dict`.

this is a source-code corpus compression product, not a static retrieval breakthrough, high-density knowledge compression, arbitrary chat, full nm behavior, strict 600x proof, or paid-compute authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_global_stream_corpus_codec.py`
- tests: `tests/test_local_100k_source_subtoken_global_stream_corpus_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_global_stream_hard/local_100k_source_subtoken_global_stream_corpus_codec/local_100k_source_subtoken_global_stream_corpus_codec_metrics.json`

the hard profile uses the same five frozen source-code blocks as the previous corpus products and checks block hashes before promotion.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_global_stream_corpus_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_global_stream_hard --timeout-sec 600
```

result:

- suite result: pass.
- block count: `5`.
- exact reconstruction success minimum: `1.0`.
- frozen manifest hash success minimum: `1.0`.
- aggregate standard payload bits: `849752`.
- global raw standard payload bits: `736504`.
- prior shared-dictionary corpus payload bits: `803400`.
- prior source-subtoken corpus payload bits: `812688`.
- aggregate selected payload bits: `699144`.
- aggregate payload improvement over standard: `0.1772375940274339`.
- global raw standard payload improvement: `0.05072613319140154`.
- margin over global raw standard baseline: `37360` bits.
- aggregate payload margin over prior shared-dictionary corpus: `104256` bits.
- aggregate payload margin over prior source-subtoken corpus: `113544` bits.
- aggregate improvement delta over prior shared-dictionary corpus: `0.12976848394324123`.
- shared token count: `256`.
- one-byte token count: `120`.
- local token count per block: `0`.
- shared dictionary payload bits: `10400`.
- global count payload bits: `21272`.
- global body payload bits: `665112`.
- global length payload bits: `312`.
- global header bits: `2048`.
- global raw standard header bits: `64`.
- model-state codec payload used: `1.0`.
- state-dict buffer payload used: `1.0`.
- model-state exact reconstruction success: `1.0`.
- state-dict reload reconstruction success: `1.0`.
- state-dict raw source block retained: `0.0`.
- margin over charged zstd trained-dictionary baseline: `283696` bits.
- margin over undercharged zstd trained-dictionary diagnostic: `250848` bits.
- random-label payload improvement: `-0.0003177134598372809`.
- random-label global raw payload improvement: `-0.0003089892190172897`.
- random-label payload incompressible: `1.0`.
- random-label global raw payload incompressible: `1.0`.
- controls collapse: `1.0`.

## controls

- all selected blocks reconstruct exactly from the shared dictionary, global count payload, global body payload, and global length payload.
- frozen block hashes must match the manifest.
- random-label byte twins must not receive compression gain against either per-block standard compression or global raw standard compression.
- wrong-indent, shared-dictionary-disabled, shuffled shared-dictionary, shuffled body-payload, shuffled count-payload, and shuffled length-payload controls collapse.
- reload from module `state_dict` must reconstruct exactly from the charged buffers.
- raw source blocks are not retained.
- formula or schema labels are not present.
- seed oracle authorization remains `0.0`.
- broad nm, chat, knowledge, full nm, external simulator, paid-compute, static breakthrough, and strict breakthrough authorization stay `0.0`.

## category check

implemented operation:

lossless source-code corpus byte compression over a fixed local manifest using source-structure splitting, one charged shared identifier-subtoken dictionary, a global compressed count stream, a global compressed body stream, a charged compressed framing stream, and a torch module state surface for the charged payload.

strongest baseline:

the previous shared-dictionary corpus product at `803400` bits, the previous source-subtoken corpus product at `812688` bits, the best same-block standard codec sweep at `849752` bits, the stronger global raw standard baseline at `736504` bits with only a `64` bit header, and the zstd trained-dictionary audit lines at `982840` charged bits and `949992` undercharged payload-only bits.

what passed:

the global-stream corpus codec reconstructs all five blocks exactly, lowers aggregate selected payload bits from `803400` to `699144`, improves aggregate payload over the best same-block standard sweep by `0.1772375940274339`, beats the stronger global raw standard baseline by `37360` bits, beats the charged public zstd dictionary line by `283696` bits, beats the undercharged public dictionary diagnostic by `250848` bits, reloads from module `state_dict`, and keeps random-label payload gain negative.

what failed or remains limited:

the result is still a source-code codec. it does not prove arbitrary opaque-key knowledge compression, learned semantic recall, base-weight implicit storage, broad model compression, strict 600x neuron density, arbitrary chat, or full nm behavior. it also does not solve the bounded qa adapter's fair same-subtoken content-scan blocker.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_global_stream_corpus_codec.py -q
python -m pytest tests\test_local_100k_source_subtoken_global_stream_corpus_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_global_stream_corpus_codec --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_global_stream_corpus_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_global_stream_hard --timeout-sec 600
python -m pytest tests\test_local_100k_source_subtoken_global_stream_corpus_codec.py tests\test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_local_100k_source_subtoken_structure_corpus_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_global_stream_corpus_codec.py tests\test_local_100k_source_subtoken_global_stream_corpus_codec.py neuroloc\simulations\suite_registry.py neuroloc\simulations\suite_runner.py
git diff --check
```

results:

- focused tests: `5 passed, 1 warning`.
- focused tests plus registry contract: `6 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- broader regression bundle: `18 passed, 1 warning`.
- python compile check: pass.
- fixed-string scans for `todo`, `#`, and triple-quote comment/docstring markers in the new simulation and focused test: no matches.
- diff whitespace check: pass, with git lf/crlf warnings only.

## decision

promote as the current broader local source-code corpus codec product. do not promote as high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
