# local 100k source-subtoken shared-dictionary corpus codec

status: superseded by [[tests/local_100k_source_subtoken_global_stream_corpus_codec]] (as of 2026-05-13).

## summary

`local_100k_source_subtoken_shared_dictionary_corpus_codec` superseded `local_100k_source_subtoken_structure_corpus_codec` and is now superseded by `local_100k_source_subtoken_global_stream_corpus_codec` as the current broader local source-code corpus codec product.

it keeps count-delta structure coding, but changes the corpus body dictionary from fully separate per-block target dictionaries to one charged corpus-shared identifier-subtoken dictionary plus small charged per-block local dictionaries. the shared dictionary carries `112` tokens, each block carries up to `16` local tokens, and all dictionary payloads, headers, and selectors are charged.

this is a source-code corpus compression product, not a static retrieval breakthrough, high-density knowledge compression, arbitrary chat, full nm behavior, strict 600x proof, or paid-compute authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_shared_dictionary_corpus_codec.py`
- tests: `tests/test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_shared_dictionary_hard/local_100k_source_subtoken_shared_dictionary_corpus_codec/local_100k_source_subtoken_shared_dictionary_corpus_codec_metrics.json`

the hard profile uses the same five frozen source-code blocks as the previous corpus product and checks block hashes before promotion.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_shared_dictionary_corpus_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_shared_dictionary_hard --timeout-sec 600
```

result:

- suite result: pass.
- block count: `5`.
- exact reconstruction success minimum: `1.0`.
- frozen manifest hash success minimum: `1.0`.
- aggregate standard payload bits: `849752`.
- prior source-subtoken corpus payload bits: `812688`.
- aggregate selected payload bits: `803400`.
- aggregate payload improvement over standard: `0.05454767979363391`.
- aggregate payload margin over prior: `9288` bits.
- aggregate improvement delta over prior: `0.011428740180733565`.
- shared token count: `112`.
- local token count per block: `16`.
- shared dictionary payload bits: `4360`.
- shared header bits: `896`.
- local header bits per block: `16`.
- selector bits per block: `16`.
- margin over charged zstd trained-dictionary baseline: `179440` bits.
- margin over undercharged zstd trained-dictionary diagnostic: `146592` bits.
- random-label payload improvement: `-0.00020807116781500356`.
- random-label payload incompressible: `1.0`.
- controls collapse: `1.0`.

## controls

- all selected blocks reconstruct exactly from count, body, shared-dictionary, and local-dictionary payloads.
- frozen block hashes must match the manifest.
- random-label byte twins must not receive compression gain.
- wrong-indent, shared-dictionary-disabled, shuffled shared-dictionary, and shuffled body-payload controls collapse.
- raw source blocks are not retained.
- formula or schema labels are not present.
- seed oracle authorization remains `0.0`.
- broad nm, chat, knowledge, full nm, external simulator, paid-compute, static breakthrough, and strict breakthrough authorization stay `0.0`.

## category check

implemented operation:

lossless source-code corpus byte compression over a fixed local manifest using source-structure splitting, a charged corpus-shared identifier-subtoken dictionary, and charged per-block local dictionaries.

strongest baseline:

the previous source-subtoken corpus product at `812688` bits, the best same-block standard codec sweep at `849752` bits, and the zstd trained-dictionary audit lines at `982840` charged bits and `949992` undercharged payload-only bits.

what passed:

the shared-dictionary corpus codec reconstructs all five blocks exactly, lowers aggregate selected payload bits from `812688` to `803400`, improves aggregate payload over the best same-block standard sweep by `0.05454767979363391`, beats the charged public zstd dictionary line by `179440` bits, beats the undercharged public dictionary diagnostic by `146592` bits, and keeps random-label payload gain negative.

what failed or remains limited:

the result is still a source-code codec. it does not prove arbitrary opaque-key knowledge compression, learned semantic recall, base-weight implicit storage, broad model compression, strict 600x neuron density, arbitrary chat, or full nm behavior. it also does not solve the bounded qa adapter's fair same-subtoken content-scan blocker.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py -q
python -m pytest tests\test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python -m pytest tests\test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_local_100k_source_subtoken_structure_corpus_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_shared_dictionary_corpus_codec --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_shared_dictionary_corpus_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_shared_dictionary_hard --timeout-sec 600
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_shared_dictionary_corpus_codec.py tests\test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py neuroloc\simulations\memory\local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py neuroloc\simulations\suite_registry.py
git diff --check
```

results:

- focused tests: `4 passed, 1 warning`.
- focused tests plus registry contract: `5 passed, 1 warning`.
- focused corpus and public-baseline tests plus registry contract: `13 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- py compile: passed.
- diff check: passed with line-ending warnings only.

## decision

keep as a superseded source-code corpus codec baseline. do not promote as high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
