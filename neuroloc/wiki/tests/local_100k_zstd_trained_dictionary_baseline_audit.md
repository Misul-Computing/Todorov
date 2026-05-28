# local 100k zstd trained-dictionary baseline audit

status: current (as of 2026-05-13).

## summary

`local_100k_zstd_trained_dictionary_baseline_audit` adds a public trained-dictionary pressure test for the current source-subtoken source-code codecs.

the audit trains zstd dictionaries only from explicit train files after excluding target paths and target hashes, charges dictionary bytes plus header and selector bits, also reports an undercharged payload-only diagnostic line, and compares both against the current source-subtoken block and frozen-corpus payloads.

this is a baseline audit, not a new codec product, not a high-density knowledge result, not learned semantic recall, not arbitrary chat, not full nm behavior, not strict 600x proof, and not broad public breakthrough authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_zstd_trained_dictionary_baseline_audit.py`
- tests: `tests/test_local_100k_zstd_trained_dictionary_baseline_audit.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_zstd_dict_audit_hard/local_100k_zstd_trained_dictionary_baseline_audit/local_100k_zstd_trained_dictionary_baseline_audit_metrics.json`

the hard profile evaluates the four-file held-out source block and the five-block frozen source-code corpus. the baseline sweeps trained zstd dictionary sizes and compression levels, then charges the selected dictionary, selected payload, a `64` bit public-dictionary header, and a `16` bit selector.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_zstd_trained_dictionary_baseline_audit --profile hard --output-root codex_local_output\suite_zstd_dict_audit_hard --timeout-sec 600
```

result:

- suite result: pass.
- engineering pass: `1.0`.
- public trained dictionary baseline used: `1.0`.
- block train-only success: `1.0`.
- block public-dictionary exact reconstruction: `1.0`.
- block current source-subtoken payload bits: `120952`.
- block best generic standard payload bits: `128816`.
- block public-dictionary charged bits: `151376`.
- block public-dictionary undercharged bits: `147200`.
- block public-dictionary dictionary bits: `4096`.
- block public-dictionary payload bits: `147200`.
- block current payload margin over charged public dictionary: `30424` bits.
- block current payload margin over undercharged public dictionary: `26248` bits.
- block random-label charged improvement over best standard: `-0.005342501453400959`.
- block random-label undercharged improvement over best standard: `-0.00011025800372872521`.
- block path overlap count: `0.0`.
- block hash overlap count: `0.0`.
- corpus train-only success: `1.0`.
- corpus public-dictionary exact reconstruction: `1.0`.
- corpus current source-subtoken payload bits: `812688`.
- corpus best generic standard payload bits: `849752`.
- corpus public-dictionary charged bits: `982840`.
- corpus public-dictionary undercharged bits: `949992`.
- corpus public-dictionary dictionary bits: `32768`.
- corpus public-dictionary payload bits: `949992`.
- corpus current payload margin over charged public dictionary: `170152` bits.
- corpus current payload margin over undercharged public dictionary: `137304` bits.
- corpus random-label charged improvement over best standard: `-0.0007263801846475873`.
- corpus random-label undercharged improvement over best standard: `-0.00007600204333362405`.
- corpus path overlap count: `0.0`.
- corpus hash overlap count: `0.0`.
- controls collapse: `1.0`.

## controls

- block dictionary-disabled reconstruction: `0.0`.
- block shuffled-dictionary reconstruction: `0.0`.
- corpus dictionary-disabled reconstruction: `0.0`.
- corpus shuffled-dictionary reconstruction: `0.0`.
- block train/test path and hash overlap counts: `0.0`.
- corpus train/test path and hash overlap counts: `0.0`.
- charged dictionary bytes are included in the primary public baseline.
- the undercharged dictionary line is reported only as a diagnostic and is still beaten.
- random-label byte twins do not benefit from the trained dictionary baseline.
- source-code product, strict breakthrough, broad nm, chat, knowledge, full nm, external simulator, and paid-compute authorizations stay `0.0`.

## category check

implemented operation:

public trained-dictionary baseline audit for lossless source-code byte compression.

strongest baseline:

trained zstd dictionaries over train-only local source files, with dictionary size and level sweep. the audit reports both charged dictionary accounting and an undercharged payload-only diagnostic.

what passed:

the current source-subtoken block payload beats the charged public trained-dictionary baseline by `30424` bits and the undercharged payload-only line by `26248` bits. the current source-subtoken corpus payload beats the charged public trained-dictionary baseline by `170152` bits and the undercharged payload-only line by `137304` bits. exact reconstruction and random-label controls pass.

what failed or remains limited:

this audit strengthens the source-code codec claim, but it does not solve the broader goal. the source-subtoken qa adapter still does not beat the fair same-subtoken content scan. the result is not arbitrary unknown-structure knowledge compression, learned semantic recall, base-weight implicit storage, broad model compression, strict 600x neuron density, arbitrary chat, or full nm behavior.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py -q
python -m pytest tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python -m pytest tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_local_100k_source_subtoken_structure_block_codec.py tests\test_local_100k_source_subtoken_structure_corpus_codec.py tests\test_local_100k_source_subtoken_qa_adapter.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_zstd_trained_dictionary_baseline_audit --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_zstd_trained_dictionary_baseline_audit --profile hard --output-root codex_local_output\suite_zstd_dict_audit_hard --timeout-sec 600
python -m py_compile neuroloc\simulations\memory\local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py neuroloc\simulations\suite_registry.py
git diff --check
```

results:

- focused tests: `6 passed, 1 warning`.
- focused tests plus registry contract: `7 passed, 1 warning`.
- focused current-compression tests plus registry contract: `15 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- py compile: passed.
- diff check: passed with line-ending warnings only.

## decision

accept as a public trained-dictionary baseline audit that strengthens the current narrow and corpus source-subtoken codec products. do not promote it as a new compression product or strict breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
