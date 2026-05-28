# local 100k source-subtoken disjoint retrieval codec

status: current (as of 2026-05-14).

## summary

`local_100k_source_subtoken_disjoint_retrieval_codec` is a narrow exact-retrieval product built from the global-stream source-subtoken codec.

it uses only frozen target source blocks with zero train-path and train-hash overlap against the small structure-training source set. it stores the charged shared dictionary, global count stream, global body stream, and length stream inside torch module state, reloads from `state_dict`, reconstructs the source blocks exactly, and retrieves aligned 32-byte chunks exactly from the reloaded state.

this is not fake reconstruction: hard validation reloads an empty module from only `global_header`, `shared_dictionary_payload`, `count_payload`, `body_payload`, and `length_payload`, then reconstructs and retrieves chunks. the original blocks, expected answers, manifest rows, and raw source prefixes are not used during decode.

this is a source-code exact-retrieval codec product, not learned semantic knowledge, arbitrary chat, full nm behavior, strict 600x proof, broad compression breakthrough, or paid-compute authorization.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_subtoken_disjoint_retrieval_codec.py`
- tests: `tests/test_local_100k_source_subtoken_disjoint_retrieval_codec.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_subtoken_disjoint_retrieval_hard/local_100k_source_subtoken_disjoint_retrieval_codec/local_100k_source_subtoken_disjoint_retrieval_codec_metrics.json`

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_disjoint_retrieval_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_disjoint_retrieval_hard --timeout-sec 600
```

result:

- suite result: pass.
- block count: `3`.
- retrieval fact count: `14715`.
- chunk bytes: `32`.
- exact reconstruction success: `1.0`.
- heldout chunk retrieval success: `1.0`.
- state-dict reload chunk retrieval success: `1.0`.
- source train-test path overlap count: `0.0`.
- source train-test hash overlap count: `0.0`.
- selected payload bits: `429488`.
- selected retrieval accounted bits: `431536`.
- standard retrieval accounted bits: `473008`.
- raw content-scan accounted bits: `451128`.
- undercharged mph accounted bits: `451144`.
- margin over standard retrieval bits: `41472`.
- margin over raw content scan bits: `19592`.
- margin over undercharged mph bits: `19608`.
- strict multiplier: `55.86800637721998`.
- random-label payload incompressible: `1.0`.
- controls collapse: `1.0`.
- model-state codec payload used: `1.0`.
- state-dict buffer payload used: `1.0`.
- state-dict raw source block retained: `0.0`.
- source-code retrieval codec product authorized: `1.0`.
- source-code retrieval breakthrough authorized: `0.0`.
- static breakthrough authorized: `0.0`.
- strict breakthrough authorized: `0.0`.

## controls

- exact reconstruction and chunk retrieval must succeed from the reloaded module state.
- train source and target source path/hash overlap counts must be zero.
- raw content-scan and undercharged mph diagnostics are measured on the same chunk-retrieval surface.
- random-label byte twins must not receive a compression gain against standard retrieval or raw content-scan retrieval.
- wrong-indent, shared-dictionary-disabled, shuffled shared-dictionary, shuffled body-payload, shuffled count-payload, and shuffled length-payload controls collapse.
- state dict must not retain a raw source block prefix.
- broad knowledge, broad nm, broad chat, full nm, paid compute, static breakthrough, and strict breakthrough authorization stay `0.0`.

## category check

implemented operation:

lossless source-code retrieval from a disjoint frozen corpus using source-structure splitting, one charged shared identifier-subtoken dictionary, a global compressed count stream, a global compressed body stream, a charged compressed framing stream, and torch module state for the charged payload.

strongest baseline on this surface:

the executable raw content-scan diagnostic at `451128` accounted bits and the undercharged mph diagnostic at `451144` accounted bits.

what passed:

the product retrieves `14715` aligned 32-byte chunks exactly after `state_dict` reload, uses no train-overlapping target source files, lowers accounted retrieval bits to `431536`, beats raw content scan by `19592` bits, beats the undercharged mph diagnostic by `19608` bits, and keeps random-label/control gates passing.

what failed or remains limited:

this is still offset-style source-code chunk retrieval. it does not solve the bounded qa adapter's fair same-subtoken content-scan blocker, learned semantic retrieval, arbitrary unknown-structure knowledge compression, implicit base-weight storage, arbitrary chat, full nm behavior, or strict 600x density.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_source_subtoken_disjoint_retrieval_codec.py -q
python -m pytest tests\test_local_100k_source_subtoken_disjoint_retrieval_codec.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_disjoint_retrieval_codec --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_subtoken_disjoint_retrieval_codec --profile hard --output-root codex_local_output\suite_l100k_source_subtoken_disjoint_retrieval_hard --timeout-sec 600
python -m pytest tests\test_local_100k_source_subtoken_disjoint_retrieval_codec.py tests\test_local_100k_source_subtoken_global_stream_corpus_codec.py tests\test_local_100k_zstd_trained_dictionary_baseline_audit.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python -m py_compile neuroloc\simulations\memory\local_100k_source_subtoken_disjoint_retrieval_codec.py tests\test_local_100k_source_subtoken_disjoint_retrieval_codec.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests: `4 passed, 1 warning`.
- focused tests plus registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- broader regression bundle: `16 passed, 1 warning`.
- python compile check: pass.

## decision

promote as the current narrow source-code exact-retrieval codec product. do not promote as learned semantic retrieval, static qa breakthrough, high-density knowledge compression, general neural compression, chat, full nm, strict 600x proof, or broad breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
