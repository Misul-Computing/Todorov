# local 100k external relation adapter

status: current (as of 2026-05-14).

## summary

`local_100k_external_relation_adapter` is a bounded llm-adoptable relation adapter product. it stores source-authored relation facts from pinned public cpython 3.12.3 source files in a charged torch module state, then exposes that state through tiny transformer-style, recurrent/state-style, and state-space-style hosts.

the claim is narrow:

```text
a model-state relation adapter can carry exact bounded source-authored qa knowledge from an external public corpus with materially higher useful relation density than a compressed source-scan baseline, while preserving exact answers, provenance, reload, random-label collapse, false-hit controls, and honest category limits.
```

it is not proof that ordinary base-model neurons implicitly learned the facts. it is not arbitrary chat, not full nm, not 600x strict compression, and not broad high-density knowledge compression.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_external_relation_adapter.py`
- tests: `tests/test_local_100k_external_relation_adapter.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_external_relation_adapter_hard/local_100k_external_relation_adapter/local_100k_external_relation_adapter_metrics.json`

## external corpus

the hard profile downloads or reuses cached pinned public files from cpython `v3.12.3`:

- `lib/argparse.py`
- `lib/asyncio/base_events.py`
- `lib/enum.py`
- `lib/dataclasses.py`
- `lib/pathlib.py`
- `lib/typing.py`
- `lib/unittest/mock.py`

each file is checked against a frozen sha256 and byte length before metrics are accepted.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_external_relation_adapter --profile hard --output-root codex_local_output\suite_l100k_external_relation_adapter_hard --timeout-sec 900
```

result:

- suite result: pass.
- external source count: `7`.
- external source total bytes: `596308`.
- relation fact count: `6247`.
- selected relation accounted bits: `413600`.
- model package accounted bits: `417696`.
- useful retrievable bits: `2304104`.
- strict multiplier: `35.65344680851064`.
- model package strict multiplier: `35.303822875967214`.
- paq8px level 2 source-scan accounted bits: `606072`.
- margin over paq8px level 2 source scan: `192472`.
- honest mph relation index bits: `3511720`.
- margin over honest mph relation index bits: `3098120`.
- undercharged mph relation bits: `2518447`.
- margin over undercharged mph relation bits: `2104847`.
- exact relation answer success: `1.0`.
- paraphrased relation answer success: `1.0`.
- random-label twin success: `0.0`.
- random-label rebuild exact success: `1.0`.
- random-label rebuild selected relation accounted bits: `1558016`.
- decoder-disabled success: `0.0`.
- parser-disabled prefixed success: `0.0`.
- shuffled-fingerprint success: `0.0`.
- wrong-query hit rate: `0.0`.
- transformer host reload success: `1.0`.
- recurrent host reload success: `1.0`.
- state-space host reload success: `1.0`.
- host parameter count max: `6592`.
- external relation adapter product authorization: `1.0`.
- llm-adoptable relation adapter candidate: `1.0`.
- true base-weight implicit storage authorization: `0.0`.
- broad breakthrough authorization: `0.0`.
- strict 600x authorization: `0.0`.
- broad knowledge authorization: `0.0`.
- arbitrary chat authorization: `0.0`.
- full nm authorization: `0.0`.

## controls

- external source hashes and sizes must match the frozen manifest.
- exact and bounded paraphrased relation questions must answer with exact value and provenance.
- random-label twin scored against the selected adapter must fail.
- separately rebuilt random-label adapter must answer exactly but cost more.
- decoder, read, adapter, code, and prefixed-query parser disabled paths must fail.
- shuffled fingerprints and wrong-query variants must fail.
- torch `state_dict` reload must preserve exact answers.
- tiny transformer-style, recurrent/state-style, and state-space-style hosts must carry the adapter tensors inside their own state dicts.
- serialized module state must not contain raw source blocks or the full question table.
- paq8px v214 level 2 source-scan pressure must be recomputed in-run.

## adoption interpretation

the result supports a peft-like or quantization-package-like route: a domain relation pack can live in model state, reload with the model, answer bounded factual relation queries exactly, and be compared in bits against compressed source scan and indexed memory baselines.

it does not prove ordinary transformer, mamba, or recurrent weights have internalized the facts through gradient training. that remains the next, harder route: either train a host to call the adapter reliably from natural prompts, or compile the same charged relation stream into a standard adapter format with an update/decompression lifecycle comparable to lora, qlora, model editing, and memory-layer baselines.

## category check

implemented operation:

external public source-authored relation qa from a charged model-state adapter.

strongest baselines:

paq8px v214 level 2 source scan, honest mph relation index, undercharged mph diagnostic, disabled controls, random-label twin, random-label rebuild-density control, and transformer/recurrent host state reload.

what passed:

the hard profile answers all `6247` relation facts exactly from charged module state, beats the recomputed paq8px source-scan baseline by `192472` bits, beats honest mph indexing by `3098120` bits, survives public external corpus hashing, and passes transformer, recurrent, and state-space host probes.

what remains limited:

the result is still a keyed relation adapter. it is not a base-weight neuron proof, not arbitrary chat, not broad knowledge compression, not a static source-code compressor, not 600x strict density, and not full nm completion. the original in-repo relation product still has the higher strict multiplier (`67.90445687825584x`), while this external product has the stronger public-corpus adoption and paq8px-margin evidence.

## verification

commands run:

```text
python -m pytest tests\test_local_100k_external_relation_adapter.py -q
python -m pytest tests\test_local_100k_external_relation_adapter.py tests\test_simulation_suite.py::test_suite_registry_contract -q
python neuroloc\simulations\suite_runner.py --simulation local_100k_external_relation_adapter --profile smoke --timeout-sec 300
python neuroloc\simulations\suite_runner.py --simulation local_100k_external_relation_adapter --profile hard --output-root codex_local_output\suite_l100k_external_relation_adapter_hard --timeout-sec 900
python -m py_compile neuroloc\simulations\memory\local_100k_external_relation_adapter.py tests\test_local_100k_external_relation_adapter.py neuroloc\simulations\suite_registry.py
```

results:

- focused tests: `4 passed, 1 warning`.
- focused tests plus registry contract: `5 passed, 1 warning`.
- smoke suite: `1/1 passed`.
- hard suite: `1/1 passed`.
- python compile check: pass.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_relation_mph_codec]]
- [[tests/paq8px_public_context_mixing_baseline_audit]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
