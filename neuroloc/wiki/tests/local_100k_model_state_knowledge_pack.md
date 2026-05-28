# local 100k model state knowledge pack

status: current (as of 2026-05-14).

## summary

`local_100k_model_state_knowledge_pack` is the broader paper-facing successor to `local_100k_external_relation_adapter`.

it packages exact bounded relation knowledge from pinned public cpython `v3.12.3` source, documentation, and config surfaces into charged torch module state. it then proves exact answer, provenance, host reload, standard adapter export, recompress update, rollback, random-label collapse, false-hit rejection, and charged baseline wins.

the claim is still bounded:

```text
a model-state knowledge pack can carry exact authored relation knowledge across multiple public surfaces with materially higher useful relation density than public-compressor source scan, zstd source scan, honest indexed memory, product-key-style storage, rag/knn storage, lora-style exact payload storage, and model-edit payload lower-bound baselines under strict accounting.
```

it is not proof that ordinary base weights learned the facts. it is not arbitrary chat, not a full nm, not 600x strict compression, and not broad open-domain knowledge compression.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_model_state_knowledge_pack.py`
- tests: `tests/test_local_100k_model_state_knowledge_pack.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_model_state_knowledge_pack_hard/local_100k_model_state_knowledge_pack/local_100k_model_state_knowledge_pack_metrics.json`

## public corpus

the hard profile downloads or reuses cached pinned public files from cpython `v3.12.3`:

- seven source files: `argparse.py`, `asyncio/base_events.py`, `enum.py`, `dataclasses.py`, `pathlib.py`, `typing.py`, and `unittest/mock.py`.
- seven documentation files: `argparse.rst`, `asyncio-eventloop.rst`, `enum.rst`, `dataclasses.rst`, `pathlib.rst`, `typing.rst`, and `unittest.mock.rst`.
- one config file: `configure.ac`.

each file is checked against a frozen sha256 and byte length before metrics are accepted.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_model_state_knowledge_pack --profile hard --output-root codex_local_output\suite_l100k_model_state_knowledge_pack_hard --timeout-sec 1200
```

current direct hard run result before suite output-root rerun:

- public surface count: `15`.
- public surface total bytes: `1323186`.
- source surface count: `7`.
- document surface count: `7`.
- config surface count: `1`.
- relation fact count: `9754`.
- selected relation accounted bits: `679400`.
- model package accounted bits: `687592`.
- useful retrievable bits: `3992464`.
- strict multiplier: `37.6093164556962`.
- model package strict multiplier: `37.1612374780393`.
- paq8px level 2 source-scan accounted bits: `1385264`.
- margin over paq8px level 2 source scan: `705864`.
- zstd level 19 source-scan accounted bits: `2195016`.
- margin over zstd level 19 source scan: `1515616`.
- strongest baseline accounted bits: `1385264`.
- margin over strongest baseline: `705864`.
- honest mph relation index bits: `5873424`.
- margin over honest mph relation index: `5194024`.
- product-key memory storage bits: `13947736`.
- margin over product-key memory storage: `13268336`.
- rag/knn retrieval storage bits: `2641968`.
- margin over rag/knn retrieval storage: `1962568`.
- lora-style exact payload lower-bound bits: `4312784`.
- margin over lora-style exact payload lower-bound: `3633384`.
- model-edit exact payload lower-bound bits: `4624912`.
- margin over model-edit exact payload lower-bound: `3945512`.
- exact relation answer success: `1.0`.
- paraphrased relation answer success: `1.0`.
- same-interface scanner success: `1.0`.
- random-label twin success: `0.0`.
- random-label rebuild exact success: `1.0`.
- random-label rebuild selected relation accounted bits: `2429560`.
- decoder-disabled success: `0.0`.
- parser-disabled prefixed success: `0.0`.
- shuffled-fingerprint success: `0.0`.
- wrong-query hit rate: `0.0`.
- adapter export reload success: `1.0`.
- update lifecycle pass: `1.0`.
- update patch accounted bits: `198312`.
- updated full recompress accounted bits: `780432`.
- rollback reload success: `1.0`.
- transformer host reload success: `1.0`.
- recurrent host reload success: `1.0`.
- state-space host reload success: `1.0`.
- host parameter count max: `6592`.
- product authorization: `1.0`.
- paper-ready bounded knowledge pack candidate: `1.0`.
- true base-weight implicit storage authorization: `0.0`.
- broad breakthrough authorization: `0.0`.
- strict 600x authorization: `0.0`.
- broad knowledge authorization: `0.0`.
- arbitrary chat authorization: `0.0`.
- full nm authorization: `0.0`.

## controls

- public source hashes and sizes must match the frozen manifest.
- exact and bounded paraphrased relation questions must answer with exact value and provenance.
- source, document, and config relation families must all be present.
- non-source document/config questions must not contain the answer value.
- same-interface scanner success is charged through the compressed source-scan baseline.
- random-label twin scored against the selected adapter must fail.
- separately rebuilt random-label adapter must answer exactly but cost more.
- decoder, read, adapter, code, and prefixed-query parser disabled paths must fail.
- shuffled fingerprints and wrong-query variants must fail.
- torch `state_dict` reload must preserve exact answers.
- adapter export reload must preserve exact answers.
- update recompression and rollback must preserve exact answers.
- tiny transformer-style, recurrent/state-style, and state-space-style hosts must carry the adapter tensors inside their own state dicts.
- serialized module state must not contain raw public surfaces or the full question table.
- paq8px v214 level 2 source-scan pressure must be recomputed in-run.
- zstd level 19 source-scan pressure must be recomputed in-run.

## category check

implemented operation:

external public authored relation qa from a charged model-state knowledge pack, covering source, documentation, and config surfaces.

strongest baselines:

paq8px v214 level 2 source scan, zstd level 19 source scan, honest mph relation index, undercharged mph diagnostic, product-key-style storage, rag/knn storage, lora-style exact payload lower bound, model-edit exact payload lower bound, disabled controls, random-label twin, random-label rebuild-density control, same-interface scanner, and transformer/recurrent host reload.

what passed:

the hard profile answers all `9754` relation facts exactly from charged module state, beats the recomputed paq8px source-scan baseline by `705864` bits, beats the zstd source-scan baseline by `1515616` bits, beats honest mph indexing by `5194024` bits, survives public external corpus hashing, and passes export, update, rollback, transformer, recurrent, and state-space host probes.

what remains limited:

the result is still a bounded relation knowledge pack. it is not proof of implicit base-weight learning, not arbitrary chat, not full nm completion, not 600x strict density, not broad open-domain knowledge compression, and not a claim that lora, memit, rag, or product-key memory were beaten on their native training benchmarks. those methods are used here as charged storage pressure lines on the same exact relation surface.

## verification

commands run so far:

```text
python neuroloc\simulations\memory\local_100k_model_state_knowledge_pack.py smoke
python neuroloc\simulations\memory\local_100k_model_state_knowledge_pack.py hard
python -m pytest tests\test_local_100k_model_state_knowledge_pack.py -q
```

results:

- direct smoke simulation: pass, `3875` facts, `36.5978345847746x` strict multiplier, `374032` bit paq margin.
- direct hard simulation: pass, `9754` facts, `37.6093164556962x` strict multiplier, `705864` bit paq margin.
- focused tests: `6 passed, 1 warning`.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_external_relation_adapter]]
- [[tests/local_100k_source_relation_mph_codec]]
- [[tests/paq8px_public_context_mixing_baseline_audit]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
