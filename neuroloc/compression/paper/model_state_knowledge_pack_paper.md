# model state knowledge pack paper package

status: current (as of 2026-05-14).

## paper claim

bounded authored relation knowledge can be compiled into a charged model-state knowledge pack, reloaded through standard neural-module state, updated by recompression, and queried exactly across public source, documentation, and config surfaces while beating strong storage and retrieval pressure lines on the same exact relation task.

## abstract draft

large models often acquire or update factual behavior through finetuning, adapters, retrieval, or model editing, but these routes rarely report exact useful knowledge bits per charged stored bit under false-hit and random-label controls. this paper studies a narrower exact surface: authored relation facts extracted from public software source, documentation, and config files. we introduce a model-state knowledge pack that stores the relation router, value stream, provenance stream, and decoder state inside torch module state, so the payload can be carried by transformer-style, recurrent/state-style, or state-space-style hosts. on pinned cpython `v3.12.3` public files, the hard profile answers `9754` exact relation queries with provenance, charges `679400` relation bits, reaches `37.6093164556962x` strict multiplier against a `2.5` bits-per-parameter ordinary factual baseline, and beats paq8px level 2 source scan by `705864` bits. it also beats zstd level 19 source scan, honest indexed memory, product-key-style storage, rag/knn storage, lora-style exact payload lower bound, and model-edit payload lower bound on the same exact task. random-label, disabled, shuffled, wrong-query, raw-retention, and full-question-table controls collapse. the result does not prove implicit base-weight learning, arbitrary chat, broad knowledge compression, or 600x strict density; it establishes a reproducible exact relation knowledge-pack result and a publishable bridge toward adapter-like deployment.

## mechanism

the product has four charged parts:

- a bounded query normalizer for fixed relation prompts.
- a minimal-hash-style relation router with fingerprints.
- compressed value and provenance id streams plus compressed value and provenance dictionaries.
- a fixed charged decoder and model-package header.

the packed state lives in torch buffers and is present in the `state_dict` of three tiny host families:

- transformer-style host.
- recurrent/state-style host.
- state-space-style host.

the host wrappers are adoption surfaces, not proof that host weights learned the facts. the paper must say this plainly.

## dataset

the hard profile uses public cpython `v3.12.3` files pinned by sha256:

- source: seven python library files.
- documentation: seven library rst files.
- config: `configure.ac`.

the exact fact families are:

- source definition parent relations.
- source statement enclosing-signature relations.
- source control-statement enclosing-signature relations.
- documentation heading relations.
- documentation statement-to-heading context relations.
- config assignment value relations.
- config macro-name relations.
- config macro-payload relations.

the documentation and config questions do not carry answer values in their question text. source relations remain source-authored relation questions and are compared against the same-interface scanner and public compressor pressure.

## baselines

implemented local baselines:

- paq8px v214 level 2 source scan.
- zstd level 19 source scan.
- same-interface scanner charged through compressed source scan.
- honest mph/index relation store.
- undercharged mph diagnostic.
- verbatim table.
- product-key-style storage accounting.
- rag/knn retrieval storage accounting.
- lora-style exact payload lower bound.
- model-edit exact payload lower bound.
- random-label twin and rebuilt random-label codec.
- disabled parser/read/adapter/code/decoder controls.
- shuffled fingerprint and shuffled value controls.
- wrong-query, partial-overlap, and marker-injection false-hit controls.

related-work pressure:

- [lora](https://arxiv.org/abs/2106.09685) and [qlora](https://arxiv.org/abs/2305.14314) establish adapter and quantized-finetuning deployment pressure, but they do not provide exact relation storage under this accounting.
- [rome](https://arxiv.org/abs/2202.05262) and [memit](https://arxiv.org/abs/2210.07229) establish model-editing pressure, but the present product is an explicit charged state pack, not a hidden base-weight edit.
- [rag](https://arxiv.org/abs/2005.11401), [retro](https://arxiv.org/abs/2112.04426), [knn-lm](https://arxiv.org/abs/1911.00172), and [memorizing transformers](https://arxiv.org/abs/2203.08913) establish external retrieval pressure.
- [product-key memory](https://arxiv.org/abs/1907.05242), [memory layers at scale](https://arxiv.org/abs/2412.09764), [titans](https://arxiv.org/abs/2501.00663), and [atlas](https://arxiv.org/abs/2505.23735) establish trainable and test-time memory pressure.
- [gptq](https://arxiv.org/abs/2210.17323), [awq](https://arxiv.org/abs/2306.00978), [zstd](https://github.com/facebook/zstd), and [paq8px](https://github.com/hxim/paq8px) establish model-compression and public-compressor deployment pressure.

## hard result

- public surfaces: `15`.
- total public bytes: `1323186`.
- exact relation facts: `9754`.
- selected relation bits: `679400`.
- model package bits: `687592`.
- useful retrievable bits: `3992464`.
- strict multiplier: `37.6093164556962x`.
- model-package strict multiplier: `37.1612374780393x`.
- paq8px source-scan bits: `1385264`.
- margin over paq8px: `705864`.
- zstd source-scan bits: `2195016`.
- margin over zstd: `1515616`.
- strongest baseline bits: `1385264`.
- margin over strongest baseline: `705864`.
- exact/paraphrased answer success: `1.0`.
- random-label twin success: `0.0`.
- rebuilt random-label exact success: `1.0`.
- rebuilt random-label bits: `2429560`.
- adapter export reload success: `1.0`.
- update lifecycle pass: `1.0`.
- transformer/recurrent/state-space host pass: `1.0`.

## proof obligations satisfied

- public corpus, frozen manifest, hash and byte-length checks.
- multi-surface facts instead of source-only facts.
- exact value and provenance answers.
- charged model-state payload, not raw public-file retention.
- no full question table in state.
- public compressor pressure recomputed in-run.
- explicit storage-pressure ladder.
- standard adapter export and reload.
- recompress update and rollback.
- random-label collapse and random-label rebuild-density control.
- wrong-query, marker-injection, partial-overlap, disabled, and shuffled controls.
- conservative non-claims kept at `0.0`.

## non-claims

the paper must not claim:

- base weights learned the facts.
- broad open-domain knowledge compression.
- arbitrary chat.
- full nm completion.
- 600x strict density.
- superiority to lora, qlora, rome, memit, rag, retro, product-key memory, or memory layers on their native training and generation benchmarks.

the paper may claim:

- stronger exact relation density than the implemented storage and retrieval baselines on this public multi-surface relation benchmark.
- a concrete adapter-like packaging route for exact knowledge packs.
- a reproducible benchmark for future learned host routers, model edits, and memory-layer variants.

## next paper-ready requirements

remaining before external submission:

- run the registered smoke and hard suite commands from a clean checkout.
- write the actual paper draft from this package.
- add license note for cpython source and docs.
- freeze the output metrics json as an artifact.
- optionally add a second public repository family to show cross-project transfer.
- optionally implement a trained bounded prompt router that maps natural prompts into the same pack without adding paraphrase rows.

## see also

- ../README.md
- ../benchmarks/baseline_stack.md
- ../candidates/external_relation_adapter_adoption_path.md
- ../../wiki/tests/local_100k_model_state_knowledge_pack.md
