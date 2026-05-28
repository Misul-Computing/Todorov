# external relation adapter adoption path

status: current (as of 2026-05-14).

## claim boundary

the external relation adapter result proves a narrower but more deployable point than the high-density neuron claim:

```text
bounded exact relation knowledge can be packaged inside a model state adapter, tested on a public external corpus, and stored more compactly than compressed source scan and relation-index baselines under strict accounting.
```

it does not prove that ordinary transformer or recurrent weights implicitly learned the facts. it proves a model-package route: a charged relation payload can ride beside transformer, mamba-like state-space, rwkv-like recurrent, or memory-layer hosts and answer bounded relation questions after `state_dict` reload.

## result

`local_100k_external_relation_adapter` uses pinned cpython `v3.12.3` files as an external corpus. hard validation answers `6247` relation facts exactly, charges `413600` selected relation bits, reaches `35.65344680851064x` strict density and `35.303822875967214x` model-package strict density, beats the in-run paq8px v214 level 2 source-scan baseline by `192472` bits, and beats honest mph relation indexing by `3098120` bits.

the broader successor is `local_100k_model_state_knowledge_pack`. it uses pinned public source, documentation, and config surfaces, answers `9754` relation facts exactly, charges `679400` selected relation bits, reaches `37.6093164556962x` strict density and `37.1612374780393x` model-package strict density, beats paq8px v214 level 2 source-scan pressure by `705864` bits, beats zstd level 19 source scan by `1515616` bits, and passes adapter export, recompress update, rollback, and three host-family reload probes.

the older in-repo `local_100k_source_relation_mph_codec` remains the highest strict multiplier at `67.90445687825584x`, but the model-state knowledge pack is the stronger paper-facing adoption proof because it is public, multi-surface, reloadable, update-tested, and compared against more deployment-shaped baselines.

## adoption routes

the best first route is a peft-like adapter:

- train or compile a bounded relation pack.
- store the charged payload as a model-state shard.
- attach a small read module to a transformer or recurrent host.
- ship it like an adapter, not like a prompt, raw file, or hidden database.
- support decode, edit, recompress, and reload as first-class operations.

the second route is a quantization-like packaging step:

- freeze the host model.
- compile relation facts into a compact relation stream.
- include the stream in the model package next to quantized tensors.
- charge every payload, header, decoder, and routing bit.
- expose deterministic update and rollback.

the third route is a memory-layer backend:

- treat the relation codec as a compressed factual shard behind a learned or routed read head.
- compare against product-key memory, memory layers at scale, rag, retro, memorizing transformers, and test-time memory.
- require exactness, false-hit rejection, update locality, and lower storage or read work.

## public pressure

primary pressure sources:

- lora and qlora prove that adapter-side model changes are the accepted deployment route for cheap llm customization, but not exact high-density relation storage.
- gptq, bitsandbytes, and autogptq prove that quantization packaging is mature and adoption depends on drop-in tooling, not just a clever encoding.
- rome and memit prove that factual associations can be edited inside transformer mechanisms, but do not prove dense lossless relation storage without side effects.
- rag, retro, and memorizing transformers prove external memory and retrieval are strong baselines; a relation adapter must beat compressed source scan or retrieval storage where those baselines have the same evidence.
- product-key memory and memory layers at scale prove sparse trainable value tables are serious capacity baselines.
- titans and infini-attention prove current architectures are actively moving toward test-time memory and compressive long-context state; the adapter must become compatible with those routes rather than ignore them.

## completed implementation gate

the 2026-05-14 model-state knowledge pack completed three of the previously listed gates:

- standard adapter export format with load/save/update/recompress lifecycle.
- multi-corpus public benchmark across source, documentation, and config factual surfaces.
- charged lora-style, model-edit-style, rag/knn, product-key, paq8px, zstd, mph, and scanner comparison lines on the same relation set.

the remaining next gate should not chase a bigger label. it should add one of these:

- a trained host router that maps natural bounded prompts to the relation adapter without storing paraphrase rows.
- a second public repository family so the paper can report cross-project robustness.
- a real model-edit comparison where the same relation set is attempted through lora-style parameters or memit-style edits under charged storage and exact-success controls.

promotion requires exact success `>=0.95`, random-label collapse, public-compressor recomputation, host reload, no hidden tables, and an explicit statement that base-weight implicit storage remains unproved unless a real trained host weight path passes.

## see also

- ../README.md
- weight_carried_knowledge_adapter.md
- ../benchmarks/baseline_stack.md
- ../../wiki/tests/local_100k_external_relation_adapter.md
