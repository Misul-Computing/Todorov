# weight-carried knowledge adapter

status: current (as of 2026-05-13).

## claim shape

the product claim is deliberately narrow:

```text
a weight-carried exact-qa codec adapter can store bounded source-heldout qa knowledge in model state with more than 22x strict charged paper-surface density over raw retrievable bits, while preserving exact answers, provenance, random-label collapse, false-hit rejection, host reload, and disabled-path controls.
```

this is not the 600x neuron claim. it is not arbitrary chat. it is not learned semantic memory. it is not proof that base transformer or mamba weights have internalized the facts. it is a deployable adapter-shaped compression surface.

## why this is useful

current llm practice already accepts weight-side adaptation. [lora](https://arxiv.org/abs/2106.09685) shows that small adapter matrices can alter a large transformer without updating the whole base model. [qlora](https://arxiv.org/abs/2305.14314) shows that quantized bases plus adapters can support low-memory finetuning. model-editing work such as [rome](https://arxiv.org/abs/2202.05262), [memit](https://arxiv.org/abs/2210.07229), and [knowledge neurons](https://arxiv.org/abs/2104.08696) shows that factual associations can be localized or changed in transformer modules, but not that the result is dense lossless knowledge compression. memory-layer work such as [product-key memory](https://arxiv.org/abs/1907.05242) and [memory layers at scale](https://arxiv.org/abs/2412.09764) shows that sparse trainable key-value storage is a strong capacity baseline. [titans](https://arxiv.org/abs/2501.00663) shows that test-time memory is an active architecture direction. [mamba](https://arxiv.org/abs/2312.00752), [rwkv](https://arxiv.org/abs/2305.13048), and [retnet](https://arxiv.org/abs/2307.08621) show that recurrent or retention state is a serious alternative to full attention, but not exact lossless long-term fact storage by itself.

the adapter product fits this landscape as a compression layer that can ride beside those systems. it does not replace them. it gives a measurable exact-storage surface that can later be attached to transformer, mamba, rwkv, retnet, or memory-layer backbones.

## implementation contract

the first implementation is `local_100k_weight_carried_qa_codec`. the first paper-facing benchmark implementation is `local_100k_paper_ready_adapter_benchmark`. the current margin/update implementation is `local_100k_margin_recompression_adapter`. the current external relation-adoption implementation is `local_100k_external_relation_adapter`.

it stores:

- one compressed source-heldout payload stream inside torch module state.
- one small charged model-state header.
- a fixed parser and decoder budget.

it does not store:

- raw source block cache.
- external payload path.
- stored manifest.
- source id or offset fields in test facts.
- content digest target keys.
- answer digest target keys.
- assignment rows.
- per-fact value rows.
- per-fact residual rows.

the read path:

- receives a bounded natural-language question.
- derives a token-signature handle at runtime.
- reads the compressed adapter payload from model state.
- decompresses the adapter payload.
- scans candidate anchor windows.
- returns the exact following answer bytes and block-local provenance.

the update path:

- decode the current adapter payload.
- apply the intended knowledge edit or finetuning update.
- recompress the payload.
- write the new payload back into model state.
- rerun exact qa and control gates.

this is the decompression and recompression lifecycle needed for finetuning, rlhf-style post-training, or domain update workflows. the current margin product carries a tiny trained update controller that gates valid recompression edits, while still leaving learned semantic retrieval and implicit base-weight storage unproved.

the paper-facing benchmark adds:

- a transformer-style torch host with the adapter payload inside `state_dict`.
- a recurrent/state-style torch host with the adapter payload inside `state_dict`.
- four source domains: knowledge, wiki, compression, and code.
- paraphrase-stable bounded questions over the same evidence terms without storing paraphrase rows.
- local baseline comparison against lora-style storage, qlora-style storage, model-editing storage, product-key memory, memory-layer, sparse-read, codec-index, minimal-perfect-hash payload, and the previous weight-carried product.

hard validation reaches exact answer success `1.0`, paraphrase-stable answer success `1.0`, controls collapse `1.0`, paper-ready requirement count `5.0`, adapter strict multiplier `16.641752137599937x`, and paper-surface strict multiplier `16.474416229698146x`.

the margin/update benchmark adds:

- a stable four-domain source block with no train-source files.
- source path, hash, and ngram overlap counts at `0.0`.
- wrong, unanswerable, partial-overlap, and marker-injection false-hit controls at `0.0`.
- a trained four-parameter recompression update controller in model state.
- explicit same-block content-scan and undercharged minimal-perfect-hash diagnostics.

hard validation reaches exact answer success `1.0`, paraphrase-stable answer success `1.0`, controls collapse `1.0`, trainable recompression update success `1.0`, update-controller-disabled success `0.0`, paper-ready requirement count `5.0`, adapter strict multiplier `22.732738950163952x`, and paper-surface strict multiplier `22.421639537059313x`. strict breakthrough authorization remains `0.0` because the executable same-block content-scan diagnostic reaches `22.73766839237796x`.

the external relation adapter adds:

- pinned public cpython `v3.12.3` source files with sha256 and byte-length checks.
- exact source-authored relation qa rather than local-only source chunks.
- tiny transformer-style, recurrent/state-style, and state-space-style hosts carrying the relation payload inside their own `state_dict`.
- in-run paq8px v214 level 2 source-scan pressure.
- random-label rebuild-density, parser-disabled, false-hit, raw-source-retention, and full-question-table controls.

hard validation reaches exact relation answer success `1.0`, paraphrased relation answer success `1.0`, transformer, recurrent, and state-space host reload success `1.0`, selected relation bits `413600`, model package bits `417696`, strict multiplier `35.65344680851064x`, model-package strict multiplier `35.303822875967214x`, and margin over paq8px source scan `192472` bits. it remains an adapter-side relation product, not base-weight implicit storage and not a broad breakthrough.

## marketing boundary

strong wording:

```text
a model-state knowledge adapter that stores exact bounded qa knowledge at 15.215x strict charged density in local source-heldout tests.
```

safe product wording:

```text
a compact adapter-side knowledge layer for transformer, mamba, and rwkv-style models. it carries exact domain qa payloads in model state, supports recompression after updates, and reports strict bit accounting with random-label and disabled-path controls.
```

paper-facing wording:

```text
a local model-state qa adapter benchmark that preserves exact answers and provenance across transformer and recurrent/state-style host packages, supports paraphrased bounded questions and recompression updates, and reaches 16.64x strict adapter density under local exact-answer controls.
```

updated paper-facing wording:

```text
a local model-state qa adapter/update benchmark that preserves exact answers and provenance across transformer and recurrent/state-style host packages, supports paraphrased bounded questions, rejects false-hit queries, carries a trained recompression update controller, and reaches 22.42x paper-surface strict density under local exact-answer controls.
```

external-adoption wording:

```text
a public-corpus model-state relation adapter that packages exact source-authored qa knowledge beside transformer and recurrent hosts, answers 6247 relation facts exactly after reload, and beats an in-run paq8px source-scan baseline by 192472 bits under strict accounting.
```

unsafe wording:

```text
600x neural knowledge compression.
```

```text
facts stored inside ordinary neurons.
```

```text
general chat memory.
```

```text
learned semantic understanding.
```

## next proof gate

the next gate should keep the `22.421639537059313x` paper-surface product line and add one of:

- learned semantic handles that work beyond evidence-token paraphrase templates.
- a direct larger-model integration shim where the adapter state lives in the model package and survives save/load.
- a payload-side learned codec that beats the executable same-block content-scan diagnostic.

promotion requires beating the same exact baselines, including product-key memory, sparse read, mph payload, verbatim storage, lora-style adapter storage, and model-editing baselines under equal exact-success accounting.
