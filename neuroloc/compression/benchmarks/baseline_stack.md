# required baseline stack

status: current (as of 2026-05-14).

## role

this file defines the baseline ladder that the next compression simulation must beat. the purpose is to prevent another false positive that only beats no-memory.

## target data

the next target remains non-generated unknown-structure local corpus facts unless the canonical plan changes.

minimum required split:

- source-heldout train and test files.
- train/test key overlap equal to `0.0`.
- opaque keys for the associative setting.
- exact byte value and exact provenance labels.
- random-label twin with the same keys, lengths, and provenance surface.

## baseline 0: no-memory and recency controls

must fail:

- no-memory
- recency-only
- wrong-key
- wrong-provenance
- shuffled-value
- shuffled-key
- shuffled-provenance

purpose:

these catch trivial leakage and template shortcuts. they are not sufficient as positive baselines.

## baseline 1: verbatim table

charge:

- full keys if stored
- full values
- provenance
- index and manifest

purpose:

this is the upper-storage exact success baseline.

## baseline 2: charged corpus codec

charge:

- compressed payload
- decoder
- dictionary
- manifest
- key or offset representation
- provenance

current numbers:

`local_100k_unknown_structure_density_probe` reports `13.941917871967359x` strict multiplier. this is the current exact unknown-structure baseline to beat.

`local_100k_llm_semantic_qa_codec` reports `14.06935717190861x` strict multiplier on a bounded natural-language qa surface, but the minimal-perfect-hash payload line remains slightly stronger at `14.070360120095941x`.

`local_100k_weight_carried_qa_codec` reports `15.215221373768888x` strict multiplier by carrying the compressed source qa payload inside model state, removing the stored manifest, and charging a smaller fixed parser and decoder budget. under its no-manifest accounting, the matched minimal-perfect-hash payload line is `15.214669447719235x`, so the product beats that line narrowly.

`local_100k_paper_ready_adapter_benchmark` reports `16.641752137599937x` adapter strict multiplier and `16.474416229698146x` paper-surface strict multiplier by using a compact four-domain source-heldout block, carrying the compressed payload inside model state, and testing transformer/recurrent host packaging, paraphrase-stable bounded qa, and recompression update. under that local compact-block accounting, the matched minimal-perfect-hash payload line is `16.64109186851554x`, so the product beats that line narrowly while also beating the previous weight-carried product.

`local_100k_margin_recompression_adapter` reports `22.732738950163952x` adapter strict multiplier and `22.421639537059313x` paper-surface strict multiplier by using a smaller stable four-domain source block, carrying the compressed payload inside model state, repairing the prior source-holdout overlap, testing transformer/recurrent host packaging, rejecting false-hit queries, and carrying a trained recompression update controller. it beats the prior `16.641752137599937x` adapter line by a large margin, but it does not beat the executable same-block content-scan diagnostic at `22.73766839237796x` or the same-block undercharged minimal-perfect-hash diagnostic, so it is a bounded adapter/update product rather than a strict breakthrough.

`local_100k_indent_token_block_codec` reports a different source-code byte-compression surface rather than a qa retrieval surface. it learns a four-byte indentation token from disjoint train files, charges the token map, reconstructs the held-out block exactly, and improves strict charged bits over the best same-block standard codec by `0.020192022171632188`. the random-label byte twin has negative compression gain, and no raw/transformed/restored block is retained as state. this is a narrow source-block codec product, not a high-density knowledge result, static retrieval breakthrough, or broad breakthrough authorization.

`local_100k_source_structure_block_codec` superseded the indentation-token product on the same held-out source-code block. it learns a four-space indentation unit from disjoint train files, separates indentation counts from body bytes, charges a `256` bit structure header, reconstructs exactly from compressed count/body streams, and improves strict charged bits over the best same-block standard codec by `0.028555874492724932`.

`local_100k_source_token_structure_block_codec` superseded the source-structure product on the same held-out source-code block. it keeps the indentation split, delta-codes the count stream, adds a fully charged target identifier dictionary, reconstructs exactly from compressed count/body/dictionary streams, and improves strict charged bits over the best same-block standard codec by `0.03543501930119766`. charged payload bits fall from `124200` to `123088`.

`local_100k_source_subtoken_structure_block_codec` supersedes the source-token product on the same held-out source-code block. it keeps the indentation split and charged dictionary accounting but uses reversible longest-match subtoken substitution in the body stream. charged payload bits fall from `123088` to `120952`, and strict charged-bit improvement over the best same-block standard codec rises to `0.048648916163515785`. this is the current narrow source-code byte-compression product, still not a high-density knowledge result, static retrieval breakthrough, or broad breakthrough authorization.

`local_100k_source_subtoken_structure_corpus_codec` broadened the same transform to a frozen five-block source-code manifest. all five blocks reconstruct exactly, aggregate selected payload bits fall from the same-block standard sweep's `849752` to `812688`, aggregate payload improvement is `0.043617431909545375`, and random-label/control gates pass. this is now superseded by the shared-dictionary corpus product.

`local_100k_source_subtoken_shared_dictionary_corpus_codec` superseded the first frozen corpus product. it keeps count-delta source-structure coding but amortizes the target dictionary across the frozen corpus with one charged `112` token shared dictionary plus charged `16` token local dictionaries per block. aggregate selected payload bits fall from `812688` to `803400`, aggregate improvement over the same-block standard sweep rises to `0.05454767979363391`, and random-label/control gates pass. this is now superseded by the global-stream corpus product.

`local_100k_source_subtoken_global_stream_corpus_codec` is the current broader source-code corpus codec product. it keeps the frozen five-block manifest, one charged `256` token shared dictionary, and count/body source-structure split, but compresses one global count stream, one global body stream, and one charged length/framing stream across the corpus. the first `120` shared tokens use one-byte codes and the remaining tokens use escaped varint codes. aggregate selected payload bits fall from `803400` to `699144`, aggregate improvement over the same-block standard sweep rises to `0.1772375940274339`, and the product also beats a stronger global raw standard baseline at `736504` bits by `37360` bits. random-label/control gates pass, and the charged payload reconstructs after torch module `state_dict` reload. this is still not a high-density knowledge result, static retrieval breakthrough, or broad breakthrough authorization.

`local_100k_source_subtoken_disjoint_retrieval_codec` adds a narrow exact-retrieval surface over disjoint frozen source blocks. it retrieves `14715` aligned 32-byte chunks exactly after torch `state_dict` reload, uses zero train path/hash overlap, charges `431536` retrieval bits, beats executable raw content scan at `451128` bits by `19592` bits, beats an undercharged mph diagnostic at `451144` bits by `19608` bits, and keeps random-label/control gates passing. this is source-code chunk retrieval, not learned semantic qa and not broad breakthrough authorization.

`local_100k_source_authored_relation_diagnostic` tests source-authored definition/import relation qa after the fixed-stride mistake. it answers `337` relation queries exactly and beats raw/undercharged diagnostics by the global-stream payload margin, but a fair unlimited relation-aware scanner solves the same task and an honest mph relation index is cheaper by `132664` bits. keep it as a diagnostic, not a product.

`local_100k_source_dense_authored_relation_diagnostic` expands the relation surface to `3741` source-authored definition-parent, statement-enclosing, and control-statement-enclosing queries after removing the answer-contained `definition_signature` leak documented in [[mistakes/dense_relation_signature_query_leakage]]. it answers exactly and beats honest relation mph/index by `2928400` bits, but loses to the paq8px level 2 relation payload line by `176536` bits and does not beat the fair unlimited scanner. keep it as an amortization diagnostic, not a product.

`local_100k_source_relation_mph_codec` is the current narrow source-authored relation-index product. it stores a charged minimal-perfect-hash-style router, 17-bit fingerprints, compressed value/provenance id streams, and compressed value/provenance dictionaries inside torch module state. hard validation answers `3741` relation queries exactly after `state_dict` reload, charges `248784` selected relation bits, recomputes the paq8px level 2 relation line in-run at `252952` payload bits plus `8192` decoder bits, beats that paq8px line by `12360` bits, beats raw-source paq content scan by `165104` bits, beats the undercharged mph relation diagnostic by `2522477` bits, beats the honest mph relation index by `3117296` bits, and reaches `67.90445687825584x` strict multiplier. cross-scored random labels, disabled decode, shuffled fingerprints, and hardened wrong-query variants collapse; a separately rebuilt random-label codec stores exactly but costs `934984` bits, so random-label density gain collapses. this is a narrow keyed relation-index product only: self-contained paq8px baseline-win authorization is `1.0`, while static breakthrough, strict 600x, broad knowledge, chat, and full nm authorization remain `0.0`.

`local_100k_external_relation_adapter` is the current public-corpus llm-adoptable relation adapter product. it stores pinned cpython `v3.12.3` source-authored relation facts in charged torch module state and packages the payload inside tiny transformer-style and recurrent/state-style hosts. hard validation answers `6247` exact relation facts, charges `413600` selected relation bits plus a `4096` bit model-package header, reaches `35.65344680851064x` strict multiplier and `35.303822875967214x` model-package strict multiplier, recomputes paq8px v214 level 2 source-scan pressure at `606072` accounted bits, beats that source-scan line by `192472` bits, beats honest mph relation indexing by `3098120` bits, and keeps random-label, disabled, parser-disabled, shuffled-fingerprint, wrong-query, raw-source-retention, and full-question-table controls collapsed. this is an adapter-side adoption product, not base-weight implicit storage or broad breakthrough authorization.

`local_100k_model_state_knowledge_pack` is the current broader paper-facing model-state knowledge pack product. it extends the external relation adapter from source-only facts to pinned public source, documentation, and config surfaces from cpython `v3.12.3`. hard validation answers `9754` exact relation facts, charges `679400` selected relation bits plus an `8192` bit model-package header, reaches `37.6093164556962x` strict multiplier and `37.1612374780393x` model-package strict multiplier, recomputes paq8px v214 level 2 source-scan pressure at `1385264` accounted bits, beats that strongest checked baseline by `705864` bits, beats zstd level 19 source scan by `1515616` bits, beats honest mph indexing by `5194024` bits, beats product-key-style storage, rag/knn storage, lora-style exact payload lower bound, and model-edit exact payload lower bound, and passes state-dict reload, standard adapter export, recompress update, rollback, transformer, recurrent, and state-space host probes. this is the current paper-facing bounded exact knowledge-pack result. base-weight implicit storage, broad open-domain knowledge, arbitrary chat, full nm, strict 600x, and broad breakthrough authorization remain `0.0`.

`local_100k_zstd_trained_dictionary_baseline_audit` adds the public trained-dictionary pressure line for the source-code codec products. charged zstd dictionaries trained only on target-excluded local source files lose to the current source-subtoken block payload by `30424` bits and to the current global-stream frozen-corpus payload by `283696` bits. even the undercharged payload-only dictionary diagnostic loses by `26248` bits on the block and `250848` bits on the corpus. this strengthens the source-code byte-compression products but does not change the breakthrough, knowledge, chat, full nm, or 600x authorization flags.

the 2026-05-14 paq8px v214 scratch audit adds a stronger public context-mixing pressure line. level 2 compressed the hard raw joined source from `802589` bytes to `50712` archive bytes, or `405696` payload bits. level 1 compressed the same source to `51889` bytes and the transformed body stream to `52665` bytes. this beats `local_100k_source_subtoken_global_stream_corpus_codec` at `699144` selected payload bits and beats the current disjoint retrieval payload before retrieval wrapper accounting. source-code byte-compression promotion now requires beating this paq8px line or explicitly staying a pre-paq local diagnostic. see [[mistakes/public_context_mixing_baseline_missing]].

`local_100k_source_structure_qa_adapter` applies the source-structure payload to the bounded exact qa adapter surface. it reaches `24.03612607449857x` adapter strict multiplier and beats raw content scan plus raw undercharged mph diagnostics, but it does not beat the fair same-structure content scan at `24.041637051473117x`.

`local_100k_source_subtoken_qa_adapter` supersedes the source-structure qa adapter on the same bounded exact qa surface. it carries count-delta, body-subtoken, and charged dictionary streams inside torch module state, preserves transformer/recurrent hosts and trained recompression update, reaches `24.19976921301639x` adapter strict multiplier and `23.847532408460314x` paper-surface strict multiplier, but it does not beat the fair same-subtoken content scan at `24.20535549399815x`.

`local_100k_source_subtoken_delta_update_adapter` adds a narrow delta-update surface on top of the source-subtoken qa adapter. it stores varint-delta offsets plus full replacement bytes for `512` updated facts in a charged `138104` bit model-state patch stream, answers updated and unchanged facts exactly after `state_dict` reload, beats full updated-adapter recompression as the same-block content-scan update baseline by `242448` delta bits, and beats an undercharged mph update table by `50312` bits. the matched delta-patch content-scan diagnostic is equal at `138104` bits and is not beaten. it is not a total static compression product because base-plus-delta total static bits lose to full recompression by `1992` bits.

## baseline 3: minimal perfect hash plus payload codec

charge:

- hash tables
- hash seeds
- key fingerprints if used
- payload codec
- decoder
- manifest
- provenance

purpose:

this is the exact classical key-routing baseline. it exposes learned methods that are only row stores with a different address function.

## baseline 4: product-key memory

charge:

- query network
- product subkeys
- selected value table
- full value table or equivalent trainable value state
- decoder
- manifest
- provenance

purpose:

this is the learned sparse value-table baseline.

## baseline 5: content-routed sparse read

charge:

- selected records
- routing keys
- cached corpus records
- decoder
- manifest

purpose:

this baseline tests whether selecting a few raw records beats compression.

## baseline 6: hdc or sdm superposition

charge:

- item dictionary
- cleanup dictionary
- hypervector seeds or actual vectors
- superposed state
- counters or hard locations
- decoder and provenance mapping

purpose:

this tests whether distributed associative storage gives useful partial-cue behavior or strict density. random-label collapse is mandatory.

## baseline 7: learned entropy-coded codec

charge:

- model parameters
- codebooks
- tokenizer or phrase model
- entropy model
- arithmetic-coded latent stream
- manifest
- provenance
- any exceptions

purpose:

this is the honest non-row candidate and a baseline for stronger cells.

## promotion comparison

promotion requires:

```text
exact_success >= 0.95
random_label_exact_success near 0.0
charged_bits < min(codec_bits, current_margin_adapter_bits, executable_content_scan_bits, mph_payload_bits, product_key_bits, sparse_read_bits, verbatim_bits)
```

plus all disabled and shuffled controls collapse.

for source-code byte-compression claims, `codec_bits` must include the paq8px public context-mixing line unless a narrower operation makes that baseline inapplicable and the reason is documented. if the paq8px line is imported as a constant rather than recomputed in-run, the run must report that fact and cannot claim a self-contained public-baseline win.

## useful density lines

report both:

- params-only density
- strict params-plus-state density

only strict density can support a real high-density claim.
