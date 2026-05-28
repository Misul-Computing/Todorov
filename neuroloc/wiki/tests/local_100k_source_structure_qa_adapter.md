# local 100k source-structure qa adapter

status: historical context only. frozen as of 2026-05-13. do not edit.

## summary

`local_100k_source_structure_qa_adapter` is a bounded exact qa adapter product built from the source-structure split codec.

it stores a source-heldout exact qa payload inside torch module state as compressed indentation-count and body streams, reloads that payload through `state_dict`, answers bounded evidence-token questions and paraphrases exactly, and keeps the trained recompression-update controller from the margin adapter line.

this is a stronger bounded adapter payload than the raw margin adapter for the raw content-scan and raw undercharged mph diagnostics. it is not a strict static-retrieval breakthrough because a fair same-structure content scan using the same split payload remains slightly stronger.

## implementation

- simulation: `neuroloc/simulations/memory/local_100k_source_structure_qa_adapter.py`
- tests: `tests/test_local_100k_source_structure_qa_adapter.py`
- suite: `compression_mirror`
- hard output: `codex_local_output/suite_l100k_source_structure_qa_hard/local_100k_source_structure_qa_adapter/local_100k_source_structure_qa_adapter_metrics.json`

the source block is the same stable four-domain margin-adapter source surface. the structure transform is trained from disjoint source files, then the held-out source block is stored as compressed count and body streams with a charged `256` bit structure header.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_source_structure_qa_adapter --profile hard --output-root codex_local_output\suite_l100k_source_structure_qa_hard --timeout-sec 1200
```

result:

- suite result: pass.
- fact count: `4096`.
- source block bytes: `154873`.
- source files: `14`.
- adapter parameter count: `4`.
- exact answer success: `1.0`.
- heldout exact answer success: `1.0`.
- paraphrase-stable answer success: `1.0`.
- random-label twin success: `0.0`.
- controls collapse: `1.0`.
- transformer surface pass: `1.0`.
- recurrent surface pass: `1.0`.
- trainable recompression update success: `1.0`.
- update-controller-disabled success: `0.0`.
- model-state adapter payload used: `1.0`.
- state-dict buffer payload used: `1.0`.
- external payload store used: `0.0`.
- stored manifest used: `0.0`.
- raw source block retained: `0.0`.
- per-fact value row count: `0.0`.
- assignment row count: `0.0`.
- block payload bits: `246328`.
- committed state bits: `279136`.
- paper-surface accounted bits: `283232`.
- useful retrievable bits: `1048576`.
- adapter strict multiplier: `24.03612607449857x`.
- paper-surface strict multiplier: `23.688602733536655x`.
- raw best standard payload bits: `262336`.
- raw executable content-scan bits: `295144`.
- raw executable content-scan multiplier: `22.73766839237796x`.
- raw undercharged mph bits: `295160`.
- raw undercharged mph multiplier: `22.73643583141347x`.
- raw content scan beaten: `1.0`.
- raw undercharged mph beaten: `1.0`.
- same-structure content scan success: `1.0`.
- same-structure content scan bits: `279136`.
- same-structure content scan multiplier: `24.041637051473117x`.
- same-structure content scan beaten: `0.0`.
- source-structure train/test path overlap count: `0.0`.
- source-structure train/test hash overlap count: `0.0`.
- source-structure train/test sliding `64` byte ngram overlap count: `1512.0`.

## controls

- no-memory success: `0.0`.
- recency-only success: `0.000244140625`.
- shuffled-question success: `0.0`.
- shuffled-paraphrase success: `0.0`.
- shuffled-value success: `0.0`.
- shuffled-provenance success: `0.0`.
- wrong-question success: `0.0`.
- unanswerable-question success: `0.0`.
- read-disabled success: `0.0`.
- decoder-disabled success: `0.0`.
- parser-disabled success: `0.0`.
- adapter-disabled success: `0.0`.
- code-disabled success: `0.0`.
- wrong-query hit rate: `0.0`.
- unanswerable-query hit rate: `0.0`.
- partial-overlap query hit rate: `0.0`.
- marker-injection query hit rate: `0.0`.
- hidden fact value row detected: `0.0`.
- hidden raw source prefix detected: `0.0`.
- formula or schema labels present: `0.0`.
- seed oracle authorized: `0.0`.

## category check

implemented operation:

exact bounded qa retrieval from a source-structure compressed payload carried inside torch module state, plus trained update-controller packaging and transformer/recurrent host probes.

strongest baseline:

the fair same-structure content scan at `24.041637051473117x`. it uses the same compressed count/body payload and decoder budget without charging the four adapter parameters, so it is slightly stronger on static retrieval.

what passed:

the adapter beats the raw content-scan and raw undercharged mph diagnostics, answers exact and paraphrased questions at `1.0`, preserves provenance, reloads from model state, collapses random-label and disabled controls, and keeps the broad authorization flags at `0.0`.

what failed or remains unproved:

the adapter does not beat the fair same-structure content scan. it does not prove learned semantic recall, implicit base-weight storage, arbitrary chat, full nm behavior, high-density knowledge compression, strict 600x density, or paid-scale trainability.

why not promoted to breakthrough:

the main gain is a better charged source-code payload transform. once the content scanner is given the same transform, static retrieval remains slightly cheaper because the model also carries the trained update controller.

## decision

accept as a bounded source-structure exact qa adapter product and as evidence that the source-structure payload improves the qa surface. do not promote to static compression breakthrough or broad high-density knowledge compression.

## correction (2026-05-13)

this run card is frozen evidence. later work superseded it with [[tests/local_100k_source_subtoken_qa_adapter]], which lowers block payload bits from `246328` to `244440` and raises paper-surface strict multiplier from `23.688602733536655x` to `23.847532408460314x`. the source-structure qa adapter is no longer the current strongest bounded exact qa adapter product.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[tests/local_100k_source_structure_block_codec]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[mistakes/answer_surface_codec_static_scan_not_beaten]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
