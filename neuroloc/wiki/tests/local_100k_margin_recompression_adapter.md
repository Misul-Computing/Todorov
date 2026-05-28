# local 100k margin recompression adapter

status: historical context only. frozen as of 2026-05-13. do not edit.

## summary

`local_100k_margin_recompression_adapter` is the next bounded model-state adapter engineering product after `local_100k_paper_ready_adapter_benchmark`.

it keeps the exact source-heldout qa adapter shape, but repairs the source-holdout flaw found in the prior benchmark, uses a smaller stable four-domain source block, adds explicit false-hit controls, and attaches a tiny trained recompression update controller to the adapter module state.

this is a high-margin local adapter/update engineering product, not a paper-ready static compression claim and not a strict public breakthrough. it still uses lexical token-signature routing for exact retrieval, and an executable same-block content-scan diagnostic remains at least as strong on static retrieval.

## implemented operation

- exact bounded qa retrieval from a compressed payload carried inside torch module state.
- transformer and recurrent/state host wrappers that carry the adapter payload and update controller in `state_dict`.
- a trained four-parameter recompression controller that gates decode-edit-recompress updates from decoded adapter state.
- source-heldout stable source selection with no train source files, no path overlap, no hash overlap, and no train/test n-gram overlap by construction.
- invalid-query false-hit controls for wrong, unanswerable, partial-overlap, and marker-injection queries.
- hostile reload checks requiring preload failure and state-dict reload success.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_margin_recompression_adapter --profile hard --output-root codex_local_output\suite_l100k_margin_recompression_hard --timeout-sec 1200
```

result:

- suite result: pass.
- fact count: `4096`.
- train fact count: `0`.
- source files: `14`.
- source domains: `4`.
- source block bytes: `154873`.
- maximum host parameter count: `6596`.
- adapter parameter count: `4`.
- exact answer success: `1.0`.
- paraphrase-stable answer success: `1.0`.
- random-label twin success: `0.0`.
- controls collapse: `1.0`.
- wrong query hit rate: `0.0`.
- unanswerable query hit rate: `0.0`.
- partial-overlap query hit rate: `0.0`.
- marker-injection query hit rate: `0.0`.
- transformer surface pass: `1.0`.
- recurrent surface pass: `1.0`.
- source holdout pass: `1.0`.
- large-margin-over-mph pass: `1.0`.
- static public baseline pass: `1.0`.
- bounded adapter engineering pass: `1.0`.
- paper-ready candidate: `0.0`.
- static compression publishable candidate: `0.0`.
- trainable recompression update success: `1.0`.
- update-controller-disabled success: `0.0`.
- adapter state-dict preload success: `0.0`.
- adapter state-dict reload success: `1.0`.
- block payload bits: `262336`.
- committed state bits: `295144`.
- paper-surface accounted bits: `299240`.
- useful retrievable bits: `1048576`.
- adapter strict multiplier: `22.732738950163952x`.
- paper-surface strict multiplier: `22.421639537059313x`.
- strongest prior static public baseline multiplier: `16.641752137599937x`.
- executable same-block content-scan multiplier: `22.73766839237796x`.
- same-block undercharged mph multiplier: `22.73643583141347x`.
- content-scan not beaten: `1.0`.
- same-block undercharged mph not beaten: `1.0`.
- strict 600x pass: `0.0`.
- strict breakthrough authorization: `0.0`.
- learned semantic retrieval authorization: `0.0`.
- arbitrary chat authorization: `0.0`.

## category check

implemented operation: a compressed model-state exact-qa adapter with trained update gating and host integration.

strongest baseline: the executable same-block content-scan diagnostic at `22.73766839237796x`, which is slightly stronger than the adapter on static retrieval because the adapter charges four trained update-controller parameters.

what passed: the product beats the previous adapter line by a large margin on paper-surface accounting, passes exact retrieval, host integration, source-holdout, false-hit, update, reload, and disabled-path controls, and keeps random-label success at `0.0`.

what failed or remains unproved: it does not beat the executable same-block content-scan diagnostic, does not beat the same-block undercharged mph diagnostic, does not prove learned semantic retrieval, does not prove implicit base-weight storage, and does not approach the strict 600x target.

why not promoted to breakthrough: the win comes from a smaller stable source block plus adapter/update hardening, while static retrieval is still content-scan over a compressed payload. the generic paper-ready authorization flag is fixed at `0.0` until static retrieval beats the fair content-scan and undercharged mph diagnostics, and the update path gets a matched update baseline.

## decision

accepted as the next bounded local adapter/update engineering product and the new product line to beat: `22.421639537059313x` paper-surface multiplier. paper-ready candidate and strict breakthrough authorization both remain `0.0`.

the next real breakthrough attempt must beat the executable same-block content-scan diagnostic or replace lexical token-signature routing with learned semantic retrieval under the same false-hit and source-holdout controls.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_paper_ready_adapter_benchmark]]
- [[mistakes/paper_ready_adapter_source_holdout_overlap]]
- [[mistakes/paper_ready_adapter_reload_false_positive]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
