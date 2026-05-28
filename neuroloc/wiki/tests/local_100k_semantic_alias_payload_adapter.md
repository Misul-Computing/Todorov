# local 100k semantic alias payload adapter

status: historical context only. frozen as of 2026-05-13. do not edit.

## summary

`local_100k_semantic_alias_payload_adapter` is a diagnostic successor to `local_100k_margin_recompression_adapter`.

it adds a reversible payload transform that reduces the compressed model-state payload, replaces visible token-signature questions with generated alias questions, and keeps exact retrieval from the compressed adapter payload inside torch module state.

the implementation passes as an engineering diagnostic only. it is not a publishable breakthrough because the alias question surface is generated from source anchors, and a fair same-interface alias content-scan baseline retrieves the same answers exactly.

## implemented operation

- exact bounded qa retrieval from a transformed compressed source-heldout payload carried inside torch module state.
- transformer and recurrent/state host wrappers that reload the adapter payload through `state_dict`.
- generated alias questions that hide lexical token signatures from the old parser but remain recoverable by the same alias function.
- fair alias content-scan diagnostic over the same decoded block.
- explicit demotion metrics for formula or schema labels, fair scan not beaten, publishable breakthrough authorization, and strict breakthrough authorization.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_semantic_alias_payload_adapter --profile hard --output-root codex_local_output\suite_l100k_semantic_alias_hard --timeout-sec 1200
```

result:

- suite result: pass.
- fact count: `4096`.
- source files: `14`.
- source domains: `4`.
- source block bytes: `154873`.
- maximum host parameter count: `6592`.
- adapter parameter count: `0`.
- exact answer success: `1.0`.
- heldout exact answer success: `1.0`.
- random-label twin success: `0.0`.
- controls collapse: `1.0`.
- wrong query hit rate: `0.0`.
- unanswerable query hit rate: `0.0`.
- partial-overlap query hit rate: `0.0`.
- marker-injection query hit rate: `0.0`.
- lexical content-scan success: `0.0`.
- alias content-scan success: `1.0`.
- transformer surface pass: `1.0`.
- recurrent surface pass: `1.0`.
- source holdout pass: `1.0`.
- block payload bits: `258144`.
- committed state bits: `290952`.
- paper-surface accounted bits: `295048`.
- useful retrievable bits: `1048576`.
- adapter strict multiplier: `23.06526987269378x`.
- paper-surface strict multiplier: `22.7450665654402x`.
- strongest static baseline multiplier after fair alias scan: `23.06526987269378x`.
- semantic alias diagnostic candidate: `1.0`.
- publishable breakthrough candidate: `0.0`.
- strict breakthrough authorization: `0.0`.
- formula or schema labels present: `1.0`.
- strict 600x pass: `0.0`.
- arbitrary chat authorization: `0.0`.

## category check

implemented operation: a transformed compressed model-state exact-qa adapter with generated alias handles and host integration.

strongest baseline: the fair same-interface alias content-scan baseline at `23.06526987269378x`.

what passed: payload bits dropped below the margin adapter line, exact retrieval remained `1.0`, disabled controls collapsed, false-hit rates remained `0.0`, host wrappers reloaded payload state, and the old lexical parser could not solve the alias questions.

what failed or remains unproved: the alias labels are generated from source anchors, the fair alias content scan solves the same task exactly, the static public baseline gate is `0.0`, learned semantic retrieval is not authorized, and the strict breakthrough claim remains blocked.

why not promoted to breakthrough: the apparent win came from changing the query interface. once the baseline receives the same alias parser, it matches the adapter and wins on the same static-retrieval operation. this is a diagnostic for parser-mismatch risk, not a new compression result.

## decision

accepted only as a demoted diagnostic. the current bounded compression product remains `local_100k_margin_recompression_adapter` until a candidate beats the fair same-block scan or proves a different operation under matched baselines.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[mistakes/semantic_alias_payload_adapter_formula_alias_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
