# source native relation stride rule category error

status: historical context only. frozen as of 2026-05-13. do not edit.

## what happened

`local_100k_source_native_relation_adapter` attempted to leave the alias family by adding a source-native relation router. the first version used a fixed stride relation: parse anchor terms, scan to the anchor, move `7 * 32` bytes forward, and return that span.

review found that this relation was a preplanned formula, not learned semantic routing. a fair stride-aware content scan using the same parser and stride solved the hard profile at `1.0` exact success.

## why it mattered

the project rule rejects targets recoverable from deterministic rules, compact generators, schema labels, or preplanned relations. a relation benchmark is only useful if the relation comes from the source or is learned from data under held-out controls, not if the implementation creates the relation by offset arithmetic.

## fix

the simulation now reports:

- stride-aware content-scan success: `1.0`.
- fair stride content scan not beaten: `1.0`.
- legacy static public baseline pass: `1.0`.
- static public baseline pass after fair stride-aware scan: `0.0`.
- fixed relation constant used: `1.0`.
- learned relation router used: `0.0`.
- formula or schema labels present: `1.0`.
- publishable relation breakthrough candidate: `0.0`.
- strict breakthrough authorization: `0.0`.

the registry and tests now require the demoted diagnostic state rather than promotion.

## prevention

future source-native relation work must use relations already authored in the corpus, such as wiki links, file imports, definitions, headings, or documented references. if a relation rule is introduced by the benchmark, a baseline with the same rule must be considered fair and must not be counted as beaten.

## see also

- [[tests/local_100k_source_native_relation_adapter]]
- [[mistakes/semantic_alias_payload_adapter_formula_alias_category_error]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
