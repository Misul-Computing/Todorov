# semantic alias payload adapter formula alias category error

status: historical context only. frozen as of 2026-05-13. do not edit.

## what happened

`local_100k_semantic_alias_payload_adapter` reduced the compressed payload and made the old lexical content-scan parser fail by replacing visible token-signature questions with generated alias questions.

review found that the alias questions were deterministically generated from source-anchor tokens. a fair content-scan baseline using the same alias parser retrieved every answer exactly.

## why it mattered

the high-density knowledge gate forbids formula-generated labels, planned schemas, hidden side channels, and unfair parser mismatches. a baseline that scans the same decoded block must receive the same question interface before a candidate can claim it was beaten.

the first version risked promoting a parser mismatch as a compression breakthrough. that would repeat the structured-target mistake: the task would be easier because the benchmark surface was designed by the implementation, not because the model learned a better compression mechanism.

## fix

the simulation now reports:

- alias content-scan success: `1.0`.
- fair alias content scan not beaten: `1.0`.
- formula or schema labels present: `1.0`.
- static public baseline pass: `0.0`.
- publishable breakthrough candidate: `0.0`.
- strict breakthrough authorization: `0.0`.

the registry requires the diagnostic state rather than promotion. the run card records the payload reduction and exact retrieval result as engineering evidence only.

## prevention

future compression attempts must include a fair same-interface scan baseline whenever the query parser, handle, alias, paraphrase surface, or source-addressing rule changes.

generated questions, generated aliases, deterministic guards, and schema-shaped labels must be charged or marked as formula labels. they cannot support a breakthrough claim unless held-out controls prove the structure was learned from data and the matched scanner still loses.

## see also

- [[tests/local_100k_semantic_alias_payload_adapter]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
