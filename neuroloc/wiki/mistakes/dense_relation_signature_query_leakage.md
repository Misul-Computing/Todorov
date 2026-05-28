# dense relation signature query leakage

status: historical context only. frozen as of 2026-05-14. do not edit.

## mistake

`local_100k_source_dense_authored_relation_diagnostic` initially included a `definition_signature` relation whose question contained the exact signature and whose answer was the same signature.

that subset was not valid reconstruction evidence. the answer could be copied from the question surface, so the random-label and decoder-disabled controls were not sufficient to prove that those facts came from the charged compressed source state.

## why it matters

the project rule forbids template leakage and answer-contained queries. a diagnostic may know the task interface, but it cannot count facts where the query already contains the target answer.

the issue did not promote the artifact to a product, because product and breakthrough authorization were already fixed at `0.0`. it still inflated the relation count, useful retrievable bits, honest mph margin, and strict multiplier for the dense relation diagnostic.

## correction

the `definition_signature` relation was removed from the diagnostic. the hard relation count drops from `4055` to `3741`, the honest mph relation index bits drop from `3646856` to `3366080`, the margin over honest mph/index drops from `3209176` to `2928400`, useful retrievable bits drop from `2860104` to `2639616`, and strict multiplier drops from `41.8220288795467` to `38.59793090842625`.

the diagnostic still answers the remaining source-authored relations exactly after `state_dict` reload and still beats honest mph/index, but it remains a diagnostic because paq8px level 2 is cheaper by `176536` bits and the fair unlimited relation-aware scanner still solves the task.

## prevention

future relation surfaces must include an answer-not-in-query check. if a relation asks for a value that is included verbatim in the query, the fact must be removed or the query must be redesigned before metrics are counted.

## see also

- [[tests/local_100k_source_dense_authored_relation_diagnostic]]
- [[mistakes/public_context_mixing_baseline_missing]]
- [[mistakes/source_native_relation_stride_rule_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
