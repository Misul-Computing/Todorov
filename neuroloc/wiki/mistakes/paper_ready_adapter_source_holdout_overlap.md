# paper-ready adapter source holdout overlap

status: historical context only. frozen as of 2026-05-13. do not edit.

## what happened

review of `local_100k_paper_ready_adapter_benchmark` found that `neuroloc/wiki/synthesis/timescale_separation.md` appeared in both the test source list and the train source list.

the benchmark still retrieved from the compressed test payload, and the prior metrics remain useful as a bounded adapter result, but the self-reported `source_holdout_used = 1.0` was too weak. it did not prove path-disjoint source selection.

## why it mattered

any learned semantic, learned dictionary, learned codec, or learned update result can leak if train and test sources share a path, hash, or long byte sequence. a high-density or paper-facing compression claim cannot rely on a self-reported holdout flag.

## fix

`local_100k_margin_recompression_adapter` removes train source files from the source compression path and adds explicit holdout metrics:

- source train/test path overlap count: `0.0`.
- source train/test hash overlap count: `0.0`.
- source train/test n-gram overlap count: `0.0`.
- source holdout pass: `1.0`.

the new test file also checks that source paths and hashes are unique on the test surface.

## prevention

future compression benchmarks must report concrete overlap counts, not only a boolean holdout flag. any benchmark that uses learned train-source information must gate path overlap, hash overlap, sampled answer overlap, and long n-gram overlap before it can claim source-heldout evidence.

## see also

- [[tests/local_100k_paper_ready_adapter_benchmark]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[mistakes/paper_ready_adapter_reload_false_positive]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
