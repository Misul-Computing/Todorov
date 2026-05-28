# source-block codec raw cache category error

status: current (as of 2026-05-12).

## mistake

an earlier `local_100k_source_block_codec` implementation compressed `source_block` and charged only the compressed block bits, but also retained a raw decoded source block cache for reads.

that made the reported compression accounting false. the product could claim to read from a compact block stream while the actual read path still had access to the uncompressed source bytes.

## why it mattered

this was a hidden side channel. if raw decoded source bytes remain available after compression, then exact retrieval may be carried by the uncharged cache rather than by the charged compressed representation.

the category error is the same class as an uncharged residual table or hidden lookup row. it can turn an honest source-block codec into a false compression result, because the retrieval path is not forced to pay for the information it uses.

## detection

the accounting problem was detected by checking whether the cell retained or read from a decoded source block after compression.

the repaired tests now assert that the product has no `decoded_block` attribute, that `decompression_count` increments when a read occurs, and that the summary reports `raw_source_block_retained=0.0`, `reads_from_compressed_block=1.0`, and `raw_source_block_bits_charged=0.0`.

## fix

the fixed product removes raw decoded-block retention. it stores the compressed `block_stream` and reads by decompressing that stream on demand.

the metric contract now states that raw source block retention is absent, reads come from the compressed block path, and no raw source block bits are charged because no raw decoded source block is retained.

## verification

verified surfaces:

- `tests/test_local_100k_source_block_codec.py` asserts `raw_source_block_retained == 0.0`.
- `tests/test_local_100k_source_block_codec.py` asserts `reads_from_compressed_block == 1.0`.
- `tests/test_local_100k_source_block_codec.py` asserts the cell has no `decoded_block`.
- `tests/test_local_100k_source_block_codec.py` asserts `decompression_count` increases after a read.
- `neuroloc/wiki/tests/local_100k_source_block_codec.md` records `raw source block retained: 0.0`, `reads from compressed block: 1.0`, and `raw source block bits charged: 0.0`.

## prevention

future source-codec and block-codec products must fail closed unless the retrieval path is audited for raw payload retention.

the required checks are:

- no decoded raw source block remains as a readable cache after compression.
- reads must exercise the charged compressed stream or a charged decoder state.
- any retained raw payload, residual row, decoded cache, seed oracle, manifest payload, or reconstruction table must be charged in strict bits.
- tests must include a structural no-cache assertion, not just success metrics.
- telemetry must separately report raw payload retained, compressed-path read usage, and raw payload bits charged.

## see also

- [[tests/local_100k_source_block_codec]]
- [[tests/local_100k_shared_predictor_exact_codec]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[mistakes/learned_unknown_structure_residual_table_defeat]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[PROJECT_PLAN]]
