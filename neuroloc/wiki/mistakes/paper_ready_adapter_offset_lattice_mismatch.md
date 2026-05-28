# paper-ready adapter offset lattice mismatch

status: historical context only. frozen as of 2026-05-12. do not edit.

## what happened

during implementation of `local_100k_paper_ready_adapter_benchmark`, the first multi-domain sampler selected offsets by each source file's local `anchor_bytes + n * chunk_bytes` lattice. the read path scanned the packed model-state block by the global `anchor_bytes + n * chunk_bytes` lattice.

because two-byte separators shift later source files off the global lattice, most sampled answers were never visited by the read path. the first smoke probe reached exact answer success `0.083984375`. after switching sampling to the same global packed-block candidate lattice, exact answer success improved but remained below target because the parser removed legitimate evidence tokens. after fixing the parser to preserve the same token signature used by the stored anchors, exact answer success and paraphrase-stable answer success reached `1.0`.

## why it mattered

this was a real implementation bug in the benchmark path. it could have produced a false negative or encouraged adding an uncharged routing map to recover the missing offsets. the correct fix was to align the generator and reader to the same packed-block offset lattice, not to add a hidden table.

## prevention

future compressed-block simulations must sample target offsets from the same candidate-offset function used by the read path, or must explicitly charge and test any separate offset map. paraphrase parsers must preserve the exact evidence-token signature used by the stored anchors unless the change is tested as a learned semantic handle.

## see also

- [[tests/local_100k_paper_ready_adapter_benchmark]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
