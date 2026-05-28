# paper-ready adapter reload false positive

status: historical context only. frozen as of 2026-05-12. do not edit.

## what happened

during review of `local_100k_paper_ready_adapter_benchmark`, the first reload probes created the reload cell from the same correct source block before loading the saved state dict. that meant reload success could pass even if the adapter payload was not actually restored from saved model state.

the recompression update probe also edited a caller-supplied raw source block rather than first decoding the adapter payload from model state.

## why it mattered

the benchmark's paper-facing claim depends on persistent model-state adapter payloads and decode-edit-recompress update. a reload probe that starts with the correct payload is a false positive, and an update probe that edits caller-side raw bytes is a side-channel risk.

## fix

the reload probes now corrupt the adapter payload before loading saved state and require preload success `0.0` followed by reload success `1.0`. the update probe now derives the editable block from `decoded_adapter_block()`, edits that decoded adapter-state block, recompresses it, corrupts the reload payload, and then requires state-dict reload to restore the edited answer.

the test facts no longer expose `semantic_handle`; handles are computed from runtime questions.

## prevention

every future state-dict reload gate must prove the pre-load object fails the target task, then prove `load_state_dict` alone restores it. every update gate must derive the editable state from the model's charged payload, not from an uncharged raw side input.

## see also

- [[tests/local_100k_paper_ready_adapter_benchmark]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[mistakes/paper_ready_adapter_source_holdout_overlap]]
- [[mistakes/paper_ready_adapter_offset_lattice_mismatch]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
