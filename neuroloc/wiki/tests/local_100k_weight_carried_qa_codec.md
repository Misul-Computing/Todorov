# local 100k weight-carried qa codec

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a tested weight-carried exact-qa codec product. it moves the bounded llm-facing qa surface from an external compressed source-block artifact into a charged model-state adapter payload, removes the stored manifest, cuts the fixed parser and decoder budget to `32768` bits, and raises strict density above the requested `15x` line.

this is not arbitrary chat, not learned semantic memory, not implicit base-weight storage, not a high-density breakthrough, and not the 600x target.

## artifact tested

- simulation: `local_100k_weight_carried_qa_codec`
- hard output root: `codex_local_output/suite_l100k_weight_carried_qa_hard`
- metrics artifact: `codex_local_output/suite_l100k_weight_carried_qa_hard/local_100k_weight_carried_qa_codec/local_100k_weight_carried_qa_codec_metrics.json`

## what was done

the simulation builds bounded natural-language questions from source-heldout corpus anchors. stored test facts do not expose source ids, offsets, content digests, answer digests, assignment keys, payload rows, external payload paths, or stored manifest rows.

the compressed byte stream is registered inside a torch module state dict as an adapter payload buffer plus a small charged model header. the read path recovers the adapter payload from model state, decompresses it, scans candidate anchor windows, computes the token-signature handle from the runtime question, and returns the exact following 32-byte answer plus provenance.

the adapter records an explicit recompression update path for later finetuning or post-training integration: changed knowledge must be decoded from the adapter stream, updated, recompressed, and written back into model state. this is the intended bridge for transformer, mamba, rwkv, and related sequence-model use. it is still a buffer-backed adapter payload, not proof that ordinary base weights or individual neurons implicitly store the facts.

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- parameter count: `0`
- source block bytes: `173730`
- candidate scan count: `5428`
- useful retrievable bits: `1048576`
- block payload bits: `408256`
- model header bits: `40`
- fixed parser and decoder bits: `32768`
- manifest bits: `0`
- committed state bits: `441064`
- strict accounted bits: `441064`
- exact answer success: `1.0`
- heldout exact answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- selected semantic collision count: `0.0`
- ambiguous match count: `0.0`
- model-state adapter payload used: `1.0`
- state-dict buffer payload used: `1.0`
- external payload store used: `0.0`
- stored manifest used: `0.0`
- adapter recompression update path: `1.0`
- adapter recompression update success: `1.0`
- adapter state-dict reload success: `1.0`
- true base-weight implicit storage authorization: `0.0`
- strict multiplier: `15.215221373768888`
- llm-facing qa baseline multiplier: `14.06935717190861`
- mph payload baseline multiplier: `15.214669447719235`
- beats llm-facing qa baseline: `1.0`
- beats mph payload baseline: `1.0`
- strict 600x pass: `0.0`
- arbitrary chat authorization: `0.0`
- breakthrough authorization: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_weight_carried_qa_codec.py -q`
- `python -m pytest tests/test_local_100k_weight_carried_qa_codec.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_weight_carried_qa_codec --profile smoke --timeout-sec 300`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_weight_carried_qa_codec --profile hard --output-root codex_local_output\suite_l100k_weight_carried_qa_hard --timeout-sec 1200`

## category check

implemented operation: exact bounded natural-language qa retrieval from a compressed model-state adapter payload.

strongest baseline: the prior bounded llm-facing qa codec at strict multiplier `14.06935717190861`, plus the minimal-perfect-hash payload line at `15.214669447719235` under this no-manifest accounting.

what passed: exact answer success is `1.0`, random-label twin success is `0.0`, controls collapse is `1.0`, external payload store is `0.0`, stored manifest is `0.0`, assignment row count is `0.0`, per-fact value row count is `0.0`, adapter recompression update success is `1.0`, adapter state-dict reload success is `1.0`, strict multiplier is `15.215221373768888`, and the product beats the current llm-facing qa and mph payload baselines.

what failed or remains weaker: strict 600x pass remains `0.0`, breakthrough authorization remains `0.0`, learned semantic retrieval authorization remains `0.0`, and base-weight implicit storage authorization remains `0.0`.

what is not proved: high-density knowledge compression, learned semantic recall, arbitrary opaque-key associative retrieval, a chat model, a full neural model, a 600x neuron-cell, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is not promoted further: the query still reduces to a lexical token-signature handle. the product improves deployment shape and density by carrying the payload in model state and eliminating stored manifest cost, but it does not yet train the base model to internalize the knowledge.

## verdict

accepted as the next presentable compression product. it crosses the requested `15x` strict multiplier line, beats the previous llm-facing product, and beats the matched minimal-perfect-hash payload line by a small margin under the same no-manifest accounting. the next real step is paraphrase-stable learned question handling or a true trainable adapter update that preserves this density without adding hidden rows.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_llm_semantic_qa_codec]]
- [[tests/local_100k_content_addressed_source_codec]]
- [[tests/local_100k_source_block_codec]]
- [[tests/local_100k_paper_ready_adapter_benchmark]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
