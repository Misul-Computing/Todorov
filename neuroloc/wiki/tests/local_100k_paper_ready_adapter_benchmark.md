# local 100k paper-ready adapter benchmark

status: current (as of 2026-05-12).

## date run

2026-05-12.

## status

passed as a paper-ready local model-state adapter benchmark candidate. it extends the weight-carried qa codec into the five paper-facing gates: transformer host integration, recurrent/state-style host integration, public baseline comparison, multi-domain source-heldout corpus slices, and paraphrase-stable exact qa plus adapter recompression.

this is not arbitrary chat, not learned semantic memory, not implicit base-weight storage, not a strict 600x result, and not broad full-nm authorization. it is a bounded exact-codec adapter benchmark with strong local evidence.

## artifact tested

- simulation: `local_100k_paper_ready_adapter_benchmark`
- hard output root: `codex_local_output/suite_l100k_paper_ready_adapter_hard`
- metrics artifact: `codex_local_output/suite_l100k_paper_ready_adapter_hard/local_100k_paper_ready_adapter_benchmark/local_100k_paper_ready_adapter_benchmark_metrics.json`

## what was done

the benchmark builds a source-heldout multi-domain corpus block from local project knowledge, wiki, compression, and code surfaces. test facts expose bounded natural-language questions, exact 32-byte answers, provenance, and domain labels, but do not expose source ids, source offsets, content digests, answer digests, assignment rows, payload rows, external payload paths, or stored manifests.

the compressed source-heldout payload is carried inside torch module state as an adapter buffer. two tiny host modules wrap that adapter state: a transformer-style host and a recurrent/state-style host. both hosts keep the adapter payload in `state_dict`, answer paraphrased questions exactly through the adapter surface, and preserve behavior after save/load.

the parser accepts multiple question phrasings over the same evidence terms without storing paraphrase rows. the update probe decodes the adapter payload, changes one answer in the source block, recompresses the payload, writes it back into model state, and confirms the edited answer plus provenance after reload.

## key hard outputs

- fact count: `4096`
- train fact count: `2048`
- source domains: `4`
- source files: `8`
- source block bytes: `145992`
- candidate scan count: `4559`
- adapter parameter count: `0`
- maximum host parameter count: `6592`
- useful retrievable bits: `1048576`
- block payload bits: `370448`
- model header bits: `40`
- fixed parser and decoder bits: `32768`
- paraphrase parser contract bits: `4096`
- committed state bits: `403256`
- paper-surface accounted bits: `407352`
- exact answer success: `1.0`
- heldout exact answer success: `1.0`
- paraphrase-stable answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- transformer surface pass: `1.0`
- recurrent surface pass: `1.0`
- public baseline stack pass: `1.0`
- multi-domain pass: `1.0`
- paraphrase or update pass: `1.0`
- paper-ready requirement count: `5.0`
- adapter recompression update success: `1.0`
- adapter state-dict preload success: `0.0`
- adapter state-dict reload success: `1.0`
- transformer state-dict preload success: `0.0`
- transformer state-dict reload success: `1.0`
- recurrent state-dict preload success: `0.0`
- recurrent state-dict reload success: `1.0`
- adapter strict multiplier: `16.641752137599937`
- paper-surface strict multiplier: `16.474416229698146`
- strongest public baseline multiplier in the local stack: `16.64109186851554`
- beats weight-carried baseline: `1.0`
- beats minimal-perfect-hash payload baseline: `1.0`
- beats product-key memory baseline: `1.0`
- beats memory-layer baseline: `1.0`
- beats sparse-read baseline: `1.0`
- beats lora-style storage baseline: `1.0`
- beats qlora-style storage baseline: `1.0`
- beats model-editing storage baseline: `1.0`
- strict 600x pass: `0.0`
- learned semantic retrieval authorization: `0.0`
- arbitrary chat authorization: `0.0`
- strict breakthrough authorization: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_paper_ready_adapter_benchmark.py -q`
- `python -m pytest tests/test_local_100k_paper_ready_adapter_benchmark.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_paper_ready_adapter_benchmark --profile smoke --timeout-sec 300`
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_paper_ready_adapter_benchmark --profile hard --output-root codex_local_output\suite_l100k_paper_ready_adapter_hard --timeout-sec 1200`
- `python -m py_compile neuroloc\simulations\memory\local_100k_paper_ready_adapter_benchmark.py neuroloc\simulations\suite_registry.py tests\test_local_100k_paper_ready_adapter_benchmark.py`

## category check

implemented operation: bounded exact qa retrieval from a compressed adapter payload carried inside torch model state, with transformer and recurrent/state-style host integration, paraphrase-stable handle parsing, exact provenance, and recompression update.

strongest baseline: the local minimal-perfect-hash payload line at `16.64109186851554x` under the same compact multi-domain block accounting, plus the previous weight-carried qa product at `15.215221373768888x`.

what passed: exact answer success is `1.0`, paraphrase-stable answer success is `1.0`, controls collapse is `1.0`, transformer and recurrent host surfaces pass, adapter update and reload pass, public baseline stack pass is `1.0`, and the adapter strict multiplier reaches `16.641752137599937x`.

what failed or remains weaker: strict 600x pass remains `0.0`, learned semantic retrieval authorization remains `0.0`, base-weight implicit storage authorization remains `0.0`, and strict breakthrough authorization remains `0.0`.

what is not proved: learned semantic memory, arbitrary opaque-key associative retrieval, facts stored inside ordinary base-model weights, arbitrary chat, full neural-model completion, paid-scale trainability, biological neuron-density proof, or external simulator transfer.

why this is not promoted further: the benchmark still relies on token-signature evidence terms and a compressed adapter buffer. the new result is stronger because it proves a host-integrated, paraphrase-stable, multi-domain, baseline-beating adapter benchmark, but it does not yet show that a base transformer or recurrent model internalizes the facts through ordinary weight training.

## verdict

accepted as the next presentable compression product and paper-ready local evidence package. compared with the prior `15.215221373768888x` weight-carried qa product, the new hard result raises strict adapter density to `16.641752137599937x`, adds transformer and recurrent/state host integration, adds paraphrase-stable exact qa, tests recompression update and reload, and beats the local public-baseline stack under exact-answer accounting.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_weight_carried_qa_codec]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[tests/local_100k_llm_semantic_qa_codec]]
- [[tests/local_100k_content_addressed_source_codec]]
- [[tests/local_100k_source_block_codec]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/neural_model_related_work_pressure_matrix]]
- [[mistakes/paper_ready_adapter_source_holdout_overlap]]
- [[mistakes/paper_ready_adapter_offset_lattice_mismatch]]
- [[mistakes/paper_ready_adapter_reload_false_positive]]
