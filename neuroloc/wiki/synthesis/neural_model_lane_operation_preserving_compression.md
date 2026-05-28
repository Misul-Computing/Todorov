# neural model lane: operation-preserving compression

status: current (as of 2026-05-14).

## thesis

compression is accepted only when it preserves the operation the memory object must support. the target is not tensor reconstruction by itself. the target is lower committed bits while preserving routing, recall, reconstruction, replay rewrite, imagined-branch use, world-state update, and action success.

the useful claim is conditional: an operation-preserving compression stack may be novel if the project proves replaceable codecs across memory surfaces under oracle bounds, trainable controls, telemetry, and related-work separation.

## ranked unknowns

1. what oracle compression ratios are possible on the hard symbolic worlds.
2. which task families can support 10x, 100x, or higher useful compression without discarding task state.
3. which compression objects matter first: addresses, payloads, episodes, imagined branches, replay rewrites, or world state.
4. how to define task-relative rate-distortion when the preserved output is an action, not a reconstruction.
5. where the prior-art boundary lies between known vector quantization, latent compression, schema memory, replay, and the project's compound interface.
6. when content-routed sparse read over verbatim context is already sufficient, and when compressed memory objects add useful capability beyond it.

## evidence base

- [[cellular_state_storage_gap_map]] proposes ranked local-state candidates that may affect write frequency, active forgetting, and useful bits per episode.
- [[neural_model_dossier_eligibility_gated_local_commit]] defines the first candidate whose compression claim depends on fewer committed bits at equal task success.
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]] defines the first mechanism-specific bit fields and storage policies for oracle compression.
- [[tests/eligibility_gated_local_commit_test_material]] documents the implemented mechanism-specific symbolic/oracle surface for those counters.
- [[neural_model_compression_stack]] defines the current stack-wide compression contract.
- [[oracle_compression_analysis_plan]] records the oracle-bound analysis sequence and the handoff from completed counters to the repaired source-observability contract and current learned-generalization repair.
- [[tests/oracle_compression_analysis_results]] documents the first oracle-bound result: clean controls, no leakage, eight strong families, and six weak families below 10x.
- [[oracle_compression_frontier_split]] separates accepted frontier families from weak frontier families and ranks the next narrow learned-codec proof-package candidates.
- [[neural_model_dossier_compression_under_bit_budget_codec]] defines the first narrow learned-codec proof package before implementation.
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]] defines the local implementation contract for the first tiny learned-codec mirror.
- [[tests/compression_under_bit_budget_mirror]] documents the first local dataset, guard, baseline, learned-codec, source-diagnostic, bit-accounting, sparse-read, matched-budget sparse-read, distributed-evidence, tiny local learned model, factor-heldout gate, factorized structured local codec, 10k-scale local data-heavy gate, and telemetry surface for that mirror. [[tests/language_grounded_state_density_mirror]] documents the first constrained generated-language bridge, the parser-resistant learned token-count negative gate, the parser-supported event-binding foundation, and the typed trainable event-binding local-state pass. [[tests/local_v1_language_model]], [[tests/local_10k_chat_model]], and [[tests/local_foundation_neural_model]] are demoted scaffolds after [[mistakes/local_foundation_lookup_scaffold_category_error]] and do not count as compression evidence. the source-observability contract is repaired, the visible-source codec solves the smoke task, content-routed sparse read also solves it from two legal observation records at 40 committed bits, matched-budget sparse read fails at 20 bits, and the original learned-codec smoke result remains negative because it does not beat sparse read on held-out operation success. the original source-pair mirror is demoted as compression evidence. the shared nonlinear factor-heldout model fails at 0.03125 test joint success, but the 9,792-parameter factorized structured local codec reaches minimum test joint success 1.0 across four heldout axes and three smoke seeds at 19 bits. the parser-resistant learned token-count gate then fails with minimum learned joint/state success 0.0 after stable prefixes are removed and the path is forced through bounded local state. the typed trainable segment binder reaches minimum test joint/state/action success 0.9583333333333334 with 8,856 parameters on the same local gate. this is useful-state-density evidence on constrained symbolic-message surfaces, not a full compression result.
- [[content_routed_sparse_read_prior]] records the 2026 subq / selective sparse attention prior-art boundary. it validates content-dependent routing and functional context as load-bearing problems, but it does not prove memory-object compression, replay compression, imagined-branch compression, or local-neuron storage. it adds a required baseline family for future compression claims.
- [[neural_model_related_work_pressure_matrix]] consolidates the six-lane prior-art pass and sharpens the operation-preserving compression boundary: task-relative compression, latent/context compression, vector/cache compression, and schema/provenance engineering are all prior-art-covered pieces; the remaining project edge is a learned memory-object code that preserves operations under exact controls.
- [[tests/local_100k_full_nm]] documents the first local full small nm candidate with one trainable module and learned binary compression that improves useful density over the 51-bit exact-state 3d baseline under 44-bit accounting.
- [[high_density_neuron_cell_related_work_pressure_matrix]] and [[tests/local_100k_high_density_cell]] document the first high-density neuron-cell side proof. it clears params-only 600x density on exact associative facts, but strict params-plus-committed-state density reaches only 6.1709981167608285x and therefore rejects the breakthrough claim.
- [[schema_density_cell_boundary]], [[tests/local_100k_schema_density_cell]], and [[mistakes/schema_density_cell_structured_target_category_error]] document a demoted boundary artifact: structured schema-generated associative facts clear a formula-compression gate, but that does not satisfy the high-density knowledge-compression target.
- [[tests/local_100k_unstructured_density_cell]] and [[tests/local_100k_unknown_structure_density_probe]] split the post-error compression target. independent random-label exact facts hit the entropy wall, while non-generated local corpus chunks are exactly recoverable through charged standard corpus compression but only reach a 13.941917871967359x strict multiplier, not the required 600x.
- [[tests/local_100k_learned_unknown_structure_density_cell]] tests the learned dictionary/residual version of the unknown-structure target. exact heldout retrieval reaches 1.0 and cross-label random scoring is 0.0, but a separately built random-label twin stores at 1.0, hard strict multiplier is only 3.0525410753623334x, and the path depends on per-fact residual rows, so [[mistakes/learned_unknown_structure_residual_table_defeat]] blocks promotion.
- `neuroloc/compression/` is the dedicated compression research corpus for the next attempt. it records the full in-repo attempt history, modern memory/compression pressure, open-source and passion-project baselines, theory limits, required baseline stack, and the first non-row shared-predictor exact codec product spec.
- [[tests/local_100k_shared_predictor_exact_codec]] implements the first opaque-key exact-codec product. it removes per-fact value slices, but hard strict multiplier is only 4.352614012398447x and it loses to charged corpus-codec and mph-payload baselines.
- [[tests/local_100k_source_block_codec]] implements the first stronger source/offset exact-codec product. it retrieves 4,096 source-heldout chunks exactly from one compressed source block, collapses the random-label twin to 0.0, and reaches strict multiplier 14.06876726917481x, beating the prior 13.941917871967359x charged corpus-codec baseline on this bounded source/offset target.
- [[tests/local_100k_content_addressed_source_codec]] implements the first content-addressed source-codec product. it removes source/offset query fields and assignment rows, retrieves 4,096 source-heldout chunks exactly through 16-bit content digest handles, and reaches strict multiplier 14.06888524576417x, slightly beating the source-block product while remaining a privileged content-handle codec.
- [[tests/local_100k_llm_semantic_qa_codec]] implements the first bounded llm-facing source-codec product. it uses natural-language questions over token-signature evidence, removes source/offset fields and content-digest target keys, retrieves 4,096 source-heldout answers exactly from one compressed source block, collapses random-label and disabled controls, and reaches strict multiplier 14.06935717190861x. it slightly beats the content-addressed product but remains a lexical token-signature codec and does not beat the minimal-perfect-hash payload line.
- [[tests/local_100k_weight_carried_qa_codec]] implements the first weight-carried bounded qa codec product. it stores the compressed source-heldout qa payload inside torch module state, removes the stored manifest, charges a 40-bit model header and 32,768-bit parser/decoder budget, retrieves 4,096 answers exactly, collapses random-label and disabled controls, and reaches strict multiplier 15.215221373768888x. it beats the prior llm-facing product and the matched no-manifest minimal-perfect-hash payload line, but remains a token-signature exact codec rather than learned semantic recall or implicit base-weight storage.
- [[tests/local_100k_paper_ready_adapter_benchmark]] implements the first paper-facing local model-state adapter benchmark. it keeps the compressed adapter payload inside torch module state, adds transformer and recurrent/state-style host modules, uses four source domains, accepts multiple paraphrase templates over the same evidence terms without storing paraphrase rows, tests decode-edit-recompress update and reload, and compares against lora-style storage, qlora-style storage, model-editing storage, product-key memory, memory-layer, sparse-read, codec-index, minimal-perfect-hash payload, and the previous weight-carried product. hard validation reaches exact answer success 1.0, paraphrase-stable answer success 1.0, controls collapse 1.0, adapter strict multiplier 16.641752137599937x, paper-surface strict multiplier 16.474416229698146x, and paper-ready requirement count 5.0. strict breakthrough authorization stays 0.0 because the path is still a token-signature exact adapter codec, not learned semantic memory or implicit base-weight storage.
- [[tests/local_100k_margin_recompression_adapter]] implements the high-margin model-state adapter/update product. it repairs the prior source-holdout overlap, uses a stable four-domain source block with no train-source files, reports concrete path/hash/ngram holdout counts at 0.0, keeps transformer and recurrent/state-style host surfaces, carries a trained four-parameter recompression update controller in model state, rejects wrong, unanswerable, partial-overlap, and marker-injection queries, and reaches adapter strict multiplier 22.732738950163952x plus paper-surface strict multiplier 22.421639537059313x. strict breakthrough authorization stays 0.0 because the executable same-block content-scan diagnostic reaches 22.73766839237796x and the route is still token-signature based.
- [[tests/local_100k_source_structure_qa_adapter]] implements the superseded bounded exact qa adapter product. it replaces the raw compressed source payload with source-structure count/body streams carried inside torch module state, keeps transformer/recurrent hosts and trained recompression update, retrieves 4,096 answers exactly, beats raw content scan and raw undercharged mph diagnostics, and reaches adapter strict multiplier 24.03612607449857x plus paper-surface strict multiplier 23.688602733536655x. strict breakthrough authorization stays 0.0 because the fair same-structure content scan reaches 24.041637051473117x.
- [[tests/local_100k_source_subtoken_qa_adapter]] implements the current bounded exact qa adapter product. it carries source-subtoken count/body/dictionary streams inside torch module state, keeps transformer/recurrent hosts and trained recompression update, retrieves 4,096 answers exactly, beats raw content scan and raw undercharged mph diagnostics, and reaches adapter strict multiplier 24.19976921301639x plus paper-surface strict multiplier 23.847532408460314x. strict breakthrough authorization stays 0.0 because the fair same-subtoken content scan reaches 24.20535549399815x.
- [[tests/local_100k_source_subtoken_delta_update_adapter]] implements the current delta-update product for an existing compressed model-state qa adapter. it stores arbitrary replacement bytes for 512 updated facts in a charged 138,104 bit varint-delta-offset patch stream, beats full updated-adapter recompression as the same-block content-scan update baseline by 242,448 delta bits, beats an undercharged mph update table by 50,312 bits, reports an equal-cost matched delta-patch content scan, and keeps total static compression authorization at 0.0 because base-plus-delta loses to full recompression by 1,992 bits.
- [[tests/local_100k_source_token_structure_block_codec]] implements the superseded source-token source-code byte-compression product. it keeps the train-learned indentation split, delta-codes the count stream, adds a fully charged target identifier dictionary, reconstructs 99,761 held-out source bytes exactly, lowers charged payload bits from 124,200 to 123,088, and improves strict charged bits over the best same-block standard codec by 0.03543501930119766. this is a byte-codec result, not high-density knowledge compression.
- [[tests/local_100k_source_subtoken_structure_block_codec]] implements the current narrow source-code byte-compression product. it keeps the same count/body/dictionary accounting but changes the body transform to reversible longest-match subtoken substitution, reconstructs 99,761 held-out source bytes exactly, lowers charged payload bits from 123,088 to 120,952, and improves strict charged bits over the best same-block standard codec by 0.048648916163515785.
- [[tests/local_100k_source_subtoken_structure_corpus_codec]] implements the superseded broader source-code corpus codec product. it freezes a five-block manifest, verifies hashes, charges selectors and standard-fallback headers, reconstructs every block exactly, lowers aggregate selected payload bits from 849,752 to 812,688, and reaches aggregate payload improvement 0.043617431909545375 with random-label/control collapse.
- [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec]] implements the superseded shared-dictionary source-code corpus product. it keeps the frozen five-block manifest but amortizes dictionary cost with one charged shared subtoken dictionary plus small charged local dictionaries per block, lowers aggregate selected payload bits to 803,400, and raises aggregate payload improvement to 0.05454767979363391.
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]] implements the current broader source-code corpus codec product. it keeps the frozen five-block manifest and the charged shared subtoken dictionary, but compresses one global count stream, one global body stream, and one charged framing stream across the corpus, lowering aggregate selected payload bits to 699,144 and raising aggregate payload improvement to 0.1772375940274339. it also beats a stronger global raw standard baseline at 736,504 bits by 37,360 bits and reconstructs after torch module `state_dict` reload from the charged payload buffers.
- [[tests/local_100k_source_subtoken_disjoint_retrieval_codec]] adds a narrow exact-retrieval surface over disjoint frozen source blocks. it retrieves 14,715 aligned 32-byte chunks exactly after torch `state_dict` reload, uses zero train path/hash overlap, charges 431,536 retrieval bits, beats executable raw content scan by 19,592 bits, and beats an undercharged mph diagnostic by 19,608 bits. it does not solve learned semantic qa or broad knowledge compression.
- [[tests/local_100k_source_authored_relation_diagnostic]] tests source-authored definition/import relation qa after the fixed-stride mistake. it answers 337 authored relation queries exactly and shows read-work gain, but a fair unlimited relation-aware scanner solves the task and an honest mph relation index is cheaper by 132,664 bits. it is a diagnostic, not a product.
- [[tests/local_100k_source_dense_authored_relation_diagnostic]] expands the authored relation surface to 3,741 exact source relations after removing the answer-contained signature-query leak in [[mistakes/dense_relation_signature_query_leakage]]. it beats honest relation mph/index by 2,928,400 bits, but paq8px level 2 remains cheaper by 176,536 bits and the fair unlimited scanner still solves the task. it is an amortization diagnostic, not a product.
- [[tests/local_100k_source_relation_mph_codec]] implements the current narrow source-authored relation-index product. it stores a charged minimal-perfect-hash-style router, 17-bit fingerprints, compressed value/provenance id streams, and compressed dictionaries in torch module state, answers 3,741 keyed relation queries exactly after `state_dict` reload, recomputes the paq8px level 2 relation line in-run at 252,952 payload bits plus 8,192 decoder bits, beats that paq8px relation line by 12,360 bits, beats raw-source paq content scan by 165,104 bits, beats undercharged and honest mph relation diagnostics, and reaches 67.90445687825584x strict multiplier. it is not a static breakthrough because the operation remains a narrow keyed relation index.
- [[mistakes/public_context_mixing_baseline_missing]] records the 2026-05-14 paq8px baseline miss. paq8px v214 level 2 compresses the same hard raw joined source to 50,712 bytes, or 405,696 payload bits, which beats the current global-stream corpus payload at 699,144 bits. source-code byte-compression claims now need to beat that public context-mixing pressure line or stay explicitly local and pre-paq.
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]] implements the public trained-dictionary pressure line for those source-code codecs. charged zstd dictionaries trained only on target-excluded source files lose to the current source-subtoken block payload by 30,424 bits and to the current global-stream frozen-corpus payload by 283,696 bits. even the undercharged payload-only diagnostic loses by 26,248 bits on the block and 250,848 bits on the corpus.
- [[indexed_reconstruction_compression]] defines compact handles, schema or residual codes, provenance, and reconstruction.
- [[neural_model_dossier_compression]] defines the mechanism dossier for compression claims.
- [[tests/hard_symbolic_nm_test_material]] provides the symbolic worlds that expose hidden state, controls, and bit-budget tasks.

## proof gates

- compute verbatim trace bits, latent-state bits, schema/residual bits, and imagined-branch program bits per family.
- report ratios only beside preserved operations and control behavior.
- reject any ratio that drops task-relevant state.
- compare against verbatim storage, no-memory, recency-only, shuffled-address, and family-specific controls.
- when a verbatim memory field is available, compare against content-routed sparse read before making a strong compression claim.
- train a tiny mirror only after oracle ratios justify the family.

## side-paper candidates

- operation-preserving compression stack, if the oracle and learned results show a Pareto improvement over verbatim storage.
- task-relative rate-distortion for memory objects, if it cleanly predicts state/action success under bit budgets.
- replay and imagination codecs, if branches and rewrites are stored as compact programs rather than traces.

## kill conditions

- oracle ratios are weak on the constructed worlds.
- compression wins only because the evaluator supplied schema labels unavailable to a model.
- reconstruction improves while action success or joint success falls.
- compressed codes beat no-memory but fail against verbatim storage or shuffled-address controls.
- learned codecs do not approach the oracle direction in the tiny mirror.

## current result

the first oracle compression analysis and frontier split are implemented. useful compression is not uniform across the symbolic surfaces. hard-profile controls are clean and leakage-free, but direct associative recall, correlated-key interference, delayed relevance local commit, bounded output exposure, crossed commit/exposure split, and commit compression frontier remain weak under the current 10x threshold. the first learned-codec mirror for the original source-pair `compression_under_bit_budget` task fails held-out operation preservation. the source-observability contract is now clean: source event observed rate, required fields visible rate, source state reconstructable rate, and visible-source-codec joint success are all 1.0. that older source-pair failure is learned address, payload color, payload position, velocity, action, and decoder generalization, not the whole current compact-state surface.

the subq prior-art update does not change that result. it tightened the comparison standard, and the first implemented sparse-read baseline now shows why: the repaired `compression_under_bit_budget` smoke task can be solved by selecting two legal raw records. this baseline commits 40 bits and is outside the compact-code budget, so it does not kill the compact-code goal, but it demotes the current source-pair mirror from "compression evidence" to "source selection plus budget pressure". the added matched-budget sparse-read control fails at 20 bits, and the distributed-evidence probe now splits the answer across four legal fragments with no commit markers: uncapped sparse read solves at 80 bits, matched-budget sparse read fails at 20 bits. the shared nonlinear tiny model learns the ordinary-split distributed-evidence compact-code task but fails factor-heldout recombination; the factorized structured local codec clears four heldout factor axes across three smoke seeds at 19 bits and minimum test joint success 1.0; the structured constrained-language bridge clears four heldout axes across two smoke seeds. the parser-resistant learned token-count gate fails with minimum learned joint/state success 0.0, so flat token-count learning is demoted. the event-binding parser baseline restores randomized-message success by preserving event segments as bounded local state, and the typed trainable segment binder clears the same local gate with 8,856 parameters and minimum joint/state/action success 0.9583333333333334. the local v1, local 10k, and local foundation artifacts are demoted scaffolds and do not advance the compression claim.

the current full local result is [[tests/local_100k_full_nm]]. it compresses the exact-state 3d candidate's 51 accounted bits to 44 accounted bits through a learned 24-bit binary latent state plus 20 fixed bridge/schema/answer bits in one trainable module. hard validation preserves initial world state, object permanence, occluded localization, action consequence, replay, rewrite, learned branch transition, and bounded answer success at 1.0 while no-memory, code-disabled, shuffled-code, decoder-disabled, no-replay, random-replay, no-branch, wrong-branch, no-integration, and wrong-dynamics controls collapse to 0.0. this is the first accepted local full small nm compression result, but it is still synthetic and code-supervised on an engineered exact-state bridge, not an unsupervised world-code discovery claim and not broad full-model authorization.

the current high-density cell result is [[tests/local_100k_high_density_cell]]. it stores and retrieves exact associative facts through a bounded hybrid cell with a tiny trainable control/read module and explicit committed-state accounting. hard validation reaches exact retrieval success 1.0, 4,096 facts, 8 trainable parameters, 114,688 useful retrievable bits, params-only density 14,336 useful bits per parameter-equivalent, and params-only multiplier 5,734.4x over the 2.5-bit ordinary factual baseline. strict accounting charges 118,816 committed state bits and reaches only 15.427495291902071 useful bits per parameter-equivalent, or 6.1709981167608285x, so [[mistakes/local_100k_high_density_cell_strict_600x_not_met]] blocks the strict 600x claim. this result is useful because it exposes the state-cost wall directly; it is not a hidden authorization to call a lookup table a compressed neuron.

the schema-density attempt [[tests/local_100k_schema_density_cell]] is demoted. it changed the fact family from independent labels to structured schema-generated exact facts, which made the result formula compression rather than the requested knowledge-compression target. hard validation remains useful as boundary telemetry, but [[mistakes/schema_density_cell_structured_target_category_error]] blocks promotion. the current high-density knowledge-compression result remains unsolved after [[tests/local_100k_high_density_cell]] exposed strict state-cost failure and the schema attempt exposed structured-target invalidity.

the unstructured exact-fact probe [[tests/local_100k_unstructured_density_cell]] closes the other false path. it uses random-label exact facts with no formula labels, seed oracle, schema labels, or per-fact committed rows. hard validation records 122,880 useful retrievable bits against a 1,246.72-bit 600x state budget, exact retrieval success 0.0, and information-theoretic 600x possible 0.0. [[mistakes/unstructured_exact_600x_entropy_wall]] records this as an entropy boundary, not an implementation failure.

the unknown-structure corpus probe [[tests/local_100k_unknown_structure_density_probe]] is the first non-generated real-data boundary after the structured-target error. it freezes current wiki corpus material, retrieves 4,096 unique 32-byte chunks exactly from charged compressed state, and collapses the random-label twin to 0.0. hard validation reports 1,048,576 useful bits, 481,282 committed state bits, strict density 34.8547946799184, and strict multiplier 13.941917871967359x. this proves real corpus structure gives measurable compression while confirming that the current standard-codec offset-key probe is far below 600x and cannot be promoted to a high-density knowledge-compression breakthrough.

the learned unknown-structure residual cell [[tests/local_100k_learned_unknown_structure_density_cell]] is now a hard defeat. it uses source-heldout non-generated corpus chunks, opaque associative keys, a learned byte-phrase dictionary, charged residual streams, and the required random-label and disabled-path controls. hard validation reaches exact retrieval success 1.0 and controls collapse 1.0, but strict multiplier is only 3.0525410753623334x, selected standard-codec multiplier is 5.029465628030584x, and the prior charged corpus-codec baseline remains 13.941917871967359x. cross-label random scoring is 0.0, but a separately built random-label twin stores at 1.0. because exact retrieval still requires per-fact residual/key records, this is not a learned high-density neuron-cell.

the current bounded llm-facing compression product is [[tests/local_100k_source_subtoken_qa_adapter]]. it keeps a source-subtoken compressed source stream inside model state as an adapter payload, wraps it in transformer and recurrent/state-style host modules, adds a paraphrase-stable bounded qa surface over four stable source domains, and carries a trained recompression update controller in model state. hard validation reaches exact answer success 1.0, paraphrase-stable answer success 1.0, random-label twin success 0.0, controls collapse 1.0, false-hit rates 0.0 for wrong, unanswerable, partial-overlap, and marker-injection queries, transformer surface pass 1.0, recurrent surface pass 1.0, trainable recompression update success 1.0, update-controller-disabled success 0.0, adapter strict multiplier 24.19976921301639x, and paper-surface strict multiplier 23.847532408460314x. it beats raw content scan and raw undercharged mph diagnostics, but not the fair same-subtoken content scan. this is real exact-retrieval compression on a bounded model-state adapter/update surface, not high-density arbitrary associative memory, learned semantic recall, implicit base-weight storage, static breakthrough, or chat.

the current narrow byte-compression product is [[tests/local_100k_source_subtoken_structure_block_codec]]. it beats the fair same-block standard codec sweep on the held-out source-code block after charging the target dictionary and all headers. the current broader source-code corpus product is [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]. it beats the same-block standard codec sweep across a frozen five-block source-code manifest after charging the shared dictionary payload, global count/body/framing payloads, and the transform header. [[tests/local_100k_zstd_trained_dictionary_baseline_audit]] now adds the public trained-dictionary pressure line and shows the source-subtoken products still win under charged and undercharged dictionary accounting. these are live source-code codec baselines to package into the qa adapter, but they do not solve the fair same-interface qa scanner blocker.

the product is marketable as a compact model-state knowledge adapter for transformer, mamba, rwkv, and related sequence models. public work on [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), [rome](https://arxiv.org/abs/2202.05262), [memit](https://arxiv.org/abs/2210.07229), [product-key memory](https://arxiv.org/abs/1907.05242), [memory layers at scale](https://arxiv.org/abs/2412.09764), [titans](https://arxiv.org/abs/2501.00663), [mamba](https://arxiv.org/abs/2312.00752), [rwkv](https://arxiv.org/abs/2305.13048), and [retnet](https://arxiv.org/abs/2307.08621) establishes that model-side adapters, edit locations, sparse memories, test-time memories, and recurrent state are serious implementation surfaces. none of them removes the need for exact bit accounting. the adapter should therefore be compared against those surfaces as baselines, not sold as replacing them.

## next action

the next compression probe must broaden the current self-contained relation-product baseline, add a useful operation that a same-subtoken content scanner cannot legally perform under matched bits, or move from token signatures toward learned semantic handles. it must keep the model-state adapter shape when making adapter claims, preserve or beat the 23.847532408460314x source-subtoken qa paper-surface product line, and avoid per-fact residual rows, assignment tables disguised as key maps, formula-generated labels, hidden lookup tables, uncharged schema bits, prompt-context storage, and seed oracles. broadening [[tests/local_100k_full_nm]] remains valid only with explicit falsification gates. do not move to paid compute or external simulators until a local result is prosecutor-clean and still beats fair baselines on useful density.

## see also

- [[PROJECT_PLAN]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_frontier_split]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[tests/compression_under_bit_budget_mirror]]
- [[tests/language_grounded_state_density_mirror]]
- [[tests/local_v1_language_model]]
- [[tests/local_10k_chat_model]]
- [[tests/local_foundation_neural_model]]
- [[tests/local_100k_full_nm]]
- [[tests/local_100k_high_density_cell]]
- [[tests/local_100k_schema_density_cell]]
- [[tests/local_100k_unstructured_density_cell]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[tests/local_100k_learned_unknown_structure_density_cell]]
- [[tests/local_100k_shared_predictor_exact_codec]]
- [[tests/local_100k_source_block_codec]]
- [[tests/local_100k_content_addressed_source_codec]]
- [[tests/local_100k_llm_semantic_qa_codec]]
- [[tests/local_100k_weight_carried_qa_codec]]
- [[tests/local_100k_paper_ready_adapter_benchmark]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[tests/local_100k_source_structure_qa_adapter]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[tests/local_100k_source_token_structure_block_codec]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_relation_mph_codec]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[mistakes/local_foundation_lookup_scaffold_category_error]]
- [[mistakes/local_100k_full_nm_soft_code_false_pass]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[mistakes/unstructured_exact_600x_entropy_wall]]
- [[mistakes/learned_unknown_structure_residual_table_defeat]]
- [[mistakes/paper_ready_adapter_offset_lattice_mismatch]]
- [[mistakes/paper_ready_adapter_reload_false_positive]]
- [[mistakes/paper_ready_adapter_source_holdout_overlap]]
- [[schema_density_cell_boundary]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[neural_model_dossier_compression]]
- [[tests/hard_symbolic_nm_test_material]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_3d_world_physics]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
- [[content_routed_sparse_read_prior]]
- [[neural_model_related_work_pressure_matrix]]
- [[high_density_neuron_cell_related_work_pressure_matrix]]
