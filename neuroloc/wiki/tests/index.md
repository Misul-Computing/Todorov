# tests

status: current (as of 2026-05-14).

this folder records concrete simulations and experiments that were actually
run, plus a small number of frozen supporting prototype notes that were
directly attached to those runs or artifacts.

the design and gating method for the architecture backlog now lives in
`wiki/synthesis/phase1_evaluation_surface_for_neural_models.md` and
`wiki/synthesis/synthetic_shared_world_bridge.md`. this index remains a
catalog of executed evidence records and committed test-material artifacts,
not a planning page. individual dated test records should be added here
only after each simulation family is run and archived as its own evidence
page; package-level entries are allowed when they document a committed
symbolic validation surface.

each test page should include:
- date run
- status
- exact script or artifact tested
- what was done
- key quantitative outputs
- verdict
- limitations
- evolution link when the test extends an earlier baseline

## catalog

### paid-run cards

- [[tests/god_run_results]] -- first paid neural-machine run (2026-04-11, 283M, bundle of all features; val_bpb 1.3950, passkey 0/20)
- [[tests/god_run_v2_results]] -- paid re-run with 17+14 prosecutor fixes (2026-04-12, 283M; val_bpb 1.4453, passkey 0/100)
- [[tests/run1_baseline_noerasure_results]] -- paid run with all bundle features off (2026-04-14, 353M; val_bpb 1.4499, passkey 0/100)
- [[tests/run2_slot_memory_first_launch_results]] -- first slot-memory paid run (2026-04-15, 355M; inherited retention bug; val_bpb 1.5107, passkey 0/100)
- [[tests/run2_slot_memory_retention_fixed_results]] -- fifth paid run (2026-04-15, 355M; retention fixed, FLA active; val_bpb 1.4777, passkey 0/100)
- [[tests/run3_cognition_phase1_results]] -- sixth paid run (2026-04-17, 355M; synthetic cognition corpus 50% passkey / 30% kv recall / 20% copy; val_bpb plateaued at alphabet prior 6.3519 from step 150; passkey 0/100 at 256 and 1024; triggered substrate_requires_architectural_change.md)

### pilot experiments

- [[tests/2026-04-07_pattern_completion_baseline|2026-04-07 pattern completion baseline]] -- ca3-like attractor baseline with shuffled-weight control, corruption/load/scaling sweeps, and machine-readable metrics
- [[tests/2026-04-07_kwta_vs_threshold_pilot|2026-04-07 k-wta vs threshold pilot]] -- matched-sparsity bridge pilot showing stronger exact support recovery for k-wta at moderate noise and exact active-fraction control at higher noise
- [[tests/2026-04-08_leak_vs_carry_pilot|2026-04-08 leak vs carry pilot]] -- matched discrete-time bridge pilot showing that explicit leak improves gap retention but loses to atmn-style carry on anchor match and long-sequence drift
- [[tests/2026-04-09_bcm_alpha_pilot|2026-04-09 bcm-like adaptive alpha pilot]] -- gamma=0.3-0.5 significantly stabilizes kda state norm over long sequences (p=0.001) without degrading retrieval. bcm-like forgetting works as predicted.
- [[tests/2026-04-09_gp_vs_bilinear_pilot|2026-04-09 gp vs bilinear pilot]] -- pga provides no advantage over random bilinear or elementwise at random init. geometric structure benefit must come from trained weight interaction, not raw algebra.

### historical matrix-memory series

- [[tests/matrix_memory_capacity_series|matrix-memory capacity series]] --
  grouped landing page for the early 2026-04-12 evidence line:
  encoding round a, encoding round b, head-dimension sweep, decay sweep,
  and overwrite sweep

### later simulation results and analyses

- [[tests/hard_symbolic_nm_test_material|hard symbolic neural-model test material]] -- current symbolic contract package for the neural-model hard tasks; documents what the tests prove, what they do not prove, and why oracle compression analysis must precede any tiny trainable mirror
- [[tests/eligibility_gated_local_commit_test_material|eligibility-gated local commit test material]] -- first mechanism-specific symbolic/oracle package for cellular local-state storage; documents the generator, deterministic controls, leakage checks, committed-distractor exposure gate, validation record, and the oracle-compression handoff
- [[tests/oracle_compression_analysis_results|oracle compression analysis results]] -- first oracle compression-bound package over `hard_symbolic_nm` and `eligibility_commit`; records clean controls, no leakage, eight strong families, six weak families below 10x, and no global tiny-mirror recommendation
- [[tests/compression_under_bit_budget_mirror|compression under bit budget mirror]] -- first local dataset, forbidden-input guard, baseline-control, learned-codec, source-diagnostic, bit-accounting, and telemetry surface for the accepted compression-under-bit-budget tiny mirror
- [[tests/language_grounded_state_density_mirror|language grounded state density mirror]] -- constrained generated-language message-response bridge for the useful-state-density local surface
- [[tests/local_state_write_read_mirror|local state write/read mirror]] -- learned compact-state write/read/update component mirror for the 100k compression-first path
- [[tests/local_100k_replay_answer_mirror|local 100k replay-answer mirror]] -- first finished local 100k model candidate on the symbolic-language surface, with learned replay/rewrite, branch rollout, and bounded answer decoding
- [[tests/local_100k_3d_nm_mirror|local 100k exact-state 3d nm mirror]] -- current top local exact-state 3d nm candidate, with deterministic synthetic 3d worlds, learned compact-state replay/rewrite, bounded answers, and exact branch transition checks under controls
- [[tests/local_100k_full_nm|local 100k full nm]] -- current top local full small nm candidate, with one trainable module, 44-bit accounted learned compression, recurrent state, replay, rewrite, learned branch transition, bounded answers, and hard-code controls
- [[tests/local_100k_high_density_cell|local 100k high-density cell]] -- bounded hybrid exact associative cell with params-only 600x density passing, strict params-plus-committed-state 600x failing, and controls documenting the state-cost wall
- [[tests/local_100k_schema_density_cell|local 100k schema-density cell]] -- demoted formula-compression boundary artifact; structured schema-generated facts are planned by construction and do not satisfy the high-density knowledge-compression target
- [[tests/local_100k_unstructured_density_cell|local 100k unstructured density cell]] -- negative entropy-bound probe showing exact independent random-label facts cannot satisfy 600x strict storage under honest accounting
- [[tests/local_100k_unknown_structure_density_probe|local 100k unknown-structure density probe]] -- non-generated local corpus boundary probe; exact charged corpus retrieval passes, random-label twin collapses, and strict 600x remains blocked
- [[tests/local_100k_learned_unknown_structure_density_cell|local 100k learned unknown-structure density cell]] -- source-heldout learned dictionary/residual associative cell; exact retrieval passes, but a separately built random-label twin also stores at 1.0, strict density loses to the charged corpus-codec baseline, and per-fact residual rows block promotion
- [[tests/local_100k_shared_predictor_exact_codec|local 100k shared-predictor exact codec]] -- tested exact-codec product with one block value stream and no per-fact value slices; exact retrieval passes, random-label payload cost rises, but strict density remains below charged corpus-codec and mph-payload baselines
- [[tests/local_100k_source_block_codec|local 100k source-block codec]] -- tested source/offset exact-codec product with one compressed source block; exact retrieval passes, random-label twin fails, and strict density beats the prior charged corpus-codec baseline while remaining below the high-density target
- [[tests/local_100k_content_addressed_source_codec|local 100k content-addressed source codec]] -- tested content-digest exact-codec product with one compressed source block; exact retrieval passes without source/offset query fields or assignment rows, and strict density slightly beats the previous source-block product while remaining below the high-density target
- [[tests/local_100k_llm_semantic_qa_codec|local 100k llm-facing semantic qa codec]] -- tested bounded natural-language qa exact-codec product with one compressed source block and token-signature question handles; exact answer success passes, random-label twin fails, strict density slightly beats the content-addressed product, and minimal-perfect-hash payload remains slightly stronger
- [[tests/local_100k_weight_carried_qa_codec|local 100k weight-carried qa codec]] -- tested model-state adapter exact-codec product; exact answer success passes, random-label twin fails, external payload and stored manifest are zero, strict density reaches `15.215221373768888x`, and the product beats the prior llm-facing qa and matched minimal-perfect-hash payload lines while remaining below the 600x target
- [[tests/local_100k_paper_ready_adapter_benchmark|local 100k paper-ready adapter benchmark]] -- tested paper-facing local model-state adapter benchmark; transformer and recurrent/state host integration pass, paraphrase-stable exact qa passes, four source domains pass, public baseline stack pass is `1.0`, strict adapter density reaches `16.641752137599937x`, and strict breakthrough authorization remains `0.0`
- [[tests/local_100k_margin_recompression_adapter|local 100k margin recompression adapter]] -- tested high-margin adapter/update engineering product; source-holdout overlap is repaired, transformer and recurrent/state host integration pass, trained recompression update passes, paper-surface strict density reaches `22.421639537059313x`, and paper-ready plus strict breakthrough authorization remain `0.0` because same-block content scan is not beaten
- [[tests/local_100k_semantic_alias_payload_adapter|local 100k semantic alias payload adapter]] -- demoted alias diagnostic; payload bits improve and exact retrieval passes, but generated alias labels make a fair same-interface alias content scan succeed at `1.0`, so publishable and strict breakthrough authorization remain `0.0`
- [[tests/local_100k_source_native_relation_adapter|local 100k source-native relation adapter]] -- demoted fixed-stride relation diagnostic; exact and paraphrase retrieval pass and relationless scan fails, but a fair stride-aware content scan succeeds at `1.0`, so publishable and strict breakthrough authorization remain `0.0`
- [[tests/local_100k_weight_mantissa_payload_adapter|local 100k weight mantissa payload adapter]] -- demoted model-state steganography diagnostic; exact retrieval passes from fp32 parameter mantissas, but honest fp32-state accounting drops paper-surface density to `16.94360217334222x` while same-block content scan remains `23.06526987269378x`, so publishable and strict breakthrough authorization remain `0.0`
- [[tests/local_100k_indent_token_block_codec|local 100k indent-token block codec]] -- narrow local source-code block codec product; a train-learned indentation token plus fair standard codec sweep reconstructs held-out bytes exactly, improves strict charged bits over the best same-block codec by `0.020192022171632188`, reports sliding 64-byte ngram overlap `3104.0`, and keeps broad breakthrough, knowledge, chat, full nm, and static retrieval-wrapper authorization at `0.0`
- [[tests/local_100k_source_structure_block_codec|local 100k source-structure block codec]] -- superseded narrow local source-code block codec product; a train-learned indentation unit splits held-out source bytes into count/body planes carried in a reloadable torch module state dict, reconstructs exactly, improves strict charged bits over the best same-block codec by `0.028555874492724932`, beats the indent-token product by `0.008363852321092744`, and keeps broad breakthrough, knowledge, chat, full nm, and static retrieval-wrapper authorization at `0.0`
- [[tests/local_100k_source_token_structure_block_codec|local 100k source-token-structure block codec]] -- superseded narrow local source-code block codec product; count-delta structure coding plus a fully charged target identifier dictionary reconstructs held-out bytes exactly, lowers charged payload bits from `124200` to `123088`, improves strict charged bits over the best same-block codec by `0.03543501930119766`, and keeps broad breakthrough, knowledge, chat, full nm, and strict breakthrough authorization at `0.0`
- [[tests/local_100k_source_subtoken_structure_block_codec|local 100k source-subtoken-structure block codec]] -- current narrow local source-code block codec product; count-delta structure coding plus a fully charged longest-match identifier-subtoken dictionary reconstructs `99761` held-out bytes exactly, lowers charged payload bits from `123088` to `120952`, improves strict charged bits over the best same-block codec by `0.048648916163515785`, and keeps broad breakthrough, knowledge, chat, full nm, and strict breakthrough authorization at `0.0`
- [[tests/local_100k_source_subtoken_structure_corpus_codec|local 100k source-subtoken-structure corpus codec]] -- superseded broader frozen-manifest source-code corpus codec product; five source-code blocks reconstruct exactly, aggregate selected payload bits fall from `849752` to `812688`, aggregate payload improvement reaches `0.043617431909545375`, random-label and disabled controls pass, and broad breakthrough, knowledge, chat, full nm, and strict breakthrough authorization remain `0.0`
- [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec|local 100k source-subtoken shared-dictionary corpus codec]] -- superseded broader frozen-manifest source-code corpus codec product; one charged shared subtoken dictionary plus charged per-block local dictionaries reconstruct all five blocks exactly, lower aggregate selected payload bits from `812688` to `803400`, reach aggregate payload improvement `0.05454767979363391`, beat the charged zstd trained-dictionary line by `179440` bits, and keep broad breakthrough, knowledge, chat, full nm, and strict breakthrough authorization at `0.0`
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec|local 100k source-subtoken global-stream corpus codec]] -- current broader frozen-manifest source-code corpus codec product; one charged shared subtoken dictionary plus global count, body, and framing streams reconstruct all five blocks exactly, reload from torch module `state_dict`, lower aggregate selected payload bits from `803400` to `699144`, reach aggregate payload improvement `0.1772375940274339`, beat a stronger global raw standard baseline by `37360` bits, beat the charged zstd trained-dictionary line by `283696` bits, and keep broad breakthrough, knowledge, chat, full nm, and strict breakthrough authorization at `0.0`
- [[tests/local_100k_source_subtoken_delta_update_adapter|local 100k source-subtoken delta-update adapter]] -- narrow model-state delta-update product; stores arbitrary replacement bytes for `512` updated facts in a charged `138104` bit varint-delta-offset patch stream, beats full updated-adapter recompression as the same-block content-scan update baseline by `242448` delta bits, beats an undercharged mph update table by `50312` bits, reports an equal-cost matched delta-patch content scan, and keeps total static compression authorization at `0.0`
- [[tests/local_100k_source_subtoken_disjoint_retrieval_codec|local 100k source-subtoken disjoint retrieval codec]] -- current narrow source-code exact-retrieval codec product; retrieves `14715` aligned 32-byte chunks from disjoint frozen source blocks after torch `state_dict` reload, uses zero train path/hash overlap, beats raw content scan by `19592` bits, beats an undercharged mph diagnostic by `19608` bits, and keeps broad breakthrough, knowledge, chat, full nm, and strict breakthrough authorization at `0.0`
- [[tests/local_100k_source_authored_relation_diagnostic|local 100k source-authored relation diagnostic]] -- diagnostic for source-authored definition/import relation qa after the fixed-stride mistake; answers `337` authored relation queries exactly and shows read-work gain, but loses to an honest mph relation index by `132664` bits and keeps product/breakthrough authorization at `0.0`
- [[tests/local_100k_source_dense_authored_relation_diagnostic|local 100k source-dense authored relation diagnostic]] -- dense authored-relation diagnostic; answers `3741` exact source-authored relation queries and beats honest mph/index by `2928400` bits after removing an answer-contained signature query leak, but loses to paq8px level 2 by `176536` bits and keeps product/breakthrough authorization at `0.0`
- [[tests/local_100k_source_relation_mph_codec|local 100k source relation mph codec]] -- narrow source-authored relation-index product; answers `3741` exact keyed relation queries from charged module state, recomputes and beats paq8px level 2 on the relation surface by `12360` bits, beats raw-source paq content scan by `165104` bits, collapses cross-scored random-label, random-label rebuild-density, disabled, shuffled-fingerprint, and hardened wrong-query controls, and keeps static breakthrough, broad breakthrough, 600x, chat, and full nm authorization at `0.0`
- [[tests/local_100k_external_relation_adapter|local 100k external relation adapter]] -- public-corpus llm-adoptable relation adapter product; answers `6247` exact cpython `v3.12.3` source-authored relation facts from charged module state, packages the payload inside tiny transformer, recurrent, and state-space hosts, recomputes and beats paq8px level 2 source-scan pressure by `192472` bits, beats honest mph indexing by `3098120` bits, and keeps true base-weight implicit storage, broad breakthrough, 600x, chat, and full nm authorization at `0.0`
- [[tests/local_100k_model_state_knowledge_pack|local 100k model state knowledge pack]] -- broader paper-facing bounded knowledge-pack product; answers `9754` exact public cpython `v3.12.3` source, documentation, and config relation facts from charged module state, packages the payload inside tiny transformer, recurrent, and state-space hosts, recomputes and beats paq8px level 2 source-scan pressure by `705864` bits, beats zstd level 19 source scan by `1515616` bits, beats honest mph, product-key-style, rag/knn, lora-style, and model-edit storage pressure lines, passes adapter export, recompress update, rollback, and controls, and keeps true base-weight implicit storage, broad breakthrough, 600x, chat, and full nm authorization at `0.0`
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit|local 100k zstd trained-dictionary baseline audit]] -- public trained-dictionary pressure audit; charged zstd dictionaries trained on target-excluded local source lose to the source-subtoken block payload by `30424` bits and to the global-stream frozen-corpus payload by `283696` bits, the undercharged payload-only lines also lose, random-label controls collapse, and strict breakthrough authorization remains `0.0`
- [[tests/paq8px_public_context_mixing_baseline_audit|paq8px public context-mixing baseline audit]] -- public context-mixing pressure audit; paq8px v214 level 2 compresses the hard raw joined source to `50712` bytes, beating the current `699144`-bit global-stream corpus payload and blocking public-compressor breakthrough wording for source-code byte compression
- [[tests/local_100k_source_structure_qa_adapter|local 100k source-structure qa adapter]] -- superseded bounded exact qa adapter product; source-structure count/body streams carried in torch module state answer 4,096 bounded source-heldout questions exactly, beat raw content-scan and raw undercharged mph diagnostics, but do not beat the fair same-structure content scan, so static breakthrough authorization remains `0.0`
- [[tests/local_100k_source_subtoken_qa_adapter|local 100k source-subtoken qa adapter]] -- current strongest bounded exact qa adapter product; source-subtoken streams carried in torch module state answer 4,096 bounded source-heldout questions exactly, preserve transformer/recurrent hosts and trained recompression update, raise paper-surface strict multiplier to `23.847532408460314x`, but do not beat the fair same-subtoken content scan, so strict breakthrough authorization remains `0.0`
- [[tests/local_v1_language_model|local v1 language model]] -- demoted constrained state-router scaffold with record update, targeted retrieval, and branch-copy controls
- [[tests/local_10k_chat_model|local 10k chat model]] -- demoted under-10k constrained responder scaffold, not a chat model in the project sense
- [[tests/local_foundation_neural_model|local foundation neural model]] -- demoted record-routing and compartment-boundary scaffold with identity codec bookkeeping
- [[tests/correction_field_trained_prediction_results|correction-field trained-prediction results]] -- trained-predictor correction-field sim; memory_capacity_delta=0 at every quality
- [[tests/multi_resolution_head_split_results|multi-resolution head split results]] -- fast/medium/slow heads with surprise gates; rare-class recall improves
- [[tests/thinking_loop_prototype_results|thinking-loop prototype results]] -- recurrent hidden-state refinement pilot on modular arithmetic

### historical long-form syntheses

- [[tests/god_run_findings|god_run findings]] -- original long-form synthesis of god_run's results

### supporting prototype notes

- [[tests/aesthetic_logger_prototype|aesthetic logger prototype]] -- frozen prototype note for the phase 6a logging module; not a live module-status page

## see also

- [[PROJECT_PLAN]]
- [[tests/hard_symbolic_nm_test_material]]
- [[tests/eligibility_gated_local_commit_test_material]]
- [[tests/oracle_compression_analysis_results]]
- [[tests/compression_under_bit_budget_mirror]]
- [[tests/language_grounded_state_density_mirror]]
- [[tests/local_v1_language_model]]
- [[tests/local_100k_3d_nm_mirror]]
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
- [[tests/local_100k_semantic_alias_payload_adapter]]
- [[tests/local_100k_source_native_relation_adapter]]
- [[tests/local_100k_weight_mantissa_payload_adapter]]
- [[tests/local_100k_indent_token_block_codec]]
- [[tests/local_100k_source_structure_block_codec]]
- [[tests/local_100k_source_token_structure_block_codec]]
- [[tests/local_100k_source_subtoken_structure_block_codec]]
- [[tests/local_100k_source_subtoken_structure_corpus_codec]]
- [[tests/local_100k_source_subtoken_shared_dictionary_corpus_codec]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_subtoken_delta_update_adapter]]
- [[tests/local_100k_source_relation_mph_codec]]
- [[tests/local_100k_external_relation_adapter]]
- [[tests/local_100k_model_state_knowledge_pack]]
- [[tests/local_100k_source_dense_authored_relation_diagnostic]]
- [[tests/paq8px_public_context_mixing_baseline_audit]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[tests/local_100k_source_subtoken_qa_adapter]]
- [[tests/local_100k_source_structure_qa_adapter]]
- [[tests/local_10k_chat_model]]
- [[tests/local_foundation_neural_model]]
- [[mistakes/local_foundation_lookup_scaffold_category_error]]
- [[mistakes/local_100k_3d_branch_rollout_overfit]]
- [[mistakes/local_100k_full_nm_soft_code_false_pass]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[mistakes/unstructured_exact_600x_entropy_wall]]
- [[mistakes/learned_unknown_structure_residual_table_defeat]]
- [[mistakes/source_block_codec_raw_cache_category_error]]
- [[mistakes/paper_ready_adapter_source_holdout_overlap]]
- [[mistakes/semantic_alias_payload_adapter_formula_alias_category_error]]
- [[mistakes/source_native_relation_stride_rule_category_error]]
- [[mistakes/weight_mantissa_payload_steganographic_accounting_error]]
- [[mistakes/answer_surface_codec_static_scan_not_beaten]]
- [[mistakes/neural_coordinate_codec_random_label_memorizer]]
- [[mistakes/learned_block_codec_frontier_loses_to_standard_codec]]
- [[synthesis/high_density_neuron_cell_related_work_pressure_matrix]]
- [[synthesis/schema_density_cell_boundary]]
- [[synthesis/phase1_evaluation_surface_for_neural_models]]
- [[synthesis/synthetic_shared_world_bridge]]
