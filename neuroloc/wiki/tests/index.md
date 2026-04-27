# tests

status: current (as of 2026-04-27).

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
- [[synthesis/phase1_evaluation_surface_for_neural_models]]
- [[synthesis/synthetic_shared_world_bridge]]
