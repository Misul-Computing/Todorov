# eligibility-gated local commit test material

status: current (as of 2026-04-27).

test type: symbolic/oracle mechanism-gate package

script:
- `neuroloc/simulations/memory/eligibility_gated_local_commit.py`

source contract:
- [[synthesis/neural_model_symbolic_contract_eligibility_gated_local_commit]]

implemented surface:
- `neuroloc/data/nm_worlds.py`
- `tests/test_eligibility_gated_local_commit_worlds.py`
- `tests/test_eligibility_gated_local_commit_suite.py`
- `neuroloc/simulations/suite_registry.py`

## what was done

implemented the first mechanism-specific symbolic test-material package for the neural model: eligibility-gated local commit with bounded output exposure.

the generator now creates deterministic symbolic episodes with four families:

- delayed relevance local commit
- bounded output exposure
- crossed commit and exposure split
- commit compression frontier

each contract exposes hidden state, observation stream, query, target, candidate events, relevance events, commit targets, read queries, exposure targets, relevant positions, distractor positions, negative commit positions, trace-eligible positions, commit positions, exposure positions, difficulty, bit budget, output budget, oracle codes, expected controls, telemetry, leakage checks, and kill conditions.

deterministic policies now cover oracle, no-memory, recency-only, shuffled-address, no-trace, random-trace, always-commit unlimited, always-commit matched budget, oracle-mark/no-commit, no-commit/oracle-exposure, fixed closed exposure, fixed open exposure, hand-opened exposure, crossed oracle/degraded split controls, matched residual capacity, and matched compute budget.

the implementation also closes the first reviewer findings on the symbolic package:

- target events are not the most recent compatible candidates.
- at least one distractor shares a surface feature with the target.
- bounded exposure and crossed split families include committed distractor competitors.
- no-trace, matched residual-capacity, and matched compute-budget controls are explicitly exercised.
- summary metrics include `state_probe_accuracy`, `action_success`, `joint_success`, delayed-use success, writes per successful episode, and memory-output norm versus residual norm.

## validation

commands run on 2026-04-27:

- `python -m pytest tests/test_eligibility_gated_local_commit_worlds.py tests/test_eligibility_gated_local_commit_suite.py -q`
- result: 13 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_nm_hard_symbolic_worlds.py tests/test_hard_symbolic_nm_suite.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 14 passed, 1 known numpy-on-windows warning
- `python neuroloc/simulations/suite_runner.py --suite eligibility_commit --profile smoke --output-root .pytest_tmp_final_428 --write-summary`
- result: 1/1 simulation passed

the warning is the existing local numpy-on-windows experimental-build warning, not a test failure.

## what this proves

the symbolic material is now strict enough to test whether a proposed mechanism can separate five phases:

1. mark candidate information before relevance is known.
2. wait until delayed relevance resolves the useful candidate.
3. commit only the resolved candidate under a bit budget.
4. read the committed candidate by address.
5. expose only the useful committed item under an output-capacity budget.

the controls prove that oracle success is not explained by current observation, recency leakage, shuffled address mapping, unlimited write capacity, fixed-open output exposure, residual capacity, or matched compute alone.

the compression counter is only a symbolic/oracle bound. it shows that selective commit can write fewer bits than always-write while preserving the constructed task state on this toy surface. it does not prove learned compression, learned trainability, biological novelty, or model-scale usefulness.

## limitations

this package is deterministic symbolic material. it does not train a neural model. it does not touch `god_machine.py`, paid presets, h200, kaggle, pod paths, simulator selection, or full-model integration.

the next research question was whether oracle compression counters over this and the broader hard-symbolic worlds produced strong enough bounds to justify a tiny trainable mirror. [[tests/oracle_compression_analysis_results]] records the first bound, and [[synthesis/oracle_compression_frontier_split]] records the split: the eligibility-specific families are control-clean but remain below the current 10x useful-compression threshold, so global mirror training is not justified from this package alone.

## verdict

accepted as a current symbolic/oracle test-material gate for the first cellular/local-state mechanism. the oracle compression follow-up is documented in [[tests/oracle_compression_analysis_results]], the family split is documented in [[synthesis/oracle_compression_frontier_split]], and the first accepted-family proof package is [[synthesis/neural_model_dossier_compression_under_bit_budget_codec]]. the next no-paid action is the tiny local mirror contract for the accepted compression-under-bit-budget family rather than global mirror training.

## see also

- [[PROJECT_PLAN]]
- [[tests/index]]
- [[synthesis/neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[synthesis/neural_model_dossier_eligibility_gated_local_commit]]
- [[synthesis/cellular_state_storage_gap_map]]
- [[synthesis/neural_model_lane_cellular_state_storage]]
- [[synthesis/neural_model_lane_trainability_evaluation]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/oracle_compression_analysis_plan]]
- [[tests/oracle_compression_analysis_results]]
- [[synthesis/oracle_compression_frontier_split]]
- [[synthesis/neural_model_dossier_compression_under_bit_budget_codec]]
- [[tests/hard_symbolic_nm_test_material]]
