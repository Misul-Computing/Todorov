# oracle compression analysis results

status: current (as of 2026-04-27).

test type: symbolic/oracle compression-bound package

script:
- `neuroloc/simulations/memory/oracle_compression_analysis.py`

source plan:
- [[synthesis/oracle_compression_analysis_plan]]

implemented surface:
- `neuroloc/data/nm_worlds.py`
- `neuroloc/simulations/memory/oracle_compression_analysis.py`
- `tests/test_oracle_compression_analysis.py`
- `neuroloc/simulations/suite_registry.py`

## what was done

implemented the first oracle compression analysis over two symbolic surfaces:

- the broader `hard_symbolic_nm` worlds documented in [[tests/hard_symbolic_nm_test_material]]
- the mechanism-specific `eligibility_commit` worlds documented in [[tests/eligibility_gated_local_commit_test_material]]

the analysis emits one record per contract with verbatim trace bits, latent-state bits, schema/residual bits, imagined-branch program bits, oracle ratios, operation flags, controls, leakage flags, acceptance, and kill conditions.

the accepted result is deliberately narrow. it is an oracle-bound counter, not learned compression. it tests whether the constructed worlds have enough structured redundancy to justify a later tiny trainable mirror.

## validation

commands run on 2026-04-27:

- `python -m pytest tests/test_oracle_compression_analysis.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 6 passed, 1 known numpy-on-windows warning
- `python neuroloc/simulations/suite_runner.py --suite oracle_compression --profile smoke --output-root .pytest_tmp_final_434 --write-summary`
- result: 1/1 simulation passed
- `python neuroloc/simulations/suite_runner.py --suite oracle_compression --profile hard --output-root .pytest_tmp_final_435 --write-summary`
- result: 1/1 simulation passed
- `python -m py_compile neuroloc/simulations/memory/oracle_compression_analysis.py tests/test_oracle_compression_analysis.py`
- result: passed

the warning is the existing local numpy-on-windows experimental-build warning, not a test failure.

## key hard-profile outputs

- contracts: 448
- surfaces: 2
- families: 14
- operation preservation rate: 1.0
- controls preservation rate: 1.0
- leakage-free rate: 1.0
- accepted rate: 0.5714
- strong oracle families: 8
- weak oracle families: 6
- kill-condition count: 192
- minimum best oracle ratio: 7.09x
- maximum best oracle ratio: 39.0x
- hard-symbolic schema-ratio mean: 11.23x
- eligibility commit ratio versus always-write: 3.61x
- imagination branch-program ratio mean: 39.0x
- global trainable mirror recommendation: 0.0

the weak families are:

- associative recall
- correlated-key interference
- delayed relevance local commit
- bounded output exposure
- crossed commit/exposure split
- commit compression frontier

each weak family fails for the same reason: the oracle ratio is below the current 10x useful-compression threshold. this is a useful negative bound, not an implementation failure.

## what this proves

the implemented symbolic worlds now have a first oracle compression frontier.

the analysis proves that the controls remain clean while bit accounting is applied: oracle succeeds, no-memory fails, recency-only fails, shuffled-address fails, and leakage checks stay clear.

it also shows, under this oracle-bound symbolic surface, that useful compression is not uniform across the project. belief-state, delayed-use, episodic-reuse, context-routing, explicit bit-budget compression, replay/rewrite, iterative rollout, and imagination/recombination clear the current strong-oracle threshold on the hard profile. direct associative recall, correlated-key interference, and the first eligibility-commit families do not clear it yet.

## what this does not prove

this does not prove learned compression.

it does not prove novelty.

it does not prove that a neural model can infer the schema, residual, or branch-program codes from observations.

it does not authorize a paid pod, full-model integration, simulator implementation, or intervention preset.

it does not justify a global tiny trainable mirror yet, because six families remain below the threshold. a narrow mirror may later be scoped only for families whose oracle ratio and controls justify the attempt.

## verdict

accepted as the first oracle compression-bound package. the frontier split is documented in [[synthesis/oracle_compression_frontier_split]], and the first family-specific proof package is [[synthesis/neural_model_dossier_compression_under_bit_budget_codec]]. the next research action is a tiny local mirror contract for that accepted family, not a global mirror.

## see also

- [[PROJECT_PLAN]]
- [[tests/index]]
- [[synthesis/neural_model_paper_spine]]
- [[synthesis/oracle_compression_analysis_plan]]
- [[synthesis/oracle_compression_frontier_split]]
- [[synthesis/neural_model_dossier_compression_under_bit_budget_codec]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/neural_model_lane_trainability_evaluation]]
- [[synthesis/neural_model_compression_stack]]
- [[synthesis/indexed_reconstruction_compression]]
- [[tests/hard_symbolic_nm_test_material]]
- [[tests/eligibility_gated_local_commit_test_material]]
