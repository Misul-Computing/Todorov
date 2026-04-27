# compression under bit budget mirror

status: current (as of 2026-04-27).

test type: local tiny-mirror dataset and baseline-control surface

script:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`

source contract:
- [[synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]

implemented surface:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`
- `tests/test_compression_under_bit_budget_mirror.py`
- `neuroloc/simulations/suite_registry.py`

## what was done

implemented the first local code surface for the accepted `compression_under_bit_budget` family. this is not a learned codec yet. it is the dataset, guard, baseline, and telemetry shell that a learned codec must satisfy before any result can count.

the surface now provides:

- deterministic train, validation, and test splits by seed.
- filtering to the `compression_under_bit_budget` family only.
- observation streams trimmed to the task query time so post-query observations cannot leak into inputs.
- model-input records that exclude exact hidden state, target answers, oracle codes, family-label shortcuts, future observations, and memory-relevant positions.
- labels and diagnostics kept outside the model-input path.
- evaluation rows computed from the exact contract attached to the dataset record, not from a regenerated episode.
- deterministic controls for oracle codec, verbatim storage, compressed oracle storage, no-memory, recency-only, shuffled-address, random codebook, matched-bit random code, matched-compute no-code, and trainability split placeholders.
- explicit bit accounting and telemetry fields for committed bits, address margin, address entropy, read concentration, write frequency, reconstruction error, memory-output norm, and confidence intervals.
- suite-runner maximum gates that reject future-observation leakage, learned-result leakage, full-model authorization, paid-compute authorization, and blocked-authorization violations.
- suite-registry entry `compression_mirror`.

## validation

commands run on 2026-04-27:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q`
- result: 10 passed, 1 known numpy-on-windows warning
- `python -m py_compile neuroloc/simulations/memory/compression_under_bit_budget_mirror.py neuroloc/simulations/suite_registry.py neuroloc/simulations/suite_runner.py tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py`
- result: passed
- `python neuroloc/simulations/suite_runner.py --suite compression_mirror --profile smoke --output-root .pytest_tmp_final_442 --write-summary`
- result: 1/1 simulation passed
- `python -m pytest tests --collect-only -q`
- result: 317 tests collected, 1 known numpy-on-windows warning

the warning is the existing local numpy-on-windows experimental-build warning, not a test failure.

## key smoke outputs

- family count: 1
- policy count: 15
- forbidden input violation count: 0
- future observation violation count: 0
- oracle joint success: 1.0
- compressed oracle joint success: 1.0
- verbatim joint success: 1.0
- no-memory joint success: 0.0
- recency-only joint success: 0.0
- shuffled-address joint success: 0.0
- random-codebook joint success: 0.0
- verbatim within budget: 0.0
- compressed oracle within budget: 1.0
- local mirror code authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- blocked authorization violation count: 0.0

## what this proves

this proves that the first mirror surface can be generated deterministically, limited to the chosen family, protected against forbidden oracle inputs and post-query observations, and compared against the required controls with explicit bit accounting.

it also proves that the suite registry can run this surface as a local smoke suite and reject blocked authorization flags if they turn on.

## what this does not prove

this does not prove learned compression.

it does not prove that a neural model can infer schema, residual, address, or provenance codes from observations.

it does not prove novelty.

it does not authorize full-model integration, simulator work, h200, kaggle, pod, runpod, or paid compute.

## verdict

accepted as the local dataset and baseline-control surface for the first `compression_under_bit_budget` tiny mirror. the next no-paid work is the learned codec implementation on this local surface, with the existing controls and forbidden-input guards preserved.

## see also

- [[PROJECT_PLAN]]
- [[tests/index]]
- [[synthesis/neural_model_paper_spine]]
- [[synthesis/neural_model_research_test_material_plan]]
- [[synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[synthesis/neural_model_dossier_compression_under_bit_budget_codec]]
- [[synthesis/oracle_compression_frontier_split]]
- [[tests/oracle_compression_analysis_results]]
- [[tests/hard_symbolic_nm_test_material]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/neural_model_lane_trainability_evaluation]]
