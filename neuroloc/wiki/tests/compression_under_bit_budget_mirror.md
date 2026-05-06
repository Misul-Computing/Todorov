# compression under bit budget mirror

status: current (as of 2026-05-07).

test type: local tiny-mirror dataset, baseline-control, content-routed sparse-read, learned-codec, source-availability, and diagnostic-localization surface

script:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`

source contract:
- [[synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]

implemented surface:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`
- `tests/test_compression_under_bit_budget_mirror.py`
- `neuroloc/simulations/suite_registry.py`

## what was done

implemented the first local code surface for the accepted `compression_under_bit_budget` family, the first trainable learned-codec smoke result, the first diagnostic-only failure-localization controls, the first source-availability/action-ambiguity checks, the repaired source-observability contract, and the first content-routed sparse-read baseline over legal observation events. this is not a passing learned-compression result. it is a repaired problem-specification result plus a controlled negative trainability result: the visible-source codec and the content-routed sparse-read baseline solve the repaired task from legal non-oracle inputs, while the tiny learned codec memorizes train split and remains far below the sparse-read baseline on held-out episodes.

the surface now provides:

- deterministic train, validation, and test splits by seed.
- filtering to the `compression_under_bit_budget` family only.
- observation streams trimmed to the task query time so post-query observations cannot leak into inputs.
- model-input records that exclude exact hidden state, target answers, oracle codes, family-label shortcuts, future observations, and memory-relevant positions while exposing explicit commit-time markers for the legal source event.
- labels and diagnostics kept outside the model-input path.
- evaluation rows computed from the exact contract attached to the dataset record, not from a regenerated episode.
- deterministic controls for oracle codec, verbatim storage, content-routed sparse read, compressed oracle storage, no-memory, recency-only, shuffled-address, random codebook, matched-bit random code, matched-compute no-code, frozen-random encoder learned-decoder, and learned-encoder frozen-random decoder.
- a trainable field codec with non-oracle observation features, predicted state/action, compact code fields, address/schema/residual/action/provenance fields, parameter counts, train loss, and held-out split metrics.
- diagnostic-only controls for learned-code/oracle-decoder, oracle-code/learned-decoder, learned-address/oracle-payload, oracle-address/learned-payload, provenance-exposed learned codec, visible-source codec, visible-source-state/oracle-action/oracle-decoder, source-observation/learned-action, provenance-exposed/oracle-decoder, learned-state/oracle-action/oracle-decoder, and oracle-state/learned-action/oracle-decoder.
- source-availability telemetry for source event presence, observation, required visible fields, reconstructable source state, source-query gap, and source-signature action ambiguity.
- explicit diagnostic flags that prevent oracle-exposed rows from counting as accepted learned-codec results.
- explicit bit accounting and telemetry fields for committed bits, address margin, address entropy, read concentration, write frequency, reconstruction error, memory-output norm, and confidence intervals.
- sparse-read telemetry for selected-record count, source-selection recall, next-source-selection recall, false-source-selection rate, record-bit cost, total committed bits, within-budget status, and compression ratio versus verbatim storage.
- suite-runner gates that require learned-result telemetry while rejecting future-observation leakage, full-model authorization, paid-compute authorization, and blocked-authorization violations.
- suite-registry entry `compression_mirror`.

## validation

commands run on 2026-04-28 after the source-observability contract repair:

- `python -m pytest tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q`
- result: 31 passed, 2 local warnings
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_nm_hard_symbolic_worlds.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q`
- result: 33 passed, 1 known numpy-on-windows warning; this command required an unsandboxed local run because pytest's temporary-directory fixture hit windows access-denied errors inside the sandbox.
- `python neuroloc/simulations/suite_runner.py --suite compression_mirror --profile smoke --output-root reports/validation/compression_mirror_contract_repair --write-summary`
- result: 1/1 simulation passed
- `python -m py_compile neuroloc/data/nm_worlds.py neuroloc/simulations/memory/compression_under_bit_budget_mirror.py neuroloc/simulations/suite_registry.py tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py`
- result: passed
- `python -c "import yaml; yaml.safe_load(open('state/program_status.yaml', encoding='utf-8')); print('yaml ok')"`
- result: passed
- `python -m pytest tests --collect-only -q`
- result: 330 tests collected, 1 known numpy-on-windows warning

the numpy warning is the existing local numpy-on-windows experimental-build warning, not a test failure.

commands run on 2026-05-07 after adding the content-routed sparse-read baseline:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py -q`
- result: 22 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_simulation_suite.py::test_suite_registry_contract tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q`
- result: 34 passed, 1 known numpy-on-windows warning
- `python -m neuroloc.simulations.memory.compression_under_bit_budget_mirror smoke`
- result: wrote `neuroloc/output/simulation_runs/memory/compression_under_bit_budget_mirror/compression_under_bit_budget_mirror_metrics.json`
- `python -m pytest tests --collect-only -q`
- result: 332 tests collected, 1 known numpy-on-windows warning

## key smoke outputs

- family count: 1
- policy count: 24
- dataset record count: 24
- learned result count: 24
- diagnostic result count: 264
- forbidden input violation count: 0
- future observation violation count: 0
- oracle joint success: 1.0
- compressed oracle joint success: 1.0
- verbatim joint success: 1.0
- content-routed sparse-read joint success: 1.0
- content-routed sparse-read state success: 1.0
- content-routed sparse-read action success: 1.0
- content-routed sparse-read selected-record count: 2.0
- content-routed sparse-read source-selection recall: 1.0
- content-routed sparse-read next-source-selection recall: 1.0
- content-routed sparse-read false-source-selection rate: 0.0
- content-routed sparse-read total committed bits: 40.0
- content-routed sparse-read within budget: 0.0
- content-routed sparse-read compression ratio versus verbatim: 1.3x
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
- learned-codec train joint success: 1.0
- learned-codec validation joint success: 0.0
- learned-codec test joint success: 0.125
- learned-codec test state-probe accuracy: 0.125
- learned-codec test action success: 0.625
- learned-codec encoder address accuracy: 0.875
- learned-codec encoder payload accuracy: 0.125
- learned-codec encoder payload color accuracy: 0.9375
- learned-codec encoder payload shape accuracy: 0.9375
- learned-codec encoder payload position accuracy: 0.4375
- learned-codec encoder payload velocity accuracy: 0.4375
- learned-codec encoder action accuracy: 0.375
- learned-codec encoder provenance accuracy: 0.9375
- learned-codec bits per successful held-out episode: 152.0
- learned-codec compression ratio versus verbatim: 2.74x
- learned-codec engineering pass: 0.0
- learned-codec paper-track pass: 0.0
- learned-codec kill-condition count: 1
- source event observed rate: 1.0
- source required fields visible rate: 1.0
- source state reconstructable rate: 1.0
- source-signature action ambiguity rate: 0.0
- visible-source-codec joint success: 1.0
- visible-source-codec state success: 1.0
- visible-source-codec action success: 1.0
- visible-source-state/oracle-action/oracle-decoder joint success: 1.0
- visible-source-state/oracle-action/oracle-decoder state success: 1.0
- learned-code/oracle-decoder joint success: 0.125
- oracle-code/learned-decoder train joint success: 1.0
- oracle-code/learned-decoder validation joint success: 0.0
- oracle-code/learned-decoder test joint success: 0.0
- oracle-code/learned-decoder train-test joint gap: 1.0
- learned-address/oracle-payload joint success: 0.25
- oracle-address/learned-payload joint success: 0.125
- provenance-exposed learned-codec joint success: 0.125
- provenance-exposed/oracle-decoder joint success: 0.0
- learned-state/oracle-action/oracle-decoder joint success: 0.0
- oracle-state/learned-action/oracle-decoder joint success: 0.0
- oracle-state/learned-action/oracle-decoder action success: 0.0
- learned action-only failure rate: 1.0
- learned-minus-content-routed-sparse-read joint gap: -0.875
- content-routed sparse-read rescue delta over learned codec: 0.875
- strongest nonlearned source diagnostic rescue delta: 1.0 from the visible-source codec

## what this proves

this proves that the first mirror surface can be generated deterministically, limited to the chosen family, protected against forbidden oracle inputs and post-query observations, and compared against the required controls with explicit bit accounting.

it also shows that the first tiny trainable codec can memorize the training split while failing held-out operation preservation. after the source-observability repair, that is evidence for a local learned-generalization gap, not evidence that the task is asking for hidden fields.

the diagnostic controls localize the current failure more sharply. on the smoke profile, legal source fields are now fully visible and reconstructable. the visible-source codec solves state, action, and joint success at 1.0, so the repaired contract is legally solvable from non-oracle input. the content-routed sparse-read baseline also solves state, action, and joint success at 1.0 by selecting two legal observation records: the committed source event and its next-source velocity support event. it commits 40 bits, compresses the full verbatim record by only 1.3x, and does not fit inside the learned codec budget. this makes the current mirror a source-selection benchmark unless a learned codec can beat sparse read at a tighter useful-bit budget.

the latest learned path remains weak after the sparse-read addition: train joint success is 1.0, validation joint success is 0.0, test joint success is 0.125, learned compression ratio versus verbatim is 2.74x, and the learned-minus-sparse-read joint gap is -0.875. this points next at a learned-codec problem and at a tighter bit-efficiency comparison: the source state and action target are legally observable, but the current local learner does not yet generalize address, payload, velocity, action, or the decoder well enough to compete with shallow legal sparse read.

it also proves that the suite registry can run this surface as a local smoke suite and reject blocked authorization flags if they turn on.

## what this does not prove

this does not prove learned compression.

it does not prove that the current `compression_under_bit_budget` task is a strong compression benchmark. the content-routed sparse-read baseline solves it from legal observations, although at a higher bit cost than the learned compact-code budget.

it does not prove that a neural model can infer schema, residual, address, or provenance codes from observations.

it does not prove that diagnostic oracle exposure is acceptable as a model input. the diagnostic rows are explicitly marked as controls and do not count as learned-codec results.

it does not prove novelty.

it does not authorize full-model integration, simulator work, h200, kaggle, pod, runpod, or paid compute.

## verdict

accepted as the repaired local dataset, baseline-control, sparse-read baseline, first learned-codec smoke surface, source-availability audit, and diagnostic-localization surface for the `compression_under_bit_budget` tiny mirror. the source-observability contract now passes. sparse read solves the task with two legal records but exceeds the compact-code budget. the learned result still fails the engineering and paper-track gates and does not beat sparse read. the next no-paid work is to repair or replace the local learned address/payload/action and decoder-generalization path, then compare it against sparse read under explicit useful-bit accounting before any broader mirror, full-model integration, or paid compute.

## see also

- [[PROJECT_PLAN]]
- [[tests/index]]
- [[synthesis/neural_model_paper_spine]]
- [[synthesis/neural_model_research_test_material_plan]]
- [[synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[synthesis/neural_model_dossier_compression_under_bit_budget_codec]]
- [[synthesis/content_routed_sparse_read_prior]]
- [[synthesis/neural_model_related_work_pressure_matrix]]
- [[synthesis/oracle_compression_frontier_split]]
- [[tests/oracle_compression_analysis_results]]
- [[tests/hard_symbolic_nm_test_material]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/neural_model_lane_trainability_evaluation]]
