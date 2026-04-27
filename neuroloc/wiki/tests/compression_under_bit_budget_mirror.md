# compression under bit budget mirror

status: current (as of 2026-04-27).

test type: local tiny-mirror dataset, baseline-control, learned-codec, source-availability, and diagnostic-localization surface

script:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`

source contract:
- [[synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]

implemented surface:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`
- `tests/test_compression_under_bit_budget_mirror.py`
- `neuroloc/simulations/suite_registry.py`

## what was done

implemented the first local code surface for the accepted `compression_under_bit_budget` family, the first trainable learned-codec smoke result, the first diagnostic-only failure-localization controls, and the first source-availability/action-ambiguity checks. this is not a passing compression result. it is a controlled negative problem-specification and trainability result: the tiny codec memorizes train split but does not preserve the target operation on held-out episodes, and the legal query-time input does not reliably expose the source fields the codec is asked to preserve.

the surface now provides:

- deterministic train, validation, and test splits by seed.
- filtering to the `compression_under_bit_budget` family only.
- observation streams trimmed to the task query time so post-query observations cannot leak into inputs.
- model-input records that exclude exact hidden state, target answers, oracle codes, family-label shortcuts, future observations, and memory-relevant positions.
- labels and diagnostics kept outside the model-input path.
- evaluation rows computed from the exact contract attached to the dataset record, not from a regenerated episode.
- deterministic controls for oracle codec, verbatim storage, compressed oracle storage, no-memory, recency-only, shuffled-address, random codebook, matched-bit random code, matched-compute no-code, frozen-random encoder learned-decoder, and learned-encoder frozen-random decoder.
- a trainable field codec with non-oracle observation features, predicted state/action, compact code fields, address/schema/residual/action/provenance fields, parameter counts, train loss, and held-out split metrics.
- diagnostic-only controls for learned-code/oracle-decoder, oracle-code/learned-decoder, learned-address/oracle-payload, oracle-address/learned-payload, provenance-exposed learned codec, visible-source-state/oracle-action/oracle-decoder, source-observation/learned-action, provenance-exposed/oracle-decoder, learned-state/oracle-action/oracle-decoder, and oracle-state/learned-action/oracle-decoder.
- source-availability telemetry for source event presence, observation, required visible fields, reconstructable source state, source-query gap, and source-signature action ambiguity.
- explicit diagnostic flags that prevent oracle-exposed rows from counting as accepted learned-codec results.
- explicit bit accounting and telemetry fields for committed bits, address margin, address entropy, read concentration, write frequency, reconstruction error, memory-output norm, and confidence intervals.
- suite-runner gates that require learned-result telemetry while rejecting future-observation leakage, full-model authorization, paid-compute authorization, and blocked-authorization violations.
- suite-registry entry `compression_mirror`.

## validation

commands run on 2026-04-27 after the source-diagnostic update:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py -q`
- result: 19 passed, 2 local warnings
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract tests/test_simulation_suite.py::test_validate_simulation_output_rejects_summary_above_maximum -q`
- result: 21 passed, 1 known numpy-on-windows warning; this command required an unsandboxed local run because pytest's temporary-directory fixture hit windows access-denied errors inside the sandbox.
- `python neuroloc/simulations/suite_runner.py --suite compression_mirror --profile smoke --output-root reports/validation/compression_mirror_source_diagnostics --write-summary`
- result: 1/1 simulation passed
- `python -m py_compile neuroloc/simulations/memory/compression_under_bit_budget_mirror.py neuroloc/simulations/suite_registry.py tests/test_compression_under_bit_budget_mirror.py`
- result: passed
- `python -c "import yaml; yaml.safe_load(open('state/program_status.yaml', encoding='utf-8')); print('yaml ok')"`
- result: passed
- `python -m pytest tests --collect-only -q`
- result: 328 tests collected, 1 known numpy-on-windows warning

the numpy warning is the existing local numpy-on-windows experimental-build warning, not a test failure.

## key smoke outputs

- family count: 1
- policy count: 22
- dataset record count: 24
- learned result count: 24
- diagnostic result count: 240
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
- learned-codec train joint success: 1.0
- learned-codec validation joint success: 0.0
- learned-codec test joint success: 0.0
- learned-codec test state-probe accuracy: 0.0
- learned-codec test action success: 0.0
- learned-codec encoder address accuracy: 0.5
- learned-codec encoder payload accuracy: 0.0
- learned-codec encoder payload position accuracy: 0.0
- learned-codec encoder payload velocity accuracy: 0.25
- learned-codec encoder action accuracy: 0.0
- learned-codec encoder provenance accuracy: 0.75
- learned-codec bits per successful held-out episode: undefined
- learned-codec compression ratio versus verbatim: 2.74x
- learned-codec engineering pass: 0.0
- learned-codec paper-track pass: 0.0
- learned-codec kill-condition count: 1
- source event observed rate: 0.25
- source required fields visible rate: 0.0
- source state reconstructable rate: 0.0
- source-signature action ambiguity rate: 0.0
- visible-source-state/oracle-action/oracle-decoder joint success: 0.0
- visible-source-state/oracle-action/oracle-decoder state success: 0.0
- learned-code/oracle-decoder joint success: 0.0
- oracle-code/learned-decoder train joint success: 1.0
- oracle-code/learned-decoder validation joint success: 0.0
- oracle-code/learned-decoder test joint success: 0.0
- oracle-code/learned-decoder train-test joint gap: 1.0
- learned-address/oracle-payload joint success: 0.5
- oracle-address/learned-payload joint success: 0.0
- provenance-exposed learned-codec joint success: 0.0
- provenance-exposed/oracle-decoder joint success: 0.0
- learned-state/oracle-action/oracle-decoder joint success: 0.0
- oracle-state/learned-action/oracle-decoder joint success: 0.0
- oracle-state/learned-action/oracle-decoder action success: 0.0
- learned action-only failure rate: 1.0
- strongest diagnostic rescue delta: 0.5 from oracle payload with learned address

## what this proves

this proves that the first mirror surface can be generated deterministically, limited to the chosen family, protected against forbidden oracle inputs and post-query observations, and compared against the required controls with explicit bit accounting.

it also shows that the first tiny trainable codec can memorize the training split while failing held-out operation preservation. that is evidence for a trainability or problem-specification gap, not evidence for a useful learned compression mechanism.

the diagnostic controls localize the first failure more sharply. on the smoke profile, the learned address is partly useful, because replacing the payload with oracle values while keeping the learned address rescues joint success to 0.5. the learned payload is not useful: replacing the address with oracle values while keeping the learned payload remains 0.0. the visible-source-state/oracle-action/oracle-decoder control also remains 0.0 because the source event is observed in only 0.25 of test records and the required source fields are visible in 0.0 of test records. the oracle-code/learned-decoder control memorizes train split at 1.0 but scores 0.0 on validation and test, so the learned decoder generalization failure is explicit rather than inferred. provenance exposure does not rescue either the learned codec or the oracle-decoder variant.

this points first at a family/mirror contract problem: the model is asked to learn operation-preserving compression for fields that are not reliably inferable from the legal non-oracle input. only after the source-state and action target are made legally observable, or the contract is redefined as a commit-time memory task with an explicit earlier observation, does it make sense to train another local codec.

it also proves that the suite registry can run this surface as a local smoke suite and reject blocked authorization flags if they turn on.

## what this does not prove

this does not prove learned compression.

it does not prove that a neural model can infer schema, residual, address, or provenance codes from observations.

it does not prove that diagnostic oracle exposure is acceptable as a model input. the diagnostic rows are explicitly marked as controls and do not count as learned-codec results.

it does not prove novelty.

it does not authorize full-model integration, simulator work, h200, kaggle, pod, runpod, or paid compute.

## verdict

accepted as the local dataset, baseline-control, first learned-codec smoke surface, source-availability audit, and diagnostic-localization surface for the `compression_under_bit_budget` tiny mirror. the learned result fails the engineering and paper-track gates. the next no-paid work is to revise the family/mirror contract so source state and action are legally inferable from the non-oracle input path, then rerun the same local controls before any broader mirror, full-model integration, or paid compute.

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
