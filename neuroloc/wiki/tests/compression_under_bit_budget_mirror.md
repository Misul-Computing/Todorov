# compression under bit budget mirror

status: current (as of 2026-05-08).

test type: local tiny-mirror dataset, baseline-control, content-routed sparse-read, matched-budget sparse-read, distributed-evidence probe, tiny local learned model, factor-heldout local falsification gate, factorized structured local codec, 10k-scale local data-heavy robustness gate, learned-codec, source-availability, and diagnostic-localization surface

script:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`

source contract:
- [[synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]

implemented surface:
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`
- `tests/test_compression_under_bit_budget_mirror.py`
- `neuroloc/simulations/suite_registry.py`

## what was done

implemented the first local code surface for the accepted `compression_under_bit_budget` family, the first trainable learned-codec smoke result, the first diagnostic-only failure-localization controls, the first source-availability/action-ambiguity checks, the repaired source-observability contract, the first content-routed sparse-read baseline over legal observation events, a matched-budget sparse-read control, a distributed-evidence probe, a tiny local learned model trained only on distributed evidence, a factor-heldout local falsification gate, a factorized structured local codec, and a 10k-scale local data-heavy robustness gate. the original source-pair mirror remains demoted as compression evidence and kept as a source-selection, useful-bit, and diagnostic-localization benchmark. the tiny distributed model is a narrow ordinary-split sanity pass, the shared nonlinear tiny model fails factor-heldout compositional generalization, and the factorized structured codec clears four local heldout factor axes across three smoke seeds. it is not a paper claim or full neural-model result.

the surface now provides:

- deterministic train, validation, and test splits by seed.
- filtering to the `compression_under_bit_budget` family only.
- observation streams trimmed to the task query time so post-query observations cannot leak into inputs.
- model-input records that exclude exact hidden state, target answers, oracle codes, family-label shortcuts, future observations, and memory-relevant positions while exposing explicit commit-time markers for the legal source event.
- labels and diagnostics kept outside the model-input path.
- evaluation rows computed from the exact contract attached to the dataset record, not from a regenerated episode.
- deterministic controls for oracle codec, verbatim storage, content-routed sparse read, matched-budget sparse read, compressed oracle storage, no-memory, recency-only, shuffled-address, random codebook, matched-bit random code, matched-compute no-code, frozen-random encoder learned-decoder, and learned-encoder frozen-random decoder.
- a trainable field codec with non-oracle observation features, predicted state/action, compact code fields, address/schema/residual/action/provenance fields, parameter counts, train loss, and held-out split metrics.
- diagnostic-only controls for learned-code/oracle-decoder, oracle-code/learned-decoder, learned-address/oracle-payload, oracle-address/learned-payload, provenance-exposed learned codec, visible-source codec, visible-source-state/oracle-action/oracle-decoder, source-observation/learned-action, provenance-exposed/oracle-decoder, learned-state/oracle-action/oracle-decoder, and oracle-state/learned-action/oracle-decoder.
- source-availability telemetry for source event presence, observation, required visible fields, reconstructable source state, source-query gap, and source-signature action ambiguity.
- explicit diagnostic flags that prevent oracle-exposed rows from counting as accepted learned-codec results.
- explicit bit accounting and telemetry fields for committed bits, address margin, address entropy, read concentration, write frequency, reconstruction error, memory-output norm, and confidence intervals.
- sparse-read telemetry for selected-record count, source-selection recall, next-source-selection recall, false-source-selection rate, record-bit cost, total committed bits, within-budget status, and compression ratio versus verbatim storage.
- distributed-evidence telemetry where the answer is split across four legal observation fragments with no commit markers; uncapped sparse read can solve this probe, while matched-budget sparse read cannot.
- tiny distributed local model telemetry for train/validation/test joint success, oracle-code to learned-decoder success, learned-code to oracle-decoder success, parameter count, committed bits, matched-budget sparse-read gap, and authorization guards.
- factor-heldout telemetry for color-shape pair-band split integrity, train/validation/test overlap, seen marginal classes, learned-codec split success, oracle-code/learned-decoder success, learned-code/oracle-decoder success, matched-budget sparse-read success, committed bits, engineering pass, and authorization guards.
- factorized structured local codec telemetry for legal field-specific input projection, small per-field nonlinear heads, deterministic action computation from learned code fields, factor-heldout split success, fieldwise accuracy, committed bits, matched-budget sparse-read gap, parameter count, and authorization guards.
- 10k-scale local telemetry for four heldout axes, seed count, run count, generated train/validation/test record counts, minimum joint/state/action success, field accuracy floor, matched-budget sparse-read ceiling, committed bits, parameter count, useful operation success per committed bit, matched sparse success per committed bit, useful state density advantage, split cleanliness, seen-marginal rate, and authorization guards.
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
- result: 333 tests collected, 1 known numpy-on-windows warning
- `python -m pytest tests --collect-only -q`
- result: 332 tests collected, 1 known numpy-on-windows warning

commands run on 2026-05-07 after demoting the original mirror as compression evidence and adding matched-budget sparse read plus distributed evidence:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py -q`
- result: 23 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 24 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_simulation_suite.py::test_suite_registry_contract tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q`
- result: 35 passed, 1 known numpy-on-windows warning
- `python -m py_compile neuroloc/simulations/memory/compression_under_bit_budget_mirror.py neuroloc/simulations/suite_registry.py tests/test_compression_under_bit_budget_mirror.py`
- result: passed
- `python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('state/program_status.yaml').read_text(encoding='utf-8')); print('yaml ok')"`
- result: yaml ok
- `python -m neuroloc.simulations.memory.compression_under_bit_budget_mirror smoke`
- result: wrote `neuroloc/output/simulation_runs/memory/compression_under_bit_budget_mirror/compression_under_bit_budget_mirror_metrics.json`

commands run on 2026-05-07 after adding the tiny distributed local learned model:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 25 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_simulation_suite.py::test_suite_registry_contract tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q`
- result: 36 passed, 1 known numpy-on-windows warning
- `python -m neuroloc.simulations.memory.compression_under_bit_budget_mirror smoke`
- result: wrote `neuroloc/output/simulation_runs/memory/compression_under_bit_budget_mirror/compression_under_bit_budget_mirror_metrics.json`
- `python -m pytest tests --collect-only -q`
- result: 334 tests collected, 1 known numpy-on-windows warning

commands run on 2026-05-07 after adding the factor-heldout local falsification gate:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py::test_compression_mirror_factor_heldout_split_is_combinational_and_local_only tests/test_compression_under_bit_budget_mirror.py::test_compression_mirror_factor_heldout_local_model_falsifies_current_tiny_win -q`
- result: 2 passed, 1 known numpy-on-windows warning
- `python -m neuroloc.simulations.memory.compression_under_bit_budget_mirror smoke`
- result: wrote `neuroloc/output/simulation_runs/memory/compression_under_bit_budget_mirror/compression_under_bit_budget_mirror_metrics.json`
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 27 passed, 1 known numpy-on-windows warning

commands run on 2026-05-07 after adding and prosecutor-repairing the factorized structured local codec:

- `python -m pytest tests/test_compression_under_bit_budget_mirror.py::test_compression_mirror_factorized_structured_local_model_repairs_factor_holdout -q`
- result: 1 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py::test_compression_mirror_factorized_vectorizer_ignores_evaluation_contract tests/test_compression_under_bit_budget_mirror.py::test_compression_mirror_factorized_structured_local_model_repairs_factor_holdout -q`
- result: 2 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 30 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_compression_under_bit_budget_mirror.py::test_compression_mirror_tenk_general_local_model_clears_multiple_factor_gates -q`
- result: 1 passed, 1 known numpy-on-windows warning
- `python -m pytest tests/test_simulation_suite.py::test_suite_registry_contract tests/test_nm_hard_symbolic_worlds.py tests/test_compression_under_bit_budget_mirror.py -q`
- result: 41 passed, 1 known numpy-on-windows warning
- `python -m pytest tests --collect-only -q`
- result: 339 tests collected, 1 known numpy-on-windows warning
- `CUBB_TENK_SEED_COUNT=3 CUBB_TENK_TRAIN_EPISODES=2048 CUBB_TENK_VAL_EPISODES=96 CUBB_TENK_TEST_EPISODES=96 CUBB_TENK_EPOCHS=240 python -m neuroloc.simulations.memory.compression_under_bit_budget_mirror smoke`
- result: wrote `neuroloc/output/simulation_runs/memory/compression_under_bit_budget_mirror/compression_under_bit_budget_mirror_metrics.json`

## key smoke outputs

- family count: 1
- policy count: 25
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
- matched-budget sparse-read joint success: 0.0
- matched-budget sparse-read total committed bits: 20.0
- matched-budget sparse-read within budget: 1.0
- distributed-evidence record count: 16
- distributed-evidence sparse-read joint success: 1.0
- distributed-evidence matched-budget sparse-read joint success: 0.0
- distributed-evidence sparse-read total committed bits: 80.0
- distributed-evidence matched-budget sparse-read total committed bits: 20.0
- distributed-evidence sparse-read within budget: 0.0
- distributed-evidence matched-budget sparse-read within budget: 1.0
- distributed-evidence compression-needed flag: 1.0
- tiny distributed local model authorized: 1.0
- tiny distributed full model authorized: 0.0
- tiny distributed paid compute authorized: 0.0
- tiny distributed train record count: 1536
- tiny distributed validation record count: 128
- tiny distributed test record count: 128
- tiny distributed train epochs: 120
- tiny distributed parameter count: 25975
- tiny distributed learned-codec train joint success: 1.0
- tiny distributed learned-codec validation joint success: 0.9921875
- tiny distributed learned-codec test joint success: 1.0
- tiny distributed learned-codec test state success: 1.0
- tiny distributed learned-codec test action success: 1.0
- tiny distributed learned-code/oracle-decoder test joint success: 1.0
- tiny distributed oracle-code/learned-decoder test joint success: 1.0
- tiny distributed matched-budget sparse-read test joint success: 0.0
- tiny distributed learned-codec total committed bits: 19.0
- tiny distributed matched-budget sparse-read total committed bits: 20.0
- tiny distributed learned-minus-matched-budget sparse-read: 1.0
- tiny distributed engineering pass: 1.0
- factor-heldout local model authorized: 1.0
- factor-heldout full model authorized: 0.0
- factor-heldout paid compute authorized: 0.0
- factor-heldout split key: color-shape pair band
- factor-heldout train record count: 512
- factor-heldout validation record count: 64
- factor-heldout test record count: 64
- factor-heldout train epochs: 100
- factor-heldout parameter count: 25975
- factor-heldout train/test bucket overlap: 0
- factor-heldout validation/test bucket overlap: 0
- factor-heldout test colors seen in train: 1.0
- factor-heldout test shapes seen in train: 1.0
- factor-heldout learned-codec train joint success: 1.0
- factor-heldout learned-codec validation joint success: 0.1875
- factor-heldout learned-codec test joint success: 0.03125
- factor-heldout learned-codec test state success: 0.03125
- factor-heldout learned-codec test action success: 0.75
- factor-heldout learned-code/oracle-decoder test joint success: 0.046875
- factor-heldout oracle-code/learned-decoder test joint success: 0.4375
- factor-heldout matched-budget sparse-read test joint success: 0.0
- factor-heldout learned-codec total committed bits: 19.0
- factor-heldout matched-budget sparse-read total committed bits: 20.0
- factor-heldout learned-minus-matched-budget sparse-read: 0.03125
- factor-heldout engineering pass: 0.0
- factorized structured local model authorized: 1.0
- factorized structured full model authorized: 0.0
- factorized structured paid compute authorized: 0.0
- factorized structured train record count: 4096
- factorized structured validation record count: 128
- factorized structured test record count: 128
- factorized structured train epochs: 300
- factorized structured parameter count: 9792
- factorized structured train/test bucket overlap: 0
- factorized structured test colors seen in train: 1.0
- factorized structured test shapes seen in train: 1.0
- factorized structured learned-codec validation joint success: 1.0
- factorized structured learned-codec test joint success: 1.0
- factorized structured learned-codec test state success: 1.0
- factorized structured learned-codec test action success: 1.0
- factorized structured encoder address accuracy: 1.0
- factorized structured encoder payload color accuracy: 1.0
- factorized structured encoder payload shape accuracy: 1.0
- factorized structured encoder payload position accuracy: 1.0
- factorized structured encoder payload velocity accuracy: 1.0
- factorized structured matched-budget sparse-read test joint success: 0.0
- factorized structured learned-codec total committed bits: 19.0
- factorized structured matched-budget sparse-read total committed bits: 20.0
- factorized structured engineering pass: 1.0
- 10k general local model authorized: 1.0
- 10k general full model authorized: 0.0
- 10k general paid compute authorized: 0.0
- 10k general axis count: 4
- 10k general seed count: 3
- 10k general run count: 12
- 10k general total local train record count: 24576
- 10k general total local validation record count: 1152
- 10k general total local test record count: 1152
- 10k general train record count per run: 2048
- 10k general validation record count per run: 96
- 10k general test record count per run: 96
- 10k general train epochs: 240
- 10k general parameter count max: 9792
- 10k general learned-codec test joint success min: 1.0
- 10k general learned-codec test state success min: 1.0
- 10k general learned-codec test action success min: 1.0
- 10k general field accuracy floor: 1.0
- 10k general matched-budget sparse-read test joint success max: 0.0
- 10k general learned-codec total committed bits max: 19.0
- 10k general matched-budget sparse-read total committed bits min: 20.0
- 10k general useful operation success per committed bit min: 0.05263157894736842
- 10k general matched sparse operation success per committed bit max: 0.0
- 10k general useful state density advantage min: 0.05263157894736842
- 10k general bucket clean rate: 1.0
- 10k general marginal seen rate: 1.0
- 10k general axis-seed pass rate: 1.0
- 10k general engineering pass: 1.0
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
- learned-minus-matched-budget-sparse-read joint gap: 0.125
- content-routed sparse-read rescue delta over learned codec: 0.875
- strongest nonlearned source diagnostic rescue delta: 1.0 from the visible-source codec

## what this proves

this proves that the first mirror surface can be generated deterministically, limited to the chosen family, protected against forbidden oracle inputs and post-query observations, and compared against the required controls with explicit bit accounting.

it also shows that the first tiny trainable codec can memorize the training split while failing held-out operation preservation. after the source-observability repair, that is evidence for a local learned-generalization gap, not evidence that the task is asking for hidden fields.

the diagnostic controls localize the current failure more sharply. on the smoke profile, legal source fields are now fully visible and reconstructable. the visible-source codec solves state, action, and joint success at 1.0, so the repaired contract is legally solvable from non-oracle input. the content-routed sparse-read baseline also solves state, action, and joint success at 1.0 by selecting two legal observation records: the committed source event and its next-source velocity support event. it commits 40 bits, compresses the full verbatim record by only 1.3x, and does not fit inside the learned codec budget. matched-budget sparse read commits 20 bits and fails at joint success 0.0. this demotes the current mirror as compression evidence and preserves it as a source-selection and useful-bit benchmark.

the distributed-evidence probe is a stricter local slice. it removes commit markers and splits color/shape, position, and velocity evidence across four legal observation fragments. uncapped sparse read solves it at joint success 1.0 with 80 committed bits and within-budget 0.0. matched-budget sparse read commits 20 bits, stays within budget, and fails at joint success 0.0.

the tiny distributed local model is the first narrow positive learned result on ordinary deterministic splits. using 1536 local train records, 128 validation records, 128 test records, 120 epochs, and 25,975 parameters, it reaches validation joint success 0.9921875 and test joint/state/action success 1.0. oracle-code to learned-decoder test joint success is 1.0, learned-code to oracle-decoder test joint success is 1.0, and end-to-end learned compact-code success is 1.0. it commits 19 bits against matched-budget sparse read at 20 bits and 0.0 joint success. this is local evidence that the distributed-evidence task can be learned by a tiny cpu-trainable model under ordinary splits.

the factor-heldout gate falsifies the current tiny win as a compositional-generalization result. the split withholds color-shape pair bands while keeping every individual test color and test shape represented in train. train/test factor overlap is 0, test colors seen in train is 1.0, and test shapes seen in train is 1.0. under this stronger local split, the same tiny model reaches train joint success 1.0 but validation joint success 0.1875 and test joint success 0.03125. learned-code/oracle-decoder test joint success is 0.046875, oracle-code/learned-decoder test joint success is 0.4375, matched-budget sparse read remains 0.0, and engineering pass is 0.0. this is a useful negative result: the current encoder/decoder can fit ordinary split structure, but it does not yet recombine held-out factor pairs.

the factorized structured local codec repairs this specific failure after prosecutor review caught and removed an evaluator-source shortcut from the encoder path. it replaces the shared nonlinear trunk with legal field-specific input projections, small per-field nonlinear heads, and a deterministic action computation from the learned address, velocity, and position fields. with 4,096 train records, 128 validation records, 128 test records, 300 epochs, and 9,792 parameters, it reaches validation/test joint/state/action success 1.0 on the original factor-heldout split. address, color, shape, position, velocity, and provenance accuracy are all 1.0. it commits 19 bits while matched-budget sparse read remains at 20 bits and 0.0 joint success.

the 10k-scale local data-heavy gate extends that repair across four heldout axes: color-shape, color-velocity, shape-velocity, and position-velocity phase. with three smoke seeds, 12 local training runs, 24,576 generated train records, 1,152 validation records, 1,152 test records, and the same 9,792-parameter model interface, the minimum test joint/state/action success is 1.0, field accuracy floor is 1.0, matched-budget sparse-read max joint success is 0.0, useful operation success per committed bit is 0.05263157894736842, matched sparse success per committed bit is 0.0, useful state density advantage is 0.05263157894736842, and engineering pass is 1.0. this is evidence that the local operation wants factorized code formation and a structured operation decoder rather than a shared tuple-memorizing trunk. it is not arbitrary chat and does not prove the general storage/compression stack.

the latest learned path remains weak after the sparse-read addition: train joint success is 1.0, validation joint success is 0.0, test joint success is 0.125, learned compression ratio versus verbatim is 2.74x, and the learned-minus-sparse-read joint gap is -0.875. this points next at a learned-codec problem and at a tighter bit-efficiency comparison: the source state and action target are legally observable, but the current local learner does not yet generalize address, payload, velocity, action, or the decoder well enough to compete with shallow legal sparse read.

it also proves that the suite registry can run this surface as a local smoke suite and reject blocked authorization flags if they turn on.

## what this does not prove

this does not prove learned compression as a broad project claim.

the tiny distributed local model is a local positive on ordinary deterministic splits only, not a full compression paper result.

the factor-heldout gate does not prove the mechanism is impossible. it proves that the current tiny encoder/decoder/training setup fails the first recombination falsification gate.

the factorized structured codec does not prove the full storage/compression stack. it proves one local symbolic repair and one 10k-scale local data-heavy robustness gate across four heldout axes and three smoke seeds. the follow-up language bridge now includes a negative parser-resistant token-count gate and a positive typed trainable event-binding pass, but the proof package still needs hard-profile sweeps, other world families, memory update, replay, imagination/branch-state, longer dialogue, provenance controls, and stronger learned binding before it can expand.

it does not prove that the current `compression_under_bit_budget` source-pair task is a strong compression benchmark. the content-routed sparse-read baseline solves it from legal observations, although at a higher bit cost than the learned compact-code budget.

it does not prove that a neural model can infer schema, residual, address, or provenance codes from observations.

it does not prove that diagnostic oracle exposure is acceptable as a model input. the diagnostic rows are explicitly marked as controls and do not count as learned-codec results.

it does not prove novelty.

it does not authorize full-model integration, simulator work, h200, kaggle, pod, runpod, or paid compute.

## verdict

accepted as the repaired local dataset, baseline-control, sparse-read baseline, matched-budget sparse-read control, distributed-evidence probe, first learned-codec smoke surface, tiny distributed local learned model, factor-heldout falsification gate, factorized structured local codec, 10k-scale local data-heavy robustness gate, source-availability audit, and diagnostic-localization surface for the `compression_under_bit_budget` tiny mirror. the original source-pair task remains demoted as compression evidence. the ordinary distributed-evidence slice has a narrow local pass, the shared nonlinear factor-heldout model fails hard, and the factorized structured local codec clears four heldout factor axes across three smoke seeds. follow-up bridges now include [[language_grounded_state_density_mirror]] for constrained symbolic-message state and [[local_v1_language_model]] for the first dataset-grounded local v1 state responder with constrained update, targeted replay, random-replay failure, and branch-state controls. the next no-paid work is robustness: longer grounded dialogue, provenance-preserving rewrites, factor-heldout query forms, larger local dataset ingestion, fair sparse-read baselines, and stronger learned binding before any broader mirror, full-model integration, or paid compute.

## see also

- [[PROJECT_PLAN]]
- [[tests/index]]
- [[language_grounded_state_density_mirror]]
- [[local_v1_language_model]]
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
