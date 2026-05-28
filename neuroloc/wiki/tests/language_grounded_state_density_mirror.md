# language_grounded_state_density_mirror

status: current (as of 2026-05-08).

test type: local symbolic-language state-density bridge for constrained message-response behavior

## summary

`language_grounded_state_density_mirror` is the first no-paid bridge from the symbolic useful-state-density surface toward a constrained message-response surface. it generates text observations and text queries from the same distributed-evidence worlds used by `compression_under_bit_budget_mirror`, trains a small local model from the generated message surface, and emits a text answer.

this does not make the model generally conversational. it proves only that a constrained generated-language message can feed the local state-density mechanism and produce a constrained answer under exact controls.

the parser-resistant learned token-count extension is implemented and negative. when stable `time_`, `slot_`, `color_`, `shape_`, and `pos_` prefixes are removed, prompt templates are randomized, event order is shuffled, irrelevant clauses are injected, and the learned encoder is restricted to a bounded local state from token counts, the learned path does not preserve the operation. that result demotes the learned-token-count claim.

the follow-up event-binding parser baseline keeps the randomized message surface but preserves event segments before committing a bounded local state. it reaches 1.0 minimum joint/state/action success across the same four heldout axes and two smoke seeds, emits a constrained answer, and collapses under zero-state or shuffled-state controls. it also charges 37 parser/schema bits in addition to the 19 state bits. the typed trainable segment binder then learns event fields from randomized event segments with 8,856 parameters and clears the local engineering gate at minimum heldout joint/state/action success 0.9583333333333334. this is a constrained grounded typed local-state responder, not arbitrary chat, and not solved compression.

## implemented surface

- generated prompt form: `observations ... question action_for slot_n`
- generated response form: `answer action_n color_n shape_n pos_n vel_n`
- four heldout axes: color-shape, color-velocity, shape-velocity, position-velocity phase
- two smoke seeds
- eight local axis-seed runs
- 16,384 generated train records
- 768 validation records
- 768 test records
- 3,847 maximum trainable parameters
- 19 committed learned bits
- matched sparse read is measured on the same generated-record contracts used to produce the language prompts, and remains a zero-success compact-budget control
- parser-resistant prompt form removes the stable prefix tokens and uses four event template families plus four query template families
- parser-resistant learned text encoder uses no handcrafted field extraction, commits through a bounded local state, and reports zero-state plus shuffled-state ablations
- parser-resistant result is negative: the learned text encoder has minimum test joint success 0.0 and engineering pass 0.0, while uncapped sparse read remains 1.0
- event-binding parser baseline keeps randomized prompts, removed stable prefixes, shuffled order, irrelevant clauses, and bounded local-state commitment
- event-binding parser-supported result is positive as a baseline: minimum test joint/state/action success 1.0, 19 state bits plus 37 parser/schema bits for 56 accounted bits, matched-budget sparse read 0.0 at 20 bits, uncapped sparse read 1.0, zero-state joint max 0.0, shuffled-state joint max 0.020833333333333332, and parser-supported foundation pass 1.0
- typed trainable event-binding result is positive as a local foundation gate: 8,856 parameters, minimum test joint/state/action success 0.9583333333333334, field floor 0.9583333333333334, zero-state joint max 0.0, shuffled-state joint max 0.020833333333333332, and engineering pass 1.0

## commands

commands run on 2026-05-07:

- `python -m pytest tests/test_language_grounded_state_density_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 4 passed, known numpy-on-windows warning, pytest cache warning
- `python -m pytest tests/test_language_grounded_state_density_mirror.py tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 37 passed, known numpy-on-windows warning; the typed event-binding gate makes this focused suite slow on windows
- `python -m pytest tests/test_language_grounded_state_density_mirror.py -q`
- result: 7 passed, known numpy-on-windows warning
- `python -m pytest tests/test_language_grounded_state_density_mirror.py::test_event_binding_foundation_clears_randomized_local_state_gate -q`
- result: 1 passed, known numpy-on-windows warning
- `python -m pytest tests/test_language_grounded_state_density_mirror.py::test_event_binding_foundation_clears_randomized_local_state_gate tests/test_language_grounded_state_density_mirror.py::test_event_binding_responder_uses_bounded_state_for_coherent_answer -q`
- result: 2 passed, known numpy-on-windows warning
- `python -m pytest tests/test_language_grounded_state_density_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 8 passed, known numpy-on-windows warning
- `python neuroloc\simulations\suite_runner.py --simulation language_grounded_state_density_mirror --profile smoke --output-root codex_local_output\suite_lgsd_event_binding --timeout 600`
- result: suite completed 1/1 passed
- `python neuroloc\simulations\suite_runner.py --simulation language_grounded_state_density_mirror --profile smoke --output-root codex_local_output\suite_lgsd_event_binding_typed --timeout 900`
- result: suite completed 1/1 passed
- `python -m pytest --collect-only -q`
- result: 346 tests collected, known numpy-on-windows warning
- `python -m pytest tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 1 passed, known numpy-on-windows warning
- `python neuroloc\simulations\suite_runner.py --simulation language_grounded_state_density_mirror --profile smoke --output-root codex_local_output\suite_lgsd_parser_resistant --timeout 300`
- result: suite completed 1/1 passed
- `$env:PYTHONPYCACHEPREFIX='C:\Users\deyan\Projects\todorov\codex_local_output\pycache'; python -m py_compile neuroloc\simulations\memory\language_grounded_state_density_mirror.py neuroloc\simulations\suite_registry.py tests\test_language_grounded_state_density_mirror.py`
- result: passed
- `python neuroloc\simulations\suite_runner.py --simulation language_grounded_state_density_mirror --profile smoke --output-root codex_local_output\suite_lgsd --timeout 300`
- result: suite completed 1/1 passed
- `python neuroloc\simulations\suite_runner.py --simulation language_grounded_state_density_mirror --profile smoke --output-root codex_local_output\suite_lgsd_verify --timeout 300`
- result: suite completed 1/1 passed
- `python neuroloc\simulations\suite_runner.py --simulation language_grounded_state_density_mirror --profile smoke --output-root codex_local_output\suite_lgsd_verify2 --timeout 300`
- result: suite completed 1/1 passed after replacing constant sparse-read fields with the measured matched-budget sparse-read control
- `$env:SIM_OUTPUT_DIR='C:\Users\deyan\Projects\todorov\codex_local_output\lgsd_default'; python -m neuroloc.simulations.memory.language_grounded_state_density_mirror smoke`
- result: wrote `codex_local_output/lgsd_default/language_grounded_state_density_mirror_metrics.json`
- `python -c "import yaml; yaml.safe_load(open('state/program_status.yaml', encoding='utf-8')); print('yaml ok')"`
- result: yaml ok
- `git diff --check`
- result: passed

## key smoke outputs

- local model authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- constrained message-response supported: 1.0
- axis count: 4
- seed count: 2
- run count: 8
- total train records: 16,384
- total validation records: 768
- total test records: 768
- maximum parameter count: 3,847
- minimum test joint success: 1.0
- minimum test state success: 1.0
- minimum test action success: 1.0
- field accuracy floor: 1.0
- learned committed bits: 19
- matched sparse-read bits: 20
- matched sparse-read joint success max: 0.0
- useful operation success per committed bit: 0.05263157894736842
- useful state density advantage: 0.05263157894736842
- engineering pass: 1.0

## parser-resistant gate output

- gate evaluated: 1.0
- local model authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- event template families: 4
- query template families: 4
- stable prefix dependency removed: 1.0
- learned text encoder reported: 1.0
- deterministic parser reported: 1.0
- local-state ablation reported: 1.0
- total train records: 16,384
- total test records: 768
- maximum parameter count: 1,779
- learned committed bits: 19
- parser/schema cost bits reported: 37
- minimum learned test joint success: 0.0
- minimum learned test state success: 0.0
- minimum learned action success: 0.13541666666666666
- field accuracy floor: 0.052083333333333336
- matched-budget sparse read joint success max: 0.0
- uncapped sparse read joint success min: 1.0
- zero-state joint success max: 0.0
- shuffled-state joint success max: 0.0
- engineering pass: 0.0
- claim downgraded to structured bridge: 1.0

## event-binding local-state foundation output

- foundation evaluated: 1.0
- parser baseline reported: 1.0
- trainable encoder reported: 1.0
- local mechanism authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- stable prefix dependency removed: 1.0
- event template families: 4
- query template families: 4
- axis count: 4
- seed count: 2
- run count: 8
- total train records: 16,384
- total validation records: 768
- total test records: 768
- maximum rule cost score: 629
- committed state bits: 19
- parser/schema cost bits: 37
- accounted bits: 56
- minimum test joint success: 1.0
- minimum test state success: 1.0
- minimum test action success: 1.0
- field accuracy floor: 1.0
- zero-state joint success max: 0.0
- shuffled-state joint success max: 0.020833333333333332
- matched-budget sparse read joint success max: 0.0
- matched-budget sparse read bits: 20
- uncapped sparse read joint success min: 1.0
- useful operation success per accounted bit: 0.017857142857142856
- useful state density advantage after parser/schema cost: 0.017857142857142856
- parser-supported foundation pass: 1.0
- trainable segment parameter count: 8,856
- trainable segment minimum joint success: 0.9583333333333334
- trainable segment minimum state success: 0.9583333333333334
- trainable segment minimum action success: 0.9583333333333334
- trainable segment field accuracy floor: 0.9583333333333334
- trainable segment zero-state joint success max: 0.0
- trainable segment shuffled-state joint success max: 0.020833333333333332
- trainable segment loss mean: 12.421865701675415 to 0.07196207623928785
- trainable useful operation success per accounted bit: 0.01711309523809524
- trainable useful state density advantage after parser/schema cost: 0.01711309523809524
- trainable engineering pass: 1.0
- claim downgraded to parser-supported foundation: 0.0

## example constrained exchange

prompt:

```text
observations time_0 slot_0 color_0 shape_1 ; time_1 slot_0 pos_2 ; time_1 slot_3 color_3 shape_0 pos_9 ; time_2 slot_2 color_2 shape_3 pos_8 ; time_3 slot_0 pos_4 ; time_3 slot_1 color_1 shape_2 pos_7 ; time_4 slot_0 pos_5 question action_for slot_0
```

response:

```text
answer action_3 color_0 shape_1 pos_4 vel_1
```

## interpretation

the first bag-of-words attempt failed because flat token counts destroyed the time-slot relations. the accepted bridge uses a structured message parser that preserves event, slot, time, and observed-field structure from the generated text. the parser-resistant learned token-count extension confirms that this is not a minor limitation: a token-count learned encoder with a bounded local state fails completely on the randomized prompt surface.

the event-binding foundation is the first repair baseline and the first typed trainable local-state responder on the randomized symbolic-message surface. it does not return to the stable prefix parser, and it does not use arbitrary future state; it binds randomized event segments into a compact local state and answers from that state. the trainable segment binder passes because field-specific typed heads prevent the heldout color-shape recombination shortcut that broke unconstrained shared learners. the next proof target is memory update, replay, imagination, provenance over longer grounded dialogue, and stricter learned binding without hiding parser/schema cost.

this is a bridge toward message-response training, not evidence that arbitrary chat works. the later [[local_10k_chat_model]] and [[local_foundation_neural_model]] artifacts are demoted scaffolds after [[../mistakes/local_foundation_lookup_scaffold_category_error]]. their command-line response, targeted retrieval, and branch-copy paths do not prove chat, memory, replay, imagination, or the full storage/compression thesis.

## see also

- [[local_v1_language_model]]
- [[local_10k_chat_model]]
- [[local_foundation_neural_model]]
- [[../mistakes/local_foundation_lookup_scaffold_category_error]]
- [[compression_under_bit_budget_mirror]]
- [[../synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[../synthesis/neural_model_paper_spine]]
- [[../synthesis/neural_model_lane_operation_preserving_compression]]
- [[../synthesis/neural_model_lane_trainability_evaluation]]
- [[../PROJECT_PLAN]]
