# language_grounded_state_density_mirror

status: current (as of 2026-05-07).

test type: local symbolic-language state-density bridge for constrained message-response behavior

## summary

`language_grounded_state_density_mirror` is the first no-paid bridge from the symbolic useful-state-density surface toward a chat-like interface. it generates text observations and text queries from the same distributed-evidence worlds used by `compression_under_bit_budget_mirror`, trains a small local model from the generated message surface, and emits a text answer.

this does not make the model generally conversational. it proves only that a constrained generated-language message can feed the local state-density mechanism and produce a constrained answer under exact controls.

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

## commands

commands run on 2026-05-07:

- `python -m pytest tests/test_language_grounded_state_density_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 4 passed, known numpy-on-windows warning, pytest cache warning
- `python -m pytest tests/test_language_grounded_state_density_mirror.py tests/test_compression_under_bit_budget_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 33 passed, known numpy-on-windows warning, pytest cache warning
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

the first bag-of-words attempt failed because flat token counts destroyed the time-slot relations. the accepted bridge uses a structured message parser that preserves event, slot, time, and observed-field structure from the generated text. this is still hand-shaped and constrained. the next proof target is to reduce that hand shaping while preserving the useful-state-density gate.

this is a bridge toward message-response training, not evidence that arbitrary chat works. it is also not proof of the full storage/compression thesis.

## see also

- [[compression_under_bit_budget_mirror]]
- [[../synthesis/neural_model_tiny_mirror_contract_compression_under_bit_budget]]
- [[../synthesis/neural_model_lane_operation_preserving_compression]]
- [[../synthesis/neural_model_lane_trainability_evaluation]]
- [[../PROJECT_PLAN]]
