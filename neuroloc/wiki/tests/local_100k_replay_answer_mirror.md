# local_100k_replay_answer_mirror

status: current (as of 2026-05-09).

test type: compression-first 100k local model candidate over the generated symbolic-language world surface

## summary

`local_100k_replay_answer_mirror` is the first finished local 100k model candidate on the current no-paid symbolic-language surface. it composes four learned parts: typed event binding from randomized generated-language event segments, compact-state write/read/update, delayed compact-code reactivation or rewrite, and bounded answer decoding from internal state. it also adds learned branch rollout from bounded state and a branch program. smoke and hard profiles now both pass the registered suite.

this is not arbitrary chat, not paid-scale trainability, not natural language generation, not broad 3d embodied intelligence, and not full project completion. it is the current finished local model candidate for the compression-first symbolic-language surface. the result is allowed to update the local path because the core gates pass under factor-heldout axes and category controls.

## implemented surface

- script: `neuroloc/simulations/memory/local_100k_replay_answer_mirror.py`
- tests: `tests/test_local_100k_replay_answer_mirror.py`
- suite id: `local_100k_replay_answer_mirror`
- suite category: `compression_mirror`
- four heldout axes: color-shape, color-velocity, shape-velocity, position-velocity phase
- one registered smoke seed
- 2,048 generated local train records across axis runs
- 64 validation records across axis runs
- 128 test records across axis runs
- 89,877 maximum trainable parameters
- 19 committed compact-state bits
- 56 accounted bits after parser/schema cost
- learned bounded answer surface: `answer action_n color_n shape_n pos_n vel_n`
- delayed reactivation after distractor corruption
- compact-state rewrite
- learned branch rollout from bounded state plus branch program

## commands

commands run on 2026-05-09:

- `python -m pytest tests/test_local_100k_replay_answer_mirror.py -q`
- result: 4 passed, known numpy-on-windows warning
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_replay_answer_mirror --profile smoke --timeout-sec 300`
- result: suite completed 1/1 passed
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_replay_answer_mirror --profile hard --timeout-sec 1200`
- first result before repair: failed because hard-profile generated-language positions 21-30 wrapped through a 21-word number vocabulary, making the language surface lossy
- repair: extended `NUMBER_WORDS` in `language_grounded_state_density_mirror.py` through `thirty`, then recalibrated hard-profile bit maximums to 21 committed bits and 60 accounted bits
- final result: suite completed 1/1 passed
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_replay_answer_mirror --profile hard --output-root codex_local_output\suite_l100k_hard --timeout-sec 1200`
- result: suite completed 1/1 passed, hard metrics preserved at `codex_local_output/suite_l100k_hard/local_100k_replay_answer_mirror/local_100k_replay_answer_mirror_metrics.json`
- `$env:PYTHONPYCACHEPREFIX='C:\Users\deyan\Projects\todorov\codex_local_output\pycache_l100k'; python -m py_compile neuroloc\simulations\memory\local_100k_replay_answer_mirror.py tests\test_local_100k_replay_answer_mirror.py neuroloc\simulations\suite_registry.py`
- result: passed

## key smoke outputs

- local model candidate authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- axis count: 4
- run count: 4
- parameter count max: 89,877
- initial joint success min: 0.96875
- initial state success min: 0.96875
- initial action success min: 0.96875
- field accuracy floor: 0.96875
- targeted replay success min: 0.96875
- no replay success max: 0.0
- random replay success max: 0.03125
- recency replay success max: 0.0
- matched-compute dummy replay success max: 0.0
- decoder disabled success max: 0.0
- shuffled answer success max: 0.03125
- rewrite success min: 0.96875
- no rewrite success max: 0.0
- random rewrite success max: 0.0
- branch rollout success min: 1.0
- no branch success max: 0.0
- wrong branch success max: 0.0
- random branch success max: 0.0
- hard-case branch gain min: 1.0
- easy-case branch gain max: 0.0
- matched-budget sparse-read joint success max: 0.0
- uncapped sparse-read joint success min: 1.0
- committed bits max: 19.0
- accounted bits max: 56.0
- engineering pass: 1.0
- example response: `answer action_6 color_1 shape_2 pos_3 vel_-2`

## key hard outputs

- local model candidate authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- parameter count max: 98,817
- initial joint success min: 1.0
- targeted replay success min: 1.0
- rewrite success min: 0.96875
- branch rollout success min: 0.984375
- committed bits max: 21.0
- accounted bits max: 60.0
- engineering pass: 1.0

## category check

implemented operation: learned generated-language event binding, learned compact-state write/read/update, delayed compact-code reactivation after distractors, compact rewrite, bounded answer decoding from state, and learned branch rollout from state plus branch program.

strongest baseline: matched-budget sparse read at 20 bits, uncapped sparse read at 32 selected records, no replay, random replay, recency replay, matched-compute dummy replay, decoder disabled, shuffled answer, no rewrite, random rewrite, no branch, wrong branch, and random branch.

what failed: the first hard-profile attempt failed because the generated-language number vocabulary was too short for hard-profile positions. this was a test-surface bug, not a model-success claim. after repair, no registered smoke or hard gate fails. random replay and shuffled answer retain small accidental smoke success at 0.03125, below their maximum gate and far below targeted replay and initial answer success.

what is not proved: open-ended dialogue, unconstrained natural-language generation, real 3d embodied simulator grounding, long-term consolidation across arbitrary domains, paid-scale optimization, and robust hard-profile performance.

why this can be called the current local 100k model candidate: unlike the demoted responder scaffolds, the artifact uses learned internal state paths for write/read/update, delayed reactivation, rewrite, branch rollout, and answer decoding, while all category controls remain explicit and full-model or paid-compute authorization remains zero.

## verdict

accepted as the first finished 100k local model candidate for the compression-first symbolic-language surface. the next proof target is exact-state 3d world grounding, not another command-line responder harness.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_3d_nm_mirror]]
- [[tests/local_state_write_read_mirror]]
- [[tests/language_grounded_state_density_mirror]]
- [[tests/compression_under_bit_budget_mirror]]
- [[synthesis/neural_model_lane_memory_replay_imagination]]
