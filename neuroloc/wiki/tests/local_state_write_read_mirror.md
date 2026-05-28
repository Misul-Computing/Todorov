# local_state_write_read_mirror

status: current (as of 2026-05-09).

test type: local learned compact-state write/read/update component mirror for the 100k compression-first model path

## summary

`local_state_write_read_mirror` composes the typed event-binding bridge from `language_grounded_state_density_mirror` with a separate learned compact-state write/read/update cell. the text/event binder converts randomized symbolic-language event segments into compact fields. the local state cell then writes those fields into an internal recurrent state, reads them back, and performs a second learned update write before readout.

this is a local component mirror, not the final 100k model, not arbitrary chat, not replay, not imagination, and not paid-compute authorization. the result proves a narrower mechanism step: under factor-heldout axes, a learned state cell can preserve a compact operation-bearing code after language-grounded event binding, while zero-state, shuffled-state, no-update, random-update, and matched-budget sparse-read controls fail.

## implemented surface

- script: `neuroloc/simulations/memory/local_state_write_read_mirror.py`
- tests: `tests/test_local_state_write_read_mirror.py`
- suite id: `local_state_write_read_mirror`
- suite category: `compression_mirror`
- four heldout axes: color-shape, color-velocity, shape-velocity, position-velocity phase
- one smoke seed in the registered smoke suite
- 2,048 generated local train records across axis runs
- 64 validation records across axis runs
- 128 test records across axis runs
- 37,719 maximum trainable parameters
- 19 committed state bits
- 56 accounted bits after charging the 37-bit parser/schema surface
- matched-budget sparse read remains the compact-budget control
- uncapped sparse read remains the upper-control verifier

## commands

commands run on 2026-05-09:

- `python -m pytest tests/test_local_state_write_read_mirror.py -q`
- result: 4 passed, known numpy-on-windows warning
- `python neuroloc\simulations\suite_runner.py --simulation local_state_write_read_mirror --profile smoke --timeout-sec 240`
- result: suite completed 1/1 passed

## key smoke outputs

- local mechanism authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- axis count: 4
- run count: 4
- parameter count max: 37,719
- joint success min: 0.96875
- state success min: 0.96875
- action success min: 0.96875
- field accuracy floor: 0.96875
- update joint success min: 1.0
- zero-state joint success max: 0.0
- shuffled-state joint success max: 0.0
- no-update joint success max: 0.0
- random-update joint success max: 0.0
- matched-budget sparse-read joint success max: 0.0
- uncapped sparse-read joint success min: 1.0
- accounted bits max: 56.0
- engineering pass: 1.0

## category check

implemented operation: typed randomized event segments are converted to compact fields by a learned binder, then a separate learned recurrent state cell writes, reads, and updates those fields before operation scoring.

strongest baseline: matched-budget sparse read at 20 bits and uncapped sparse read at 32 selected records. matched-budget sparse read stays at 0.0 joint success; uncapped sparse read stays at 1.0.

what failed: no category claim failed in the registered smoke, but this still does not implement replay, imagination, open-ended language generation, long-term memory consolidation, or a full 100k model.

what is not proved: arbitrary dialogue, natural-language generation, learned latent rollout, replay rewrite, robust multi-step embodied world reasoning, full 100k integration, or paid-scale trainability.

why this is not promoted to the neural model claim: it is a component mirror over generated symbolic-language records. the answer surface is still constrained, and the mechanism lacks replay/rewrite, imagination/rollout, 3d world state, and a complete model training objective.

## verdict

accepted as the next compression-first local component result for the 100k path. the next executable step is to integrate this compact-state write/read/update cell with replay or rewrite and a constrained language-answer decoder under the same category controls.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_3d_nm_mirror]]
- [[tests/compression_under_bit_budget_mirror]]
- [[tests/language_grounded_state_density_mirror]]
