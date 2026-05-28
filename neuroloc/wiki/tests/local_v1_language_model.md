# local_v1_language_model

status: current (as of 2026-05-08).

test type: demoted local dataset-grounded state-router scaffold

## summary

`local_v1_language_model` is now demoted to a constrained state-router scaffold. it trains a tiny router over dataset-derived records, then answers from the predicted bounded record object rather than from a next-token decoder. it also adds a local dialogue loop for record update, targeted retrieval, and branch-copy rehearsal under zero-state and random-replay controls.

this is not arbitrary chat, not solved compression, not replay, not imagination, not brain-like memory, and not full model integration. it is a constrained dataset-grounded responder that tests whether a small local model can learn prompt-to-record-state routing and preserve state, action, and provenance under zero-state and shuffled-state controls.

## implemented surface

- source dataset: local text records extracted from project documents by default, with test injection of explicit source texts
- training objective: state-first cue-to-record binding, not next-token prediction
- response form: `v1 answer: <cue> links to <payload>. source record_<n>.`
- state object: selected record id plus cue, payload, action-equivalent payload, update state, retrieval target, branch-copy state, and provenance
- controls: zero-state answer collapse, shuffled-state answer degradation, and random-replay failure
- local parameter budget: less than 10,000 trainable routing weights
- output artifact: `neuroloc/output/simulation_runs/memory/local_v1_language_model/local_v1_language_model_metrics.json`
- suite: `compression_mirror`

## commands

commands run on 2026-05-08:

- `python -m pytest tests/test_local_v1_language_model.py -q`
- result: 4 passed, known numpy-on-windows warning
- `python -m pytest tests/test_local_v1_language_model.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 5 passed, known numpy-on-windows warning
- `$env:V1_LANGUAGE_PROFILE='smoke'; $env:V1_LANGUAGE_MAX_RECORDS='32'; $env:V1_LANGUAGE_SEED_COUNT='2'; python neuroloc\simulations\memory\local_v1_language_model.py`
- result: wrote `neuroloc/output/simulation_runs/memory/local_v1_language_model/local_v1_language_model_metrics.json`

## key smoke outputs

- dataset grounded: 1.0
- state-first training used: 1.0
- next-token training used: 0.0
- local model authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- seed count: 2
- run count: 2
- record count: 32
- maximum vocabulary size: 85
- maximum trainable parameter count: 2,720
- minimum test joint success: 1.0
- minimum test state success: 1.0
- minimum test action success: 1.0
- minimum provenance success: 1.0
- zero-state joint success max: 0.0
- shuffled-state joint success max: 0.0
- accounted bits: 28
- useful operation success per accounted bit: 0.03571428571428571
- interactive response supported: 1.0
- engineering pass: 1.0
- dialogue gate evaluated: 1.0
- memory update success min: 1.0
- legacy targeted-replay metric, now interpreted as targeted retrieval success min: 1.0
- random replay success max: 0.0
- legacy branch-state metric, now interpreted as branch-copy success min: 1.0
- dialogue final joint success min: 1.0
- dialogue zero-state joint success max: 0.0
- dialogue engineering pass: 1.0

## interpretation

the result proves a narrow scaffold: a very small model can be trained on a local dataset to route natural-ish queries into a bounded record state and answer from that state with provenance. after the later `local_10k_chat_model` documentation changed the default corpus, the cue-binding path was hardened so common status words cannot pull grounded queries away from their target record. it also proves that success depends on the state object, because removing or shuffling that state collapses the operation. the dialogue extension proves constrained record operations: a record can be updated, targeted retrieval can recover the updated record after distraction, random retrieval fails, and a branch-copy response can be created from the current record.

the result does not prove natural conversation, general language understanding, learned compression of arbitrary data, dreaming, simulator grounding, replay, imagination, memory, or full neural-model v1. the corrected interpretation is recorded in [[../mistakes/local_foundation_lookup_scaffold_category_error]]. the next proof step is a real local v1 architecture contract before more code.

## see also

- [[language_grounded_state_density_mirror]]
- [[compression_under_bit_budget_mirror]]
- [[local_10k_chat_model]]
- [[../mistakes/local_foundation_lookup_scaffold_category_error]]
- [[index]]
- [[../synthesis/neural_model_paper_spine]]
- [[../synthesis/neural_model_lane_operation_preserving_compression]]
- [[../synthesis/neural_model_lane_memory_replay_imagination]]
- [[../synthesis/neural_model_lane_trainability_evaluation]]
- [[../PROJECT_PLAN]]
