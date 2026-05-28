# local 100k exact-state 3d nm mirror

status: historical context only. frozen as of 2026-05-09 except supersession note.

## date run

2026-05-09.

## status

passed as a local exact-state 3d nm candidate.

this is not an arbitrary chat model, not a paid-scale model, not an external simulator result, and not a full embodied intelligence claim. it is a local deterministic exact-state 3d proof surface over generated episodes, with learned compact-state write/read/replay/rewrite and a bounded answer decoder over compact world fields.

## artifact tested

- `neuroloc/data/nm_3d_worlds.py`
- `neuroloc/simulations/memory/local_100k_3d_nm_mirror.py`
- `tests/test_local_100k_3d_nm_mirror.py`
- `neuroloc/simulations/suite_registry.py`
- hard artifact: `codex_local_output/suite_l100k_3d_hard/local_100k_3d_nm_mirror/local_100k_3d_nm_mirror_metrics.json`

## what was done

the 100k symbolic-language candidate was extended with a deterministic synthetic exact-state 3d episode contract. each episode has hidden object state, occluded observations, action traces, query type, answer labels, memory-relevant positions, distractors, provenance, and bit accounting.

the world tests object permanence, occluded localization, delayed use, action consequence, compact replay, rewrite, and counterfactual exact transition over decoded compact state. the current observation hides the target object position at query time, so matched-budget sparse read cannot solve the task from the present frame.

the implemented branch operation is an exact compact transition over the decoded learned compact state. it proves that the learned internal compact state can be decoded, transformed, and checked under branch controls. it does not prove a learned physics network, learned imagination dynamics, or broad simulator competence.

## key quantitative outputs

hard profile, registered suite:

- maximum trainable parameters: `66559`
- accounted bits: `51`
- initial world-state joint success: `1.0`
- object permanence success: `1.0`
- occluded localization success: `1.0`
- action consequence success: `1.0`
- targeted replay success: `1.0`
- rewrite success: `0.9666666666666667`
- counterfactual exact transition success: `1.0`
- hard-case branch gain: `1.0`
- matched-budget sparse-read success: `0.0`
- no-memory success: `0.0`
- recency-only success: `0.0`
- shuffled-state success: `0.0`
- random-replay success: `0.0`
- no-integration success: `0.0`
- wrong-dynamics success: `0.0`
- no-branch success: `0.0`
- wrong-branch success: `0.0`
- decoder-disabled success: `0.0`
- rewrite provenance success: `1.0`
- branch provenance success: `1.0`
- full-model authorization: `0.0`
- paid-compute authorization: `0.0`
- arbitrary-chat authorization: `0.0`
- engineering pass: `1.0`

## verification commands

- `python -m pytest tests/test_local_100k_3d_nm_mirror.py -q` passed: 7 passed.
- `python -m pytest tests/test_local_100k_3d_nm_mirror.py tests/test_simulation_suite.py::test_suite_registry_contract -q` passed: 8 passed.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_3d_nm_mirror --profile smoke --timeout-sec 300` passed.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_3d_nm_mirror --profile hard --output-root codex_local_output\suite_l100k_3d_hard --timeout-sec 1200` passed.

## category check

implemented operation: deterministic exact-state 3d episode generation plus learned compact-state write/read/replay/rewrite and bounded answer decoding under a local 100k parameter cap.

strongest baseline: matched-budget sparse read over current observation, no-memory, recency-only, shuffled-state, random-replay, no-branch, wrong-branch, decoder-disabled, shuffled-answer, and no-rewrite controls.

what failed during implementation: an attempted learned branch transition over generated 3d trajectories did not generalize, and an early hard-profile registry seed exposed rewrite instability. see [[mistakes/local_100k_3d_branch_rollout_overfit]].

what is not proved: learned physics dynamics, learned latent imagination rollout, external simulator transfer, arbitrary chat, broad language generation, paid-scale trainability, visual grounding, or a full neural-model v1.

why this is promoted only to local exact-state 3d nm candidate: the learned compact-state path passes exact-state 3d controls and preserves provenance under replay/rewrite/exact-transition checks, but the 3d bridge remains deterministic and synthetic.

## verdict

accepted as the prior exact-state 3d proof layer: a local exact-state 3d nm candidate over deterministic synthetic worlds. it superseded the prior 100k symbolic-language candidate, and is now superseded as the top local result by [[tests/local_100k_full_nm]] while remaining the 51-bit baseline for the local full small nm candidate.

## limitations

the synthetic bridge is internal and exact. no external simulator is selected. the branch transition is exact over compact decoded state, not a learned physics network. the model speaks through bounded answer fields and cannot hold open-ended dialogue. no-integration and wrong-dynamics controls now collapse to `0.0`, so the accepted bridge is not merely current-observation lookup.

## evolution link

this extends [[tests/local_100k_replay_answer_mirror]] from symbolic-language state tasks into exact-state 3d grounding.

[[tests/local_100k_full_nm]] extends this result by replacing the component-style compact-state mirror with one trainable local model under 100k parameters and a learned compressed internal state.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_replay_answer_mirror]]
- [[tests/local_100k_full_nm]]
- [[tests/local_state_write_read_mirror]]
- [[synthesis/neural_model_lane_3d_world_physics]]
- [[mistakes/local_100k_3d_branch_rollout_overfit]]
