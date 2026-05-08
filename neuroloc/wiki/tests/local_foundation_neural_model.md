# local_foundation_neural_model

status: current (as of 2026-05-08).

test type: laptop-only neural-model foundation architecture gate

## summary

`local_foundation_neural_model` is the first local v1 surface that makes the intended architecture boundary explicit. it keeps the model under 10k trainable parameters while separating context, working memory, episodic memory, long-term memory, replay buffer, branch memory, and a replaceable memory-object codec boundary.

this is not the extreme compression result. the codec is deliberately simple: a schema/residual identity boundary that records address, schema, payload, action, source, version, and vector state. the point of this pass is to prevent compression from becoming a late bolt-on. every answer, action, update, replay, and branch operation must pass through an encoded memory object that can later be replaced by a stronger codec.

## implemented surface

- trainable neural parameters: token-state embeddings plus memory-route state weights
- training method: state-first memory-object binding, not next-token prediction
- codec boundary: `schema_residual_identity_v0`
- memory surfaces: context state, working memory, episodic memory, long-term memory, replay buffer, and branch memory
- operations: grounded answer, action answer, memory update, short-term context recall, targeted replay, branch-state rehearsal, and unknown-prompt refusal
- command-line prompt mode: `python neuroloc\simulations\memory\local_foundation_neural_model.py --chat --prompt "what does cortex preserve"`
- interactive stdin mode: `python neuroloc\simulations\memory\local_foundation_neural_model.py --chat`
- response form: `foundation chat: <cue> -> <payload>; action <action>; source <source>.`
- suite: `compression_mirror`

## commands

commands run on 2026-05-08:

- `python -m pytest tests/test_local_foundation_neural_model.py -q`
- result: 4 passed, known numpy-on-windows warning
- `python -m pytest tests/test_local_foundation_neural_model.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 5 passed, known numpy-on-windows warning
- `python neuroloc\simulations\suite_runner.py --simulation local_foundation_neural_model --profile smoke --output-root codex_local_output\suite_local_foundation --timeout 120`
- result: suite completed, 1/1 passed
- `python neuroloc\simulations\memory\local_foundation_neural_model.py --chat --prompt "what does cortex preserve"`
- result: `foundation chat: cortex -> memory; action bind; source record_4.`
- `python neuroloc\simulations\memory\local_foundation_neural_model.py --chat --prompt "action cortex"`
- result: `foundation chat: action bind for cortex. source record_4.`
- `python neuroloc\simulations\memory\local_foundation_neural_model.py --chat --prompt "imagine cortex branch rehearsal"`
- result: `foundation chat: branch rehearsal for cortex -> memory; action bind; source branch_from_record_4.`

## key smoke outputs

- local model authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- next-token training used: 0.0
- state-first training used: 1.0
- codec boundary used: 1.0
- simple codec used: 1.0
- extreme compression claimed: 0.0
- trainable parameter count: 5,560
- accounted bits: 62
- grounded response success: 1.0
- unknown refusal success: 1.0
- route-disabled grounded success: 0.041666666666666664
- shuffled-route grounded success: 0.0
- codec-disabled grounded success: 0.0
- replay-disabled success: 0.0
- branch-disabled success: 0.0
- action success: 1.0
- memory update success: 1.0
- context success: 1.0
- targeted replay success: 1.0
- branch-state success: 1.0
- final dialogue joint success: 1.0
- working memory used: 1.0
- episodic memory used: 1.0
- long-term memory used: 1.0
- replay buffer used: 1.0
- branch memory used: 1.0
- engineering pass: 1.0

## interpretation

the result moves the project onto the right track for v1: build the foundation first, keep compression replaceable, and force memory operations through named surfaces. route-disabled, shuffled-route, codec-disabled, replay-disabled, and branch-disabled controls collapse their corresponding operation paths. this avoids the failure mode where a chat surface works but compression has no natural place to enter later.

the result still does not prove broad language competence, open-ended imagination, solved compression, or a paper-worthy architecture. it proves that the local foundation can run as a small state-and-memory architecture with explicit compartments and controls.

## next

harden this surface before any paid compute or full integration:

- ingest larger local text corpora into memory objects
- add factor-heldout query forms over the foundation architecture
- add zero-working-memory, zero-long-term-memory, shuffled-code, and sparse-read baselines
- make branch-state tests require reconstruction or action improvement
- replace `schema_residual_identity_v0` with a learned codec only after the simple boundary is stable

## see also

- [[local_10k_chat_model]]
- [[local_v1_language_model]]
- [[language_grounded_state_density_mirror]]
- [[compression_under_bit_budget_mirror]]
- [[index]]
- [[../PROJECT_PLAN]]
- [[../synthesis/neural_model_paper_spine]]
- [[../synthesis/neural_model_lane_operation_preserving_compression]]
- [[../synthesis/neural_model_lane_memory_replay_imagination]]
- [[../synthesis/neural_model_lane_trainability_evaluation]]
- [[../synthesis/neural_model_related_work_pressure_matrix]]
