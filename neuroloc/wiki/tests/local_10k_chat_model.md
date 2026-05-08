# local_10k_chat_model

status: current (as of 2026-05-08).

test type: laptop-only integrated 10k constrained chat model gate

## summary

`local_10k_chat_model` is the first integrated local chat artifact for the neural-model v1 direction. it trains a compact state-first neural router over dataset-derived memory records, keeps a bounded memory table, supports short-term dialogue state, updates memory records, performs targeted replay, creates a tiny branch-state rehearsal object, refuses unknown prompts, and exposes a command-line chat surface.

this is the current runnable 10k chat model surface. it is still constrained: it does not prove arbitrary chat, open-ended imagination, solved compression, or full-scale language modeling. it proves that the local neural-model foundation can be executed on the laptop as a small state-and-memory responder instead of only as isolated symbolic tests.

## implemented surface

- trainable neural parameters: token-state embeddings plus record-routing state weights
- training method: state-first local binding, not next-token prediction
- memory surfaces: long-term dataset records, short-term dialogue state, memory update, targeted replay, and branch-state rehearsal
- command-line prompt mode: `python neuroloc\simulations\memory\local_10k_chat_model.py --chat --prompt "what does cortex preserve for action"`
- interactive stdin mode: `python neuroloc\simulations\memory\local_10k_chat_model.py --chat`
- response form: `v1 chat: <cue> links to <payload>. source record_<n>.`
- unknown prompt response: `v1 chat: outside grounded memory scope`
- suite: `compression_mirror`

## commands

commands run on 2026-05-08:

- `python -m pytest tests/test_local_10k_chat_model.py -q`
- result: 4 passed, known numpy-on-windows warning
- `python -m pytest tests/test_local_10k_chat_model.py tests/test_simulation_suite.py::test_suite_registry_contract -q`
- result: 5 passed, known numpy-on-windows warning
- `python neuroloc\simulations\suite_runner.py --simulation local_10k_chat_model --profile smoke --output-root codex_local_output\suite_local_10k_chat --timeout 120`
- result: suite completed, 1/1 passed
- `python neuroloc\simulations\memory\local_10k_chat_model.py --chat --prompt "what does cortex preserve for action"`
- result: `v1 chat: cortex links to memory. source record_27.`
- `python neuroloc\simulations\memory\local_10k_chat_model.py --chat --prompt "write a pirate poem about thunder"`
- result: `v1 chat: outside grounded memory scope`

## key smoke outputs

- local model authorized: 1.0
- full model authorized: 0.0
- paid compute authorized: 0.0
- arbitrary chat authorized: 0.0
- next-token training used: 0.0
- state-first training used: 1.0
- trainable parameter count: 7,968
- accounted bits: 61
- grounded response success: 0.96875
- unknown refusal success: 1.0
- route-disabled grounded success: 0.03125
- shuffled-route grounded success: 0.0
- memory update success: at least 0.9
- short-term context success: at least 0.9
- targeted replay success: at least 0.9
- branch-state success: at least 0.9
- final dialogue joint success: at least 0.9
- engineering pass: 1.0

## interpretation

the result is the first runnable local v1 chat foundation. the model has a compact state router and explicit stateful memory operations, so it is closer to the intended neural model than the earlier isolated proof gates. route-disabled and shuffled-route controls collapse grounded response, so the positive grounded-answer score depends on the routed state surface rather than only direct cue lookup. it can be talked to through the command line, but it is grounded only in its dataset-derived records and refuses prompts outside that grounded memory scope.

the result does not prove broad language competence. it does not learn open-ended generation. it does not prove 600x compression. it does not replace the need for fair sparse-read baselines, factor-heldout query forms, larger local datasets, provenance-preserving rewrites, and harder dialogue tests.

## next

harden this surface before any paid compute or full integration:

- larger local dataset ingestion
- factor-heldout query forms
- provenance-preserving rewrite tests
- fair sparse-read and verbatim-memory baselines
- longer multi-turn dialogue
- branch-state tests that require action improvement or reconstruction, not only branch creation

## see also

- [[local_v1_language_model]]
- [[language_grounded_state_density_mirror]]
- [[compression_under_bit_budget_mirror]]
- [[index]]
- [[../PROJECT_PLAN]]
- [[../synthesis/neural_model_paper_spine]]
- [[../synthesis/neural_model_lane_operation_preserving_compression]]
- [[../synthesis/neural_model_lane_memory_replay_imagination]]
- [[../synthesis/neural_model_lane_trainability_evaluation]]
