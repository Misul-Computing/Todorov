# local_10k_chat_model

status: current (as of 2026-05-08).

test type: demoted laptop-only under-10k constrained responder scaffold

## summary

`local_10k_chat_model` is now demoted to a constrained responder scaffold. it trains a compact state-first router over dataset-derived records, keeps a bounded record table, supports short-term dialogue state, updates records, performs targeted retrieval, creates a tiny branch-copy object, refuses unknown prompts, and exposes a command-line prompt surface.

this is not a chat model in the project sense. it does not prove arbitrary chat, open-ended imagination, solved compression, learned memory, replay, or full-scale language modeling. it proves only that a small command-line harness can route bounded prompts to tiny local records under controls.

## implemented surface

- trainable neural parameters: token-state embeddings plus record-routing state weights
- training method: state-first local binding, not next-token prediction
- scaffold surfaces: long-term dataset records, short-term dialogue state, record update, targeted retrieval, and branch-copy bookkeeping
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
- legacy targeted-replay metric, now interpreted as targeted retrieval success: at least 0.9
- legacy branch-state metric, now interpreted as branch-copy success: at least 0.9
- final dialogue joint success: at least 0.9
- engineering pass: 1.0

## interpretation

the result is a runnable constrained responder scaffold. route-disabled and shuffled-route controls collapse grounded response, so the positive grounded-answer score depends on the routed record surface rather than only direct cue text. it can be prompted through the command line, but it is grounded only in its dataset-derived records and refuses prompts outside that tiny scope.

the result does not prove broad language competence, learned open-ended generation, 600x compression, memory, replay, imagination, or neural-model v1. the corrected interpretation is recorded in [[../mistakes/local_foundation_lookup_scaffold_category_error]].

## next

harden this surface before any paid compute or full integration:

- keep this artifact as a scaffold or baseline
- write the real local v1 architecture contract before more code
- require branch-state tests to prove latent rollout, reconstruction, or hard-case action improvement before using imagination language

## see also

- [[local_foundation_neural_model]]
- [[local_v1_language_model]]
- [[language_grounded_state_density_mirror]]
- [[compression_under_bit_budget_mirror]]
- [[../mistakes/local_foundation_lookup_scaffold_category_error]]
- [[index]]
- [[../PROJECT_PLAN]]
- [[../synthesis/neural_model_paper_spine]]
- [[../synthesis/neural_model_lane_operation_preserving_compression]]
- [[../synthesis/neural_model_lane_memory_replay_imagination]]
- [[../synthesis/neural_model_lane_trainability_evaluation]]
