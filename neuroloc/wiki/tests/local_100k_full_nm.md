# local 100k full nm

status: historical context only. frozen as of 2026-05-09. do not edit.

## date run

2026-05-09.

## status

passed as the first local full small nm candidate.

this is a single trainable local model object under 100k parameters. it is not a suite of separate component mirrors. it remains local synthetic evidence, not paid-scale trainability, not external-simulator transfer, and not arbitrary chat.

## artifact tested

- `neuroloc/simulations/memory/local_100k_full_nm.py`
- `tests/test_local_100k_full_nm.py`
- `neuroloc/simulations/suite_registry.py`
- hard artifact: `codex_local_output/suite_l100k_full_nm_hard/local_100k_full_nm/local_100k_full_nm_metrics.json`

## what was done

the exact-state 3d generator remains the data source, but success now routes through one `torch.nn.Module` with a shared encoder, learned binary bottleneck, recurrent state cell, replay path, rewrite path, learned branch-transition path, and decoders for world fields, actions, provenance, and bounded language-answer fields.

the model receives legal observation/query features. current-state code supervision is derived from legal observed bridge fields, branch code supervision is derived from branch labels, and target compact world fields are not passed into the model forward path or state cell as inputs.

## key hard outputs

- single trainable module: `1.0`
- local full candidate authorized: `1.0`
- full model authorized: `0.0`
- paid compute authorized: `0.0`
- external simulator authorized: `0.0`
- arbitrary chat authorized: `0.0`
- maximum trainable parameters: `81070`
- learned latent state bits: `24`
- fixed bridge/schema/answer bits: `20`
- accounted bits: `44`
- baseline exact-state 3d accounted bits: `51`
- useful density advantage over 3d baseline: `0.003119429590017826`
- initial world-state joint success: `1.0`
- object permanence success: `1.0`
- occluded localization success: `1.0`
- action consequence success: `1.0`
- targeted replay success: `1.0`
- rewrite success: `1.0`
- learned branch transition success: `1.0`
- bounded language-answer success: `1.0`
- no-memory success: `0.0`
- code-disabled success: `0.0`
- shuffled-code success: `0.0`
- decoder-disabled success: `0.0`
- no-replay success: `0.0`
- random-replay success: `0.0`
- no-branch success: `0.0`
- wrong-branch success: `0.0`
- no-integration success: `0.0`
- wrong-dynamics success: `0.0`
- engineering pass: `1.0`

## verification commands

- `python -m pytest tests/test_local_100k_full_nm.py -q` passed: 5 passed, 1 warning, 162.31 seconds.
- `python -m pytest tests/test_local_100k_full_nm.py tests/test_simulation_suite.py::test_suite_registry_contract -q` passed: 6 passed, 1 warning, 317.44 seconds.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_full_nm --profile smoke --timeout-sec 300` passed: 165.8 seconds.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_full_nm --profile hard --output-root codex_local_output\suite_l100k_full_nm_hard --timeout-sec 1200` passed: 604.5 seconds.

## category check

implemented operation: one trainable local model with learned binary compression, recurrent internal state, replay, rewrite, learned branch transition, action decoding, provenance decoding, and bounded answer decoding.

strongest baseline: current exact-state 3d candidate at 51 accounted bits, no-memory, code-disabled, shuffled-code, decoder-disabled, no-replay, random-replay, no-branch, wrong-branch, no-integration, and wrong-dynamics controls.

what failed during implementation: soft continuous latent codes trained well but failed when thresholded into hard compressed bits; this was not acceptable as compression. see [[mistakes/local_100k_full_nm_soft_code_false_pass]].

what is not proved: arbitrary dialogue, broad natural-language generation, visual grounding, external simulator transfer, paid-scale optimization, open-ended world modeling, or robust multi-seed/multi-axis scaling beyond this registered local proof.

why this can be called a local full small nm candidate: unlike the earlier component mirrors, the model is a single trainable `nn.Module` and all accepted operations route through its learned compressed internal state.

## verdict

accepted as the current top local result and the first local full small nm candidate. it supersedes [[tests/local_100k_3d_nm_mirror]] as the active local model result while keeping the same no-paid-compute, no-broad-full-model, and no-arbitrary-chat limits.

## limitations

the compression code is supervised by binary code targets derived from legal observed bridge state and branch labels. this proves operation-preserving learned compression under the local exact-state bridge, not unsupervised discovery of a world code. hard validation is one registered axis and one seed because the current local training cost is high.

## evolution link

this extends [[tests/local_100k_3d_nm_mirror]] by replacing the explicit compact-state cell with one single trainable compressed-state model.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_3d_nm_mirror]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/neural_model_lane_3d_world_physics]]
- [[mistakes/local_100k_full_nm_soft_code_false_pass]]
