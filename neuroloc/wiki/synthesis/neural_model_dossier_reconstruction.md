# neural model dossier: reconstruction

status: current (as of 2026-04-23).

## claim

the neural model may not need verbatim reconstruction, but it must prove that reconstructed content preserves the state needed for the task. semantic success and exact success must be measured separately.

## mathematical operation

the candidate read path is:

```text
p_hat = prior_decoder(q, address, schema, context)
r_hat = residual_decoder(q, address, schema, code)
h_hat = p_hat + r_hat
```

the model succeeds only if `h_hat` supports the required probe, answer, or action.

## evidence basis

prediction plus residual correction improved reconstruction in the correction-field simulations even when the memory substrate did not store more patterns. world-model and discrete-latent work support reconstructing through a learned prior rather than storing full observations.

## failure mode targeted

the model may optimize a reconstruction loss that looks good while losing the variables needed for downstream action, or it may preserve verbatim details that do not matter.

## required test material

episodes must expose both exact target values and latent task variables. include cases where semantically equivalent reconstructions are acceptable and cases where exact values are required.

## success metrics

- exact reconstruction
- semantic reconstruction
- downstream `action_success`
- `joint_success`
- reconstruction error conditioned on task relevance

## controls

- verbatim readout
- prediction-only
- residual-only
- random decoder
- oracle latent state
- no-memory

## telemetry

- prior contribution norm
- residual contribution norm
- reconstruction error by variable
- task-relevant error
- decoder uncertainty or calibration

## kill condition

kill the mechanism if reconstruction improves while action success does not, or if semantic scoring hides failures on exact-recall tasks.

## see also

- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
