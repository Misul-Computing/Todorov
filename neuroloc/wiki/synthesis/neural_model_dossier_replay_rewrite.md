# neural model dossier: replay and rewrite

status: current (as of 2026-04-23).

## claim

retrieval should be tested as a possible rewrite window. replay is useful only if it improves later reuse or compression without corrupting task-relevant state.

## mathematical operation

the candidate loop is:

```text
h_hat_i = read(M, q_i)
c_i = compress(h_hat_i, context)
M' = rewrite(M, i, c_i)
```

the rewrite must either improve later retrieval, reduce stored bits at equal success, or improve reuse after distractors.

## evidence basis

biological reconsolidation and replay motivate the idea, but the project has not proved a model-side rewrite benefit. the current indexed reconstruction frame treats replay rewrite as a proof obligation, not a validated component.

## failure mode targeted

the model may retrieve memories passively, never improve compression, or rewrite useful records into a worse representation.

## required test material

use recurring latent episodes with distractor intervals. include repeated access opportunities, delayed reuse, and a fixed memory budget so rewrite pressure is real.

## success metrics

- reuse advantage
- bits written per successful repeated episode
- post-rewrite recall
- drift after repeated rewrites
- improvement over random replay

## controls

- no replay
- random replay
- oracle targeted replay
- learned targeted replay
- no rewrite
- matched replay compute

## telemetry

- replay selection frequency
- rewrite acceptance rate
- pre/post code size
- pre/post recall
- drift in reconstructed variables

## kill condition

kill the mechanism if learned replay does not beat random replay, if rewrite reduces bits by losing task state, or if gains disappear under matched compute.

## see also

- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_research_test_material_plan]]
