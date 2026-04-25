# neural model dossier: compression

status: current (as of 2026-04-25).

## claim

compression should mean fewer committed bits per useful memory, not more raw bits per neuron. the candidate mechanism must change what is stored and prove that task-relevant state survives.

## mathematical operation

the current compact-memory object is:

```text
m_i = (a_i, s_i, Q(r_i), source_i)
h_hat_i = P(q, a_i, s_i, context_t) + G(q, a_i, s_i, Q(r_i))
```

where the stored record contains an address, a schema or latent id, a compact residual code, and provenance. reconstruction happens through a shared decoder or model prior.

## evidence basis

the correction-field simulations found no memory-side capacity gain from residual values alone. the indexed reconstruction frame keeps the useful part: store compact handles and reconstruct through a shared prior under a rate-distortion objective.

## failure mode targeted

the model may store too much verbatim content, saturate fixed memory, or reduce bits in a way that destroys the state needed for action.

## required test material

use latent episodes where multiple storage policies can solve the same task: verbatim, surprise-only, compact-address, schema-residual, and no-memory. expose exact bits written per episode.

## success metrics

- bits written per successful episode
- `joint_success`
- rate-distortion Pareto frontier
- compression budget at fixed success
- semantic retention under reconstruction

## controls

- verbatim storage
- no-memory
- recency-only
- random codebook
- oracle schema id
- matched stored-bit budget

## telemetry

- bits committed by field
- write frequency
- codebook usage entropy
- residual norm
- reconstruction error
- task distortion

## kill condition

kill the mechanism if it reduces stored bits but loses action-relevant state, if it cannot beat verbatim storage on the Pareto frontier, or if the gain comes only from oracle schema labels.

## see also

- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[neural_model_research_test_material_plan]]
