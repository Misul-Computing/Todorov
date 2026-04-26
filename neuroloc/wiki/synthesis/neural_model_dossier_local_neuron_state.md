# neural model dossier: local neuron state

status: current (as of 2026-04-26).

## claim

a useful neural-model unit should be treated as a small local state machine, not as one scalar activation. the candidate local state may include polarity-separated accumulators, membrane or subthreshold carry state, and short eligibility or surprise traces.

## mathematical operation

the minimal candidate state is:

```text
u_t = alpha u_{t-1} + W_x x_t
p_t = split_pos_neg(u_t)
e_t = beta e_{t-1} + phi(u_t, x_t)
y_t = read(g_t, p_t, e_t)
```

where `u_t` is a local carry state, `p_t` separates positive and negative evidence, `e_t` is a short trace, and `g_t` controls how much of the local state is exposed to the rest of the model.

## evidence basis

the complemented-neuron ternary-snn paper and neuronspark-0.9b both point away from hard event purity. the useful translation is preserving membrane or leakage state around discrete events. dendritic-computation evidence also supports branch-local state and nonlinear local coincidence before final output.

## failure mode targeted

hard quantization can discard sign, magnitude, and subthreshold history before memory formation. this can make later write, address, and gate decisions depend on impoverished events rather than on the local evidence that produced them.

## required test material

use aliased latent-world observations where the same immediate symbol can require different hidden-state beliefs depending on recent history. include polarity reversals, subthreshold cues, and distractor events.

## success metrics

- `state_probe_accuracy`
- delayed belief retention
- polarity-confusion rate
- action success after subthreshold-only cues

## controls

- scalar activation only
- sign-only state
- no local carry state
- oracle hidden-state readout
- matched parameter budget

## telemetry

- local-state norm
- positive versus negative accumulator balance
- trace half-life
- gate-open fraction
- downstream contribution norm

## kill condition

kill the mechanism if it improves local probe accuracy but not delayed action success, if the downstream model ignores the local state, or if scalar matched-budget controls perform the same.

## see also

- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_research_test_material_plan]]
