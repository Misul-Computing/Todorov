# neural model dossier: trainability

status: current (as of 2026-04-27).

## claim

trainability is a separate mechanism. a substrate that can solve a task under oracle placement has not proved that gradient descent can discover the write, address, read, and output convention.

## mathematical operation

the trainable loop contains four coupled stages:

```text
write = W_theta(observation, state)
address = A_theta(observation, state)
read = R_theta(memory, query)
output = G_theta(read, residual)
```

the test must localize which stage fails.

## evidence basis

cpu gates showed that hand-placed slot memory can retrieve. paid runs showed that the learned loop still produced zero retrieval. the likely fixed points include closed output gates, low address entropy, weak target-position gradients, and memory outputs treated as noise.

[[neural_model_dossier_eligibility_gated_local_commit]] is the first mechanism dossier that requires oracle/learned splits for trace, commit, and output exposure before model code.

## failure mode targeted

gradient descent may never open the path that would make the mechanism useful. the rest of the model can learn around the memory path before memory becomes aligned.

## required test material

use the same latent worlds under oracle-write, oracle-read, open-gate, auxiliary-target, and address-initialization variants so the failure can be localized without changing the task.

## success metrics

- learned versus oracle gap
- gate-opening time
- address entropy over training
- target-position learning curve
- `joint_success`

## controls

- oracle write / learned read
- learned write / oracle read
- hand-opened gate
- gate-init sweep
- auxiliary-loss sweep
- orthogonal-address initialization
- no-memory and recency-only

## telemetry

- gradient norm by stage
- gate logits
- memory-output norm
- slot/address entropy
- target-position loss
- train/eval recurrence parity

## kill condition

kill the mechanism if oracle variants pass but no trainable path closes the gap, if gradient telemetry shows the mechanism is not receiving useful signal, or if auxiliary loss improves loss without `joint_success`.

## see also

- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_research_test_material_plan]]
