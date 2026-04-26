# neural model dossier: addressing

status: current (as of 2026-04-26).

## claim

addressing must be tested as its own mechanism. a memory substrate has no usable capacity if the model cannot form separated addresses and read them without collapse.

## mathematical operation

for a slot-like read:

```text
l_i = q_t k_i / sqrt(d)
w_i = softmax(l_i)
o_t = sum_i w_i v_i
margin = l_target - max_non_target(l_i)
```

retrieval requires a positive address margin, high read concentration on the intended slot, and enough slot entropy across writes to avoid collapse.

## evidence basis

linear reads mix all stored terms under correlated keys. softmax addressing can suppress non-target slots, but only if the model learns or is initialized with separated addresses. the paid slot runs did not prove learned address formation.

## failure mode targeted

the model may collapse to one slot, spread reads across many slots, or learn addresses that correlate with distractors instead of task identity.

## required test material

use key-value tasks with controlled key correlation, repeated keys with different contexts, and distractor keys that share features with the target.

## success metrics

- exact recall
- degraded-cue recall
- address margin
- slot-usage entropy
- read concentration
- recall versus key-correlation slope

## controls

- shuffled-address
- orthogonal-address initialization
- random-slot write
- oracle address / learned value
- learned address / oracle value
- no-memory and recency-only

## telemetry

- address logits
- address margin distribution
- slot entropy
- slot age
- read concentration
- write collision count

## kill condition

kill the mechanism if recall improves only under oracle addresses, if slot entropy collapses early, or if shuffled-address controls perform similarly.

## see also

- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_research_test_material_plan]]
