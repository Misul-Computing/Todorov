# neural model dossier: memory formation

status: current (as of 2026-04-27).

## claim

the neural model must prove that it can decide what to write before any storage substrate is credited. memory formation is a learned write-selection problem, not only a capacity problem.

## mathematical operation

the candidate write path is:

```text
s_t = surprise(h_t, context_t)
a_t = address(h_t, context_t)
v_t = value(h_t, context_t)
z_t = write_gate(s_t, h_t, context_t)
M_t = update(M_{t-1}, a_t, v_t, z_t)
```

the write gate `z_t` must open for task-relevant events and stay closed for distractors.

[[neural_model_dossier_eligibility_gated_local_commit]] specializes this write path into a delayed mark-then-commit operation with a separate output-exposure state.

## evidence basis

the paid slot-memory runs showed that a substrate can retrieve under hand placement yet fail under learned routing. the output gate also plausibly creates a fixed point: the memory needs an open gate to become useful, and the gate needs useful memory to open.

## failure mode targeted

the model may never write useful information, may write every token indiscriminately, or may keep the memory path closed long enough that gradient descent routes around it.

## required test material

episodes must mark relevant and irrelevant events separately. include cases where the relevant event is rare, delayed, and surrounded by distractors that share surface features.

## success metrics

- write precision and recall on memory-relevant positions
- `joint_success`
- gate-open fraction at relevant versus irrelevant positions
- delayed action success after relevant writes

## controls

- oracle write / learned read
- learned write / oracle read
- hand-opened output gate
- no-memory
- recency-only
- matched always-write baseline

## telemetry

- write frequency
- gate logits and gate-open fraction
- memory-output norm versus residual norm
- gradient norm through write path
- relevant-position coverage

## kill condition

kill the mechanism if writes do not align with relevant events, if oracle writes pass but learned writes fail without a fixable localization, or if always-write performs the same under matched budget.

## see also

- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_research_test_material_plan]]
