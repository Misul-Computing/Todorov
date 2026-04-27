# neural model dossier: eligibility-gated local commit

status: current (as of 2026-04-27).

## claim

eligibility-gated local commit is the first cellular-state mechanism candidate for the neural model. the claim is conditional: a useful memory path must separate possible writes, durable commits, and output exposure.

the mechanism is not a biological copy. it is a mathematical contract:

- mark candidate information locally when it may become useful.
- commit it only when later evidence makes it relevant.
- expose committed memory through a bounded output-capacity state so useful memory can affect action without opening every stored item into the residual stream.

the mechanism is accepted only if it improves delayed state/action/joint success above matched controls and uses fewer committed bits than always-write or verbatim storage at equal success.

## mathematical operation

the candidate loop is:

```text
u_t = alpha u_{t-1} + f_x(o_t, h_{t-1})
e_t = lambda_e e_{t-1} + phi(u_t, o_t, local_context_t, surprise_t)

c_t = candidate_value(u_t, o_t, h_{t-1})
a_t = address(u_t, context_t)

k_t = sigmoid(w_e e_t + w_r relevance_t + w_s stability_t + b_k)
commit_t = k_t * c_t

M_t = update(M_{t-1}, a_t, commit_t)

r_t = clip(lambda_r r_{t-1} + insert_t - remove_t, 0, r_max)
m_t = read(M_t, query_t)
y_t = residual_t + r_t * output_gate_t * m_t
```

where:

- `u_t` is fast local carry state.
- `e_t` is a write-permission trace, not a permanent memory.
- `c_t` is the candidate payload.
- `a_t` is the candidate address.
- `k_t` is the commit gate.
- `M_t` is the committed memory object.
- `r_t` is bounded output-capacity state.
- `m_t` is the memory read.
- `y_t` is the residual stream after bounded memory exposure.

this separates four questions that previous runs blurred:

1. did the system notice the possible memory event?
2. did it commit the right event when relevance became clear?
3. did it read the committed event by the right address?
4. did the read affect the answer or action?

## evidence basis

[[cellular_state_storage_gap_map]] ranks this as the first local-state hypothesis cluster. [[cellular_molecular_computational_primitives]] supports the three-plane split: fast carry, intermediate write-permission state, and slow stability or control. [[neural_model_dossier_local_neuron_state]] supplies the local-state substrate. [[neural_model_dossier_memory_formation]] supplies the write-selection failure frame. [[neural_model_dossier_trainability]] supplies the learned/oracle split.

the biological evidence is used only as evidence for the operation shape:

- local biochemical and electrical state can mark recent events before permanent change.
- delayed modulatory or relevance signals can convert a local mark into a longer-lived update.
- output efficacy can change separately from whether a state exists.
- slow homeostatic state can keep gates and outputs from collapsing into closed or noisy fixed points.

none of this proves that the neural model can learn the mechanism. it only makes the mechanism worth testing.

## failure mode targeted

the mechanism targets two coupled failures observed in the project:

1. learned writes do not align with memory-relevant positions, so the model stores distractors, stores everything, or stores nothing useful.
2. useful stored state remains silent because the output path is closed, too weak, or treated as noise.

the target is not lower loss by itself. the target is delayed state/action/joint success with telemetry proving the intended path was used.

## required test material

the first symbolic contract should include these families:

1. delayed relevance local commit: a source event appears early, distractors follow, and a late relevance signal identifies which earlier event matters.
2. negative commit trap: an early candidate looks useful, but later context makes it irrelevant.
3. commit under interference: target and distractor share features, and only one receives late relevance.
4. commit for delayed use: the target must be committed while visible, retained through occlusion, and used when the query is masked.
5. episodic commit reuse: a committed event must be reused after unrelated distractor episodes.
6. contextual commit permission: the same cue requires different commits under different contexts.
7. bounded output exposure: memory content exists, but answer/action succeeds only if the right committed item is exposed.
8. bounded output-capacity competition: several committed states exist, but output capacity is smaller than the number of stored states.
9. crossed commit/exposure split: oracle commit plus learned exposure, learned commit plus oracle exposure, oracle both, learned both.
10. commit compression frontier: compare eligibility-gated commits against always-write and verbatim storage under equal task success.

each episode must expose exact hidden state, observations, query, answer/action, memory-relevant positions, distractor positions, difficulty parameters, bit budget, and expected behavior for oracle and controls.

## success metrics

top-line metrics:

- `state_probe_accuracy`
- `action_success`
- `joint_success`
- exact recall
- delayed-use success
- context-routed action success

write metrics:

- `write_precision`
- `write_recall`
- `commit_f1`
- `commit_latency`
- false-commit rate on tempting distractors
- writes per successful episode

exposure metrics:

- output-capacity precision
- output-capacity recall
- memory-output norm versus residual norm
- exposure noise cost under fixed-open exposure

compression metrics:

- bits committed per successful episode
- useful bits fraction
- always-write penalty under matched budget
- compression frontier against verbatim storage

failure-localization metrics:

- oracle-write / learned-read gap
- learned-write / oracle-read gap
- oracle-commit / learned-exposure gap
- learned-commit / oracle-exposure gap

## falsifying controls

minimum controls:

- oracle
- no-memory
- recency-only
- shuffled-address
- no trace
- random trace
- always-write under matched bit budget
- oracle trace / learned commit
- learned trace / oracle commit
- oracle commit / learned exposure
- learned commit / oracle exposure
- fixed closed exposure
- fixed open exposure
- hand-opened exposure
- matched residual-capacity baseline
- matched compute and parameter budget

the mechanism does not pass if it only beats no-memory. it must also beat recency-only, shuffled-address, always-write, and matched residual or compute controls where relevant.

## telemetry

minimum telemetry:

- local carry norm
- write-permission trace norm
- trace half-life
- commit gate logits
- commit gate open fraction
- write precision and recall by memory-relevant position
- output-capacity state
- output gate logits
- output exposure fraction at answer/action time
- memory-output norm versus residual norm
- address entropy
- address margin
- read concentration
- retention over delay
- bits committed per episode
- gradient norm through trace, commit, read, exposure, and decoder paths
- failure split by write-side, read-side, exposure-side, and decoder-side

telemetry is evidence only when it moves with task metrics. healthy-looking telemetry with unchanged `joint_success` is a failure.

## leakage risks

- the query observation exposes the target answer directly.
- the late relevance marker names the target identity rather than disambiguating context.
- source time, query time, or object index encodes the answer.
- low-cardinality color or shape combinations leak the action formula.
- distractors are easier than targets because only targets receive complete observations.
- recency fails trivially because the target is never recent.
- always-write fails only because of artificial bit accounting rather than interference or exposure competition.
- output exposure can be solved from residual/query features without reading committed state.
- oracle controls use information unavailable to the learned condition.
- randomized controls are not seed-locked.

## kill condition

kill the mechanism if any of these occur:

- always-write performs the same at matched bit or output budget.
- matched residual-capacity state performs the same.
- learned traces do not align with memory-relevant positions.
- gains require oracle labels at test time.
- output gates remain closed.
- output gates open into noise.
- telemetry looks healthy but `joint_success` does not improve.
- compressed commits drop task-relevant state.
- oracle-write passes but learned-write cannot close the gap under a tiny mirror.
- oracle exposure passes but learned exposure cannot affect action.
- no-memory, recency-only, or shuffled-address controls approach the same score.

## next action

the symbolic episode contract is now [[neural_model_symbolic_contract_eligibility_gated_local_commit]]. next, implement the symbolic generator and deterministic evaluators for that contract. do not implement model code. do not add a paid preset. the next executable work is still symbolic/oracle test material plus oracle compression analysis on the existing hard symbolic worlds.

## see also

- [[PROJECT_PLAN]]
- [[cellular_state_storage_gap_map]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_symbolic_contract_eligibility_gated_local_commit]]
- [[cellular_molecular_computational_primitives]]
- [[neural_model_dossier_local_neuron_state]]
- [[neural_model_dossier_memory_formation]]
- [[neural_model_dossier_trainability]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_operation_preserving_compression]]
- [[oracle_compression_analysis_plan]]
- [[tests/hard_symbolic_nm_test_material]]
- [[neural_model_paper_spine]]
- [[substrate_requires_architectural_change]]
- [[INDEX]]
