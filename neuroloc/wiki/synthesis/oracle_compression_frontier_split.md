# oracle compression frontier split

status: current (as of 2026-04-27).

## purpose

this page turns the first oracle compression result into the next research contract. it is not a new claim of learned compression. it separates the symbolic families into:

- accepted frontier families: oracle compression clears the current 10x useful-compression threshold while operations and controls remain clean.
- weak frontier families: operations and controls remain clean, but the best oracle ratio is below 10x.

the split exists to prevent a global tiny mirror from averaging away weak families. future learned-codec work must be family-specific.

## source result

source: [[tests/oracle_compression_analysis_results]].

hard profile:

- 448 contracts
- 64 episodes
- 14 families
- operation preservation rate 1.0
- controls preservation rate 1.0
- leakage-free rate 1.0
- accepted rate 0.5714
- eight strong families
- six weak families
- global tiny-mirror recommendation 0.0

## accepted frontier

these families clear the current oracle threshold and may later support narrow learned-codec tests. they do not authorize full-model work or paid compute.

### belief-state formation

best oracle ratio: 13.0x.

interpretation: masked repeated observations can be replaced by a smaller latent state when the world structure is known. this is a candidate for learned latent-state coding, not for payload compression.

next proof package: define a learned state-code mirror that must preserve `state_probe_accuracy`, `action_success`, and `joint_success` while logging reconstruction error and bits written.

### delayed use under partial observability

best oracle ratio: 13.0x.

interpretation: a compact task state can replace the full delayed trace. this is a candidate for memory-as-state, not memory-as-verbatim-copy.

next proof package: train a tiny mirror only if it can beat no-memory and recency-only under delay sweeps while preserving action success.

### episodic reuse after distractors

best oracle ratio: 13.0x.

interpretation: repeated episode structure can be stored as schema plus residual while distractor resistance remains measurable.

next proof package: test whether learned reuse can preserve provenance and resist distractors without storing the entire episode.

### context-gated routing

best oracle ratio: 13.0x.

interpretation: context/action maps can be compact handles when the routing variable is preserved.

next proof package: isolate whether learned compression preserves the route, not only the answer.

### compression under bit budget

best oracle ratio: 13.0x.

interpretation: the constructed bit-budget family exposes a schema/residual advantage over verbatim storage.

next proof package: keep this as the first narrow learned-codec candidate if a mirror is built, because the task already tests the compression claim directly.

### replay/rewrite

best oracle ratio: 26.0x.

interpretation: targeted replay can promote repeated traces into a smaller invariant while preserving provenance.

next proof package: define a replay-rewrite codec that must show rewrite reduces bits without corrupting the original source identity.

### iterative hard-case rollout

best oracle ratio: 19.5x.

interpretation: extra internal compute can be represented as a compact program-like object for hard cases. this is not evidence of learned reasoning.

next proof package: test whether learned rollout gain is larger on hard cases than easy cases under matched compute.

### imagination/recombination

best oracle ratio: 39.0x.

interpretation: imagined branches can be stored as compact branch programs on the current symbolic world. this is the strongest current compression frontier, but it is still oracle-only.

next proof package: define branch reconstruction metrics, uncertainty, provenance, and action-use checks before any learned imagination codec is attempted.

## weak frontier

these families do not justify learned compression yet. the failure is not leakage or control failure. the failure is weak oracle ratio under the current task and codec definitions.

### associative recall

best oracle ratio: 7.09x.

reason: arbitrary payload recall has limited reusable structure. the task is closer to high-entropy lookup than world-state compression.

decision: do not use this family to claim strong compression. keep it as an addressability and exact-recall control unless a future task introduces repeated schemas or lower-entropy payload structure.

### correlated-key interference

best oracle ratio: 7.09x.

reason: the family mainly tests address separation under similar keys. compression is secondary and can hide the real failure if overemphasized.

decision: keep it as an addressing/interference gate. a compression claim here requires a separate address-code proof, not generic schema compression.

### delayed relevance local commit

best oracle ratio: 9.625x.

reason: selective commit helps, but the current code is just below the threshold. the useful operation is delayed write permission, not large compression.

decision: preserve it as a cellular-state gate and do not call it a strong compression result until the task or codec exposes more redundant candidate structure.

### bounded output exposure

best oracle ratio: 9.625x.

reason: bounded exposure limits what leaves memory, but it does not by itself create much additional storage compression.

decision: treat it as an output-capacity and noise-control mechanism, not a compression mechanism.

### crossed commit/exposure split

best oracle ratio: 9.625x.

reason: the family is designed to localize commit-side versus exposure-side failure. the compression counter is subordinate to that localization role.

decision: keep it as a failure-localization test. do not train a codec against it as the first compression target.

### commit compression frontier

best oracle ratio: 9.625x.

reason: the current frontier is near the threshold, but it does not clear it. this makes it the most promising weak family, not an accepted result.

decision: revise this family first if the project wants a cellular-state compression side paper. likely revisions are richer candidate pools, repeated candidate schemas, and a stricter always-write comparison.

## next proof package

the next package is not code in the full model. it is a narrow learned-codec plan for one accepted family.

ranked candidate:

1. compression under bit budget, because the task directly tests the compression claim. proof package: [[neural_model_dossier_compression_under_bit_budget_codec]].
2. imagination/recombination, because the oracle branch-program ratio is strongest but the learned-codec risk is high.
3. replay/rewrite, because provenance and rewrite corruption are concrete falsifiers.
4. belief-state or delayed-use coding, because these are safer state-compression tests but less paper-sharp.

do not start with associative recall, correlated-key interference, or eligibility-commit families as compression claims. keep them as controls and localization surfaces until the weak frontier is revised.

## decision rule

a tiny mirror may be scoped only for a named accepted family with:

- a mechanism dossier or addendum
- exact preserved operation
- non-oracle input fields
- baseline verbatim budget
- no-memory, recency-only, shuffled-address, and verbatim controls
- telemetry proving the compressed path is used
- kill condition for losing state, action, provenance, or address separation

no global mirror, full model integration, paid compute, pod, h200, kaggle, or intervention preset is authorized by this split.

## see also

- [[PROJECT_PLAN]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[tests/oracle_compression_analysis_results]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_compression_stack]]
- [[indexed_reconstruction_compression]]
- [[tests/hard_symbolic_nm_test_material]]
- [[tests/eligibility_gated_local_commit_test_material]]
