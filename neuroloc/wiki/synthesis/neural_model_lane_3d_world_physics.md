# neural model lane: 3d world model and physics

status: current (as of 2026-04-26).

## thesis

the target model is a 3d latent world-memory model. it learns memory, physics, reasoning, and imagination in embodied worlds where exact hidden state is available for evaluation. language and discrete actions are co-primary output surfaces. language is not the sole objective. drawing or image generation is not an early output surface.

the simulator is not chosen yet. simulator selection is a research gate.

## ranked unknowns

1. which existing simulator can provide customizable embodied 3d worlds, deterministic runs, exact hidden state, physics variables, occlusion, object permanence, discrete actions, and cheap local validation.
2. how to export an episode contract compatible with the hard symbolic tests.
3. how world-grounded language should describe exact state without leaking answers.
4. which first tasks best test object permanence, occlusion, delayed use, simple dynamics, and counterfactual physics queries.
5. whether candidate representation losses improve world state without becoming the assumed core method.

## evidence base

- [[synthetic_shared_world_bridge]] defines the exact-state multi-view bridge.
- [[world_models_imagination_and_planning]] frames latent rollout, action, and planning.
- [[phase1_evaluation_surface_for_neural_models]] defines the state/action evaluation surface that 3d worlds must preserve.
- [[neural_model_paper_spine]] states the future flagship paper frame.

## simulator-selection gate

a candidate simulator must support:

- customizable embodied 3d worlds
- deterministic runs
- exact hidden state
- physics variables
- object permanence and occlusion tasks
- discrete actions
- world-grounded language generation
- cheap local or cpu validation where possible
- exportable episode contracts compatible with hard symbolic tests

no simulator is selected until this gate is filled with candidates and failure modes.

## proof gates

- define hidden state, observations, actions, queries, answers, memory-relevant positions, distractors, difficulty parameters, and oracle/no-memory behavior.
- prove that world-grounded text does not leak the target answer.
- require symbolic equivalents before training.
- require object permanence, occlusion, delayed use, simple dynamics, and counterfactual physics tasks to beat no-memory and recency-only controls.
- preserve exact state for state_probe_accuracy, action_success, joint_success, and compression accounting.

## side-paper candidates

- exact-state embodied 3d neural-model test material, if the simulator export exposes a reusable benchmark contract.
- world-grounded language plus action evaluation, if it cleanly separates language fluency from world-state competence.
- counterfactual physics under compact world state, if compression and imagination interact cleanly.

## kill conditions

- the simulator cannot expose exact hidden state or deterministic replay.
- language descriptions leak answers or make memory unnecessary.
- tasks can be solved by local observation or recency-only policies.
- physics variables are too inaccessible or too expensive for local validation.
- the environment forces image generation or unrelated multimodal scope before the memory and action gates are proved.

## next action

create a simulator-selection matrix. do not pick a simulator until the matrix records candidates, supported gates, exact-state access, local cost, language support, and incompatibilities.

## see also

- [[PROJECT_PLAN]]
- [[neural_model_paper_spine]]
- [[neural_model_research_test_material_plan]]
- [[synthetic_shared_world_bridge]]
- [[world_models_imagination_and_planning]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_memory_replay_imagination]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_lane_project_operations]]
