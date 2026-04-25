# synthetic shared-world bridge

status: current (as of 2026-04-25).

## the claim

phase 2 should not be a bolt-on collection of image, audio, and video tests.
it should be the first extension of the same latent world used in phase 1.
the clean bridge is a synthetic shared world with exact hidden state and several
renderers. one world, many views.

this matters because the project's current risk is not lack of modality. it is
confounded failure. if recognition, recollection, prediction, counterfactuals,
and action-conditioned rollout all fail inside one large multimodal stack, the
result will be uninterpretable. the bridge therefore has to preserve exact
ground truth while letting the same substrate see several sensory surfaces.

## why synthetic first

the recent benchmark landscape splits into three layers:

- synthetic diagnostic worlds
- embodied multimodal simulators
- 2025-2026 world-model benchmark suites

for neuroloc, the first layer should dominate the early bridge. the second layer
is useful after the substrate behaves reliably in controlled worlds. the third
layer is best treated as later reporting and audit infrastructure.

the strongest reasons to start synthetic are:

- exact hidden state is available
- bias can be controlled directly
- online reward and offline probe scores can be separated
- counterfactuals can be generated cheaply and exactly
- multimodal agreement can be checked against the same underlying state

## the design rule: one latent world, several renderers

the right bridge is closer to mugen than to a full household benchmark. start
with one procedural world that emits:

- a symbolic state trace
- a small image stream
- a toy audio stream
- descriptive, predictive, and counterfactual queries derived from the same
  hidden state

the symbolic renderer is not a fallback. it is the audit channel. when the
visual or audio surface fails, the symbolic state tells us whether the failure
is in recognition, memory, or action-conditioned prediction.

## current local footing (2026-04-22)

phase 2 should be designed on top of the phase-1 battery that now actually
exists in the repo, not on top of an abstract wish list. as of 2026-04-22 the
local symbolic core already includes:

- contextual recall and recognition surfaces in `contextual_recall_world.py`
- delayed use in `delayed_cue_world.py`
- one-cue many-value recollection in `multi_association_recall.py`
- interference sweeps in `slot_key_interference_sweep.py`
- episodic separation / completion / novelty in
  `episodic_separation_completion.py`
- episodic replay / reuse in `episodic_replay_reuse.py`
- context-gated routing in `contextual_gate_routing.py`

the bridge should preserve those task families. it should not replace them with
"vision benchmarks" that change the latent question at the same time they add a
new sensory surface.

## the 2026-04-23 support layer

the new research pass makes the bridge easier to define cleanly:

- [[world_models_imagination_and_planning]] gives the interpretation of
  imagination and planning as latent rollout rather than text production
- [[beyond_next_token_for_neural_models]] and
  [[dreamer_muzero_jepa_titans]] give the external-ai comparison frame
- [[visuals_to_phase1_nm_tests]] and [[visuals_to_curriculum_chapters]]
  translate the first visual batch into testing and teaching use
- [[canonical_visual_narratives_world_models]] carries the first stable
  world-model visual set

that means phase 2 no longer needs to improvise its narrative. it now has a
literature shelf, a comparison layer, a translation layer, and a visual layer.

## what the bridge should borrow

### from worldsense

borrow the bias discipline. decorrelate vocabulary, layouts, roles, and answer
keys. do not let the query template reveal the solution class. held-out
compositions matter more than random splits.

### from popgym and memory maze

borrow the memory ladder. phase 2 should still contain cheap partial-observability
tests and offline probes, not only pretty multimodal scenes. short-memory and
long-memory tests should coexist.

### from phyre, cater, clevrer, physion, and intphys 2

borrow the task types:

- object permanence under occlusion
- delayed reveal
- future-event prediction
- possible vs impossible event discrimination
- action-conditioned physical consequence prediction
- explanatory and counterfactual queries

if only one video-style reasoning family is imported early, clevrer is the best
single template because it already separates descriptive, explanatory,
predictive, and counterfactual questions.

### from perception test

borrow the scorecard. the best ontology is not one average score. it is a grid:

- recognition
- memory
- physics
- semantics

crossed with:

- descriptive
- predictive
- counterfactual

this ontology should guide the local simulation battery even if the actual
perception-test dataset is never used.

## what phase 2 should test first

the first bridge tasks should still be small:

- see object, lose sight of it, later localize it
- hear a source, later identify location from vision
- see a hidden interaction, later answer from audio continuation
- bind object identity across disappearance and reappearance
- predict the next state after an action
- reject counterfactual distractors that share start or end state but not the
  valid action history

the important point is that all of these should be judged against the same
underlying latent world. the modalities are different views of one state, not
different tasks glued together afterward.

## what should wait

these are valuable, but they should not be early inner-loop tests:

- real scanned 3d audio-visual simulation
- household embodied benchmarks like alfred and teach
- broad world-generation leaderboards
- prompt-following or judge-model-heavy world-model evaluations

the new 2025-2026 wave is useful here as a warning. worldtest / autumnbench,
mind, worldarena, worldbench, and related suites are important, but most of
them are better as later audits than as the first bridge. they can easily
reward visual plausibility, benchmark-specific interface engineering, or large
stack integration before the substrate itself is trustworthy.

## how this integrates with phase 1 instead of stacking after it

phase 2 should not replace the phase-1 battery. it should reuse it inside a
shared world:

- phase-1 associative recall becomes multimodal identity binding
- phase-1 delayed-use tasks become out-of-sight localization
- phase-1 iterative reasoning becomes counterfactual rollout over the shared
  latent state
- phase-1 episodic reuse becomes recurring-world reuse across multiple episodes

in other words, phase 2 is not "after memory, now vision." it is "the same
latent-world tests, now rendered through more than one sensory surface."

the one major phase-1 gap still open on 2026-04-22 is the latent-world
deliberation / iterative-rollout probe. that is exactly why the bridge should
stay synthetic first. the shared world can host that missing gate instead of
forcing a separate reasoning benchmark later.

## see also

- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[world_models_imagination_and_planning]]
- [[beyond_next_token_for_neural_models]]
- [[dreamer_muzero_jepa_titans]]
- [[canonical_visual_narratives_world_models]]
- [[visuals_to_curriculum_chapters]]
- [[training_objective_vs_architectural_goal]]
- [[substrate_requires_architectural_change]]
- [[slot_memory_design]]
- [[tests/index]]
- [[PROJECT_PLAN]]
- [[tests/run3_cognition_phase1_results]]

## references

- [worldsense](https://arxiv.org/abs/2311.15930)
- [popgym](https://arxiv.org/abs/2303.01859)
- [evaluating long-term memory in 3d mazes](https://arxiv.org/abs/2210.13383)
- [phyre](https://arxiv.org/abs/1908.05656)
- [cater](https://arxiv.org/abs/1910.04744)
- [clevrer](https://arxiv.org/abs/1910.01442)
- [physion](https://arxiv.org/abs/2106.08261)
- [intphys 2](https://openreview.net/forum?id=Xpf5x3mLvn)
- [mugen](https://arxiv.org/abs/2204.08058)
- [soundspaces 2.0](https://arxiv.org/abs/2206.08312)
- [perception test](https://arxiv.org/abs/2305.13786)
- [benchmarking world-model learning](https://arxiv.org/abs/2510.19788)
- [mind](https://arxiv.org/abs/2602.08025)
- [worldarena](https://arxiv.org/abs/2602.08971)
- [worldbench](https://arxiv.org/abs/2601.21282)
- [iworld-bench](https://iworld-bench.com/)
- [v-jepa 2](https://arxiv.org/abs/2506.09985)
