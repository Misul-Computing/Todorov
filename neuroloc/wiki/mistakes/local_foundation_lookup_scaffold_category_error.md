# local foundation lookup scaffold category error

status: historical context only. frozen as of 2026-05-08. do not edit.

## summary

the local v1, local 10k, and local foundation artifacts were documented with
names and interpretations that were too close to the final neural-model claim.
the actual code surfaces are useful local scaffolds, but they are not a working
neural model, not a chat model, not brain-like memory, not compression, and not
imagination.

the most serious error was treating branch-state bookkeeping as if it were
related to imagination. in the local foundation surface, the `imagine` command
selects an existing memory code, copies it, adds a branch label, changes the
source string, appends it to branch memory, and renders a template response.
that proves only that the branch-memory list is used. it does not prove latent
rollout, recombination, counterfactual prediction, world-state dynamics, or
hard-case action improvement.

## category map

### memory

what was implemented: routed selection over tiny dataset-derived memory
records, followed by rendering of selected fields.

what was overclaimed or risked: brain-like memory, episodic memory, working
memory, long-term memory, and foundation memory.

correct interpretation: constrained record routing with explicit storage
compartments. the compartment names are architecture targets, not achieved
biological mechanisms.

minimum future gate: learned write and read over internal state, partial-cue
retrieval, interference behavior, forgetting or consolidation behavior,
provenance preservation, and controls against no-memory, recency-only,
shuffled-address, sparse-read, and verbatim-store baselines.

### replay

what was implemented: targeted retrieval or copying of an already selected
record after distraction.

what was overclaimed or risked: replay, consolidation, rewrite, and offline
credit assignment.

correct interpretation: targeted recall scaffold.

minimum future gate: delayed reactivation must change future recall,
compression, rewrite quality, interference resistance, or action success. it
must beat no-replay and random-replay controls under matched compute while
preserving provenance.

### imagination

what was implemented: copy an existing record into a branch object and render a
branch template.

what was overclaimed or risked: imagination, dreaming, branch-state rehearsal,
latent rollout, and counterfactual reasoning.

correct interpretation: branch bookkeeping scaffold.

minimum future gate: learned latent branch rollout or recombination, learned or
explicit dynamics, branch-local uncertainty, provenance, reconstruction or
action improvement, and controls for no-branch, shuffled-branch,
wrong-dynamics, random-branch, and matched compute. hard-case rollout gain must
exceed easy-case gain.

### compression

what was implemented: an identity schema/residual codec boundary and earlier
field codecs on symbolic surfaces.

what was overclaimed or risked: solved compression, storage compression, and
memory-object compression.

correct interpretation: codec boundary and narrow useful-state-density
scaffold. the original source-pair mirror is already demoted as compression
evidence because legal sparse read solved it.

minimum future gate: fewer accounted bits at matched operation success against
fair sparse-read and verbatim baselines, with decoder, schema, parser, and
action-map costs charged or fixed identically across methods.

### chat and language

what was implemented: command-line prompt handling over bounded local records,
with fixed response templates and refusal outside the tiny grounded scope.

what was overclaimed or risked: 10k chat model, local v1 language model, broad
language competence, or arbitrary conversation.

correct interpretation: constrained responder and command-line harness.

minimum future gate: learned response composition or generation over held-out
dialogue, not fixed rendering of selected record fields. it must keep
provenance, refuse unsupported prompts, and survive query-form and factor-held
out tests.

### neural model v1

what was implemented: small local proof scaffolds for state routing, response
plumbing, and explicit compartment boundaries.

what was overclaimed or risked: a working v1 neural model or breakthrough
architecture.

correct interpretation: architecture-boundary scaffold only.

minimum future gate: a written architecture contract before code, specifying
state variables, write/read dynamics, replay, imagination, action and answer
decoding, training objective, controls, telemetry, and kill conditions. the
implementation must use learned internal dynamics rather than a record lookup
path.

## required correction

future documentation must demote the local v1, local 10k, and local foundation
artifacts to scaffolds unless a later proof gate explicitly upgrades them.
status files must stop calling them the current foundation step. new work must
start from a mechanism contract, not from building a prompt wrapper.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_v1_language_model]]
- [[tests/local_10k_chat_model]]
- [[tests/local_foundation_neural_model]]
- [[synthesis/neural_model_lane_memory_replay_imagination]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[synthesis/world_models_imagination_and_planning]]
