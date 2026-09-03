# Start here

status: definitional. last fact-checked 2026-08-13.

## What Neuroloc is

Neuroloc is a computational-neuroscience wiki and architecture evidence corpus.
It contains 61 mechanism articles, 27 bridge documents, 15 comparison analyses,
58 synthesis pages, 49 research shelves, 33 entity notes, 75 wiki test records, and
26 mistake records. Its canonical plan separates current architecture decisions
from retained historical evidence.

## Who this is for

This reading path assumes familiarity with machine learning, attention,
gradient descent, and linear algebra. It introduces the neuroscience needed to
evaluate which biological mechanisms are useful engineering constraints and
which analogies do not survive scrutiny.

## Current architecture orientation

Use [[Home]] as the landing page, [[INDEX]] as the flat catalog, and
[[PROJECT_PLAN]] as the only canonical project state. The current architecture
contract is [[synthesis/modular_neural_model_stack]]. The latest local trainer
direction is recorded in [[tests/todorov_cls_macbook_session_202608]]. Older
biological bridges, the compressed rotational bilinear recurrence, Candidate G,
and the compression-first path remain evidence and backlog rather than the
live plan.

## Reading path

Start with the introductory material:

1. [[the_brain_in_one_page]] for the compressed neuroscience overview.
2. [[neuroscience_for_ml_engineers]] for a deeper engineering-oriented primer.
3. [[canonical_visual_narratives_neuroscience]] and
   [[canonical_visual_narratives_mind_and_memory]] for the visual summaries.

Then read the adversarial bridge documents:

4. [[plasticity_to_matrix_memory_delta_rule]] asks how far the matrix update
   corresponds to Hebbian models.
5. [[sparse_coding_to_ternary_spikes]] tests the sparse-coding analogy.
6. [[energy_efficiency_to_ternary_spikes]] tests the energy claim.
7. [[dendritic_computation_to_swiglu]] tests the dendritic-gating analogy.
8. [[state_action_memory_architecture_direction]] records the earlier
   state-action research direction as supporting evidence.
9. [[synthesis/modular_neural_model_stack]] defines the present composition and
   proof order.

Use [[glossary]] for unfamiliar terms and [[INDEX]] for the remaining mechanism,
bridge, comparison, synthesis, knowledge, test, and mistake documents.

## Main lesson from the biology audit

Most biological analogies in the historical Todorov architecture are partial:

- Three-value activations are not equivalent to cortical sparse coding.
- The matrix update resembles Hebbian outer-product models but does not
  establish that the brain performs the same key-value binding operation.
- Layer normalization is not biological divisive normalization.
- A residual stream is not a global workspace.
- Complex rotation in a sequence model is not biological oscillatory
  coordination.
- Transformer layers are not cortical layers.

The useful outcome is narrower and more defensible: recurrent state,
competition, adaptive thresholds, local gating, routing, replay, and
multi-timescale dynamics can be tested as mathematical mechanisms without
claiming literal biological identity.

## See also

- [[Home]]
- [[INDEX]]
- [[PROJECT_PLAN]]
- [[synthesis/modular_neural_model_stack]]
- [[the_brain_in_one_page]]
- [[neuroscience_for_ml_engineers]]
- [[research_implications_for_neural_model_direction]]
- [[mathematical_foundations]]
- [[todorov_biology_map]]
- [[glossary]]
