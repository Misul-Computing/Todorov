# cellular molecular computational primitives

status: current (as of 2026-04-26).

## thesis

the strongest cell-level lesson for this project is not “copy a channel” or “copy a receptor.” it is that biological computation is split across multiple local state variables operating on different timescales. the most useful abstraction is a three-plane model:

- fast electrical state
- intermediate eligibility or calcium-like state
- slow metabolic or homeostatic control state

that split is better grounded than a single recurrent content state that is forced to carry activation, memory, control, and stability at once.

## what looks established

- membranes and channels create fast thresholded dynamics and adaptation
- spines and dendritic branches create local compartments with partially independent state
- nmda coincidence and dendritic plateaus create branch-local commit events
- vesicle release and biochemical gating create stochastic write conditions rather than deterministic write-through
- astrocytes, oligodendrocytes, and metabolic constraints support a slower regulation plane

## what this implies for architecture

the project should treat local write events as gated commitments instead of uniform writes. future architecture work should prefer:

- explicit multi-timescale state
- local write gates tied to more than one condition
- a slow control plane for stability and resource allocation

this does not mean importing molecular detail literally. it means refusing to collapse fast activity, memory eligibility, and stability control into one state variable without a reason.

the current ranked project translation is [[cellular_state_storage_gap_map]].

## what remains speculative

- direct glia-like controller modules
- trainable delay fabrics as a myelination import
- vesicle-style staged writes as a literal memory interface

these ideas are worth preserving, but they are not yet action-ready.

## see also

- [[neural_model_lane_cellular_state_storage]]
- [[cellular_state_storage_gap_map]]
- [[cellular_molecular_neurobiology_research]]
- [[visual_sources_systems_neuroscience]]
- [[cross_scale_building_blocks_for_biological_computation]]
- [[state_action_memory_architecture_direction]]
- [[indexed_reconstruction_compression]]
- [[canonical_visual_narratives_neuroscience]]
