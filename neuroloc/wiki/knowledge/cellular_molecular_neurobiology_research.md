# cellular molecular neurobiology research

status: current (as of 2026-04-26).

curated research shelf for cellular and molecular neurobiology as computational substrate. the focus is not generic cell biology. the focus is which molecular and compartmental mechanisms imply reusable compute, memory, timing, or control abstractions for neural models.

## strongest takeaways

- the cell is a multi-timescale system, not a single activation unit
- fast electrical state, intermediate calcium and eligibility state, and slow metabolic or homeostatic state are the most useful abstraction split
- nmda coincidence, dendritic plateaus, and spine-local signaling all argue for local commit-style write events instead of uniform write-through memory
- glia and myelination matter most as slow support, timing, and resource-control planes

## load-bearing source clusters

### membrane, channels, and receptor families

- hodgkin and huxley 1952 — canonical membrane-current formalism
- eyal et al. 2016 — neuron-class differences in signal transfer
- glutamatergic signaling reviews — ionotropic versus metabotropic split
- modern ion-channel and clustered-gating literature

### local compartments and local write events

- calcium influx into hippocampal spines
- nanoscale molecular architecture and calcium diffusion
- dendritic plateau and spike-timing control papers
- btsp and camkii papers linking delayed dendritic signals to memory formation

### vesicle and biochemical control

- synaptic vesicle release machinery reviews
- calmodulin and buffer-control papers
- akap and scaffolded-signaling papers on local biochemical namespaces

### glial and metabolic support

- astrocyte-neuron lactate transport and long-term memory
- neuroglial potassium cycle
- oligodendrocyte-axon metabolic coupling
- metabolic constraints on synaptic learning and memory

## figure candidates from this shelf

- `c1_membrane_and_channel_timescales`
- `c2_receptor_family_timescale_split`
- `c3_spine_microdomain_and_calcium_paths`
- `c6_glial_support_plane`

## what this shelf is for

- grounding new synthesis on multi-timescale state
- supporting future curriculum chapters on cells, synapses, and compartmental neurons
- preventing vague “biological inspiration” by tying claims to local mechanisms and timescales

## see also

- [[neural_model_lane_cellular_state_storage]]
- [[systems_neuroscience_research]]
- [[cross_scale_building_blocks_research]]
- [[cellular_molecular_computational_primitives]]
- [[state_action_memory_architecture_direction]]
- [[canonical_visual_narratives_neuroscience]]
