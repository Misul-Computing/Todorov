# eptesicus laboratories research wiki

status: current (as of 2026-06-18, active lane reframed to the neural-machine architecture program; the teaching curriculum moved to backlog).

## active workstream: neural-machine architecture program

the active lane is the neural-machine architecture program (approach a: attention does exact recall, a recurrent state does cheap world-tracking, eidetic compression deferred). it is in cpu validation now -- the v0.1 toy (`v01/`) and the feel bench, where a located, accumulated touch sense turns an otherwise unsolvable task solved (blind 0.14, real-touch 1.00, fake-touch 0.15 control). candidate F (affect-gated write) closed negative; its shuffled-surprise control is the live anomaly, promoted to candidate G (stochastic write gain). the six-lane neural-model research and the compression results are the research substrate that feeds the machine. the teaching PDF curriculum (`~/.claude/plans/compressed-dancing-haven.md`) is preserved and reviewable but is backlog; see `INDEX.md` section "teaching curriculum" for the chapter list. paid compute is gated on funding plus a cpu-validated intervention.

![curriculum scale summary](assets/statistics/s2_curriculum_scale_summary.svg)

## neuroloc -- biological neural computation

the neuroloc wiki maps brain computation mechanisms to todorov's CRBR framework. 61 mechanism articles, 28 bridge documents, 55 synthesis articles, 15 comparison articles, 33 entity notes, 70 test records, 22 mistake docs, 49 knowledge articles, and 7 concept articles.

the architecture program is no longer described only as "pick the next paid run." the wiki records a broader cpu-first method: phase 1 is judged by state + action, not next-token prediction alone, and the implemented symbolic `biology_phase1` battery is part of the active cpu-validation surface.

the 2026-04-23 research pass added a second layer on top of that backlog method: new literature shelves, new synthesis pages on replay, routing, world models, and multi-timescale computation, plus a first batch of recreated visuals that can be reused in both the wiki and the teaching curriculum.

![wiki catalog structure summary](assets/statistics/s5_wiki_catalog_structure.svg)

start here: [[start_here]]

role split:
- this page = landing page and current orientation
- [[start_here]] = guided reading path for a new reader
- [[INDEX|catalog]] = flat catalog of all wiki compartments
- [[PROJECT_PLAN]] = authoritative current state and decision rules

### quick navigation
- [[INDEX|catalog]] -- master catalog of all articles
- [[PROJECT_PLAN]] -- canonical project state
- [[OPERATING_DIRECTIVE]] -- binding rules for wiki and state updates
- [[log]] -- chronological record of all wiki operations
- [[tests/index|tests]] -- dated records of completed simulations and experiments
- [[research_implications_for_neural_model_direction]] -- ranked summary of what the latest research changes

### key bridge documents (biology -> todorov)
- [[neuron_models_to_atmn|neuron models -> ATMN]]
- [[plasticity_to_matrix_memory_delta_rule|plasticity -> matrix-memory delta rule]]
- [[sparse_coding_to_ternary_spikes|sparse coding -> ternary spikes]]
- [[lateral_inhibition_to_adaptive_threshold|lateral inhibition -> adaptive threshold]]
- [[oscillations_to_mamba3_rotation|oscillations -> Mamba3 rotation]]
- [[memory_systems_to_matrix_memory_and_compressed_attention|memory systems -> matrix memory + compressed attention]]
- [[dendritic_computation_to_swiglu|dendritic computation -> SwiGLU]]
- [[spatial_computation_to_pga|spatial computation -> PGA]]
- [[global_workspace_to_residual_stream|global workspace -> residual stream]]
- [[positional_encoding_to_rope|positional/phase coding -> RoPE]]
- [[normalization_to_rmsnorm|biological normalization -> RMSNorm]]

### synthesis (cross-cutting themes)
- [[phase1_evaluation_surface_for_neural_models|phase 1 evaluation surface for neural models]]
- [[synthetic_shared_world_bridge|synthetic shared-world bridge]]
- [[substrate_requires_architectural_change|substrate requires architectural change]]
- [[training_objective_vs_architectural_goal|training objective vs architectural goal]]
- [[research_implications_for_neural_model_direction|research implications for neural model direction]]
- [[beyond_next_token_for_neural_models|beyond next-token for neural models]]
- [[world_models_imagination_and_planning|world models, imagination, and planning]]
- [[working_memory_as_controlled_access|working memory as controlled access]]
- [[attention_as_precision_and_routing|attention as precision and routing]]
- [[cross_scale_building_blocks_for_biological_computation|cross-scale building blocks for biological computation]]
- [[sparsity_from_biology_to_ternary_spikes|sparsity: biology to ternary spikes]]
- [[timescale_separation|timescale separation]]
- [[local_vs_global_computation|local vs global computation]]
- [[compression_and_bottlenecks|compression and bottlenecks]]
- [[recurrence_vs_feedforward|recurrence vs feedforward]]

### current backlog method
- [[phase1_evaluation_surface_for_neural_models|phase 1 battery]] -- recognition, recollection, interference resistance, delayed use, episodic reuse, and later iterative reasoning, with state/action/joint-success metrics
- [[synthetic_shared_world_bridge|phase 2 bridge]] -- extend the same latent-world tests into symbolic + image + toy-audio views of one exact hidden state
- [[state_action_memory_architecture_direction|state-action memory direction]] -- the current architecture-level translation of the new research cluster
- [[substrate_requires_architectural_change|architectural interventions]] -- the ranked A-E candidate list; candidate G (stochastic write gain) is the current live lead

### new research shelves and visual narratives
- [[systems_neuroscience_research|systems neuroscience research]]
- [[cellular_molecular_neurobiology_research|cellular and molecular neurobiology research]]
- [[cognitive_architecture_research|cognitive architecture research]]
- [[cross_scale_building_blocks_research|cross-scale building blocks research]]
- [[architectures_beyond_next_token_research|architectures beyond next-token research]]
- [[canonical_visual_narratives_neuroscience|canonical visual narratives: neuroscience]]
- [[canonical_visual_narratives_mind_and_memory|canonical visual narratives: mind and memory]]
- [[canonical_visual_narratives_world_models|canonical visual narratives: world models]]

### mechanism domains
- [[leaky_integrate_and_fire|single neuron models]] (4 articles)
- [[hebbian_learning|synaptic plasticity]] (6 articles)
- [[sparse_coding|neural coding]] (4 articles)
- [[lateral_inhibition|lateral inhibition]] (4 articles)
- [[predictive_coding|predictive processing]] (3 articles)
- [[cortical_column|cortical microcircuits]] (4 articles)
- [[gamma_oscillations|oscillatory dynamics]] (3 articles)
- [[dopamine_system|neuromodulation]] (5 articles)
- [[hippocampal_memory|memory systems]] (4 articles)
- [[dendritic_computation|dendritic computation]] (4 articles)
- [[brain_energy_budget|energy and metabolism]] (3 articles)
- [[selective_attention|attention]] (3 articles)
- [[critical_periods|development and learning]] (3 articles)
- [[place_cells|spatial computation]] (4 articles)
- [[global_workspace_theory|consciousness and integration]] (4 articles)
- [[gaba_signaling|inhibitory signaling]] (2 articles)
- [[basal_ganglia|action selection]] (1 article)

### introductory articles
- [[start_here]] -- entry point and reading order
- [[the_brain_in_one_page]] -- 80/20 neuroscience for ML engineers
- [[neuroscience_for_ml_engineers]] -- the big primer
- [[mathematical_foundations]] -- math with worked examples
- [[todorov_biology_map]] -- every component mapped to biology
- [[glossary]] -- 73 terms in plain language

### todorov architecture knowledge
- [[unified_theory]] -- CRBR framework
- [[ternary_spikes]] -- spike theory and validation
- [[kda_channel_gating]] -- delta rule and forgetting
- [[mla_compression]] -- latent attention
- [[mamba3_architecture]] -- state space models
- [[geometric_algebra]] -- PGA G(3,0,1)
- [[delta_rule_theory]] -- online learning
- [[hybrid_architectures]] -- layer ratios
- [[training_efficiency]] -- optimization
- [[context_extension]] -- long context
- [[papers_library]] -- full paper database

## see also

- [[INDEX]]
- [[PROJECT_PLAN]]
- [[OPERATING_DIRECTIVE]]
- [[log]]
