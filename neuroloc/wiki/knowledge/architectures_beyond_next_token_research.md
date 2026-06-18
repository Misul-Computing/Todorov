# architectures beyond next token research

status: current (as of 2026-04-23).

curated research shelf for ai architectures that do not reduce the whole system to next-token continuation. the emphasis is on latent state learning, action-conditioned dynamics, planning, bounded memory, and entity-structured state.

## strongest takeaways

- the strongest state-action stack is still world-model centered: belief state, transition model, and action or value head
- jepa and data2vec are the strongest current latent-target alternatives for state formation
- muzero and td-mpc2 show that task-relevant latent planning can outperform reconstruction-heavy designs when control is the goal
- object-centric systems remain the strongest route to clean factorized state
- explicit memory systems help, but retrieval alone is not a substitute for world models

## load-bearing source clusters

### latent-target representation learning

- data2vec
- i-jepa
- v-jepa
- v-jepa 2 and follow-ons

### world models and latent planning

- plaNet
- dreamer
- dreamerv3
- muzero
- td-mpc2

### bounded memory and retrieval support

- memorizing transformers
- retro
- titans
- larimar

### factorized state and abstraction

- slot attention
- recurrent independent mechanisms
- large concept models

## figure candidates from this shelf

- `a1_jepa_family_timeline`
- `a2_world_model_control_loop_comparison`
- `a3_memory_architecture_design_matrix`
- `a4_object_binding_loop`
- `a5_abstraction_ladder`

## what this shelf is for

- grounding the external-ai side of the backlog
- keeping project-native component naming separate from imported external architecture names
- supporting wiki articles on world models, associative memory, and paper implementation

## see also

- [[cognitive_architecture_research]]
- [[beyond_next_token_for_neural_models]]
- [[world_models_imagination_and_planning]]
- [[state_action_memory_architecture_direction]]
- [[dreamer_muzero_jepa_titans]]
- [[canonical_visual_narratives_world_models]]
