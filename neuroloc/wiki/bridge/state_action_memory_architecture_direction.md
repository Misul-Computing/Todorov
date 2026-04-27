# bridge: state action memory architecture direction

status: current (as of 2026-04-27).

## translation

the strongest joint lesson from the biology and ai research is not “choose one trick.” it is a stack:

- latent state formation
- bounded memory and indexing
- action-conditioned dynamics
- controlled routing
- planning or value readout

that stack is a better target than any one next-token architecture or retrieval-only architecture.

## biology side

- hippocampal indexing supports compact handles plus reinstatement
- replay supports consolidation and prospective use
- routing and gating are repeated control motifs
- multi-timescale local state supports staged write and stabilization logic

## ai side

- jepa and data2vec support state formation without surface reconstruction as the main objective
- dreamer, muzero, and td-mpc2 support explicit latent dynamics and planning
- titans-style memory supports bounded surprise-gated memory rather than unbounded cache growth

## practical direction

when the backlog resumes, the project should compare candidate designs against this stack rather than against transformer baselines alone.

## see also

- [[architectures_beyond_next_token_research]]
- [[cellular_molecular_computational_primitives]]
- [[world_models_imagination_and_planning]]
- [[beyond_next_token_for_neural_models]]
- [[research_implications_for_neural_model_direction]]
