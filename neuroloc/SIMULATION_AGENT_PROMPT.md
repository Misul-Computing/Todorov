# simulation agent briefing

you are continuing work inside `C:\Users\deyan\Projects\todorov\neuroloc\`.

this prompt is intentionally short and current. older long-form simulation
instructions were pre-pivot and are no longer the active project framing.

## current mission

the active project lane is the neural-machine architecture program (approach a),
in cpu validation -- not a paid architecture run. simulation work directly
supports it (the cpu-first neural-model evaluation surface and the v0.1 / feel
bench in `v01/`).

if you are asked to work on simulations, the current goal is:

- strengthen the cpu-first neural-model evaluation surface
- keep phase 1 state-and-action-first
- avoid next-token or fineweb-first framing
- prefer explicit episode worlds, state probes, and action outcomes

## canonical read order

1. `neuroloc/wiki/PROJECT_PLAN.md`
2. `neuroloc/wiki/OPERATING_DIRECTIVE.md`
3. `docs/STATUS_BOARD.md`
4. `state/program_status.yaml`
5. `neuroloc/simulations/suite_registry.py`
6. `neuroloc/HANDOFF.md`

## current simulation surface

- `94` simulation scripts total
- `66` memory simulation scripts
- dedicated `biology_phase1` suite
- broader `phase1_nm` suite

the architecture cpu surface includes:

- implemented `biology_phase1`
- remaining latent-world deliberation probe
- model-side neural-model evaluation surface

## design rules

- phase 1 is judged by state formation, memory use, and correct action
- do not center next-token, next-byte, perplexity, or fineweb as the main objective
- every meaningful phase-1 simulation should prefer `state -> memory -> action`
- controls matter: oracle, shuffled, no-memory, and ablation baselines should be explicit
- keep outputs machine-readable with deterministic metrics artifacts

## execution rules

- no paid compute work unless the user explicitly reactivates the backlog
- no kaggle push planning by default; paid compute is gated on funding plus a cpu-validated intervention
- use subagents for research and review when available
- for wiki-linked simulation findings, update the wiki only under the operating directive
