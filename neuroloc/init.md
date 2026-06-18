# neuroloc

this directory is the research memory and simulation substrate for the todorov
project.

the active project lane is the neural-machine architecture program (approach a),
in cpu validation. neuroloc is its reference material, simulation infrastructure,
and architecture-backlog memory.

canonical persistent project state lives in `neuroloc/wiki/PROJECT_PLAN.md`.

## current status

- the neural-machine architecture program (approach a) is the active lane, in cpu validation (the v0.1 toy and feel bench in `v01/`)
- candidate F closed negative; candidate G (stochastic write gain) is the current live lead
- paid compute is gated on funding plus a cpu-validated intervention
- the architecture cpu surface includes the implemented `biology_phase1` battery, latent-world deliberation, and model-side neural-model evaluation

## structure

```text
neuroloc/
  wiki/           canonical wiki and project state
  simulations/    cpu-first biology and neural-model simulations
  spec/           neural-machine design and backlog plans
  results/        historical experiment summaries
  print/          print-oriented neuroloc documents
  raw/            immutable source material
  HANDOFF.md      operator handoff for neuroloc work
  init.md         this file
```

## current counts

- `346` wiki markdown files
- `55` synthesis articles
- `94` simulation scripts
- `66` memory simulation scripts

## entry points

1. `neuroloc/wiki/PROJECT_PLAN.md`
2. `neuroloc/wiki/OPERATING_DIRECTIVE.md`
3. `neuroloc/HANDOFF.md`
4. `neuroloc/wiki/Home.md`

## note on history

older neuroloc documents may describe the pre-pivot architecture phase in
detail. keep them as historical record unless they are an overview or handoff
surface that must match the current state.
