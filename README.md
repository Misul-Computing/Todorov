# todorov

todorov is a research program building one neural machine: a 3d latent
world-memory model that fuses a language model with a sensory world-model.
the architecture is built on one mathematical object, the compressed rotational
bilinear recurrence (crbr): every layer instantiates
`z_t = Q(R(B(C(x_t), C(h_{t-1}))))`, with compression, bilinear interaction,
rotational structure, and output shaping treated as one composable family.

## what works, and what does not

two results are solid and reproducible.

- grounding works. on a cpu bench, giving the toy model a located, accumulated
  sense (touch) turns an otherwise unsolvable task into a solved one: blind
  recall `0.14`, real-touch recall `1.00`, with a fake-touch control at `0.15`
  that proves the gain comes from the felt content, not from extra tokens or a
  leak. the same sense also feeds an integration over the whole sweep
  (counting), not just single-fact lookup. toy scale, cpu, but verified by the
  honesty control.
- the architecture beats a matched transformer on language at scale. the
  strongest archived result is the 267m phase-5 baseline: `0.663x`
  bits-per-byte versus a matched transformer (33.7% better), spike mutual
  information `1.168`, spike cka `0.732`.

one wall is open and fully documented. across six paid runs the recurrent
memory substrate fit the next-byte distribution well but never learned verbatim
retrieval (`0/100` passkey across two substrates and two corpora). the diagnosis
is architectural, not a training-corpus problem, and the falsification trail is
recorded run by run. the first sgd-trained non-chance retrieval on a recurrent
memory substrate in this project appeared as a control anomaly in the candidate
F experiment (mqar exact `0.360`) and is the current live lead, promoted to
candidate G (stochastic write gain).

## the live workstream

the active lane is the neural-machine architecture program. the current design
stance is approach a, the division of labour that survived the project's
17-run history: attention does exact recall (the part that always worked), a
recurrent state does cheap world-tracking, and eidetic compression is deferred.
the machine is in cpu validation now (the v0.1 toy and the feel bench above);
paid compute returns with funding and a cpu-validated intervention.

the six-lane neural-model research and the compression results (oracle bounds,
the 100k local relation and knowledge-pack adapters) are the research substrate
that feeds the machine, not a separate track. the teaching curriculum in
`pdf_curriculum/` is preserved and reviewable but is backlog, not the active
lane.

canonical persistent project state lives in `neuroloc/wiki/PROJECT_PLAN.md`.

## neuroloc

neuroloc (`neuroloc/`) is the repository-native research wiki, simulation
corpus, and architecture-backlog memory:

- `346` wiki markdown files (`55` synthesis articles, `61` mechanism articles)
- `94` simulation scripts (`66` memory simulations)
- `70` experiment run cards and `22` mistake / post-mortem docs -- the documented falsification trail

main entry points:

- `neuroloc/wiki/Home.md`
- `neuroloc/wiki/INDEX.md`
- `neuroloc/wiki/PROJECT_PLAN.md`
- `neuroloc/HANDOFF.md`

## repository map

- `neuroloc/` wiki, simulations, specs, and the neural-machine design surface
- `v01/` the v0.1 toy codebase and the feel bench
- `src/` todorov library code
- `tests/` repo test suite
- `pdf_curriculum/` preserved teaching curriculum (backlog)
- `docs/` and `state/` human- and machine-readable project status

## funding

the program is compute-limited. the immediate need is funding for the
cpu-validated architectural intervention that lifts the retrieval wall, and for
the first paid run of the grounded neural machine at scale. the evidence base is
a reproducible grounding result, a matched-transformer language win, and a
rigorous, fully documented falsification trail that has already ruled out the
substrate dead-ends.

eptesicus laboratories.
