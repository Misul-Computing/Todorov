# Todorov

Todorov is building a neural machine that learns the way a mind does: grounding language in sensory experience and memory rather than in text alone. It is a 3D latent world-memory model that fuses a language model with a sensory world-model, so that words, perceptions, and remembered structure are learned in one system instead of three.

The whole architecture is a single idea, expressed once. Every layer is an instance of the Compressed Rotational Bilinear Recurrence (CRBR):

`z_t = Q(R(B(C(x_t), C(h_{t-1}))))`

A compression, a bilinear interaction, a rotation, and an output map, composed in that order. Attention, recurrent state, and gated memory are not separate subsystems bolted together. They are the same operator under different settings: one object, many behaviors. That economy is what keeps the machine analyzable and lets it scale without turning into a patchwork.

## Why it is promising

Sensory grounding measurably changes what the model can learn, and the effect is controlled. Give the model a sense it can locate and accumulate (touch), and a task it cannot otherwise solve becomes solvable: recall rises from 0.14 with no sense to 1.00 with a real one. A fake, randomized sense leaves performance at 0.15, so the gain comes from felt content, not from extra signal or a leak. The same sense also drives integration across a whole trajectory, not just point lookup. The demonstration is small and on CPU, but it is clean, and it is the kind of result the larger machine is built to scale.

The architecture already beats a strong baseline. At 267M parameters it reaches 0.663x bits-per-byte against a matched transformer, a 33.7% improvement, with healthy internal signal (spike mutual information 1.168, representational similarity 0.732). At matched scale, the biological constraints help.

The remaining hard problem is sharply localized rather than open-ended. Long-range verbatim retrieval through the recurrent memory is the frontier: the memory models language well but does not yet teach itself to recall under gradient descent. That question now has a CPU-validated lead, candidate G, a stochastic write-gain mechanism that produced the first above-chance trained retrieval on this substrate. The next experiment is defined, inexpensive, and falsifiable.

## How it is built

The design, called Approach A, follows the evidence. Attention does exact recall, the capability that has held up across the whole project. A recurrent state does cheap, continuous world-tracking. Heavy eidetic compression is deferred until the simpler parts are solid. Everything is validated on CPU first, on the v0.1 toy and the feel bench above, so each claim is controlled before any compute is bought. A six-lane research program (cellular state, compression, memory and replay, 3D world physics, trainability, and operations) and a stack of compression results supply the machine with grounded design choices rather than guesses.

Canonical project state lives in `neuroloc/wiki/PROJECT_PLAN.md`.

## Repository

- `neuroloc/`: research wiki, simulation corpus, specifications, and the neural-machine design surface
- `v01/`: the v0.1 toy codebase and the feel bench
- `src/`: Todorov library code
- `quant/`: data-free weight quantization research for transformer LLMs
- `tests/`: test suite
- `docs/` and `state/`: human- and machine-readable project status

The `neuroloc/` wiki is the project's working memory: 346 markdown articles (55 synthesis, 61 mechanism), 94 simulation scripts, and a full trail of run cards. Every claim above traces to a recorded experiment. Start at `neuroloc/wiki/Home.md` and `neuroloc/wiki/PROJECT_PLAN.md`.

## Weight quantization workstream

`quant/` is a separate research thread on data-free post-training weight quantization for transformer language models. It runs independently of the neural-machine program and is validated on Qwen2.5 and Qwen3 models on CPU, Kaggle T4, and a rented RTX PRO 6000.

The positive results: E8 lattice vector quantization (8D, two-stage RVQ) reaches 2.1% degradation at 4.125 bpw on Qwen3-4B, below a 5% target, and a per-tensor bit-width picker beats uniform quantization at equal average budget on both perplexity and reasoning. Block-wise fine-tuning of LayerNorm and bias parameters recovers a real but modest share of the residual gap on top of E8.

The negative result is documented and explained: sub-1-bit post-training quantization (1.5B at 0.1 BPW matching a 1B FP16 model) fails by several orders of magnitude, and the bit-budget math shows why. The information deficit is structural; recovering it requires quantization-aware training, not a better PTQ scheme. Cross-layer redundancy probes confirm the weights carry no exploitable redundancy at the scale a 100x compression target would need.

The full timeline, results tables, and file inventory are in `quant/docs/STATUS.md`, `quant/docs/CHANGELOG.md`, and `quant/docs/BIT_BUDGET_ANALYSIS.md`. The package lives in `quant/quant/`, with reproduction and attack scripts at the `quant/` root and unit tests in `quant/tests/`.

## Funding

The science is CPU-validated and the next step is concrete. Funding buys two things: the compute to run the retrieval intervention at scale, and the first full training run of the grounded neural machine. The foundation is already in place: a controlled grounding result, a win over a matched transformer, a single composable architecture, and a precisely localized open problem with a live lead.

Eptesicus Laboratories.
