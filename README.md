# Todorov

Todorov is a research program for one neural machine that learns language,
world state, and action in a shared latent process. The current architecture
separates three jobs that earlier versions blurred together:

- Routed selected-set attention is the candidate for exact nonlocal recall.
- Recurrent state handles cheap within-sequence world tracking.
- Token-local feature mixing transforms each position without owning recall or
  state.

The smallest current composition uses Transformerov as the tested host and
recurrent path, Monodratic as the routed recall candidate, and a published
nested-reciprocal feature mixer as an optional feed-forward candidate. Karkasov
contributes only a later training protocol for separately released specialists
and zero-gate docking. Laplace is outside the architecture.

These are evidence-bounded roles, not a merged implementation. A later local
session defined a Todorov-owned complementary-learning-systems trainer for the
Apple M5 Pro. Its fast private state, frozen-trunk specialist, byte-level
FineWeb-Edu stream, and local evidence are recorded in
`neuroloc/wiki/tests/todorov_cls_macbook_session_202608.md`. The trainer and
its generated artifacts remain local and are not part of this public source
push. No paid compute or broad model claim is authorized by that session.

## Evidence boundary

Transformerov has bounded recurrent counting and symbolic sensor-channel
necessity results.
Monodratic has component tests, associative-recall results, and a small host
integration proof. Neither result establishes the combined model. Karkasov's
phase is incomplete and its current replacement comparison is negative. The
nested-reciprocal feature mixer comes from published work and has no local
Todorov result.

The first combined proof must require successful routed recall and recurrent
tracking in one matched CPU package. Disabling either module must break only
its assigned job, and dense attention must remain an explicit control rather
than a hidden fallback. Feature-mixer and specialist-training experiments come
later and change one variable at a time.

The canonical state is `neuroloc/wiki/PROJECT_PLAN.md`. The architecture
contract is `neuroloc/wiki/synthesis/modular_neural_model_stack.md`, and the
feature-mixer proof package is
`neuroloc/wiki/synthesis/neural_model_dossier_nested_reciprocal_feature_mixer.md`.

## Repository

- `neuroloc/` contains the research wiki, proof packages, simulations, and
  historical evidence.
- `v01/` contains the local architecture toy and feel-bench work.
- `src/` contains the legacy library source.
- `tests/` contains repository tests.
- `docs/` and `state/` mirror the canonical project status.
- `quant/` is a separate weight-quantization research workstream.

Start with `neuroloc/wiki/PROJECT_PLAN.md`, then
`neuroloc/wiki/OPERATING_DIRECTIVE.md`, then `neuroloc/HANDOFF.md`.

## Historical architecture

Older documents describe the compressed rotational bilinear recurrence and a
compression-first local path as the active design. Those records explain prior
experiments but do not override the current modular contract.

## Local trainer boundary

The August 2026 CLS trainer session used local MLX Metal on an Apple M5 Pro.
The working `train/` directory contains copied trainer code, generated
checkpoints, datasets, and result files. It is intentionally excluded from
the public repository commit because it contains local scratch material and
large generated artifacts. The public evidence summary does not replace a
reproducible trainer release.

Paid compute remains separately gated by CPU evidence, review, cost, expected
value, and explicit authorization.
