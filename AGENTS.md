# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

# single plan rule (2026-05-08)

`neuroloc/wiki/PROJECT_PLAN.md` is the only canonical project plan to follow.
every new session reads this file after AGENTS.md and treats it as the binding
recovery point for the whole project.

other files are subordinate evidence, logs, handoffs, mirrors, or procedure
docs. they are not competing plans and cannot override
`neuroloc/wiki/PROJECT_PLAN.md`:

- `~/.claude/plans/compressed-dancing-haven.md` is the detailed curriculum
  specification and chapter-production protocol.
- `pdf_curriculum/index/curriculum_status.md` is a lightweight curriculum index.
- `state/program_status.yaml` is structured run/status data.
- `docs/STATUS_BOARD.md` is a per-run/status-board summary.
- `reports/` contains read-only reports and recovery dossiers.
- `neuroloc/wiki/OPERATING_DIRECTIVE.md` is the binding workflow directive for
  wiki and state-file edits, not a competing scientific plan.

when any document disagrees with the project direction, repair
`neuroloc/wiki/PROJECT_PLAN.md` first. sync a subordinate file only when the
current task directly touches that surface. do not require future sessions to
reconcile multiple plan-like files before recovering the project state.

# researcher workflow and evidence logging (2026-05-08)

every task starts from `neuroloc/wiki/PROJECT_PLAN.md`, then inspects the
current code and docs before acting. no claim is established by memory,
conversation, a prior summary, or a runnable scaffold alone. every claim remains
unproved until checked against code, tests, wiki evidence, controls, and the
canonical plan.

agents must think outside the box, but every proposed idea must be grounded in
math, biology or ml evidence, a falsifiable test, controls, telemetry, and a
kill condition. creative hypotheses are welcome only when they become testable
mechanism claims.

simplicity beats everything else, but it is never easy. prefer the simplest
mechanism that can honestly pass the gate, then do the hard work required to
prove it is not a shortcut, hidden table, side channel, or category error.

roadblocks, novel mechanisms, ambiguous failures, and scientific uncertainty
require focused research or review agents before implementation continues. the
resulting evidence must be written to the wiki surface that owns it, then folded
into `neuroloc/wiki/PROJECT_PLAN.md` when it changes project direction.

tests run, exact results, mistakes, roadblocks, decisions, and evidence
locations must be recorded in the correct place: `neuroloc/wiki/tests/` for
test records, `neuroloc/wiki/mistakes/` for mistakes, synthesis pages for
research decisions, and the append-only update history in
`neuroloc/wiki/PROJECT_PLAN.md` for canonical direction changes.

when deyan sets a `/goal`, treat it as a delivered-product objective unless
he explicitly says the goal is only planning, brainstorming, or research. do
not mark a goal complete for a literature survey, map, plan, scaffold, or
documentation corpus alone when the requested object is a working artifact.
the goal remains active until code, tests, simulations, metrics, and canonical
documentation prove a product or until deyan explicitly changes or cancels the
goal. documented failures are required evidence, not a stopping condition; if
one path fails, write the mistake or boundary record and continue to the next
best path until the goal ends in a positive tested result.

scaffolds, harnesses, diagnostics, baselines, and real model components must be
labelled separately. violating this workflow is a blocking scientific error:
progress cannot be claimed until the workflow failure is corrected and
documented.

# neural model research/test-material phase (2026-04-23)

**the active scientific object is the neural model.** the old todorov architecture and prior paid runs are historical evidence about failed substrates and trainability, not the live design identity. the approved no-paid-compute preparation lane is `neuroloc/wiki/synthesis/neural_model_research_test_material_plan.md`: every proposed local-neuron, memory, addressing, interference, compression, reconstruction, replay, rollout, and trainability mechanism must have a concise mechanism dossier before any metric code, model code, or intervention preset is accepted.

chapter 1 remains review-ready after the 2026-04-23 math correction pass, and chapter 2 remains outline-only awaiting approval. this does not authorize chapter 2 research, architecture execution, paid compute, h200, kaggle, or pod work.

the first hard symbolic test-material package is implemented as the `hard_symbolic_nm` suite. it covers belief-state formation, associative recall, correlated-key interference, delayed use, episodic reuse, context-gated routing, compression under bit budget, replay/rewrite, iterative rollout, and imagination/recombination. the current next action is governed only by `neuroloc/wiki/PROJECT_PLAN.md`; as of 2026-05-09, deyan has explicitly reactivated local code and simulation work and raised the active local target to 100k parameters for coherence. the active path is compression-first local implementation toward an actual 100k-parameter model, with wiki-grounded evidence, controls, telemetry, and documentation before any stronger claim.

# category-error guardrails (2026-05-08)

the 2026-05-08 local responder/foundation artifacts exposed a serious
category error: a routed record table, copied branch record, or command-line
template responder must never be presented as a working neural model, chat
model, imagination system, compression result, or brain-like memory system.
future agents must enforce the distinctions below before writing code, docs,
status summaries, or user-facing claims.

**scaffold naming rule.** until the required proof gate passes, use only
`scaffold`, `harness`, `diagnostic`, `surface`, `router`, or `baseline`.
do not call an artifact `v1`, `foundation neural model`, `chat model`,
`imagination`, `dreaming`, `memory`, or `compression` unless it satisfies
the corresponding gate below. filenames may retain historical names, but new
text must state that those names are legacy labels when they overclaim.

**memory gate.** record lookup, table routing, nearest-record selection, or
copying a stored payload is not memory in the project sense. a memory claim
requires learned write and read over internal state, partial-cue retrieval,
interference/forgetting behavior, provenance, and controls against no-memory,
recency-only, shuffled-address, sparse-read, and verbatim-store baselines.

**imagination gate.** copying a retrieved record into a branch object, adding
a branch label, or rendering a template is not imagination. an imagination
claim requires learned latent branch rollout or recombination, a dynamics or
world-state transition, branch-local uncertainty/provenance, and evidence that
the branch improves hard-case action, reconstruction, or counterfactual
prediction more than easy cases. controls must include no-branch, shuffled-
branch, wrong-dynamics, matched-compute, and random-branch baselines.

**replay gate.** returning a previously selected record is not replay. replay
requires delayed reactivation that changes future recall, compression,
rewrite quality, interference, or action success. it must beat random replay
and no-replay under matched compute while preserving provenance.

**compression gate.** fewer visible fields or a hand-coded schema boundary is
not compression. compression requires fewer accounted bits at matched
operation success against fair sparse-read and verbatim baselines, with
decoder/schema/parser costs either charged or explicitly fixed across methods.
retaining a decoded raw payload cache, residual table, source block, seed
oracle, or reconstruction row while charging only a compressed stream is false
compression unless the retained surface is charged in strict bits. the
2026-05-12 source-block raw-cache mistake at
`neuroloc/wiki/mistakes/source_block_codec_raw_cache_category_error.md` is the
blocking precedent: future codecs must prove the read path uses the charged
compressed representation or a charged decoder state.

**high-density knowledge gate.** structured, planned, formula-generated, or
hand-designed fact families are not evidence for the high-density
knowledge-compression target. if values, provenance, labels, or answers are
recoverable because the agent chose a compact generator, schema, deterministic
rule, template, seed, or preplanned relation, the result is formula compression
or a boundary artifact, not knowledge compression. the required target is
unstructured or unknown-structure knowledge: no hand-designed answer formula,
no schema-generated labels, no hidden lookup table, no per-fact committed rows,
no uncharged side channel, and no template leakage. any discovered structure
must be learned from data under held-out controls, charged in the accounting,
and tested against random-label entropy controls, sparse read, verbatim
storage, and product-key-style memory. the 2026-05-09 schema-density cell
mistake at
`neuroloc/wiki/mistakes/schema_density_cell_structured_target_category_error.md`
is the blocking precedent: do not repeat it. the 2026-05-12 learned
unknown-structure residual-row defeat at
`neuroloc/wiki/mistakes/learned_unknown_structure_residual_table_defeat.md`
is the second blocking precedent: a learned dictionary plus per-fact residual
table is not the requested high-density knowledge-compression cell.

**chat and language gate.** a command-line prompt surface over bounded records
is not a chat model. a chat-model claim requires learned language generation
or learned response composition over held-out dialogue, not fixed templates or
record rendering. otherwise call it a constrained responder or command-line
harness.

**v1 gate.** `v1` is reserved for a coherent neural architecture with learned
internal dynamics, not a local proof scaffold. a v1 candidate must specify the
architecture contract first: state variables, write/read dynamics, replay,
imagination, action/answer decoding, training objective, controls, telemetry,
and kill conditions.

every new run card or status update for local scaffolds must include a
category check with: implemented operation, strongest baseline, what failed,
what is not proved, and why the artifact is not being promoted to the neural
model claim.

# teaching curriculum package (2026-04-17; preserved after 2026-04-23)

**the teaching PDF curriculum remains preserved and reviewable** in the plan file at `~/.claude/plans/compressed-dancing-haven.md`, but the current scientific planning lane is the 2026-04-23 neural model research/test-material phase above. paid compute is paused indefinitely. the architectural-intervention track (A-E candidates in `neuroloc/wiki/synthesis/substrate_requires_architectural_change.md`) remains in the research backlog but is NOT the active workstream; it resumes only after dossier-driven test material, cpu controls, and a selected intervention exist.

the curriculum specifies 36 chapters across 6 phases (math fundamentals, biology fundamentals, brain computation, ML math, advanced architectures and compression, paper implementation) at 20-25 pages per chapter in English LaTeX, produced with parallel research agents and prosecutor validation. full detail in the plan file and in `neuroloc/wiki/PROJECT_PLAN.md` section "pivot to teaching curriculum" + section "curriculum track".

**curriculum rules** (apply when editing curriculum material):

1. **no paid compute run is authorised.** the pod stays stopped through the curriculum and the neural-model research/test-material phase. paid compute can return only after dossier-driven test material, a tiny trainable mirror, cpu controls, telemetry, prosecutor-clean docs, and one explicit selected hypothesis exist.

2. **published-technique names are restricted.** "MLA", "KDA", "Hopfield", "Hebbian", "STDP", "BCM", "k-WTA", "ternary spike" and other named published techniques are permitted in curriculum chapters (and in wiki/code going forward) ONLY when (a) quoting an external source or (b) naming an external paper's architecture (e.g., "in Ramsauer 2020, the architecture is called 'modern Hopfield networks'"). the project's own components are always named by what they do mathematically or by the project glossary terms: `matrix memory` (the asymmetric outer-product state with decay and optional write-side erasure), `compressed attention` (the low-rank projected dot-product attention with rotary positional encoding), `slot memory`, `output gate`, `surprise ratio`, etc. every chapter's research output and draft passes a naming-compliance prosecutor.

3. **per-chapter research method**: launch many parallel research agents across the unrestricted source inventory (wiki + cited papers + classical textbooks + online resources + reference implementations + recent papers). follow with prosecutor agents that validate research for factual accuracy, internal consistency, completeness, naming compliance, and pedagogical coverage before drafting begins. zero-findings rule applies to both the research-validation prosecutor and the draft-validation prosecutor.

4. **writing voice**: English prose in the pedagogical voice of `/Projects/dzipobel` adapted from Bulgarian. second-person imperative when teaching, short declarative sentences, explicit warnings for anti-misconception points, worked examples inline, zero emojis, zero AI attribution.

5. **user executes chapter production**. any agent working on this project during the curriculum era produces outlines, research gather, drafts, and compilations; the user approves outlines before research begins and approves drafts before finalization.

6. **the private curriculum plan is detailed protocol, not a second project plan.** the master curriculum specification at `~/.claude/plans/compressed-dancing-haven.md` must stay detailed and current when chapter status or curriculum protocol changes, but `neuroloc/wiki/PROJECT_PLAN.md` remains the single session-start plan. if they diverge, fix `PROJECT_PLAN.md` first, then sync the curriculum plan so it agrees on curriculum detail.

# historical architecture background

the text below records the old todorov architecture theory as historical
background. it is evidence for why the project moved, not the live design
identity and not the current model being built. the active architecture is the
neural model defined by `neuroloc/wiki/PROJECT_PLAN.md`; as of 2026-05-09 the
next step is local compression-first implementation and simulation toward an
actual 100k-parameter model.

the old architecture attempted to express biological neural computation
principles -- ternary spikes, adaptive thresholds, recurrent state accumulation
-- through one mathematical object, the Compressed Rotational Bilinear
Recurrence (CRBR). every layer instantiated:

    z_t = Q(R(B(C(x_t), C(h_{t-1}))))

where C is compression (ternary spikes, latent projection, gating), B is a
bilinear interaction (outer product, dot product, geometric product), R is
rotational structure (RoPE, complex dynamics, PGA rotors), and Q is output
quantization. the three layer types are different parameterizations of the
same CRBR framework:

- delta-rule layers: C=spike, B=outer product, R=RoPE, recurrence via
  error-correcting writes to a matrix-valued state
- continuous-dynamics layers: C=gate, B=element-wise product, R=complex
  rotation, recurrence via discretized differential equations
- latent-attention layers: C=low-rank projection, B=dot product, R=RoPE,
  exact retrieval over compressed per-token representations

all paths pass through ternary spike quantization ({-1, 0, +1}), which is
the biological constraint that forces sparse information representation.
the spatial module uses g(3,0,1) projective geometric algebra (B_GP) as
additive self-interaction -- the most general bilinear map in the framework,
subsuming dot product and element-wise product as grade restrictions.

the full unified theory is in knowledge/unified_theory.md. the four families
(C, B, R, Q) and their composition rules define the entire architecture.
everything is C, B, R, or Q. everything composes.

todorov architecture development is historical evidence, not the live design
identity. the active scientific lane is neural model research/test-material
preparation, while the teaching curriculum in `pdf_curriculum/` remains
review-ready and preserved. `neuroloc/spec/blueprint.md` is a paused backlog
design target, not the live workstream. paid training resumes only after
dossier-driven test material, a tiny trainable mirror, cpu controls, telemetry,
and a selected intervention.

# how to think

this project requires unconventional ideas. the architecture is not a
standard transformer and the solutions to its problems are not in standard
transformer playbooks. you must think outside the box.

ground yourself before acting. read the knowledge/ files for published
research and prior findings. read docs/EXPERIMENT_LOG.md for what has
already been tried and why it worked or failed. search the web for recent
papers and benchmarks when your assumptions are uncertain. do not guess
when you can verify.

do not rely solely on your training data. this architecture combines
mechanisms (kda channel-wise gating, ternary spikes, geometric algebra)
that have limited coverage in your pretraining corpus. the knowledge files
contain curated research that is more reliable than your recall. when the
knowledge files and your memory disagree, trust the knowledge files.

eara and its rules are crucial. if the autoresearch loop protocol is not
followed -- every step, every subagent, every artifact -- the project is
jeopardized. previous violations pushed broken code, produced confounded
results, and wasted gpu time. the protocol is not overhead. it is the
difference between science and guessing. follow it literally.

# rules

zero comments in code. no inline, no block, no docstrings, no TODOs.
zero emojis. anywhere. code, docs, commits, prints, logs, filenames.
zero AI attribution. no Co-authored-by. no "written with AI." sole author: Deyan Todorov.
lowercase in all docs, commits, and readme. no tables in readme.

fix absolutely every single thing the prosecutor flags. no exceptions. no
cherry-picking. P0, P1, P2, P3, "minor," "nit," "stylistic," "worth tracking
but not blocking" -- all of it gets fixed before the run launches. priorities
determine ORDER, not SELECTION. every finding is either fixed, or explicitly
waived in writing by deyan (not by you). "i'll look at it later" is not a
waiver. "don't worry about it" is not a waiver. a waiver is "waive finding
F<N>" from deyan, full stop.

treat every prosecutor finding as a bug CLASS, not an instance. if the
prosecutor flags a hardcoded event dict dropping a schema key at line X, the
required response is: (1) fix line X, (2) grep every analogous call site in
the codebase, (3) fix them all, (4) add a structural guard (smoke test,
assertion, refactor) so the class is impossible to reintroduce, (5) re-run
the prosecutor on the modified file. repeat until zero findings.

the prosecutor must return zero unfixed findings before any paid compute
runs. if the prosecutor was not invoked on a file that is about to go into
paid compute, invoke it. pushing code to paid compute without a clean
prosecutor pass is the same class of offense as pushing without smoke test.

use the non-Codex-suffix prosecutor agent (`feature-dev:code-reviewer`) for
every code or wiki review. do not use `prosecutor-Codex`. directive from
deyan, 2026-04-12. exceptions require explicit user approval.

`neuroloc/wiki/PROJECT_PLAN.md` is the canonical project state. it MUST be
updated whenever any of the following happens: (a) a training run starts,
(b) a training run completes, (c) a hypothesis is added or falsified,
(d) the experimental method changes, (e) a new project term enters the
glossary, (f) a prior research question is answered. failing to update this
file is the same class of error as failing to run the prosecutor: both are
blockers for paid compute launches. an outdated PROJECT_PLAN.md is a bug to
fix, not noise to ignore. the file's "update history" section is append-only
and every change gets a new entry with date, author, and a one-line
description of what changed and why. wiki updates that touch PROJECT_PLAN.md
or that document research findings also pass through prosecutor.

design from first principles. do not bind the architecture or its components
to published technique names. the project has its own glossary in
`neuroloc/wiki/PROJECT_PLAN.md`. when describing a new component, describe
what it DOES mathematically; do not use names like "k-WTA", "ternary spike",
"Hopfield", "MLA", "KDA" in code, comments, commit messages, or wiki
articles unless quoting an external source. published technique names import
assumptions and baggage the project does not want.

`neuroloc/wiki/OPERATING_DIRECTIVE.md` is binding for every wiki and state-
file change. it defines the source-of-truth hierarchy when documents
disagree (code wins over docs, later commit wins over earlier, named
canonical roles for PROJECT_PLAN / program_status.yaml / STATUS_BOARD /
AGENTS.md), the four article lifecycle states (current / superseded /
historical context only / definitional), the banner format every article
must carry as its first non-heading line, the migration policy for the
pre-existing wiki articles that do not yet have banners, the bidirectional
cross-reference rule, the append-only rule for mistakes and update_history
sections, the prosecutor protocol on wiki changes, the run-card location
rule (`wiki/tests/` canonical), and the rules for handling disagreement
between the agent and the user. read it before any wiki work. the
directive itself is append-only at its update_history section.

these rules apply to every file you create or modify. no exceptions.

# commands

run the full repo test suite:
    pytest tests/ -q

run a single test file:
    pytest tests/test_kda.py -q

run a single test case:
    pytest tests/test_kda.py::test_parallel_matches_sequential -q

evaluate gates from a run artifact:
    python scripts/evaluate_gates.py <run_dir>

push a training run to kaggle:
    python scripts/kaggle_exec.py push --config eara.yaml

pull completed kaggle run results:
    python scripts/pull_kaggle_results.py

# autoresearch loop

this project uses eara (program.md + eara.yaml). when running the experiment
loop, you MUST follow scripts/autoresearch_loop.md LITERALLY. the protocol
has 7 pre-push steps and 4 post-completion agents.

steps 2-5 (research, plan compliance, self-critique, smoke test) MUST each be
launched as SEPARATE SUBAGENTS using the Agent tool. you may not do them
inline. you may not combine them. the push gate at step 7 blocks any push
without all 4 subagent artifacts.

this is non-negotiable. the protocol exists because an agent skipped steps
2-4 in run_009 and pushed code with a shape mismatch bug and a confounded
experimental design. read the anti-rationalization rules in the loop protocol
before you convince yourself you can skip a step.

# phase sequencing (historical backlog reference)

phases 0-3 are complete. phase 4 (multimodal) is skipped at this scale.

these sequencing notes are preserved for when the architecture backlog resumes.
they are NOT an authorization to run architecture work during the curriculum.

phase 5 sequencing is mandatory only when the backlog is explicitly reactivated:
1. phase 5 baseline: 300m params with current simple spikes (replicate phase 1)
2. phase 5a: swap ternarystrike for atmn (isolate neuron model impact)
3. phase 5b: expand to 132-point placement with winner from 5a (isolate placement)
4. sbds distillation only if bpb ratio degrades at 300m scale

three runs, not one. each produces an evidence-backed claim. NEVER activate
multiple untested features simultaneously. always distinguish "implemented"
from "validated" in reports.

implemented but not yet validated (unit tests pass, never trained):
- atmn spike neurons (membrane potential dynamics, per-neuron v_th = exp(a))
- 132-point expanded spike placement (spike_all_projections=True in config)
- sbds bidirectional distillation loss (0.2 fwd kl + 0.7 rev kl + 0.1 feature)

neural machine sequencing (from neuroloc/spec/blueprint.md, backlog only):
1. run 1: baseline 350m, replicate 0.663x bpb ratio at larger scale
2. run 2: rate-coded k-wta compression (replace ternary spikes)
3. run 3: delta rule erasure + bcm-like activity-dependent forgetting
4. run 4: multi-compartment swiglu (k=4 block-diagonal sub-gates)
5. run 5: exact retrieval as a mode (replace separate mla layers)

each feature is validated in isolation. each run that improves or maintains
quality is KEPT. each run that degrades is DROPPED. do NOT bundle features.
this is the same discipline as phase 5a/5b but for the neural machine.

# architecture rules

never simplify recurrent state computation. it is the memory, not diagnostics.
the matmul path state_approx bug (run_003-007) was caused by taking only the
last timestep instead of accumulating full state. this made all chunked
evaluation silently wrong for context > 256.

gp self-interaction is additive after the swiglu down projection. it adds zero
computational overhead. it can be on by default. the "optional spatial module"
from chimera-edge is validated: gp as self-interaction works, gp as neighbor
interaction does not (echoloc thesis).

spike health thresholds: mi > 0.1, cka > 0.3, firing rate 30-60%. the
adaptive threshold (alpha * mean(|x|)) with alpha=1.0 gives ~41% firing rate.
stable across 11 runs (including 267m scale). mi reached all-time high of
1.311 with gp active (6m scale). at 267m scale: mi 1.168, cka 0.732, fr 40.8%.

layer pattern: 6:1:1 ratio (75% kda, 12.5% mamba3, 12.5% mla). this ratio has been
adopted and validated by kimi, qwen3, olmo, and systematic analysis papers.

# kaggle

kernel: dttdrv/todorov-autoresearch
auth: read ~/.kaggle/kaggle.json, set KAGGLE_API_TOKEN env var BEFORE
importing the kaggle package. the package auto-authenticates on import.
always use acc="NvidiaTeslaT4" in kernels_push.

fla on t4: triton 3.3+ dropped sm_75. pin torch==2.6.0 + triton==3.2.0.
t4 lacks bf16 cores: use fp16 for all fla kernels. use chunk_kda (not
chunk_gated_delta_rule). hybrid threshold: matmul for seq<512, fla for
seq>=512.

api quirks: kernels_status() 404s for ~60s after push on new kernels.
kernels_output() returns last COMPLETED run, not current. no streaming
log api. the mcp-server-kaggle-exec package is BROKEN -- use
scripts/kaggle_exec.py instead.

kaggle t4 is not viable for 300m params. phase 5 runs on h200 (confirmed
in run_010: 267m params trained successfully on h200).

# repository layout

src/             library code. source of truth. do NOT modify during experiment loops.
  algebra/       geometric product (G(3,0,1) cayley table, sandwich product)
  layers/        kda.py, mamba3.py, mla.py, swiglu.py
  spikes/        ternary_spike.py, atmn_spike.py, spiking_brain.py
  model/         full model assembly
  training/      optimizer, schedule, loss
notebooks/autoresearch/train.py   sole file modified during experiment loops
scripts/         kaggle_exec.py (push runs), evaluate_gates.py, autoresearch_loop.md
tests/           repo test suite and focused nm regression batteries
state/           program_status.yaml is a subordinate structured status mirror
docs/            STATUS_BOARD.md is a subordinate status-board mirror
pdf_curriculum/  active teaching pdf curriculum production
knowledge/       published research and prior findings (read before acting)
neuroloc/        biological grounding wiki + neural machine spec
  spec/blueprint.md           paused neural-machine backlog design target
  spec/implementation_plan.md archived pre-pivot implementation sketch
  wiki/                       216 markdown files (obsidian)
  simulations/                45 python validation scripts
  results/                    experiment registry

# critical files

notebooks/autoresearch/train.py -- the sole file modified during experiment
loops. ~1550 lines. self-contained for kaggle. all model code inlined.

scripts/autoresearch_loop.md -- the MANDATORY loop protocol. read this
before every experiment session.

program.md -- universal eara agent instructions.
eara.yaml -- project-specific config (metric, gates, compute backend).
state/program_status.yaml -- current phase, run history, gate results.
docs/STATUS_BOARD.md -- what is passing, what is failing, what is next.

src/ -- library code. do NOT modify during experiment loops. the library
is the source of truth. if train.py and src/ diverge, fix src/.

# bug history

run_003-007: matmul path state_approx only captured last timestep. fixed
in run_008 with full accumulation loop.

run_009: src/layers/swiglu.py spatial_mode added gp output (d_model) to
hidden state (hidden_dim). fixed in both train.py and library: gp residual
applied after down projection.

run_009: step count mismatch (gp=500, nogp=200, transformer=200) confounded
language degradation measurement. the -18.8% "improvement" is training
budget, not gp. fix: always train comparison models at matched step counts.

run_009: equivariance gate tested the cayley table algebra, not the trained
model. this gate is vacuous (always passes at float32 precision). fix: test
full swiglu pathway equivariance, not raw multivector algebra.

run_009: shape classification p~0.07 on 200 trials. not statistically
significant. fix: increase to 1000 trials for p<0.01.

run_010: fla chunk_kda produces NaN at d_model=1024 (works at d=384).
likely a numerical stability issue in the triton kernel at larger head
dimensions. fix: investigate chunk_kda kernel, may need fp32 accumulator
or different chunking strategy at d=1024.

run_010: mamba3 sequential scan ~15s/step at T=2048 with 267m params.
not viable for training. fix: implement parallel scan kernel for mamba3,
or use an existing fused implementation.

run_011: fla chunk_kda NaN at d=1024 fixed via l2 normalization on k/v
before the triton kernel. verified stable across 2000 steps at 280m.

run_011: firing-rate telemetry reads zero under gradient checkpointing.
the hook captures re-materialized activations, not the forward pass. fix:
disable gc on spike-metric layers, or read fr from the cached forward value.

run_011: mamba3 sequential scan is the dominant training cost at 280m
(~10s/step overhead, 6x slower than kda+mla alone). fix: parallel scan
kernel or drop the dynamics layers until the kernel exists.

god_run: god_machine.py step-logger metrics_logger.log({...}) was a
hardcoded dict literal that cherry-picked 5 keys out of collect_god_metrics's
40+ key return value. result: 4000-step h200 run produced a metrics.jsonl
missing every probe metric (imag_contribution_ratio_per_layer,
pc_error_l2_per_layer, kwta_k_rate_per_layer, alpha_base_mean_per_layer,
mlp_compartment_l2_per_layer). the prosecutor had already caught the same
bug class on neural_machine.py final_validation event (F3 finding) but i
only fixed the single named instance and did not generalize. fix: merge
full spike_stats dict into step record; smoke test now round-trips through
a real MetricsLogger to disk and asserts every aux key survives. new hard
rule: fix absolutely every prosecutor finding and treat each as a bug class,
not an instance. grep for analogous sites and fix all of them.

god_run: bcm train/eval path divergence. _fla_delta_rule_path computes
alpha_eff from running_state_norm (buffer, ema-updated); _recurrent_with_erasure
computed alpha_eff from the live per-step state frobenius norm. training
always takes the fla path, eval continuation takes the recurrent path after
first chunk. model is trained under one dynamical rule and evaluated under
another. this is a credible contributor to the 0% passkey finding. fix:
align recurrent path to use _effective_log_alpha, which reads the buffer.

god_run: running_state_norm buffer mutated during validation. both the fla
and recurrent update sites were unconditional with torch.no_grad() but no
self.training gate. each validation pass applied 50 ema updates
(mul_(0.99).add_(..., alpha=0.01)), polluting the bcm threshold. fix: gate
both updates on self.bcm_enabled and self.training.

god_run: imag_filter and pc_head were nn.Linear(d, d) which is 1m params
per delta layer, ~24m total for each, and always active on every forward
pass. fix: low-rank factorization nn.Linear(d, rank) -> nn.Linear(rank, d)
with rank = max(8, min(64, d//16)). gives ~131k params per layer for d=1024.
added post-val warning if imag_ratio_mean stays below 0.02 over 50 steps.

god_run: ByteDataset.__init__ called .long() on the full dataset, turning
uint8 into int64 and using 8x the necessary ram (5gb of data materialized
to 40gb). fix: keep self.data as uint8 and cast to long inside __getitem__
on the small returned slice.

run2_slot_memory_retention_fixed: first launch attempt of this run took 17
min of paid h200 to log only step 0 at 655 tok/s (vs the 33k tok/s the
fused triton kernel gives). root cause: `flash-linear-attention` was not
installed on the pod. import failed at module load, FLA_AVAILABLE was set
to False, every SlotMemory layer fell through to the pure-python
`_recurrent_slot_update` loop, the model trained at ~50s/step, and 4000
steps would have taken 56 hours of paid compute. the fall-through was
silent — no warning anywhere in the launch logs. fix: commit edcfe5d adds
`flash-linear-attention>=0.4.0` and `datasets>=2.19.0` as hard pins in
`requirements.txt`. structural improvement still outstanding (see
PROJECT_PLAN.md): god_machine.py startup should warn or fail when any
layer's `use_fla=False` after a preset asks for FLA. mistake doc:
wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md.

god_run: passkey_retrieval_test and selective_copy_test were run with
num_trials=20. 0/20 gives a 95% wilson upper bound of ~14%, which is too
weak to support an architectural claim. fix: bumped to num_trials=100. added
_wilson_upper_95 helper and ci in log lines.

god_run: imagination test was really a state-structure probe (closed-gate
readout of the delta state with novel random keys, measuring structured vs
random output norm and pairwise orthogonality). name was a misleading
metaphor from wiki/knowledge/imagination_computation_research.md that
suggested image generation, but the model is byte-level text with no vision
capability. fix: renamed to run_delta_state_structure_probe.

# results summary

phase 1: bpb 0.84x (16% better than transformer). mi 1.275 (29x gerhard).
cka 0.913 (46x gerhard). all 4 gates pass.

phase 2: perplexity +4% over 16x extrapolation (pass). mla cache linear
(pass). selective copy 0% (deferred, needs 130m+). fla o(t) scaling
confirmed (1.94-1.97x per 2x length).

phase 3: n-body 29% mae improvement (real). shape classification 30% vs
25% (not yet real, p~0.07). equivariance 1.34e-07 (vacuous gate). language
bpb not degraded (confounded by step count mismatch). gp overhead: zero.
spike mi all-time high (1.311).

phase 5 baseline (run_010): 267m params on h200, kda+mla only (mamba3
dropped, fla disabled). bpb 2.375, ratio 0.663x (33.7% better than
transformer). spike mi 1.168, cka 0.732, fr 40.8%. all 4 gates pass.
mamba3 sequential scan ~15s/step at T=2048 (not viable without parallel
scan). fla chunk_kda produces NaN at d_model=1024.

phase 5 full arch (run_011): 280m params on h200, full 6:1:1 (18 kda + 3
mamba3 + 3 mla). bpb 2.592, ratio 0.722x (27.8% better than transformer).
spike mi 1.246, cka 0.802. fr measurement bug from gradient checkpointing
(reported 0%, actual ~41%). 3/4 gates pass. mamba3 sequential scan
dominates training time (~10s/step overhead). atmn too slow at 280m with
gradient checkpointing.

god_run: god_machine.py at 283m params on h200, fineweb-edu byte-level,
131m tokens, 4000 steps, 53 min. all 5 blueprint features active (k-wta
20%, delta erasure, bcm alpha gamma=0.3, multi-compartment k=4, compressed
attention via sdpa) plus always-on imagination probe and pc diagnostic
head. best_val_bpb 1.3950 (0.390x ratio, 2.57x better than transformer
baseline). firing rate 0.200 exactly (k-wta target met). no dead neurons.
RETRIEVAL FAILED AT ALL TESTED LENGTHS: passkey 0/20 @ 256/1024/4096;
selective copy 0/20 @ 256/512/1024/2048. perplexity at length scales
correctly with context: 1.9354 @256, 1.8437 @512, 1.4909 @1024, 1.4110
@2048, 1.3751 @4096. delta_state_structure_probe mean_structure_ratio 0.981,
pairwise_cos -0.003, random_pairwise_cos 0.000 (state is high-dimensional
noise, not content-addressable). diagnosis: compressed-attention + mlp
path fit the next-byte distribution; delta-rule memory is noise. 17
prosecutor findings f1-f17 fixed before re-run.

god_run_v2: god_machine.py re-run with 17 F + 14 G prosecutor fixes. h200,
283m params, fineweb-edu 131m tokens, 4000 steps, 59 min. critical fix F1:
torch.exp instead of sigmoid(logsigmoid()) so recurrent path matches FLA's
exp(g) gate convention. best_val_bpb 1.4453 (+0.050 vs v1, 0.404x ratio).
RETRIEVAL STILL 0: passkey 0/100 ci 3.7% at 256/1024/4096, copy 0/100 at
256/512/1024/2048. delta_state_structure_probe ratio 0.977 pairwise_cos 0.003
(noise). f1 fix did not recover passkey. external review listed 8 candidate
contributing mechanisms; theory was provisional before controls.

run1_baseline_noerasure: god_machine.py with all 5 bundle features off.
h200, 353m params, fineweb-edu 131m tokens, 4000 steps, 42 min. final
val_bpb 1.4499 same as v2 within noise. passkey 0/100 at 256, copy 0/100
at every length. confirmed bundle-broken at the substrate level: matrix
memory itself does not retrieve.

run2_slot_memory: pivot to slot memory substrate (softmax addressing over
64 prototype keys per head, surprise-gated lru writes, FLA via
fused_recurrent_simple_gla). h200, 355m params, fineweb-edu 131m tokens,
4000 steps. final val_bpb 1.5107, passkey 0/100 at every length. SELF-
INFLICTED CONFIG BUG: the slot preset inherited alpha_log_mean=-0.5 from
Config defaults, reproducing the state-evaporation bug documented four
days earlier in linear_attention_retrieval_wall.md. fix in commit 7abb781.
audit revealed run1_baseline_noerasure and run4_erasure_ablation also
inherited the broken default; fix in commit f0e1a9a applies to all gated
presets; structural guard `_assert_preset_retention_safe` enforces explicit
alpha_log_mean at config resolution. mistake doc:
wiki/mistakes/run2_slot_memory_decay_copy_paste.md.

run2_slot_memory_retention_fixed: slot memory with alpha_log_mean=5.0
explicit and FLA actually active. h200, 355m params, 4000 steps in 72 min
at 33000 tok/s. final val_bpb 1.4777 (best). partial eval before user
halted pod: passkey@256 = 0/100, passkey@1024 = 0/100. FIVE consecutive
paid runs at 0% passkey across two substrates and two retention regimes.
at that point the diagnosis was "the language-modelling training objective
on fineweb-edu does not exercise the memory substrate". the proposed
discriminant (`wiki/synthesis/training_objective_vs_architectural_goal.md`)
was to train on a corpus that rewards retrieval. that discriminant was
then executed as run3_cognition_phase1.

run3_cognition_phase1: first paid test on a retrieval-shaped corpus. h200,
355m params, synthetic cognition corpus (50% passkey / 30% kv recall /
20% copy, hex ASCII alphabet, from `neuroloc/data/cognition_corpus.py`),
4000 steps in 72 min at 32800 tok/s, git f27b81c. best val_bpb 6.3519 —
training plateaued at the alphabet prior from step 150 and never descended.
loss stayed in band [4.34, 4.42] for steps 50-4000. partial eval before
user halted pod: passkey@256 = 0/100, passkey@1024 = 0/100. SIXTH
consecutive 0% passkey, and the first one where the corpus explicitly
rewards retrieval. the
`training_objective_vs_architectural_goal.md` article predicted:

> "if phase one cannot produce any non-zero passkey accuracy on synthetic
> data, the substrate genuinely cannot be trained by sgd at this scale and
> the architecture needs deeper changes."

phase one produced 0%. the architecture-cannot-be-trained-by-sgd branch
fires. diagnosis is now architectural, not training-corpus. post-run
analysis at `wiki/synthesis/substrate_requires_architectural_change.md`
catalogs and ranks five candidate interventions: (A) output gate init 0
instead of -4 (sigmoid(-4)=0.018 appears to be a fixed point of sgd; the
gate stayed at 0.018 through two paid runs including this one), (B)
auxiliary retrieval loss weighting MARK_QUERY-following positions 10-100x
(attacks the 3%-of-tokens-that-matter dilution at byte level), (C)
orthogonal prototype key init (break one-slot softmax collapse at init),
(D) warm start from hand-placed (key, value) pairs, (E) substrate
replacement (titans fast-weight, larimar, differentiable kv table with
hard attention). mistake doc from this run: no new mistake — the run
executed cleanly, the result just falsified the corpus-pivot hypothesis.
run card: `wiki/tests/run3_cognition_phase1_results.md`.

17 runs (9 kaggle t4 + 8 h200, ~$19-24 of paid compute spent).
architecture development is paused. the A-E list above is historical backlog
evidence, not an active launch plan. another paid run on the current slot
substrate with no architectural changes is strictly predicted to produce 0%
passkey and must not be authorised. no intervention bundle is authorised. the
active no-paid-compute path is the neural-model research/test-material plan:
mechanism dossiers, explicit test material, symbolic verification, a tiny
trainable mirror, one-mechanism cpu controls, telemetry, prosecutor-clean docs,
and only then a selected paid hypothesis.

# keeping this file current

this file MUST be updated after every completed run. the documentation agent
(step 9d of the autoresearch loop) is responsible for updating AGENTS.md
along with all other docs and state files.

what to update after each run:
- results summary: add new phase results, update run count and budget
- bug history: add any new bugs found, with root cause and fix
- phase sequencing: advance completed phases, update "what comes next"
- architecture rules: add any new rules learned from the run
- kaggle section: update if compute backend changes for phase 5

if AGENTS.md and the project direction ever disagree,
`neuroloc/wiki/PROJECT_PLAN.md` is authoritative. repair the plan first if it
is wrong, then update AGENTS.md or subordinate status mirrors only when the
current task directly touches them.

# read order for new sessions

minimum recovery path:

1. this file (`AGENTS.md`)
2. `neuroloc/wiki/PROJECT_PLAN.md`

task-specific reads after that:

- read `neuroloc/wiki/OPERATING_DIRECTIVE.md` before wiki or state edits
- read `~/.claude/plans/compressed-dancing-haven.md` only for detailed curriculum protocol or chapter-production work
- read `pdf_curriculum/index/curriculum_status.md` only for lightweight chapter package paths
- read `state/program_status.yaml` or `docs/STATUS_BOARD.md` only when checking structured run/status details
- read `eara.yaml`, `program.md`, and `scripts/autoresearch_loop.md` only if the architecture loop is explicitly reactivated
- read `knowledge/`, `neuroloc/wiki/knowledge/`, `neuroloc/wiki/synthesis/`, and `neuroloc/wiki/bridge/` files relevant to the current research question
- read `neuroloc/spec/blueprint.md`, `neuroloc/spec/next_gen.md`, or `neuroloc/spec/implementation_plan.md` only when working on the paused neural-machine backlog
- read `neuroloc/HANDOFF.md` only when recovering wiki/simulation details not already captured in `PROJECT_PLAN.md`
