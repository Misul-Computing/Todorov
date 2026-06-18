# training objective vs architectural goal: why a memory substrate trained on next-byte loss never learns to memorise

status: current (as of 2026-04-22).

## the realisation

2026-04-15. the second paid run2_slot_memory training, with retention fixed
(`alpha_log_mean=5.0`) and the FLA kernel actually active, completed 4000
steps on fineweb-edu cleanly. final val_bpb 1.4777, within ~2% of god_run_v2's
1.4453. eval suite then reported passkey@256 = 0/100, passkey@1024 = 0/100.
identical retrieval failure to every prior paid run, including the matrix-memory
runs.

the user reframed: "this is supposed to be a neural model, not a language one.
not a token predictor."

## the mismatch

the architecture is specified (`spec/blueprint.md`, `spec/next_gen.md`) as a
substrate for neural computation: memory, compression, thinking, imagination.
those are computation primitives, not language primitives. but the entire
training and evaluation pipeline is shaped around language modelling:

- training corpus: fineweb-edu, natural english text
- training loss: cross-entropy over next byte
- training duration: 131M tokens, sized to land on a token-prediction trajectory
- best-checkpoint criterion: lowest val_bpb on held-out fineweb chunks

evaluation includes cognitive probes (passkey, selective_copy, perplexity-at-length,
delta_state_structure_probe), but those are evaluated on a checkpoint that
was selected for low val_bpb, not for cognitive ability. the training never
applied gradient pressure to use the memory substrate.

## why this is the root cause of the 0% passkey result

natural text rewards local distributional structure: bigram, syntactic, short-
range semantic. the four attention layers (one in every seven, total 4 of 28)
already provide enough short-range integration to drive val_bpb down. the 24
slot or matrix memory layers are unconstrained by the loss as long as their
output does not actively harm the local prediction, so gradient descent
collapses them toward a benign null contribution: low gate activation, high
prototype-key entropy, no information actually routed.

the architecture has not failed at retrieval. it was never trained for
retrieval. the optimisation found the easiest path to the supplied loss, and
that path uses none of the substrate the architecture exposes.

## the consequence: gates A and B are sufficient, the paid runs are
testing the wrong thing

cpu gates A (slot buffer capacity) and B (slot surprise writes) measured the
slot mechanism in isolation, with addresses placed by hand. both passed at
cosine 0.9999. the mechanism CAN retrieve. it just is not asked to.

cpu gate C (slot vs matrix integration) tried to bridge by training a tiny
model on synthetic data, but iterations v1, v2, v2b all converged to the
alphabet prior because the tiny scale could not learn the operation either,
and the substrate comparison plateaued indistinguishably.

paid runs trained the substrate on a corpus that has no signal for retrieval.
the resulting checkpoint has no retrieval ability. the eval at the end of
the run measures retrieval. it is 0% by construction, not by architecture.

## what would test this properly

the architecture's claim is that it CAN store and recall content addressably.
to test that claim, the training task must reward storage and recall. three
candidate reformulations:

1. **synthetic cognition curriculum.** generate corpora that consist
   exclusively of memory-task structures: passkey at varied distance, copy-
   sequence-of-length-N, key-value associative recall, simple algorithmic
   tasks (sort, count, deduplicate). val_bpb on these corpora drops only if
   the substrate learns the task. eval on held-out instances of the same
   tasks. no fineweb at all.

2. **mixed pretraining.** roughly 70% fineweb-edu plus 30% synthetic
   cognition, sampled in every batch. the memory substrate gets gradient
   pressure to be useful while general language competence develops in
   parallel. eval reports both val_bpb on text and accuracy on cognition
   probes; both must improve.

3. **two-phase pretraining.** phase one: pure synthetic cognition until
   passkey accuracy on held-out instances exceeds a threshold. phase two:
   fineweb-edu to add general competence. eval phase two on both metrics.
   if phase one passkey is non-zero, the architecture's substrate can be
   trained; phase two then tells you whether general training preserves or
   destroys the trained memory.

option 3 is the cleanest discriminant. if phase one cannot produce any
non-zero passkey accuracy on synthetic data, the substrate genuinely cannot
be trained by sgd at this scale and the architecture needs deeper changes.
if phase one passes and phase two erases the memory, the issue is task
interference and the answer is option 2 or a revised loss. if phase one
passes and phase two preserves, the project has a working memory substrate
and the only remaining question is scale.

## the quantitative bar: bits per parameter

allen-zhu et al. 2024 ("physics of language models: part 3.3, knowledge
capacity scaling laws") established that vanilla transformer LLMs store
roughly 2 to 3 bits of factual knowledge per parameter, largely independent
of architectural details among the standard variants. that constant is the
project's competitive bar. a 355M parameter model trained on natural text
that ties a same-size transformer at val_bpb is delivering the same
~1 gigabit of effective knowledge — no architectural advantage. matching
the bar is not the point.

the architecture's design promise is to exceed it via two compounding
mechanisms documented in `wiki/synthesis/compression_beyond_quantization.md`:

1. **content-addressable substrate.** softmax addressing over slots gives
   ramsauer-style exponential capacity per address dimension, theoretically
   `~2^(d/2)` patterns per head for binary content — orders of magnitude
   more bits per parameter spent on the slot bank than dense weights would
   yield. that potential is wasted if the substrate stores noise (current
   state) but is realised when the substrate stores compressed content.

2. **quantised activations.** the ternary spike, k-WTA, and adaptive
   threshold mechanisms in `spec/blueprint.md` and the existing
   `neuroloc/model/neural_machine.py` reduce activation entropy without
   destroying retrievability (gates A and B verified retention under hard
   quantisation). the output of the substrate is therefore inherently
   lossy in a controlled way that matches what the next layer can consume.
   parameters spent on quantised pathways earn higher bits-per-parameter
   than dense fp16 pathways because the quantisation IS the compression.

these two mechanisms compound: addressable substrates store more per byte
of state, and quantised activations make those bytes count for more in the
downstream computation. the project's whole reason to exist is the product.
if neither is exercised by the training objective, the resulting model is
strictly worse than a same-budget transformer (more flops per token, no
matching knowledge density). that is the failure pattern of all five paid
runs to date.

success therefore requires BOTH:

- a training task that uses memory (so the substrate is exercised at all),
  per the three options above, AND
- an explicit measurement of bits-per-parameter against a same-size
  transformer baseline on the same task. a successful run is not "non-zero
  passkey"; it is "passkey accuracy at distance D with parameter budget P
  exceeds what a same-P transformer achieves at the same D".

without the second criterion the project can produce a non-zero passkey
result that nobody can interpret as architectural progress, only as
"a 355M model can memorize a 5-byte string when trained on copies of it".
that is not the bar.

## what does NOT need to change

- the architecture itself (slot memory, retention init, structural guard,
  output gate). these are validated in isolation and the paid run trained
  cleanly with no instability.
- the eval suite. the cognitive probes are exactly the right measurement;
  they are simply being applied to checkpoints that were not asked to learn
  cognition.
- the paid compute infrastructure. the run completed cleanly in 72 minutes
  at ~33k tokens per second. cost was acceptable. the gates and provenance
  enforcement worked correctly.

## what changes in the project plan as a result

> **note (2026-04-17)**: the guidance in this section reflects the
> project plan as of 2026-04-16, before the cognition-corpus paid run
> was executed. that run (run3_cognition_phase1) has now completed and
> returned 0% passkey; see the "empirical result" section at the end of
> this article and `wiki/synthesis/substrate_requires_architectural_change.md`.
> the concrete "next paid run on a memory-shaped corpus" recommendation
> below was acted on and produced the predicted discriminant's negative
> branch. the article's reasoning is preserved here for evidence
> continuity but the forward-looking guidance is superseded by the
> architectural-intervention article.

the next paid run is no longer "another attempt to verify slot memory works
on fineweb-edu". the next paid run is "the first run on a corpus designed to
exercise the architecture's memory substrate". val_bpb becomes a sanity
metric for trainability rather than a primary outcome.

five paid runs have shown 0% passkey on fineweb-edu. that pattern is
now explained, not by the architecture being broken, but by a category
mismatch between the training objective and the architectural goal. there
is no reason to expect a fifth paid run on fineweb-edu to be any different,
and there is corresponding reason to expect a paid run on a memory-shaped
corpus to behave differently.

## empirical result (2026-04-17)

phase 1 of the "two-phase pretraining" option proposed in this article
was executed as the paid run `run3_cognition_phase1` on 2026-04-17.
355M parameters, slot substrate, retention fixed at alpha_log_mean=5.0,
FLA active on H200, trained for 4000 steps (131M tokens) on a synthetic
cognition corpus with 50% passkey / 30% kv recall / 20% copy blocks.
phase 2 (fineweb-edu to add general competence) was not attempted
because phase 1 did not produce non-zero passkey, which was the
threshold for entering phase 2 per the proposal above.

result: val_bpb plateaued at 6.3519 from step 150 and never improved.
partial eval before pod stop reported passkey @ 256 = 0/100 and passkey
@ 1024 = 0/100. training did not learn the retrieval operation the
corpus explicitly encoded.

this executes the article's predicted discriminant. the text above
(section "what would test this properly") explicitly states:

> if phase one cannot produce any non-zero passkey accuracy on
> synthetic data, the substrate genuinely cannot be trained by sgd at
> this scale and the architecture needs deeper changes.

phase one produced 0% passkey. the architecture-cannot-be-trained-by-sgd
branch therefore fires. the next analysis is at
`wiki/synthesis/substrate_requires_architectural_change.md`, which
catalogs candidate architectural changes (output gate init, auxiliary
retrieval loss, orthogonal prototype keys, hand-placed-address warm
start, substrate replacement) and ranks them by cost and risk.

this article's reasoning structure remains correct. its prediction
that a cognition-shaped corpus would discriminate between "LM loss
is the problem" and "the architecture cannot be trained" is confirmed.
the answer came back "the architecture cannot be trained" under the
configuration the project has been using.

note (2026-04-21 / 2026-04-22): this article remains the evidence chain
for why corpus-only fixes were necessary to test and then insufficient
to solve the retrieval problem. the current backlog method is broader
than passkey on a cognition corpus alone. when an architecture intervention
is selected, candidates first clear the cpu phase-1
battery in `wiki/synthesis/phase1_evaluation_surface_for_neural_models.md`
and then extend through the shared latent-world bridge in
`wiki/synthesis/synthetic_shared_world_bridge.md`.

## see also

- `wiki/tests/run2_slot_memory_retention_fixed_results.md` - the run card
  for the fifth paid run, the empirical trigger for this article
- `wiki/synthesis/slot_memory_design.md` - the substrate design; cpu gates
  A and B; the substrate that was tested at paid scale here
- `wiki/synthesis/phase1_evaluation_surface_for_neural_models.md` - the external-synthesis article that replaces passkey-only evaluation with a broader phase-1 battery
- `wiki/synthesis/synthetic_shared_world_bridge.md` - the follow-on bridge for phase 2 once the phase-1 battery is grounded
- `wiki/mistakes/run2_slot_memory_decay_copy_paste.md` - the first-launch
  retention bug that obscured the substrate question for one run
- `wiki/mistakes/run2_slot_memory_fla_silent_fall_through.md` - the
  second-launch FLA-not-installed silent slowdown that delayed this run
- `wiki/tests/encoding_simulation_round_b.md` - capacity ceiling under the
  matrix-memory operation
- `wiki/synthesis/linear_attention_retrieval_wall.md` - the failure-modes
  catalogue; this realisation supersedes its causal claim by identifying
  a more fundamental cause (the loss does not exercise the substrate)
- `spec/blueprint.md` - the architecture's stated computational goal
- `spec/next_gen.md` - the longer-horizon roadmap; cognition-shaped
  pretraining belongs here
- `wiki/synthesis/substrate_requires_architectural_change.md` - the post-run-3 analysis after this article's proposed discriminant (cognition training) was executed and returned 0% passkey, confirming the architecture-cannot-be-trained branch the article predicted
- `wiki/tests/run3_cognition_phase1_results.md` - the run card of the paid test this article proposed
