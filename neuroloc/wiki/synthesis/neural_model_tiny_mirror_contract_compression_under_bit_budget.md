# neural model tiny mirror contract: compression under bit budget

status: current (as of 2026-05-07).

## purpose

this page is the implementation contract for the first tiny trainable mirror allowed by the compression proof package. it is still no-paid, local, and family-specific. it narrows [[neural_model_dossier_compression_under_bit_budget_codec]] into an executable design without touching the full model.

the contract exists to prevent three errors:

- training a global mirror before the weak oracle families are fixed.
- giving the model oracle schema, hidden-state, or answer shortcuts.
- accepting fewer bits as compression when task-relevant state is lost.

## authorized scope

authorized:

- a tiny local mirror for the `compression_under_bit_budget` family only.
- cpu-first implementation under `neuroloc/simulations/memory/` or a similarly local test-material path.
- deterministic generation from the existing `neuroloc/data/nm_worlds.py` hard symbolic contract.
- pytest coverage and suite-registry wiring if code is written after this contract.

not authorized:

- global mirror training across all families.
- `god_machine.py`.
- paid presets.
- h200, kaggle, pod, or runpod paths.
- simulator selection or 3d embodied-world implementation.
- full model integration.

## source family

source generator: `generate_nm_hard_symbolic_batch`.

source family: `compression_under_bit_budget`.

the mirror must filter the generated contracts to this family and ignore all other families except as future negative-transfer research. the oracle split in [[oracle_compression_frontier_split]] selected this family because it clears the current strong-oracle threshold at 13.0x and directly tests compression rather than addressability or eligibility side effects.

## allowed inputs

the model path may receive only observation-available fields:

- observation stream fields exposed by the contract.
- query fields.
- current timestep or relative position.
- visible object attributes.
- visible context tags.
- explicit remaining bit budget.
- observable event markers if present in the observation contract.

the model path may maintain its own recurrent or local hidden state. that state is not the generator's exact hidden state.

## forbidden inputs

the model path must not receive:

- exact hidden state.
- target answer or target action.
- oracle schema id.
- oracle residual id.
- oracle latent-state code.
- family label as an input feature.
- future observations.
- policy result flags.
- kill-condition labels.
- memory-relevant positions unless they are encoded as observable event markers.

the training loop may use exact hidden state and oracle codes only as labels, diagnostics, or ablation targets. any run that uses them as inference inputs is diagnostic only and cannot count as a learned-codec result.

## dataset split

the implementation must produce deterministic, non-overlapping splits:

- smoke train: 64 episodes.
- smoke validation: 16 episodes.
- smoke test: 16 episodes.
- hard train: 1024 episodes.
- hard validation: 256 episodes.
- hard test: 512 episodes.

the split key is seed, not array position. validation and test seeds must not overlap train seeds. no metric may be selected on the test split. the test split is read once after thresholds are fixed.

the mirror must also include factor-heldout local falsification before an ordinary-split result is treated as mechanism evidence. the current factor split withholds color-shape pair bands while keeping every individual test color and test shape represented in train. the artifact must report train/validation/test factor-bucket overlap, seen-marginal checks, learned-code/oracle-decoder success, oracle-code/learned-decoder success, matched-budget sparse-read success, committed bits, and engineering pass.

difficulty sweeps must vary at least:

- distractor count.
- delay.
- visible-feature masking.
- bit-budget fraction.
- payload entropy.
- address similarity.

## model contract

the mirror may be simple, but it must contain the same functional interfaces as the claimed mechanism:

```text
z_t = encoder(o_t, q, b_t, z_{t-1})
c_t = codec(z_t)
m_t = write(memory_{t-1}, c_t)
r_t = read(m_t, q)
y_t = decoder(r_t, q)
```

required emitted fields per episode:

- predicted state.
- predicted action or answer.
- compact code fields.
- address field.
- schema-like field if learned.
- residual field if learned.
- provenance field.
- bits committed by field.
- total bits committed.
- reconstruction error.
- telemetry fields listed below.

the codec may use a learned discrete bottleneck, continuous bottleneck with measured bit proxy, or explicit field-wise code. the first implementation must pick one representation and document its bit accounting before training.

## baselines and controls

required non-trainable controls:

- oracle codec.
- verbatim store.
- compressed oracle store.
- no-memory.
- recency-only.
- shuffled-address.
- random codebook.
- matched-bit random code.
- matched-compute no-code baseline.

required trainability splits:

- learned code with oracle decoder.
- oracle code with learned decoder.
- learned address with oracle payload.
- oracle address with learned payload.
- frozen random encoder with learned decoder.
- learned encoder with frozen random decoder.

these splits must localize whether failure is encoder-side, decoder-side, address-side, payload-side, or optimization-side.

if the ordinary split passes but factor-heldout fails, the result is a local sanity pass only. if a factorized local repair passes, the next action must harden that repair locally before broader mirrors, full-model code, simulator work, or paid compute.

## losses

the primary objective is task success under a bit budget:

```text
loss = task_loss + lambda_bits * committed_bits + lambda_recon * task_reconstruction_loss
```

allowed auxiliary losses:

- visible-field reconstruction.
- state probe on training and validation only.
- address contrastive loss against non-target distractors.
- budget overflow penalty.
- provenance classification if provenance is observable or used only as a label.

diagnostic-only auxiliary losses:

- oracle schema id prediction.
- oracle latent-state code prediction.
- oracle residual id prediction.

a model that passes only with diagnostic-only oracle labels does not pass the learned-codec gate.

## metrics

top-line metrics:

- `state_probe_accuracy`
- `action_success`
- `joint_success`
- `bits_committed_per_successful_episode`
- `compression_ratio_vs_verbatim`
- `rate_distortion_frontier`

control gaps:

- learned minus no-memory.
- learned minus recency-only.
- learned minus shuffled-address.
- learned minus random-codebook.
- learned versus verbatim at matched success.
- learned versus oracle codec.

failure localization:

- learned-code / oracle-decoder gap.
- oracle-code / learned-decoder gap.
- learned-address / oracle-payload gap.
- oracle-address / learned-payload gap.
- frozen-random-encoder / learned-decoder gap.
- learned-encoder / frozen-random-decoder gap.

## telemetry schema

each run artifact must log:

- split.
- seed.
- episode id.
- family.
- difficulty.
- model parameter count.
- trainable parameter count.
- committed bits by field.
- total committed bits.
- budget overflow.
- code usage entropy.
- address entropy.
- address margin.
- read concentration.
- write frequency.
- residual norm.
- reconstruction error.
- memory-output norm versus residual norm.
- task loss.
- bit penalty.
- state probe accuracy.
- action success.
- joint success.
- confidence interval fields.

telemetry-only success is not a pass. telemetry explains a result; it does not replace state/action/joint success.

## acceptance thresholds

smoke profile:

- oracle codec `joint_success >= 0.98`.
- learned codec `joint_success >= 0.85`.
- learned codec beats no-memory, recency-only, shuffled-address, and random-codebook controls by at least 0.40 absolute `joint_success`.
- learned compression ratio versus verbatim is at least 3.0x.
- no forbidden input appears in the model input record.

hard profile:

- oracle codec `joint_success >= 0.98`.
- learned codec `joint_success >= 0.95`.
- `state_probe_accuracy >= 0.95`.
- `action_success >= 0.95`.
- learned codec beats no-memory, recency-only, shuffled-address, and random-codebook controls by at least 0.50 absolute `joint_success`.
- learned compression ratio versus verbatim is at least 4.0x for an engineering pass.
- learned compression ratio versus verbatim is at least 6.5x for a paper-track pass, because the oracle frontier for this family is 13.0x.
- verbatim storage must not satisfy the compressed budget.
- shuffled-address must fail the address-dependent result.
- confidence intervals must exclude the control scores.
- factor-heldout learned codec `joint_success >= 0.95` before an ordinary-split tiny result counts as mechanism evidence.

anything below the engineering pass may still be useful as a negative trainability result, but it does not advance the compression claim.

## kill conditions

kill or redesign the mirror if:

- it needs oracle schema labels at inference.
- it needs exact hidden state at inference.
- it reduces bits while losing `joint_success`.
- it beats no-memory but not recency-only.
- shuffled-address succeeds.
- random codebook matches learned code.
- learned-code / oracle-decoder succeeds but oracle-code / learned-decoder fails and decoder fixes do not help.
- oracle-code / learned-decoder succeeds but learned-code / oracle-decoder fails and encoder fixes do not help.
- address entropy collapses.
- budget usage does not correlate with success.
- telemetry shows memory output is unused.
- the result appears only on smoke and vanishes on hard.
- the result appears on ordinary deterministic splits but fails factor-heldout recombination.

## implementation handoff

the first local dataset, learned-codec, and source-diagnostic surface is now documented in [[tests/compression_under_bit_budget_mirror]]. it produced a negative smoke result, exposed a source-observability contract problem, repaired that contract so legal visible-source extraction solves the smoke task, added sparse-read and matched-budget sparse-read controls, added distributed evidence, produced an ordinary-split tiny local pass, and then failed factor-heldout recombination.

the next code slice should modify the local learned-codec path, not the full model. the first factorized structured codec repaired the color-shape pair-band gate, so the next slice should harden it with multiple seeds, hard-profile/local larger sweeps, other heldout axes, and less hand-shaped event pooling while preserving deterministic splits, forbidden-input guards, baselines, telemetry artifacts, and focused tests.

## see also

- [[PROJECT_PLAN]]
- [[tests/compression_under_bit_budget_mirror]]
- [[neural_model_dossier_compression_under_bit_budget_codec]]
- [[oracle_compression_frontier_split]]
- [[tests/oracle_compression_analysis_results]]
- [[tests/hard_symbolic_nm_test_material]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_paper_spine]]
- [[neural_model_research_test_material_plan]]
- [[neural_model_compression_stack]]
