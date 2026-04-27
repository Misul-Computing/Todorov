# cellular state storage gap map

status: current (as of 2026-04-27).

## purpose

this page records the first no-paid-compute research gap map for the biology-led lane. it asks which cellular, molecular, membrane, dendritic, synaptic, glial, metabolic, and information-theoretic mechanisms can be translated into candidate mathematical operations for the neural model.

the answer is conditional. no biological mechanism is accepted as project-native because it is plausible, recent, or named in the literature. it is accepted only if it becomes a compact operation that improves delayed state, action, memory, compression, replay, or trainability metrics under controls.

## working conclusion

the strongest candidate is not "more bits in a neuron." the stronger target is more retrievable, task-relevant state per committed bit under noise, delay, interference, and a real decoder.

the useful abstraction is a three-plane local state machine:

```text
fast carry state -> eligibility / write-permission trace -> slow stability / control state
```

this structure is supported independently by membrane and dendritic evidence, calcium and synaptic-tagging evidence, synaptic-capacity work, and glial or homeostatic evidence. it also matches the project's failed paid-run evidence: large nominal state existed, but learned write, read, gate, and output usage did not become useful.

## ranked candidate operations

### 1. eligibility-gated local commit

mathematical translation:

```text
e_t = lambda_e e_{t-1} + phi(local_pre_t, local_post_t, context_t, surprise_t)
write_t = gate(e_t, relevance_t, stability_t) * candidate_t
```

preserved operation: delayed write permission. a local event marks a possible write, but a later relevance, surprise, reward, or query-error signal decides whether it becomes durable.

evidence basis: calcium transients, dopamine-gated eligibility traces, synaptic tagging and capture, and local protein-synthesis evidence all support a two-stage mark-then-commit structure. the project should translate that structure into write selection, not literal biochemistry.

failure mode targeted: the prior memory substrates could carry state but did not learn useful writes under the training path. this operation attacks learned-write misalignment directly.

first proof gate: delayed-use hard symbolic worlds where the useful event is only known after distractors. compare oracle trace, learned trace, no trace, random trace, always-write, no-memory, recency-only, and shuffled-address controls.

telemetry: trace norm, trace half-life, write precision and recall on memory-relevant positions, gate-open fraction, downstream contribution norm, gradient flow through write and gate paths.

kill condition: reject if always-write or matched residual state performs the same, if learned traces do not align with memory-relevant positions, or if gains require oracle labels at test time.

### 2. branch-local compartment state

mathematical translation:

```text
b^j_t = lambda_j b^j_{t-1} + W_j x^j_t
y_t = W_o sum_j g_j(context_t) * sigma_j(b^j_t)
```

preserved operation: source segregation and local nonlinear routing before global mixing.

evidence basis: dendritic branch computation, local plateaus, receptor coincidence, and microdomain signaling support partially independent local state. the safe translation is grouped local accumulators and branch gates, not literal dendritic morphology.

failure mode targeted: false binding and interference when multiple sources, contexts, or cues share one mixed state.

first proof gate: context-gated routing and false-bind worlds where the same observation has different meaning under different latent context. shuffled branch assignment and matched-width single-state controls must fail relative to the compartmental candidate.

telemetry: branch usage entropy, branch-specific state norm, gate selectivity by context, false-bind rate, ablation delta per branch.

kill condition: reject if branches collapse to one active branch, if a wider point-state control matches performance, or if branch state helps probes but not actions.

### 3. membrane or subthreshold carry state

mathematical translation:

```text
u_t = alpha u_{t-1} + W_x x_t
p_t = split_pos_neg(u_t)
y_t = expose(g_t, p_t)
```

preserved operation: weak evidence and polarity retention around sparse or discrete events.

evidence basis: membrane-current models, adaptive-threshold spiking work, recent complemented-neuron ternary spiking work, and NeuronSpark-style language-model work all point away from event-only purity. they support preserving subthreshold or leakage state around discrete emissions, but they do not prove world-memory retrieval.

failure mode targeted: hard events can discard sign, magnitude, and history before write, address, or gate decisions.

first proof gate: aliased hidden-state worlds with polarity reversals and subthreshold-only cues. compare scalar activation, sign-only state, no carry state, matched parameter state, and oracle hidden-state readout.

telemetry: local-state norm, positive and negative accumulator balance, polarity-confusion rate, state persistence by delay, downstream contribution norm.

kill condition: reject if it improves local probe accuracy but not delayed action success, or if scalar matched-budget controls perform the same.

### 4. bounded output-capacity state

mathematical translation:

```text
r_t = clip(r_{t-1} + insert_t - remove_t, 0, r_max)
output_t = r_t * memory_read_t
```

preserved operation: memory exposure capacity. this separates "state exists" from "state is allowed to affect the residual stream."

evidence basis: receptor trafficking makes synaptic state usable by changing output efficacy. the project translation is an output-capacity or exposure plane, not receptor biology.

failure mode targeted: prior runs showed plausible memory paths that did not affect answers. a memory path that stores content but remains silent is operationally useless.

first proof gate: learned-read and oracle-write splits where hidden memory content is available, but output exposure must be learned. compare hand-opened gate, learned gate, fixed open gate, fixed closed gate, and matched residual controls.

telemetry: output-capacity state, gate-open fraction, memory-output norm versus residual norm, read contribution to answer logits or action choice, gradient norm through output gate.

kill condition: reject if output gates open into noise, remain closed, or look healthy while `joint_success` is unchanged.

### 5. slow homeostatic and stability control

mathematical translation:

```text
h_t = rho h_{t-1} + (1 - rho) activity_t
scale_t = exp(eta(target - h_t))
threshold_t = threshold_0 + W_h h_t
```

preserved operation: anti-saturation and anti-dead-path stabilization.

evidence basis: homeostatic plasticity, glial modulation, potassium buffering, metabolic support, and adaptive threshold evidence all support a slow control plane that regulates excitability and resource use.

failure mode targeted: closed gates, noisy residual injection, state explosion, state decay to zero, and seed-fragile recurrent dynamics.

first proof gate: burst and interference curricula where unregulated state saturates or dies. compare homeostatic state against layer normalization, clipping, static thresholds, and matched learned scalar gates.

telemetry: state norm, threshold or scale trajectory, address entropy, gate-open fraction, read concentration, gradient flow, early-warning correlation between stability metrics and task failure.

kill condition: reject if simple normalization or clipping matches it, or if stability telemetry improves without state/action/joint success.

### 6. active forgetting and utility-weighted cleanup

mathematical translation:

```text
utility_i = ema(contribution_i * gradient_i - interference_i)
state_i <- decay_or_delete(state_i) when utility_i < floor
```

preserved operation: targeted removal of harmful or stale local state.

evidence basis: microglial pruning and forgetting evidence supports active information removal as a biological operation. the project translation is utility-weighted state cleanup, not literal immune-like machinery.

failure mode targeted: continual-write drift, interference accumulation, and storing nuisance detail because there is no controlled deletion.

first proof gate: nonstationary associative recall and replay/rewrite worlds with protected memories and obsolete memories. compare ordinary decay, magnitude pruning, no forgetting, random deletion, and oracle deletion.

telemetry: utility score distribution, protected-memory survival, obsolete-memory removal, interference slope, replay rewrite quality, bits written per successful episode.

kill condition: reject if ordinary decay or magnitude pruning matches it, or if forgetting improves compression while losing task-relevant state.

### 7. resource-budgeted write and compute control

mathematical translation:

```text
budget_t = rho budget_{t-1} + supply_t - cost(reads_t, writes_t, active_units_t)
use_path_t = gate(candidate_t, budget_t, expected_value_t)
```

preserved operation: task-relative allocation under compute or memory cost.

evidence basis: neural information is energy constrained, and glial or metabolic systems regulate support and resource availability. the project translation is cost-aware write and route selection.

failure mode targeted: storing everything, routing every token through expensive memory, or accepting compression ratios that hide a compute-cost increase.

first proof gate: bit-budget and compute-budget worlds where relevant state is sparse among distractors. compare static sparsity, fixed top-k, no budget, oracle budget, and learned budget.

telemetry: active compute, write frequency, budget trajectory, success per active unit, failure prediction from budget exhaustion.

kill condition: reject if static sparsity or fixed top-k matches it.

### 8. slow route consolidation

mathematical translation:

```text
u_i = ema(route_i_contribution)
gain_i <- gain_i + eta u_i
delay_i <- delay_i - eta u_i
```

preserved operation: repeated useful routes become easier or faster to use.

evidence basis: activity-dependent myelination supports slow timing and route adjustment. this is biologically strong, but first implementation risk is high because it can collapse into static skip or gain parameters.

failure mode targeted: timing mismatch and unreliable long-horizon internal routes.

first proof gate: delayed-use and iterative-rollout worlds where timing or route reliability matters. compare static skip, static gain, learned route without slow utility, and shuffled temporal controls.

telemetry: route utility, delay or gain changes, hard-case rollout gain, easy-case rollout gain, route ablation delta.

kill condition: defer if static routing matches it or if no task requires timing adaptation.

## information constraints

raw local-state capacity is only an upper bound:

```text
B_raw = sum_i log2 |A_i|
```

for noisy continuous state, effective bits are closer to:

```text
B_eff ~= 0.5 log2 det(I + Sigma_signal Sigma_noise^-1)
```

the hierarchy that matters is:

```text
stored_bits >= retrievable_bits >= task_useful_bits
```

compression must therefore be measured by task-relative rate-distortion:

```text
L = D_task + lambda R + mu I_interference + rho C_rewrite
```

where `D_task` includes state-probe failure, action failure, reconstruction failure, and joint failure. a smaller code that preserves tensor reconstruction while losing task action is not compression for this project.

## what not to do

- do not treat biological capacity estimates as direct evidence that a model has retrievable memory.
- do not copy calcium, dendrite, glial, myelin, receptor, or channel machinery literally.
- do not accept a mechanism that improves next-token loss without state/action/joint success.
- do not accept a mechanism that only works with oracle labels, hand-opened gates, or hand-placed addresses.
- do not use a published technique name as a project-native component name.
- do not use passkey-style recall as the only win.

## missing proof obligations

1. prove that local state improves delayed hidden-state belief and action success, not only local probes.
2. prove that write permission aligns with memory-relevant positions under distractors.
3. prove that branch-local state reduces false binding rather than only adding capacity.
4. prove that subthreshold carry survives noise and delay with read margins above controls.
5. prove that output-capacity state prevents silent memory paths without injecting noise.
6. prove that homeostatic control is more useful than ordinary normalization or clipping.
7. prove that active forgetting improves a rate-distortion frontier rather than erasing useful knowledge.
8. prove that any compression claim beats verbatim storage at equal task success.

## first symbolic tests

the next implementation should still be symbolic and oracle-first. no full model path is justified by this page.

1. local-write permission test: delayed relevance after distractors; compare oracle trace, learned trace, no trace, random trace, always-write, no-memory, recency-only, and shuffled-address controls.
2. compartmental route test: multiple source streams with false-bind traps; compare branch-local state against matched-width single-state and shuffled-branch controls.
3. membrane-summary test: aliased observations with subthreshold polarity history; compare carry state against scalar, sign-only, no-carry, and matched-parameter controls.
4. output-exposure test: memory content is available but must affect action; compare learned exposure, hand-opened exposure, fixed open, fixed closed, and matched residual controls.
5. homeostatic fixed-point test: burst and interference curricula; compare slow set-point state against normalization, clipping, static thresholds, and matched learned scalar controls.
6. active-forgetting test: nonstationary associative recall with protected and obsolete memories; compare utility cleanup against ordinary decay, magnitude pruning, random deletion, and oracle deletion.

## relation to compression

cellular-state mechanisms are useful for compression only if they reduce committed bits at equal or better task success. the likely path is indirect:

- eligibility traces reduce writes by delaying commit until relevance is known.
- branch-local state reduces false binding, improving useful bits per stored association.
- subthreshold carry preserves evidence before it becomes worth committing.
- active forgetting removes harmful or obsolete state.
- resource budgets force a Pareto frontier between active compute, stored bits, and success.

none of these is a compression win until oracle bounds and symbolic controls show fewer committed bits per successful episode against verbatim storage.

## side-paper trigger

this lane becomes paper-worthy only if a candidate operation clears all of these:

- the operation is stated mathematically.
- the prior-art boundary is explicit.
- the preserved operation is not merely a tensor reconstruction.
- oracle or symbolic bounds show why the operation should help.
- a trainable mirror learns it above matched controls.
- telemetry proves the intended path is used.
- kill conditions are strict enough to reject plausible but useless biological imports.

## next action

the first mechanism dossier from this gap map is now [[neural_model_dossier_eligibility_gated_local_commit]]. the next work is to define the symbolic/oracle episode contract for delayed relevance local commit, bounded output exposure, crossed commit/exposure splits, and commit compression frontier. in parallel, start the oracle compression analysis on the hard symbolic worlds so the project can decide whether reduced writes produce a real rate-distortion advantage.

## see also

- [[PROJECT_PLAN]]
- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_dossier_eligibility_gated_local_commit]]
- [[neural_model_dossier_local_neuron_state]]
- [[cellular_molecular_computational_primitives]]
- [[cellular_molecular_neurobiology_research]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_lane_trainability_evaluation]]
- [[neural_model_research_test_material_plan]]
- [[substrate_requires_architectural_change]]
- [[tests/hard_symbolic_nm_test_material]]
- [[INDEX]]

## references

- bartol et al. 2015, synaptic nanoconnectomic estimates: https://elifesciences.org/articles/10778
- gerstner et al. 2018, eligibility traces and three-factor rules: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00053/full
- yagishita et al. 2014, dopamine timing window for spine plasticity: https://www.science.org/doi/10.1126/science.1255514
- frey and morris 1997, synaptic tagging and capture: https://www.nature.com/articles/385533a0
- redondo and morris 2011, tagging and capture review: https://www.nature.com/articles/nrn2963
- sanhueza and lisman 2023, synaptic memory and camkii: https://pmc.ncbi.nlm.nih.gov/articles/PMC10642921/
- shen et al. 2024, camkii autophosphorylation and synaptic memory: https://pubmed.ncbi.nlm.nih.gov/38889145/
- zhang et al. 2023, structural camkii functions in ltp induction: https://www.nature.com/articles/s41586-023-06465-y
- london and hausser 2005, dendritic computation: https://www.annualreviews.org/content/journals/10.1146/annurev.neuro.28.061604.135703
- polsky et al. 2004, branch-local subunits: https://www.nature.com/articles/nn1253
- larkum et al. 1999, bac firing: https://www.nature.com/articles/18686
- henneberger et al. 2010, astrocyte d-serine and plasticity gating: https://www.nature.com/articles/nature08673
- suzuki et al. 2011, astrocyte lactate and memory: https://pmc.ncbi.nlm.nih.gov/articles/PMC3073831/
- stellwagen and malenka 2006, glial tnf-alpha and synaptic scaling: https://www.nature.com/articles/nature04671
- wang et al. 2020, microglial forgetting: https://pubmed.ncbi.nlm.nih.gov/32029629/
- mckenzie et al. 2014, activity-dependent myelination: https://pubmed.ncbi.nlm.nih.gov/25324381/
- fields 2015, activity-dependent myelination review: https://www.nature.com/articles/nrn4023
- niven and laughlin 2008, energy as a neural constraint: https://journals.biologists.com/jeb/article/211/11/1792/9506/Energy-limitation-as-a-selective-pressure-on-the
- benna and fusi 2016, multi-timescale synaptic memory: https://www.nature.com/articles/nn.4401
- cabannes et al. 2024, associative-memory interference from correlated embeddings: https://arxiv.org/abs/2402.18724
- complemented-neuron ternary spiking paper: https://arxiv.org/abs/2601.15598
- neuronspark 0.9b paper: https://arxiv.org/abs/2603.16148
