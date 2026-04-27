# indexed reconstruction compression

status: current (as of 2026-04-27).

## thesis

the current research conclusion is narrow:

- do not pursue "more bits per neuron" as the primary novelty target
- pursue fewer committed bits per useful memory by changing what is stored
- store compact addresses, residuals, schema ids, and provenance
- reconstruct content through a shared decoder or model prior
- rewrite memories through replay when compression improves without losing task-relevant state

the mathematical target is not a larger scalar neuron. it is a local state machine that decides when an event is worth committing, where it should be addressed, how much residual information must be stored, and when it should be rewritten into a cheaper representation.

## why raw bit capacity is not the bottleneck

bartol et al. measured at least 26 distinguishable synaptic strengths in hippocampal ca1, about 4.7 bits per synapse. that supports few-bit local state as biologically plausible, but it does not make a synapse an independently addressable memory cell. the same study notes that stochastic synaptic activation requires averaging over minutes.

for this project, the paid failures do not look like insufficient numeric precision. six paid runs reached 0 percent passkey across two substrates, two retention regimes, and two corpora. the stronger diagnosis is that useful information is not reliably written, protected, addressed, trained, and read.

the cell-level lesson points the same way. pyramidal neurons are better understood as branch-local nonlinear subunits feeding a final output than as one scalar activation. dendrites add local coincidence, local state, and local commit conditions. energy-budget work also favors sparse distributed activity over dense continuous activity. the useful import is not molecular detail. it is conditional write permission plus state separated across timescales.

## evidence that changes the direction

- synaptic precision is real but bounded: about 4.7 bits per synapse in the ca1 measurement, with slow averaging needed for precision.
- single neurons expose branch-local nonlinear computation, so a "neuron" should be treated as a small local state system, not one number.
- predictive coding and sparse coding both reduce entropy before later processing: predictable activity should not be committed with the same force as surprising activity.
- hippocampal indexing and complementary learning systems point away from full-content storage and toward compact handles that reinstate wider cortical state.
- reconsolidation evidence supports treating retrieval as a possible rewrite window rather than a purely passive read.
- the local correction-field simulations found no memory-side capacity gain from residual values, but did preserve the reconstruction-side interpretation: prediction plus residual correction can improve reconstructed content even when the memory substrate itself does not store more patterns.
- the current symbolic phase-1 battery is useful, but the model-side evaluation still needs to ask whether state, action, compression, replay, and hard-case rollout improve together.

## candidate mathematical object

use a fixed-size memory record of this form:

```text
p_t = f_theta(h_{t-1}, c_{t-1})
r_t = h_t - p_t
u_t = ||r_t||^2 / (||h_t||^2 + eps)
e_t = lambda_e * e_{t-1} + g_theta(h_t, c_t)
a_t = A_theta(h_t, e_t, c_t)
z_t = gate_theta(u_t, e_t, c_t)
m_i = (a_i, s_i, Q_phi(r_i), source_i)
p_hat_i = P_phi(q, a_i, s_i, c_t)
h_hat_i = p_hat_i + G_phi(q, a_i, s_i, Q_phi(r_i))
```

where:

- `p_t` is the model's local prediction
- `r_t` is the residual that remains worth storing
- `u_t` is the surprise ratio
- `e_t` is an eligibility-like local trace
- `a_t` is the address
- `z_t` is the write decision
- `s_i` is a schema or coarse latent id
- `Q_phi(r_i)` is a compact residual code
- `source_i` is a provenance id or hash, not an input to reconstruction
- `P_phi` is a shared prior decoder that reconstructs the predictable part at read time
- `G_phi` reconstructs the residual correction from query, address, schema, and residual

this changes the stored variable. the memory no longer tries to store every hidden vector or an unbudgeted prediction vector. it stores a compact handle, provenance, and the part of the hidden vector that the shared prior and schema cannot reconstruct.

## what would be novel if proved

the defensible novelty target is:

fixed-size memory that stores compact semantic handles plus schema and residual codes, reconstructs through a shared decoder, and rewrites itself through replay under an explicit rate-distortion objective.

that claim is not yet proved. the point of this page is to make the proof obligations explicit before any architecture work resumes.

the nearest components already exist separately in the literature: predictive filtering, sparse codes, indexing theory, complementary learning systems, discrete latent codebooks, world models, and reconsolidation. the project opportunity is the compound system, not any one imported mechanism.

## recent spiking-source check

two 2026 spiking sources sharpen the conclusion rather than reversing it.

- zhang et al.'s complemented ternary spiking neuron adds a learnable complement term that preserves historical input information in the membrane potential, then regularizes membrane-potential aggregation across layers and timesteps. the portable math is polarity/history-preserving local state plus membrane-summary training, not a proof that a discrete spike alone stores more useful task bits.
- neuronspark-0.9b trains a 0.9b-parameter spiking language model from random initialization with next-token prediction and surrogate gradients. its useful design signal is leakage-current inter-layer communication: downstream layers consume floating-point membrane-leakage signals by default rather than only binary spikes.
- project implication: if spiking work resumes, the cpu probes should test polarity-separated local state, membrane-summary channels, and variable internal timesteps under state/action/compression gates. hard event quantization alone should not be treated as the compression breakthrough.

## required cpu proof obligations

before architecture resumes, the project needs model-native probes rather than another language-model run:

1. model-side phase-1 mirror harness: run a tiny local model on the same latent worlds as the symbolic battery and report `state_probe_accuracy`, `action_success`, and `joint_success`.
2. trainability localization: test oracle write / learned read, learned write / oracle read, hand-opened gates, gate-init sweeps, and address-orthogonality sweeps.
3. multi-timescale state ladder: test short cue, medium episode, and slow context retention at multiple gaps.
4. compression-under-budget: compare verbatim, surprise-only, compact-address, schema-residual, and no-memory policies by bits written per episode versus state/action/joint success.
5. replay-rewrite: compare no replay, random replay, oracle targeted replay, and learned targeted replay; require improvement over both no replay and random replay.
6. iterative rollout: test 0, 1, 3, and 5 internal iterations and require hard-case gain larger than easy-case gain.
7. explicit rate-distortion gate: optimize or select with an objective of the form `D_task + lambda * R(memory) + mu * I_interference + rho * C_rewrite`, sweep `lambda`, and show a Pareto improvement over verbatim, surprise-only, schema-residual, and no-memory baselines.

## see also

- [[neural_model_paper_spine]]
- [[oracle_compression_analysis_plan]]
- [[tests/oracle_compression_analysis_results]]
- [[oracle_compression_frontier_split]]
- [[neural_model_lane_cellular_state_storage]]
- [[neural_model_lane_operation_preserving_compression]]
- [[cellular_molecular_computational_primitives]]
- [[correction_field_memory]]
- [[compression_beyond_quantization]]
- [[generative_memory_research]]
- [[memory_compression_to_tiered_architecture]]
- [[neural_model_compression_stack]]
- [[neural_model_dossier_compression]]
- [[neural_model_research_test_material_plan]]
- [[phase1_evaluation_surface_for_neural_models]]
- [[research_implications_for_neural_model_direction]]
- [[substrate_requires_architectural_change]]

## references

- [bartol et al. 2015, nanoconnectomic upper bound on synaptic plasticity](https://elifesciences.org/articles/10778)
- [poirazi, brannon, mel 2003, pyramidal neuron as two-layer neural network](https://pubmed.ncbi.nlm.nih.gov/12670427/)
- [stuart and spruston 2015, dendritic integration](https://www.nature.com/articles/nn.4157)
- [major, larkum, schiller 2013, active dendritic properties](https://www.annualreviews.org/doi/10.1146/annurev-neuro-062111-150343)
- [rao and ballard 1999, predictive coding in visual cortex](https://www.nature.com/articles/nn0199_79)
- [olshausen and field 1996, sparse coding of natural images](https://www.nature.com/articles/381607a0)
- [attwell and laughlin 2001, energy budget for signaling](https://journals.sagepub.com/doi/10.1097/00004647-200110000-00001)
- [teyler and discenna 1986, hippocampal memory indexing theory](https://pubmed.ncbi.nlm.nih.gov/3008780/)
- [bartlett 1932, remembering: a study in experimental and social psychology](https://archive.org/details/in.ernet.dli.2015.188085)
- [mcclelland, mcnaughton, o'reilly 1995, complementary learning systems](https://doi.org/10.1037/0033-295X.102.3.419)
- [nader, schafe, ledoux 2000, reconsolidation after retrieval](https://www.nature.com/articles/35021052)
- [van den oord, vinyals, kavukcuoglu 2017, neural discrete representation learning](https://arxiv.org/abs/1711.00937)
- [hafner et al. 2023, mastering diverse domains through world models](https://arxiv.org/abs/2301.04104)
- [balle et al. 2018, variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436)
- [zhang et al. 2026, ternary spiking neural networks enhanced by complemented neurons and membrane potential aggregation](https://arxiv.org/abs/2601.15598)
- [enhanced ternary snn implementation](https://github.com/ZBX05/Enhanced-TernarySNN)
- [tang 2026, neuronspark: a spiking neural network language model with selective state space dynamics](https://arxiv.org/abs/2603.16148)
- [brain2nd neuronspark-0.9b model card](https://huggingface.co/Brain2nd/NeuronSpark-0.9B)
