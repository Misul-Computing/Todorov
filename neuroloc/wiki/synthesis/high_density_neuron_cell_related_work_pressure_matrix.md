# high density neuron-cell related-work pressure matrix

status: current (as of 2026-05-09).

## role

this page records the pressure matrix for the proposed high-density neuron-cell compression target. it separates public evidence from project claims. public work is used as pressure and mechanism evidence, not as authorization to overclaim.

## current local evidence

the current upstream local result is [[tests/local_100k_full_nm]]. it proves supervised operation-preserving compression on the deterministic exact-state 3d bridge: one trainable module preserves world-state, replay, rewrite, branch-transition, action, provenance, and bounded-answer operations while reducing accounted code from `51` to `44` bits. it does not prove a 600x neuron-density claim.

the first high-density cell proof is [[tests/local_100k_high_density_cell]]. it stores exact associative facts in a bounded hybrid cell and reports two density lines. the params-only line clears the 600x target. the strict params-plus-committed-state line does not. therefore no strict 600x breakthrough is accepted.

the schema-density attempt [[tests/local_100k_schema_density_cell]] is now demoted by [[mistakes/schema_density_cell_structured_target_category_error]]. structured schema-generated facts are planned by construction, so the result is formula compression rather than the requested high-density knowledge-compression target. the target remains unsolved.

the post-error boundary split is now explicit. [[tests/local_100k_unstructured_density_cell]] shows independent random-label exact facts cannot satisfy a 600x strict budget because the entropy lower bound is far above the allowed state. [[tests/local_100k_unknown_structure_density_probe]] shows non-generated wiki corpus chunks are exactly recoverable through charged standard compression and a random-label twin collapses, but strict density is only 34.8547946799184 useful bits per parameter-equivalent, or 13.941917871967359x. the next claim must beat this charged corpus-codec baseline, not only no-memory.

[[tests/local_100k_learned_unknown_structure_density_cell]] tests the obvious learned unknown-structure next step: source-heldout chunks, opaque associative keys, a learned byte-phrase dictionary, exact residual decoding, and random-label controls. it passes exact heldout retrieval and cross-label random scoring collapses, but a separately built random-label twin also stores at 1.0. hard strict multiplier is only 3.0525410753623334x, selected standard-codec comparison is 5.029465628030584x, and the prior charged corpus-codec baseline remains stronger at 13.941917871967359x. [[mistakes/learned_unknown_structure_residual_table_defeat]] records the failure mode: learned dictionary plus per-fact residual rows is still a charged residual table, not a high-density neuron-cell.

## pressure matrix

- titans and atlas: prove test-time neural memory is an active big-lab direction and make surprise-gated long-term memory a mandatory comparison. they do not prove high-density exact associative storage under strict state accounting. project use: compare gated commit and test-time write behavior against a mini titans or miras-style baseline. sources: [titans](https://arxiv.org/abs/2501.00663), [atlas](https://arxiv.org/abs/2505.23735).
- miras: frames transformers, recurrent models, and titans as associative memory modules defined by memory architecture, attentional-bias objective, retention gate, and learning rule. project use: treat the high-density cell as one point in this design space, not as an isolated invention. source: [miras](https://arxiv.org/abs/2504.13173).
- memory layers and product-key memory: prove that large factual capacity can be bought through sparse trainable key-value memory without proportional flops. they impose the strongest "is this just a memory table" baseline. project use: compare useful density against product-key and memory-layer-style lookups. sources: [memory layers at scale](https://proceedings.mlr.press/v267/berges25a.html), [large memory layers with product keys](https://arxiv.org/abs/1907.05242).
- content-routed sparse read: proves content-dependent selection over verbatim memory is a mandatory baseline for any compression claim. project use: compressed cells must beat or complement sparse read on useful bits, not merely beat no-memory. source: [[content_routed_sparse_read_prior]].
- bounded recurrent and state-space models: mamba, retnet, rwkv, griffin/hawk, hyena, infini-attention, transformerfam, xlstm, and deltanet pressure any claim that fixed or recurrent state is new. project use: report whether the cell improves retrievable useful state rather than merely using a bounded state variable. sources: [mamba](https://arxiv.org/abs/2312.00752), [retnet](https://arxiv.org/abs/2307.08621), [rwkv](https://arxiv.org/abs/2305.13048), [griffin and hawk](https://arxiv.org/abs/2402.19427), [hyena](https://arxiv.org/abs/2302.10866), [infini-attention](https://arxiv.org/abs/2404.07143), [transformerfam](https://arxiv.org/abs/2404.09173).
- hdc/vsa, perceiver, and object slots: pressure compact-code and object-world claims because distributed symbolic vectors, latent arrays, and object-centric slots already cover much of the "structured compact state" language. project use: require exact operation preservation, provenance, and accounting rather than only representation elegance. sources: [hdc/vsa survey](https://arxiv.org/abs/2111.06077), [perceiver io](https://arxiv.org/abs/2107.14795), [slot attention](https://arxiv.org/abs/2006.15055).
- open-source memory baselines: product-key memory and hdc/vsa implementations make lookup and distributed-symbolic baselines cheap to reproduce locally. project use: future cells should add at least one real open-source product-key or hdc/vsa ablation when the target moves beyond the current codec/residual defeat. sources: [product-key memory repo](https://github.com/lucidrains/product-key-memory), [torchhd paper](https://jmlr.org/papers/v24/23-0300.html), [torchhd repo](https://github.com/hyperdimensional-computing/torchhd).
- tufts neuro-symbolic robotics: shows explicit symbolic structure can beat end-to-end vla models on structured long-horizon manipulation with far lower energy. project use: explicit factor structure is allowed, but it must be charged and must not become a hidden unaccounted table. source: [the price is not right](https://hrilab.tufts.edu/publications/dugganetal26icra.pdf).
- tribe and tribe v2: pressure biological-alignment language, not memory compression. project use: future brain-alignment sanity checks may use tribe-style multimodal brain response modeling, but it cannot justify high-density storage claims. source: [tribe](https://arxiv.org/abs/2507.22229).
- public reverse-engineered or leaked architecture claims: useful only as low-confidence idea leads. they cannot justify acceptance gates, novelty language, or project-native naming.

## acceptance wording

accepted wording for the current high-density cell result:

```text
accepted as a local params-only high-density associative-cell candidate. exact associative retrieval passes and the params-only density clears the 600x target, but the strict params-plus-committed-state line does not. this is not a strict 600x compression breakthrough, not biological neuron-density proof, not arbitrary chat, not paid-scale trainability, and not broad nm completion.
```

for any future 600x claim:

```text
no 600x bits-per-parameter claim is accepted unless the denominator is named, fixed schema/parser/decoder/model costs are charged or explicitly amortized, preserved operations are reported, and the candidate beats sparse-read, latent-token, bounded-recurrent-state, product-key, and verbatim baselines at equal success.
```

## see also

- [[PROJECT_PLAN]]
- [[neural_model_lane_operation_preserving_compression]]
- [[neural_model_related_work_pressure_matrix]]
- [[compression_beyond_quantization]]
- [[cellular_state_storage_gap_map]]
- [[tests/local_100k_full_nm]]
- [[tests/local_100k_high_density_cell]]
- [[tests/local_100k_schema_density_cell]]
- [[tests/local_100k_unstructured_density_cell]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[tests/local_100k_learned_unknown_structure_density_cell]]
- [[schema_density_cell_boundary]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[mistakes/unstructured_exact_600x_entropy_wall]]
- [[mistakes/learned_unknown_structure_residual_table_defeat]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
