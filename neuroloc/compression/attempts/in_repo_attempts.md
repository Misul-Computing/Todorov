# in-repo compression attempts

status: current (as of 2026-05-14).

## role

this file is the compression archaeology ledger. it records what was tried, why it was tried, what it measured, why it passed or failed, and what constraint remains.

## hard symbolic nm compression material

files:

- `neuroloc/wiki/tests/hard_symbolic_nm_test_material.md`
- `neuroloc/simulations/memory/nm_hard_symbolic_test_material.py`
- `tests/test_hard_symbolic_nm_suite.py`
- `tests/test_nm_hard_symbolic_worlds.py`

why tried:

before model work, the project needed non-leaky symbolic task material for belief state, associative recall, correlated interference, delayed use, context routing, compression under bit budget, replay, rewrite, rollout, and recombination.

result:

the test material validated task and control separation. it did not prove learned compression.

constraint left:

future compression claims must preserve operations under explicit controls. reconstruction without action, memory, replay, or rollout success is not enough.

## oracle compression analysis

files:

- `neuroloc/wiki/tests/oracle_compression_analysis_results.md`
- `neuroloc/simulations/memory/oracle_compression_analysis.py`
- `tests/test_oracle_compression_analysis.py`

why tried:

the project needed to know whether the symbolic worlds had enough structured redundancy to justify a learned mirror.

key metrics:

- contracts: `448`
- families: `14`
- operation preservation: `1.0`
- controls preservation: `1.0`
- leakage-free rate: `1.0`
- accepted rate: `0.5714`
- strong oracle families: `8`
- weak oracle families: `6`
- kill-condition count: `192`
- best oracle ratio range: `7.09x` to `39.0x`
- hard-symbolic schema-ratio mean: `11.23x`
- imagination branch-program ratio mean: `39.0x`
- global trainable mirror recommendation: `0.0`

why limited:

oracle coding showed a frontier, not a learned model. six families missed the 10x useful-compression threshold.

constraint left:

only narrow family mirrors were justified. no global compression model was authorized from oracle counters alone.

## compression under bit budget mirror

files:

- `neuroloc/wiki/tests/compression_under_bit_budget_mirror.md`
- `neuroloc/simulations/memory/compression_under_bit_budget_mirror.py`
- `tests/test_compression_under_bit_budget_mirror.py`

why tried:

this was the first local learned-codec mirror for the accepted bit-budget family.

key results:

- original learned codec train joint success: `1.0`
- original learned codec validation joint success: `0.0`
- original learned codec test joint success: `0.125`
- learned compression ratio versus verbatim: `2.74x`
- content-routed sparse-read joint success: `1.0`
- content-routed sparse-read committed bits: `40`
- matched-budget sparse-read joint success: `0.0`
- matched-budget sparse-read committed bits: `20`
- distributed-evidence sparse read joint success: `1.0` at `80` bits
- distributed-evidence matched-budget sparse read joint success: `0.0` at `20` bits
- tiny distributed model ordinary-split test joint success: `1.0` at `19` bits
- factor-heldout shared model test joint success: `0.03125`
- factorized structured local codec minimum joint success across four axes and three seeds: `1.0`
- factorized structured codec parameter count: `9792`

why mixed:

the source-pair task turned out to be solvable by legal sparse read if enough bits were allowed. the ordinary split learned model passed, but the factor-heldout gate killed the shared tuple-memorizing path. field-specific factorized heads repaired that local symbolic surface.

constraint left:

sparse read is a mandatory baseline. learned compression must survive factor-heldout recombination and must not depend on evaluator-source shortcuts.

## local state write/read mirror

files:

- `neuroloc/wiki/tests/local_state_write_read_mirror.md`
- `neuroloc/simulations/memory/local_state_write_read_mirror.py`
- `tests/test_local_state_write_read_mirror.py`

why tried:

the project needed a compact-state write/read/update component after the bit-budget work.

key metrics:

- maximum trainable parameters: `37719`
- accounted bits: `56`
- minimum joint/state/action success: `0.96875`
- update success: `1.0`
- matched-budget sparse-read success: `0.0`
- zero-state, shuffled-state, no-update, and random-update controls: `0.0`

why limited:

this was component evidence. it did not contain replay, imagination, 3d world state, or a full objective.

constraint left:

component mirrors need promotion gates before model-language claims.

## local 100k replay answer mirror

files:

- `neuroloc/wiki/tests/local_100k_replay_answer_mirror.md`
- `neuroloc/simulations/memory/local_100k_replay_answer_mirror.py`
- `tests/test_local_100k_replay_answer_mirror.py`

why tried:

the next local model needed delayed compact-code reactivation, rewrite, bounded answer decoding, and branch rollout.

key metrics:

- smoke maximum trainable parameters: `89877`
- smoke accounted bits: `56`
- smoke minimum initial, replay, and rewrite success: `0.96875`
- smoke branch rollout success: `1.0`
- hard maximum trainable parameters: `98817`
- hard accounted bits: `60`
- hard initial and replay success: `1.0`
- hard rewrite success: `0.96875`
- hard branch rollout success: `0.984375`

failure repaired:

hard-profile language numbers initially wrapped positions `21` to `30`; the vocabulary was repaired.

constraint left:

symbolic-language success is not arbitrary chat and not 3d grounding.

## local 100k exact-state 3d nm mirror

files:

- `neuroloc/wiki/tests/local_100k_3d_nm_mirror.md`
- `neuroloc/simulations/memory/local_100k_3d_nm_mirror.py`
- `tests/test_local_100k_3d_nm_mirror.py`
- `neuroloc/wiki/mistakes/local_100k_3d_branch_rollout_overfit.md`

why tried:

the symbolic-language candidate needed a deterministic exact-state 3d bridge for object permanence, occlusion, delayed use, simple dynamics, and counterfactual queries.

key metrics:

- maximum trainable parameters: `66559`
- accounted bits: `51`
- initial world-state success: `1.0`
- object permanence success: `1.0`
- occluded localization success: `1.0`
- action consequence success: `1.0`
- targeted replay success: `1.0`
- rewrite success: `0.9666666666666667`
- counterfactual transition success: `1.0`
- listed controls: `0.0`

failure:

the learned branch transition over generated trajectories overfit. the accepted branch path became exact compact transition over decoded state.

constraint left:

this is a baseline exact-state 3d candidate, not learned physics.

## local 100k full nm

files:

- `neuroloc/wiki/tests/local_100k_full_nm.md`
- `neuroloc/simulations/memory/local_100k_full_nm.py`
- `tests/test_local_100k_full_nm.py`
- `neuroloc/wiki/mistakes/local_100k_full_nm_soft_code_false_pass.md`

why tried:

the component mirror had to become one trainable `torch.nn.Module` with a single forward path.

key metrics:

- maximum trainable parameters: `81070`
- learned latent state bits: `24`
- fixed bridge/schema/answer bits: `20`
- accounted bits: `44`
- exact-state 3d baseline bits: `51`
- useful density advantage over 3d baseline: `0.003119429590017826`
- all main hard operation successes: `1.0`
- no-memory, code-disabled, shuffled-code, decoder-disabled, no-replay, random-replay, no-branch, wrong-branch, no-integration, wrong-dynamics controls: `0.0`

failure repaired:

soft continuous latent codes trained well but failed when forced into hard compressed bits. the soft path was rejected.

constraint left:

this is the current top local full small nm candidate, but it is synthetic and supervised on the exact-state bridge. it does not prove arbitrary chat, external simulator transfer, paid-scale trainability, or unsupervised world-code discovery.

## local 100k high-density cell

files:

- `neuroloc/wiki/tests/local_100k_high_density_cell.md`
- `neuroloc/simulations/memory/local_100k_high_density_cell.py`
- `tests/test_local_100k_high_density_cell.py`
- `neuroloc/wiki/mistakes/local_100k_high_density_cell_strict_600x_not_met.md`

why tried:

the project needed to test the requested high-density associative-cell target directly.

key metrics:

- fact count: `4096`
- trainable parameters: `8`
- useful retrievable bits: `114688`
- committed state bits: `118816`
- exact retrieval success: `1.0`
- params-only multiplier over 2.5 bits per parameter: `5734.4x`
- strict multiplier after committed-state accounting: `6.1709981167608285x`
- strict 600x pass: `0.0`

why limited:

params-only density cleared the target, but strict accounting killed it. the stored state was the actual denominator.

constraint left:

state bits must be charged. params-only density cannot substitute for strict density.

## schema density cell

files:

- `neuroloc/wiki/tests/local_100k_schema_density_cell.md`
- `neuroloc/wiki/synthesis/schema_density_cell_boundary.md`
- `neuroloc/wiki/mistakes/schema_density_cell_structured_target_category_error.md`

why tried:

the attempt explored compact schema coefficients over generated exact facts.

key metrics:

- fact count: `32768`
- train facts: `24`
- trainable parameters: `24`
- useful retrievable bits: `983040`
- committed bits: `520`
- exact retrieval success: `1.0`
- strict multiplier: `6959.575221238939x`
- random entropy control: `0.0`

why invalid:

the target was structured by construction. the answer came from a planned formula, not unknown knowledge compression.

constraint left:

structured formula compression is useful only as a boundary artifact.

## unstructured exact-fact density probe

files:

- `neuroloc/wiki/tests/local_100k_unstructured_density_cell.md`
- `neuroloc/wiki/mistakes/unstructured_exact_600x_entropy_wall.md`

why tried:

the project needed to test exact independent random-label facts with no formula, seed oracle, or hidden table.

key metrics:

- fact count: `4096`
- useful retrievable bits: `122880`
- 600x state budget: `1246.72` bits
- entropy gap: `121633.28` bits
- entropy gap multiplier: `98.56262833675565`
- exact retrieval success: `0.0`
- strict 600x possible: `0.0`

why failed:

this is an entropy boundary, not an implementation failure.

constraint left:

future work must use unknown-structure real data where redundancy may exist, with a random-label twin as the lie detector.

## unknown-structure corpus probe

files:

- `neuroloc/wiki/tests/local_100k_unknown_structure_density_probe.md`
- `neuroloc/simulations/memory/local_100k_unknown_structure_density_probe.py`
- `tests/test_local_100k_unknown_structure_density_probe.py`

why tried:

after the structured-target error, the project needed a non-generated corpus baseline.

key metrics:

- fact count: `4096`
- corpus file count: `7`
- corpus bytes: `168591`
- compressed bytes: `50973`
- useful retrievable bits: `1048576`
- committed state bits: `481282`
- decoder bits: `65536`
- manifest bits: `7944`
- query key bits: `73728`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- strict multiplier over 2.5 bits per parameter: `13.941917871967359x`
- strict 600x pass: `0.0`

why useful:

it proved real project corpus redundancy exists and set the charged codec baseline.

constraint left:

any learned unknown-structure cell must beat this result, not just no-memory.

## learned unknown-structure residual cell

files:

- `neuroloc/wiki/tests/local_100k_learned_unknown_structure_density_cell.md`
- `neuroloc/simulations/memory/local_100k_learned_unknown_structure_density_cell.py`
- `tests/test_local_100k_learned_unknown_structure_density_cell.py`
- `neuroloc/wiki/mistakes/learned_unknown_structure_residual_table_defeat.md`

why tried:

this was the obvious learned next step: source-heldout corpus chunks, opaque associative keys, learned byte-phrase dictionary, exact residual decoding, provenance, and random-label controls.

key metrics:

- fact count: `4096`
- train fact count: `2048`
- useful retrievable bits: `1048576`
- committed state bits: `1674075`
- strict accounted bits: `2198363`
- exact retrieval success: `1.0`
- heldout exact retrieval success: `1.0`
- random-label twin storage success: `1.0`
- random-label cross-label success: `0.0`
- strict multiplier: `3.0525410753623334x`
- selected standard-codec multiplier: `5.029465628030584x`
- prior charged corpus-codec baseline: `13.941917871967359x`
- no per-fact committed rows: `0.0`
- strict 600x pass: `0.0`

why failed:

the learned dictionary helped, but exact retrieval still depended on per-fact residual/key rows. the random-label twin could also store exactly, proving the mechanism remained table-shaped.

constraint left:

do not iterate this path. the next candidate must remove per-fact residual rows or prove a genuinely amortized shared code.

## shared-predictor exact codec

files:

- `neuroloc/wiki/tests/local_100k_shared_predictor_exact_codec.md`
- `neuroloc/simulations/memory/local_100k_shared_predictor_exact_codec.py`
- `tests/test_local_100k_shared_predictor_exact_codec.py`

why tried:

after the residual-row defeat, the project needed a no-per-fact-value-slice exact codec product.

key metrics:

- fact count: `4096`
- train fact count: `2048`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- strict multiplier: `4.352614012398447x`
- beats charged codec baseline: `0.0`
- beats mph payload baseline: `0.0`

why limited:

one shared block stream removed per-fact value rows, but opaque-key routing and payload accounting still lost to the charged corpus-codec and minimal-perfect-hash payload baselines.

constraint left:

exactness without rows is not enough. the next path must also beat the classical compressed-source baselines.

## source-block exact codec

files:

- `neuroloc/wiki/tests/local_100k_source_block_codec.md`
- `neuroloc/simulations/memory/local_100k_source_block_codec.py`
- `tests/test_local_100k_source_block_codec.py`
- `neuroloc/wiki/mistakes/source_block_codec_raw_cache_category_error.md`

why tried:

the project needed to test whether the source-heldout corpus could be stored once as a compressed source stream and queried by source id plus offset.

key metrics:

- fact count: `4096`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- strict accounted bits: `476958`
- strict multiplier: `14.06876726917481x`
- raw source block retained: `0.0`
- reads from compressed block: `1.0`

why limited:

the result used source id and offset query fields. it is an exact source codec product, not arbitrary associative memory.

constraint left:

remove source-offset routing without adding assignment rows or a raw decoded source cache.

## content-addressed source codec

files:

- `neuroloc/wiki/tests/local_100k_content_addressed_source_codec.md`
- `neuroloc/simulations/memory/local_100k_content_addressed_source_codec.py`
- `tests/test_local_100k_content_addressed_source_codec.py`

why tried:

the next step was to remove visible source id and offset fields while preserving the one-block compressed source product.

key metrics:

- fact count: `4096`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- strict accounted bits: `476954`
- strict multiplier: `14.06888524576417x`
- selected digest collision count: `0.0`
- source-offset routing: `0.0`
- key assignment bits: `0.0`

why limited:

the query handle was still derived from the target content window. this removed source-offset routing but did not prove semantic retrieval or arbitrary opaque-key memory.

constraint left:

move to a bounded question surface without returning to source offsets, answer digests, assignment rows, or raw caches.

## llm-facing semantic qa codec

files:

- `neuroloc/wiki/tests/local_100k_llm_semantic_qa_codec.md`
- `neuroloc/simulations/memory/local_100k_llm_semantic_qa_codec.py`
- `tests/test_local_100k_llm_semantic_qa_codec.py`

why tried:

the next product needed a bounded natural-language qa surface for llm-facing use, while preserving charged compressed-state accounting.

key metrics:

- fact count: `4096`
- train fact count: `2048`
- parameter count: `3`
- exact answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- strict accounted bits: `476938`
- strict multiplier: `14.06935717190861x`
- beats content-addressed baseline: `1.0`
- beats mph payload baseline: `0.0`
- strict 600x pass: `0.0`

why limited:

the question handle is a lexical token signature. it is useful as a bounded llm-facing qa substrate, but it is not learned semantic recall and it still loses to the minimal-perfect-hash payload line.

constraint left:

learn paraphrase-stable or generative retrieval that beats the minimal-perfect-hash payload baseline without introducing per-fact rows, uncharged maps, or prompt-context storage.

## weight-carried qa codec

files:

- `neuroloc/wiki/tests/local_100k_weight_carried_qa_codec.md`
- `neuroloc/simulations/memory/local_100k_weight_carried_qa_codec.py`
- `tests/test_local_100k_weight_carried_qa_codec.py`
- `neuroloc/compression/candidates/weight_carried_knowledge_adapter.md`

why tried:

the previous llm-facing product was close to the minimal-perfect-hash payload line but still treated the compressed source stream as an external artifact. the next product needed to carry the compressed knowledge inside model state, remove stored manifest cost, lower the fixed parser and decoder budget, and cross the requested `15x` strict multiplier line.

key metrics:

- fact count: `4096`
- train fact count: `2048`
- parameter count: `0`
- exact answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- model-state adapter payload used: `1.0`
- external payload store used: `0.0`
- stored manifest used: `0.0`
- adapter recompression update path: `1.0`
- adapter recompression update success: `1.0`
- adapter state-dict reload success: `1.0`
- committed state bits: `441064`
- strict multiplier: `15.215221373768888x`
- beats llm-facing qa baseline: `1.0`
- beats mph payload baseline: `1.0`
- strict 600x pass: `0.0`

why limited:

the product carries exact qa knowledge in a model-state adapter payload, but the read handle is still a lexical token signature. there is no learned paraphrase-stable semantic retrieval, no ordinary base-weight implicit storage, and no broad chat behavior.

constraint left:

keep the model-state adapter shape and `15x` density line, then add paraphrase-stable learned handles or a trainable update rule that performs decompression and recompression without adding hidden rows.

## paper-ready adapter benchmark

files:

- `neuroloc/wiki/tests/local_100k_paper_ready_adapter_benchmark.md`
- `neuroloc/simulations/memory/local_100k_paper_ready_adapter_benchmark.py`
- `tests/test_local_100k_paper_ready_adapter_benchmark.py`
- `neuroloc/compression/candidates/weight_carried_knowledge_adapter.md`

why tried:

the previous product crossed the `15x` line, but it did not yet prove the five local adapter requirements: host integration, public baseline comparison, multi-domain data, paraphrase-stable bounded qa, and documented category limits.

key metrics:

- fact count: `4096`
- train fact count: `2048`
- source domains: `4`
- maximum host parameter count: `6592`
- exact answer success: `1.0`
- paraphrase-stable answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- transformer surface pass: `1.0`
- recurrent surface pass: `1.0`
- public baseline stack pass: `1.0`
- adapter recompression update success: `1.0`
- adapter state-dict reload success: `1.0`
- committed state bits: `403256`
- paper-surface accounted bits: `407352`
- adapter strict multiplier: `16.641752137599937x`
- paper-surface strict multiplier: `16.474416229698146x`
- strict 600x pass: `0.0`

why limited:

the benchmark is still a token-signature exact adapter codec. paraphrase stability is over bounded evidence-term templates, not learned semantic recall. the payload is in adapter state, not implicitly internalized by ordinary base-model weights.

constraint left:

beat the new `16.641752137599937x` line with learned semantic handles, learned generative retrieval, or a trainable adapter update rule, while preserving exact answers, provenance, random-label collapse, and no hidden rows.

## margin recompression adapter

files:

- `neuroloc/wiki/tests/local_100k_margin_recompression_adapter.md`
- `neuroloc/simulations/memory/local_100k_margin_recompression_adapter.py`
- `tests/test_local_100k_margin_recompression_adapter.py`
- `neuroloc/wiki/mistakes/paper_ready_adapter_source_holdout_overlap.md`

why tried:

the previous adapter result beat the matched minimal-perfect-hash line by only a rounding-width margin and Pauli's review found a train/test source overlap in the source block. the next product needed a large margin over the prior adapter line, concrete source-holdout checks, false-hit controls, and a real trained recompression-update surface while keeping the model-state adapter shape.

key metrics:

- fact count: `4096`
- train fact count: `0`
- source domains: `4`
- maximum host parameter count: `6596`
- update controller parameters: `4`
- exact answer success: `1.0`
- paraphrase-stable answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- false-hit rates: `0.0`
- source holdout pass: `1.0`
- transformer surface pass: `1.0`
- recurrent surface pass: `1.0`
- trainable recompression update success: `1.0`
- update-controller-disabled success: `0.0`
- adapter state-dict reload success: `1.0`
- committed state bits: `295144`
- paper-surface accounted bits: `299240`
- adapter strict multiplier: `22.732738950163952x`
- paper-surface strict multiplier: `22.421639537059313x`
- executable content-scan multiplier: `22.73766839237796x`
- content-scan not beaten: `1.0`
- same-block undercharged mph not beaten: `1.0`
- bounded adapter engineering pass: `1.0`
- paper-ready candidate: `0.0`
- strict breakthrough authorization: `0.0`

why limited:

the product is a better bounded model-state adapter/update engineering benchmark, but the read path is still token-signature based. it does not beat the executable same-block content-scan diagnostic or prove implicit base-weight storage, learned semantic recall, arbitrary chat, paper-ready static compression, or 600x high-density knowledge compression.

constraint left:

preserve or beat the `22.421639537059313x` paper-surface line while replacing token-signature routing with learned semantic retrieval or reducing payload bits below the same-block content-scan diagnostic.

## semantic alias payload adapter

files:

- `neuroloc/wiki/tests/local_100k_semantic_alias_payload_adapter.md`
- `neuroloc/simulations/memory/local_100k_semantic_alias_payload_adapter.py`
- `tests/test_local_100k_semantic_alias_payload_adapter.py`
- `neuroloc/wiki/mistakes/semantic_alias_payload_adapter_formula_alias_category_error.md`

why tried:

the margin adapter lost to the executable content-scan diagnostic by a narrow static-retrieval margin. the next attempt reduced payload bits and replaced visible lexical token signatures with generated aliases to test whether the query surface could move beyond old parser matching.

key metrics:

- fact count: `4096`
- exact answer success: `1.0`
- lexical content-scan success: `0.0`
- alias content-scan success: `1.0`
- controls collapse: `1.0`
- block payload bits: `258144`
- committed state bits: `290952`
- paper-surface multiplier: `22.7450665654402x`
- publishable breakthrough authorization: `0.0`

why limited:

the alias labels were generated from source anchors. when the scanner received the same alias parser, it solved the task exactly.

constraint left:

parser changes must give the scanner the same interface before a baseline win can be claimed.

## source-native fixed-stride relation adapter

files:

- `neuroloc/wiki/tests/local_100k_source_native_relation_adapter.md`
- `neuroloc/simulations/memory/local_100k_source_native_relation_adapter.py`
- `tests/test_local_100k_source_native_relation_adapter.py`
- `neuroloc/wiki/mistakes/source_native_relation_stride_rule_category_error.md`

why tried:

after the alias failure, the next attempt moved toward source-native relations: parse evidence terms, use a charged relation code, and return a related source span rather than the adjacent chunk.

key metrics:

- fact count: `4096`
- exact answer success: `1.0`
- paraphrase-stable answer success: `1.0`
- relationless content-scan success: `0.0`
- stride-aware content-scan success: `1.0`
- controls collapse: `1.0`
- source train/test overlap counts: `0.0`
- block payload bits: `258144`
- committed state bits: `290960`
- paper-surface multiplier: `22.744449867143864x`
- publishable relation breakthrough authorization: `0.0`

why limited:

the relation was still formula-generated: target equals anchor plus a fixed stride. a fair scanner with the same stride relation solved the task exactly.

constraint left:

next relation work must use relations authored in the source corpus, such as wiki links, imports, definitions, or reference edges, and must include a matched relation-aware content-scan baseline.

## weight-mantissa payload adapter

files:

- `neuroloc/wiki/tests/local_100k_weight_mantissa_payload_adapter.md`
- `neuroloc/simulations/memory/local_100k_weight_mantissa_payload_adapter.py`
- `tests/test_local_100k_weight_mantissa_payload_adapter.py`
- `neuroloc/wiki/mistakes/weight_mantissa_payload_steganographic_accounting_error.md`

why tried:

after the source-relation failures, the next attempt tested whether the payload could be carried inside trainable torch parameters instead of a normal state-dict payload buffer.

key metrics:

- fact count: `4096`
- exact answer success: `1.0`
- controls collapse: `1.0`
- adapter parameter count: `11224`
- block payload bits: `258144`
- committed state bits after fp32 charging: `391976`
- paper-surface accounted bits after fp32 charging: `396072`
- apparent mantissa paper-surface multiplier: `30.998884002808474x`
- fp32 paper-surface multiplier: `16.94360217334222x`
- same-block content-scan multiplier: `23.06526987269378x`
- same-block undercharged mph multiplier: `23.064001539688213x`
- static public baseline pass: `0.0`
- publishable weight-payload candidate: `0.0`
- strict breakthrough authorization: `0.0`

why limited:

the apparent multiplier is produced by storing payload bits in fp32 mantissas while charging only parameter count. once the payload bits or physical fp32 state are charged, the result loses to the fair same-block scanner and mph diagnostics.

constraint left:

weight-native compression must prove useful knowledge carried by ordinary learned weights or fully charged adapter state, not bit-pattern payload steganography.

## indent-token source-code block codec

files:

- `neuroloc/wiki/tests/local_100k_indent_token_block_codec.md`
- `neuroloc/simulations/memory/local_100k_indent_token_block_codec.py`
- `tests/test_local_100k_indent_token_block_codec.py`

why tried:

the exact qa adapter line kept losing to same-interface content scan and mph diagnostics. the next positive path moved away from query-answer wrappers and tested a simpler fundamental claim: a train-learned source-code byte token can improve lossless held-out block compression after charging the token map and shared decoder.

key metrics:

- exact reconstruction success: `1.0`
- model-state restore success: `1.0`
- model-state reload success: `1.0`
- model-state payload used: `1.0`
- external payload store used: `0.0`
- compressed-stream read success: `1.0`
- target block bytes: `99761`
- best standard strict bits: `161648`
- learned strict bits: `158384`
- payload improvement over best standard: `0.025338467271146442`
- strict improvement over best standard: `0.020192022171632188`
- paper-surface improvement over best standard: `0.019693020561830293`
- random-label payload improvement: `-0.008129021911272377`
- path/hash train-test overlap counts: `0.0`
- sliding 64-byte ngram overlap count: `3104.0`
- source-block codec product authorization: `1.0`
- source-block codec breakthrough authorization: `0.0`
- strict/general knowledge/chat/full-nm authorization: `0.0`

why limited:

this is a source-code byte-compression product, not high-density knowledge compression. it learns an indentation token from source style and improves a strict codec denominator; it does not prove semantic memory, arbitrary opaque-key exact recall, implicit weight storage, or 600x factual density.

constraint left:

superseded by the source-structure split product below. the next target is wider source-code corpora or model-state integration that still beats same-block standard codecs, content scan, and mph diagnostics after all headers, dictionaries, and decoder bits are charged.

## source-structure split source-code block codec

files:

- `neuroloc/wiki/tests/local_100k_source_structure_block_codec.md`
- `neuroloc/simulations/memory/local_100k_source_structure_block_codec.py`
- `tests/test_local_100k_source_structure_block_codec.py`

why tried:

the indentation-token product proved a narrow exact byte-compression win but only by about `2.019%` strict charged bits. the next simple path separated source indentation structure from body bytes so ordinary codecs could compress each plane more efficiently.

key metrics:

- exact reconstruction success: `1.0`
- compressed-stream read success: `1.0`
- target block bytes: `99761`
- best standard payload bits: `128816`
- learned count payload bits: `4416`
- learned body payload bits: `119528`
- learned structure header bits: `256`
- learned payload bits: `124200`
- payload improvement over best standard: `0.03583405788100857`
- strict improvement over best standard: `0.028555874492724932`
- paper-surface improvement over best standard: `0.027850178588666858`
- strict-improvement delta over indent-token product: `0.008363852321092744`
- random-label payload improvement: `-0.0005412665637591965`
- path/hash train-test overlap counts: `0.0`
- sliding 64-byte ngram overlap count: `3104.0`
- source-block codec product authorization: `1.0`
- source-block codec breakthrough authorization: `0.0`
- strict/general knowledge/chat/full-nm authorization: `0.0`

why limited:

this is still a source-code byte-compression product, not high-density knowledge compression. it uses a simple learned source-structure transform and standard codecs; it does not prove semantic memory, arbitrary opaque-key exact recall, implicit weight storage, or 600x factual density.

constraint left:

superseded by the source-token-structure product below. broaden the corpus and baseline stack, then test whether the transform can be packaged into model-state adapters without losing the same-block codec advantage.

## source-token-structure source-code block codec

files:

- `neuroloc/wiki/tests/local_100k_source_token_structure_block_codec.md`
- `neuroloc/simulations/memory/local_100k_source_token_structure_block_codec.py`
- `tests/test_local_100k_source_token_structure_block_codec.py`

why tried:

the source-structure split improved strict charged bits by `2.855%`, but the identifier-heavy body plane still contained repeated source-code terms. the next simple path kept the count/body split, delta-coded the indentation-count plane, and added a fully charged target identifier dictionary for whole-word body substitution.

key metrics:

- exact reconstruction success: `1.0`
- compressed-stream read success: `1.0`
- target block bytes: `99761`
- best standard payload bits: `128816`
- prior source-structure payload bits: `124200`
- learned count-delta payload bits: `4096`
- learned body-token payload bits: `112864`
- learned dictionary payload bits: `5232`
- learned token-structure header bits: `896`
- learned payload bits: `123088`
- payload improvement over best standard: `0.04446652589740405`
- strict improvement over best standard: `0.03543501930119766`
- strict-improvement delta over source-structure: `0.00687914480847273`
- target charged token count: `120`
- random-label payload improvement: `-0.0012328849507848366`
- controls collapse: `1.0`
- source-block codec product authorization: `1.0`
- strict/general knowledge/chat/full-nm authorization: `0.0`

why limited:

this is still a source-code byte-compression product, not high-density knowledge compression. the target identifier dictionary is charged honestly, so it is valid as a codec component, but it is not evidence that unknown facts are stored inside ordinary neural weights or that broad semantic retrieval is solved.

constraint left:

superseded by the source-subtoken-structure products below. the whole-word token boundary left a measurable payload gap.

## source-subtoken-structure source-code block codec

files:

- `neuroloc/wiki/tests/local_100k_source_subtoken_structure_block_codec.md`
- `neuroloc/simulations/memory/local_100k_source_subtoken_structure_block_codec.py`
- `tests/test_local_100k_source_subtoken_structure_block_codec.py`

why tried:

the source-token-structure product improved strict charged bits by `3.5435%`, but whole-word identifier substitution missed repeated identifier substrings and repeated source-code morphemes. the next simple path kept the count/body/dictionary accounting but changed the body transform to reversible longest-match subtoken substitution.

key metrics:

- exact reconstruction success: `1.0`
- model-state reload success: `1.0`
- target block bytes: `99761`
- best standard payload bits: `128816`
- prior source-token payload bits: `123088`
- learned count-delta payload bits: `4096`
- learned body-subtoken payload bits: `110728`
- learned dictionary payload bits: `5232`
- learned subtoken-structure header bits: `896`
- learned payload bits: `120952`
- payload improvement over best standard: `0.061048316979257236`
- strict improvement over best standard: `0.048648916163515785`
- strict-improvement delta over source-token: `0.013213896862318122`
- target charged token count: `120`
- random-label payload improvement: `-0.0013331194996291321`
- controls collapse: `1.0`
- source-block codec product authorization: `1.0`
- strict/general knowledge/chat/full-nm authorization: `0.0`

why limited:

this is still a source-code byte-compression product, not high-density knowledge compression. the target dictionary is charged honestly, so it is valid as a codec component, but it is not evidence that unknown facts are stored inside ordinary neural weights or that broad semantic retrieval is solved.

constraint left:

the byte-codec path now needs broader external corpus comparison and a model-state adapter variant that beats a fair same-interface scanner, or it must remain a narrow source-code compression result.

## source-subtoken-structure frozen corpus codec

files:

- `neuroloc/wiki/tests/local_100k_source_subtoken_structure_corpus_codec.md`
- `neuroloc/simulations/memory/local_100k_source_subtoken_structure_corpus_codec.py`
- `tests/test_local_100k_source_subtoken_structure_corpus_codec.py`

why tried:

the first token corpus benchmark broadened the transform but had live-checkout drift and a weak control gate. the frozen corpus codec fixes that by hardcoding a five-block manifest, checking hashes, charging selectors and fallback headers, and requiring random-label/control collapse.

key metrics:

- block count: `5`
- exact reconstruction success minimum: `1.0`
- frozen manifest hash success minimum: `1.0`
- aggregate standard payload bits: `849752`
- aggregate selected payload bits: `812688`
- aggregate payload improvement: `0.043617431909545375`
- subtoken-structure selected block count: `5`
- standard fallback selected block count: `0`
- random-label payload incompressible minimum: `1.0`
- random-label payload improvement max: `-0.0006119877602447951`
- controls collapse: `1.0`
- source-code corpus codec product authorization: `1.0`
- strict/general knowledge/chat/full-nm authorization: `0.0`

why limited:

the target is still source-code byte compression, not arbitrary exact associative knowledge. it proves a local transform can beat a strong standard-codec sweep on a fixed source-code manifest under charged dictionary/header accounting, not that broad model knowledge has 600x density.

constraint left:

compare the transform against stronger external source-code compressors and package it into the bounded qa adapter surface without losing the same-interface content-scan comparison.

## source-structure qa adapter

files:

- `neuroloc/wiki/tests/local_100k_source_structure_qa_adapter.md`
- `neuroloc/simulations/memory/local_100k_source_structure_qa_adapter.py`
- `tests/test_local_100k_source_structure_qa_adapter.py`

why tried:

the source-structure split improved the source-code byte payload, so the next adapter step packaged the same count/body split into the bounded qa model-state adapter surface and compared it against raw content scan, raw undercharged mph, and a fair same-structure content scan.

key metrics:

- exact answer success: `1.0`
- heldout exact answer success: `1.0`
- paraphrase-stable answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- transformer surface pass: `1.0`
- recurrent surface pass: `1.0`
- trainable recompression update success: `1.0`
- block payload bits: `246328`
- committed state bits: `279136`
- adapter strict multiplier: `24.03612607449857x`
- paper-surface strict multiplier: `23.688602733536655x`
- raw content scan beaten: `1.0`
- raw undercharged mph beaten: `1.0`
- same-structure content scan multiplier: `24.041637051473117x`
- same-structure content scan beaten: `0.0`

why limited:

once the content scanner receives the same source-structure payload and decoder budget, it remains slightly stronger than the adapter because the adapter also carries four trained update-controller parameters. this is a bounded exact qa product, not a static-retrieval breakthrough.

constraint left:

superseded by the source-subtoken qa adapter below. the next qa adapter still must either reduce payload bits further or add a useful operation that the fair scanner cannot legally perform under the same accounting.

## source-subtoken qa adapter

files:

- `neuroloc/wiki/tests/local_100k_source_subtoken_qa_adapter.md`
- `neuroloc/simulations/memory/local_100k_source_subtoken_qa_adapter.py`
- `tests/test_local_100k_source_subtoken_qa_adapter.py`

why tried:

the source-subtoken byte codec improved the source-code payload. the next adapter step moved the same count-delta, body-subtoken, and charged dictionary streams into the bounded qa model-state adapter surface.

key metrics:

- exact answer success: `1.0`
- heldout exact answer success: `1.0`
- paraphrase-stable answer success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- transformer surface pass: `1.0`
- recurrent surface pass: `1.0`
- trainable recompression update success: `1.0`
- block payload bits: `244440`
- committed state bits: `277248`
- adapter strict multiplier: `24.19976921301639x`
- paper-surface strict multiplier: `23.847532408460314x`
- raw content scan beaten: `1.0`
- raw undercharged mph beaten: `1.0`
- same-subtoken content scan multiplier: `24.20535549399815x`
- same-subtoken content scan beaten: `0.0`

why limited:

the fair content scanner receiving the same subtoken payload and decoder budget remains slightly stronger than the adapter. this is a stronger bounded exact qa/update product, not a static-retrieval breakthrough.

constraint left:

the next qa adapter must add an operation that the fair scanner cannot legally perform under matched bits, or move to learned semantic handles without parser mismatch.

## source-subtoken delta-update adapter

files:

- `neuroloc/simulations/memory/local_100k_source_subtoken_delta_update_adapter.py`
- `tests/test_local_100k_source_subtoken_delta_update_adapter.py`
- `neuroloc/wiki/tests/local_100k_source_subtoken_delta_update_adapter.md`

why tried:

the source-subtoken qa adapter was blocked as a static compression breakthrough because a fair same-subtoken content scan was slightly stronger. the next useful operation was incremental update compression: if the base compressed adapter already exists in model state, store only a charged patch stream for changed facts and compare that patch stream against full recompression and an undercharged mph update table.

key metrics:

- fact count: `4096`
- update fact count: `512`
- unchanged fact count: `3584`
- base payload bits: `244440`
- delta patch bits: `138104`
- total updated adapter bits: `382544`
- full recompress updated bits: `380552`
- same-block content scan update bits: `380552`
- undercharged mph update bits: `188416`
- matched delta-patch content scan bits: `138104`
- margin over full recompress bits: `242448`
- margin over same-block content scan update bits: `242448`
- margin over undercharged mph update bits: `50312`
- margin over matched delta-patch content scan bits: `0`
- total static margin over full recompress bits: `-1992`
- exact updated answer success: `1.0`
- unchanged answer success: `1.0`
- state-dict reload success: `1.0`
- random patch control success: `0.0`
- patch-disabled success: `0.0`
- shuffled-patch success: `0.0`
- controls collapse: `1.0`

why useful:

the repaired patch stream stores varint-delta offsets plus full replacement bytes. it is not a generated xor mask or value formula. as a delta-update product, it beats full updated-adapter recompression as the same-block content-scan update baseline and undercharged mph update storage while preserving exact updated and unchanged answers from model state.

why limited:

total static compression is not solved. base-plus-delta bits lose to full recompression by `1992` bits on hard. the matched delta-patch content scan is equal at `138104` bits and is not beaten. this product is an incremental update result for an existing compressed adapter, not a public source-byte compression breakthrough, not total static knowledge compression, not learned semantic recall, and not full nm behavior.

constraint left:

the next step is either to turn this delta-update path into a learned recompression policy with arbitrary updates and stronger update batches, or return to static compression with a method that actually beats same-subtoken scan and paq8px pressure. do not count generated update programs or deterministic value formulas as compression.

## failed block-codec diagnostics from the same search

files:

- `neuroloc/simulations/memory/local_100k_answer_surface_codec.py`
- `tests/test_local_100k_answer_surface_codec.py`
- `neuroloc/wiki/mistakes/answer_surface_codec_static_scan_not_beaten.md`
- `neuroloc/simulations/memory/local_100k_neural_coordinate_codec.py`
- `tests/test_local_100k_neural_coordinate_codec.py`
- `neuroloc/wiki/mistakes/neural_coordinate_codec_random_label_memorizer.md`
- `neuroloc/simulations/memory/local_100k_learned_block_codec_frontier.py`
- `tests/test_local_100k_learned_block_codec_frontier.py`
- `neuroloc/wiki/mistakes/learned_block_codec_frontier_loses_to_standard_codec.md`

why tried:

parallel probes tested whether answer-only surfaces, neural-coordinate maps, or learned phrase block codecs could produce a stronger compression claim than static retrieval wrappers.

key outcomes:

- answer-surface codec passed exact retrieval but matched same-interface scan and mph payload diagnostics, so it is only a diagnostic.
- neural-coordinate codec failed the hard target and a random-label twin memorized better than the real target, so it is a memorizer boundary.
- learned phrase block codec reconstructed exactly but lost badly to standard codecs, with learned bits `275003` versus best standard bits `112876`.

constraint left:

do not promote these diagnostics. keep them as negative evidence that exact retrieval wrappers, coordinate memorization, and weak learned phrase substitution are not enough.

## zstd trained-dictionary public baseline audit

files:

- `neuroloc/simulations/memory/local_100k_zstd_trained_dictionary_baseline_audit.py`
- `tests/test_local_100k_zstd_trained_dictionary_baseline_audit.py`
- `neuroloc/wiki/tests/local_100k_zstd_trained_dictionary_baseline_audit.md`

why tried:

the source-subtoken block and corpus codecs beat the in-repo standard codec sweep, but a publishable compression direction needs pressure from public trained-dictionary codecs too. the audit tests zstd dictionaries trained only on target-excluded local source files, charges dictionary bytes, header bits, and selector bits, and also reports an undercharged payload-only diagnostic line.

key metrics:

- block current source-subtoken payload bits: `120952`
- block charged public dictionary bits: `151376`
- block undercharged public dictionary bits: `147200`
- block current margin over charged public dictionary: `30424` bits
- block current margin over undercharged public dictionary: `26248` bits
- corpus then-current source-subtoken payload bits: `812688`
- corpus charged public dictionary bits: `982840`
- corpus undercharged public dictionary bits: `949992`
- corpus then-current margin over charged public dictionary: `170152` bits
- corpus then-current margin over undercharged public dictionary: `137304` bits
- train/test path overlap count: `0.0` on block and corpus
- train/test hash overlap count: `0.0` on block and corpus
- random-label charged and undercharged improvements: below `0.0`

why limited:

this strengthens the source-code byte-compression products against a public trained-dictionary baseline, but it is not a new product and not a breakthrough. the bounded qa adapter is still limited by the fair same-subtoken content scanner.

constraint left:

public trained-dictionary pressure is covered for the source-code codec products. the later global-stream corpus product extends the margin over the same trained-dictionary lines; the next useful move must beat the same-interface qa scanner or move to a learned semantic handle without parser mismatch or hidden rows.

## source-subtoken shared-dictionary corpus codec

files:

- `neuroloc/simulations/memory/local_100k_source_subtoken_shared_dictionary_corpus_codec.py`
- `tests/test_local_100k_source_subtoken_shared_dictionary_corpus_codec.py`
- `neuroloc/wiki/tests/local_100k_source_subtoken_shared_dictionary_corpus_codec.md`

why tried:

the prior source-subtoken corpus product charged a separate target dictionary for every block. a broader corpus codec can amortize cross-block identifier-subtoken redundancy by charging one shared dictionary once and only small local dictionaries per block.

key metrics:

- aggregate standard payload bits: `849752`
- prior source-subtoken corpus payload bits: `812688`
- aggregate selected payload bits: `803400`
- aggregate payload improvement over standard: `0.05454767979363391`
- aggregate payload margin over prior: `9288` bits
- shared token count: `112`
- local token count per block: `16`
- shared dictionary payload bits: `4360`
- margin over charged zstd trained-dictionary baseline: `179440` bits
- margin over undercharged zstd trained-dictionary diagnostic: `146592` bits
- random-label payload improvement: `-0.00020807116781500356`
- controls collapse: `1.0`

why limited:

this is a stronger source-code corpus byte codec, not arbitrary unknown-structure knowledge compression and not a qa/static-retrieval breakthrough. it does not beat the same-subtoken qa content scanner because it does not change that surface.

constraint left:

superseded by the global-stream corpus product below. the next qa-facing attempt must still either beat the fair same-interface scanner with a real operation or move to learned semantic handles without parser mismatch.

## source-subtoken global-stream corpus codec

files:

- `neuroloc/simulations/memory/local_100k_source_subtoken_global_stream_corpus_codec.py`
- `tests/test_local_100k_source_subtoken_global_stream_corpus_codec.py`
- `neuroloc/wiki/tests/local_100k_source_subtoken_global_stream_corpus_codec.md`

why tried:

the shared-dictionary corpus product proved that a dictionary can be amortized across the frozen source-code corpus, but the body payload still dominated the bit budget. the next simple path keeps the charged shared dictionary and source-structure split, removes per-block local dictionaries, compresses one global count stream, one global body stream, and one charged framing stream across the corpus, then carries the charged payload through torch module state.

key metrics:

- aggregate standard payload bits: `849752`
- prior shared-dictionary corpus payload bits: `803400`
- prior source-subtoken corpus payload bits: `812688`
- global raw standard payload bits: `736504`
- aggregate selected payload bits: `699144`
- aggregate payload improvement over standard: `0.1772375940274339`
- margin over global raw standard baseline: `37360` bits
- aggregate payload margin over prior shared-dictionary corpus: `104256` bits
- aggregate payload margin over prior source-subtoken corpus: `113544` bits
- shared token count: `256`
- one-byte token count: `120`
- local token count per block: `0`
- shared dictionary payload bits: `10400`
- global count payload bits: `21272`
- global body payload bits: `665112`
- global length payload bits: `312`
- global header bits: `2048`
- model-state codec payload used: `1.0`
- state-dict reload reconstruction success: `1.0`
- state-dict raw source block retained: `0.0`
- margin over charged zstd trained-dictionary baseline: `283696` bits
- margin over undercharged zstd trained-dictionary diagnostic: `250848` bits
- random-label payload improvement: `-0.0003177134598372809`
- controls collapse: `1.0`

why limited:

this is a stronger source-code corpus byte codec with module-state reload proof, not arbitrary unknown-structure knowledge compression and not a qa/static-retrieval breakthrough. it does not beat the same-subtoken qa content scanner because it does not change that surface.

constraint left:

use this as the current source-code corpus codec baseline. the next qa-facing attempt must either beat the fair same-interface scanner with a real operation or move to learned semantic handles without parser mismatch.

## source-subtoken disjoint retrieval codec

files:

- `neuroloc/simulations/memory/local_100k_source_subtoken_disjoint_retrieval_codec.py`
- `tests/test_local_100k_source_subtoken_disjoint_retrieval_codec.py`
- `neuroloc/wiki/tests/local_100k_source_subtoken_disjoint_retrieval_codec.md`

why tried:

the global-stream corpus product beat raw corpus compression, but it was only a reconstruction codec. the next narrow positive step turns that payload into an exact retrieval surface over disjoint frozen source blocks, with raw content-scan and undercharged mph diagnostics measured on the same aligned 32-byte chunk retrieval task.

key metrics:

- block count: `3`
- retrieval fact count: `14715`
- chunk bytes: `32`
- selected retrieval accounted bits: `431536`
- standard retrieval accounted bits: `473008`
- raw content-scan accounted bits: `451128`
- undercharged mph accounted bits: `451144`
- margin over standard retrieval bits: `41472`
- margin over raw content scan bits: `19592`
- margin over undercharged mph bits: `19608`
- strict multiplier: `55.86800637721998`
- source train-test path overlap count: `0.0`
- source train-test hash overlap count: `0.0`
- state-dict reload chunk retrieval success: `1.0`
- random-label payload incompressible: `1.0`
- controls collapse: `1.0`

why limited:

this is a source-code chunk retrieval codec with disjoint-source proof and module-state reload proof. it is not learned semantic qa, not arbitrary unknown-structure knowledge compression, not implicit base-weight storage, and not broad breakthrough authorization. it does not solve the bounded qa adapter's fair same-subtoken content-scan blocker.

constraint left:

use this as the current narrow source-code exact-retrieval baseline. the next higher-value attempt should follow the review recommendation: source-authored relation qa with explicit read-work accounting, while keeping fair same-interface scanner and mph diagnostics in the report.

## source-authored relation diagnostic

files:

- `neuroloc/simulations/memory/local_100k_source_authored_relation_diagnostic.py`
- `tests/test_local_100k_source_authored_relation_diagnostic.py`
- `neuroloc/wiki/tests/local_100k_source_authored_relation_diagnostic.md`

why tried:

after the fixed-stride relation mistake, the next safe branch had to use relations literally authored in the source. this diagnostic extracts unique python definition signatures and unique import statements from the disjoint frozen source blocks after `state_dict` reload, then compares against fair scanner and mph baselines.

key metrics:

- relation fact count: `337`
- definition relation count: `303`
- import relation count: `34`
- exact relation answer success: `1.0`
- selected relation accounted bits: `437680`
- raw relation content-scan bits: `457272`
- undercharged relation mph bits: `457288`
- honest mph relation index bits: `305016`
- margin over raw relation content scan bits: `19592`
- margin over undercharged relation mph bits: `19608`
- margin over honest mph relation index bits: `-132664`
- relation-aware unlimited scanner success: `1.0`
- read-work gain over unlimited scan: `337.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`

why limited:

the diagnostic avoids generated aliases and fixed strides, but it still loses to an honest relation mph/index and does not beat a fair unlimited relation-aware scanner. relation-product authorization and static breakthrough authorization stay `0.0`.

constraint left:

do not promote authored relation qa unless it beats the honest index baseline or proves a stricter read-work claim where the fair scanner is given the same parser and accounting. the next relation surface needs harder authored links than definition/import line lookup.

## dense source-authored relation diagnostic

files:

- `neuroloc/simulations/memory/local_100k_source_dense_authored_relation_diagnostic.py`
- `tests/test_local_100k_source_dense_authored_relation_diagnostic.py`
- `neuroloc/wiki/tests/local_100k_source_dense_authored_relation_diagnostic.md`
- `neuroloc/wiki/mistakes/dense_relation_signature_query_leakage.md`

why tried:

the sparse authored relation diagnostic lost to an honest relation mph/index. the next attempt tested whether the same charged compressed source payload could amortize many more authored source relations: definition-parent links, statement-to-enclosing-signature links, and control-statement-to-enclosing-signature links.

key metrics:

- relation fact count: `3741`
- definition parent relation count: `314`
- definition signature relation count: `0`
- statement enclosing relation count: `2808`
- control statement enclosing relation count: `619`
- exact relation answer success: `1.0`
- selected relation accounted bits: `437680`
- honest mph relation index bits: `3366080`
- margin over honest mph relation index bits: `2928400`
- paq8px level 2 relation recomputed payload bits: `252952`
- paq8px level 2 relation accounted bits: `261144`
- paq8px relation recomputed matches constant: `1.0`
- margin over paq8px level 2 relation bits: `-176536`
- useful retrievable bits: `2639616`
- strict multiplier: `38.59793090842625`
- random-label twin success: `0.0`
- shuffled-value success: `0.0`
- relation-decoder-disabled success: `0.0`
- wrong-query hit rate: `0.0`
- controls collapse: `1.0`

failure repaired:

the first version counted a `definition_signature` relation whose question contained the exact answer. [[mistakes/dense_relation_signature_query_leakage]] records the issue. the relation was removed before keeping the diagnostic metrics.

why limited:

the repaired diagnostic beats honest relation mph/index by a large margin, but paq8px level 2 is still cheaper and the fair unlimited relation-aware scanner solves the same task. product authorization and breakthrough authorization stay `0.0`.

constraint left:

future relation surfaces need an answer-not-in-query guard and a public-compressor comparison before any product wording. a relation result can be useful only if it beats honest indexing and either beats paq8px pressure or proves an operation where raw-source reconstruction plus scanner is not the fair comparator.

## source relation mph codec

files:

- `neuroloc/simulations/memory/local_100k_source_relation_mph_codec.py`
- `tests/test_local_100k_source_relation_mph_codec.py`
- `neuroloc/wiki/tests/local_100k_source_relation_mph_codec.md`

why tried:

the dense authored relation diagnostic proved that the `3741` repaired relation surface was useful but still lost to paq8px. the next simple relation path stopped reconstructing source and stored only a charged exact keyed relation index: a minimal-perfect-hash-style router, fingerprints, compressed value/provenance id streams, and compressed dictionaries inside torch module state.

key metrics:

- relation fact count: `3741`
- exact relation answer success: `1.0`
- selected relation accounted bits: `248784`
- paq8px level 2 relation accounted bits: `261144`
- margin over paq8px level 2 relation bits: `12360`
- raw-source paq content-scan bits: `413888`
- margin over raw-source paq content scan bits: `165104`
- undercharged mph relation bits: `2771261`
- honest mph relation index bits: `3366080`
- margin over undercharged mph relation bits: `2522477`
- margin over honest mph relation index bits: `3117296`
- useful retrievable bits: `2639616`
- strict multiplier: `67.90445687825584`
- cross-scored random-label twin success: `0.0`
- rebuilt random-label exact success: `1.0`
- rebuilt random-label selected relation accounted bits: `934984`
- rebuilt random-label density-control collapse: `1.0`
- decoder-disabled success: `0.0`
- shuffled-fingerprint success: `0.0`
- wrong-query variant count: `7482`
- wrong-query hit rate: `0.0`
- state-dict exact reload answer success: `1.0`
- stored question substring hit count: `0.0`
- raw source block substring hit count: `0.0`
- source relation index product candidate: `1.0`
- source relation static breakthrough candidate: `0.0`
- self-contained paq8px baseline-win authorization: `1.0`

why limited:

this is a narrow exact keyed relation-index product, not broad high-density knowledge compression. it uses a classical keyed router plus compressed relation payloads, not learned semantic memory or implicit base-weight storage. the paq8px relation baseline is recomputed inside the simulation; the wider raw-source content-scan baseline remains imported from the paq8px audit record.

constraint left:

future promotion needs either a broader independently reproduced public-baseline artifact, a larger relation corpus, or a learned operation where a raw-source scan is not the fair comparator. do not call this static breakthrough, 600x, chat, full nm, or broad knowledge compression.

## rejected all-five source relation broadening probe

files:

- no committed simulation file

why tried:

after the three-block source relation mph codec produced a self-contained public-baseline win, the next obvious strengthening move was to use all five frozen source blocks from the global-stream corpus surface.

key metrics:

- block count: `5`
- joined source bytes: `802589`
- relation fact count: `5906`
- useful retrievable bits: `4222128`
- selected relation accounted bits: `401912`
- recomputed paq8px level 2 accounted bits: `413888`
- margin over recomputed paq8px relation bits: `11976`
- strict multiplier: `67.23267580962997`
- source train/test path overlap count: `6`
- source train/test hash overlap count: `6`
- product authorization: `0.0`

why rejected:

the apparent broader paq8px win is invalid for product promotion because the added frozen blocks overlap the source-training set used by the current source-code compression stack. the clean hard relation product therefore stays on the non-overlap `(0, 3, 4)` block subset.

constraint left:

broader relation products need a new frozen manifest with zero train path/hash overlap, not a larger count obtained by adding already-used source blocks.

## paq8px public baseline scratch audit

files:

- no committed simulation file
- `codex_local_output/compression_tools/paq8px_v214/paq8px.exe`
- `neuroloc/wiki/mistakes/public_context_mixing_baseline_missing.md`

why tried:

the zstd trained-dictionary audit was not enough to establish the strongest public source-code compression pressure line. a current public context-mixing compressor had to be checked before any competitor-beating source-code byte-compression claim could be trusted.

key metrics:

- tool: `paq8px` v214 levels 1 and 2
- hard raw joined source bytes: `802589`
- hard raw joined level 1 archive bytes: `51889`
- hard raw joined level 2 archive bytes: `50712`
- hard raw joined level 2 payload bits: `405696`
- transformed body stream bytes: `472621`
- transformed body level 1 archive bytes: `52665`
- current global-stream corpus selected payload bits: `699144`
- current disjoint retrieval accounted bits: `431536`

why limited:

this is a public compressor audit, not a neuroloc product. it does not provide learned semantic retrieval, query answering, or a neural memory mechanism. it does, however, beat the current source-code byte-compression payload by a large margin.

constraint left:

source-code byte-compression promotion now requires beating the paq8px pressure line under matched payload and operation accounting, or explicitly staying a local diagnostic. zstd trained dictionaries are no longer the strongest public compressor pressure line.

## rejected context-entropy body probe

files:

- no committed simulation file

why tried:

an independent review identified the current global-stream body payload as the bottleneck: `83139` bytes, or `665112` bits, out of the `699144` selected corpus payload bits. the shortest plausible path was to keep the current count/body/dictionary/framing front half and replace only the brotli-compressed substituted body stream with an adaptive context model.

key metrics:

- current global-stream body payload bits: `665112`
- current global-stream selected payload bits: `699144`
- best simple ppm estimate: `801051.0` body bits
- best simple ppm projected selected payload bits: `835083`
- best simple ppm gap versus current selected payload: `-135939`
- byte high-low split projected selected payload bits: `967120`
- line-length separated body projected selected payload bits: `778864`
- token-count sweep best setting remained the existing `256` shared tokens and `120` one-byte tokens at `699144` bits
- generic n-gram phrase dictionary best tested selected payload bits: `723976`

why rejected:

none of the fast probes beat the current global-stream body payload, and all are far above the new paq8px public pressure line. the simple adaptive model was especially weak because sparse higher-order byte contexts had poor escape efficiency on the substituted source body.

constraint left:

do not register a context-entropy body codec unless a probe first beats the current `83139` byte body payload. a serious future attempt needs a real range coder with stronger source-conditioned mixing or a different source transform; simple ppm, byte-class splitting, line-length separation, and frequency-ranked n-gram phrase substitution are exhausted on this surface.

## rejected paq transform probes

files:

- no committed simulation file
- `codex_local_output/paq_transform_probe_worker/`
- `codex_local_output/paq_transform_probe_main2/`
- `codex_local_output/line_grammar_paq_probe/`
- `codex_local_output/bpe_grammar_probe/`

why tried:

after paq8px became the public pressure line, the next possible rescue was a reversible source-aware transform whose streams compressed better under paq than raw source. the probes tested whether paq was merely missing easy structure such as indentation, identifiers, line dictionaries, or repeated byte-pair grammar.

key metrics:

- raw paq8px level 2 target: `50712` bytes, or `405696` bits.
- worker split-indent transform: `63891` bytes.
- worker raw token-id transform: `63272` bytes.
- worker case-normalized body transform: `69283` bytes.
- worker token-id body with indentation transform: `77395` bytes.
- worker class-normalized body transform: `86694` bytes.
- main level-1 line split transform: `53385` bytes, versus raw level-1 `51885` bytes in that probe.
- main level-1 identifier dictionary transform: `73374` bytes.
- main level-1 word-number dictionary transform: `73664` bytes.
- main level-1 keyword-identifier dictionary transform: `75258` bytes.
- main level-1 best subtoken transform: `55223` bytes.
- line dictionary level-2 transform: `74488` bytes.
- stripped frequent-line best level-2 transform: `65628` bytes.
- regex skeleton line transform: `82742` bytes.
- byte-pair grammar, 512 merges, paq level 1: `67915` bytes.
- byte-pair grammar, 1024 merges, paq level 1: `71033` bytes.

why rejected:

all exact-reconstructable transforms measured above the corresponding raw paq line. the file-deduplication probe also exposed a reconstruction mismatch when newline join separators were not modeled, so it was rejected rather than counted.

constraint left:

do not retry shallow source preprocessing against paq8px. the exhausted set now includes indentation split, identifier dictionaries, keyword dictionaries, line dictionaries, regex skeletons, case normalization, class normalization, simple token ids, source-subtoken substitution before paq, and naive byte-pair grammar. the next source-byte attempt needs a genuinely stronger entropy model or a different benchmark operation where raw-source reconstruction plus scanner is not the fair comparator.

## rejected concrete-token gap probe

files:

- no committed simulation file

why tried:

an independent review suggested that a concrete syntax or token-gap split might give a more defensible source-code structural codec than identifier-subtoken substitution alone.

key metrics:

- token/gap split total with `1024` bit header: `771280` bits
- token/gap split total with `2048` bit header: `772304` bits
- current global-stream corpus product: `699144` bits
- tokenizer error count: `1`

why rejected:

the split loses by at least `68416` bits to the current global-stream codec before a product-grade exact restore implementation exists. it is not worth promoting into the suite.

constraint left:

source-code syntax splitting must beat the global count/body/framing stream before it becomes a product candidate.

## rejected current-source token and line grammar probe

files:

- no committed simulation file
- `codex_local_output/line_grammar_paq_probe/`
- `codex_local_output/relation_index_probe/`

why tried:

after the relation mph codec became the first narrow positive relation-index product, the next source-byte attempt checked whether current-source token gaps, class splits, or line dictionaries could beat the paq8px raw-source pressure line or expose a cheaper exact relation surface.

key metrics:

- hard raw source bytes: `802585`
- best standard raw-source payload: `736288` bits
- paq8px raw-source line: `405696` bits
- token-gap total with `2048` bit header: `1142304` bits
- token-gap margin versus paq8px: `-736608` bits
- class-split total with `2048` bit header: `1078480` bits
- class-split margin versus paq8px: `-672784` bits
- best line-dictionary total: `832752` bits
- best line-dictionary margin versus paq8px: `-427056` bits
- no-key relation tsv paq archive bytes: `11331`
- no-key relation id paq archive bytes: `14280`

why rejected:

all reversible source grammar/token transforms lost badly to paq8px. the no-key relation ordinal surfaces compressed very well, but they are invalid as keyed knowledge products because the query-address burden is removed; paq also beats those no-key surfaces directly.

constraint left:

do not promote grammar, token-gap, class-split, line-dictionary, or no-key ordinal relation storage without a keyed operation and a direct public-compressor comparison. the positive relation path remains the charged keyed mph relation codec, not no-key ordinal table compression.

## accepted bounded model-state knowledge pack

files:

- `neuroloc/simulations/memory/local_100k_model_state_knowledge_pack.py`
- `tests/test_local_100k_model_state_knowledge_pack.py`
- `neuroloc/wiki/tests/local_100k_model_state_knowledge_pack.md`
- `neuroloc/compression/paper/model_state_knowledge_pack_paper.md`

why tried:

the source-only external relation adapter was correct but still too narrow for a paper-facing claim. the next product needed public multi-surface facts, charged deployment-shaped baselines, and a real adapter lifecycle.

key metrics:

- public cpython `v3.12.3` surfaces: `15`.
- public bytes: `1323186`.
- exact relation facts: `9754`.
- selected relation bits: `679400`.
- model package bits: `687592`.
- useful retrievable bits: `3992464`.
- strict multiplier: `37.6093164556962x`.
- paq8px level 2 source-scan bits: `1385264`.
- margin over paq8px: `705864`.
- zstd level 19 source-scan bits: `2195016`.
- margin over zstd: `1515616`.
- honest mph margin: `5194024`.
- random-label twin success: `0.0`.
- random-label rebuild bits: `2429560`.
- export, update, rollback, and host reload success: `1.0`.

why accepted:

it satisfies the bounded exact relation knowledge-pack gate: public multi-surface corpus, exact answers and provenance, charged module-state payload, no raw public surface retention, no full question table, paq8px and zstd pressure, storage baselines, random-label collapse, false-hit controls, adapter export, update recompression, rollback, and three host-family reload probes.

constraint left:

this is paper-facing within a bounded exact relation claim only. it is not implicit base-weight learning, broad open-domain knowledge compression, arbitrary chat, full nm completion, strict 600x proof, or a native benchmark win over lora, qlora, rome, memit, rag, retro, product-key memory, or memory layers. the next external-strengthening step is a trained bounded host router, a second public repository family, or a real lora/model-edit training comparison on the same relation set.

## older architecture compression lessons

the old architecture explored quantization, low-rank cache compression, correction-field memory, split retention, matrix memory, slot memory, and compressed attention. the most important lessons are:

- compression improved byte prediction more easily than exact retrieval.
- correction-field memory helped reconstruction-side prediction but did not increase memory capacity.
- matrix-memory direct accumulation had a structural capacity wall.
- slot memory restored a better retrieval substrate in isolation but paid runs still ended at zero passkey under the tested training setup.
- compressed/statistical paths must be evaluated against exact retrieval tasks, not only validation loss.

primary older files:

- `neuroloc/wiki/synthesis/compression_beyond_quantization.md`
- `neuroloc/wiki/synthesis/neural_model_compression_stack.md`
- `neuroloc/wiki/synthesis/linear_attention_retrieval_wall.md`
- `neuroloc/wiki/synthesis/slot_memory_design.md`
- `neuroloc/wiki/knowledge/mla_compression.md`
- `neuroloc/wiki/knowledge/ternary_compression_research.md`
- `docs/STATUS_BOARD.md`
- `state/program_status.yaml`

## current rule

the current local full nm result is real within its synthetic scope. the current bounded qa adapter product is `local_100k_source_subtoken_qa_adapter`, the current narrow source-code byte-compression product is `local_100k_source_subtoken_structure_block_codec`, the current broader source-code corpus codec product is `local_100k_source_subtoken_global_stream_corpus_codec`, the current narrow source-authored relation-index product is `local_100k_source_relation_mph_codec`, and the current broader paper-facing bounded knowledge-pack product is `local_100k_model_state_knowledge_pack`. `local_100k_zstd_trained_dictionary_baseline_audit` strengthens the byte-codec claim against trained public dictionaries but does not change the qa scanner blocker; paq8px remains the stronger public pressure line for source bytes. the high-density knowledge-compression result is not solved. the next attempt must train a bounded host router, add a second public repository family, or attempt the same relation set through a real lora/model-edit path under charged accounting.
