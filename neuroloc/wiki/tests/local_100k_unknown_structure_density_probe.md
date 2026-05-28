# local 100k unknown-structure density probe

status: current (as of 2026-05-09).

## date run

2026-05-09.

## status

passed as an unknown-structure corpus boundary probe. no 600x high-density knowledge-compression breakthrough is authorized.

## artifact tested

- `neuroloc/simulations/memory/local_100k_unknown_structure_density_probe.py`
- `tests/test_local_100k_unknown_structure_density_probe.py`
- `neuroloc/simulations/suite_registry.py`
- hard artifact: `codex_local_output/suite_l100k_unknown_structure_density_probe_hard/local_100k_unknown_structure_density_probe/local_100k_unknown_structure_density_probe_metrics.json`

## what was done

the simulation freezes a non-generated local corpus from current project wiki files, samples unique non-overlapping byte chunks as exact facts, and stores the corpus through a charged standard compressed representation. the key is an offset query, not a random associative key. the random-label twin keeps the same keys but replaces values and provenance with independent random bytes.

the probe charges compressed payload bits, decoder bits, manifest bits, and trainable control parameters. it explicitly reports that it is a sequence-offset corpus target with a standard codec dependency, not a general associative knowledge-compression breakthrough.

## key hard outputs

- fact count: `4096`
- corpus file count: `7`
- corpus bytes: `168591`
- compressed bytes: `50973`
- trainable parameter count: `4`
- committed state bits: `481282`
- decoder bits: `65536`
- manifest bits: `7944`
- query key bits: `73728`
- useful retrievable bits: `1048576`
- unique source bits: `1048576`
- compressed ratio after charged decoder and manifest: `2.8023653492131433`
- raw zlib payload ratio before decoder and manifest charge: `3.3074568889412044`
- strict density: `34.8547946799184`
- strict multiplier versus 2.5 bits per parameter: `13.941917871967359`
- target density: `1500.0`
- exact retrieval success: `1.0`
- random-label twin success: `0.0`
- controls collapse: `1.0`
- strict 600x pass: `0.0`
- strict breakthrough authorized: `0.0`
- general unknown-structure breakthrough authorized: `0.0`

## verification commands

- `python -m pytest tests/test_local_100k_unknown_structure_density_probe.py -q` passed: 6 passed, 1 warning.
- `python -m pytest tests/test_local_100k_unknown_structure_density_probe.py tests/test_simulation_suite.py::test_suite_registry_contract -q` passed: 7 passed, 1 warning.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_unknown_structure_density_probe --profile smoke --timeout-sec 300` passed.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_unknown_structure_density_probe --profile hard --output-root codex_local_output\suite_l100k_unknown_structure_density_probe_hard --timeout-sec 1200` passed.

## category check

implemented operation: exact retrieval of non-generated local corpus byte chunks from a charged compressed corpus state.

strongest baseline: random-label twin, no-memory, recency-only, shuffled-key, shuffled-value, shuffled-provenance, read-disabled, decoder-disabled, charged standard-codec accounting, and the prior unstructured entropy wall.

what passed: exact corpus retrieval is `1.0`, random-label twin success is `0.0`, and disabled or shuffled controls collapse.

what failed: strict density reaches only `34.8547946799184`, far below the `1500.0` density needed for 600x over the ordinary 2.5 bits-per-parameter bar.

what is not proved: high-density associative knowledge compression, broad nm competence, arbitrary chat, paid-scale trainability, external simulator transfer, learned compressor superiority, or a general breakthrough.

why this is useful: it establishes the first fair non-generated unknown-structure measurement after the structured-target error. real corpus structure is compressible, independent random labels are not, and the remaining target is a learned charged cell that beats standard corpus compression and associative lookup baselines without relying on offset keys.

## verdict

accepted as a boundary and baseline probe. the high-density knowledge-compression target remains unsolved.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_high_density_cell]]
- [[tests/local_100k_schema_density_cell]]
- [[tests/local_100k_unstructured_density_cell]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[mistakes/unstructured_exact_600x_entropy_wall]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
