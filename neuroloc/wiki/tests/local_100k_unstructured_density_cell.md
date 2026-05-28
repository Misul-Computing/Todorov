# local 100k unstructured density cell

status: current (as of 2026-05-09).

## date run

2026-05-09.

## status

passed as a negative entropy-bound probe. no high-density knowledge-compression breakthrough is authorized.

## artifact tested

- `neuroloc/simulations/memory/local_100k_unstructured_density_cell.py`
- `tests/test_local_100k_unstructured_density_cell.py`
- `neuroloc/simulations/suite_registry.py`
- hard artifact: `codex_local_output/suite_l100k_unstructured_density_cell_hard/local_100k_unstructured_density_cell/local_100k_unstructured_density_cell_metrics.json`

## what was done

the simulation generates exact unstructured key-value-provenance facts with random labels. it forbids seed-oracle recovery, formula labels, schema-generated labels, per-fact committed rows, and table-backed values. the cell is a compressed sketch under the 600x strict budget. the gate is expected to fail exact retrieval and prove the entropy wall rather than overclaim success.

## key hard outputs

- fact count: `4096`
- useful retrievable bits: `122880`
- committed state bits under 600x budget: `1218`
- target state budget bits: `1246.72`
- entropy lower bound bits: `122880`
- entropy budget gap bits: `121633.28`
- entropy gap multiplier: `98.56262833675565`
- exact retrieval success: `0.0`
- strict density of attempted sketch: `1533.603744149766`
- strict multiplier versus 2.5 bits per parameter: `613.4414976599064`
- information-theoretic 600x possible: `0.0`
- strict 600x pass: `0.0`
- useful negative result: `1.0`
- strict breakthrough authorized: `0.0`
- general independent-fact breakthrough authorized: `0.0`
- controls collapse: `1.0`

## verification commands

- `python -m py_compile neuroloc\simulations\memory\local_100k_unstructured_density_cell.py tests\test_local_100k_unstructured_density_cell.py neuroloc\simulations\suite_registry.py` passed.
- `python -m pytest tests/test_local_100k_unstructured_density_cell.py tests/test_simulation_suite.py::test_suite_registry_contract -q` passed: 7 passed, 1 warning.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_unstructured_density_cell --profile smoke --timeout-sec 300` passed.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_unstructured_density_cell --profile hard --output-root codex_local_output\suite_l100k_unstructured_density_cell_hard --timeout-sec 1200` passed.

## category check

implemented operation: compressed-sketch exact-retrieval attempt over unstructured random-label facts.

strongest baseline: entropy lower bound, verbatim table, product-key-style lookup, content-routed sparse read, no-memory, recency-only, shuffled-key, shuffled-value, shuffled-provenance, write-disabled, read-disabled, and decoder-disabled controls.

what passed: the gate cleanly shows the 600x state budget is far below the entropy lower bound for exact independent labels. no formula labels, schema labels, seed oracle, or per-fact rows are authorized.

what failed: exact retrieval is `0.0`, and strict 600x pass is `0.0`.

what is not proved: high-density knowledge compression, broad nm competence, arbitrary chat, paid-scale trainability, or external simulator transfer.

why this is useful: it prevents the project from chasing or reporting an impossible exact random-label compression claim. any future success must exploit learned structure in unknown-structure data, not independent entropy and not hand-planned formulas.

## verdict

accepted as a negative boundary result. the high-density knowledge-compression target remains unsolved.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_high_density_cell]]
- [[tests/local_100k_schema_density_cell]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
