# local 100k high density cell

status: historical context only. frozen as of 2026-05-09. do not edit.

## date run

2026-05-09.

## status

passed as a local params-only high-density associative-cell candidate.

this is not a strict 600x compression breakthrough. it is not biological neuron-density proof, not arbitrary chat, not paid-scale trainability, not external simulator transfer, and not broad nm completion.

## artifact tested

- `neuroloc/simulations/memory/local_100k_high_density_cell.py`
- `tests/test_local_100k_high_density_cell.py`
- `neuroloc/simulations/suite_registry.py`
- hard artifact: `codex_local_output/suite_l100k_high_density_cell_hard/local_100k_high_density_cell/local_100k_high_density_cell_metrics.json`

## what was done

the simulation builds a bounded hybrid cell with three planes: a gated commit plane, a factorized associative address plane, and an exact decoder/read plane for values and provenance.

the knowledge unit is an independent exact associative fact. train and test keys are disjoint. the cell writes test facts at evaluation time and answers exact key queries. no target labels are passed through a chat surface or language template.

## key hard outputs

- fact count: `4096`
- train facts held apart from test facts: `128`
- trainable parameters: `8`
- useful retrievable bits: `114688`
- committed state bits: `118816`
- exact retrieval success: `1.0`
- params-only density: `14336.0`
- params-only multiplier versus 2.5 bits per parameter: `5734.4`
- params-only 600x pass: `1.0`
- strict density after committed-state accounting: `15.427495291902071`
- strict multiplier versus 2.5 bits per parameter: `6.1709981167608285`
- strict 600x pass: `0.0`
- strict breakthrough authorized: `0.0`
- strict density advantage over best strict baseline: `5.245677110083889`
- no-key-leakage: `1.0`
- controls collapse: `1.0`
- no-memory success: `0.0`
- recency-only success: `0.000244140625`
- shuffled-key success: `0.0`
- shuffled-value success: `0.0`
- shuffled-provenance success: `0.0`
- write-disabled success: `0.0`
- read-disabled success: `0.0`
- decoder-disabled success: `0.0`
- local partial candidate authorized: `1.0`
- claim downgraded to params-only: `1.0`

## verification commands

- `python -m pytest tests/test_local_100k_high_density_cell.py -q` passed: 6 passed, 1 warning.
- `python -m pytest tests/test_local_100k_high_density_cell.py tests/test_simulation_suite.py::test_suite_registry_contract -q` passed: 7 passed, 1 warning.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_high_density_cell --profile smoke --timeout-sec 300` passed.
- `python neuroloc\simulations\suite_runner.py --simulation local_100k_high_density_cell --profile hard --output-root codex_local_output\suite_l100k_high_density_cell_hard --timeout-sec 1200` passed.
- `python -m py_compile neuroloc\simulations\memory\local_100k_high_density_cell.py tests\test_local_100k_high_density_cell.py neuroloc\simulations\suite_registry.py` passed.
- `git diff --check` passed with line-ending warnings only.

## category check

implemented operation: exact associative write/read through a bounded hybrid cell with factorized addressing and value/provenance decoding.

strongest baseline: verbatim table, product-key-style lookup, content-routed sparse read, mini titans/miras-style bounded test-time memory, scalar mlp capacity, no-memory, recency-only, shuffled-key, shuffled-value, shuffled-provenance, write-disabled, read-disabled, and decoder-disabled controls.

what failed: strict 600x density did not pass after committed state was charged. see [[mistakes/local_100k_high_density_cell_strict_600x_not_met]].

what is not proved: strict 600x compression, biological neuron-density storage, unsupervised world-code discovery, broad neural-model competence, arbitrary chat, paid-scale trainability, or external simulator transfer.

why this can be accepted only as a params-only candidate: exact retrieval passes and params-only density clears the 600x target, but exact independent facts still require committed state bits. the strict density line is only about `6.17x` the ordinary 2.5 bits-per-parameter bar, not 600x.

## verdict

accepted as a useful pressure-test artifact and partial high-density cell candidate. it does not supersede [[tests/local_100k_full_nm]] as the current top local nm result. it opens the next compression roadblock: reducing committed state cost without losing exact operation preservation.

## see also

- [[PROJECT_PLAN]]
- [[synthesis/high_density_neuron_cell_related_work_pressure_matrix]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
- [[tests/local_100k_full_nm]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
