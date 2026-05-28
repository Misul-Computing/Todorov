# unstructured exact 600x entropy wall

status: current (as of 2026-05-09).

## summary

the unstructured-density cell confirms that exact independent random-label facts cannot be compressed by 600x under honest strict accounting.

## what happened

`local_100k_unstructured_density_cell` tested exact retrieval over unstructured random-label key-value-provenance facts. the hard profile had `122880` useful retrievable bits, while the 600x strict state budget allowed only about `1246.72` state bits before the density target broke.

the entropy lower bound exceeds the budget by `121633.28` bits, or a factor of `98.56262833675565`. exact retrieval success was `0.0`, and both strict breakthrough authorization flags remained `0.0`.

## why it matters

this is not an implementation bug. exact independent labels require their information to exist somewhere. without a seed oracle, schema generator, hidden table, or lossy output, the requested 600x exact random-label storage target violates the entropy lower bound.

## prevention

future work must target unknown-structure data where learnable redundancy may exist, not independent random labels and not hand-designed formulas. every future claim must keep the random-label twin as a negative control and charge all discovered structure, residuals, codebooks, decoder state, examples, seeds, and side channels.

## see also

- [[tests/local_100k_unstructured_density_cell]]
- [[tests/local_100k_unknown_structure_density_probe]]
- [[tests/local_100k_high_density_cell]]
- [[tests/local_100k_schema_density_cell]]
- [[mistakes/local_100k_high_density_cell_strict_600x_not_met]]
- [[mistakes/schema_density_cell_structured_target_category_error]]
- [[PROJECT_PLAN]]
