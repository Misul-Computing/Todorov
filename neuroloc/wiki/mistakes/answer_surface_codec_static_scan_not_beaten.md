# answer-surface codec static scan not beaten

status: current

## mistake

`local_100k_answer_surface_codec` produced exact answers on a bounded answer surface but did not produce a publishable compression claim. the same-interface scan and minimal-perfect-hash diagnostics reached the same useful surface, so the result was another static retrieval diagnostic rather than a new compression mechanism.

## why it matters

the project has repeatedly failed when answer routing, token signatures, or compact handles are mistaken for compression. exact retrieval is not enough when a scanner with the same interface can solve the task at equal or better useful density.

## correction

the result stays unregistered as a diagnostic. future answer-surface work must beat same-interface content scan and mph baselines after the scanner receives the same parser, handle surface, and payload access.

## evidence

- simulation: `neuroloc/simulations/memory/local_100k_answer_surface_codec.py`
- tests: `tests/test_local_100k_answer_surface_codec.py`
- observed hard metrics: exact success `1.0`, same-interface scan success `1.0`, undercharged mph multiplier matched the surface, publishable authorization `0.0`
