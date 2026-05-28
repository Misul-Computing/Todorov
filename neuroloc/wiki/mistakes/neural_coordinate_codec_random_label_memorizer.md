# neural-coordinate codec random-label memorizer

status: current

## mistake

`local_100k_neural_coordinate_codec` tested a neural coordinate-style compression surface, but the hard profile failed the real target and the random-label twin remained high. this means the path behaved like a memorizer or coordinate table rather than a useful learned compression mechanism.

## why it matters

random-label controls are the cleanest guard against hidden table behavior. if a method preserves random labels as well as or better than real data, it has not learned useful structure.

## correction

the result stays unregistered as a diagnostic. future coordinate or latent-code attempts must require real-target success above `0.95`, random-label cost collapse, and a strict win over standard codecs and content-scan baselines before any promotion.

## evidence

- simulation: `neuroloc/simulations/memory/local_100k_neural_coordinate_codec.py`
- tests: `tests/test_local_100k_neural_coordinate_codec.py`
- observed metrics: smoke exact success `0.96875`, random-label twin exact success `0.984375`, hard exact success `0.1914`, hard random-label twin exact success about `0.914`, publishable authorization `0.0`
