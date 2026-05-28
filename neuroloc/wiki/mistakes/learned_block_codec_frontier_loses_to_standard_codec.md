# learned block-codec frontier loses to standard codec

status: current

## mistake

`local_100k_learned_block_codec_frontier` reconstructed its target exactly, but the learned phrase codec lost badly to the best standard codec baseline. exact reconstruction alone did not matter because the charged learned stream was much larger than the standard compressed payload.

## why it matters

a learned codec must beat strong ordinary codecs on the same bytes before it can support a compression breakthrough claim. otherwise it is only a reconstruction demo.

## correction

the result stays unregistered as a diagnostic. the accepted next path became the simpler indentation-token transform because it learned from disjoint train source files, charged the token map, and beat the best same-block standard codec under strict bits.

## evidence

- simulation: `neuroloc/simulations/memory/local_100k_learned_block_codec_frontier.py`
- tests: `tests/test_local_100k_learned_block_codec_frontier.py`
- observed metrics: exact reconstruction `1.0`, learned phrase codec bits `275003`, best standard codec bits `112876`, publishable authorization `0.0`
