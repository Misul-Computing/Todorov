# local 100k full nm soft code false pass

status: historical context only. frozen as of 2026-05-09. do not edit.

## summary

during implementation of [[tests/local_100k_full_nm]], the first single-model version learned with soft continuous latent codes but failed when evaluated through hard binary compressed codes. that would have been a false compression pass.

## what happened

the model trained with soft latent probabilities and low training loss. when evaluation used thresholded bits, held-out world-state success collapsed. a follow-up straight-through hard-code version also memorized the train split before the bottleneck was given explicit binary-code supervision.

## why it was wrong

continuous probabilities are not the same as an accounted compressed memory code. claiming compression from the soft path would have repeated the project category error: a surface looked like the desired mechanism but did not actually satisfy the gate when the accounted code was enforced.

## fix

the accepted model trains and evaluates through a hard straight-through binary bottleneck. it also receives binary code supervision derived from legal observed world state, and all accepted operations are decoded from the recurrent state written from that hard code.

## prevention

future compression work must report both soft-training behavior and hard-code evaluation. a compression claim can pass only when the hard accounted code preserves the operation under controls.

## see also

- [[tests/local_100k_full_nm]]
- [[tests/local_100k_3d_nm_mirror]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
