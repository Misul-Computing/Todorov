# theory limits for exact knowledge compression

status: current (as of 2026-05-12).

## bottom line

for exact associative knowledge compression, the basic bound is:

```text
total_charged_bits >= k(data | fixed_public_decoder) - allowed_error_savings
```

for practical accounting:

```text
l_total >= l_model + l_index + l_keys + l_decoder + l_dictionary + l_residuals + l_manifest + l_training_side_info
```

for exact independent random labels:

```text
h(v_1...v_n | k_1...k_n) = sum_i h(v_i)
```

the bits can move into a dictionary, model, seed, residual table, hash assignment, latent code, provenance record, or decoder. they cannot disappear.

## shannon boundary

source: [shannon 1948](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf).

expected lossless code length cannot beat source entropy except by exploiting source structure. arithmetic coding is the practical form: if a model assigns probability `p(x)`, the ideal code length is about `-log2 p(x)` bits.

implication:

better prediction gives better compression. coding does not create compression without prediction.

## mdl boundary

the honest target is:

```text
min_model l(model) + l(data | model)
```

if the model stores a giant decoder, tokenizer, dictionary, seed, example set, or residual table, mdl charges it. exact kolmogorov complexity is not computable in practice, so the repo uses charged compressors, ablations, random-label controls, and bit ledgers as upper-bound evidence.

## classical codecs

sources: [ziv-lempel 1977](https://nemenmanlab.org/~ilya/images/e/e9/Ziv-lempel-77.pdf), [arithmetic coding](https://web.stanford.edu/class/ee398a/handouts/papers/WittenACM87ArithmCoding.pdf), [rfc 8878 zstandard](https://www.rfc-editor.org/rfc/rfc8878.html), [rfc 7932 brotli](https://www.rfc-editor.org/rfc/rfc7932).

lz-style systems reuse repeated substrings. zstd and brotli add block structure, entropy coding, and dictionary machinery. if neuroloc uses an out-of-band dictionary, it must be charged. if a learned method does not beat charged zstd or brotli on the same held-out values, it is not a breakthrough.

## perfect hashing

sources: [fks perfect hashing](https://www.cs.dartmouth.edu/~ac/Teach/CS105-Winter05/Handouts/fks-perfecthash.pdf), [recsplit](https://arxiv.org/abs/1910.06416), [pthash](https://github.com/jermp/pthash).

a minimal perfect hash maps known keys to dense indices. it stores routing, not values. even near the lower bound of roughly `log2(e)` bits per key, the payload still has to be stored.

implication:

minimal perfect hashing plus a compressed payload is a mandatory exact retrieval baseline and a common hidden-table trap.

## bloom and bloomier filters

sources: [bloom 1970](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p422-bloom.pdf), [bloomier filters](https://www.cs.princeton.edu/~chazelle/pubs/soda04.pdf), [quotient filters](https://www.fsl.cs.stonybrook.edu/docs/vldb12qfflash/vldb12qfflash.pdf).

bloom filters answer approximate membership. bloomier filters approximate static functions. quotient filters improve locality and related operations.

implication:

these are excellent routing and membership baselines. they do not prove exact arbitrary key-value recovery unless errors are allowed or the values are stored elsewhere.

## error-correcting codes

source: [hamming 1950](https://bioinfo.uib.es/~joemiro/comptrans/HammingIntro.pdf).

error-correcting codes spend redundancy to survive noise. they protect information; they do not compress independent exact labels.

implication:

ecc may make a compact neural state robust, but its parity bits must be charged.

## associative memories and distributed representations

sources: [kanerva sparse distributed memory](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19890017031.pdf), [hopfield 1982](https://authors.library.caltech.edu/records/w41x7-8bn13), [torchhd](https://www.jmlr.org/papers/v24/23-0300.html), [vsa capacity](https://openreview.net/forum?id=FIW9uXiCkI).

associative memories, cam, sdm, vsa, and hdc are useful for partial cues, noise tolerance, distributed routing, and graceful degradation. superposition creates interference. cleanup dictionaries and item memories must be charged.

implication:

if exact many-item recovery succeeds on random labels, the system is storing a table in another form.

## product quantization and learned indexes

source: [learned indexes](https://research.google/pubs/the-case-for-learned-index-structures/).

product quantization compresses vectors for approximate similarity search. learned indexes exploit key distribution. neither stores exact values for free.

implication:

these help routing and approximate search. opaque random keys destroy most learned-index advantage.

## random-label twin rule

the random-label twin is the main lie detector:

- keep the same keys, query protocol, value length, provenance schema, and accounting.
- replace values with independent random bytes.
- rebuild the candidate under the same rules.

interpretation:

- if exact retrieval still passes, there is a row store, residual store, seed oracle, or hidden side channel.
- if it collapses, the original success may exploit real structure.
- if it beats random labels but loses to zstd or brotli, it is a learned compressor but not a breakthrough.
- if it beats codecs and random labels fail, there is a real candidate.

## must charge list

charge all of the following:

- model weights
- quantization scale metadata
- decoder code and decoder parameters
- tokenizer, parser, schema, and grammar
- learned dictionary and static dictionary
- retained training corpus or examples
- residual payloads and residual indices
- key fingerprints and full query keys when stored
- minimal perfect hash tables and seeds
- manifests and chunk assignments
- provenance
- codebooks and latent priors
- error-correcting parity
- fallback records and exception ledgers
- calibration windows and search windows
- any generated relation rule

## hidden lookup table tests

before accepting a positive result:

- run a random-label twin.
- shuffle keys, values, and provenance.
- remove residual stream, dictionary, decoder, and assignment table one at a time.
- inspect whether storage scales as `o(n)` or as one row per fact.
- count per-fact bits directly.
- test adversarial incompressible labels.
- test key permutation invariance.
- require the same decoder for train and held-out facts.
- compare to zstd, brotli, minimal-perfect-hash plus payload, sparse read, product-key memory, and verbatim storage.
- report whether random labels can be stored by rebuilding the mechanism.

## simple remaining possibility

the only clean exact path is a predictor plus entropy coder:

```text
learn p(value | shared_model, key_context)
store arithmetic_code(value under p)
decode exactly with the same model
```

this can beat standard codecs only if the learned model captures structure they miss. it cannot beat entropy on independent labels.

