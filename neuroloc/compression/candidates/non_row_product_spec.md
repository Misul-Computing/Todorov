# non-row compression product spec

status: current (as of 2026-05-12).

## product name

working name: shared-predictor exact codec cell.

this is not a promoted result. it is the first buildable candidate after the residual-row defeat.

## simple idea

do not store values in rows.

learn one shared predictor over the corpus family. for each held-out value, store only the arithmetic-coded surprise of that value under the predictor, plus the minimum key-to-code routing needed to decode it. if the predictor understands the data better than zstd or brotli, the arithmetic stream gets shorter. if it does not, the candidate dies cleanly.

that statement is a baseline path, not automatic promotion. one independently addressable coded slice per fact is still a payload table, even if the slice was produced by arithmetic coding. the promoted cell must either use a block-level shared stream with fully charged offsets and provenance and label that result as a codec baseline, or show sublinear/no-per-fact value payload while preserving exact retrieval.

```text
shared model learns p(value | context)
stored code is arithmetic_code(value under p)
decoder reconstructs value exactly
random labels collapse because p assigns them near-uniform probability
```

## why this is different from the failed residual table

the failed learned unknown-structure cell stored dictionary tokens plus per-fact residual bytes. random labels could be stored exactly by building the same residual rows.

this candidate must satisfy:

- no literal per-fact residual rows.
- no per-fact value payload outside the entropy-coded stream.
- no independently addressable entropy-coded value slice per fact for a promoted non-row cell.
- one shared decoder for all train and held-out values.
- block-level arithmetic streams are allowed only when their offsets, provenance, and key routing are charged and the result is reported as a codec baseline unless the per-fact payload disappears.
- random-label twin cannot pass unless it pays near-raw entropy.
- exception ledger has a hard cap and is reported separately.

## architecture

### shared predictor

small byte or phrase predictor trained on source files disjoint from test files.

possible implementations:

- byte-level n-gram plus neural probability correction.
- phrase graph with learned probabilities.
- discrete latent codec with entropy model.
- small recurrent or transformer predictor if local time allows.

### encoder

for each held-out fact:

- produce an opaque key.
- select only allowed context fields.
- encode the exact value under the shared predictor only through a charged block stream or through a proven sublinear shared code.
- record provenance through the same code path or a separate charged stream.

### decoder

given key and stored code:

- recover the code stream.
- decode exact bytes using the shared model.
- decode provenance.
- answer exact retrieval query.

### accounting

charge:

- model weights
- quantization metadata
- tokenizer or phrase graph
- entropy model
- arithmetic streams
- key map
- provenance stream
- manifest
- training-side information
- exception ledger
- checksums or validation hashes

## acceptance gates

minimum local hard gate:

- train/test key overlap: `0.0`
- source holdout: `1.0`
- exact held-out retrieval: at least `0.95`
- random-label twin exact retrieval: near `0.0` under the same budget
- strict multiplier above `13.941917871967359x`
- beats minimal perfect hash plus charged payload codec
- beats product-key memory under strict accounting
- beats sparse read under strict accounting
- no per-fact residual rows
- disabled predictor, disabled code, disabled decoder, shuffled key, shuffled value, shuffled provenance, wrong-model, and wrong-code controls collapse

target gate:

- strict useful bits per parameter-equivalent approaches the 600x bar only after all state and model bits are charged.

## kill conditions

kill the candidate if:

- exact retrieval requires a per-fact literal stream close to raw value length.
- random-label twin stores at high success.
- strict density does not beat the current corpus-codec baseline.
- the exception ledger grows with fact count.
- the key map becomes a minimal perfect hash plus payload table.
- the arithmetic coder produces one independently addressable value slice per fact and that path is used for promotion instead of being reported as a charged codec baseline.
- the predictor only works because train and test sources overlap.
- exactness depends on retries, checksums, or search windows that are not charged.

## first implementation sequence

1. freeze the existing unknown-structure corpus split.
2. add a classical baseline run: zstd or brotli if available, plus minimal-perfect-hash accounting.
3. implement a simple arithmetic coder around a byte probability model.
4. start with a deterministic n-gram predictor and charge the model table.
5. add a small learned probability correction only if the n-gram baseline is clean.
6. run exact held-out retrieval and random-label twin.
7. compare strict density against the existing `13.941917871967359x` baseline.
8. only then add neural latent code variants.

## upgrade path

if the simple predictor loses to zstd:

- try phrase graph induction with arithmetic-coded derivations.
- try discrete latent codec with exact reconstruction.
- try hdc/sdm only as routing plus candidate generation, not value storage.

if it beats zstd but stays far below 600x:

- keep it as a valid learned codec baseline.
- do not call it a breakthrough.
- use it as the foundation for operation-preserving compression where exact task variables are smaller than exact bytes.

## paper-worthy claim if proved

possible wording only after gates pass:

```text
a shared-predictor exact codec cell can store source-heldout unknown-structure associative facts with lower strict charged bits than standard codecs, sparse read, product-key memory, and verbatim storage, while random-label controls collapse.
```

that would be a real compression result. it would still not imply arbitrary chat, paid-scale trainability, biological neuron density, or broad nm completion.
