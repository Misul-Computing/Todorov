# open-source and passion-project baselines

status: current (as of 2026-05-14).

## role

this file records open-source and informal alternatives that should pressure the next compression attempt. none of them is accepted as a solved neuroloc high-density compressor. they are baselines, controls, or idea sources.

## exact indexing and payload baselines

### minimal perfect hash plus charged payload

sources: [pthash](https://github.com/jermp/pthash), [recsplit](https://arxiv.org/abs/1910.06416), [bbhash go](https://github.com/relab/bbhash).

fit:

excellent exact-key routing baseline.

risk:

payload storage is still a table. this is the clean baseline for detecting whether a learned method is only reimplementing row lookup.

local use:

build a minimal-perfect-hash style accounting baseline over opaque keys, then store values through zstd, brotli, or the current charged corpus codec. report bits per key and value bits separately.

## learned sparse memory baselines

### lucidrains product-key-memory

source: [product-key-memory](https://github.com/lucidrains/product-key-memory).

fit:

good learned-address baseline. weak as a breakthrough because value embeddings are the memory.

local use:

create a product-key baseline over corpus chunk ids or latent value codes. charge subkeys, value table, query network, and all parameters.

### facebookresearch memory

source: [facebookresearch memory](https://github.com/facebookresearch/memory).

fit:

official scaled memory-layer implementation for product-key style memory.

local use:

use as design pressure for sparse trainable memory and as a reference for what counts as value-table storage.

## content-routed sparse read and cache compression

### msa style sparse routing

source: [evermind-ai msa](https://github.com/EverMind-AI/MSA).

fit:

strong content-routed sparse-read baseline. not a high-density compressor by itself.

local use:

mirror the idea with corpus chunks, pooled routing keys, top-k document selection, and exact answer from selected chunks. charge cached chunks and routing keys.

### nsa

sources: [nvidia nsa docs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html), [nsa paper](https://arxiv.org/abs/2502.11089).

fit:

compression attention plus selection attention plus sliding window. useful as a routing/compressed-kv control.

local use:

implement a naive cpu block selector and compressed-kv baseline, not the gpu kernel.

### h2o, kvpress, streamingllm, rocketkv

sources: [h2o](https://github.com/FMInference/H2O), [kvpress](https://github.com/NVIDIA/kvpress), [streamingllm](https://github.com/mit-han-lab/streaming-llm), [rocketkv](https://github.com/NVlabs/RocketKV).

fit:

good cache-retention and eviction baselines. weak for durable exact knowledge.

local use:

use them as controls for what can be kept under a context or kv budget. they should fail arbitrary old exact fact retrieval unless the retained window contains the fact.

## distributed-symbolic and biological-ish baselines

### torchhd

sources: [torchhd repo](https://github.com/hyperdimensional-computing/torchhd), [torchhd jmlr](https://www.jmlr.org/papers/v24/23-0300.html).

fit:

solid vsa/hdc library for binding, bundling, permutation, and cleanup memory.

risk:

item dictionaries, cleanup memories, dimensions, seeds, and labels are side channels.

local use:

encode key, value, and provenance triples; sweep dimension, sparsity, and load; require random-label twin collapse.

### kanerva sdm

sources: [kanervasdm](https://pypi.org/project/KanervaSDM/), [sdm framework](https://sdm-framework.readthedocs.io/), [kanerva nasa report](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19890017031.pdf).

fit:

direct sparse distributed memory baseline for noisy associative recall.

risk:

hard-location addresses and counters dominate storage.

local use:

build exact-fact baseline, charge address matrix and counters, compare against hdc and sparse read.

### triadic memory

sources: [triadic memory report](https://peterovermann.com/TriadicMemory.pdf), [triadic-memory crate](https://lib.rs/crates/triadic-memory).

fit:

interesting sparse binary relation-completion idea.

risk:

low software maturity and high side-channel risk.

local use:

use only as a relation/provenance ablation until exact accounting exists.

## learned codec baselines

### paq8px context mixing

source: [paq8px](https://github.com/hxim/paq8px).

fit:

strong public text/source-code compression pressure line. it is not a neuroloc mechanism, but it is now a required source-code byte-compression baseline because it beats the current global-stream corpus payload on the hard frozen source blocks.

risk:

it is slow, has large fixed decoder/tool state, and is not an associative knowledge model. when used as an operation baseline, report payload bits separately from any retrieval/index/query wrapper bits. if the tool or dictionaries are bundled as model state, charge them; if treated as a fixed public decoder, label it that way.

local use:

the 2026-05-14 scratch audit downloaded `paq8px` v214 and ran levels 1 and 2 on the hard corpus. raw joined source bytes compressed from `802589` bytes to `51889` archive bytes at level 1 and `50712` archive bytes at level 2, or `405696` payload bits for the stronger checked line. the transformed body stream compressed from `472621` bytes to `52665` archive bytes at level 1. this demotes the current source-subtoken global-stream corpus codec from competitor-beating source-code compression to a pre-paq local codec product until a registered in-repo result beats this line or proves a fair operation that the public compressor cannot provide under matched accounting. see [[mistakes/public_context_mixing_baseline_missing]].

### zstd trained dictionaries

source: [zstandard](https://github.com/facebook/zstd).

fit:

strong public byte-compression baseline for local source-code corpora when trained dictionaries are allowed.

risk:

dictionary bytes are model state or side information. if they are trained from the target or treated as free, the comparison becomes an uncharged side channel.

local use:

`local_100k_zstd_trained_dictionary_baseline_audit` trains dictionaries only from target-excluded local source files, charges dictionary bytes plus header and selector bits, and also reports an undercharged payload-only diagnostic. hard validation shows the current source-subtoken block codec beats the charged trained-dictionary line by `30424` bits and the undercharged payload-only line by `26248` bits. the current global-stream frozen-corpus codec beats the charged trained-dictionary line by `283696` bits and the undercharged payload-only line by `250848` bits. this strengthens the source-code codec claim but does not authorize broad breakthrough wording.

### compressai

source: [compressai](https://github.com/InterDigitalInc/CompressAI).

fit:

mature learned compression toolkit with entropy bottlenecks and hyperpriors.

risk:

mostly image-oriented and not associative retrieval. model, decoder, and entropy tables must be charged.

local use:

adapt the learned-codec pattern to corpus chunks only if exact byte reconstruction is enforced and all model bits are charged.

### vector-quantize-pytorch

source: [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch).

fit:

practical vq, residual-vq, finite scalar quantization, and related discrete bottlenecks.

risk:

approximate reconstruction is not exact knowledge.

local use:

use for the learned non-row amortized code candidate, with arithmetic-coded indices and exact byte reconstruction as the gate.

## membership-only baselines

### neural bloom filter

source: [neural bloom filter](https://arxiv.org/abs/1906.04304).

fit:

learned approximate membership.

risk:

membership is not value retrieval. false positives are allowed.

local use:

use only for "does this key exist?" probes, not as exact fact storage.

## passion projects and low-confidence leads

examples found include sparse-coordinate storage prototypes, hdc engines, and bit-operation vsa/sdm claims. these may inspire tests, but they cannot justify acceptance unless source code, storage accounting, random-label controls, and exact retrieval gates are reproduced locally.

## current conclusion

there is no open-source system found here that already proves exact unknown-structure key-value knowledge retrieval at the 600x density target under honest accounting. however, paq8px now beats the current source-code byte-compression payload by a large margin, so source-code byte compression claims must clear that public context-mixing line before competitor-beating wording is allowed. the useful imports are baseline pressure and mechanism parts:

- minimal perfect hash plus payload codec
- product-key or memory-layer value table
- hdc and sdm superposition
- content-routed sparse read
- learned entropy-coded codec
- public context-mixing source-code compression
