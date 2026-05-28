# paq8px public context-mixing baseline audit

status: historical context only. frozen as of 2026-05-14. do not edit.

## summary

this audit checks a strong public context-mixing compressor against the current hard frozen source-code corpus used by the source-subtoken global-stream codec and disjoint retrieval codec.

it is a baseline audit, not a neuroloc product. the result demotes competitor-beating wording for the current source-code byte-compression line because paq8px v214 beats the current global-stream payload by a large margin.

## implementation

- tool: `paq8px` v214 windows x64.
- source: `https://github.com/hxim/paq8px/releases/download/v214/paq8px_v214_windows_x64.7z`
- local tool path: `codex_local_output/compression_tools/paq8px_v214/paq8px.exe`
- scratch input path: `codex_local_output/compression_scratch/raw_joined.bin`
- scratch body input path: `codex_local_output/compression_scratch/body_stream.bin`

## commands

```text
paq8px.exe -1 codex_local_output\compression_scratch\body_stream.bin codex_local_output\compression_scratch\body_stream_paq1.paq8px214
paq8px.exe -1 codex_local_output\compression_scratch\raw_joined.bin codex_local_output\compression_scratch\raw_joined_paq1.paq8px214
paq8px.exe -2 codex_local_output\compression_scratch\raw_joined.bin codex_local_output\compression_scratch\raw_joined_paq2.paq8px214
```

## result

- hard raw joined source bytes: `802589`.
- hard transformed body stream bytes: `472621`.
- current source-subtoken global-stream corpus selected payload bits: `699144`.
- current global-stream body payload bytes: `83139`.
- current global-stream body payload bits: `665112`.
- paq8px v214 level 1 raw joined archive bytes: `51889`.
- paq8px v214 level 1 raw joined payload bits: `415112`.
- paq8px v214 level 2 raw joined archive bytes: `50712`.
- paq8px v214 level 2 raw joined payload bits: `405696`.
- paq8px v214 level 1 transformed body archive bytes: `52665`.
- paq8px v214 level 1 transformed body payload bits: `421320`.

## decision

paq8px v214 level 2 is now the strongest checked public source-code byte-compression pressure line for the hard corpus. future source-code byte-compression promotion must beat this line under matched payload and operation accounting, or stay explicitly local and pre-paq.

the current global-stream corpus codec remains useful as an in-repo exact reconstruction and model-state payload product against the earlier standard/zstd baselines, but it is not a public-compressor breakthrough.

## see also

- [[PROJECT_PLAN]]
- [[mistakes/public_context_mixing_baseline_missing]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_subtoken_disjoint_retrieval_codec]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
