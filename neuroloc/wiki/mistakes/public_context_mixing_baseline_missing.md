# public context-mixing baseline missing

status: historical context only. frozen as of 2026-05-14. do not edit.

## mistake

the source-code byte-compression line compared against standard in-repo codecs and charged zstd trained dictionaries, but it did not include a strong public context-mixing compressor in the pressure stack before product wording was used.

on 2026-05-14, `paq8px` v214 level 2 compressed the same hard raw joined source block used by the global-stream corpus codec to `50712` bytes. this is `405696` payload bits before any operation-specific retrieval or wrapper accounting, far below the current `local_100k_source_subtoken_global_stream_corpus_codec` `699144` selected payload bits.

## why it matters

the current source-subtoken global-stream codec still has useful in-repo evidence: exact reconstruction, torch `state_dict` payload use, random-label controls, and wins over the earlier raw standard and zstd trained-dictionary lines. but it is not competitor-beating source-code compression while a public context-mixing compressor beats it by this margin.

this also weakens the disjoint retrieval surface as a compression breakthrough candidate. the retrieval wrapper beats the executable raw content-scan and undercharged mph diagnostics that were implemented in-repo, but the underlying payload is not stronger than the newly checked public context-mixing pressure line.

## correction

add public context-mixing compression to the required baseline stack for source-code byte-compression claims. future source-code byte-compression product claims must either beat the paq8px pressure line under matched payload and operation accounting, or state explicitly that they are pre-paq local diagnostics.

do not treat zstd trained dictionaries as the strongest public compressor pressure line. they remain useful, but they are not enough.

## evidence

- downloaded tool: `codex_local_output/compression_tools/paq8px_v214/paq8px.exe`
- source: `https://github.com/hxim/paq8px/releases/download/v214/paq8px_v214_windows_x64.7z`
- command shape: `paq8px.exe -2 <input> <archive>`
- hard raw joined source bytes: `802589`
- hard raw joined paq8px v214 level 1 archive bytes: `51889`
- hard raw joined paq8px v214 level 2 archive bytes: `50712`
- hard transformed body bytes: `472621`
- hard transformed body paq8px v214 level 1 archive bytes: `52665`
- current global-stream corpus selected payload bits: `699144`
- current disjoint retrieval accounted bits: `431536`
- paq raw joined level 2 payload bits before operation wrapper: `405696`

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_source_subtoken_global_stream_corpus_codec]]
- [[tests/local_100k_source_subtoken_disjoint_retrieval_codec]]
- [[tests/local_100k_zstd_trained_dictionary_baseline_audit]]
- [[tests/paq8px_public_context_mixing_baseline_audit]]
- [[synthesis/neural_model_lane_operation_preserving_compression]]
