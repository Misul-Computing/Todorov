# transformerov

a hybrid sequence model. one attention layer does recall, three gated-delta layers do the cheap state tracking, a short conv feeds them, and the whole thing grounds in a sense. the name is the point: a bridge between the neural memory where it works and the transformer where it does not.

## the problem it answers

a recurrent or linear memory does not learn verbatim recall under sgd. this is the retrieval wall, and it held across six paid runs and a from-scratch reproduction here: the descent memory and the ungated delta both cliff on associative recall after four to six key-value pairs. attention does recall for free. so the design is not about replacing attention or replacing the linear memory. it uses each where it earns its place: attention for recall, gated delta for everything else.

## architecture

four blocks, pattern [attn, delta, delta, delta]. attention is the first block, not the last. placement is not cosmetic. with attention last or in the middle the hybrid cliffs exactly like pure delta (recall 0.007 at 16 pairs), because by the time the residual reaches it the three delta layers have mixed away the raw key and value tokens the induction circuit needs. attention first reads them clean off the embedding and recovers full recall.

each block: rmsnorm, then a short causal depthwise conv (kernel 4, the token shift induction needs), then the mixer, then rmsnorm and a swiglu mlp. embedding and head weights tied.

## the gated delta

the delta layers run the gated delta rule, chunkwise, on the mac gpu.

per token, per head, a decay g in (0,1):

    S_t = g_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t = q_t^T S_t

chunkwise form. let G be the cumulative decay from the chunk start. set S_hat = S / G. the gated rule then becomes the plain ungated delta on rescaled values v/G, read by rescaled queries q*G, with the chunk total decay carried across the chunk boundary. this reuses the ungated chunkwise machinery unchanged and matches the gated recurrent to 5e-16 at chunk sizes 8, 16, 32. mps runs it 2.4x faster than cpu at 8k context.

stability is the hard part, and a single-seed run hides it. that v/G rescaling amplifies gradients by 1/G, and when the learned decay gets strong over a long chunk, G underflows and the gradient overflows to nan. it shows up as an intermittent nan a couple hundred steps in, on some seeds and not others, because mps is not bit-deterministic, so it only surfaces under a multi-seed harden. three guardrails fix it together:

- floor the decay so g stays in (0.975, 1). the within-chunk amplification is g_min^chunk, so this holds 1/G under about 5 at chunk 64. a shorter chunk would allow a lower floor, but chunk 64 is ~1.65x faster than chunk 16 here, so the floor is the cheaper knob.
- a per-head rmsnorm on the memory output, the retnet/gla output norm, bounding what enters the residual.
- a state-norm clamp at 100 as a safety net, which the ungated path already had and the gated path had lost.

with all three, every seed trains clean. the decay bias starts at 0.4 so g begins near 0.99 and the state barely forgets at init; the model learns its forgetting rate within the floor.

## results

toy scale, single seed, m5 pro mps, byte level. what matters is the shape of the three curves, not the absolute numbers.

recall, mqar token accuracy, 4 layers, d=128:

    pairs   attn    delta   hybrid
    4       0.980   0.743   0.990
    8       0.938   0.284   0.969
    16      0.941   0.007   0.944

language, byte bits-per-byte, 512 context, d=256:

    attn 1.651   delta 1.641   hybrid 1.647

grounding, feel touch-world, three arms:

    real 1.000   blind 0.140   fake 0.150

one attention layer in four buys transformer-grade recall and transformer-grade language. the gated delta on its own beats attention on language but cannot recall. the hybrid takes both, at a quarter of the attention layers and a quarter of the kv-cache growth, and it still grounds: a real sense solves the task, a fake sense does not.

## files

    chunk_delta.py   ungated and gated chunkwise delta, correctness and speed checks
    bench.py         the hybrid lm (gated delta memory, attention, conv) and byte bpb
    mqar_bench.py    recall sweep, any block pattern via --mixers
    feel_hybrid.py   grounding, three arms

run on the gpu with --device mps. the venv is .venv-feel.

## honest edges

toy scale (about 4.5m params), one seed, one corpus. the hybrid is slower in wall-clock than pure attention at 512 context, because the delta layers only win on speed past about 4k, so the gain at 512 is kv-cache memory, not time. recall past 32 pairs collapses for all three at this size, a model-capacity ceiling rather than a property of the bridge. next step: scale the hybrid up and rerun the three axes at size.
