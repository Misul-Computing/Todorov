"""novelquant data-free weight quantization package.

Modules:
  rotate  , Walsh-Hadamard transforms and randomized incoherence rotation.
  codebook, NF4, Lloyd-Max (Gaussian), data-fit Lloyd, and uniform quantizers.
  quantize, per-tensor quant/dequant dispatch + whole-model application.
  eval    , canonical WikiText-2 perplexity (concatenated, 40k tokens).
  picker  , (workstream 2) per-tensor bit-width assignment.  Placeholder.
  pack    , (workstream 4) bit-packed storage + dequant.  Placeholder.
"""
from . import rotate, codebook, quantize, eval  # noqa: A004,F401  (shadows builtin; intentional)
