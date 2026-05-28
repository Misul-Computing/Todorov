# weight mantissa payload steganographic accounting error

status: historical context only. frozen as of 2026-05-13. do not edit.

## what happened

`local_100k_weight_mantissa_payload_adapter` tried to move the compressed payload into trainable torch parameters by packing payload bits into fp32 mantissas.

the first accounting line treated each carrier parameter as one ordinary trainable parameter while using `23` mantissa bits per parameter for payload. that produced an apparent paper-surface multiplier of `30.998884002808474x`.

## why it mattered

the project target is useful knowledge density in model state, not a hidden byte stream inside floating-point mantissas. if a method stores a payload inside the bit pattern of parameters, those payload bits and the physical parameter state must be charged. otherwise the result is a steganographic undercharge.

## fix

the simulation now reports the demoted state:

- payload-bit paper-surface multiplier: `22.7450665654402x`.
- fp32 paper-surface multiplier: `16.94360217334222x`.
- same-block content-scan multiplier: `23.06526987269378x`.
- same-block undercharged mph multiplier: `23.064001539688213x`.
- static public baseline pass: `0.0`.
- beats same-block content-scan baseline: `0.0`.
- beats same-block undercharged mph baseline: `0.0`.
- mantissa steganography diagnostic: `1.0`.
- publishable weight-payload candidate: `0.0`.
- strict breakthrough authorization: `0.0`.

the registry and tests now require the demoted diagnostic state.

## prevention

future weight-native compression claims must report three lines:

- apparent parameter-count density.
- payload-bit charged density.
- physical model-state density.

promotion can use only the charged lines, and any fair scanner or minimal-perfect-hash diagnostic with the same address surface must be treated as a baseline.

## see also

- [[tests/local_100k_weight_mantissa_payload_adapter]]
- [[mistakes/semantic_alias_payload_adapter_formula_alias_category_error]]
- [[mistakes/source_native_relation_stride_rule_category_error]]
