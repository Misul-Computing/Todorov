# local 100k weight mantissa payload adapter

status: historical context only. frozen as of 2026-05-13. do not edit.

## summary

`local_100k_weight_mantissa_payload_adapter` is a demoted weight-state payload diagnostic.

it packs the compressed source payload into the mantissa bits of trainable fp32 torch parameters, then reconstructs the payload from the model state and answers bounded exact qa from the decoded source block.

the mechanical retrieval path works, but the result is not promoted. after charging the real fp32 model state, the candidate falls below the executable same-block content-scan and minimal-perfect-hash diagnostics. the apparent `30.998884002808474x` paper-surface line is a steganographic undercharge, not a publishable density result.

## hard validation

command:

```text
python neuroloc\simulations\suite_runner.py --simulation local_100k_weight_mantissa_payload_adapter --profile hard --output-root codex_local_output\suite_l100k_weight_mantissa_hard --timeout-sec 1200
```

result:

- suite result: pass.
- fact count: `4096`.
- exact answer success: `1.0`.
- heldout exact answer success: `1.0`.
- random-label twin success: `0.0`.
- controls collapse: `1.0`.
- adapter parameter count: `11224`.
- mantissa bits per parameter: `23`.
- block payload bits: `258144`.
- committed state bits after fp32 charging: `391976`.
- paper-surface accounted bits after fp32 charging: `396072`.
- adapter strict multiplier after fp32 charging: `17.12065636671633x`.
- paper-surface strict multiplier after fp32 charging: `16.94360217334222x`.
- apparent mantissa paper-surface multiplier: `30.998884002808474x`.
- same-block content-scan multiplier: `23.06526987269378x`.
- same-block undercharged mph multiplier: `23.064001539688213x`.
- static public baseline pass: `0.0`.
- beats same-block content-scan baseline: `0.0`.
- beats same-block undercharged mph baseline: `0.0`.
- mantissa payload carrier used: `1.0`.
- mantissa steganography diagnostic: `1.0`.
- publishable weight-payload candidate: `0.0`.
- strict breakthrough authorization: `0.0`.
- strict 600x pass: `0.0`.
- arbitrary chat authorization: `0.0`.

## category check

implemented operation: fp32 parameter-mantissa payload packing for one compressed source block, followed by bounded exact qa.

strongest baseline: executable same-block content scan and same-block undercharged minimal-perfect-hash payload accounting, both at about `23.06x`.

what passed: exact answer recovery, random-label collapse, disabled-control collapse, state-dict parameter payload recovery, and the absence of a separate state-dict payload buffer.

what failed or remains unproved: the information is hidden in trainable parameter mantissas. charging only the number of fp32 parameters creates an apparent density win, but charging either the payload bits or the physical fp32 state removes the win.

why not promoted to breakthrough: it is a steganographic model-state payload, not implicit knowledge stored in ordinary learned weights. the fair fp32-state accounting line is below the same-interface scanner and undercharged mph diagnostics.

## decision

accepted only as a diagnostic. future weight-native work must distinguish ordinary useful learned weights from payload carriers and must report payload-bit, physical-state, and fair scanner baselines before any promotion.

## see also

- [[PROJECT_PLAN]]
- [[tests/local_100k_margin_recompression_adapter]]
- [[tests/local_100k_semantic_alias_payload_adapter]]
- [[tests/local_100k_source_native_relation_adapter]]
- [[mistakes/weight_mantissa_payload_steganographic_accounting_error]]
