# feel imagination goal

substrate: linear matrix descent memory (v01 SequenceModel, mem_mode=linear), cpu, .venv-feel.

## established baseline (do not regress)

- working feel model: recall real 1.00 / blind 0.14 / fake 0.15; count real 1.00 / blind 0.33 / fake 0.26.
- imagination as inference (occlusion recall, 1 hidden cell): occ_structured 0.87, occ_random 0.58, fake 0.44, blind 0.56, observed 0.98.
- imagination feeding the count (occlusion count, 1 hidden cell, no mechanism): occ_structured 0.66, occ_random 0.45, fake 0.14, blind 0.24, observed_full 1.00.

## goal

a fill-back mechanism that makes imagination drive the count inline: during the sweep the model predicts each hidden cell (directly supervised, b1 style) and feeds the imagined value into the running computation, so the count integrates the imagined cell rather than missing it.

## success criteria (all must hold)

1. occ_structured (count, 1 hidden) with fill-back >= 0.80, up from the 0.66 no-fill baseline, toward observed_full 1.00.
2. no-fill control collapses to ~0.66; random-fill control collapses to ~0.45.
3. occ_random (random world, fill on) stays <= 0.55 (imagination needs dynamics to roll).
4. fake and blind stay <= 0.40 (chance for count).
5. causal_no_future_leak passes on every arm.
6. the lift holds across >= 2 seeds.

## metric

occ_structured token accuracy on occlusion-count. direction: higher.

## kill condition

if after a few iterations fill-back cannot beat the no-fill baseline by >= 0.08 on occ_structured with controls intact, log the negative and stop. an honest negative is an acceptable terminal state, consistent with the feel-and-predict negative-transfer history.

## outcome (achieved, loop closed)

converged in two iterations, both confirmed across 2 seeds, .venv-feel, cpu.

- iteration 1, causal fill-back (imagine the hidden cell from left context only), feel_imagine_fillback.py:
  fillback 0.789 / 0.758 (seeds 0/1), nofill 0.430 / 0.523, oracle 0.992 / 0.969. lifts the count ~+0.29 over no-fill but parks at the causal ceiling ~0.77, just under target: a cell hidden at position 0 has no left context, predicted ceiling (0.5 + 7*0.85)/8 = 0.806.
- iteration 2, bidirectional fill-back (imagine from the whole sensed sweep, b1 full context), feel_imagine_fillback2.py:
  fillback 0.836 / 0.805 (seeds 0/1, mean 0.82), nofill 0.656 / 0.609, oracle 1.000 / 1.000, random 0.453 / 0.406, occ_random 0.367 / 0.453, fake 0.102 / 0.148, blind 0.109 / 0.156.

all criteria met across both seeds: fillback >= 0.80, +0.18 to +0.20 over no-fill, oracle ~1.0, random/occ_random/fake/blind collapse, causal clean. the imagined value (not just any fill) drives the count: fillback beats random-fill by ~0.38.

caveats: toy scale (8-cell strip), 1 hidden cell, linear matrix memory (approach a working substrate, not the descent/candidate-g memory), cpu, 2 seeds.
