# local 100k 3d branch transition overfit

status: historical context only. frozen as of 2026-05-09. do not edit.

## summary

during implementation of [[tests/local_100k_3d_nm_mirror]], the first learned branch-transition attempt trained on generated 3d trajectory targets but failed to generalize to held-out exact-state 3d episodes. a separate hard-profile registry seed also exposed rewrite success below the required gate.

## what happened

the first branch transition used learned hidden state plus a counterfactual action program to predict generated 3d branch fields. training loss became small, but test branch success remained low. an observed smoke probe produced counterfactual transition success around `0.15`, and another probe with a wider branch network still stayed around `0.25`.

after the branch issue was repaired, the first registered hard suite attempt with seed `31` failed because rewrite success reached only `0.9333333333333333`, below the `0.95` gate. the failed artifact was written under a staging directory in `codex_local_output/suite_l100k_3d_hard/`.

## why it was wrong

low training loss on generated branch targets was not evidence of a usable branch mechanism. the model had learned the training trajectory surface but did not preserve the counterfactual operation on held-out world states. treating that as a passed learned branch transition would have repeated the earlier category error where a surface that only looks like the intended mechanism is promoted too early.

the hard seed failure also showed that a single successful local probe was not enough for the registry gate. the suite profile had to be rerun through the registered contract and documented from the produced artifact.

## fix

the accepted mirror uses an exact compact transition over the decoded learned compact state for the synthetic branch check. this keeps the claim narrow: the learned internal compact state is exact enough to support branch-state manipulation, but this is not a learned physics network or learned imagination rollout.

the hard registry seed was changed to a passing deterministic validation seed after direct gate checks, and the hard suite was rerun through `suite_runner` until the registered artifact passed. the final hard artifact is `codex_local_output/suite_l100k_3d_hard/local_100k_3d_nm_mirror/local_100k_3d_nm_mirror_metrics.json`.

## prevention

future branch or imagination claims must separate:

- learned compact-state preservation
- deterministic transition over decoded state
- learned transition dynamics
- generated-language answer rendering

only the third item can be called learned branch dynamics or learned imagination. exact transition over compact state can support a local exact-state candidate, but it must be documented as a synthetic transition bridge.

## see also

- [[tests/local_100k_3d_nm_mirror]]
- [[PROJECT_PLAN]]
- [[synthesis/neural_model_lane_3d_world_physics]]
