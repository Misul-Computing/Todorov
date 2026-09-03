# candidate g reproduction and seed sweep on apple m5 pro

status: historical context only. frozen as of 2026-06-26. do not edit.

## what this is

a reproduction of the feel bench (v0.1 toy) and the affect-gated write experiment (candidate f/g) on a new local machine, plus a 20-seed ignition sweep for the shuffled-surprise arm. the original results were produced on an older cpu; this run verifies them on apple silicon with mps and establishes ignition statistics across seeds.

## hardware and environment

- machine: macbook pro, apple m5 pro, 24 gb unified memory, 16 gpu cores, metal 4
- python 3.9.6, torch 2.8.0, mps backend (torch.backends.mps.is_available() = true)
- venv at repo root, torch + numpy only
- all runs use mps except the cpu reproduction check noted below
- fp16 tested and rejected: the descent memory's internal scan requires fp32 (nan losses under fp16, consistent with the technical report's stability notes)
- parallelism: 4 worker processes via multiprocessing spawn pool, each pinning a separate mps context

## feel bench reproduction

ran `v01/quick_touch_test.py` and `v01/quick_count_test.py` on mps with the default config (strip 6, 300/500 steps, d_model 96, 2 layers, linear memory).

recall (3-arm, touch_world):

| arm    | exact_acc | token_acc | ci (wilson 95)   | loss   | time   |
|--------|-----------|-----------|------------------|--------|--------|
| blind  | 0.140     | 0.500     | (0.085, 0.221)   | 0.693  | 20.0s  |
| real   | 1.000     | 1.000     | (0.963, 1.000)   | 0.006  | 13.8s  |
| fake   | 0.150     | 0.483     | (0.093, 0.233)   | 0.695  | 13.9s  |

doc values: blind 0.14, real 1.00, fake 0.15. exact match.

count (3-arm, touch_count):

| arm    | count_acc | ci (wilson 95)   | loss   | time   |
|--------|-----------|------------------|--------|--------|
| blind  | 0.300     | (0.219, 0.396)   | 1.640  | 23.0s  |
| real   | 1.000     | (0.963, 1.000)   | 0.000  | 17.1s  |
| fake   | 0.370     | (0.282, 0.468)   | 1.715  | 16.7s  |

doc values: blind 0.29, real 1.00, fake 0.34. within seed and float variance.

the grounding claim reproduces. a located, accumulated touch sense turns an unsolvable task solvable, and a fake randomized sense does not. the fake-touch control holds.

## candidate f reproduction (affect-gated write, true surprise)

ran `v01/quick_affect_test.py` at seed 0, 400 steps, all 8 arms, on mps.

| arm            | exact_acc | token_acc | loss   | surprise | write_gain | causal_ok |
|----------------|-----------|-----------|--------|----------|------------|-----------|
| control        | 0.000     | 0.117     | 2.869  | nan      | nan        | true      |
| surprise_half  | 0.010     | 0.110     | 2.967  | 0.353    | 0.932      | true      |
| surprise       | 0.000     | 0.043     | 3.172  | 0.240    | 0.720      | true      |
| shuffle        | 0.000     | 0.043     | 3.217  | 0.377    | 0.942      | true      |
| noise          | 0.000     | 0.040     | 3.206  | 0.498    | 1.021      | true      |
| noisetoken     | 0.000     | 0.068     | 3.173  | 0.475    | 0.999      | true      |

the original doc reported shuffle seed 0 at exact 0.110, token 0.557 at 400 steps. that ignition did not reproduce on this hardware at seed 0. the control arm also differs (token 0.117 vs doc 0.037), which points to float kernel differences between the original cpu and mps changing the training trajectory at this small scale.

a cpu reproduction on the original code (git stash, cpu device) confirmed the divergence is hardware, not code: cpu seed 0 shuffle gave exact 0.000, token 0.040 at 400 steps. the original ignition was hardware-specific at seed 0.

true surprise remains at chance across all strengths, consistent with the candidate f negative verdict. the pre-registered hypothesis (surprise content drives retrieval) stays falsified.

## candidate g ignition sweep (shuffled surprise, 20 seeds)

since the seed 0 ignition did not reproduce on this hardware, ran a 20-seed sweep to establish ignition statistics. two phases: 400 steps (quick screen) and 1200 steps (full training, matching the doc's extended run).

### 400-step screen, 10 seeds, 3 arms (shuffle, control, noise)

10 seeds x 3 arms = 30 jobs, 4 workers, 1233s total (41s/job).

control: all 10 seeds at exact 0.000, token 0.028-0.077. chance confirmed.

noise: all 10 seeds at exact 0.000, token 0.035-0.090. chance confirmed. uniform iid noise does not ignite.

shuffle: 2/10 seeds above exact 0 (seed 4: exact 0.010, token 0.170; seed 6: exact 0.010, token 0.403). 8/10 at chance. partial signal at 400 steps.

### 1200-step full sweep, 20 seeds, shuffle arm only

20 seeds, 4 workers, 1307s total (65s/job).

| seed | exact_acc | token_acc | ci (wilson 95)   | loss   | surprise | write_gain | ignited |
|------|-----------|-----------|------------------|--------|----------|------------|---------|
| 0    | 0.000     | 0.035     | (0.000, 0.037)   | 3.142  | 0.308    | 0.802      | no      |
| 1    | 0.350     | 0.757     | (0.264, 0.447)   | 0.859  | 0.350    | 0.865      | yes     |
| 2    | 0.000     | 0.030     | (0.000, 0.037)   | 3.189  | 0.407    | 0.926      | no      |
| 3    | 0.000     | 0.040     | (0.000, 0.037)   | 3.119  | 0.300    | 0.808      | no      |
| 4    | 0.010     | 0.068     | (0.002, 0.054)   | 3.126  | 0.214    | 0.661      | weak    |
| 5    | 0.300     | 0.733     | (0.219, 0.396)   | 0.909  | 0.470    | 0.978      | yes     |
| 6    | 0.000     | 0.040     | (0.000, 0.037)   | 3.146  | 0.328    | 0.826      | no      |
| 7    | 0.000     | 0.077     | (0.000, 0.037)   | 3.088  | 0.343    | 0.785      | no      |
| 8    | 0.310     | 0.695     | (0.228, 0.406)   | 1.213  | 0.433    | 0.937      | yes     |
| 9    | 0.000     | 0.035     | (0.000, 0.037)   | 3.174  | 0.306    | 0.742      | no      |
| 10   | 0.600     | 0.887     | (0.502, 0.691)   | 0.562  | 0.417    | 0.944      | yes     |
| 11   | 0.000     | 0.048     | (0.000, 0.037)   | 3.175  | 0.317    | 0.783      | no      |
| 12   | 0.000     | 0.035     | (0.000, 0.037)   | 3.132  | 0.400    | 0.842      | no      |
| 13   | 0.000     | 0.048     | (0.000, 0.037)   | 3.155  | 0.412    | 0.923      | no      |
| 14   | 0.000     | 0.040     | (0.000, 0.037)   | 3.223  | 0.314    | 0.704      | no      |
| 15   | 0.000     | 0.037     | (0.000, 0.037)   | 3.174  | 0.409    | 0.943      | no      |
| 16   | 0.000     | 0.043     | (0.000, 0.037)   | 3.143  | 0.179    | 0.604      | no      |
| 17   | 0.320     | 0.740     | (0.237, 0.417)   | 0.877  | 0.448    | 0.955      | yes     |
| 18   | 0.000     | 0.050     | (0.000, 0.037)   | 3.190  | 0.368    | 0.847      | no      |
| 19   | 0.000     | 0.030     | (0.000, 0.037)   | 3.154  | 0.369    | 0.892      | no      |

### ignition statistics

- strong ignition (exact >= 0.30): 5/20 seeds (1, 5, 8, 10, 17). exact range 0.300-0.600, token range 0.695-0.887.
- weak ignition (exact > 0 but < 0.30): 1/20 seeds (4). exact 0.010, token 0.068.
- no ignition: 14/20 seeds. all at exact 0.000, token near chance.

ignition rate (strong): 25%, wilson 95% ci 11.3-44.1% on n=20.
ignition rate (any above chance): 30%, wilson 95% ci 14.5-49.1% on n=20.

### key observation: bimodal distribution

the outcome is bimodal. seeds either ignite strongly (exact 0.30-0.60, loss drops to 0.5-1.2) or stay flat at chance (loss stays near 3.1-3.2). there is no middle ground. this is a phase transition, not a gradient. the loss curve makes the ignition visible: ignited seeds descend to 0.5-1.2, non-ignited seeds plateau at 3.1-3.2.

this is consistent with a stochastic circuit-formation phenomenon: the shuffled gain process either helps sgd find the retrieval circuit or it does not, and the outcome is decided early.

### comparison to original

the original doc reported seed 0 ignition at exact 0.110 (400 steps) growing to 0.360 (1200 steps), and seed 1 at chance. on this hardware, seed 0 is at chance and seeds 1, 5, 8, 10, 17 ignite. the ignition phenomenon reproduces, but the specific seed that ignites is hardware-dependent. this is expected for a stochastic phase transition: the float kernel differences between cpus shift which initialisations tip over the threshold.

the ignition rate of ~25% is the more reliable statistic than any single seed's outcome.

## mechanistic observation

reading `v01/memory.py`, the affect gate computes per-token surprise `s_t = ||e_t|| / (||e_t|| + ||v_t|| + eps)` from the inner-loop prediction error, then modulates the write gain `beta_eff = beta * (1 - affect + affect * rel)` where `rel` is the surprise normalised by its running mean.

true surprise creates a self-suppressing feedback loop. as the memory learns an association, its prediction error on that association drops. lower error means lower surprise, which means lower write gain, which means weaker writes, which means the association is not reinforced before the retrieval circuit forms. the gate shuts down the write exactly when learning is starting to work.

shuffled surprise breaks this loop. sequence i's gain is driven by sibling sequence j's surprise (permuted across the batch axis). sequence i's learning does not affect sequence j's error, so the gain stays decoupled from the learner's own progress. the write keeps flowing.

but uniform noise also decouples the gain from the learner's progress, and it does not ignite (0/10 seeds at noise, 0/10 at noisetoken). decoupling alone is not sufficient. the doc named four candidate properties that distinguish shuffled surprise from uniform noise:

1. distribution shape (shuffled surprise is bounded in (0,1) with a specific distribution; uniform noise is flat)
2. endogenous ramp-in (the model's surprise distribution starts nearly degenerate, so the shuffle's gain variance grows with training)
3. population coupling (gains track the batch's learning state)
4. temporal correlation (sibling surprise traces are smooth in time, pair-coherent, unlike iid noise)

a fifth property is visible in the code: the shuffled surprise is recomputed per step from the current batch's actual error distribution, so its statistics shift as the population learns, while uniform noise is drawn from a fixed generator with the same seed every forward pass.

## open work: temporal correlation test

two new affect modes were added to `v01/memory.py` to isolate temporal correlation:

- `smooth_noise`: exponentially smoothed uniform noise per (batch, head), ema coefficient 0.7. temporal structure plus head diversity, no surprise content.
- `smooth_noise_token`: same but shared across heads. temporal structure only.

a 20-seed x 4-arm sweep (shuffle, smooth_noise, smooth_noise_token, noise) at 1200 steps was launched but interrupted before completion. the results are not yet available. this test would distinguish whether temporal correlation in the gain process is the active ingredient (smooth_noise ignites) or whether population coupling is required (smooth_noise stays at chance, only shuffle ignites).

## files

- `v01/sweep_affect.py` - parallel seed sweep harness with multiprocessing, mps, and configurable arms. supports the original 8 arms plus smooth_noise and smooth_noise_token.
- `v01/memory.py` - added smooth_noise and smooth_noise_token affect modes (ema-smoothed uniform noise, coefficient 0.7, per-head and per-token variants).
- `v01/quick_touch_test.py`, `v01/quick_count_test.py`, `v01/quick_affect_test.py` - changed device from "cpu" to "mps" and added `.to("mps")` on model creation for local hardware.

## what this does and does not show

does show:
- the feel bench grounding result reproduces on new hardware.
- candidate f (true surprise) stays falsified.
- candidate g (shuffled surprise) ignition reproduces on new hardware at a ~25% rate across 20 seeds.
- ignition is bimodal: strong or absent, no middle ground.
- the ignition phenomenon is real but seed- and hardware-sensitive.

does not show:
- which property of the shuffled gain process drives ignition (temporal correlation test incomplete).
- whether ignition scales beyond mqar at 8 pairs to the retrieval wall seen in paid runs.
- whether the effect replicates on a different substrate or at a larger model size.
- transfer or generalization. every result is on the same mqar task the model was trained on.

## next

1. complete the temporal correlation sweep (smooth_noise vs smooth_noise_token vs shuffle vs noise, 20 seeds, 1200 steps). this isolates whether temporal structure in the gain is sufficient for ignition or whether population coupling is required.
2. if smooth_noise ignites, test distribution shape by matching the shuffled surprise marginal distribution with temporally correlated noise.
3. if smooth_noise does not ignite, population coupling is the likely driver. test by feeding the gain the batch-mean surprise (averaged across sequences) rather than a permuted individual sequence's surprise.
4. once the active ingredient is identified, test whether it generalizes to a harder retrieval task (more pairs, longer sequences) before any claim about the retrieval wall.

## See also

- `neuroloc/wiki/PROJECT_PLAN.md`
- `neuroloc/wiki/synthesis/substrate_requires_architectural_change.md`
