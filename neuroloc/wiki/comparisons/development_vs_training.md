# development vs training

status: current (as of 2026-04-22).

## the question

biological neural development (critical periods, synaptic pruning, self-organization, curriculum of increasing complexity) shapes circuits through a multi-stage process spanning years. ML training (warmup, learning rate decay, progressive training, pruning/distillation) shapes parameters through gradient descent spanning hours to days. both produce functional networks from unstructured initial conditions. how deep does the analogy go, and where does it break?

this comparison is motivated by a specific practical question: should todorov's training recipe incorporate developmental staging -- layer-specific learning rate schedules, progressive activation of architectural features, or explicit pruning phases?

## dimension 1: plasticity modulation

### biology

plasticity is not constant. it varies by region, layer, and developmental stage:

- [[critical_periods]] open and close on region-specific schedules: V1 matures by age 4-6, PFC matures by age 16-20 (Huttenlocher and Dabholkar 1997)
- the trigger is internal (PV+ interneuron maturation, see [[critical_periods]]), not externally scheduled
- molecular brakes (PNNs, myelin, epigenetic modifications) actively terminate plasticity, not just reduce it
- plasticity can be partially reopened in adults (chondroitinase, interneuron transplantation, pharmacological interventions)

the key feature: plasticity modulation is LAYER-SPECIFIC and SELF-ORGANIZED. it emerges from the circuit's own activity and maturational state, not from an external controller.

### ML training

learning rate schedules modulate plasticity:

- warmup: gradual increase over first few thousand steps, preventing early gradient instability
- cosine/linear decay: gradual decrease, reducing the magnitude of late-training parameter updates
- sometimes: cyclical or restart schedules (cosine annealing with warm restarts)

the key feature: learning rate modulation is GLOBAL (all parameters use the same schedule) and EXTERNALLY IMPOSED. there is no analog of layer-specific maturation or self-organized plasticity closure.

### comparison

| property | biology | ML training |
|---|---|---|
| schedule | region-specific, heterochronous | global, uniform |
| trigger | internal (PV+ maturation, activity-dependent) | external (hyperparameter schedule) |
| termination | structural (PNNs, myelin, epigenetics) | gradual decay (no structural lock-in) |
| reopening | possible with intervention | change learning rate (trivial) |
| spatial specificity | layer-specific, cell-type-specific | same for all parameters |

**verdict:** the analogy between learning rate decay and critical period closure is WEAK. the most important feature of biological plasticity modulation -- its REGIONAL SPECIFICITY and SELF-ORGANIZED timing -- has no counterpart in standard ML training.

**dissent:** Achille, Rovere, and Soatto (2019) showed that deep networks exhibit "critical learning periods" analogous to biological ones: brief exposure to corrupted data early in training permanently impairs performance in a way that later exposure does not. this occurs despite a uniform learning rate. the critical period emerges from the dynamics of weight optimization (specifically, the Fisher information matrix of the weights), not from an externally imposed schedule. this suggests that critical-period-like phenomena may be intrinsic to gradient-based optimization, regardless of whether the learning rate varies by layer.

## dimension 2: information ordering (curriculum)

### biology

the brain encounters information in a developmentally ordered sequence:

- spontaneous retinal waves before visual experience (see [[developmental_self_organization]])
- simple sensory stimulation (faces, edges, basic sounds) dominates early experience due to limited motor control and attention span
- progressively complex sensory-motor experience as the organism develops
- language: phonemes -> words -> phrases -> complex syntax (see [[critical_periods]], language acquisition section)

this is not explicitly designed as a curriculum. it is an emergent consequence of (a) maturational constraints on sensory and motor systems, (b) environmental structure (caregivers simplify input for infants -- "motherese"), and (c) cognitive capacity limitations (limited working memory forces attention to simpler patterns).

Elman (1993) showed that this matters computationally: a simple recurrent network failed to learn complex grammatical structures when trained on the full distribution from the start, but succeeded when trained with limited working memory that gradually expanded. the constraint forced the network to learn simple structures first, which then served as scaffolding for complex structures. the limitation was not a handicap -- it was a prerequisite.

### ML training

curriculum learning in ML explicitly orders training data from simple to complex:

- Bengio et al. (2009) formalized curriculum learning, showing that training on easy examples first can accelerate convergence and sometimes improve final performance
- todorov uses progressive context extension: 256 -> 512 -> 1024 -> 2048 tokens over training, which is a form of curriculum (shorter contexts are simpler)
- self-paced learning (Kumar et al. 2010) orders examples by current model loss, training on easy examples first

### comparison

| property | biology | ML training |
|---|---|---|
| ordering mechanism | emergent (maturation + environment) | explicit (designed schedule) |
| what is ordered | sensory complexity + motor repertoire | data difficulty or context length |
| capacity constraint | real (working memory, attention) | simulated (truncation, filtering) |
| is it necessary? | Elman (1993): yes, for some tasks | Bengio (2009): helps convergence, unclear for final performance |

**verdict:** progressive context extension in todorov is a genuine curriculum that has a biological parallel. the MECHANISM is different (explicit schedule vs emergent constraint), but the PRINCIPLE is the same: expose the learner to simple patterns first. the open question is whether more aggressive curriculum (e.g., starting with simple syntactic patterns before complex ones, not just short before long) would help. the evidence from Elman (1993) suggests it might for syntactic learning specifically.

## dimension 3: capacity management

### biology

the brain manages representational capacity through:

- [[synaptic_pruning]]: reducing connectivity by ~50% during development, removing redundant or unused connections
- [[critical_periods|critical period closure]]: locking in stable representations in early-maturing areas while later areas remain plastic
- myelination: increasing processing speed in mature circuits at the cost of structural flexibility
- the net effect: capacity is progressively specialized. the early brain has high capacity and low specificity; the mature brain has lower capacity and high specificity

### ML training

capacity management in ML:

- fixed architecture throughout training (most common): all capacity available from step 1
- progressive growing: starting with a smaller network and adding layers/dimensions during training (Karras et al. 2018 for GANs, but rare for language models)
- post-training pruning: removing parameters after training to reduce inference cost (SparseGPT, lottery ticket hypothesis)
- knowledge distillation: training a smaller network to match a larger network's behavior
- todorov: no capacity management. 267m parameters are all active from step 1 to the last step

### comparison

| property | biology | ML training |
|---|---|---|
| initial capacity | excess (exuberant connectivity) | fixed (designed architecture) |
| reduction | pruning ~50% during development | post-training pruning (optional) |
| timing | concurrent with learning | after learning (separate phase) |
| is it activity-dependent? | yes (use it or lose it) | magnitude-based or structured (not activity-dependent in the biological sense) |
| effect on computation | specialization, faster processing | compression, faster inference |

**verdict:** the most fundamental difference: biology STARTS with excess capacity and REDUCES it during learning. ML FIXES capacity and occasionally reduces it AFTER learning. the biological approach means that early learning has access to more capacity than late learning, which is the opposite of what most ML architectures provide. whether starting with a larger-than-needed network and pruning during training would improve todorov's performance is unknown but would be an interesting experiment.

## dimension 4: timescale

### biology

development spans years, with multiple overlapping processes:

- prenatal: molecular patterning, axon guidance, initial synapse formation (~months)
- postnatal early: spontaneous activity-driven refinement, sensory critical periods (~months to years)
- childhood: continued refinement, language acquisition, myelination (~years)
- adolescence: PFC pruning, executive function maturation (~years)
- adult: ongoing synaptic plasticity at much reduced magnitude (lifetime)

total: ~20 years from initial circuit formation to full maturation.

### ML training

- warmup: ~1000-5000 steps (~minutes)
- main training: ~100K-1M steps (~hours to days)
- fine-tuning (if any): ~1K-10K steps (~minutes)

total: hours to days.

the 10^4 to 10^5 difference in timescale is not just quantitative. it means that biological development has time for multiple overlapping processes with different time constants (fast Hebbian learning, slow homeostatic scaling, very slow myelination, extremely slow pruning). ML training compresses everything into one gradient descent run, which may prevent some forms of organization that require timescale separation.

## the adversarial question: should todorov incorporate developmental staging?

### argument for

1. **progressive context extension already works.** todorov already uses curriculum (256->2048 tokens). extending this principle to other aspects of training (progressive spike activation, layer-specific learning rates, staged architectural features) could yield similar benefits.

2. **phase 5 sequencing IS developmental staging.** the manually specified progression (baseline -> ATMN -> expanded spikes) tests one new feature at a time. this is developmentally principled: you do not introduce all complexity simultaneously. but it operates across runs, not within a single run.

3. **critical learning periods are real in deep networks.** Achille et al. (2019) showed that early training dynamics permanently shape learned representations. this means the early phase of todorov training is disproportionately important, and what happens during warmup may determine the final quality. careful control of early training (simpler data, restricted architecture) could improve final performance.

4. **Elman's starting small principle.** if todorov's spike mechanism or layer interactions create optimization difficulties, starting with a simpler configuration (e.g., no spikes for the first N steps, then gradually introducing them) could help the optimizer find better solutions.

### argument against

1. **no evidence that layer-specific learning rates help language models.** the extensive hyperparameter search literature for transformers has not found consistent benefits from per-layer learning rate scaling. the architecture may not have the layer-specific functional specialization that makes regional critical periods useful in biology.

2. **developmental staging adds hyperparameters.** each staging decision (when to introduce spikes, how to schedule per-layer learning rates, when to start pruning) is an additional hyperparameter that must be tuned. with limited compute budget, the risk of introducing bad staging decisions exceeds the potential benefit.

3. **biology's developmental staging evolved over millions of years.** the specific timing of PV+ maturation, PNN formation, and heterochronous pruning was optimized by evolution for a specific sensory-motor niche. there is no reason to believe that the same timing would be optimal for next-token prediction on text data. importing biological timing without the evolutionary optimization that produced it is cargo-cult neuroscience.

4. **the one thing biology and ML agree on is that early training matters.** but they disagree on WHY. in biology, early training matters because critical periods close. in ML, early training matters because the loss landscape topology is determined early (Achille et al. 2019). the practical implication is the same (be careful with early training data), but the mechanism is different, and the biological mechanism (layer-specific closure) may not apply.

### verdict

progressive context extension is validated and should continue. phase 5 sequencing across runs is methodologically sound. within-run developmental staging (progressive spike activation, layer-specific learning rates, training-time pruning) is a speculative optimization with no strong prior support from either the ML or neuroscience literature. the expected benefit is small and the hyperparameter cost is real. defer to phase 6+ for experimental investigation, if at all.

the one exception: if todorov encounters optimization difficulties at 300m+ scale (loss spikes, spike collapse, training instability), developmental staging (e.g., training without spikes for the first N steps, then gradually introducing them) should be considered as a stabilization technique. this would be an engineering solution motivated by biological intuition, not a principled implementation of developmental neuroscience.

## see also

- [[development_to_training_schedule]] -- bridge article on training-schedule analogies and concrete proposed changes
- [[critical_periods]] -- biological mechanism behind the comparison
- [[PROJECT_PLAN]] -- authoritative current project state

## key references

- Achille, A., Rovere, M. & Soatto, S. (2019). Critical learning periods in deep networks. ICLR 2019.
- Elman, J. L. (1993). Learning and development in neural networks: the importance of starting small. Cognition, 48(1), 71-99.
- Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). Curriculum learning. ICML 2009.
- Huttenlocher, P. R. & Dabholkar, A. S. (1997). Regional differences in synaptogenesis in human cerebral cortex. Journal of Comparative Neurology, 387(2), 167-178.
- Hensch, T. K. (2005). Critical period plasticity in local cortical circuits. Nature Reviews Neuroscience, 6(11), 877-888.
- Karras, T., Aila, T., Laine, S. & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. ICLR 2018.
- Frankle, J. & Carlin, M. (2019). The lottery ticket hypothesis: finding sparse, trainable neural networks. ICLR 2019.
