# bridge: development to model training schedule

status: current (as of 2026-06-18).

this article is about model-training sequencing and staged optimization in the architecture track, which is the project's active lane (the neural-machine architecture program, approach a). it is not about the teaching PDF curriculum, which is preserved as backlog.

## the biological mechanism

biological neural development is a multi-stage process that shapes circuit structure through temporally ordered, regionally specific mechanisms:

1. **[[critical_periods]]:** time windows during which circuits are maximally plastic, triggered by PV+ interneuron maturation and terminated by molecular brakes (PNNs, myelin, epigenetic modifications). different brain regions have different critical period schedules (Hensch 2005, Huttenlocher and Dabholkar 1997).

2. **[[synaptic_pruning]]:** the brain starts with ~2x the adult number of synapses and eliminates ~50% during development through activity-dependent selection (Changeux and Danchin 1976). active synapses survive, inactive synapses are removed. pruning is heterochronous: sensory cortices prune by age 4-6, prefrontal cortex prunes through adolescence.

3. **[[developmental_self_organization]]:** ordered structure (topographic maps, orientation columns, ocular dominance columns) emerges from Hebbian competition plus lateral inhibition, driven first by spontaneous activity (retinal waves) and then by sensory experience. no explicit supervision or template is required.

4. **curriculum of increasing complexity:** emergent rather than designed. limited sensory-motor capacity in infants constrains the complexity of early experience. Elman (1993) showed that this constraint is computationally beneficial: networks that "start small" (limited working memory) learn complex grammar that fully-capable networks fail to learn.

## the current todorov implementation

training uses:
- warmup (gradual LR increase over ~1000 steps) then cosine decay to near-zero
- progressive context extension (256 -> 512 -> 1024 -> 2048 tokens over training)
- spike threshold alpha is learnable from step 1 (no progressive introduction)
- all layers active from step 1 (no staged activation)
- phase 5 sequencing: baseline -> ATMN -> expanded spikes (a form of progressive complexity across runs)

### analogy to critical periods

**warmup = "critical period" for learning (high plasticity)**

the warmup phase is the period when learning rate ramps from near-zero to peak. during this phase, the network transitions from random initialization to its first structured representations. Achille et al. (2019) showed that data deficits during this early phase permanently impair deep networks, analogous to sensory deprivation during biological critical periods.

strength of analogy: MODERATE. both are early time windows where the system is maximally sensitive to input statistics. both produce permanent effects if disrupted.

**cosine decay = closing of the critical period (reduced plasticity)**

as training progresses, the learning rate decays toward zero, progressively reducing the magnitude of weight updates. this is analogous to critical period closure, where molecular brakes reduce synaptic plasticity.

strength of analogy: WEAK. biological critical period closure is STRUCTURAL (PNNs physically constrain synapses, myelin prevents axon sprouting). cosine decay is merely parametric (the same update rule with a smaller coefficient). biological closure is IRREVERSIBLE without intervention (chondroitinase, transplantation). cosine decay is trivially reversible (change the hyperparameter). biological closure is REGIONAL (different areas close at different times). cosine decay is GLOBAL.

**no layer-specific critical periods**

all layers in todorov learn at the same rate throughout training. there is no analog of the heterochronous maturation schedule where V1 matures years before PFC. KDA layers and MLA layers use the same learning rate, the same warmup, the same decay.

this is the weakest point of the analogy. the most important feature of biological critical periods is their REGIONAL SPECIFICITY -- the fact that different circuits have different plasticity windows. a training recipe where KDA layers have a different learning rate schedule than MLA layers would be a closer analog, but there is no evidence this would help.

**no pruning**

todorov does not remove connections during training. the architecture (number of parameters, connectivity) is fixed from initialization. the ternary spike sets 59% of activations to zero on each forward pass, but this is transient and input-dependent -- all weights remain available for all inputs. see [[synaptic_pruning]] for analysis.

**progressive context extension IS curriculum learning**

the 256 -> 512 -> 1024 -> 2048 token progression is a genuine curriculum. shorter contexts are simpler (fewer long-range dependencies, less information to integrate). training on short contexts first allows the model to learn local patterns (token-level and phrase-level statistics) before encountering the full complexity of long-range dependencies. this parallels Elman's (1993) starting-small principle and the biological curriculum of increasing sensory complexity.

## the adversarial question: is spike threshold learning analogous to critical period plasticity?

### the claim

spike threshold alpha is a learnable parameter that adapts the firing threshold to the activation statistics. one might argue that alpha learning during early training constitutes a "critical period" for spike calibration: alpha converges early and then changes slowly, analogous to a critical period that opens and closes.

### the analysis

this claim does not hold under scrutiny, for three reasons:

1. **no closing mechanism.** alpha is learnable throughout training with the same learning rate as all other parameters (modulo cosine decay). there is no mechanism that specifically locks alpha after an early period. if alpha stabilizes, it is because the gradient with respect to alpha becomes small (the parameter is near a local optimum), not because plasticity is structurally terminated. this is qualitatively different from PNN-mediated critical period closure, which physically prevents synaptic rearrangement regardless of the gradient signal.

2. **no layer-specific timing.** if alpha learning constitutes a critical period, every layer should have the same critical period (since they all use the same learning rate schedule). biological critical periods are layer-specific and region-specific. a uniform alpha critical period is not analogous to the heterochronous developmental schedule that makes biological critical periods functionally important.

3. **alpha is a GLOBAL threshold, not a circuit refinement.** biological critical periods refine the fine structure of circuits (which synapses survive, which are pruned, which neurons respond to which inputs). alpha is a single scalar that sets the population firing rate. learning alpha is analogous to calibrating the OVERALL level of inhibition, not refining the SPECIFIC pattern of inhibition. the biological analog of alpha learning is more like the initial maturation of GABAergic inhibition (which sets the overall E/I balance) than like the critical period itself (which refines specific circuit connectivity).

### verdict

alpha learning is NOT analogous to critical period plasticity. it is more accurately analogous to the TRIGGER for the critical period (setting the E/I balance that enables competitive refinement) than to the critical period itself (the competitive refinement that produces circuit-specific organization).

the learning rate schedule (warmup + cosine decay) is the closest analog to a critical period in todorov's training, but it lacks regional specificity and structural closure. see [[development_vs_training]] for the full comparison.

## the proposed change

### option 1: layer-specific learning rate schedules

assign different learning rate schedules to different layer types:
- KDA layers: standard schedule (warmup + cosine decay)
- MLA layers: delayed warmup (start later, decay later), mimicking the late maturation of association cortices
- spike alpha: fast convergence schedule (rapid warmup, early plateau), mimicking the early maturation of inhibitory circuits that triggers the critical period

rationale: the 3:1 KDA:MLA ratio means that KDA layers are the "primary sensory cortex" (doing most of the computation, processing sequential input) while MLA layers are "association cortex" (integrating across the sequence, performing retrieval). biologically, sensory areas mature before association areas.

estimated probability of meaningful BPB improvement: 10-15%. the KDA/MLA functional distinction is a rough analog at best, and the evidence for layer-specific learning rates in language models is weak.

### option 2: progressive spike activation

introduce spikes progressively during training:
- steps 0-N: no ternary spikes (standard continuous activations, higher capacity, easier optimization)
- steps N-2N: linearly interpolate between continuous and ternary (alpha schedule from 0 to 1.0)
- steps 2N+: full ternary spikes

rationale: analogous to the developmental sequence where GABA is initially excitatory (no inhibition), then shifts to inhibitory (establishing the E/I balance that opens the critical period). early training without spikes allows the network to establish a good representation geometry, which spikes then discretize.

estimated probability of meaningful BPB improvement: 20-30%. this directly addresses the known optimization difficulty of the STE (straight-through estimator) gradient, which is biased. letting the network learn in continuous space first and then quantizing could reduce the impact of STE bias on the final solution. however, the network might learn representations that do not quantize well, requiring extensive relearning after spike introduction.

risk: the transition from continuous to ternary may cause a training instability (sudden change in activation statistics). biological development avoids this by making the GABA excitatory-to-inhibitory switch gradual (KCC2 expression increases slowly over development). the linear interpolation schedule is the ML equivalent.

### option 3: training-time pruning

after an initial training phase, remove parameters based on magnitude or activity:
- train for N steps with full architecture
- identify weights below a magnitude threshold or neurons with consistently near-zero activation
- remove these parameters permanently (zero them and freeze)
- continue training with reduced architecture for remaining steps

rationale: directly mimics biological synaptic pruning. the initial overconnected network has high capacity for exploration, and pruning specializes it.

estimated probability of meaningful BPB improvement: 5-10%. the lottery ticket hypothesis evidence for language models is mixed, and structured pruning at 300m parameters saves limited compute relative to the overhead of the pruning decision. the main benefit would be faster inference, not better training.

### option 4: ATMN per-neuron thresholds as developmental specialization

ATMN already provides per-neuron learnable thresholds (V_th_i = exp(a_i)). if ATMN is adopted (phase 5a), the per-neuron thresholds could develop on a schedule:
- early training: all thresholds initialized identically (uniform population)
- during training: each neuron develops its own threshold based on its activation statistics
- late training: thresholds stabilize (analogous to each neuron's critical period closing at its own time)

this would be self-organized (threshold values emerge from training dynamics) rather than externally scheduled. the per-neuron threshold diversity would parallel the diversity of firing thresholds across cortical neurons, which spans a ~10x range in biology.

estimated probability of meaningful BPB improvement: 15-20%, contingent on ATMN being adopted. the per-neuron threshold is the most biologically grounded of the proposed changes, and it creates feature-specific sparsity (different dimensions have different firing rates), which is closer to the cortical code than uniform 41% sparsity.

## implementation spec

### option 2 (recommended first test): progressive spike activation

```
phase 1 (steps 0 to warmup_end):
    spike_alpha_schedule = 0.0 (no spikes, continuous activations)
    learning_rate = linear warmup to peak

phase 2 (steps warmup_end to warmup_end + spike_ramp):
    spike_alpha_schedule = linearly interpolate from 0.0 to 1.0
    learning_rate = peak (or beginning of decay)

phase 3 (steps warmup_end + spike_ramp to end):
    spike_alpha_schedule = 1.0 (full ternary spikes, learnable alpha)
    learning_rate = cosine decay

spike_ramp = 10% of total training steps (e.g., 1000 steps for 10000-step run)
```

implementation: modify the spike function in train.py to accept a schedule parameter that interpolates between identity (no quantization) and full ternary spike. the interpolation is:

```
output = (1 - t) * x + t * spike(x, alpha)
```

where t ramps from 0 to 1 during the spike ramp phase. at t=0, the output is the continuous input. at t=1, the output is the full ternary spike. the STE gradient flows through both branches weighted by (1-t) and t respectively.

lines changed in train.py: ~10 (add schedule parameter to spike function, modify forward pass to use interpolation).

### expected impact

- **convergence speed:** likely faster in early training (no STE bias), potentially slower during spike ramp (adaptation to new activation regime)
- **final BPB:** uncertain. could be better (starting from a better initial representation geometry), worse (representations learned in continuous space may not survive quantization), or unchanged
- **spike health:** spike MI and CKA may differ from current values if the representation geometry is shaped differently by early continuous training. need to monitor throughout

### risk assessment

**risk: spike collapse during ramp.** if the continuous representations are far from the ternary manifold, the transition to spikes could cause many dimensions to cluster near the threshold, producing near-random ternary assignments. mitigation: monitor spike health during ramp; if MI drops below 0.5, slow the ramp.

**risk: alpha divergence.** alpha is learnable from the start of the spike ramp. if the initial continuous representations have unusual scale (much larger or smaller than the ternary regime expects), alpha may diverge. mitigation: clip alpha to [0.1, 10.0] during the spike ramp phase.

**risk: compute overhead.** the interpolation adds one multiply-add per activation during the spike ramp phase (~10% of training). negligible at the current scale.

### comparison to existing approach

current: spikes from step 1, alpha learnable from step 1, 41% firing rate throughout training. MI 1.168, CKA 0.732 at 267m scale.

proposed: no spikes during warmup, linear spike ramp over 10% of training, then standard. expected MI and CKA: comparable or better (continuous early training may establish better-organized representations before quantization).

this is a SINGLE CHANGE (spike scheduling) that can be validated in one run at 6m scale before deploying at 300m. it follows the phase 5 sequencing principle: one variable at a time.

## see also

- [[critical_periods]] -- the biological mechanism this bridge is testing
- [[development_vs_training]] -- comparison article for the same broad theme
- [[PROJECT_PLAN]] -- authoritative current project state and the curriculum pivot

## key references

- Hensch, T. K. (2005). Critical period plasticity in local cortical circuits. Nature Reviews Neuroscience, 6(11), 877-888.
- Elman, J. L. (1993). Learning and development in neural networks: the importance of starting small. Cognition, 48(1), 71-99.
- Achille, A., Rovere, M. & Soatto, S. (2019). Critical learning periods in deep networks. ICLR 2019.
- Changeux, J.-P. & Danchin, A. (1976). Selective stabilisation of developing synapses. Nature, 264(5588), 705-712.
- Huttenlocher, P. R. & Dabholkar, A. S. (1997). Regional differences in synaptogenesis in human cerebral cortex. Journal of Comparative Neurology, 387(2), 167-178.
- Bengio, Y. et al. (2009). Curriculum learning. ICML 2009.
