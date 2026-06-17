# three-factor learning rules

status: definitional. last fact-checked 2026-04-16.

**why this matters**: three-factor learning rules solve the credit assignment problem that pure [[hebbian_learning]] cannot: how does a synapse know whether its recent activity was useful for the organism? by adding a global modulatory signal (dopamine, norepinephrine, acetylcholine) as a third factor that gates local synaptic changes, these rules bridge unsupervised correlation detection and reward-based learning. in ML terms, the third factor is the biological implementation of the error/reward signal in reinforcement learning. for todorov, three-factor rules represent the most biologically plausible path toward replacing the straight-through estimator (STE) for spike gradient computation, because they provide a mechanism for credit assignment that respects the locality constraints of spiking networks.

## summary

three-factor learning rules extend classical two-factor [[hebbian_learning]] by conditioning synaptic modification on a global modulatory signal. the general form is:

    Delta_w = eta * f(pre, post) * M(t)

where f(pre, post) is the local two-factor term (correlation between presynaptic and postsynaptic activity, as in Hebb's rule or [[stdp]]), and M(t) is the **third factor** (a global neuromodulatory signal -- typically dopamine, norepinephrine, or acetylcholine -- that broadcasts information about reward, surprise, or behavioral relevance). the key insight is that local synaptic activity creates a temporary **eligibility trace** (a biochemical "tag" marking recently co-active synapses), which is converted into a permanent weight change only when the modulatory signal arrives. ML analog: the eligibility trace is the biological equivalent of a gradient buffer -- the gradient is computed locally and stored, but the weight update is deferred until a global signal (reward, loss) determines its sign and magnitude.

## the credit assignment problem

the fundamental problem motivating three-factor rules is **credit assignment** (determining which of the thousands of synaptic changes occurring at any moment were responsible for a behavioral outcome). two-factor Hebbian rules capture statistical regularities in neural activity but have no mechanism to evaluate whether those regularities are task-relevant. a synapse that strengthens because of correlated firing has no way to know whether that correlation led to reward or punishment. backpropagation solves this in artificial networks by propagating error signals backward through the network, but this requires non-local information (the error gradient at downstream layers), which has no obvious biological implementation. three-factor rules offer a different solution: keep the learning rule local, but gate it with a single global broadcast signal that carries evaluative information. note that stabilization mechanisms like [[bcm_theory]] and [[homeostatic_plasticity]] address a different problem (preventing runaway weights) but are complementary: three-factor rules determine *which* synapses to modify, while homeostatic mechanisms ensure the overall network remains stable.

## mechanism

### the eligibility trace

the eligibility trace is the central concept. when pre- and postsynaptic neurons are co-active (satisfying the two-factor condition), the synapse enters an **eligible** state -- a transient biochemical modification that does not yet constitute a permanent weight change. this trace persists for a time window of roughly 0.3 to 2 seconds, during which it can be converted into lasting LTP or LTD by the arrival of a neuromodulatory signal. if no modulatory signal arrives within the window, the trace decays and no lasting change occurs.

mathematically, the eligibility trace e(t) evolves as:

    de/dt = -e/tau_e + f(pre, post)

where tau_e is the trace time constant (300 ms to 2 s) and f(pre, post) is the instantaneous two-factor term (e.g., the STDP-derived spike correlation from [[stdp]]). the weight update is then:

    dw/dt = e(t) * M(t)

ML analog: this is equivalent to computing the gradient (e), storing it in a buffer, and applying the update only when a reward signal (M) arrives. the time constant tau_e determines the maximum delay between action and reward that the system can bridge.

### the neuromodulatory signal

the third factor M(t) is carried by **neuromodulatory systems** (diffuse projections from subcortical nuclei that release neurotransmitters broadly across cortical areas, modulating neural activity on timescales of hundreds of milliseconds to seconds):

- **dopamine** (from the ventral tegmental area and substantia nigra): encodes **reward prediction error** (RPE -- the difference between received and expected reward). phasic dopamine bursts signal positive RPE; dips below baseline signal negative RPE (Schultz et al. 1997). see [[dopamine_system]] for the full circuit. ML analog: this is the temporal-difference error delta in TD learning, or the advantage signal in PPO.
- **norepinephrine** (from the locus coeruleus): signals surprise, arousal, and uncertainty. broadly enhances synaptic plasticity and may act as a global "learning rate" modulator. see [[neuromodulatory_framework]].
- **acetylcholine** (from the basal forebrain): modulates attention and encoding vs. retrieval balance in hippocampal circuits. high acetylcholine favors new learning; low acetylcholine favors recall.

### the neoHebbian framework

Gerstner et al. (2018) formalized three-factor learning as the **neoHebbian** framework:

    Delta_w = [eligibility_trace] * [neuromodulatory_signal] * [postsynaptic_factor]

where the postsynaptic factor accounts for cell-type-specific and dendritic-compartment-specific modulation. this framework unifies reward-modulated STDP, dopamine-gated plasticity, and attentional gating under a single mathematical structure. the key claim is that three-factor rules are sufficient to solve temporal credit assignment in recurrent spiking networks, provided the eligibility trace has an appropriate time constant and the modulatory signal carries task-relevant information.

## biological evidence

### Yagishita et al. (2014)

the most direct evidence for eligibility traces comes from two-photon imaging of dendritic spine dynamics in medium spiny neurons of the striatum. Yagishita et al. showed that pairing pre- and postsynaptic activity (glutamate uncaging + postsynaptic depolarization) with phasic dopamine application produced lasting spine enlargement (a structural correlate of LTP) only when dopamine arrived within a ~0.3 to 2 second window after the pairing. dopamine arriving before the pairing or more than 2 seconds after produced no lasting change. this established that the eligibility trace has a well-defined temporal window and that dopamine acts as a retroactive gate, not a permissive signal.

### He et al. (2015)

He et al. demonstrated a similar mechanism in cortical synapses, showing that norepinephrine and dopamine can retroactively convert subthreshold synaptic changes into lasting plasticity within a seconds-long window. this extended the eligibility trace concept beyond the striatum to cortical circuits, suggesting it is a general principle of mammalian synaptic plasticity.

### further evidence

multiple lines of evidence converge on the eligibility trace as a real biochemical entity rather than a theoretical convenience. in the hippocampus, **synaptic tagging and capture** (STC) experiments (Frey and Morris 1997) showed that a weak stimulation can produce lasting LTP if a strong stimulation occurs at a nearby synapse within 1-2 hours, suggesting the weak stimulus sets a "tag" that is converted by plasticity-related proteins synthesized in response to the strong stimulus. while the STC timescale (hours) is longer than the dopamine-gated trace (seconds), the conceptual structure is identical: local activity marks the synapse, a global signal converts the mark into a permanent change. in invertebrate systems, Cassenaer and Laurent (2012) demonstrated that odor-reward associations in the mushroom body of locusts depend on a precisely timed dopaminergic reinforcement signal that gates [[stdp]]-like plasticity, providing evidence for three-factor rules in invertebrate systems as well.

## the locality advantage

the central appeal of three-factor rules is **locality**: each synapse computes its update using only information available at its own location (presynaptic activity, postsynaptic activity) plus a single scalar broadcast. contrast this with backpropagation, which requires each synapse to know the error gradient at every downstream layer -- information that must be propagated backward through the entire network along the same pathways used for forward computation. no biological mechanism for this symmetric backward pass has been convincingly demonstrated. three-factor rules sidestep the problem entirely: the modulatory broadcast carries a compressed, scalar summary of the network's performance, and the eligibility trace provides the local direction for the update. ML analog: this is analogous to the difference between full gradient computation (backpropagation) and zeroth-order/policy-gradient methods that estimate the gradient from scalar reward signals. the tradeoff is variance: three-factor updates are noisier than exact gradients but require no non-local computation.

## relationship to reinforcement learning

three-factor rules have a deep formal connection to policy gradient methods in RL. in the REINFORCE algorithm (Williams, 1992), the weight update is:

    Delta_w = eta * R * nabla_w log pi(a|s)

where R is the reward and nabla_w log pi(a|s) is the log-probability gradient of the selected action. the log-probability gradient depends only on local information (the input to the neuron and its output), while R is a global scalar broadcast. this is exactly the three-factor structure: nabla_w log pi = f(pre, post) is the eligibility trace, and R = M is the modulatory signal. the formal equivalence was established by several groups (Xie and Seung 2004, Pfister et al. 2006, Fremaux et al. 2010) and shows that three-factor Hebbian rules can implement policy gradient RL in spiking networks.

## alternatives to backpropagation

### e-prop (Bellec et al. 2020)

**e-prop** (eligibility propagation) is the most developed three-factor alternative to backpropagation through time (BPTT) for training recurrent spiking networks. e-prop maintains per-synapse eligibility traces that approximate the gradients computed by BPTT, but using only locally available information plus a top-down learning signal. on speech recognition and other sequential tasks, e-prop achieves performance close to BPTT-trained LSTMs while respecting biological locality constraints. ML analog: e-prop replaces the full backward pass of BPTT with forward-computed eligibility traces, making it an online, local learning rule -- conceptually similar to real-time recurrent learning (RTRL) but with O(1) per-synapse memory instead of O(N^2).

### RFLO

**RFLO** (random feedback local online learning) combines random feedback alignment (replacing the transpose weight matrix in backpropagation with fixed random projections) with eligibility traces, achieving another biologically plausible approximation to gradient descent. the key idea is that the top-down error signal does not need to travel through the exact transpose of the forward weights -- random fixed feedback connections suffice to convey directional information, and the eligibility trace provides the synapse-specific modulation. this further relaxes the biological implausibility of backpropagation by removing the weight transport requirement.

## relationship to todorov

in the current todorov architecture, the ternary spike function uses the straight-through estimator (STE) to pass gradients through the non-differentiable quantization. the data-dependent gate beta_t = sigmoid(beta_proj(x)) modulates the KDA delta rule write strength, functioning as a two-factor gate (it depends on the input x, not on a reward or error signal). this is not a three-factor rule: there is no eligibility trace and no delayed modulatory signal.

however, three-factor rules represent a potential path for replacing the STE in spike gradient computation. the **sparsity-gradient paradox** (the problem that ternary spikes kill gradients for ~60% of activations, forcing reliance on the biologically implausible STE) could be addressed by an eligibility-trace mechanism that accumulates local spike statistics and converts them into weight updates via a global error signal, bypassing the need for a differentiable surrogate gradient entirely. this remains speculative and unvalidated.

## challenges

1. **the timescale mismatch problem.** biological eligibility traces persist for 0.3 to 2 seconds, but many real-world credit assignment problems require bridging delays of minutes, hours, or longer. dopamine RPE signals can only retroactively gate synaptic changes within the trace window. how the brain assigns credit over longer timescales -- whether through hierarchical decomposition, working memory, or some other mechanism -- is unresolved, and three-factor rules alone cannot explain it.

2. **the specificity problem.** neuromodulatory signals are broadcast globally or semi-globally, but different synapses in different circuits need different error signals. a single scalar dopamine RPE cannot carry the per-synapse gradient information that backpropagation provides. proposals for addressing this include cell-type-specific receptor expression (different neurons respond differently to the same dopamine signal) and spatial gradients in neuromodulator concentration, but whether these mechanisms provide sufficient specificity for complex tasks is unclear.

3. **performance gap with backpropagation.** e-prop and related three-factor methods approach but do not match the performance of BPTT on standard benchmarks. the approximation error introduced by replacing exact gradients with eligibility traces is nonzero, and it accumulates over long sequences. whether this gap is fundamental or can be closed with better trace dynamics is an open question.

4. **computational cost of online eligibility traces.** maintaining per-synapse eligibility traces in hardware or simulation adds memory proportional to the number of synapses, not just the number of neurons. for large-scale models, this overhead may be prohibitive unless the traces can be efficiently compressed or approximated.

## see also

- `wiki/tests/affect_gated_write_cpu_experiment.md` — a project use of the third-factor rule: surprise as the modulatory signal gating the descent memory's write
- [[hebbian_learning]]
- [[stdp]]
- [[dopamine_system]]
- [[neuromodulatory_framework]]
- [[bcm_theory]]
- [[homeostatic_plasticity]]
- [[serotonin_system]]
- [[basal_ganglia]]

## key references

- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D. & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. Frontiers in Neural Circuits, 12, 53.
- Yagishita, S., Hayashi-Takagi, A., Ellis-Davies, G. C. R., Urakubo, H., Ishii, S. & Kasai, H. (2014). A critical time window for dopamine actions on the structural plasticity of dendritic spines. Science, 345(6204), 1616-1620.
- He, K., Huertas, M., Hong, S. Z., Tie, X., Hell, J. W., Shouval, H. & Kirkwood, A. (2015). Distinct eligibility traces for LTP and LTD in cortical synapses. Neuron, 88(3), 528-538.
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R. & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11, 3625.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4), 229-256.
- Fremaux, N., Sprekeler, H. & Gerstner, W. (2010). Functional requirements for reward-modulated spike-timing-dependent plasticity. Journal of Neuroscience, 30(40), 13326-13337.
- Pfister, J.-P., Toyoizumi, T., Barber, D. & Gerstner, W. (2006). Optimal spike-timing-dependent plasticity for precise action potential firing in supervised learning. Neural Computation, 18(6), 1318-1348.
- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593-1599.
