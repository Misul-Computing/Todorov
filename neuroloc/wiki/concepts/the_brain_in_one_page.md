# the brain in one page

status: definitional. last fact-checked 2026-04-16.

this is the 80/20 overview. it covers the 20% of neuroscience that gives you 80% of the understanding you need to read this wiki and design brain-inspired architectures.

## the hardware

the human brain has about 86 billion neurons connected by roughly 100 trillion synapses. it runs on about 20 watts -- less than a laptop charger. for comparison, a 300M parameter transformer has 300 million "neurons" and uses about 200 watts per GPU. the brain is 286 times larger and 10 times more power-efficient.

this efficiency gap is not about cleverness in software. it is about physics. biological computation uses 10 femtojoules per synaptic operation. silicon uses 4.6 picojoules per floating-point multiply at 45nm (and 0.3 picojoules at 5nm). ternary operations cost about 0.001 picojoules at 5nm -- closing the gap by 200-350x. see [[biological_vs_silicon_energy]] for the full analysis.

## the neuron

a neuron is a cell that receives electrical inputs, integrates them over time, and fires a spike if the total exceeds a threshold.

think of a ReLU unit in a neural network. it takes a weighted sum of inputs and outputs zero if the sum is negative, or the sum itself if positive. a biological neuron does something similar but with two critical differences. first, it has a temporal state called the **membrane potential** (the voltage across the cell membrane, typically around -65 millivolts at rest). inputs accumulate in this state over time, like a leaky bucket that fills with each input and slowly drains between inputs. second, its output is all-or-nothing: when the membrane potential crosses a threshold, the neuron fires a **spike** (also called an **action potential**), a brief ~1ms electrical pulse. then the membrane potential resets and the neuron is briefly unable to fire again. see [[leaky_integrate_and_fire]] for the simplest mathematical model.

the spike is the fundamental unit of neural communication. it is binary: fire or not fire. the neuron does not output a continuous value like a transformer activation. information is carried by which neurons spike, when they spike, and how often they spike. in todorov, ternary spikes {-1, 0, +1} extend this to three states: inhibit, silent, or excite.

a neuron has three main parts. The **dendrites** (tree-like branches) receive inputs from other neurons. The **soma** (cell body) integrates these inputs and decides whether to spike. The **axon** (long cable) carries the spike to other neurons. Recent research shows that dendrites are not passive cables -- they perform their own local computations, acting like a small neural network within the neuron. see [[dendritic_computation]] and [[two_layer_neuron]].

for ML engineers, the key takeaway: a biological neuron is an RNN cell with binary output. it has persistent state (membrane potential), a nonlinear threshold (spike), and temporal dynamics (leak, refractory period). this is fundamentally different from a feedforward ReLU unit, which has no memory of previous inputs.

## the synapse

a **synapse** is the connection point between two neurons. one neuron (presynaptic) sends a spike down its axon. when the spike arrives at the synapse, it triggers the release of chemical **neurotransmitters** (signaling molecules). these cross a tiny gap and bind to receptors on the receiving neuron (postsynaptic), causing a change in its membrane potential.

in ML terms, a synapse is a weight in a weight matrix. but biological weights are different in two ways. first, they change based on activity during inference, not just during training. **short-term plasticity** (lasting milliseconds to minutes) alters synaptic strength based on recent firing patterns. a synapse can get temporarily weaker (depression) or stronger (facilitation) depending on how fast the presynaptic neuron has been firing. see [[short_term_plasticity]]. second, long-term weight changes follow local rules, not global backpropagation. The most important is **Hebbian learning**: "neurons that fire together wire together." if neuron A repeatedly causes neuron B to fire, the synapse from A to B gets stronger. mathematically, this is an outer product: Delta_w = eta * x_pre * x_post. see [[hebbian_learning]] for details.

synapses come in two types. **Excitatory** synapses (using glutamate as their neurotransmitter) push the postsynaptic neuron toward firing. **Inhibitory** synapses (using GABA) push it away from firing. a neuron is either excitatory or inhibitory -- it cannot switch. this is called Dale's law.

## the cortex

the **cerebral cortex** is the thin (2-4 mm) wrinkled sheet of neurons covering the brain. it handles perception, language, planning, motor control, and most of what we call cognition. about 80% of cortical neurons are excitatory and 20% are inhibitory. this 80/20 ratio is remarkably consistent across brain regions and species.

the cortex is organized in six layers (L1-L6), stacked from the surface inward. each layer has different cell types and connectivity patterns. layer 4 receives sensory input from the thalamus. layers 2/3 process and relay information between cortical areas. layer 5 sends output to subcortical structures. layer 6 sends feedback to the thalamus. this layered architecture is fundamentally different from transformer layers, which are functionally identical -- see [[cortical_layers_vs_todorov_layers]] for the detailed comparison.

neurons in the cortex are organized in **cortical columns** (vertical groups of neurons spanning all 6 layers). neurons within a column tend to respond to similar features. the column is sometimes called the "repeating unit" of cortical computation, though this analogy is debated. see [[cortical_column]].

the cortex also has a repeating circuit motif called the **canonical microcircuit** (Douglas & Martin 1989). it consists of a recurrent excitatory loop between layers, stabilized by inhibitory neurons. this circuit amplifies weak inputs (like a gain control) while keeping activity bounded. see [[canonical_microcircuit]].

## how information flows

sensory information enters the brain through specialized receptors (eyes, ears, skin). it passes through the **thalamus** (a relay station that also gates and filters signals) and reaches primary sensory cortex (V1 for vision, A1 for hearing). from there it flows through a hierarchy of cortical areas, from simple feature detectors to complex object representations.

the brain is bidirectional. feedforward connections carry sensory data upward through the hierarchy. feedback connections carry predictions and contextual information downward. the **predictive coding** framework (Rao & Ballard 1999) proposes that feedforward connections carry prediction errors, not raw data. only the surprise -- the difference between what was predicted and what was received -- propagates upward. see [[predictive_coding]].

in a transformer, the residual stream is the sole channel for inter-layer communication. information flows forward through the layers and accumulates additively. there is no feedback, no prediction error, and no gating by the thalamus. the residual stream is a shared bus, not a predictive hierarchy. see [[global_workspace_to_residual_stream]] for the full analysis.

## what makes the brain different from a transformer

**sparsity.** at any given moment, only about 1-5% of cortical neurons are active. the rest are silent. this extreme sparsity is an energy constraint: each spike costs energy, so the brain minimizes the number of spikes. in a dense transformer layer, 100% of units are active for every input. see [[sparse_coding]] and [[brain_energy_budget]].

**communication.** neurons communicate with spikes -- discrete, all-or-nothing events. transformers use continuous-valued activations. spikes carry information in their timing and their identity (which neuron fired), not in their amplitude. this forces a different kind of information encoding. see [[population_coding]].

**memory.** synaptic weights change during processing, not just during training. short-term plasticity adapts synapses on millisecond timescales. the brain effectively rewrites its program while running. transformer weights are fixed at inference time. see [[short_term_plasticity]] and [[memory_systems_to_matrix_memory_and_compressed_attention]].

**learning.** the brain uses local learning rules. each synapse adjusts based on the activity of its two connected neurons, plus a global neuromodulatory signal (like dopamine for reward). there is no backpropagation of error gradients through the entire network. see [[plasticity_local_vs_global]].

**power.** 20 watts for 86 billion neurons versus 200+ watts per GPU for 300 million parameters. the gap is real but narrowing: ternary operations on modern silicon approach biological efficiency per operation. the bottleneck is data movement, not compute. see [[biological_vs_silicon_energy]].

## see also

- [[neuroscience_for_ml_engineers]] -- deeper treatment of each concept introduced here
- [[glossary]] -- definitions of every technical term in this wiki
- [[INDEX]] -- full catalog of all mechanism, bridge, and comparison articles
