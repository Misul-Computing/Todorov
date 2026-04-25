# memory systems research library

status: current (as of 2026-04-25).

curated research on how biological memory works at the cognitive and systems level. organized by domain. this library complements [[neuroscience_research_2026]], which covers cellular and circuit-level mechanisms. the present article covers the higher-level architecture of memory: how memories are formed, stored, retrieved, transformed, and lost.

## spatial memory and the method of loci

### the art of memory

yates, f. a. (1966). *the art of memory*. routledge & kegan paul.

key insight: the method of loci -- mentally placing items along a familiar spatial route -- has been used since classical antiquity to achieve extraordinary feats of memorization. the technique exploits the hippocampal spatial navigation system to scaffold non-spatial memory, converting arbitrary sequences into spatial trajectories that the brain is already optimized to encode and retrieve.

relevance to neural computer: demonstrates that spatial indexing is the brain's native addressing scheme. an architecture that uses spatial structure (e.g., [[place_cells]], [[grid_cells]], rotational position encoding) for memory addressing is exploiting the same computational primitive that makes the method of loci work.

### moonwalking with einstein

foer, j. (2011). *moonwalking with einstein: the art and science of remembering everything*. penguin press.

key insight: competitive memorizers encode information by constructing vivid, bizarre spatial narratives through memory palaces. foer, a journalist with average memory, trained for one year and won the u.s. memory championship. the technique is not innate talent but a learnable strategy that converts serial recall into spatial navigation plus associative binding.

relevance to neural computer: underscores that the bottleneck in biological memory is not storage capacity but encoding strategy. the brain has vast capacity that is unlocked by routing through the spatial system. this suggests that the key architectural choice is not how much state to carry but how to address and retrieve from it.

### navigation-related structural change in the hippocampi of taxi drivers

maguire, e. a., gadian, d. g., johnsrude, i. s., good, c. d., ashburner, j., frackowiak, r. s. j., & frith, c. d. (2000). navigation-related structural change in the hippocampi of taxi drivers. *proceedings of the national academy of sciences*, 97(8), 4398-4403. doi: 10.1073/pnas.070039597

key insight: london taxi drivers who spent years learning "the knowledge" (navigating 25,000 streets) show significantly enlarged posterior hippocampi compared to controls. the volume correlates with years of experience. this is structural neuroplasticity driven by spatial memory demands -- the brain physically reallocates tissue to the spatial indexing system under sustained load.

relevance to neural computer: provides direct evidence that the spatial memory system scales with demand and that the hippocampus is the physical substrate. an architecture that dedicates more parameters to its spatial/positional encoding system under heavier memory loads would be implementing the same principle.

### mnemonic training reshapes brain networks to support superior memory

dresler, m., shirer, w. r., konrad, b. n., muller, n. c. j., wagner, i. c., fernandez, g., czisch, m., & greicius, m. d. (2017). mnemonic training reshapes brain networks to support superior memory. *neuron*, 93(5), 1227-1235. doi: 10.1016/j.neuron.2017.02.003

key insight: 40 days of method-of-loci training in naive subjects doubled free recall of word lists (from ~26 to ~62 words) and restructured functional brain connectivity to resemble that of world-class memorizers. the critical network changes were increased connectivity between medial prefrontal cortex and hippocampal/parahippocampal regions -- the same default mode network nodes used for spatial navigation and episodic simulation.

relevance to neural computer: the training-induced connectivity changes show that memory improvement comes from better routing between encoding and retrieval systems, not from adding storage. this parallels how kda's delta rule improves with better gating of what gets written to state, not with larger state matrices.

### long-lasting improvements in memory following method of loci training

engel de abreu, p. m. j., buss, c., pietto, m. l., kaminski, a., barchfeld, p., & lipina, s. j. (2021). long-lasting improvements in memory following method of loci training. *science advances*. doi: 10.1126/sciadv.abj0437

key insight: method-of-loci improvements persist for at least 4 months after training ends, with maintained gains in both word recall and pattern recognition. this is not a short-term strategy effect but a durable restructuring of memory processes. the persistence suggests that the training induced lasting changes in how information is encoded, not just retrieval tricks.

relevance to neural computer: durable improvement from spatial encoding training implies that once the spatial addressing pathway is established, it continues to benefit non-spatial memory tasks. for todorov, this suggests that [[positional_encoding_to_rope]] may be doing more than sequence ordering -- it may be providing the addressing substrate for kda's content-addressable memory.

## episodic vs semantic memory

### episodic and semantic memory

tulving, e. (1972). episodic and semantic memory. in e. tulving & w. donaldson (eds.), *organization of memory* (pp. 381-403). academic press.

tulving, e. (1983). *elements of episodic memory*. oxford university press.

key insight: memory is not one system. episodic memory records personal experiences (what happened, where, when) and is tied to autonoetic consciousness -- the subjective sense of re-experiencing the past. semantic memory stores facts, concepts, and general knowledge without experiential context. the two systems have different encoding requirements, retrieval mechanisms, and neural substrates.

relevance to neural computer: the episodic/semantic distinction maps to a fundamental architectural choice. kda's delta rule state accumulates token-by-token history (episodic-like), while mla's compressed key-value cache stores factual associations stripped of positional context (semantic-like). the two-system architecture is not an accident -- it reflects a real computational division. see [[memory_systems_to_matrix_memory_and_compressed_attention]].

### multiple trace theory

nadel, l., & moscovitch, m. (1997). memory consolidation, retrograde amnesia and the hippocampal complex. *current opinion in neurobiology*, 7(2), 217-227. doi: 10.1016/S0959-4388(97)80010-4

key insight: contra standard consolidation theory (which holds that the hippocampus is a temporary store), multiple trace theory argues that the hippocampus is permanently required for vivid episodic recall. each retrieval creates a new memory trace, so frequently retrieved memories have many hippocampal traces (making them robust) while rarely retrieved ones have few (making them fragile). semantic memories emerge when many overlapping traces extract the gist.

relevance to neural computer: retrieval as trace creation means that every read from memory is also a write -- consistent with kda's delta rule, which updates state at every timestep. the mechanism also suggests that repeated patterns should naturally consolidate into compressed representations, which is what the kda-to-mla pathway achieves across layers.

### trace transformation theory

moscovitch, m., cabeza, r., winocur, g., & nadel, l. (2016). episodic memory and beyond: the hippocampus and neocortex in transformation. *annual review of psychology*, 67, 105-134. doi: 10.1146/annurev-psych-113011-143733

key insight: memories are not simply transferred from hippocampus to neocortex -- they are transformed. detailed, context-rich hippocampal traces are gradually transformed into schematic, gist-based neocortical representations. both the detailed and schematic versions can coexist, and which one is retrieved depends on task demands. transformation is an active process, not passive decay.

relevance to neural computer: the transformation model maps to how information flows through todorov's layer stack. early layers (kda) maintain detailed, context-rich state. later layers (mla) compress into schematic representations suitable for prediction. the 3:1 kda-to-mla ratio may reflect the biological observation that detailed representation requires more resources than schematic extraction.

### complementary learning systems

mcclelland, j. l., mcnaughton, b. l., & o'reilly, r. c. (1995). why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. *psychological review*, 102(3), 419-457. doi: 10.1037/0033-295X.102.3.419

key insight: the brain requires two learning systems with opposite properties. the hippocampus learns quickly from single episodes (fast binding, sparse representations, pattern separation) to avoid catastrophic interference. the neocortex learns slowly across many episodes (gradual weight change, distributed representations, statistical regularities) to extract structure. interleaved replay during sleep bridges the two rates.

relevance to neural computer: the cls framework is the most influential theory mapping to hybrid architectures. see [[complementary_learning_systems]] for the mechanism and [[memory_systems_to_matrix_memory_and_compressed_attention]] for the bridge analysis. note: the bridge article documents that todorov's kda+mla is NOT a direct implementation of cls -- both systems learn at the same rate during training. the analogy is structural (fast recurrence + compressed cache), not temporal (fast learning + slow learning).

### opposing representations colocated in hippocampal subfield ca1

sherman, b. e., turk-browne, n. b., & goldfarb, e. v. (2024). opposing representations colocated in hippocampal subfield ca1. *perspectives on psychological science*. doi: 10.1177/17456916241258258

key insight: hippocampal ca1 simultaneously maintains both pattern-separated (distinct) and pattern-completed (overlapping) representations of similar memories. these opposing codes coexist in the same neural population, with different subpopulations carrying each type. this resolves the long-standing puzzle of how the hippocampus can both discriminate similar memories (separation) and fill in missing details (completion).

relevance to neural computer: dual coding in the same population is directly relevant to kda's delta rule state, which must both distinguish similar input patterns (write different keys) and retrieve from partial cues (match approximate queries). the coexistence of separation and completion in ca1 suggests this is not a design tension but a feature. see [[hippocampal_memory]], [[pattern_completion]].

## working memory

### the magical number seven, plus or minus two

miller, g. a. (1956). the magical number seven, plus or minus two: some limits on our capacity for processing information. *psychological review*, 63(2), 81-97. doi: 10.1037/h0043158

key insight: immediate memory span is approximately 7 items (range 5-9), but the critical unit is the chunk, not the individual item. by recoding information into larger chunks (e.g., grouping digits into phone number segments), the number of items within reach can be expanded without increasing the number of simultaneously active representations.

relevance to neural computer: the chunk is the natural unit of working memory, and chunking is a form of compression. todorov's ternary spikes ({-1, 0, +1}) are an extreme compression that reduces each activation to ~1.58 bits. if each spike pattern represents a chunk rather than a raw feature, the effective information capacity of a layer is much higher than the bit count suggests.

### the magical number 4 in short-term memory

cowan, n. (2001). the magical number 4 in short-term memory: a reconsideration of mental storage capacity. *behavioral and brain sciences*, 24(1), 87-114. doi: 10.1017/S0140525X01003922

key insight: when rehearsal and chunking are controlled for, true working memory capacity is approximately 4 items, not 7. miller's larger estimate confounded storage capacity with strategic recoding. the 4-item limit appears to be a fundamental architectural constraint, possibly reflecting the number of independent pointers the attentional system can maintain.

relevance to neural computer: a 4-pointer limit suggests that working memory is not a buffer but an index -- a small set of pointers into a much larger store. this is consistent with [[hippocampal_memory]]'s indexing theory and with how mla maintains a small number of latent query heads that index into compressed key-value representations.

### working memory model

baddeley, a. d., & hitch, g. (1974). working memory. in g. h. bower (ed.), *the psychology of learning and motivation* (vol. 8, pp. 47-89). academic press.

baddeley, a. (2000). the episodic buffer: a new component of working memory? *trends in cognitive sciences*, 4(11), 417-423. doi: 10.1016/S1364-6613(00)01538-2

key insight: working memory is not a single store but a multi-component system: a phonological loop (verbal rehearsal), a visuospatial sketchpad (spatial/visual maintenance), a central executive (attentional control), and an episodic buffer (multimodal integration with long-term memory). the components are modality-specific buffers controlled by a domain-general executive.

relevance to neural computer: the multi-component architecture parallels how todorov uses different layer types for different aspects of working memory. kda's matrix-valued state acts as a content buffer, mamba3's continuous dynamics act as a temporal buffer, and the residual stream acts as the central executive (routing information between components). the episodic buffer's role as a binding interface maps to how mla integrates information across the other two layer types.

### long-term working memory

ericsson, k. a., & kintsch, w. (1995). long-term working memory. *psychological review*, 102(2), 211-245. doi: 10.1037/0033-295X.102.2.211

key insight: experts appear to exceed the 4-chunk working memory limit because they maintain retrieval structures in long-term memory that can be rapidly accessed. chess masters, medical diagnosticians, and expert readers do not hold more in working memory -- they have highly practiced retrieval cues that make relevant long-term memory functionally equivalent to working memory. the key is not larger buffers but faster, more reliable retrieval from the existing store.

relevance to neural computer: this reframes the working memory bottleneck. an architecture does not need a larger state matrix to have better working memory -- it needs better retrieval. kda's delta rule is a retrieval mechanism (content-addressable via key-value matching), and its effectiveness determines how much of the accumulated state is functionally available at each timestep.

## memory reconsolidation

### reconsolidation of memory

nader, k., schafe, g. e., & ledoux, j. e. (2000). fear memories require protein synthesis in the lateral amygdala for reconsolidation after retrieval. *nature*, 406, 722-726. doi: 10.1038/35021052

key insight: when a consolidated memory is retrieved, it returns to a labile state and requires new protein synthesis to restabilize (reconsolidate). if protein synthesis is blocked during this reconsolidation window (~6 hours), the memory is weakened or erased. this overturned the classical view that consolidated memories are permanent and immutable.

relevance to neural computer: every retrieval is a potential write. this is precisely what kda's delta rule implements -- reading from state and updating state happen in the same operation. the reconsolidation framework suggests this is not a design compromise but a feature: it allows memories to be updated with new information each time they are accessed.

### preventing the return of fear in humans using reconsolidation update mechanisms

schiller, d., monfils, m. h., raio, c. m., johnson, d. c., ledoux, j. e., & phelps, e. a. (2010). preventing the return of fear in humans using reconsolidation update mechanisms. *nature*, 463, 49-53. doi: 10.1038/nature08637

key insight: fear memories can be permanently updated (not just suppressed) by presenting new non-threatening information during the reconsolidation window. unlike standard extinction (which creates a competing memory that can relapse), reconsolidation-based updating modifies the original trace. the timing window is critical: new information must arrive within ~6 hours of retrieval to trigger reconsolidation rather than new encoding.

relevance to neural computer: demonstrates that memory update during retrieval is more powerful than memory suppression. for kda, this implies that the delta rule's error-correcting write (which modifies existing state based on prediction error) is computationally superior to approaches that simply add competing entries, because it modifies the original representation rather than creating interference.

### prediction error governs pharmacologically induced amnesia for learned motor movements

sevenster, d., beckers, t., & kindt, m. (2014). prediction error governs pharmacologically induced amnesia for learned motor movements. *science*. doi: 10.1126/science.1245592

key insight: reconsolidation is not triggered by retrieval alone -- it requires prediction error. when a retrieved memory matches expectations perfectly, it is re-stabilized without modification. only when the retrieval context includes a mismatch (surprise, prediction error) does the memory become labile and open to updating. prediction error is the gate that determines whether a retrieved memory is modified.

relevance to neural computer: prediction error as the gate for memory update is exactly what the delta rule computes. kda writes to state proportional to the error between the predicted and actual value, which means that well-predicted patterns leave state unchanged (no reconsolidation) while surprising patterns trigger state updates (reconsolidation). the biological mechanism and the mathematical operation align.

note on replication: human reconsolidation replication has been inconsistent (klucken et al. 2016, golkar et al. 2017, luyten et al. 2022). the boundary conditions for triggering reconsolidation in humans remain debated. the core finding in rodents is well-established; the translational gap is real.

## expertise and chunking

### thought and choice in chess

de groot, a. d. (1946/1978). *thought and choice in chess*. mouton publishers.

key insight: chess masters can reconstruct a briefly shown game position almost perfectly (~93% accuracy), while novices recall ~33%. but when pieces are placed randomly, masters and novices perform equally poorly. expertise is not better general memory but a library of ~50,000-100,000 stored chunk patterns that enable rapid recognition and compressed encoding of meaningful configurations.

relevance to neural computer: this is the original evidence that memory capacity depends on learned compression. todorov's ternary spikes implement a form of learned chunking -- the spike pattern is a compressed representation of the input, and what makes it useful is not its bit depth but its structure relative to the training distribution. random inputs would not compress well, just as random chess positions do not benefit from expertise.

### perception in chess

chase, w. g., & simon, h. a. (1973). perception in chess. *cognitive psychology*, 4(1), 55-81. doi: 10.1016/0010-0285(73)90004-2

key insight: masters encode chess positions in chunks of 4-5 pieces with specific spatial and functional relationships. the chunk is the unit of perception, memory, and planning. chunked perception is faster (masters glance at the board and "see" the structure) and more efficient (one chunk replaces several independent items in working memory). the 50,000-100,000 chunk estimate for master-level play has been replicated and refined.

relevance to neural computer: chunk-based perception is template matching with learned templates -- precisely what kda's key-value retrieval implements. the key vector is the chunk signature, and the value vector is the associated information. the delta rule's error-correcting write is how new chunks are formed: when a pattern is not well-matched by existing keys, the prediction error drives a state update that creates a new entry.

### neural efficiency hypothesis

key insight: experts performing domain-relevant tasks show lower neural activation (fewer active neurons, lower bold signal) than novices performing the same tasks at the same accuracy. expertise produces more efficient representations that require less computational resource to achieve equivalent or superior performance.

relevance to neural computer: neural efficiency maps to todorov's spike sparsity. the 41% firing rate is the trained system's level of efficiency. if the architecture is working correctly, spike health metrics (mi, cka) should improve during training as the representations become more efficient -- more information per active spike, fewer spikes needed per representation.

## prospective memory and mental time travel

### remembering

bartlett, f. c. (1932). *remembering: a study in experimental and social psychology*. cambridge university press.

key insight: memory is reconstructive, not reproductive. bartlett's "war of the ghosts" experiment showed that recall systematically distorts stories to fit the rememberer's cultural schemas -- details are omitted, added, or transformed to make the narrative more coherent. memory is not a recording that degrades; it is a constructive process that generates plausible reconstructions from stored fragments.

relevance to neural computer: reconstructive memory means that the output of memory retrieval is a generated representation, not a stored one. this is exactly what autoregressive language modeling does -- the model does not retrieve stored text but generates plausible continuations from compressed internal state. the "errors" of human memory (schema-driven distortions) are features of a generative system, not bugs of a storage system.

### the evolution of foresight

suddendorf, t., & corballis, m. c. (2007). the evolution of foresight: what is mental time travel, and is it unique to humans? *behavioral and brain sciences*, 30(3), 299-313. doi: 10.1017/S0140525X07001975

key insight: episodic memory and future simulation (mental time travel) are two applications of the same neural system. the hippocampus and default mode network are active both when remembering past events and when imagining future scenarios. memory did not evolve for accurate recording of the past -- it evolved for flexible simulation of possible futures, and remembering is a byproduct.

relevance to neural computer: if memory is fundamentally a simulation engine, then the goal of a memory architecture is not faithful storage but useful generation. kda's state does not need to perfectly preserve every past input -- it needs to maintain sufficient structure to support accurate predictions about what comes next. this reframes state decay not as information loss but as compression toward prediction-relevant features.

### remembering the past to imagine the future

addis, d. r., wong, a. t., & schacter, d. l. (2007). remembering the past to imagine the future: the prospective brain. *nature reviews neuroscience*, 8, 657-661. doi: 10.1038/nrn2213

key insight: fmri studies show that remembering past events and imagining future events activate the same core network: hippocampus, medial prefrontal cortex, posterior cingulate, lateral temporal cortex, and lateral parietal cortex. the overlap is not merely anatomical -- patients with hippocampal amnesia are impaired at both remembering the past and imagining the future. the constructive episodic simulation hypothesis holds that future thinking requires recombining elements of past experience.

relevance to neural computer: the shared substrate for memory and generation validates the autoregressive architecture's dual use of context (past tokens) for both comprehension (understanding what was said) and generation (predicting what comes next). the same state that encodes the past is the substrate for generating the future.

### adaptive constructive processes and the future of memory

schacter, d. l. (2012). adaptive constructive processes and the future of memory. *american psychologist*, 67(8), 603-613. doi: 10.1037/a0029869

key insight: the "seven sins of memory" (transience, absent-mindedness, blocking, misattribution, suggestibility, bias, persistence) are not design flaws but byproducts of a system optimized for simulation and prediction. transience (forgetting) discards irrelevant details to extract gist. misattribution and suggestibility allow flexible recombination of elements across episodes. the errors reveal the system's priorities: generalization over verbatim accuracy.

relevance to neural computer: this is the strongest argument that lossy compression in memory is not a limitation to be engineered around but a feature to be embraced. kda's state matrix will lose information over time (the delta rule overwrites old entries with new ones), and this is correct behavior for a prediction-oriented system. the relevant metric is not state fidelity but prediction accuracy.

## memory across species

### episodic-like memory in food-caching birds

clayton, n. s., & dickinson, a. (1998). episodic-like memory in food-caching birds. *nature*, 395, 272-274. doi: 10.1038/26216

key insight: western scrub jays remember what food they cached, where they cached it, and when they cached it (what-where-when). they preferentially retrieve perishable food (worms) before non-perishable food (nuts) when the interval is short, but reverse this preference when enough time has passed for worms to have decayed. this demonstrates episodic-like memory in non-mammalian species, indicating that the computational problem (binding events to spatiotemporal context) drives convergent solutions across distant lineages.

relevance to neural computer: what-where-when binding is a content-addressable memory operation -- retrieve by composite key. kda's key vectors must encode not just content (what) but positional context (where/when via rope). the cross-species convergence suggests this is not a biological quirk but a fundamental requirement for any memory system that must track time-varying, spatially organized information.

### cuttlefish episodic-like memory

jozet-alves, c., bertin, m., & clayton, n. s. (2013). evidence of episodic-like memory in cuttlefish. *current biology*, 23(23), R1033-R1036. doi: 10.1016/j.cub.2013.10.021

key insight: cuttlefish demonstrate episodic-like what-where-when memory despite having no hippocampus and a fundamentally different brain organization from vertebrates. they update their foraging preferences based on remembered past experiences with specific food types at specific locations. this shows that episodic memory is a computational capability, not a hippocampal property -- it can be implemented by any architecture that supports context-bound associative retrieval.

relevance to neural computer: the strongest evidence that the computational function (context-bound associative retrieval) is separable from the biological substrate (hippocampus). todorov does not need to replicate hippocampal anatomy -- it needs to replicate the function: binding content to context and retrieving by partial cue. kda's delta rule + rope achieves this with completely different mechanisms.

### hippocampal replay during sleep

wilson, m. a., & mcnaughton, b. l. (1994). reactivation of hippocampal ensemble memories during sleep: evidence for offline sequence compression. *science*, 265(5172), 676-679. doi: 10.1126/science.8036517

key insight: hippocampal place cell firing patterns that occurred during waking experience are reactivated during subsequent slow-wave sleep, but compressed in time (replay at ~5-20x speed). the temporal sequence of cell activations is preserved, maintaining the spatial trajectory information. this offline replay is thought to support memory consolidation by re-presenting experience to the neocortex for gradual integration.

relevance to neural computer: replay is compressed re-presentation of sequential experience for consolidation. in todorov, the training loop itself serves this function -- each training batch re-presents compressed sequences for weight updates. the question is whether an online analog of replay (re-processing stored state during inference) would benefit long-context performance. see [[memory_consolidation]].

### hippocampal preplay of future trajectories

dragoi, g., & tonegawa, s. (2011). preplay of future place cell sequences by hippocampal cellular assemblies. *nature*, 469, 397-401. doi: 10.1038/nature09633

key insight: before a rat ever runs through a novel track, hippocampal place cells already fire in sequences that correspond to the spatial layout the rat will later traverse. this "preplay" occurs during rest periods and precedes any experience of the environment. the hippocampus does not merely record experience -- it generates candidate trajectories in advance, possibly supporting planning and prediction.

relevance to neural computer: preplay is generative prediction from the memory system -- the hippocampus simulating possible futures before they happen. this aligns with [[the_brain_in_one_page]]'s view that the brain is fundamentally a prediction engine. the autoregressive generation process in language models (predicting the next token from accumulated context) is computationally analogous: generating a trajectory through token space from a state that encodes prior structure.

### hippocampus as key-value content-addressable memory

gershman, s. j., fiete, i. r., & irie, k. (2025). the hippocampus as a key-value content-addressable memory. *neuron*. doi: 10.1016/j.neuron.2025.01.001

key insight: the hippocampus can be formally modeled as a content-addressable memory (cam) that stores key-value pairs, where the key is a contextual pattern (spatial, temporal, or abstract) and the value is the associated memory content. retrieval is by partial key match (pattern completion). the cam formalism unifies diverse hippocampal functions -- spatial navigation, episodic recall, relational inference -- under a single computational description.

relevance to neural computer: this is the most direct theoretical connection between hippocampal memory and kda. the delta rule's matrix-valued state is literally a key-value store updated by error correction. the cam formalism validates the choice of content-addressable memory as the core computational primitive and provides a theoretical framework for analyzing kda's capacity, interference, and retrieval dynamics. see [[hippocampal_memory]], [[memory_systems_to_matrix_memory_and_compressed_attention]].

## key architectural concepts

### hippocampal indexing theory

teyler, t. j., & rudy, j. w. (2007). the hippocampal indexing theory and episodic memory: updating the index. *hippocampus*, 17(12), 1158-1169. doi: 10.1002/hipo.20350

the hippocampus stores pointers (indices) to neocortical representations, not the content itself. retrieval involves reactivating the index, which then reinstates the distributed neocortical pattern. this explains why hippocampal damage impairs episodic retrieval (the index is lost) but not semantic knowledge (the neocortical content is preserved). the index is small, fast to write, and content-addressable; the content is large, slow to consolidate, and distributed.

### schacter's seven sins of memory

schacter, d. l. (2001). *the seven sins of memory: how the mind forgets and remembers*. houghton mifflin.

the seven sins (transience, absent-mindedness, blocking, misattribution, suggestibility, bias, persistence) reveal that memory errors are features of a constructive system optimized for future simulation, not bugs of a storage system. transience implements automatic garbage collection. misattribution enables creative recombination. suggestibility allows social updating of beliefs.

### every retrieval is a potential write

the reconsolidation literature (nader et al. 2000, sevenster et al. 2014) establishes that retrieved memories return to a labile state and can be modified. this means the memory system has no pure read operation -- every read is a read-modify-write. kda's delta rule implements exactly this: every query against state produces an output and updates the state.

### memory and imagination share one system

the prospective memory literature (suddendorf & corballis 2007, addis et al. 2007, schacter 2012) establishes that remembering and imagining activate the same neural substrate. memory did not evolve for accurate recording but for flexible simulation. the autoregressive prediction objective aligns with this: the model's memory system exists to support generation, and generation quality is the measure of memory quality.

## relevance to todorov

the memory systems literature provides several key validations and challenges for todorov's architecture:

**validated design choices:**
- kda's delta rule as content-addressable memory is directly supported by gershman et al. 2025, which formalizes the hippocampus as a key-value cam with error-correcting updates
- the read-modify-write cycle (every retrieval updates state) matches the reconsolidation literature, where prediction error gates memory modification
- the 3:1 kda-to-mla ratio maps to the asymmetry between detailed episodic traces (kda, hippocampal-like) and compressed semantic representations (mla, neocortical-like), though the bridge analysis ([[memory_systems_to_matrix_memory_and_compressed_attention]]) documents that the analogy is structural, not temporal
- ternary spike compression as chunking is consistent with the expertise literature: the value of compressed representations depends on their learned structure, not their bit depth
- rope as spatial addressing parallels the method of loci's exploitation of spatial indexing for non-spatial memory

**challenges and open questions:**
- todorov lacks replay -- no mechanism re-presents stored state during inference for consolidation. the cls framework (mcclelland et al. 1995) and the replay literature (wilson & mcnaughton 1994) suggest this may limit long-context performance
- the 4-item working memory limit (cowan 2001) raises the question of how many independent "pointers" kda maintains per timestep. the number of effective retrieval slots is determined by the number of query heads, which may impose a cowan-like bottleneck
- dual coding in ca1 (sherman et al. 2024) suggests that the same state matrix should support both pattern separation and pattern completion. whether kda's delta rule achieves this dual function or sacrifices one for the other is untested
- preplay (dragoi & tonegawa 2011) suggests that a mature memory system should generate candidate trajectories from learned structure even without input. todorov has no offline generation mode during inference

**implications for future phases:**
- phase 6+ could explore replay-based consolidation during inference (re-processing compressed state to improve retrieval)
- the working memory bottleneck could be investigated by varying the number of kda query heads and measuring effective context utilization
- the separation/completion duality could be tested by probing kda state with similar vs dissimilar queries and measuring discrimination vs generalization

## see also

- [[hippocampal_memory]]
- [[memory_consolidation]]
- [[complementary_learning_systems]]
- [[pattern_completion]]
- [[memory_systems_to_matrix_memory_and_compressed_attention]]
- [[place_cells]]
- [[grid_cells]]
- [[compression_beyond_quantization]]
