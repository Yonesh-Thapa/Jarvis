# full, runnable code here
from collections import deque, Counter
import random

from src.neural_fabric import NeuralFabric

class MemoryCore:
    """
    Orchestrates memory formation, consolidation, and recall within the NeuralFabric.
    It operates by modifying synaptic weights, not by storing data itself.
    """
    def __init__(self, fabric: NeuralFabric, consolidation_threshold: int = 3):
        """
        Initializes the Memory Core.

        Args:
            fabric (NeuralFabric): The shared neural fabric instance.
            consolidation_threshold (int): How many times a pattern must be observed
                                           in the recent past to be consolidated.
        """
        self.fabric = fabric
        
        # A buffer of recently observed firing patterns (sets of UIDs).
        # This acts as a short-term memory.
        self.short_term_memory = deque(maxlen=100)
        
        # A list of consolidated memory traces (patterns).
        # In a real brain, this is implicitly stored, but we track it explicitly
        # to enable dreaming and other meta-operations.
        self.consolidated_patterns = []
        
        self.consolidation_threshold = consolidation_threshold
        print("MemoryCore initialized.")

    def observe(self, fired_uids: set):
        """
        Observes the latest set of fired neurons and adds it to short-term memory.
        This should be called after every simulation step.
        """
        if len(fired_uids) > 1: # Ignore trivial activations
            # We freeze the set to make it hashable for the Counter.
            self.short_term_memory.append(frozenset(fired_uids))

    def consolidate(self):
        """
        Scans short-term memory for salient patterns and strengthens their connections.
        This is the core of memory formation (learning).
        """
        if not self.short_term_memory:
            return

        # --- Principle: Identify salient patterns ---
        # Find patterns that have appeared frequently in our recent history.
        pattern_counts = Counter(self.short_term_memory)
        
        salient_patterns = [
            p for p, count in pattern_counts.items() 
            if count >= self.consolidation_threshold
        ]

        if not salient_patterns:
            return

        print(f"INFO: Found {len(salient_patterns)} salient patterns to consolidate.")
        for pattern in salient_patterns:
            self._strengthen_pattern(pattern)
            if pattern not in self.consolidated_patterns:
                self.consolidated_patterns.append(pattern)
        
        # Clear the buffer after consolidation to prevent re-consolidating the same events.
        self.short_term_memory.clear()

    def _strengthen_pattern(self, pattern_uids: frozenset):
        """
        Internal helper to strengthen synapses within a given pattern.
        This creates a "cell assembly" where neurons are tightly bound.
        """
        # --- Principle: Neurons that fire together, wire together ---
        neuron_list = list(pattern_uids)
        # For every pair of neurons in the pattern, create/strengthen a synapse.
        for i in range(len(neuron_list)):
            for j in range(i + 1, len(neuron_list)):
                source_uid = neuron_list[i]
                target_uid = neuron_list[j]
                
                # Create bidirectional connections to form a strong, auto-associative memory.
                self.fabric.connect_neurons(source_uid, target_uid, weight=0.7)
                self.fabric.connect_neurons(target_uid, source_uid, weight=0.7)
        
        print(f"  - Consolidated pattern of size {len(pattern_uids)}")


    def recall(self, cue_uids: set, activation_strength: float = 0.6) -> set:
        """
        Given a partial cue, attempts to activate the full memory pattern.
        
        Args:
            cue_uids (set): A small subset of neurons from a memory pattern.
            activation_strength (float): The signal strength to propagate to associated neurons.
            
        Returns:
            set: The set of neurons that were excited by the recall mechanism.
        """
        if not cue_uids:
            return set()
            
        excited_neurons = set()
        # For each neuron in the cue...
        for uid in cue_uids:
            # ...find all neurons it has a strong connection to.
            for target_uid, synapse in self.fabric.synapses.get(uid, {}).items():
                if synapse.weight > 0.5: # Threshold for a 'strong' connection
                    self.fabric.neurons[target_uid].receive_signal(activation_strength)
                    excited_neurons.add(target_uid)
        
        print(f"INFO: Recall cue of size {len(cue_uids)} excited {len(excited_neurons)} neurons.")
        return excited_neurons
        
    def dream(self):
        """
        Replays a random consolidated memory to further strengthen it.
        This is an offline, low-power consolidation mechanism.
        """
        if not self.consolidated_patterns:
            return
            
        # Pick a random memory to replay.
        dream_pattern = random.choice(self.consolidated_patterns)
        
        print(f"INFO: Dreaming... replaying a memory of size {len(dream_pattern)}.")
        
        # Activate the pattern in the fabric. The subsequent `step_simulation`
        # will trigger Hebbian learning, further reinforcing the synapses.
        self.fabric.activate_pattern(dream_pattern, signal_strength=1.1)

    def prune_synapses(self, prune_threshold=0.05, prune_factor=0.99):
        """
        Weakens all synapses slightly and removes very weak ones.
        This is the 'forgetting' mechanism.
        """
        pruned_count = 0
        synapses_to_prune = []

        for source_uid, targets in self.fabric.synapses.items():
            for target_uid, synapse in targets.items():
                # Decay the weight
                synapse.weight *= prune_factor
                # Mark for deletion if below threshold
                if synapse.weight < prune_threshold:
                    synapses_to_prune.append((source_uid, target_uid))

        for source_uid, target_uid in synapses_to_prune:
            del self.fabric.synapses[source_uid][target_uid]
            pruned_count += 1
            
        if pruned_count > 0:
            print(f"INFO: Pruned {pruned_count} weak synapses.")