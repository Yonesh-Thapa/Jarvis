from collections import deque, Counter
import random
from src.neural_fabric import NeuralFabric

class MemoryCore:
    def __init__(self, fabric: NeuralFabric, consolidation_threshold: int = 3):
        self.fabric = fabric
        self.short_term_memory = deque(maxlen=100)
        self.consolidated_patterns = []
        self.consolidation_threshold = consolidation_threshold
        print("MemoryCore initialized.")

    def observe(self, fired_uids: set):
        if len(fired_uids) > 1:
            self.short_term_memory.append(frozenset(fired_uids))

    def consolidate(self):
        if not self.short_term_memory: return
        
        pattern_counts = Counter(self.short_term_memory)
        salient_patterns = [p for p, count in pattern_counts.items() if count >= self.consolidation_threshold]

        if not salient_patterns:
            self.short_term_memory.clear()
            return

        print(f"  - Consolidating {len(salient_patterns)} salient patterns into long-term memory.")
        for pattern in salient_patterns:
            self._strengthen_pattern(pattern)
            if pattern not in self.consolidated_patterns:
                self.consolidated_patterns.append(pattern)
        
        self.short_term_memory.clear()

    def _strengthen_pattern(self, pattern_uids: frozenset):
        neuron_list = list(pattern_uids)
        for i in range(len(neuron_list)):
            for j in range(i + 1, len(neuron_list)):
                self.fabric.connect_neurons(neuron_list[i], neuron_list[j], weight=0.7)
                self.fabric.connect_neurons(neuron_list[j], neuron_list[i], weight=0.7)
        # print(f"    - Consolidated pattern of size {len(pattern_uids)}")

    def recognize_pattern(self, current_pattern: set, threshold: float = 0.7) -> frozenset | None:
        if not self.consolidated_patterns or len(current_pattern) < 3: return None
        best_match, max_similarity = None, 0.0
        for known_pattern in self.consolidated_patterns:
            intersection = len(current_pattern.intersection(known_pattern))
            union = len(current_pattern.union(known_pattern))
            if union == 0: continue
            similarity = intersection / union
            if similarity > max_similarity:
                max_similarity, best_match = similarity, known_pattern
        return best_match if max_similarity > threshold else None

    def recall(self, cue_uids: set, activation_strength: float = 0.6) -> set:
        if not cue_uids: return set()
        excited_neurons = set()
        for uid in cue_uids:
            for target_uid, synapse in self.fabric.synapses.get(uid, {}).items():
                if synapse.weight > 0.5:
                    self.fabric.neurons[target_uid].receive_signal(activation_strength)
                    excited_neurons.add(target_uid)
        return excited_neurons
        
    def dream(self):
        if not self.consolidated_patterns: return
        dream_pattern = random.choice(self.consolidated_patterns)
        print(f"  - Dreaming... replaying a memory of size {len(dream_pattern)}.")
        self.fabric.activate_pattern(dream_pattern, signal_strength=1.1)

    def prune_synapses(self, prune_threshold=0.05, prune_factor=0.99):
        pruned_count = 0
        synapses_to_prune = []
        with self.fabric.synapse_lock:
            for source_uid, targets in self.fabric.synapses.items():
                for target_uid, synapse in list(targets.items()): # Use list to avoid runtime dict changes
                    synapse.weight *= prune_factor
                    if synapse.weight < prune_threshold:
                        synapses_to_prune.append((source_uid, target_uid))
        
        for source_uid, target_uid in synapses_to_prune:
            if source_uid in self.fabric.synapses and target_uid in self.fabric.synapses[source_uid]:
                del self.fabric.synapses[source_uid][target_uid]
                pruned_count += 1
        
        if pruned_count > 0: 
            print(f"  - Pruned {pruned_count} weak synaptic connections.")