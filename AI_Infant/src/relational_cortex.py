# full, runnable code here
import random
from collections import deque
from .neural_fabric import NeuralFabric

class RelationalCortex:
    def __init__(self, fabric: NeuralFabric):
        self.fabric = fabric
        self.event_zone = 'general_association'
        print("RelationalCortex initialized.")

    def create_and_integrate_relation(self, subject_pattern, verb_pattern, object_pattern):
        subject_symbol = self._get_symbol_for_pattern(subject_pattern, "S")
        verb_symbol = self._get_symbol_for_pattern(verb_pattern, "V")
        object_symbol = self._get_symbol_for_pattern(object_pattern, "O")

        # Use the raw word for the event name for better readability
        event_symbol = f"event_{subject_symbol}_{verb_symbol}_{object_symbol}"
        event_pattern = self.fabric.recall(event_symbol)

        if not event_pattern:
            print(f"\n--- Creating Relation: ({subject_symbol}) -> [{verb_symbol}] -> ({object_symbol}) ---")
            event_neurons = list(self.fabric.zones[self.event_zone])
            available_neurons = [n for n in event_neurons if n not in self.fabric.used_event_neurons]
            if len(available_neurons) < 10:
                print("RELATION_FAIL: Not enough neurons in event zone."); return
            
            event_pattern = set(random.sample(available_neurons, 10))
            self.fabric.used_event_neurons.update(event_pattern)
            self.fabric.bind(event_symbol, event_pattern)
            print(f"  - Created new event pattern '{event_symbol}'.")
        else:
            print(f"\n--- Reinforcing Relation: ({subject_symbol}) -> [{verb_symbol}] -> ({object_symbol}) ---")

        integration_patterns = {subject_pattern, verb_pattern, object_pattern}
        self.fabric.logic.integrate_event_knowledge(frozenset(event_pattern), integration_patterns)
        
    def _get_symbol_for_pattern(self, pattern, default=""):
        # Helper to find a human-readable symbol for a pattern.
        # This now becomes more important for debugging.
        for word, p_map in self.fabric.language.word_to_pattern_map.items():
            if p_map == pattern:
                return word
        
        # Fallback for non-word patterns
        symbol = next((s for s, p in self.fabric.symbol_table.items() if p == pattern), None)
        if symbol: return symbol
        
        return f"concept_{default}"