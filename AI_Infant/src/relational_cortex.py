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
        """
        Creates a new "event" pattern that represents a relationship between
        three other patterns (subject, verb, object).
        """
        s_sym = self._get_symbol_for_pattern(subject_pattern, "subject")
        v_sym = self._get_symbol_for_pattern(verb_pattern, "verb")
        o_sym = self._get_symbol_for_pattern(object_pattern, "object")
        
        if not all([s_sym, v_sym, o_sym]): return

        event_symbol = f"event_{s_sym}_{v_sym}_{o_sym}"
        event_pattern = self.fabric.recall(event_symbol)
        
        if not event_pattern:
            # Verbose logging only for brand new events
            print(f"--- Creating Relation: ({s_sym}) -> [{v_sym}] -> ({o_sym}) ---")
            
            event_neurons = list(self.fabric.zones[self.event_zone])
            available_neurons = [n for n in event_neurons if n not in self.fabric.used_event_neurons]
            if len(available_neurons) < 10: return None
            
            event_pattern = set(random.sample(available_neurons, 10))
            self.fabric.used_event_neurons.update(event_pattern)
            self.fabric.bind(event_symbol, event_pattern)
        
        frozen_event_pattern = frozenset(event_pattern)
        integration_patterns = {subject_pattern, verb_pattern, object_pattern}
        self.fabric.logic.integrate_event_knowledge(frozen_event_pattern, integration_patterns)
        return frozen_event_pattern
        
    # --- START OF FINAL FIX ---
    def _get_symbol_for_pattern(self, pattern: frozenset, default: str = "") -> str:
        """
        The authoritative function to translate a neural pattern back to its
        human-readable word or symbol.
        """
        if not pattern: return default

        # The language cortex holds the definitive map from word -> pattern. Check here first.
        # This is much more reliable than iterating through the entire symbol table.
        if self.fabric.language:
            for word, p_map in self.fabric.language.word_to_pattern_map.items():
                if p_map == pattern:
                    return word

        # Fallback for non-word symbols like events, goals, etc.
        symbol = next((s for s, p in self.fabric.symbol_table.items() if p == pattern and not s.startswith("word_")), None)
        if symbol: 
            return symbol
            
        return f"concept_{hash(pattern)}"
    # --- END OF FINAL FIX ---