# full, runnable code here
import random
from collections import deque
from .neural_fabric import NeuralFabric

class RelationalCortex:
    def __init__(self, fabric: NeuralFabric):
        self.fabric = fabric
        self.event_zone = 'general_association'
        # Restore the sliding window buffer
        self.activation_sequence = deque(maxlen=3)
        print("RelationalCortex initialized.")

    def observe_activation(self, pattern: frozenset):
        if pattern and len(pattern) > 0:
            if len(self.activation_sequence) == 0 or self.activation_sequence[-1] != pattern:
                self.activation_sequence.append(pattern)

    def analyze_sequence_for_relation(self):
        if len(self.activation_sequence) < 3:
            return
        subject, verb, obj = list(self.activation_sequence)
        self.create_and_integrate_relation(subject, verb, obj)
        self.activation_sequence.popleft()

    def create_and_integrate_relation(self, subject_pattern, verb_pattern, object_pattern):
        # --- FIX: Use the internal fabric symbols, not human words ---
        subject_symbol = self._get_symbol_for_pattern(subject_pattern)
        verb_symbol = self._get_symbol_for_pattern(verb_pattern)
        object_symbol = self._get_symbol_for_pattern(object_pattern)

        if not all([subject_symbol, verb_symbol, object_symbol]): return

        event_symbol = f"event_{subject_symbol}_{verb_symbol}_{object_symbol}"
        event_pattern = self.fabric.recall(event_symbol)

        if not event_pattern:
            # We don't need verbose logging for every single relation anymore
            event_neurons = list(self.fabric.zones[self.event_zone])
            available_neurons = [n for n in event_neurons if n not in self.fabric.used_event_neurons]
            if len(available_neurons) < 10: return
            event_pattern = set(random.sample(available_neurons, 10))
            self.fabric.used_event_neurons.update(event_pattern)
            self.fabric.bind(event_symbol, event_pattern)

        frozen_event_pattern = frozenset(event_pattern)
        integration_patterns = {subject_pattern, verb_pattern, object_pattern}
        self.fabric.logic.integrate_event_knowledge(frozen_event_pattern, integration_patterns)
        
    def _get_symbol_for_pattern(self, pattern, default=""):
        # This function is now simple, robust, and has no external dependencies.
        return next((s for s, p in self.fabric.symbol_table.items() if p == pattern), None)