# full, runnable code here
from .neural_fabric import NeuralFabric
from .memory_core import MemoryCore
from collections import deque

class LogicCortex:
    def __init__(self, fabric: NeuralFabric, memory_core: MemoryCore):
        self.fabric = fabric
        self.memory = memory_core
        self.abstract_categories = {}
        print("LogicCortex initialized.")

    # --- New helper function for robust symbol resolution ---
    def _resolve_symbol_to_pattern(self, symbol: str) -> frozenset | None:
        """
        Intelligently finds the neural pattern for any given symbol,
        whether it's a word, an event, or another abstract concept.
        """
        # First, check for known abstract prefixes
        if symbol.startswith(("event_", "goal_", "meta_", "op_")):
            return self.fabric.recall(symbol)
        
        # If no prefix, assume it's a word and ask the LanguageCortex
        pattern, _ = self.fabric.language._get_or_create_pattern_for_word(symbol)
        if pattern:
            return pattern
            
        # As a final fallback, check the main fabric table again
        return self.fabric.recall(symbol)

    def bind_symbol_to_pattern(self, symbol: str, pattern_uids: set):
        self.fabric.bind(symbol, pattern_uids)

    def associate_concepts(self, new_symbol: str, existing_symbols: list, context_pattern: set):
        # ... This method is unchanged
        self.bind_symbol_to_pattern(new_symbol, context_pattern)
        all_involved_neurons = set(context_pattern)
        for symbol in existing_symbols:
            pattern = self._resolve_symbol_to_pattern(symbol)
            if pattern: all_involved_neurons.update(pattern)
        self.memory._strengthen_pattern(frozenset(all_involved_neurons))

    def integrate_event_knowledge(self, event_pattern: frozenset, component_patterns: set):
        # ... This method is unchanged
        all_involved_neurons = set(event_pattern)
        for component in component_patterns: all_involved_neurons.update(component)
        self.memory._strengthen_pattern(all_involved_neurons)

    # --- START OF FINAL FIX: QUERY USES THE RESOLVER ---
    def query_association(self, symbol_a: str, symbol_b: str) -> float:
        pattern_a = self._resolve_symbol_to_pattern(symbol_a)
        pattern_b = self._resolve_symbol_to_pattern(symbol_b)
            
        if not pattern_a or not pattern_b:
            print(f"QUERY_FAIL: One or both symbols ('{symbol_a}', '{symbol_b}') are unknown.")
            return 0.0

        # Activate the first pattern as the cue
        self.fabric.activate_pattern(pattern_a, 1.1)
        self.fabric.step_simulation() # Let the cue fire
        
        # See what downstream neurons fire in the next step
        downstream_activations = self.fabric.step_simulation()
        
        intersection = len(pattern_b.intersection(downstream_activations))
        return intersection / len(pattern_b) if len(pattern_b) > 0 else 0.0
    # --- END OF FINAL FIX ---

    def perform_inference(self, start_word: str):
        # This method is now correct and does not need changes.
        print(f"\n--- Performing Inference starting from '{start_word}' ---")
        start_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(start_word)
        if not start_pattern:
            print(f"  - INFERENCE_FAIL: Cannot start, word '{start_word}' is unknown."); return
        all_events = {s: p for s, p in self.fabric.symbol_table.items() if s.startswith("event_")}
        inferred_facts = 0
        for event_symbol, _ in all_events.items():
            try: _, subject_sym, verb_sym, object_sym = event_symbol.split('_', 3)
            except ValueError: continue
            if subject_sym == start_word:
                for sub_event_symbol, _ in all_events.items():
                    try: _, sub_subject_sym, sub_verb_sym, sub_object_sym = sub_event_symbol.split('_', 3)
                    except ValueError: continue
                    if sub_subject_sym == object_sym and verb_sym == sub_verb_sym:
                        print(f"  - CHAIN FOUND: ({start_word}) -> [{verb_sym}] -> ({object_sym}) AND ({sub_subject_sym}) -> [{sub_verb_sym}] -> ({sub_object_sym})")
                        final_object_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(sub_object_sym)
                        verb_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(verb_sym)
                        if final_object_pattern and verb_pattern:
                            inferred_event_symbol = f"event_{start_word}_{verb_sym}_{sub_object_sym}"
                            if not self.fabric.recall(inferred_event_symbol):
                                print(f"  - SYNTHESIZING new knowledge: '{start_word} {verb_sym} {sub_object_sym}'")
                                self.fabric.relation.create_and_integrate_relation(
                                    start_pattern, verb_pattern, final_object_pattern
                                )
                                inferred_facts += 1
        print(f"--- Inference Complete. Synthesized {inferred_facts} new facts. ---")