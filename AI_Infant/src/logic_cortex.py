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

    def _resolve_symbol_to_pattern(self, symbol: str) -> frozenset | None:
        if symbol.startswith(("event_", "goal_", "meta_", "op_")):
            return self.fabric.recall(symbol)
        pattern, _ = self.fabric.language._get_or_create_pattern_for_word(symbol)
        if pattern: return pattern
        return self.fabric.recall(symbol)

    def integrate_textual_knowledge(self, main_idea_pattern: frozenset, property_patterns: set):
        if not main_idea_pattern or not property_patterns: return
        main_idea_symbol = self.fabric.relation._get_symbol_for_pattern(main_idea_pattern)
        print(f"\n--- Integrating knowledge about '{main_idea_symbol}' ---")
        
        all_involved_neurons = set(main_idea_pattern)
        for prop_pattern in property_patterns:
            if prop_pattern != main_idea_pattern:
                all_involved_neurons.update(prop_pattern)
        
        frozen_pattern = frozenset(all_involved_neurons)
        self.memory._strengthen_pattern(frozen_pattern)
        
        # --- THE FIX: This learning event is important. Remember it. ---
        # We observe it multiple times to ensure it crosses the consolidation threshold.
        print(f"  - Committing integrated knowledge of size {len(frozen_pattern)} to short-term memory.")
        for _ in range(self.memory.consolidation_threshold):
            self.memory.observe(frozen_pattern)

    def integrate_event_knowledge(self, event_pattern: frozenset, component_patterns: set):
        if not event_pattern or not component_patterns: return
        
        all_involved_neurons = set(event_pattern)
        for component in component_patterns:
            all_involved_neurons.update(component)
        
        frozen_pattern = frozenset(all_involved_neurons)
        self.memory._strengthen_pattern(frozen_pattern)
        print(f"  - Integrating event with its {len(component_patterns)} components.")
        
        # --- THE FIX: This new relationship is important. Remember it. ---
        self.memory.observe(frozen_pattern)

    # All other methods below this point are unchanged and correct.
    def bind_symbol_to_pattern(self, symbol: str, pattern_uids: set):
        self.fabric.bind(symbol, pattern_uids)

    def associate_concepts(self, new_symbol: str, existing_symbols: list, context_pattern: set):
        self.bind_symbol_to_pattern(new_symbol, context_pattern)
        all_involved_neurons = frozenset(context_pattern)
        for symbol in existing_symbols:
            pattern = self._resolve_symbol_to_pattern(symbol)
            if pattern: all_involved_neurons = all_involved_neurons.union(pattern)
        self.memory._strengthen_pattern(all_involved_neurons)
        self.memory.observe(all_involved_neurons)

    def query_association(self, symbol_a: str, symbol_b: str) -> float:
        pattern_a = self._resolve_symbol_to_pattern(symbol_a)
        pattern_b = self._resolve_symbol_to_pattern(symbol_b)
        if not pattern_a or not pattern_b:
            print(f"QUERY_FAIL: One or both symbols ('{symbol_a}', '{symbol_b}') are unknown.")
            return 0.0
        self.fabric.activate_pattern(pattern_a, 1.1)
        self.fabric.step_simulation()
        downstream_activations = self.fabric.step_simulation()
        intersection = len(pattern_b.intersection(downstream_activations))
        return intersection / len(pattern_b) if len(pattern_b) > 0 else 0.0

    def perform_inference(self, start_word: str):
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