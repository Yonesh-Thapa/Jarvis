# full, runnable code here
from .neural_fabric import NeuralFabric
from .memory_core import MemoryCore

class LogicCortex:
    def __init__(self, fabric: NeuralFabric, memory_core: MemoryCore):
        self.fabric = fabric
        self.memory = memory_core

    def _resolve_symbol_to_pattern(self, symbol: str) -> frozenset | None:
        if symbol.startswith(("event_", "goal_", "meta_", "op_")):
            return self.fabric.recall(symbol)
        pattern, _ = self.fabric.language._get_or_create_pattern_for_word(symbol)
        if pattern: return pattern
        return self.fabric.recall(symbol)

    def integrate_event_knowledge(self, event_pattern: frozenset, component_patterns: set):
        if not event_pattern or not component_patterns: return
        all_involved = set(event_pattern)
        for component in component_patterns: all_involved.update(component)
        frozen_pattern = frozenset(all_involved)
        self.memory._strengthen_pattern(frozen_pattern)
        self.memory.observe(frozen_pattern)

    def query_association(self, symbol_a: str, symbol_b: str) -> float:
        pattern_a = self._resolve_symbol_to_pattern(symbol_a)
        pattern_b = self._resolve_symbol_to_pattern(symbol_b)
        if not pattern_a or not pattern_b: return 0.0
        self.fabric.activate_pattern(pattern_a, 1.1)
        self.fabric.step_simulation()
        downstream = self.fabric.step_simulation()
        intersection = len(pattern_b.intersection(downstream))
        return intersection / len(pattern_b) if len(pattern_b) > 0 else 0.0

    # --- START OF FINAL FIX: Robust Inference Engine ---
    def perform_inference(self, start_word: str):
        print(f"\n--- Performing Inference on '{start_word}' ---")
        start_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(start_word)
        if not start_pattern: return
        
        all_events = {s for s in self.fabric.symbol_table if s.startswith("event_")}
        inferred_facts = 0

        # Create a simple lookup map of all known facts
        fact_map = {}
        for event_symbol in all_events:
            try:
                _, subject, verb, obj = event_symbol.split('_', 3)
                if subject not in fact_map:
                    fact_map[subject] = []
                fact_map[subject].append({'verb': verb, 'object': obj})
            except ValueError: continue
            
        # 1. Find the first link in the chain (A -> is -> B)
        for fact1 in fact_map.get(start_word, []):
            object1 = fact1['object']
            verb1 = fact1['verb']
            
            # 2. Find the second link in the chain (B -> is -> C)
            for fact2 in fact_map.get(object1, []):
                # Ensure the relationship is the same (e.g., both are "is")
                if fact2['verb'] == verb1:
                    final_object = fact2['object']
                    print(f"  - CHAIN FOUND: ({start_word}) -> [{verb1}] -> ({object1}) AND ({object1}) -> [{verb1}] -> ({final_object})")
                    
                    # 3. Synthesize the new fact (A -> is -> C)
                    inferred_event_symbol = f"event_{start_word}_{verb1}_{final_object}"
                    if not self.fabric.recall(inferred_event_symbol):
                        print(f"  - SYNTHESIZING: '{start_word} {verb1} {final_object}'")
                        
                        verb_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(verb1)
                        final_object_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(final_object)
                        
                        if verb_pattern and final_object_pattern:
                            self.fabric.relation.create_and_integrate_relation(
                                start_pattern, verb_pattern, final_object_pattern
                            )
                            inferred_facts += 1

        print(f"--- Inference Complete. Synthesized {inferred_facts} new facts. ---")
    # --- END OF FINAL FIX ---