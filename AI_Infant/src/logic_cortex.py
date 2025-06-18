# full, runnable code here
from .neural_fabric import NeuralFabric
from .memory_core import MemoryCore

class LogicCortex:
    def __init__(self, fabric: NeuralFabric, memory_core: MemoryCore):
        self.fabric = fabric
        self.memory = memory_core
        print("LogicCortex initialized.")

    def _resolve_symbol_to_pattern(self, symbol: str) -> frozenset | None:
        """Finds the neural pattern for a given word or symbol string."""
        # First, try to find it as a non-word symbol (e.g., event_...)
        pattern = self.fabric.recall(symbol)
        if pattern: return pattern
        
        # If not found, assume it's a word and ask the language cortex
        pattern, _ = self.fabric.language._get_or_create_pattern_for_word(symbol)
        if pattern: return pattern
            
        return None

    def integrate_event_knowledge(self, event_pattern: frozenset, component_patterns: set):
        """Strengthens the connection between an event and its components."""
        if not event_pattern or not component_patterns: return
        
        all_involved = set(event_pattern)
        for component in component_patterns: all_involved.update(component)
        
        frozen_pattern = frozenset(all_involved)
        self.memory._strengthen_pattern(frozen_pattern)
        self.memory.observe(frozen_pattern)

    def query_association(self, symbol_a: str, symbol_b: str) -> float:
        """Measures how strongly two concepts are neurologically linked."""
        pattern_a = self._resolve_symbol_to_pattern(symbol_a)
        pattern_b = self._resolve_symbol_to_pattern(symbol_b)
        
        if not pattern_a or not pattern_b: return 0.0
        
        self.fabric.activate_pattern(pattern_a, 1.1)
        self.fabric.step_simulation()
        
        downstream_activations = self.fabric.step_simulation()
        
        intersection = len(pattern_b.intersection(downstream_activations))
        return intersection / len(pattern_b) if len(pattern_b) > 0 else 0.0

    # --- START OF FINAL FIX: Robust Transitive Inference Engine ---
    def perform_inference(self, start_word: str):
        """
        Performs multi-step, transitive inference.
        Example: If it knows "A is B" and "B is C", it will create "A is C".
        """
        print(f"\n--- Performing Inference on '{start_word}' ---")
        start_pattern, start_symbol_word = self.fabric.language._get_or_create_pattern_for_word(start_word)
        if not start_pattern:
            print(f"  - Cannot perform inference, '{start_word}' is an unknown concept.")
            return
            
        all_events = {s for s in self.fabric.symbol_table if s.startswith("event_")}
        inferred_facts = 0

        # Create a lookup map of all known facts for efficient searching
        # e.g., fact_map['socrates'] = [{'verb': 'is', 'object': 'man'}]
        fact_map = {}
        for event_symbol in all_events:
            try:
                _, subject, verb, obj = event_symbol.split('_', 3)
                if self.fabric.recall(subject):
                    subject_word = self.fabric.relation._get_symbol_for_pattern(self.fabric.recall(subject))
                    if subject_word not in fact_map:
                        fact_map[subject_word] = []
                    fact_map[subject_word].append({'verb': verb, 'object': obj})
            except (ValueError, TypeError):
                continue
            
        # 1. Find the first link in the chain (e.g., Socrates -> is -> Man)
        for fact1 in fact_map.get(start_symbol_word, []):
            object1_word = fact1['object']
            verb1_word = fact1['verb']
            
            # 2. Find the second link in the chain (e.g., Man -> is -> Mortal)
            for fact2 in fact_map.get(object1_word, []):
                # Ensure the relationship is the same (e.g., both are "is a type of")
                if fact2['verb'] == verb1_word:
                    final_object_word = fact2['object']
                    print(f"  - CHAIN FOUND: ({start_word}) -> [{verb1_word}] -> ({object1_word}) AND ({object1_word}) -> [{verb1_word}] -> ({final_object_word})")
                    
                    # 3. Synthesize the new fact (Socrates -> is -> Mortal)
                    inferred_event_symbol = f"event_{start_word}_{verb1_word}_{final_object_word}"
                    
                    if not self.fabric.recall(inferred_event_symbol):
                        print(f"  - SYNTHESIZING NEW KNOWLEDGE: '{start_word} {verb1_word} {final_object_word}'")
                        
                        verb_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(verb1_word)
                        final_object_pattern, _ = self.fabric.language._get_or_create_pattern_for_word(final_object_word)
                        
                        if verb_pattern and final_object_pattern:
                            self.fabric.relation.create_and_integrate_relation(
                                start_pattern, verb_pattern, final_object_pattern
                            )
                            inferred_facts += 1

        if inferred_facts > 0:
            print(f"--- Inference Complete. Synthesized {inferred_facts} new facts. Let me rest to consolidate this.")
            self.last_activity_time = time.time() - 40 # Force rest
        else:
            print(f"--- Inference Complete. No new facts were synthesized from '{start_word}'.")

    # --- END OF FINAL FIX ---