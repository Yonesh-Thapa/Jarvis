# full, runnable code here
import hashlib
import random
from collections import Counter

from .neural_fabric import NeuralFabric
from .relational_cortex import RelationalCortex

class LanguageCortex:
    def __init__(self, fabric: NeuralFabric, relational_cortex: RelationalCortex, 
                 zone_name: str, neuron_per_word: int = 5):
        self.fabric = fabric
        self.relational_cortex = relational_cortex
        self.zone_name = zone_name
        self.neuron_per_word = neuron_per_word
        self.language_neurons = list(self.fabric.zones.get(zone_name, []))
        self.used_neurons = set()
        self.word_to_pattern_map = {}
        if not self.language_neurons: raise ValueError(f"Zone '{zone_name}' has no neurons.")
        print(f"LanguageCortex initialized and connected to RelationalCortex.")

    def _get_or_create_pattern_for_word(self, word: str) -> tuple[frozenset | None, str | None]:
        if word in self.word_to_pattern_map:
            pattern = self.word_to_pattern_map[word]
            symbol = next((s for s, p in self.fabric.symbol_table.items() if p == pattern), None)
            return pattern, symbol
        word_hash = hashlib.sha256(word.encode()).hexdigest()
        symbol = f"word_{word_hash[:8]}"
        pattern_set = self.fabric.recall(symbol)
        if pattern_set:
            self.word_to_pattern_map[word] = frozenset(pattern_set)
            return frozenset(pattern_set), symbol
        available_neurons = [n for n in self.language_neurons if n not in self.used_neurons]
        if len(available_neurons) < self.neuron_per_word: return None, None
        new_pattern = set(random.sample(available_neurons, self.neuron_per_word))
        self.used_neurons.update(new_pattern)
        self.fabric.bind(symbol, new_pattern)
        frozen_pattern = frozenset(new_pattern)
        self.word_to_pattern_map[word] = frozen_pattern
        return frozen_pattern, symbol

    # --- START OF FINAL FIX: ARCHITECTURAL CHANGE TO PARSING ---
    def perceive_text_block(self, text_block: str) -> tuple[set, frozenset | None]:
        print(f"\n--- Perceiving and Structuring text block... ---")
        sentences = text_block.replace('\n', ' ').replace(';','.').replace('!','.').replace('?','.').split('.')
        all_perceived_patterns = set()
        word_counter = Counter()

        for sentence in sentences:
            words = [w for w in sentence.lower().strip().split() if w]
            if not words: continue
            
            # Update word counter for main idea heuristic
            word_counter.update(w for w in words if len(w) > 3)

            # Create patterns for all words in the sentence first
            for word in words:
                pattern, _ = self._get_or_create_pattern_for_word(word)
                if pattern: all_perceived_patterns.add(pattern)

            # Simple "X is Y" parsing for relational learning
            if len(words) >= 3 and "is" in words:
                is_index = words.index("is")
                if is_index > 0 and is_index < len(words) - 1:
                    subject_phrase = " ".join(words[:is_index])
                    object_phrase = " ".join(words[is_index+1:])
                    
                    # For simplicity, we'll treat multi-word phrases as single concepts for now
                    # A more advanced version would handle compound nouns
                    subject_word = subject_phrase.split()[-1]
                    object_word = object_phrase.split()[-1]
                    verb_word = "is"

                    subject_pattern, _ = self._get_or_create_pattern_for_word(subject_word)
                    verb_pattern, _ = self._get_or_create_pattern_for_word(verb_word)
                    object_pattern, _ = self._get_or_create_pattern_for_word(object_word)

                    if subject_pattern and verb_pattern and object_pattern:
                        # Directly create the single, correct relation
                        self.relational_cortex.create_and_integrate_relation(
                            subject_pattern, verb_pattern, object_pattern
                        )
            else:
                # For non-relational sentences, just learn the words
                for word in words:
                    self._get_or_create_pattern_for_word(word)

        main_idea_pattern = None
        if word_counter:
            most_common_word = word_counter.most_common(1)[0][0]
            main_idea_pattern, _ = self._get_or_create_pattern_for_word(most_common_word)
            print(f"  - Heuristic main idea of text: '{most_common_word}'")
        
        print(f"--- Text perception complete. Perceived {len(all_perceived_patterns)} unique concepts. ---")
        return all_perceived_patterns, main_idea_pattern
    # --- END OF FINAL FIX ---