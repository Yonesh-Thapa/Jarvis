# full, runnable code here
import hashlib
import random
from collections import Counter, deque

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
        print("LanguageCortex initialized and connected to RelationalCortex.")

    def _get_or_create_pattern_for_word(self, word: str) -> tuple[frozenset | None, str | None]:
        """
        Retrieves the neural pattern for a word. If the word is new, it creates
        a new, unique pattern for it and binds it in the fabric.
        """
        word = word.lower().strip(".,'\"?!")
        if not word: return None, None

        if word in self.word_to_pattern_map:
            pattern = self.word_to_pattern_map[word]
            symbol = self.relational_cortex._get_symbol_for_pattern(pattern, default=word)
            return pattern, symbol
        
        # Use a hash to generate a consistent but unique symbol name
        word_hash = hashlib.sha256(word.encode()).hexdigest()
        symbol = f"word_{word_hash[:8]}"
        
        pattern_set = self.fabric.recall(symbol)
        if pattern_set:
            self.word_to_pattern_map[word] = frozenset(pattern_set)
            return frozenset(pattern_set), symbol

        available_neurons = [n for n in self.language_neurons if n not in self.used_neurons]
        if len(available_neurons) < self.neuron_per_word:
            print(f"LANGUAGE_CORTEX_WARN: Not enough neurons left to learn the word '{word}'.")
            return None, None
        
        new_pattern = set(random.sample(available_neurons, self.neuron_per_word))
        self.used_neurons.update(new_pattern)
        self.fabric.bind(symbol, new_pattern)
        self.fabric.bind(word, new_pattern) # Also bind the raw word for easy lookup
        frozen_pattern = frozenset(new_pattern)
        self.word_to_pattern_map[word] = frozen_pattern
        return frozen_pattern, word

    def perceive_text_block(self, text_block: str) -> tuple[set, frozenset | None, set]:
        """
        Processes a block of text, converting it into neural activations,
        identifying relationships, and determining the main idea.
        """
        print(f"\n--- Perceiving and Analyzing text block... ---")
        sentences = text_block.replace('\n', ' ').replace(';','.').replace('!','.').replace('?','.').split('.')
        all_perceived_patterns = set()
        all_event_patterns = set()
        word_counter = Counter()
        
        # --- START OF FINAL FIX: Robust Sliding Window with Stop Word Removal ---
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'is', 'are', 'was', 'were', 'of', 'for', 'to'}

        for sentence in sentences:
            words = [w for w in sentence.lower().strip().split() if w and w not in stop_words]
            if not words: continue
            
            word_counter.update(w for w in words if len(w) > 2)
            
            word_patterns_in_sentence = deque(maxlen=3)

            for word in words:
                pattern, _ = self._get_or_create_pattern_for_word(word)
                if not pattern: continue
                all_perceived_patterns.add(pattern)
                word_patterns_in_sentence.append(pattern)
                
                if len(word_patterns_in_sentence) == 3:
                    subject, verb, obj = list(word_patterns_in_sentence)
                    event_pattern = self.relational_cortex.create_and_integrate_relation(subject, verb, obj)
                    if event_pattern:
                        all_event_patterns.add(frozenset(event_pattern))
            # --- END OF FINAL FIX ---

        main_idea_pattern = None
        if word_counter:
            most_common_word = word_counter.most_common(1)[0][0]
            main_idea_pattern, _ = self._get_or_create_pattern_for_word(most_common_word)
            print(f"  - Heuristic main idea of text: '{most_common_word}'")
        
        print(f"--- Text perception complete. ---")
        return all_perceived_patterns, main_idea_pattern, all_event_patterns