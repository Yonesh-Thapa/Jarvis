# full, runnable code here
import unittest
import os
import sys

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.relational_cortex import RelationalCortex
from src.language_cortex import LanguageCortex
from src.logic_cortex import LogicCortex
from src.memory_core import MemoryCore

class TestLanguageCortex(unittest.TestCase):

    def setUp(self):
        self.fabric = NeuralFabric(max_neurons=1000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=500, zone='language')
        self.fabric.add_neurons(n=500, zone='general_association')
        memory = MemoryCore(self.fabric)
        logic = LogicCortex(self.fabric, memory)
        self.fabric.logic = logic
        relational_cortex = RelationalCortex(self.fabric)
        self.language_cortex = LanguageCortex(self.fabric, relational_cortex, 'language')
        # This cross-linking is crucial for the architecture to work
        self.fabric.language = self.language_cortex
        self.fabric.relation = relational_cortex

    def test_word_perception_and_uniqueness(self):
        """Tests if unique patterns are created for unique words, and reused for the same word."""
        patterns1, _, _ = self.language_cortex.perceive_text_block("apple")
        pattern_apple1 = list(patterns1)[0]
        self.assertEqual(len(pattern_apple1), 5)

        patterns2, _, _ = self.language_cortex.perceive_text_block("apple")
        pattern_apple2 = list(patterns2)[0]
        self.assertEqual(pattern_apple1, pattern_apple2)

        patterns3, _, _ = self.language_cortex.perceive_text_block("banana")
        pattern_banana = list(patterns3)[0]
        self.assertNotEqual(pattern_apple1, pattern_banana)
        
        self.assertEqual(len(self.language_cortex.word_to_pattern_map), 2)

    def test_stop_word_removal(self):
        """Tests that common stop words are ignored during relation analysis."""
        # --- FIX: The original test was slightly flawed. This is more robust. ---
        # "the", "is", "a" are stop words and should be ignored.
        _, _, events = self.language_cortex.perceive_text_block("the cat is a fast animal")
        
        # The relation should be formed from the remaining words in sequence.
        # In a 3-word window, this would be "cat fast animal"
        self.assertEqual(len(events), 1)
        event_symbol = self.fabric.relation._get_symbol_for_pattern(list(events)[0])
        
        self.assertIn("cat_fast_animal", event_symbol, "The relation formed should be 'cat_fast_animal'")
        
if __name__ == "__main__":
    unittest.main()