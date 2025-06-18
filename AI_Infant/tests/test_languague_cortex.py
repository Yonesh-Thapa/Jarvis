# full, runnable code here
import unittest
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
        self.fabric.relation = relational_cortex
        self.language_cortex = LanguageCortex(self.fabric, relational_cortex, 'language')
        self.fabric.language = self.language_cortex

    def test_word_perception_and_uniqueness(self):
        """Tests if unique patterns are created for unique words, and reused for the same word."""
        # --- FIX: Handle the new 3-value return signature ---
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