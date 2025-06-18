# test script here
import unittest
import os
import sys
import random
import time

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore
from src.logic_cortex import LogicCortex
from src.emotion_module import EmotionModule

class TestEmotionModule(unittest.TestCase):

    def setUp(self):
        self.fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=100, zone='concept')
        self.memory = MemoryCore(self.fabric, consolidation_threshold=2)
        self.emotion = EmotionModule(self.fabric, self.memory)
        
        concept_uids = list(self.fabric.zones['concept'])
        self.neutral_pattern = frozenset(random.sample(concept_uids, k=10))
        self.positive_pattern = frozenset(random.sample(concept_uids, k=10))

    def test_emotional_learning_modulation(self):
        """
        Tests if high valence causes a memory to be consolidated from a single
        event, while a neutral one is not.
        """
        # Observe neutral event once (below threshold of 2)
        self.memory.observe(self.neutral_pattern)
        
        # Observe positive event once, but assess it with high valence.
        # This should cause it to be observed multiple times internally.
        self.emotion.assess(self.positive_pattern, valence=0.9)
        
        # Check short-term memory counts
        self.assertEqual(self.memory.short_term_memory.count(self.neutral_pattern), 1)
        self.assertGreater(self.memory.short_term_memory.count(self.positive_pattern), 2)
        
        # Consolidate memories
        self.memory.consolidate()
        
        self.assertNotIn(self.neutral_pattern, self.memory.consolidated_patterns)
        self.assertIn(self.positive_pattern, self.memory.consolidated_patterns)
        
    def test_valence_decay(self):
        """Tests that emotional state returns to neutral over time."""
        self.emotion.assess(self.positive_pattern, valence=0.9)
        initial_valence = self.emotion.get_current_valence()
        self.assertAlmostEqual(initial_valence, 0.9)

        time.sleep(0.1)
        self.emotion.step()
        
        self.assertLess(self.emotion.get_current_valence(), initial_valence)
        
        time.sleep(2)
        self.emotion.step()
        self.assertAlmostEqual(self.emotion.get_current_valence(), 0.0, delta=0.1)

if __name__ == "__main__":
    unittest.main()