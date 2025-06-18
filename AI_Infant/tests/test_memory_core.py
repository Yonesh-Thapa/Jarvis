# test script here
import unittest
import os
import sys
import random

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore

class TestMemoryCore(unittest.TestCase):

    def setUp(self):
        self.fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=100, zone='vision')
        self.fabric.add_neurons(n=50, zone='audio')
        self.memory = MemoryCore(self.fabric, consolidation_threshold=2)

        vision_uids = list(self.fabric.zones['vision'])
        audio_uids = list(self.fabric.zones['audio'])

        self.apple_visual = frozenset(random.sample(vision_uids, k=10))
        self.apple_audio = frozenset(random.sample(audio_uids, k=5))
        self.concept_apple = self.apple_visual.union(self.apple_audio)

    def test_association_learning_and_consolidation(self):
        """
        Tests that co-activation of patterns leads to consolidation and creates
        strong synaptic links between them.
        """
        for _ in range(3):
            self.memory.observe(self.concept_apple)

        source_neuron = list(self.apple_visual)[0]
        target_neuron = list(self.apple_audio)[0]
        self.assertNotIn(target_neuron, self.fabric.synapses.get(source_neuron, {}))

        self.memory.consolidate()

        self.assertIn(self.concept_apple, self.memory.consolidated_patterns)
        
        synapse = self.fabric.synapses[source_neuron].get(target_neuron)
        self.assertIsNotNone(synapse)
        self.assertGreater(synapse.weight, 0.6)

    def test_recall_from_partial_cue(self):
        """Tests if presenting a part of a memory can retrieve the whole."""
        for _ in range(3):
            self.memory.observe(self.concept_apple)
        self.memory.consolidate()

        self.fabric.activate_pattern(self.apple_visual, signal_strength=1.1)
        self.fabric.step_simulation()
        
        self.memory.recall(cue_uids=self.apple_visual)
        fired_after_recall = self.fabric.step_simulation()

        self.assertTrue(self.apple_audio.issubset(fired_after_recall))

    def test_dreaming(self):
        """Tests if dreaming replays a consolidated memory."""
        for _ in range(3):
            self.memory.observe(self.concept_apple)
        self.memory.consolidate()
        
        self.assertGreater(len(self.memory.consolidated_patterns), 0)
        
        self.memory.dream()
        fired_during_dream = self.fabric.step_simulation()
        
        self.assertEqual(fired_during_dream, self.concept_apple)

if __name__ == "__main__":
    unittest.main()