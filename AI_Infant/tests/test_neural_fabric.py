# full, runnable code here
import unittest
import random
import time
import os
import sys

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric, PowerBudgetExceededError

class TestNeuralFabric(unittest.TestCase):
    def setUp(self):
        """Set up a fresh fabric for each test."""
        self.fabric = NeuralFabric(max_neurons=12000, power_budget_watts=20.0)

    def test_neuron_creation_and_zoning(self):
        """Tests adding neurons and assigning them to zones."""
        self.fabric.add_neurons(n=10000, zone='vision')
        self.assertEqual(len(self.fabric.neurons), 10000)
        self.assertEqual(len(self.fabric.zones['vision']), 10000)
        
        with self.assertRaises(ValueError):
            self.fabric.add_neurons(n=3000, zone='overflow')

    def test_pattern_binding_and_recall(self):
        """Tests binding a symbol to a pattern and recalling it."""
        self.fabric.add_neurons(n=100, zone='test')
        pattern = frozenset(random.sample(list(self.fabric.zones['test']), 50))
        self.fabric.bind("TEST_A", pattern)
        recalled_pattern = self.fabric.recall("TEST_A")
        self.assertEqual(pattern, recalled_pattern)
        self.assertIsNone(self.fabric.recall("NON_EXISTENT"))

    def test_simulation_and_power_model(self):
        """--- FIX: This test validates the more accurate power model. ---"""
        self.fabric.add_neurons(n=100, zone='test')
        pattern = set(self.fabric.zones['test'])
        self.fabric.activate_pattern(pattern, signal_strength=1.1)
        
        self.assertEqual(self.fabric.get_total_estimated_watts(), 0.0)

        fired_uids = self.fabric.step_simulation()
        self.assertTrue(len(fired_uids) > 0)
        
        power = self.fabric.get_total_estimated_watts()
        self.assertGreater(power, 0)
        self.assertLess(power, 0.1, "Power for a small firing event should be minimal.")

    def test_power_budget_exception(self):
        """Tests that the power budget enforcement raises an exception."""
        self.fabric.add_neurons(n=100, zone='test')
        self.fabric.power_budget_watts = 1e-15 
        self.fabric.activate_pattern(set(self.fabric.zones['test']), 1.1)
        
        with self.assertRaises(PowerBudgetExceededError):
            self.fabric.step_simulation()

if __name__ == '__main__':
    unittest.main()