# full, runnable code here
import unittest
import random
import time
from src.neural_fabric import NeuralFabric, PowerBudgetExceededError

class TestNeuralFabric(unittest.TestCase):
    def setUp(self):
        """Set up a fresh fabric for each test."""
        self.fabric = NeuralFabric(max_neurons=12000, power_budget_watts=20.0)

    def test_dynamic_growth(self):
        """Tests adding neurons on demand."""
        self.fabric.add_neurons(n=10000, zone='vision')
        self.assertEqual(len(self.fabric.neurons), 10000)
        self.fabric.add_neurons(n=1000, zone='audio')
        self.assertEqual(len(self.fabric.neurons), 11000)

    def test_pattern_binding_and_recall(self):
        """Tests binding a symbol to a pattern and recalling it."""
        self.fabric.add_neurons(n=1000, zone='test')
        test_neuron_ids = list(self.fabric.zones['test'])
        pattern_to_bind = set(random.sample(test_neuron_ids, 50))
        self.fabric.bind("TEST_A", pattern_to_bind)
        recalled_pattern = self.fabric.recall("TEST_A")
        self.assertEqual(pattern_to_bind, recalled_pattern)

    def test_simulation_and_power(self):
        """Tests a simulation step and power estimation."""
        self.fabric.add_neurons(n=100, zone='test')
        pattern = set(self.fabric.zones['test'])
        self.fabric.activate_pattern(pattern, signal_strength=1.1)
        fired_uids = self.fabric.step_simulation()
        self.assertTrue(len(fired_uids) > 0)
        
        # --- FIX: Use the restored function ---
        power = self.fabric.get_total_estimated_watts()
        self.assertGreater(power, 0)
        self.assertLess(power, 0.1)

    def test_power_budget_exception(self):
        """Tests that the power budget enforcement raises an exception."""
        self.fabric.add_neurons(n=100, zone='test')
        self.fabric.power_budget_watts = 1e-15 # Set an impossibly low budget
        self.fabric.activate_pattern(set(self.fabric.zones['test']), 1.1)
        with self.assertRaises(PowerBudgetExceededError):
            self.fabric.step_simulation()