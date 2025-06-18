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

class TestRelationalCortex(unittest.TestCase):

    def setUp(self):
        self.fabric = NeuralFabric(max_neurons=1000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=500, zone='language')
        self.fabric.add_neurons(n=500, zone='general_association')
        
        memory = MemoryCore(self.fabric)
        logic = LogicCortex(self.fabric, memory)
        self.relational_cortex = RelationalCortex(self.fabric)
        self.language_cortex = LanguageCortex(self.fabric, self.relational_cortex, 'language')

        self.fabric.logic = logic
        self.fabric.relation = self.relational_cortex
        self.fabric.language = self.language_cortex

    def test_event_creation_from_triad(self):
        """Tests if reading a sentence creates a new 'event' concept."""
        event_symbols_before = [s for s in self.fabric.symbol_table if s.startswith("event_")]
        self.assertEqual(len(event_symbols_before), 0)

        self.language_cortex.perceive_text_block("cat chased mouse")

        event_symbols_after = [s for s in self.fabric.symbol_table if s.startswith("event_")]
        self.assertEqual(len(event_symbols_after), 1)
        
        event_symbol = event_symbols_after[0]
        self.assertEqual(event_symbol, "event_cat_chased_mouse")

        # --- FIX: Test association after consolidation ---
        self.fabric.memory = MemoryCore(self.fabric, consolidation_threshold=1)
        self.fabric.logic.memory = self.fabric.memory # ensure logic has the new memory
        
        # Manually consolidate the memory of the event
        full_event_pattern = self.fabric.recall(event_symbol)
        self.fabric.memory.observe(full_event_pattern)
        self.fabric.memory.consolidate()

        # After consolidation, the association should be strong
        association_score = self.logic.query_association(event_symbol, "cat")
        self.assertGreater(association_score, 0.9)

    def test_symbol_resolution(self):
        """--- FIX: This test validates the more reliable symbol resolution. ---"""
        pattern, word = self.language_cortex._get_or_create_pattern_for_word("testword")
        self.assertEqual(word, "testword")
        
        resolved_word = self.relational_cortex._get_symbol_for_pattern(pattern)
        self.assertEqual(resolved_word, "testword")
        
        # Create a non-word symbol
        self.fabric.bind("goal_test", {999,998,997})
        resolved_goal = self.relational_cortex._get_symbol_for_pattern({999,998,997})
        self.assertEqual(resolved_goal, "goal_test")


if __name__ == "__main__":
    unittest.main()