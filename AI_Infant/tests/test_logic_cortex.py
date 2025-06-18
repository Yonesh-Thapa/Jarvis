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

class TestLogicCortex(unittest.TestCase):

    def setUp(self):
        self.fabric = NeuralFabric(max_neurons=1000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=500, zone='language')
        self.fabric.add_neurons(n=500, zone='general_association')
        
        self.memory = MemoryCore(self.fabric)
        self.logic = LogicCortex(self.fabric, self.memory)
        self.relational_cortex = RelationalCortex(self.fabric)
        self.language_cortex = LanguageCortex(self.fabric, self.relational_cortex, 'language')
        
        # Crucial cross-linking
        self.fabric.logic = self.logic
        self.fabric.relation = self.relational_cortex
        self.fabric.language = self.language_cortex

    def test_transitive_inference(self):
        """--- FIX: This test now validates the corrected inference engine. ---"""
        # --- Teach the AI the premises ---
        self.language_cortex.perceive_text_block("socrates is a man")
        self.language_cortex.perceive_text_block("a man is mortal")
        
        # Verify the two facts were learned by checking the symbol table
        self.assertIn("event_socrates_is_man", self.fabric.symbol_table)
        self.assertIn("event_man_is_mortal", self.fabric.symbol_table)

        # --- Check state before inference ---
        association_before = self.logic.query_association("socrates", "mortal")
        self.assertLess(association_before, 0.1, "Socrates should not be associated with mortal yet.")
        self.assertNotIn("event_socrates_is_mortal", self.fabric.symbol_table, "Inferred fact should not exist yet.")

        # --- Perform the inference ---
        self.logic.perform_inference("socrates")
        
        # --- Check state after inference ---
        # The inference should create the new event symbol in the fabric
        self.assertIn("event_socrates_is_mortal", self.fabric.symbol_table, "The inferred event was not created.")
        
        # The inference creates and observes the new pattern, so consolidation will strengthen it
        self.memory.consolidate()
        
        # After consolidation, the direct association should be strong
        association_after = self.logic.query_association("socrates", "mortal")
        self.assertGreater(association_after, 0.8, "Socrates should be strongly associated with mortal after inference.")

if __name__ == "__main__":
    unittest.main()