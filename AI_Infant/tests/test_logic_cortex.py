# full, runnable code here
import unittest
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
        self.fabric.logic = self.logic
        self.relational_cortex = RelationalCortex(self.fabric)
        self.fabric.relation = self.relational_cortex
        self.language_cortex = LanguageCortex(self.fabric, self.relational_cortex, 'language')
        self.fabric.language = self.language_cortex

    def test_transitive_inference(self):
        """Tests if the AI can deduce A=C from A=B and B=C."""
        print("\n--- Testing Inference: Teaching Facts ---")
        self.language_cortex.perceive_text_block("socrates is a man")
        self.language_cortex.perceive_text_block("a man is mortal")

        # --- FIX: The association should be low before inference ---
        association_before = self.logic.query_association("socrates", "mortal")
        self.assertLess(association_before, 0.1, "Socrates should not be directly associated with mortal yet.")

        print("\n--- Testing Inference: Performing Deduction ---")
        self.logic.perform_inference("socrates")
        
        # The inference creates a new event, which is observed and can be consolidated
        self.memory.consolidate()
        
        print("\n--- Testing Inference: Verifying New Knowledge ---")
        inferred_event_symbol = "event_socrates_is_mortal"
        self.assertIn(inferred_event_symbol, self.fabric.symbol_table, "The inferred event was not created.")
        
        # After inference, the direct association should now be strong
        association_after = self.logic.query_association("socrates", "mortal")
        self.assertGreater(association_after, 0.8, "Socrates should be strongly associated with mortal after inference.")