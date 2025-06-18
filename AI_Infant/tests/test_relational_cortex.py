# full, runnable code here
import unittest
from src.neural_fabric import NeuralFabric
from src.relational_cortex import RelationalCortex
from src.language_cortex import LanguageCortex
from src.logic_cortex import LogicCortex
from src.memory_core import MemoryCore

class TestRelationalCortex(unittest.TestCase):

    def setUp(self):
        """Set up a fresh fabric and cortices for each test."""
        self.fabric = NeuralFabric(max_neurons=1000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=500, zone='language')
        self.fabric.add_neurons(n=500, zone='general_association')
        
        memory = MemoryCore(self.fabric)
        logic = LogicCortex(self.fabric, memory)
        self.fabric.logic = logic
        
        self.relational_cortex = RelationalCortex(self.fabric)
        self.fabric.relation = self.relational_cortex
        
        self.language_cortex = LanguageCortex(self.fabric, self.relational_cortex, 'language')
        self.fabric.language = self.language_cortex

    def test_event_creation_from_triad(self):
        """Tests if reading a sentence creates a new 'event' concept."""
        # Pre-check: No event symbols should exist yet
        event_symbols_before = [s for s in self.fabric.symbol_table if s.startswith("event_")]
        self.assertEqual(len(event_symbols_before), 0)

        # The AI perceives a simple sentence
        self.language_cortex.perceive_text_block("cat chased mouse")

        # Post-check: A new event symbol should now exist
        event_symbols_after = [s for s in self.fabric.symbol_table if s.startswith("event_")]
        self.assertEqual(len(event_symbols_after), 1, "A single relational event should have been created.")
        
        event_symbol = event_symbols_after[0]
        self.assertIn("cat", event_symbol)
        self.assertIn("chased", event_symbol)
        self.assertIn("mouse", event_symbol)

        # Check that the event is associated with its parts
        association_score = self.fabric.logic.query_association(event_symbol, "cat")
        self.assertGreater(association_score, 0.9, "The created event should be strongly associated with its subject.")