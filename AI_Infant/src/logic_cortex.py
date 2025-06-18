# full, runnable code here
from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore

class LogicCortex:
    """
    Provides reasoning and planning capabilities by manipulating symbolic patterns
    stored within the NeuralFabric's connections.
    """
    def __init__(self, fabric: NeuralFabric, memory_core: MemoryCore):
        """
        Initializes the Logic Cortex.

        Args:
            fabric (NeuralFabric): The shared neural fabric.
            memory_core (MemoryCore): The memory controller for recall operations.
        """
        self.fabric = fabric
        self.memory = memory_core
        print("LogicCortex initialized.")

    def bind_symbol_to_pattern(self, symbol: str, pattern_uids: set):
        """
        A formal mechanism to bind a human-readable symbol to a pattern of neurons.
        This populates the fabric's universal symbol table.

        Args:
            symbol (str): The label for the concept (e.g., "apple", "red").
            pattern_uids (set): The set of neuron UIDs representing the concept.
        """
        self.fabric.bind(symbol, pattern_uids)

    # --- START OF NEW FEATURE: Explicit Association ---
    def associate_concepts(self, new_symbol: str, existing_symbols: list, context_pattern: set):
        """
        Builds strong synaptic bridges between a new context and existing concepts.
        This is the primary tool for compositional learning.
        """
        print(f"--- Associating '{new_symbol}' with {existing_symbols} ---")
        
        # 1. Bind the new symbol to the current sensory experience.
        self.bind_symbol_to_pattern(new_symbol, context_pattern)

        all_involved_neurons = set(context_pattern)
        
        # 2. Retrieve the patterns for the existing symbols.
        for symbol in existing_symbols:
            pattern = self.fabric.recall(symbol)
            if pattern:
                all_involved_neurons.update(pattern)
            else:
                print(f"  - WARNING: Could not find existing concept '{symbol}' to associate.")

        # 3. Strengthen the connections within this entire group of concepts.
        # This is like a targeted, high-power consolidation event.
        print(f"  - Building synaptic bridges between {len(all_involved_neurons)} total neurons.")
        self.memory._strengthen_pattern(frozenset(all_involved_neurons))
    # --- END OF NEW FEATURE ---

    def query_association(self, symbol_a: str, symbol_b: str) -> float:
        """
        Measures the strength of association between two concepts.

        Args:
            symbol_a (str): The source concept (the "subject").
            symbol_b (str): The concept being queried for association (the "object").

        Returns:
            float: A score from 0.0 to 1.0 representing the association strength.
        """
        pattern_a = self.fabric.recall(symbol_a)
        pattern_b = self.fabric.recall(symbol_b)

        if not pattern_a or not pattern_b:
            return 0.0

        # --- Principle: Thought as internal simulation ---
        # 1. Activate the first pattern as a "cue".
        self.memory.recall(cue_uids=pattern_a)
        
        # 2. THE FIX: Run a simulation step to allow the cued neurons to actually fire.
        self.fabric.step_simulation()
        
        # 3. Run a SECOND step to see what downstream neurons get activated
        # by the signal propagating from the now-firing cue neurons.
        downstream_activations = self.fabric.step_simulation()
        
        # 4. Calculate the overlap.
        intersection = len(pattern_b.intersection(downstream_activations))
        if len(pattern_b) == 0: return 0.0
        
        association_strength = intersection / len(pattern_b)
        
        return association_strength
        
    def execute_thought_sequence(self, symbol_sequence: list[str]):
        """
        Executes a "train of thought" by sequentially recalling a list of symbols.
        """
        print(f"\n--- Executing thought sequence: {symbol_sequence} ---")
        final_fired_uids = set()
        for i, symbol in enumerate(symbol_sequence):
            pattern = self.fabric.recall(symbol)
            if not pattern:
                print(f"  - WARNING: Symbol '{symbol}' not recognized.")
                continue

            print(f"  - Thinking of '{symbol}'...")
            self.fabric.activate_pattern(pattern, signal_strength=1.1)
            final_fired_uids = self.fabric.step_simulation()
            
        print("--- Sequence complete. ---")
        return final_fired_uids