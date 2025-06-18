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
        # --- Principle: An internal language of symbols ---
        # This creates a dictionary entry for the AI's "vocabulary".
        self.fabric.bind(symbol, pattern_uids)

    def query_association(self, symbol_a: str, symbol_b: str) -> float:
        """
        Measures the strength of association between two concepts.

        Args:
            symbol_a (str): The source concept (the "subject").
            symbol_b (str): The concept being queried for association (the "object").

        Returns:
            float: A score from 0.0 to 1.0 representing the association strength.
                   1.0 means the patterns are identical or fire together perfectly.
                   0.0 means there is no connection.
        """
        # 1. Retrieve the neural patterns from the fabric's symbol table.
        pattern_a = self.fabric.recall(symbol_a)
        pattern_b = self.fabric.recall(symbol_b)

        if not pattern_a or not pattern_b:
            # One of the concepts is unknown to the AI.
            return 0.0

        # --- Principle: Thought as internal simulation ---
        # 2. Activate the first pattern as a "cue".
        # This is like "thinking" of symbol A.
        self.memory.recall(cue_uids=pattern_a)
        
        # 3. Step the simulation to see what other neurons get activated
        # through the synapses strengthened by the MemoryCore.
        downstream_activations = self.fabric.step_simulation()
        
        # 4. Calculate the overlap.
        # How much of pattern B was included in the downstream activations?
        intersection = len(pattern_b.intersection(downstream_activations))
        
        # The strength is the proportion of pattern B that was successfully recalled.
        association_strength = intersection / len(pattern_b)
        
        return association_strength
        
    def execute_thought_sequence(self, symbol_sequence: list[str]):
        """
        Executes a "train of thought" by sequentially recalling a list of symbols.

        Args:
            symbol_sequence (list[str]): An ordered list of symbols to "think" about.
        
        Returns:
            set: The final set of fired UIDs after the full sequence.
        """
        print(f"\n--- Executing thought sequence: {symbol_sequence} ---")
        final_fired_uids = set()
        for i, symbol in enumerate(symbol_sequence):
            pattern = self.fabric.recall(symbol)
            if not pattern:
                print(f"  - WARNING: Symbol '{symbol}' not recognized.")
                continue

            print(f"  - Thinking of '{symbol}'...")
            # Activate the pattern. For sequences, we give a strong, direct signal.
            self.fabric.activate_pattern(pattern, signal_strength=1.1)
            final_fired_uids = self.fabric.step_simulation()
            
        print("--- Sequence complete. ---")
        return final_fired_uids