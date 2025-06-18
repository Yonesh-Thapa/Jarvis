# test script here
import os
import sys
import random

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore
from src.logic_cortex import LogicCortex

def run_reasoning_test():
    """
    Tests if the LogicCortex can reason about learned associations.
    """
    print("--- Running LogicCortex Reasoning Test ---")

    # 1. Setup Environment
    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=200, zone='vision')
    fabric.add_neurons(n=100, zone='property') # An abstract zone for non-visual properties
    
    memory = MemoryCore(fabric, consolidation_threshold=2)
    logic = LogicCortex(fabric, memory)

    vision_uids = list(fabric.zones['vision'])
    property_uids = list(fabric.zones['property'])

    # 2. Define Core Concepts
    # We create distinct, non-overlapping base patterns.
    p_apple = set(random.sample(vision_uids, k=15))
    p_ball = set(random.sample(vision_uids, k=15))
    p_red = set(random.sample(property_uids, k=10))
    p_round = set(random.sample(property_uids, k=10))

    # 3. Learning Phase: Simulate seeing objects with properties.
    print("\n--- Simulating Learning Phase ---")
    
    # The AI "sees" a red, round apple three times.
    experience_1 = p_apple.union(p_red).union(p_round)
    for _ in range(3):
        fabric.activate_pattern(experience_1, 1.1)
        memory.observe(fabric.step_simulation())

    # The AI "sees" a red, round ball three times.
    experience_2 = p_ball.union(p_red).union(p_round)
    for _ in range(3):
        fabric.activate_pattern(experience_2, 1.1)
        memory.observe(fabric.step_simulation())
        
    # Consolidate these experiences into long-term memory.
    memory.consolidate()
    print("--- Learning complete. Memories consolidated. ---")

    # 4. Symbol Binding Phase: Give names to the learned concepts.
    print("\n--- Binding Symbols to Concepts ---")
    logic.bind_symbol_to_pattern("apple", p_apple)
    logic.bind_symbol_to_pattern("ball", p_ball)
    logic.bind_symbol_to_pattern("red", p_red)
    logic.bind_symbol_to_pattern("round", p_round)
    
    # 5. Querying (Reasoning) Phase
    print("\n--- Querying Associations (Reasoning) ---")
    
    # Test strong, learned associations
    apple_is_red = logic.query_association("apple", "red")
    print(f"Q: Is 'apple' associated with 'red'? A: {apple_is_red:.2f}")
    assert apple_is_red > 0.9, "Failed to associate apple with red."

    apple_is_round = logic.query_association("apple", "round")
    print(f"Q: Is 'apple' associated with 'round'? A: {apple_is_round:.2f}")
    assert apple_is_round > 0.9, "Failed to associate apple with round."

    ball_is_red = logic.query_association("ball", "red")
    print(f"Q: Is 'ball' associated with 'red'? A: {ball_is_red:.2f}")
    assert ball_is_red > 0.9, "Failed to associate ball with red."

    # Test for non-associations. 'apple' and 'ball' were never seen together.
    # Their association should be near zero (any small value is noise).
    apple_is_ball = logic.query_association("apple", "ball")
    print(f"Q: Is 'apple' associated with 'ball'? A: {apple_is_ball:.2f}")
    assert apple_is_ball < 0.1, "Incorrectly associated apple with ball."
    
    print("\nSUCCESS: Reasoning tests passed.")
    
    # 6. Test Sequential Thought
    logic.execute_thought_sequence(["apple", "red"])

    print("\n--- All LogicCortex Tests Passed ---")

if __name__ == "__main__":
    run_reasoning_test()