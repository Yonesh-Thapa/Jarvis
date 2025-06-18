# test script here
import os
import sys
import random
import time

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore
from src.logic_cortex import LogicCortex
from src.emotion_module import EmotionModule

def run_emotional_learning_test():
    """
    Tests if the EmotionModule can make a memory "stick" with a single,
    emotionally-charged exposure.
    """
    print("--- Running EmotionModule Learning Modulation Test ---")

    # 1. Setup Full System
    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=100, zone='concept')
    
    memory = MemoryCore(fabric, consolidation_threshold=2) # Note: threshold is 2
    logic = LogicCortex(fabric, memory)
    emotion = EmotionModule(fabric, memory)
    
    concept_uids = list(fabric.zones['concept'])

    # 2. Define concepts
    # A pattern for a "neutral" event (e.g., seeing a rock)
    neutral_pattern = set(random.sample(concept_uids, k=10))
    # A pattern for a "positive" event (e.g., solving a puzzle)
    positive_pattern = set(random.sample(concept_uids, k=10))
    # A property to associate with the positive event (e.g., the concept of "success")
    success_property = set(random.sample(concept_uids, k=5))
    
    # Bind symbols for later querying
    logic.bind_symbol_to_pattern("neutral_event", neutral_pattern)
    logic.bind_symbol_to_pattern("positive_event", positive_pattern)
    logic.bind_symbol_to_pattern("success", success_property)

    # 3. Learning Phase: One exposure for each event
    print("\n--- Simulating Learning Phase (Single Exposure) ---")
    
    # Expose the AI to the neutral event ONCE.
    print("Observing neutral event...")
    fabric.activate_pattern(neutral_pattern, 1.1)
    memory.observe(fabric.step_simulation())
    
    # Expose the AI to the positive event ONCE, but with high valence.
    print("Observing positive event with high valence...")
    positive_experience = positive_pattern.union(success_property)
    fabric.activate_pattern(positive_experience, 1.1)
    fired_positive = fabric.step_simulation()
    # Assess this experience as highly positive
    emotion.assess(fired_positive, valence=0.9)
    
    # 4. Attempt Consolidation
    print("\n--- Triggering Consolidation ---")
    memory.consolidate()
    
    # 5. Test Memory & Reasoning
    print("\n--- Testing Recall and Association ---")

    # Query 1: The neutral event.
    # It was only seen once, and the consolidation threshold is 2.
    # It should NOT have a strong memory.
    neutral_is_neutral = logic.query_association("neutral_event", "neutral_event")
    print(f"Q: How well is 'neutral_event' associated with itself? A: {neutral_is_neutral:.2f}")
    assert neutral_is_neutral < 0.5, "Neutral event was consolidated despite single exposure."
    print("SUCCESS: Neutral event was not strongly memorized, as expected.")

    # Query 2: The positive event.
    # It was also seen only once, but the emotional tag should have boosted
    # it into long-term memory.
    positive_is_associated_with_success = logic.query_association("positive_event", "success")
    print(f"Q: Is 'positive_event' associated with 'success'? A: {positive_is_associated_with_success:.2f}")
    assert positive_is_associated_with_success > 0.9, "Emotionally-charged event FAILED to consolidate."
    print("SUCCESS: Positive event was consolidated from a single exposure due to emotional valence!")

    # 6. Test Valence Decay
    print(f"\nValence immediately after event: {emotion.get_current_valence():.2f}")
    time.sleep(1) # Let time pass
    emotion.step() # Apply decay
    print(f"Valence after 1 second of decay: {emotion.get_current_valence():.2f}")
    assert emotion.get_current_valence() < 0.9, "Valence did not decay."
    print("SUCCESS: Valence decay is working.")
    
    print("\n--- All EmotionModule Tests Passed ---")

if __name__ == "__main__":
    run_emotional_learning_test()