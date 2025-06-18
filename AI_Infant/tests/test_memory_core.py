# test script here
import os
import sys
import random

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore

def run_association_learning_test():
    """
    A comprehensive test to verify that the MemoryCore can learn an association
    between a 'visual' and an 'auditory' pattern.
    """
    print("--- Running MemoryCore Association Learning Test ---")

    # 1. Setup Fabric, Zones, and MemoryCore
    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=100, zone='vision')
    fabric.add_neurons(n=50, zone='audio')
    
    memory = MemoryCore(fabric, consolidation_threshold=2) # Consolidate after 2 observations

    vision_uids = list(fabric.zones['vision'])
    audio_uids = list(fabric.zones['audio'])
    
    # 2. Define two distinct patterns
    # A "visual" pattern for an apple
    apple_visual_pattern = set(random.sample(vision_uids, k=10))
    # An "auditory" pattern for the word "apple"
    apple_audio_pattern = set(random.sample(audio_uids, k=5))
    # The combined, cross-modal concept of an apple
    concept_apple = apple_visual_pattern.union(apple_audio_pattern)
    
    print(f"Defined visual pattern (size {len(apple_visual_pattern)}) and audio pattern (size {len(apple_audio_pattern)}).")
    
    # 3. Simulate learning: Present the visual and audio patterns together
    print("\n--- Simulating Learning Phase (co-activation) ---")
    for i in range(3): # Present the paired stimuli 3 times
        print(f"Learning trial {i+1}...")
        # Activate the combined pattern
        fabric.activate_pattern(concept_apple, signal_strength=1.1)
        # Let the fabric fire
        fired_uids = fabric.step_simulation()
        # Let memory observe the firing
        memory.observe(fired_uids)

    # 4. Trigger consolidation
    print("\n--- Triggering Consolidation ---")
    # Verify no strong cross-modal connections exist yet
    source_neuron = list(apple_visual_pattern)[0]
    target_neuron = list(apple_audio_pattern)[0]
    assert target_neuron not in fabric.synapses.get(source_neuron, {})

    memory.consolidate()
    
    # Verify that consolidation created strong cross-modal synapses
    synapse = fabric.synapses[source_neuron][target_neuron]
    assert synapse is not None, "Synapse was not created between vision and audio zones."
    assert synapse.weight > 0.6, "Synapse is not strong enough after consolidation."
    print("SUCCESS: Consolidation created strong cross-modal connections.")

    # 5. Test Recall: Use a partial cue to retrieve the full memory
    print("\n--- Testing Recall Phase ---")
    # Activate ONLY the visual pattern (the cue)
    print("Presenting only the visual cue...")
    fabric.activate_pattern(apple_visual_pattern, signal_strength=1.1)
    fabric.step_simulation() # Let the cue neurons fire
    
    # Use the memory's recall mechanism based on the cue's firing
    memory.recall(cue_uids=apple_visual_pattern)
    fired_after_recall = fabric.step_simulation() # See what fires next
    
    print(f"Cue activated {len(apple_visual_pattern)} neurons.")
    print(f"After recall, {len(fired_after_recall)} neurons fired.")

    # The magic test: Did the audio pattern fire, even though it wasn't presented?
    assert apple_audio_pattern.issubset(fired_after_recall), \
        "Recall FAILED: Activating the visual cue did not trigger the associated audio pattern."
    
    print("SUCCESS: Recalled the associated audio pattern from a visual cue!")
    
    # 6. Test Dreaming
    print("\n--- Testing Dreaming Phase ---")
    assert len(memory.consolidated_patterns) > 0, "No patterns were consolidated to dream about."
    memory.dream()
    fired_during_dream = fabric.step_simulation()
    assert len(fired_during_dream) == len(concept_apple), "Dreaming did not activate the full memory pattern."
    print("SUCCESS: Dreaming successfully replayed a consolidated memory.")

    print("\n--- All MemoryCore Tests Passed ---")


if __name__ == "__main__":
    run_association_learning_test()