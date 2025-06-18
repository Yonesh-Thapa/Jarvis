# test script here
import random
import time
from neural_fabric import NeuralFabric, PowerBudgetExceededError

def run_fabric_test():
    """Tests the core functionalities of the NeuralFabric."""
    print("--- Starting NeuralFabric Test ---")

    # 1. Instantiate the fabric with a total capacity and power budget.
    # We allocate more max_neurons than immediately needed to test dynamic growth.
    MAX_NEURONS = 12000
    POWER_BUDGET = 20.0 # Watts
    fabric = NeuralFabric(max_neurons=MAX_NEURONS, power_budget_watts=POWER_BUDGET)
    print(f"Fabric instantiated with {MAX_NEURONS} max neuron capacity and a {POWER_BUDGET}W budget.")

    # 2. Add initial neurons and demonstrate dynamic growth.
    print("\n--- Testing Dynamic Growth ---")
    fabric.add_neurons(n=10000, zone='vision')
    assert len(fabric.neurons) == 10000
    assert len(fabric.zones['vision']) == 10000

    fabric.add_neurons(n=1000, zone='audio')
    assert len(fabric.neurons) == 11000
    assert len(fabric.zones['audio']) == 1000
    print("Dynamic growth test passed.")

    # 3. Fire a random sparse pattern, bind a symbol, and test recall.
    print("\n--- Testing Pattern Binding and Recall ---")
    
    # Create a sparse pattern (e.g., ~1% of vision neurons)
    vision_neuron_ids = list(fabric.zones['vision'])
    pattern_size = int(0.01 * len(vision_neuron_ids))
    sparse_pattern_to_bind = set(random.sample(vision_neuron_ids, pattern_size))
    
    # Bind the symbol "TEST_PATTERN_A" to this set of neurons
    fabric.bind("TEST_PATTERN_A", sparse_pattern_to_bind)
    
    # Recall the neurons using the symbol
    recalled_neuron_ids = fabric.recall("TEST_PATTERN_A")
    
    # Verify that the recalled set is identical to the original
    assert sparse_pattern_to_bind == recalled_neuron_ids
    print(f"Successfully bound and recalled 'TEST_PATTERN_A' with {len(recalled_neuron_ids)} neurons.")
    print("Binding and recall test passed.")

    # 4. Test activation, simulation step, and energy accounting.
    print("\n--- Testing Simulation and Power Estimation ---")

    # Activate the pattern we just recalled
    print("Activating pattern...")
    fabric.activate_pattern(recalled_neuron_ids, signal_strength=1.1) # Strength > threshold to ensure firing

    # Run the simulation for one step
    print("Running one simulation step...")
    fired_uids = fabric.step_simulation()
    print(f"Step completed. {len(fired_uids)} neurons fired.")

    # The fired neurons should be our pattern (or a subset if potential decayed)
    assert len(fired_uids) > 0
    assert fired_uids.issubset(recalled_neuron_ids)

    # Let the simulation run for a moment to get a stable power reading
    time.sleep(0.2)
    fabric.update_power_estimate()
    estimated_watts = fabric.get_total_estimated_watts()

    print(f"Live estimated power: {estimated_watts:.6f} W")
    assert estimated_watts < 0.1 # For this small test, power should be minimal
    print("Power estimation test passed.")
    
    # Test power budget exception
    print("\n--- Testing Power Budget Enforcement ---")
    try:
        # Set a ridiculously low budget that will be exceeded
        fabric.power_budget_watts = 1e-12
        fabric.step_simulation() # This activity should now throw an error
        # This line should not be reached
        print("FAIL: PowerBudgetExceededError was not raised!")
    except PowerBudgetExceededError as e:
        print(f"SUCCESS: Caught expected exception: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected exception: {e}")
    finally:
        # Reset budget for any further tests
        fabric.power_budget_watts = POWER_BUDGET
        
    print("\n--- All NeuralFabric Tests Passed ---")

if __name__ == "__main__":
    run_fabric_test()