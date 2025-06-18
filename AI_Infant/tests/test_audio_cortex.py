# test script here
import numpy as np
import time
import os

# Adjust path to import from the 'src' directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.audio_cortex import AudioCortex

def create_test_signal(freq, duration, sample_rate):
    """Generates a simple sine wave for predictable testing."""
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * freq * t)
    # Convert to float32 bytes, which is what process_chunk expects
    return data.astype(np.float32).tobytes()

def test_static_signal():
    """Tests the AudioCortex with a generated, predictable audio signal."""
    print("--- Running AudioCortex Static Signal Test ---")

    # 1. Setup
    SAMPLE_RATE = 44100
    N_MFCC = 13
    N_BINS = 8 # Bins per MFCC
    
    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=N_MFCC * N_BINS, zone='audio')
    
    cortex = AudioCortex(fabric, 'audio', sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, n_bins_per_mfcc=N_BINS)

    # 2. Create two different test signals
    signal_A_440hz = create_test_signal(440, 0.1, SAMPLE_RATE) # A4 note
    signal_B_880hz = create_test_signal(880, 0.1, SAMPLE_RATE) # A5 note

    # 3. Process signals and check results
    print("Processing 440 Hz tone...")
    activated_A = cortex.process_chunk(signal_A_440hz)
    fired_A = fabric.step_simulation()

    print(f"Activated {len(activated_A)} neurons. Fired {len(fired_A)} neurons.")
    
    # Assertions for signal A
    assert len(fired_A) == N_MFCC, f"Expected {N_MFCC} neurons to fire, but {len(fired_A)} did."
    assert fired_A == activated_A, "Fired neurons must match activated neurons."

    # Process the same signal again to test for determinism
    activated_A2 = cortex.process_chunk(signal_A_440hz)
    fired_A2 = fabric.step_simulation()
    assert fired_A == fired_A2, "Processing the same signal should produce the exact same firing pattern."
    print("Determinism test passed.")

    # Process signal B
    print("\nProcessing 880 Hz tone...")
    activated_B = cortex.process_chunk(signal_B_880hz)
    fired_B = fabric.step_simulation()

    print(f"Activated {len(activated_B)} neurons. Fired {len(fired_B)} neurons.")
    assert len(fired_B) == N_MFCC, f"Expected {N_MFCC} neurons to fire, but {len(fired_B)} did."
    assert fired_B != fired_A, "Different signals must produce different firing patterns."
    print("Distinct patterns for distinct sounds test passed.")

    power_watts = fabric.get_total_estimated_watts()
    print(f"\nFinal estimated power: {power_watts:.6f} W")
    assert power_watts < 1.0, "Power budget for audio subsystem test exceeded 1W!"

    print("\nStatic Signal Test Passed.\n")

def demo_live_mic():
    """Demonstrates the AudioCortex with a live microphone."""
    print("--- Running AudioCortex Live Mic Demo (10 seconds) ---")
    print("Speak or make some noise...")

    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=13 * 8, zone='audio')
    cortex = AudioCortex(fabric, 'audio')

    cortex.start_stream()
    
    start_time = time.time()
    last_fired_set = set()

    while time.time() - start_time < 10:
        fired_uids = fabric.step_simulation()
        if fired_uids and fired_uids != last_fired_set:
            print(f"Time: {time.time() - start_time:.1f}s | Sound detected! Fired {len(fired_uids)} neurons. Pattern: {sorted(list(fired_uids))[:5]}...")
            last_fired_set = fired_uids
        time.sleep(0.05) # Main loop polls for fabric activity

    cortex.stop_stream()
    print("Live Mic Demo Finished.")


if __name__ == "__main__":
    test_static_signal()
    # Uncomment to run the live microphone demonstration.
    # Note: Requires a working microphone and permissions.
    # demo_live_mic()