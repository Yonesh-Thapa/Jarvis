# test script here
import unittest
import numpy as np
import time
import os
import sys

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.audio_cortex import AudioCortex

def create_test_signal(freq, duration, sample_rate):
    """Generates a simple sine wave for predictable testing."""
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * freq * t)
    return data.astype(np.float32).tobytes()

class TestAudioCortex(unittest.TestCase):

    def setUp(self):
        self.SAMPLE_RATE = 44100
        self.N_MFCC = 13
        self.N_BINS = 8
        self.NUM_NEURONS = self.N_MFCC * self.N_BINS
        self.fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=self.NUM_NEURONS, zone='audio')
        self.cortex = AudioCortex(
            self.fabric, 'audio', 
            sample_rate=self.SAMPLE_RATE, 
            n_mfcc=self.N_MFCC, 
            n_bins_per_mfcc=self.N_BINS
        )

    def test_static_signal_processing(self):
        """Tests the AudioCortex with a generated, predictable audio signal."""
        signal_A = create_test_signal(440, 0.1, self.SAMPLE_RATE)
        signal_B = create_test_signal(880, 0.1, self.SAMPLE_RATE)

        activated_A = self.cortex.process_chunk(signal_A)
        fired_A = self.fabric.step_simulation()
        self.assertEqual(len(fired_A), self.N_MFCC)
        self.assertEqual(fired_A, activated_A)

        # Test determinism
        self.fabric.neurons[list(fired_A)[0]].activation_potential = 0
        activated_A2 = self.cortex.process_chunk(signal_A)
        fired_A2 = self.fabric.step_simulation()
        self.assertEqual(fired_A, fired_A2)

        activated_B = self.cortex.process_chunk(signal_B)
        fired_B = self.fabric.step_simulation()
        self.assertEqual(len(fired_B), self.N_MFCC)
        self.assertNotEqual(fired_A, fired_B)

    @unittest.skip("Skipping live mic test as it requires manual interaction.")
    def test_live_mic_demo(self):
        """Demonstrates the AudioCortex with a live microphone."""
        self.cortex.start_stream()
        self.assertTrue(self.cortex.is_streaming)
        time.sleep(2)
        self.cortex.stop_stream()
        self.assertFalse(self.cortex.is_streaming)

if __name__ == "__main__":
    unittest.main()