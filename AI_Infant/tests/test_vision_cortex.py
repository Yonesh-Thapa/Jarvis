# test script here
import unittest
import cv2
import numpy as np
import time
import os
import sys

# Adjust path to import from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.vision_cortex import VisionCortex

def create_test_image(resolution):
    """Creates a black image with a white rectangle for predictable testing."""
    width, height = resolution
    img = np.zeros((height, width, 3), dtype=np.uint8)
    start_point = (int(width * 0.25), int(height * 0.25))
    end_point = (int(width * 0.75), int(height * 0.75))
    cv2.rectangle(img, start_point, end_point, (255, 255, 255), 2)
    return img

class TestVisionCortex(unittest.TestCase):

    def setUp(self):
        self.INPUT_RESOLUTION = (640, 480)
        self.GRID_SIZE = (64, 48)
        self.NUM_VISION_NEURONS = self.GRID_SIZE[0] * self.GRID_SIZE[1]
        self.fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=self.NUM_VISION_NEURONS, zone='vision')
        self.cortex = VisionCortex(self.fabric, 'vision', self.INPUT_RESOLUTION, self.GRID_SIZE)

    def test_static_image_processing(self):
        """Tests the VisionCortex with a generated, predictable image."""
        test_frame = create_test_image(self.INPUT_RESOLUTION)
        activated_uids = self.cortex.process_frame(test_frame)
        
        fired_uids = self.fabric.step_simulation()
        
        self.assertGreater(len(fired_uids), 0)
        self.assertLess(len(fired_uids), self.NUM_VISION_NEURONS * 0.2)
        self.assertEqual(fired_uids, activated_uids)
    
    @unittest.skip("Skipping live webcam test as it requires a camera.")
    def test_live_webcam_demo(self):
        """Demonstrates the VisionCortex with a live webcam feed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.skipTest("Cannot open webcam.")
            
        ret, frame = cap.read()
        self.assertTrue(ret, "Failed to read from webcam.")
        
        activated_uids = self.cortex.process_frame(frame)
        self.assertIsInstance(activated_uids, set)

        cap.release()

if __name__ == "__main__":
    unittest.main()