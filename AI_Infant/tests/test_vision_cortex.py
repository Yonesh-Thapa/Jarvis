# test script here
import cv2
import numpy as np
import time
import os

# Adjust path to import from the 'src' directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_fabric import NeuralFabric
from src.vision_cortex import VisionCortex

def create_test_image(resolution):
    """Creates a black image with a white rectangle for predictable testing."""
    width, height = resolution
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a white rectangle. Start(x,y), End(x,y), Color, Thickness
    start_point = (int(width * 0.25), int(height * 0.25))
    end_point = (int(width * 0.75), int(height * 0.75))
    cv2.rectangle(img, start_point, end_point, (255, 255, 255), 2)
    
    return img

def test_static_image():
    """Tests the VisionCortex with a generated, predictable image."""
    print("--- Running VisionCortex Static Image Test ---")
    
    # 1. Setup Fabric and Cortex
    INPUT_RESOLUTION = (640, 480)
    GRID_SIZE = (64, 48) # Lower resolution perception grid
    NUM_VISION_NEURONS = GRID_SIZE[0] * GRID_SIZE[1]
    
    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=NUM_VISION_NEURONS, zone='vision')
    
    cortex = VisionCortex(fabric, 'vision', INPUT_RESOLUTION, GRID_SIZE)

    # 2. Create test image and process it
    test_frame = create_test_image(INPUT_RESOLUTION)
    activated_uids = cortex.process_frame(test_frame)
    
    # 3. Run fabric simulation and check results
    fired_uids = fabric.step_simulation()
    
    print(f"Processed static image. Activated {len(activated_uids)} neurons.")
    print(f"After simulation step, {len(fired_uids)} neurons fired.")

    # Assertions
    assert len(fired_uids) > 0, "No neurons fired for the test image."
    assert len(fired_uids) < NUM_VISION_NEURONS * 0.2, "Activation is not sparse. Too many neurons fired."
    assert fired_uids == activated_uids, "The neurons that fired should be exactly the ones activated."
    
    print("Static Image Test Passed.\n")

def test_live_webcam():
    """Demonstrates the VisionCortex with a live webcam feed."""
    print("--- Running VisionCortex Live Webcam Demo ---")
    print("Press 'q' to quit.")
    
    # 1. Setup
    INPUT_RESOLUTION = (640, 480)
    GRID_SIZE = (64, 48)
    NUM_VISION_NEURONS = GRID_SIZE[0] * GRID_SIZE[1]
    
    fabric = NeuralFabric(max_neurons=10000, power_budget_watts=20.0)
    fabric.add_neurons(n=NUM_VISION_NEURONS, zone='vision')
    cortex = VisionCortex(fabric, 'vision', INPUT_RESOLUTION, GRID_SIZE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam. Skipping live test.")
        return
        
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 2. Process frame and step simulation
        cortex.process_frame(frame)
        fired_uids = fabric.step_simulation()

        # 3. Display and Report
        # Create a visual representation of what the AI "sees"
        edge_image = cortex._frame_to_edges(frame)
        cv2.imshow('AI Vision (Edges)', edge_image)
        
        # Power and performance logging
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            power_watts = fabric.get_total_estimated_watts()
            print(f"FPS: {fps:.1f} | Fired Neurons: {len(fired_uids):>4} | Power: {power_watts:.6f} W")
            assert power_watts < 2.0, "Power budget for vision exceeded 2W!"
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live Webcam Demo Finished.")

if __name__ == "__main__":
    test_static_image()
    # Uncomment the line below to run the live webcam test.
    # test_live_webcam()