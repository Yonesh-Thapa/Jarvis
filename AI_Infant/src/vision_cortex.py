# full, runnable code here
import cv2
import numpy as np
from src.neural_fabric import NeuralFabric

# To install dependencies:
# pip install opencv-python numpy

class VisionCortex:
    """
    Processes raw video frames into sparse neural activations.
    It acts as a controller for the 'vision' zone of a NeuralFabric instance.
    """
    def __init__(self, fabric: NeuralFabric, zone_name: str, input_resolution: tuple, grid_size: tuple):
        """
        Initializes the Vision Cortex.

        Args:
            fabric (NeuralFabric): The shared neural fabric.
            zone_name (str): The name of the zone this cortex controls (e.g., 'vision').
            input_resolution (tuple): The (width, height) of the input video frames.
            grid_size (tuple): The (width, height) of the neuron grid to map features onto.
                               This determines the spatial resolution of perception.
        """
        self.fabric = fabric
        self.zone_name = zone_name
        self.input_width, self.input_height = input_resolution
        self.grid_width, self.grid_height = grid_size
        self.num_grid_cells = self.grid_width * self.grid_height

        # --- Principle: Verify resource allocation ---
        # Ensure the fabric has a 'vision' zone with enough neurons for our grid.
        if self.zone_name not in self.fabric.zones:
            raise ValueError(f"Zone '{self.zone_name}' not found in the NeuralFabric. Please add neurons to it first.")
        
        vision_zone_neurons = self.fabric.zones[self.zone_name]
        if len(vision_zone_neurons) < self.num_grid_cells:
            raise ValueError(
                f"Vision zone needs at least {self.num_grid_cells} neurons for a {grid_size} grid, "
                f"but only {len(vision_zone_neurons)} are allocated."
            )
        
        # Create a stable, ordered list of neuron UIDs for consistent mapping.
        self.neuron_map = sorted(list(vision_zone_neurons))[:self.num_grid_cells]
        print(f"VisionCortex initialized. Mapped {self.num_grid_cells} neurons to a {self.grid_width}x{self.grid_height} perceptual grid.")

    def _frame_to_edges(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts a raw video frame into a binary edge image.
        This is a low-power, high-efficiency form of feature extraction.
        
        Args:
            frame (np.ndarray): The raw BGR frame from OpenCV.

        Returns:
            np.ndarray: A binary image where white pixels represent detected edges.
        """
        # Convert to grayscale for edge detection (color is a separate feature).
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply a small blur to reduce noise, which improves Canny performance.
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Use Canny edge detection. The thresholds can be tuned.
        # Lower thresholds detect more (potentially noisy) edges.
        # Higher thresholds detect stronger edges.
        edge_image = cv2.Canny(blurred_frame, 50, 150)
        
        return edge_image

    def _edges_to_sparse_activations(self, edge_image: np.ndarray) -> set:
        """
        Maps the locations of detected edges to specific neuron UIDs.

        Args:
            edge_image (np.ndarray): A binary edge image.

        Returns:
            set: A set of unique neuron UIDs that should be activated.
        """
        activated_neuron_uids = set()
        
        # Find the (y, x) coordinates of all edge pixels.
        edge_pixels = np.argwhere(edge_image > 0)
        
        if edge_pixels.size == 0:
            return activated_neuron_uids # No edges, no activations.

        # --- Principle: Map features to neurons, don't store features ---
        # For each detected edge pixel, determine which grid cell it falls into.
        for y, x in edge_pixels:
            # Normalize coordinates to [0, 1] and then scale to grid dimensions.
            grid_x = int((x / self.input_width) * self.grid_width)
            grid_y = int((y / self.input_height) * self.grid_height)
            
            # Convert 2D grid coordinate to a 1D index.
            grid_index = grid_y * self.grid_width + grid_x
            
            # Ensure index is within bounds (can happen at the very edge).
            if grid_index < len(self.neuron_map):
                # Get the neuron UID assigned to this grid cell.
                neuron_uid = self.neuron_map[grid_index]
                activated_neuron_uids.add(neuron_uid)
                
        return activated_neuron_uids

    def process_frame(self, frame: np.ndarray) -> set:
        """
        The main public method. Takes a frame, processes it, and activates the fabric.
        
        Args:
            frame (np.ndarray): The raw BGR frame from OpenCV.

        Returns:
            set: The set of neuron UIDs that were activated in the fabric.
        """
        # --- Principle: "See once", compress, and discard raw data ---
        # The 'frame' object is local to this method and is discarded upon return.
        # The class does not store any raw pixel data.
        
        # 1. Extract symbolic features (edges).
        edge_image = self._frame_to_edges(frame)
        
        # 2. Convert features to sparse neural activations.
        activated_uids = self._edges_to_sparse_activations(edge_image)
        
        # 3. Excite the corresponding neurons in the fabric.
        # We give a signal strength > 1.0 to ensure they overcome the threshold and fire.
        if activated_uids:
            self.fabric.activate_pattern(activated_uids, signal_strength=1.1)
            
        # The return value is a set of symbolic IDs, not data.
        return activated_uids