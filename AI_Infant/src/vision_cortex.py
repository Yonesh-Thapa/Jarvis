import cv2
import numpy as np
from src.neural_fabric import NeuralFabric

class VisionCortex:
    def __init__(self, fabric: NeuralFabric, zone_name: str, input_resolution: tuple, grid_size: tuple):
        self.fabric = fabric
        self.zone_name = zone_name
        self.input_width, self.input_height = input_resolution
        self.grid_width, self.grid_height = grid_size
        self.num_grid_cells = self.grid_width * self.grid_height

        if self.zone_name not in self.fabric.zones:
            raise ValueError(f"Zone '{self.zone_name}' not found. Please add neurons to it first.")
        
        vision_zone_neurons = self.fabric.zones[self.zone_name]
        if len(vision_zone_neurons) < self.num_grid_cells:
            raise ValueError(
                f"Vision zone needs at least {self.num_grid_cells} neurons, "
                f"but only {len(vision_zone_neurons)} are allocated."
            )
        
        self.neuron_map = sorted(list(vision_zone_neurons))[:self.num_grid_cells]
        print(f"VisionCortex initialized. Mapped {self.num_grid_cells} neurons to a {self.grid_width}x{self.grid_height} grid.")

    def _frame_to_edges(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edge_image = cv2.Canny(blurred_frame, 50, 150)
        return edge_image

    def _edges_to_sparse_activations(self, edge_image: np.ndarray) -> set:
        activated_neuron_uids = set()
        edge_pixels = np.argwhere(edge_image > 0)
        
        if edge_pixels.size == 0:
            return activated_neuron_uids

        for y, x in edge_pixels:
            grid_x = int((x / self.input_width) * self.grid_width)
            grid_y = int((y / self.input_height) * self.grid_height)
            grid_index = grid_y * self.grid_width + grid_x
            
            if grid_index < len(self.neuron_map):
                neuron_uid = self.neuron_map[grid_index]
                activated_neuron_uids.add(neuron_uid)
                
        return activated_neuron_uids

    def process_frame(self, frame: np.ndarray) -> set:
        edge_image = self._frame_to_edges(frame)
        activated_uids = self._edges_to_sparse_activations(edge_image)
        
        if activated_uids:
            self.fabric.activate_pattern(activated_uids, signal_strength=1.1)
            
        return activated_uids