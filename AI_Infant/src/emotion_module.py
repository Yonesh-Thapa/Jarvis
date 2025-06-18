# full, runnable code here
from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore
import time

class EmotionModule:
    """
    Assigns valence to experiences and modulates learning and attention.
    Acts as the AI's limbic system analogue.
    """
    def __init__(self, fabric: NeuralFabric, memory_core: MemoryCore):
        """
        Initializes the Emotion Module.

        Args:
            fabric (NeuralFabric): The shared neural fabric.
            memory_core (MemoryCore): The memory controller to influence.
        """
        self.fabric = fabric
        self.memory = memory_core
        
        # --- Principle: Emotion as a learning modulator ---
        self.default_learning_rate = 0.01  # Default Hebbian learning rate
        self.current_learning_rate = self.default_learning_rate
        
        # Valence state: positive, negative, or neutral. Ranges from -1.0 to 1.0.
        self.current_valence = 0.0
        self.valence_decay_rate = 0.75 # How quickly emotion returns to neutral
        self.last_update_time = time.time()

        print("EmotionModule initialized.")

    def assess(self, pattern_uids: set, valence: float):
        """
        Assess a currently active pattern and assign a valence to it.
        This is the primary input from the "outside world" about whether an
        event was good or bad.

        Args:
            pattern_uids (set): The set of neurons representing the experience.
            valence (float): A score from -1.0 (negative) to 1.0 (positive).
        """
        if not pattern_uids:
            return

        # --- Principle: Emotion is a transient state ---
        # Update the AI's overall "mood" or valence state.
        self.current_valence = max(-1.0, min(1.0, self.current_valence + valence))
        print(f"INFO: Event assessed. New valence: {self.current_valence:.2f}")

        # --- Principle: Emotion guides attention and memory ---
        # 1. Modulate learning rate based on the new valence.
        # Positive emotions enhance learning, negative emotions also make memories
        # strong (though perhaps of a different quality, a future enhancement).
        # We use the absolute value so both strong positive and negative events are memorable.
        valence_magnitude = abs(self.current_valence)
        # Learning rate can scale from 50% to 250% of default.
        self.current_learning_rate = self.default_learning_rate * (1 + 1.5 * valence_magnitude)
        
        # 2. Make the current experience a priority for consolidation.
        # By adding it to short-term memory multiple times, we ensure it crosses
        # the MemoryCore's consolidation threshold.
        priority_boost = int(valence_magnitude * 5) # Boost based on emotional intensity
        for _ in range(priority_boost + 1):
            self.memory.observe(pattern_uids)
            
        print(f"  - Learning rate set to: {self.current_learning_rate:.4f}")
        print(f"  - Event priority boosted by: {priority_boost}")
        
    def step(self):
        """
        Applies decay to the current emotional state, returning it to neutral over time.
        This should be called periodically in the main loop.
        """
        now = time.time()
        elapsed = now - self.last_update_time
        
        # Apply exponential decay
        self.current_valence *= (self.valence_decay_rate ** elapsed)
        
        # If valence is very close to zero, reset the learning rate to default.
        if abs(self.current_valence) < 0.01:
            self.current_valence = 0.0
            self.current_learning_rate = self.default_learning_rate

        self.last_update_time = now
        
    def get_current_valence(self) -> float:
        """Returns the current emotional state of the system."""
        return self.current_valence

    def get_current_learning_rate(self) -> float:
        """Returns the learning rate, modulated by the current emotional state."""
        return self.current_learning_rate