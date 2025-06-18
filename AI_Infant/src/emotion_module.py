import time
from src.neural_fabric import NeuralFabric
from src.memory_core import MemoryCore

class EmotionModule:
    def __init__(self, fabric: NeuralFabric, memory_core: MemoryCore):
        self.fabric = fabric
        self.memory = memory_core
        self.default_learning_rate = 0.01
        self.current_learning_rate = self.default_learning_rate
        self.current_valence = 0.0
        self.valence_decay_rate = 0.75
        self.last_update_time = time.time()
        self.positive_state_pattern = None
        print("EmotionModule initialized.")

    def assess(self, pattern_uids: set, valence: float):
        if not pattern_uids: return
        self.current_valence = max(-1.0, min(1.0, self.current_valence + valence))
        print(f"INFO: Event assessed. New valence: {self.current_valence:.2f}")

        if valence > 0.5 and self.positive_state_pattern:
            experience_with_reward = pattern_uids.union(self.positive_state_pattern)
            for _ in range(5): self.memory.observe(frozenset(experience_with_reward))
            print("  - Associating experience with positive state achievement.")

        valence_magnitude = abs(self.current_valence)
        self.current_learning_rate = self.default_learning_rate * (1 + 1.5 * valence_magnitude)
        priority_boost = int(valence_magnitude * 5)
        for _ in range(priority_boost + 1): self.memory.observe(pattern_uids)
        print(f"  - Learning rate set to: {self.current_learning_rate:.4f}")
        print(f"  - Event priority boosted by: {priority_boost}")
        
    def step(self):
        now = time.time()
        elapsed = now - self.last_update_time
        self.current_valence *= (self.valence_decay_rate ** elapsed)
        if abs(self.current_valence) < 0.01:
            self.current_valence = 0.0
            self.current_learning_rate = self.default_learning_rate
        self.last_update_time = now
        
    def get_current_valence(self) -> float: return self.current_valence
    def get_current_learning_rate(self) -> float: return self.current_learning_rate