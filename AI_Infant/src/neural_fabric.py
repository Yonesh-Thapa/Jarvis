# full, runnable code here
import time, random, threading
from collections import deque, defaultdict
import networkx as nx

class PowerBudgetExceededError(Exception):
    def __init__(self, message="Operation aborted: Would exceed power budget"):
        self.message = message
        super().__init__(self.message)
class Neuron:
    def __init__(self, uid: int, zone: str):
        self.uid=uid; self.zone=zone; self.activation_potential=0.0; self.threshold=1.0
    def receive_signal(self, weight: float): self.activation_potential += weight
    def step(self) -> bool:
        if self.activation_potential >= self.threshold:
            self.activation_potential = 0.0
            return True
        self.activation_potential *= 0.9
        return False

class Synapse:
    def __init__(self, source_uid: int, target_uid: int, weight: float = 0.5):
        self.source_uid=source_uid; self.target_uid=target_uid; self.weight=weight
    def update_weight_hebbian(self, learning_rate: float = 0.01):
        self.weight = min(1.0, self.weight + learning_rate)

class NeuralFabric:
    def __init__(self, max_neurons: int, power_budget_watts: float):
        self.max_neurons, self.power_budget_watts = max_neurons, power_budget_watts
        self.neurons, self.synapses, self.zones = {}, defaultdict(dict), defaultdict(set)
        self.symbol_table = {}
        self.last_power_check_time, self.rolling_avg_power = time.time(), 0.0
        self.synapse_lock = threading.Lock()
        self.used_event_neurons = set()
        self.logic = self.relation = self.language = None
        # --- FIX: Track energy consumption correctly ---
        self.joules_this_step = 0.0

    def add_neurons(self, n: int, zone: str):
        if len(self.neurons) + n > self.max_neurons: raise ValueError("Exceeded max_neurons")
        start_uid = len(self.neurons)
        for i in range(n):
            uid = start_uid + i; self.neurons[uid] = Neuron(uid, zone); self.zones[zone].add(uid)

    def connect_neurons(self, source_uid, target_uid, weight=0.1):
        if source_uid in self.neurons and target_uid in self.neurons:
            with self.synapse_lock:
                if target_uid not in self.synapses[source_uid]:
                    self.synapses[source_uid][target_uid] = Synapse(source_uid, target_uid, weight)

    def activate_pattern(self, neuron_ids, signal_strength=1.0):
        for uid in neuron_ids:
            if uid in self.neurons: self.neurons[uid].receive_signal(signal_strength)

    def step_simulation(self):
        fired_neuron_uids = set()
        # --- FIX: Accumulate energy from firing neurons ---
        JOULES_PER_FIRING = 1e-9
        for uid, neuron in self.neurons.items():
            if neuron.step():
                fired_neuron_uids.add(uid)
                self.joules_this_step += JOULES_PER_FIRING
        
        with self.synapse_lock:
            JOULES_PER_SYNAPSE_UPDATE = 5e-10
            for source_uid in fired_neuron_uids:
                for target_uid, synapse in self.synapses.get(source_uid, {}).items():
                    self.neurons[target_uid].receive_signal(synapse.weight)
                    if target_uid in fired_neuron_uids:
                        synapse.update_weight_hebbian()
                        self.joules_this_step += JOULES_PER_SYNAPSE_UPDATE
        
        self.update_power_estimate()
        if self.rolling_avg_power > self.power_budget_watts:
            raise PowerBudgetExceededError(f"Power budget exceeded")
        return fired_neuron_uids

    def bind(self, symbol, neuron_ids):
        self.symbol_table[symbol] = frozenset(neuron_ids)

    def recall(self, symbol) -> set:
        return self.symbol_table.get(symbol, set())

    def update_power_estimate(self):
        now = time.time(); elapsed = now - self.last_power_check_time
        if elapsed == 0: return
        
        # --- FIX: Correct power calculation ---
        WATTS_IDLE_PER_NEURON = 1e-12
        current_power = (self.joules_this_step / elapsed) + (len(self.neurons) * WATTS_IDLE_PER_NEURON)
        
        self.rolling_avg_power = (0.95 * self.rolling_avg_power) + (0.05 * current_power)
        self.joules_this_step = 0.0 # Reset for next step
        self.last_power_check_time = now
        
    def get_total_estimated_watts(self) -> float:
        return self.rolling_avg_power