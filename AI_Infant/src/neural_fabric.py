import time
import random
import threading
from collections import deque, defaultdict
import networkx as nx

class PowerBudgetExceededError(Exception):
    def __init__(self, message="Operation aborted: Would exceed power budget"):
        self.message = message
        super().__init__(self.message)

JOULES_PER_FIRING = 1e-9
JOULES_PER_SYNAPSE_UPDATE = 5e-10
WATTS_IDLE_PER_NEURON = 1e-12

class Neuron:
    # ... (This class is unchanged)
    def __init__(self, uid: int, zone: str):
        self.uid=uid; self.zone=zone; self.activation_potential=0.0; self.threshold=1.0; self.last_fired_time=0.0; self.activation_history=deque(maxlen=10); self.total_energy_joules=0.0; self.created_time=time.time()
    def receive_signal(self, weight: float): self.activation_potential += weight
    def step(self, current_time: float) -> bool:
        fired = False
        if self.activation_potential >= self.threshold: fired=True; self.last_fired_time=current_time; self.activation_history.append(current_time); self.total_energy_joules+=JOULES_PER_FIRING; self.activation_potential=0.0
        else: self.activation_potential*=0.9
        return fired

class Synapse:
    # ... (This class is unchanged)
    def __init__(self, source_uid: int, target_uid: int, weight: float = 0.5):
        self.source_uid=source_uid; self.target_uid=target_uid; self.weight=weight; self.last_update_joules=0.0
    def update_weight_hebbian(self, learning_rate: float = 0.01): self.weight=min(1.0, self.weight + learning_rate); self.last_update_joules=JOULES_PER_SYNAPSE_UPDATE

class NeuralFabric:
    def __init__(self, max_neurons: int, power_budget_watts: float):
        self.max_neurons=max_neurons; self.power_budget_watts=power_budget_watts; self.neurons={}; self.synapses=defaultdict(dict); self.zones=defaultdict(set); self.symbol_table={}; self.last_power_check_time=time.time(); self.rolling_avg_power=0.0
        self.synapse_lock = threading.Lock()

    def add_neurons(self, n: int, zone: str):
        if len(self.neurons) + n > self.max_neurons: raise ValueError("Exceeded max_neurons")
        start_uid = len(self.neurons)
        for i in range(n): uid=start_uid+i; self.neurons[uid]=Neuron(uid,zone); self.zones[zone].add(uid)
        print(f"INFO: Added {n} neurons to zone '{zone}'. Total: {len(self.neurons)}")

    def connect_neurons(self, source_uid: int, target_uid: int, weight: float = 0.1):
        if source_uid in self.neurons and target_uid in self.neurons:
            with self.synapse_lock:
                if target_uid not in self.synapses[source_uid]:
                    self.synapses[source_uid][target_uid] = Synapse(source_uid, target_uid, weight)

    def activate_pattern(self, neuron_ids: set, signal_strength: float = 1.0):
        for uid in neuron_ids:
            if uid in self.neurons: self.neurons[uid].receive_signal(signal_strength)

    def step_simulation(self):
        current_time=time.time(); fired_neuron_uids=set()
        for uid, neuron in self.neurons.items():
            if neuron.step(current_time): fired_neuron_uids.add(uid)
        
        with self.synapse_lock:
            for source_uid in fired_neuron_uids:
                for target_uid, synapse in self.synapses.get(source_uid, {}).items():
                    self.neurons[target_uid].receive_signal(synapse.weight)
                    if target_uid in fired_neuron_uids: synapse.update_weight_hebbian()
        
        self.update_power_estimate()
        return fired_neuron_uids

    def bind(self, symbol: str, neuron_ids: set):
        print(f"INFO: Bound symbol '{symbol}' to {len(neuron_ids)} neurons.")
        self.symbol_table[symbol] = neuron_ids

    def recall(self, symbol: str) -> set: return self.symbol_table.get(symbol, set())

    def update_power_estimate(self):
        now=time.time(); elapsed=now-self.last_power_check_time
        if elapsed < 0.1: return
        total_idle_watts=len(self.neurons)*WATTS_IDLE_PER_NEURON; total_synapse_joules=0
        with self.synapse_lock:
            for source_map in self.synapses.values():
                for synapse in source_map.values(): total_synapse_joules+=synapse.last_update_joules; synapse.last_update_joules=0
        active_power=total_synapse_joules/elapsed if elapsed>0 else 0
        current_total_power=total_idle_watts+active_power
        self.rolling_avg_power = 0.95*self.rolling_avg_power+0.05*current_total_power
        self.last_power_check_time = now