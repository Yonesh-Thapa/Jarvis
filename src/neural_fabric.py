# full, runnable code here
import time
import random
from collections import deque, defaultdict
import networkx as nx

# --- Custom Exception for Power Management ---

class PowerBudgetExceededError(Exception):
    """Custom exception raised when an action would exceed the system's power budget."""
    def __init__(self, message="Operation aborted: Would exceed power budget"):
        self.message = message
        super().__init__(self.message)

# --- Core Constants for Energy Estimation ---
# NOTE: These are illustrative values. Real-world values would require empirical
# measurement on target hardware. The model, however, remains valid.
# Based on estimates for the human brain, scaled down.
# A human neuron firing uses ~1e-10 Joules.
JOULES_PER_FIRING = 1e-9          # Energy for a single neuron activation cycle.
JOULES_PER_SYNAPSE_UPDATE = 5e-10  # Energy for Hebbian weight update.
WATTS_IDLE_PER_NEURON = 1e-12      # Passive power consumption to maintain potential.

class Neuron:
    """
    Represents a single computational unit in the NeuralFabric.
    It operates on the principle of 'leaky integrate-and-fire'.
    """
    def __init__(self, uid: int, zone: str):
        self.uid = uid  # Unique identifier
        self.zone = zone # Zone tag (e.g., 'vision', 'audio')
        self.activation_potential = 0.0 # Current electrical potential
        self.threshold = 1.0 # Potential required to fire
        self.last_fired_time = 0.0 # Timestamp of the last firing event
        self.activation_history = deque(maxlen=10) # Stores recent activation times for sparsity checks
        self.total_energy_joules = 0.0 # Cumulative energy consumption
        self.created_time = time.time()

    def receive_signal(self, weight: float):
        """Increases activation potential based on an incoming signal."""
        self.activation_potential += weight

    def step(self, current_time: float) -> bool:
        """
        Executes one time step for the neuron. Decays potential and checks for firing.
        Returns True if the neuron fired, False otherwise.
        """
        fired = False
        # Check if potential exceeds the firing threshold
        if self.activation_potential >= self.threshold:
            fired = True
            self.last_fired_time = current_time
            self.activation_history.append(current_time)
            self.total_energy_joules += JOULES_PER_FIRING
            self.activation_potential = 0.0  # Reset potential after firing
        else:
            # Leaky behavior: potential decays over time if not fired
            decay_factor = 0.9
            self.activation_potential *= decay_factor

        return fired

    def get_estimated_watts(self) -> float:
        """Estimates the average power consumption of this neuron since its creation."""
        elapsed_time = time.time() - self.created_time
        if elapsed_time == 0:
            return WATTS_IDLE_PER_NEURON
        # Power = Energy / Time
        active_power = self.total_energy_joules / elapsed_time
        return active_power + WATTS_IDLE_PER_NEURON

class Synapse:
    """
    Represents a connection between two Neurons.
    Its weight is updated based on event-driven Hebbian learning.
    """
    def __init__(self, source_uid: int, target_uid: int, weight: float = 0.5):
        self.source_uid = source_uid
        self.target_uid = target_uid
        self.weight = weight
        self.last_update_joules = 0.0

    def update_weight_hebbian(self, learning_rate: float = 0.01):
        """
        Strengthens the connection. 'Neurons that fire together, wire together.'
        This is called when both pre- and post-synaptic neurons fire.
        """
        self.weight = min(1.0, self.weight + learning_rate)
        self.last_update_joules = JOULES_PER_SYNAPSE_UPDATE

class NeuralFabric:
    """
    The substrate for the AI's mind. Manages all neurons, synapses,
    and their interactions under a strict power budget.
    """
    def __init__(self, max_neurons: int, power_budget_watts: float):
        self.max_neurons = max_neurons
        self.power_budget_watts = power_budget_watts
        self.neurons = {} # {uid: Neuron}
        self.synapses = defaultdict(dict) # {source_uid: {target_uid: Synapse}}
        self.zones = defaultdict(set) # {zone_name: {uid1, uid2, ...}}
        self.symbol_table = {} # {symbol_str: {uid1, uid2, ...}}
        self.last_power_check_time = time.time()
        self.rolling_avg_power = 0.0

    def _check_power_budget(self):
        """Internal check to ensure we don't exceed the power envelope."""
        # This is a fast check using a rolling average. A full check is more expensive.
        if self.rolling_avg_power > self.power_budget_watts:
            raise PowerBudgetExceededError(
                f"Operation aborted: Rolling average power {self.rolling_avg_power:.4f}W "
                f"exceeds budget {self.power_budget_watts:.4f}W"
            )

    def add_neurons(self, n: int, zone: str):
        """Dynamically adds 'n' neurons to a specific zone."""
        if len(self.neurons) + n > self.max_neurons:
            raise ValueError(f"Cannot add {n} neurons. Would exceed max_neurons limit of {self.max_neurons}")
        
        self._check_power_budget() # Check if we can afford the idle power of new neurons
        
        start_uid = len(self.neurons)
        for i in range(n):
            uid = start_uid + i
            self.neurons[uid] = Neuron(uid, zone)
            self.zones[zone].add(uid)
        print(f"INFO: Added {n} neurons to zone '{zone}'. Total neurons: {len(self.neurons)}")

    def connect_neurons(self, source_uid: int, target_uid: int, weight: float = 0.1):
        """Creates a synapse between two neurons if one doesn't exist."""
        if source_uid in self.neurons and target_uid in self.neurons:
            if target_uid not in self.synapses[source_uid]:
                self.synapses[source_uid][target_uid] = Synapse(source_uid, target_uid, weight)

    def activate_pattern(self, neuron_ids: set, signal_strength: float = 1.0):
        """Directly activates a set of neurons, simulating sensory input."""
        for uid in neuron_ids:
            if uid in self.neurons:
                self.neurons[uid].receive_signal(signal_strength)

    def step_simulation(self):
        """
        Advances the simulation by one tick.
        1. Determines which neurons fire.
        2. Propagates signals through synapses.
        3. Applies Hebbian learning to synapses connecting co-activated neurons.
        """
        current_time = time.time()
        fired_neuron_uids = set()
        
        # 1. Integration and Firing Step
        for uid, neuron in self.neurons.items():
            if neuron.step(current_time):
                fired_neuron_uids.add(uid)

        # 2. Signal Propagation & 3. Hebbian Learning (Event-driven)
        if not fired_neuron_uids:
            return # No activity, no computation needed (Least Action)

        for source_uid in fired_neuron_uids:
            # Propagate signals to targets
            for target_uid, synapse in self.synapses.get(source_uid, {}).items():
                self.neurons[target_uid].receive_signal(synapse.weight)
                # Hebbian update: if the target neuron also just fired, strengthen synapse
                if target_uid in fired_neuron_uids:
                    synapse.update_weight_hebbian()

        # Update power estimate after activity
        self.update_power_estimate()
        self._check_power_budget()
        return fired_neuron_uids

    def bind(self, symbol: str, neuron_ids: set):
        """Binds a human-readable symbol to a set of co-activated neurons."""
        # Ensure all neurons exist before binding
        for uid in neuron_ids:
            if uid not in self.neurons:
                raise ValueError(f"Cannot bind symbol '{symbol}': Neuron UID {uid} does not exist.")
        self.symbol_table[symbol] = neuron_ids
        print(f"INFO: Bound symbol '{symbol}' to {len(neuron_ids)} neurons.")

    def recall(self, symbol: str) -> set:
        """Recalls the set of neuron UIDs associated with a symbol."""
        return self.symbol_table.get(symbol, set())

    def update_power_estimate(self):
        """Calculates the total estimated power consumption of the fabric."""
        now = time.time()
        elapsed = now - self.last_power_check_time
        if elapsed < 0.1: # Don't update too frequently
            return

        total_joules_in_interval = 0
        total_idle_watts = 0
        
        # This is a sampled calculation for performance.
        # In a real system, this might be offloaded.
        sampled_neurons = random.sample(list(self.neurons.values()), k=min(len(self.neurons), 500))

        for neuron in sampled_neurons:
            # We assume the energy for firing was already added to neuron.total_energy_joules
            total_idle_watts += WATTS_IDLE_PER_NEURON

        # Estimate total idle power based on the sample
        if len(sampled_neurons) > 0:
            estimated_total_idle_watts = (total_idle_watts / len(sampled_neurons)) * len(self.neurons)
        else:
            estimated_total_idle_watts = 0

        # Estimate active energy from synapses that just updated
        total_synapse_joules = 0
        for source_map in self.synapses.values():
            for synapse in source_map.values():
                total_synapse_joules += synapse.last_update_joules
                synapse.last_update_joules = 0 # Consume the energy value

        # Power = Energy / Time
        active_power = total_synapse_joules / elapsed
        
        # Simple rolling average to smooth out spikes
        current_total_power = estimated_total_idle_watts + active_power
        self.rolling_avg_power = 0.95 * self.rolling_avg_power + 0.05 * current_total_power
        
        self.last_power_check_time = now

    def get_total_estimated_watts(self) -> float:
        """Returns the current rolling average of power consumption."""
        return self.rolling_avg_power

    def to_networkx(self) -> nx.Graph:
        """Exports the current fabric state to a NetworkX graph for visualization."""
        G = nx.Graph()
        for uid, neuron in self.neurons.items():
            G.add_node(uid, zone=neuron.zone, potential=neuron.activation_potential)
        for source_uid, targets in self.synapses.items():
            for target_uid, synapse in targets.items():
                G.add_edge(source_uid, target_uid, weight=synapse.weight)
        return G