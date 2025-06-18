# full, runnable code here
import random
import re
from .neural_fabric import NeuralFabric

class CodeCortex:
    """
    A specialized cortex for perceiving and mentally simulating formal code.
    It learns to associate syntax with pre-trained operational axioms.
    """
    def __init__(self, fabric: NeuralFabric, language_cortex):
        self.fabric = fabric
        self.language_cortex = language_cortex
        self.axiom_zone = 'general_association'
        self.mental_sandbox = {} # The "virtual machine" state
        self._pre_train_axioms()
        print("CodeCortex initialized with operational axioms.")

    def _pre_train_axioms(self):
        """

        Teaches the AI the fundamental, non-discoverable concepts of code execution.
        This is like teaching a child what "plus" or "equals" means.
        """
        print("--- Pre-training CodeCortex with axioms ---")
        axiom_neurons = list(self.fabric.zones[self.axiom_zone])
        used_neurons = set()

        def create_axiom(name, num_neurons=5):
            nonlocal used_neurons
            available = list(set(axiom_neurons) - used_neurons)
            if len(available) < num_neurons:
                print(f"WARNING: Not enough neurons for axiom '{name}'")
                return set()
            pattern = set(random.sample(available, num_neurons))
            used_neurons.update(pattern)
            self.fabric.bind(f"op_{name}", pattern)
            print(f"  - Axiom '{name}' bound to {len(pattern)} neurons.")
            return pattern

        # Foundational operations
        self.op_assign = create_axiom("assign")
        self.op_print = create_axiom("print")
        self.op_add = create_axiom("add")
        
        # Ground syntax to these operations
        equals_pattern, _ = self.language_cortex._get_or_create_pattern_for_word("=")
        if equals_pattern and self.op_assign: self.fabric.connect_neurons(list(equals_pattern)[0], list(self.op_assign)[0], weight=1.0)
        
        print_pattern, _ = self.language_cortex._get_or_create_pattern_for_word("print")
        if print_pattern and self.op_print: self.fabric.connect_neurons(list(print_pattern)[0], list(self.op_print)[0], weight=1.0)
        
        plus_pattern, _ = self.language_cortex._get_or_create_pattern_for_word("+")
        if plus_pattern and self.op_add: self.fabric.connect_neurons(list(plus_pattern)[0], list(self.op_add)[0], weight=1.0)

    def _get_value(self, token: str) -> int:
        """Resolves a token to a value, from the sandbox or as a literal."""
        token = token.strip()
        if token.isdigit():
            return int(token)
        elif token in self.mental_sandbox:
            return self.mental_sandbox[token]
        else:
            # It's a variable we haven't seen before, initialize to 0
            return 0 

    def mentally_execute(self, file_content: str):
        """
        Reads and simulates a block of simple code, updating a mental sandbox.
        """
        self.mental_sandbox = {} # Reset for each execution
        lines = file_content.strip().split('\n')
        
        print("\n--- Mentally Executing Code ---")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            print(f"  - Simulating line {i+1}: `{line}`")

            assignment_match = re.match(r"(\w+)\s*=\s*(.+)", line)
            print_match = re.match(r"print\((.+)\)", line)

            if assignment_match:
                var_name = assignment_match.group(1)
                expression = assignment_match.group(2).strip()
                
                if '+' in expression:
                    val1_str, val2_str = [v.strip() for v in expression.split('+', 1)]
                    val1 = self._get_value(val1_str)
                    val2 = self._get_value(val2_str)
                    result = val1 + val2
                    if self.op_add: self.fabric.activate_pattern(self.op_add, 1.1)
                else:
                    result = self._get_value(expression)

                self.mental_sandbox[var_name] = result
                if self.op_assign: self.fabric.activate_pattern(self.op_assign, 1.1)
                print(f"    - State: {var_name} is now {result}")

            elif print_match:
                var_name = print_match.group(1).strip()
                value_to_print = self._get_value(var_name)
                print(f"    - SIMULATED_OUTPUT: {value_to_print}")
                if self.op_print: self.fabric.activate_pattern(self.op_print, 1.1)

            else:
                print(f"    - SYNTAX_UNFAMILIAR: Could not parse line.")
            
            self.fabric.step_simulation()
        
        print(f"--- Execution Complete. Final Sandbox State: {self.mental_sandbox} ---")
        return self.mental_sandbox