# full, runnable code here
import pyttsx3
import threading

from .neural_fabric import NeuralFabric
from .logic_cortex import LogicCortex

class ActionCortex:
    """
    Translates activated neural patterns into external actions, like speaking.
    """
    def __init__(self, fabric: NeuralFabric, logic_cortex: LogicCortex):
        """
        Initializes the Action Cortex.

        Args:
            fabric (NeuralFabric): The shared neural fabric.
            logic_cortex (LogicCortex): The logic controller.
        """
        self.fabric = fabric
        self.logic = logic_cortex
        
        # Action registry: maps a symbol to a function and its required pattern.
        self._action_registry = {}
        
        # Setup offline Text-to-Speech engine
        try:
            self.tts_engine = pyttsx3.init()
            print("ActionCortex initialized with TTS engine.")
        except Exception as e:
            self.tts_engine = None
            print(f"WARNING: pyttsx3 initialization failed: {e}. Speech will be disabled.")
            print("On Linux, you may need to run: sudo apt-get install espeak")

    def register_action(self, symbol: str, function):
        """
        Binds a symbolic pattern to a Python function.

        Args:
            symbol (str): The symbol representing the action (e.g., "action_speak").
            function: The function to call when the action is triggered.
        """
        pattern = self.fabric.recall(symbol)
        if not pattern:
            print(f"WARNING: Cannot register action '{symbol}'. Symbol not found in fabric.")
            return
        
        self._action_registry[symbol] = {
            "pattern": pattern,
            "function": function
        }
        print(f"Action '{symbol}' registered.")

    def _speak_action(self, context_pattern: set):
        """
        The function executed for the 'action_speak' symbol.

        Args:
            context_pattern (set): The other neurons firing alongside the action,
                                   providing the context for what to say.
        """
        if not self.tts_engine:
            print("ACTION: Speak (TTS disabled)")
            return
            
        text_to_speak = ""
        # Find a known symbol within the context pattern
        for symbol, pattern in self.fabric.symbol_table.items():
            if symbol.startswith("action_"): continue # Don't speak the action itself
            
            if pattern.issubset(context_pattern):
                # We found the subject of the speech command!
                text_to_speak = symbol.replace("_", " ") # speak "red cup" not "red_cup"
                break
        
        if text_to_speak:
            print(f"ACTION: Speaking '{text_to_speak}'")
            # Run TTS in a separate thread to prevent blocking the main loop
            threading.Thread(target=self._run_tts, args=(text_to_speak,), daemon=True).start()
        else:
            print("ACTION: Speak command received, but no known concept to speak about.")

    def _run_tts(self, text):
        """Helper to run the blocking TTS calls."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def step(self, fired_uids: set):
        """
        Checks the currently fired neurons for any registered action patterns.
        This should be called in the main AI loop.
        """
        if not fired_uids or not self._action_registry:
            return

        for symbol, action_data in self._action_registry.items():
            action_pattern = action_data["pattern"]
            
            # If the action pattern is a subset of what just fired...
            if action_pattern and action_pattern.issubset(fired_uids):
                # ...execute the associated function.
                # The context is all other neurons that fired simultaneously.
                context = fired_uids - action_pattern
                action_data["function"](context)
                # Assume one action per step for simplicity
                break