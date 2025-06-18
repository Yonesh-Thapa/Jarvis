# full, runnable code here
import time, threading, sys, random, queue, os
import pickle
from collections import defaultdict

from .neural_fabric import NeuralFabric, PowerBudgetExceededError
from .language_cortex import LanguageCortex
from .knowledge_oracle import KnowledgeOracle
from .web_browser import WebBrowser
from .code_cortex import CodeCortex
from .relational_cortex import RelationalCortex
from .memory_core import MemoryCore
from .logic_cortex import LogicCortex
from .emotion_module import EmotionModule
from .planning_cortex import PlanningCortex
from .action_cortex import ActionCortex


class InfantAI:
    def __init__(self):
        print("--- Bootstrapping Infant AI System ---")
        self.is_running = False
        self.state = "AWAKE"
        self.knowledge_dir = "knowledge"
        
        self._setup_components()
        self.last_activity_time = time.time()
        
    def _setup_components(self):
        """Initializes and connects all cognitive modules."""
        self.fabric = NeuralFabric(max_neurons=100000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=10000, zone='language')
        self.fabric.add_neurons(n=5000, zone='general_association')
        self.fabric.add_neurons(n=100, zone='goal')
        
        self.memory = MemoryCore(self.fabric, consolidation_threshold=3)
        self.logic = LogicCortex(self.fabric, self.memory)
        self.fabric.logic = self.logic # Allow fabric to access logic
        
        self.relational = RelationalCortex(self.fabric)
        self.fabric.relation = self.relational # Allow fabric to access relational
        
        self.language = LanguageCortex(self.fabric, self.relational, 'language')
        self.fabric.language = self.language # Allow fabric to access language
        
        # --- NEW: Load the mind from files if they exist ---
        self.load_mind()

        self.oracle = KnowledgeOracle()
        self.browser = WebBrowser()
        self.speech_queue = queue.Queue(maxsize=5)
        self.code = CodeCortex(self.fabric, self.language)
        self.emotion = EmotionModule(self.fabric, self.memory)
        self.action = ActionCortex(self.fabric, self.speech_queue, self.oracle, self.language, self.browser)
        self.planning = PlanningCortex(self.fabric, self.logic, self.emotion, self.memory, self.action, self.language)
        
        print("\n--- All AI components initialized successfully. ---")

    # --- START OF FULLY IMPLEMENTED PERSISTENCE ---
    def save_mind(self):
        """Saves the essential components of the AI's mind to separate files."""
        print("\n--- Saving mind state... ---")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        try:
            # Save the symbol table (the lexicon)
            with open(os.path.join(self.knowledge_dir, "symbols.pkl"), "wb") as f:
                pickle.dump(self.fabric.symbol_table, f)
            # Save the synapse graph (the wiring)
            with open(os.path.join(self.knowledge_dir, "synapses.pkl"), "wb") as f:
                pickle.dump(dict(self.fabric.synapses), f)
            # Save the consolidated memories
            with open(os.path.join(self.knowledge_dir, "memory.pkl"), "wb") as f:
                pickle.dump(self.memory.consolidated_patterns, f)
            # Save the word-to-pattern map for the language cortex
            with open(os.path.join(self.knowledge_dir, "word_map.pkl"), "wb") as f:
                pickle.dump(self.language.word_to_pattern_map, f)
            
            print("--- Mind state saved successfully. ---")
        except Exception as e:
            print(f"ERROR: Could not save mind state. Reason: {e}")

    def load_mind(self):
        """Loads the AI's mind from files if they exist."""
        print("--- Attempting to load mind state... ---")
        if not os.path.isdir(self.knowledge_dir):
            print("  - No knowledge directory found. Starting with a blank slate.")
            return

        try:
            path = os.path.join(self.knowledge_dir, "symbols.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f: self.fabric.symbol_table = pickle.load(f)
                print(f"  - Loaded {len(self.fabric.symbol_table)} symbols.")

            path = os.path.join(self.knowledge_dir, "synapses.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f: self.fabric.synapses.update(pickle.load(f))
                print(f"  - Loaded synapse graph for {len(self.fabric.synapses)} neurons.")

            path = os.path.join(self.knowledge_dir, "memory.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f: self.memory.consolidated_patterns = pickle.load(f)
                print(f"  - Loaded {len(self.memory.consolidated_patterns)} consolidated memories.")

            path = os.path.join(self.knowledge_dir, "word_map.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f: self.language.word_to_pattern_map = pickle.load(f)
                print(f"  - Loaded word map with {len(self.language.word_to_pattern_map)} entries.")

            print("--- Mind state loaded successfully. ---")
        except Exception as e:
            print(f"ERROR: Could not load mind state. Starting fresh. Reason: {e}")
    # --- END OF FULLY IMPLEMENTED PERSISTENCE ---

    def _enter_resting_state(self):
        """Handles the AI's cognitive cycle of consolidation and dreaming."""
        self.state = "RESTING"
        print("\n--- AI entered RESTING state. Consolidating memories... ---")
        self.memory.consolidate()
        if self.memory.consolidated_patterns:
            for _ in range(min(len(self.memory.consolidated_patterns), 3)):
                self.memory.dream()
                self.fabric.step_simulation()
        self.memory.prune_synapses()
        print("--- AI is returning to AWAKE state. ---\n")
        self.state = "AWAKE"
        self.last_activity_time = time.time()

    def live(self):
        """The main loop of the AI's life. Now works with main_phase2.py's UI."""
        self.is_running = True
        try:
            # Register available actions with the action cortex
            self.action.register_action("action_ask_oracle", self.action._ask_oracle_action)
            self.action.register_action("action_search_web", self.action._search_web_action)
            self.action.register_action("action_browse_page", self.action._browse_page_action)
            print("\n--- AI is now a fully autonomous reasoning agent with persistence. ---")
            
            while self.is_running:
                # Enter resting state if idle for too long
                if time.time() - self.last_activity_time > 30.0 and self.state == "AWAKE":
                    self._enter_resting_state()
                
                if self.state == "AWAKE":
                    # Planner decides the next action
                    next_plan_step = self.planning.step()
                    if next_plan_step:
                        self.last_activity_time = time.time()
                        action_result = self.action.execute_action(next_plan_step)
                        self.planning.update_plan_with_result(action_result)
                        self.fabric.step_simulation()
                    
                    self.emotion.step() # Emotional state decays over time
                    time.sleep(1) # Pace the cognitive cycle
                else:
                    time.sleep(1) # Sleep while resting
                    
        except PowerBudgetExceededError as e:
            print(f"CRITICAL: {e}. System is halting.")
        except Exception as e:
            print(f"FATAL ERROR in AI loop: {e}")
        finally:
            self.is_running = False
            self.state = "SHUTDOWN"
            self.shutdown()
            
    def shutdown(self):
        """Gracefully shuts down the AI, saving its mind."""
        print("\n--- Shutting down AI system... ---")
        self.save_mind()
        print("Shutdown complete.")