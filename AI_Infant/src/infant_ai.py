# full, runnable code here
import time, threading, sys, random, queue, os
import pickle # Import the library for saving/loading objects

from .neural_fabric import NeuralFabric, PowerBudgetExceededError
# ... other imports are correct
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
        self.is_running = False; self.state = "AWAKE"
        # --- NEW: Define the knowledge directory ---
        self.knowledge_dir = "knowledge"
        self._setup_components()
        self.user_input_thread = threading.Thread(target=self._handle_user_input, daemon=True)
        self.last_activity_time = time.time()
        
    def _setup_components(self):
        self.fabric = NeuralFabric(max_neurons=100000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=10000, zone='language'); self.fabric.add_neurons(n=5000, zone='general_association'); self.fabric.add_neurons(n=100, zone='goal')
        
        self.memory = MemoryCore(self.fabric, consolidation_threshold=3);
        self.logic = LogicCortex(self.fabric, self.memory)
        self.fabric.logic = self.logic
        
        self.relational = RelationalCortex(self.fabric); self.fabric.relation = self.relational
        self.language = LanguageCortex(self.fabric, self.relational, 'language'); self.fabric.language = self.language
        
        # --- NEW: Load the mind from files if they exist ---
        self.load_mind()

        self.oracle = KnowledgeOracle(); self.browser = WebBrowser(); self.speech_queue = queue.Queue(maxsize=5)
        self.code = CodeCortex(self.fabric, self.language); self.emotion = EmotionModule(self.fabric, self.memory)
        self.action = ActionCortex(self.fabric, self.speech_queue, self.oracle, self.language, self.browser)
        self.planning = PlanningCortex(self.fabric, self.logic, self.emotion, self.memory, self.action, self.language)
        print("\n--- All AI components initialized successfully. ---")

    # --- START OF NEW FEATURE: PERSISTENCE ---
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
                pickle.dump(dict(self.fabric.synapses), f) # Convert defaultdict to dict for pickling
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
            # Load symbols
            path = os.path.join(self.knowledge_dir, "symbols.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.fabric.symbol_table = pickle.load(f)
                print(f"  - Loaded {len(self.fabric.symbol_table)} symbols.")

            # Load synapses
            path = os.path.join(self.knowledge_dir, "synapses.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.fabric.synapses.update(pickle.load(f))
                print(f"  - Loaded synapse graph for {len(self.fabric.synapses)} neurons.")

            # Load memories
            path = os.path.join(self.knowledge_dir, "memory.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.memory.consolidated_patterns = pickle.load(f)
                print(f"  - Loaded {len(self.memory.consolidated_patterns)} consolidated memories.")

            # Load word map
            path = os.path.join(self.knowledge_dir, "word_map.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.language.word_to_pattern_map = pickle.load(f)
                print(f"  - Loaded word map with {len(self.language.word_to_pattern_map)} entries.")

            print("--- Mind state loaded successfully. ---")
        except Exception as e:
            print(f"ERROR: Could not load mind state. Starting fresh. Reason: {e}")
    # --- END OF NEW FEATURE: PERSISTENCE ---

    def _handle_user_input(self):
        # This function is unchanged
        if not sys.stdin.isatty(): print("\nWARNING: Non-interactive shell. Command prompt disabled."); return
        print("\nUser command interface is active. Type 'help' for commands.")
        while self.is_running:
            try:
                command = input("> ").lower().strip().split()
                if not command: self.last_activity_time = time.time(); continue
                cmd, args = command[0], command[1:]
                self.last_activity_time = time.time()
                if cmd == 'help': print("Commands: research, execute, learn, infer, query, status, quit")
                elif cmd == 'research' and args:
                    topic_pattern, _ = self.language._get_or_create_pattern_for_word(' '.join(args))
                    if topic_pattern: self.planning.add_curiosity_targets({topic_pattern})
                elif cmd == 'execute' and args:
                    if os.path.exists(args[0]):
                        with open(args[0], 'r') as f: self.code.mentally_execute(f.read())
                elif cmd == 'learn' and args: self.language.perceive_text_block(' '.join(args))
                elif cmd == 'infer' and args: self.logic.perform_inference(args[0])
                elif cmd == 'query' and len(args) == 2: print(f"Association: {self.logic.query_association(args[0], args[1]):.2f}")
                elif cmd == 'status': print(f"State: {self.state}, Memories: {len(self.memory.consolidated_patterns)}, Curiosity Queue: {len(self.planning.curiosity_queue)}, Visited URLs: {len(self.planning.visited_urls)}")
                elif cmd == 'quit': self.is_running = False; break
                else: print("Unknown command.")
            except (KeyboardInterrupt, EOFError): self.is_running = False; break
            except Exception as e: print(f"Error processing command: {e}")

    def _enter_resting_state(self):
        # This function is unchanged
        self.state = "RESTING"; print("\n--- AI entered RESTING state. ---")
        self.memory.consolidate()
        if self.memory.consolidated_patterns:
            for _ in range(min(len(self.memory.consolidated_patterns), 3)):
                self.memory.dream(); self.fabric.step_simulation()
        self.memory.prune_synapses()
        print("--- AI is returning to AWAKE state. ---\n")
        self.state = "AWAKE"; self.last_activity_time = time.time()

    def live(self):
        # This function is unchanged
        self.is_running = True
        try:
            self.user_input_thread.start()
            self.action.register_action("action_ask_oracle", self.action._ask_oracle_action)
            self.action.register_action("action_search_web", self.action._search_web_action)
            self.action.register_action("action_browse_page", self.action._browse_page_action)
            print("\n--- AI is now a fully autonomous reasoning agent with persistence. ---")
            while self.is_running:
                if time.time() - self.last_activity_time > 30.0 and self.state == "AWAKE":
                    self._enter_resting_state()
                if self.state == "AWAKE":
                    next_plan_step = self.planning.step()
                    if next_plan_step:
                        action_result = self.action.execute_action(next_plan_step)
                        self.planning.update_plan_with_result(action_result)
                        self.fabric.step_simulation()
                    self.emotion.step()
                    if next_plan_step: self.last_activity_time = time.time()
                    time.sleep(1)
                else:
                    time.sleep(1)
        except PowerBudgetExceededError as e: print(f"CRITICAL: {e}. System is halting.")
        except Exception as e: print(f"FATAL ERROR in AI loop: {e}")
        finally:
            self.is_running = False; self.state = "SHUTDOWN"
            # --- Call save_mind() on shutdown ---
            self.shutdown()
            
    def shutdown(self):
        print("\n--- Shutting down AI system... ---")
        self.save_mind()
        print("Shutdown complete.")