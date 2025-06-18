# full, runnable code here
import time, threading, sys, random, queue, os

# ... all other imports are correct
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
        # This function is unchanged
        print("--- Bootstrapping Infant AI System ---")
        self.is_running = False; self.state = "AWAKE"
        self._setup_components()
        self.last_activity_time = time.time()
        
    def _setup_components(self):
        # This function is unchanged
        self.fabric = NeuralFabric(max_neurons=100000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=10000, zone='language'); self.fabric.add_neurons(n=5000, zone='general_association'); self.fabric.add_neurons(n=100, zone='goal')
        self.memory = MemoryCore(self.fabric, consolidation_threshold=3); self.logic = LogicCortex(self.fabric, self.memory)
        self.fabric.logic = self.logic
        self.relational = RelationalCortex(self.fabric); self.fabric.relation = self.relational
        self.language = LanguageCortex(self.fabric, self.relational, 'language'); self.fabric.language = self.language
        self.oracle = KnowledgeOracle(); self.browser = WebBrowser(); self.speech_queue = queue.Queue(maxsize=5)
        self.code = CodeCortex(self.fabric, self.language); self.emotion = EmotionModule(self.fabric, self.memory)
        self.action = ActionCortex(self.fabric, self.speech_queue, self.oracle, self.language, self.browser)
        self.planning = PlanningCortex(self.fabric, self.logic, self.emotion, self.memory, self.action, self.language)
        print("\n--- All AI components initialized successfully. ---")
    
    # --- FIX: The user input is now handled by the main application thread ---
    def process_command(self, command_text: str):
        """Processes a single command from the main application loop."""
        try:
            command = command_text.lower().strip().split()
            if not command: return
            cmd, args = command[0], command[1:]
            self.last_activity_time = time.time()
            if cmd == 'help': print("Commands: research, execute, learn, infer, query, status, quit")
            elif cmd == 'research' and args:
                topic = ' '.join(args)
                topic_pattern, _ = self.language._get_or_create_pattern_for_word(topic)
                if topic_pattern: self.planning.add_curiosity_targets({topic_pattern})
            elif cmd == 'execute' and args:
                file_path = args[0]
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f: self.code.mentally_execute(f.read())
            elif cmd == 'learn' and args: self.language.perceive_text_block(' '.join(args))
            elif cmd == 'infer' and args: self.logic.perform_inference(args[0])
            elif cmd == 'query' and len(args) == 2: print(f"Association: {self.logic.query_association(args[0], args[1]):.2f}")
            elif cmd == 'status': print(f"State: {self.state}, Valence: {self.emotion.get_current_valence():.2f}, Memories: {len(self.memory.consolidated_patterns)}, Active Goal: {self.planning.active_goal}, Curiosity Queue: {len(self.planning.curiosity_queue)}, Visited URLs: {len(self.planning.visited_urls)}")
            elif cmd == 'quit': self.is_running = False
            else: print("Unknown command.")
        except Exception as e: print(f"Error processing command: {e}")

    def _enter_resting_state(self):
        # This function is unchanged
        self.state = "RESTING"; print("\n--- AI entered RESTING state. No new sensory input detected. ---")
        self.memory.consolidate()
        if self.memory.consolidated_patterns:
            for _ in range(min(len(self.memory.consolidated_patterns), 3)):
                self.memory.dream(); self.fabric.step_simulation()
        self.memory.prune_synapses()
        print("--- AI is returning to AWAKE state. ---\n")
        self.state = "AWAKE"; self.last_activity_time = time.time()

    def live(self):
        self.is_running = True
        try:
            # Action registration is unchanged
            action_neurons = self.fabric.zones['general_association']
            if len(action_neurons) >= 15:
                ask_p=set(random.sample(list(action_neurons),5)); self.logic.bind_symbol_to_pattern("action_ask_oracle",ask_p); self.action.register_action("action_ask_oracle",self.action._ask_oracle_action)
                search_p=set(random.sample(list(action_neurons-ask_p),5)); self.logic.bind_symbol_to_pattern("action_search_web",search_p); self.action.register_action("action_search_web",self.action._search_web_action)
                browse_p=set(random.sample(list(action_neurons-ask_p-search_p),5)); self.logic.bind_symbol_to_pattern("action_browse_page",browse_p); self.action.register_action("action_browse_page",self.action._browse_page_action)
            
            print("\n--- AI is now a fully autonomous reasoning agent. ---")
            
            last_action_result = None
            while self.is_running:
                if time.time() - self.last_activity_time > 30.0 and self.state == "AWAKE":
                    self._enter_resting_state()
                if self.state == "AWAKE":
                    self.planning.step()
                    fired_uids = self.fabric.step_simulation()
                    if fired_uids: self.last_activity_time = time.time()
                    action_result = self.action.step(fired_uids)
                    if action_result is not None:
                        self.planning.update_plan_with_result(action_result)
                    self.emotion.step()
                    time.sleep(1)
                else:
                    time.sleep(1)
        except PowerBudgetExceededError as e: print(f"CRITICAL: {e}. System is halting.")
        except Exception as e: print(f"FATAL ERROR in AI loop: {e}")
        finally:
            self.is_running = False; self.state = "SHUTDOWN"
            print("INFO: AI thread has finished.")
            
    def shutdown(self):
        print("\n--- Shutting down AI system... ---")
        print("Shutdown complete.")