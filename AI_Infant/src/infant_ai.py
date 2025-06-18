# full, runnable code here
import time, threading, cv2, pickle, sys, random, queue

from .neural_fabric import NeuralFabric, PowerBudgetExceededError
from .vision_cortex import VisionCortex
from .audio_cortex import AudioCortex
from .memory_core import MemoryCore
from .logic_cortex import LogicCortex
from .emotion_module import EmotionModule
from .action_cortex import ActionCortex

class ActionCortex:
    def __init__(self, fabric: NeuralFabric, speech_queue: queue.Queue):
        self.fabric = fabric; self.speech_queue = speech_queue; self._action_registry = {}
        print("ActionCortex initialized.")
    def register_action(self, symbol: str, function):
        pattern = self.fabric.recall(symbol)
        if pattern: self._action_registry[symbol] = {"pattern": pattern, "function": function}
    def _speak_action(self, context_pattern: set):
        text_to_speak = next((s.replace("_", " ") for s, p in self.fabric.symbol_table.items() if not s.startswith("action_") and p.issubset(context_pattern)), None)
        if text_to_speak: self.speech_queue.put(text_to_speak); print(f"ACTION: Queuing speech for '{text_to_speak}'")
    def step(self, fired_uids: set):
        if not fired_uids: return
        for symbol, action_data in self._action_registry.items():
            action_pattern = action_data.get("pattern")
            if action_pattern and action_pattern.issubset(fired_uids):
                action_data["function"](fired_uids - action_pattern); break

class InfantAI:
    def __init__(self):
        print("--- Bootstrapping Infant AI System ---")
        self.is_running = False
        self.display_queue = queue.Queue(maxsize=2)
        self.speech_queue = queue.Queue(maxsize=5)
        self._setup_components()
        self.last_salient_pattern = frozenset()
        self.user_input_thread = threading.Thread(target=self._handle_user_input, daemon=True)
        
    def _setup_components(self):
        self.fabric = NeuralFabric(max_neurons=50000, power_budget_watts=20.0)
        self.fabric.add_neurons(n=64*48, zone='vision'); self.fabric.add_neurons(n=13*8, zone='audio'); self.fabric.add_neurons(n=1000, zone='general_association')
        self.vision = VisionCortex(self.fabric, 'vision', input_resolution=(640, 480), grid_size=(64, 48))
        self.audio = AudioCortex(self.fabric, 'audio')
        self.memory = MemoryCore(self.fabric, consolidation_threshold=3)
        self.logic = LogicCortex(self.fabric, self.memory)
        self.emotion = EmotionModule(self.fabric, self.memory)
        self.action = ActionCortex(self.fabric, self.speech_queue)
        print("\n--- All AI components initialized successfully. ---")

    def _handle_user_input(self):
        if not sys.stdin.isatty(): print("\nWARNING: Non-interactive shell. Command prompt disabled."); return
        print("\nUser command interface is active. Type 'help' for commands.")
        while self.is_running:
            try:
                command = input("> ").lower().strip().split()
                if not command: continue
                cmd, args = command[0], command[1:]
                if cmd == 'help': print("Commands: bind, query, assess, do, status, quit")
                elif cmd == 'bind' and args: self.logic.bind_symbol_to_pattern(args[0], self.last_salient_pattern)
                elif cmd == 'query' and len(args) == 2: print(f"Association: {self.logic.query_association(args[0], args[1]):.2f}")
                # --- START OF NEW COMMAND ---
                elif cmd == 'associate' and len(args) >= 3 and args[1] == 'with':
                    # e.g., associate black_charger with black charger
                    new_symbol = args[0]
                    existing_symbols = args[2:]
                    self.logic.associate_concepts(new_symbol, existing_symbols, self.last_salient_pattern)
                # --- END OF NEW COMMAND ---
                elif cmd == 'assess' and args: self.emotion.assess(self.last_salient_pattern, float(args[0]))
                elif cmd == 'do' and args: self.logic.execute_thought_sequence(args)
                elif cmd == 'status': print(f"Valence: {self.emotion.get_current_valence():.2f}, Memories: {len(self.memory.consolidated_patterns)}")
                elif cmd == 'quit': self.is_running = False; break
                else: print("Unknown command.")
            except (KeyboardInterrupt, EOFError): self.is_running = False; break
            except Exception as e: print(f"Error processing command: {e}")

    def live(self):
        print("\nINFO: Running infant_ai.py with stable architecture.\n")
        cap = None; self.is_running = True
        try:
            self.audio.start_stream(); self.user_input_thread.start()
            action_speak_neurons = self.fabric.zones['general_association']
            if len(action_speak_neurons) >= 5:
                action_speak_pattern = set(random.sample(list(action_speak_neurons), 5))
                self.logic.bind_symbol_to_pattern("action_speak", action_speak_pattern)
                self.action.register_action("action_speak", self.action._speak_action)
            
            for i in range(4):
                temp_cap = cv2.VideoCapture(i)
                if temp_cap and temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None: cap = temp_cap; print(f"SUCCESS: Camera found at index {i}."); break
                    else: temp_cap.release()
            
            if not cap: raise RuntimeError("Could not find any working camera.")
            
            print("\n--- AI is now LIVE. Running perception loop... ---")
            while self.is_running:
                ret, frame = cap.read()
                if not ret: time.sleep(0.01); continue
                
                self.vision.process_frame(frame)
                fired_uids = self.fabric.step_simulation()
                if len(fired_uids) > 5: self.last_salient_pattern = frozenset(fired_uids); self.memory.observe(self.last_salient_pattern)
                self.action.step(fired_uids); self.emotion.step()
                
                display_frame = self.vision._frame_to_edges(frame)
                cv2.putText(display_frame, f"Fired: {len(fired_uids)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                try: self.display_queue.put_nowait(display_frame)
                except queue.Full: pass
        except Exception as e: print(f"FATAL ERROR in AI loop: {e}")
        finally:
            self.is_running = False
            if cap: cap.release()
            self.display_queue.put(None); self.speech_queue.put(None)
            print("INFO: AI thread has finished.")

    def shutdown(self):
        print("\n--- Shutting down AI system... ---")
        if self.audio.is_streaming: self.audio.stop_stream()
        print("Shutdown complete.")