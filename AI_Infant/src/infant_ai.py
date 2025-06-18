# full, runnable code here
import time
import threading
import cv2
import pickle
import sys # Added for shutdown message
import random
import queue

from .neural_fabric import NeuralFabric, PowerBudgetExceededError
from .vision_cortex import VisionCortex
from .audio_cortex import AudioCortex
from .memory_core import MemoryCore
from .logic_cortex import LogicCortex
from .emotion_module import EmotionModule
from .action_cortex import ActionCortex

class InfantAI:
    """
    The integrated AI system. Orchestrates all cognitive modules and runs
    the main perception-action-learning loop.
    """
    def __init__(self):
        print("--- Bootstrapping Infant AI System ---")
        self.is_running = False
        self._setup_components()
        self.last_salient_pattern = frozenset()
        self.user_input_thread = threading.Thread(target=self._handle_user_input, daemon=True)
        # --- START OF CHANGE ---
        # A thread-safe queue to pass frames from the AI thread to the main display thread.
        # maxsize=1 means we only care about the most recent frame.
        self.display_queue = queue.Queue(maxsize=1)
        # --- END OF CHANGE ---
    def _setup_components(self):
        """Initializes and connects all the AI's core modules."""
        # 1. The Neural Substrate
        self.fabric = NeuralFabric(max_neurons=50000, power_budget_watts=20.0)
        
        # 2. Allocate neurons for each functional zone
        self.fabric.add_neurons(n=64*48, zone='vision')
        self.fabric.add_neurons(n=13*8, zone='audio')
        self.fabric.add_neurons(n=1000, zone='general_association')
        
        # 3. Initialize Cortex Controllers
        self.vision = VisionCortex(self.fabric, 'vision', input_resolution=(640, 480), grid_size=(64, 48))
        self.audio = AudioCortex(self.fabric, 'audio')
        self.memory = MemoryCore(self.fabric, consolidation_threshold=3)
        self.logic = LogicCortex(self.fabric, self.memory)
        self.emotion = EmotionModule(self.fabric, self.memory)
        self.action = ActionCortex(self.fabric, self.logic)
        
        print("\n--- All AI components initialized successfully. ---")

    def _handle_user_input(self):
        """Runs in a separate thread to handle non-blocking user commands."""
        print("\nUser command interface is active. Type 'help' for commands.")
        while self.is_running:
            try:
                command = input("> ").lower().strip().split()
                if not command:
                    continue
                
                cmd = command[0]
                args = command[1:]

                if cmd == 'help':
                    print("Commands:\n"
                          "  bind <symbol>       - Binds a name to the last salient event.\n"
                          "  query <sym_a> <sym_b> - Queries the association between two symbols.\n"
                          "  assess <valence>    - Applies emotional valence (-1.0 to 1.0) to the last event.\n"
                          "  do <seq>... <action>  - Executes a thought/action sequence (e.g., 'do cup action_speak').\n"
                          "  status              - Shows the current status of the AI.\n"
                          "  save <filename>     - Saves the AI's memory (fabric state).\n"
                          "  load <filename>     - Loads a previously saved memory.\n"
                          "  quit                - Shuts down the AI.")
                elif cmd == 'bind' and args:
                    self.logic.bind_symbol_to_pattern(args[0], self.last_salient_pattern)
                elif cmd == 'query' and len(args) == 2:
                    strength = self.logic.query_association(args[0], args[1])
                    print(f"Association strength between '{args[0]}' and '{args[1]}': {strength:.2f}")
                elif cmd == 'assess' and args:
                    valence = float(args[0])
                    self.emotion.assess(self.last_salient_pattern, valence)
                elif cmd == 'do' and args:
                    self.logic.execute_thought_sequence(args)
                elif cmd == 'status':
                    print(f"Current Valence: {self.emotion.get_current_valence():.2f} | "
                          f"Learning Rate: {self.emotion.get_current_learning_rate():.4f} | "
                          f"Consolidated Memories: {len(self.memory.consolidated_patterns)}")
                elif cmd == 'save' and args:
                    self.save_state(args[0])
                elif cmd == 'load' and args:
                    self.load_state(args[0])
                elif cmd == 'quit':
                    self.is_running = False
                else:
                    print("Unknown command. Type 'help'.")
            
            except (KeyboardInterrupt, EOFError):
                # Gracefully shut down if user presses Ctrl+C or Ctrl+D
                print("\nUser interrupted. Shutting down...")
                self.is_running = False
            
            except Exception as e:
                print(f"Error processing command: {e}")

    def live(self):
        """The main cognitive loop."""
        self.is_running = True
        self.audio.start_stream()
        self.user_input_thread.start()
        
        # --- Post-bootstrapping: Create a concept for the 'speak' action ---
        action_speak_neurons = self.fabric.zones['general_association']
        if len(action_speak_neurons) < 5:
             print("CRITICAL: Not enough neurons in general_association zone for actions.")
             self.shutdown()
             return
        
        action_speak_pattern = set(random.sample(list(action_speak_neurons), 5))
        self.logic.bind_symbol_to_pattern("action_speak", action_speak_pattern)
        self.action.register_action("action_speak", self.action._speak_action)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam.")
            self.shutdown()
            return
            
        last_consolidation_time = time.time()
        last_prune_time = time.time()
        
        print("\n--- AI is now LIVE. Running perception loop... ---")
        while self.is_running:
            try:
                # 1. Perception
                ret, frame = cap.read()
                if not ret:
                    break
                
                vision_activations = self.vision.process_frame(frame)
                
                # 2. Cognition (Stepping the Fabric)
                fired_uids = self.fabric.step_simulation()
                
                # 3. Learning & State Update
                if len(fired_uids) > 5: # Only observe non-trivial patterns
                    self.last_salient_pattern = frozenset(fired_uids)
                    self.memory.observe(self.last_salient_pattern)
                
                self.action.step(fired_uids)
                self.emotion.step()

                # 4. Periodic Maintenance Tasks
                now = time.time()
                if now - last_consolidation_time > 5:
                    self.memory.consolidate()
                    last_consolidation_time = now
                
                if now - last_prune_time > 60:
                    self.memory.prune_synapses()
                    last_prune_time = now

                # --- START OF CHANGE ---
                # Instead of showing the image here, put it in the queue for the main thread.
                display_frame = self.vision._frame_to_edges(frame)
                cv2.putText(display_frame, f"Power: {self.fabric.get_total_estimated_watts():.4f}W", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(display_frame, f"Fired: {len(fired_uids)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # 5. Display & Monitoring
                edge_image = self.vision._frame_to_edges(frame)
                cv2.putText(edge_image, f"Power: {self.fabric.get_total_estimated_watts():.4f}W", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(edge_image, f"Fired: {len(fired_uids)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow('AI Perception', edge_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    
            except PowerBudgetExceededError as e:
                print(f"CRITICAL: {e}. Cooling down...")
                time.sleep(2)
            except KeyboardInterrupt:
                self.is_running = False
         # --- START OF CHANGE ---
        # Signal the display loop to terminate by putting a sentinel value in the queue.
        self.display_queue.put(None)
        cap.release()
        # --- END OF CHANGE ---

        self.shutdown()

    def save_state(self, filepath: str):
        """Saves the state of the NeuralFabric to a file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.fabric, f)
            print(f"AI state saved to {filepath}")
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self, filepath: str):
        """Loads the AI state from a file and re-initializes controllers."""
        try:
            with open(filepath, 'rb') as f:
                self.fabric = pickle.load(f)
            # Re-wire all modules to the newly loaded fabric
            self._setup_components()
            print(f"AI state loaded from {filepath}. System re-initialized.")
        except Exception as e:
            print(f"Error loading state: {e}")

    def shutdown(self):
        """Gracefully shuts down all components."""
        # Check if it's already in the process of shutting down
        if not self.is_running and not self.user_input_thread.is_alive():
            return
            
        print("\n--- Shutting down AI system... ---")
        self.is_running = False
        self.audio.stop_stream()
        cv2.destroyAllWindows()
        # Give a moment for the user input thread to see the is_running flag
        # and exit its loop, preventing a hang.
        print("Waiting for threads to close...")
        if threading.current_thread() != self.user_input_thread:
            self.user_input_thread.join(timeout=2.0)
        
        print("Shutdown complete.")
        # Added to ensure the script fully terminates in some environments
        sys.exit(0)