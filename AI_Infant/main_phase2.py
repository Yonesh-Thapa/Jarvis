# full, runnable code here
from src.infant_ai import InfantAI
import time, threading, sys

class UILoop:
    """
    A more advanced, non-blocking UI loop that allows the AI to operate
    autonomously in the background while waiting for user commands.
    """
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.user_input = ""
        self.input_event = threading.Event()
        self.input_thread = threading.Thread(target=self._get_input, daemon=True)

    def _get_input(self):
        """Dedicated thread to listen for user input without blocking."""
        while self.ai.is_running:
            try:
                self.user_input = sys.stdin.readline()
                self.input_event.set() # Signal that input is available
            except ValueError: # Catches error when stdin is closed during shutdown
                if not self.ai.is_running: break

    def start(self):
        """Starts the UI loop."""
        self.input_thread.start()
        print("\n\n" + "="*60 + "\n      THE AUTONOMOUS LEARNING AGENT\n" + "="*60)
        print("The AI is live. Its mind is loaded. Type 'help' for commands.")
        print("-" * 60)
        
        try:
            while self.ai.is_running:
                sys.stdout.write("> ")
                sys.stdout.flush()
                
                # Wait for user input, but with a timeout so the AI's idle
                # cycle (for resting/consolidation) is not blocked.
                self.input_event.wait(timeout=35) 
                
                if self.input_event.is_set():
                    # If input was received, process it.
                    if self.user_input:
                        self.process_command(self.user_input)
                    self.user_input = ""
                    self.input_event.clear() # Reset for the next wait

        except (KeyboardInterrupt, EOFError):
            print("\nShutdown signal received.")
            self.ai.is_running = False

    def process_command(self, user_input):
        """Processes user commands and resets the AI's idle timer."""
        # Any interaction means the AI is not idle.
        self.ai.last_activity_time = time.time()
        
        command = user_input.lower().strip().split()
        if not command: return
        
        cmd, args = command[0], command[1:]
        
        if cmd == 'help':
            print("Commands:")
            print("  research <topic>      - Primes the AI to learn about a topic.")
            print("  learn <text>          - AI will read and learn from the provided text.")
            print("  infer <word>          - AI performs deductive inference (e.g., 'infer socrates').")
            print("  query <word1> <word2> - Check the association strength between two concepts.")
            print("  status                - Display the current state of the AI.")
            print("  quit                  - Shuts down the AI and saves its mind.")
        elif cmd == 'research' and args:
            topic_pattern, _ = self.ai.language._get_or_create_pattern_for_word(' '.join(args))
            if topic_pattern: self.ai.planning.add_curiosity_targets({topic_pattern})
        elif cmd == 'learn' and args:
            self.ai.language.perceive_text_block(' '.join(args))
        elif cmd == 'infer' and args:
            self.ai.logic.perform_inference(args[0])
        elif cmd == 'query' and len(args) == 2:
            association = self.ai.logic.query_association(args[0], args[1])
            print(f"Association strength: {association:.2f}")
        elif cmd == 'status':
            print(f"State: {self.ai.state}, Memories: {len(self.ai.memory.consolidated_patterns)}, "
                  f"Curiosity Queue: {len(self.ai.planning.curiosity_queue)}, "
                  f"Visited URLs: {len(self.ai.planning.visited_urls)}")
        elif cmd == 'quit':
            self.ai.is_running = False
        else:
            print("Unknown command. Type 'help' for a list of commands.")


def run_autonomous_learning():
    print("Initializing AI... please wait.")
    ai = InfantAI()
    
    # Run the AI's main 'live' loop in a separate thread
    ai_thread = threading.Thread(target=ai.live, daemon=True)
    ai_thread.start()
    
    time.sleep(3) 
    if not ai_thread.is_alive():
        print("CRITICAL: AI thread failed to start. Exiting."); return

    # Run the UI loop in the main thread
    ui = UILoop(ai)
    ui.start()
    
    # Wait for the AI thread to finish if it's still running
    if ai_thread.is_alive():
        ai_thread.join()
    
    # The AI's shutdown sequence is called within its 'live' method's finally block.
    print("Program finished.")

if __name__ == "__main__":
    run_autonomous_learning()