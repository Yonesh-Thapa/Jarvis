# full, runnable code here
from src.infant_ai import InfantAI
import time, threading, sys

class UILoop:
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.user_input = ""
        self.input_event = threading.Event()
        self.input_thread = threading.Thread(target=self._get_input, daemon=True)

    def _get_input(self):
        while self.ai.is_running:
            self.user_input = sys.stdin.readline()
            self.input_event.set()

    def start(self):
        self.input_thread.start()
        print("\n\n" + "="*60 + "\n      THE AUTONOMOUS LEARNING AGENT\n" + "="*60)
        print("The AI is live. Give it a research topic or type 'help' for other commands.")
        print("-" * 60)
        
        try:
            while self.ai.is_running:
                sys.stdout.write("> ")
                sys.stdout.flush()
                
                # Wait for user input without blocking the AI's resting cycle
                self.input_event.wait(timeout=35) # Wait up to 35 seconds
                
                if self.input_event.is_set():
                    if self.user_input:
                        self.ai.process_command(self.user_input)
                    self.user_input = ""
                    self.input_event.clear()

        except (KeyboardInterrupt, EOFError):
            print("\nShutdown signal received.")
            self.ai.is_running = False

def run_autonomous_learning():
    print("Initializing AI... please wait.")
    ai = InfantAI()
    
    ai_thread = threading.Thread(target=ai.live, daemon=True)
    ai_thread.start()
    
    time.sleep(3) 
    if not ai_thread.is_alive():
        print("CRITICAL: AI thread failed to start. Exiting."); return

    ui = UILoop(ai)
    ui.start()
    
    if ai_thread.is_alive():
        ai_thread.join()
    
    ai.shutdown()
    print("Program finished.")

if __name__ == "__main__":
    run_autonomous_learning()