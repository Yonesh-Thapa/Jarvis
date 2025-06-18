# full, runnable code here
from src.infant_ai import InfantAI
import time, threading, sys

# --- A more advanced, non-blocking way to handle user input ---
import select

def user_input_is_available():
    """Checks if there is any input waiting on stdin."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def run_autonomous_learning():
    print("Initializing AI... please wait.")
    ai = InfantAI()
    
    # The AI's brain runs in a separate thread
    ai_thread = threading.Thread(target=ai.live, daemon=True)
    ai_thread.start()
    
    time.sleep(3) 
    if not ai_thread.is_alive():
        print("CRITICAL: AI thread failed to start. Exiting."); return

    print("\n\n" + "="*60 + "\n      THE AUTONOMOUS LEARNING AGENT\n" + "="*60)
    print("The AI is live. Give it a research topic or type 'help' for other commands.")
    print("The AI will learn and rest on its own. Press Enter to get a new prompt if needed.")
    print("-" * 60)
    
    # --- START OF FINAL FIX: The Non-Blocking Main Loop ---
    sys.stdout.write("> ")
    sys.stdout.flush()
    
    try:
        while ai.is_running:
            # Check for user input without blocking
            if user_input_is_available():
                command = sys.stdin.readline().strip()
                if command:
                    ai.process_command(command)
                
                # After processing a command, show the prompt again
                sys.stdout.write("> ")
                sys.stdout.flush()

            # Let other threads run
            time.sleep(0.1)

    except (KeyboardInterrupt, EOFError):
        print("\nShutdown signal received.")
        ai.is_running = False # Signal the AI thread to stop
    
    # Wait for the AI thread to finish its last cycle
    if ai_thread.is_alive():
        ai_thread.join(timeout=2.0)
    
    ai.shutdown()
    print("Program finished.")

if __name__ == "__main__":
    run_autonomous_learning()