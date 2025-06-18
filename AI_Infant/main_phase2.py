# full, runnable code here
from src.infant_ai import InfantAI
import time, threading

def run_autonomous_learning():
    print("Initializing AI... please wait.")
    ai = InfantAI()
    
    ai_thread = threading.Thread(target=ai.live, daemon=True)
    ai_thread.start()
    
    time.sleep(3) 
    if not ai_thread.is_alive():
        print("CRITICAL: AI thread failed to start. Exiting."); return

    print("\n\n" + "="*60 + "\n      PHASE 10: DEDUCTIVE INFERENCE\n" + "="*60)
    print("The AI can now synthesize new knowledge from existing facts.")
    print("-" * 60)
    print("\nLESSON 1: Teaching Foundational Facts")
    print("  - Teach the AI two separate, but related, facts.")
    print("  - Type: learn socrates is a man")
    print("  - Type: learn a man is mortal")

    print("\nLESSON 2: Triggering Inference")
    print("  - Now, command the AI to think about the first concept.")
    print("  - Type: infer socrates")
    print("  - Observe as it finds the chain (socrates -> man -> mortal) and")
    print("    synthesizes the new event: 'event_socrates_is_mortal'.")
    
    print("\nLESSON 3: Verifying the New Knowledge")
    print("  - The AI's brain has now physically changed. The new fact is permanent.")
    print("  - You can query this new, synthesized knowledge.")
    print("  - Try: query event_socrates_is_mortal word_socrates")
    print("  - The association should now be very high.")

    print("\n" + "="*60 + "\n")
    
    try:
        ai_thread.join()
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
        ai.is_running = False
        ai_thread.join(timeout=2.0)
    
    ai.shutdown()
    print("Program finished.")

if __name__ == "__main__":
    run_autonomous_learning()