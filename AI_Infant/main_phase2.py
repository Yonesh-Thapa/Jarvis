# full, runnable code here
from src.infant_ai import InfantAI
import time, threading, sys, queue, cv2
import pyttsx3

def run_learning_curriculum():
    print("Initializing AI... please wait.")
    ai = InfantAI()
    
    try:
        tts_engine = pyttsx3.init()
        if tts_engine._driver is None: raise RuntimeError("No TTS driver found")
    except Exception as e:
        print(f"WARNING: pyttsx3 init failed: {e}. Speech disabled."); tts_engine = None

    ai_thread = threading.Thread(target=ai.live, daemon=True)
    ai_thread.start()
    
    time.sleep(5) 
    if not ai_thread.is_alive():
        print("CRITICAL: AI thread failed to start. Exiting."); return

    print("\n\n" + "="*60 + "\n      PHASE 2: LANGUAGE GROUNDING CURRICULUM\n" + "="*60)
    print("The AI is live. Follow the steps below, typing into the '>' prompt.")
    print("-" * 60)
    print("\nLESSON 1: Learning 'red'\n  - Hold a black object. Type: bind black\n  - Then type: assess 0.7")
    print("\nLESSON 2: Learning 'cup'\n  - Hold a charger. Type: bind charger\n  - Then type: assess 0.7")
    print("\nLESSON 3: Association\n  - Hold the BLACK CHARGER. Type: assess 0.8")
    print("\nLESSON 4: Query\n  - Test the memory. Type: query cup red")
    print("\nLESSON 5: Action\n  - Command the AI. Type: do cup action_speak")
    print("\n" + "="*60 + "\nCurriculum complete. Type 'quit' to exit.\n" + "="*60 + "\n")
    
    # This is the main application loop, controlled by the AI's running state.
    try:
        while ai.is_running:
            # Handle Speech
            if tts_engine:
                try:
                    text = ai.speech_queue.get_nowait()
                    if text is None: break
                    tts_engine.say(text); tts_engine.runAndWait()
                except queue.Empty: pass
            
            # Handle Display
            try:
                frame = ai.display_queue.get_nowait()
                if frame is None: break
                cv2.imshow('AI Perception', frame)
            except queue.Empty: pass

            if cv2.waitKey(30) & 0xFF == ord('q'):
                ai.is_running = False
    finally:
        # This block is GUARANTEED to run, ensuring a clean shutdown.
        print("\nMain loop exited. Orchestrating final shutdown.")
        ai.is_running = False # Ensure AI thread knows to stop
        if ai_thread.is_alive(): ai_thread.join(timeout=2.0)
        ai.shutdown()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    run_learning_curriculum()