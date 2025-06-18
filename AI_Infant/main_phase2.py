# full, runnable code here
from src.infant_ai import InfantAI
import time
import threading
import sys
import queue
import cv2

def run_learning_curriculum():
    """
    A guided script for a human to teach the AI its first concepts.
    This script only PRINTS instructions. The user types into the AI's prompt.
    """
    ai = InfantAI()
    
    # Start the main loop in a background thread.
    ai_thread = threading.Thread(target=ai.live, daemon=True)
    ai_thread.start()
    
    # Wait for the AI to boot up and for its command prompt to appear.
    time.sleep(5) 
    
    # --- The instructions are printed to the console for the user to follow ---
    print("\n\n" + "="*60)
    print("      PHASE 2: LANGUAGE GROUNDING CURRICULUM")
    print("="*60)
    print("The AI is now live and waiting for your commands at the '>' prompt.")
    print("Follow the steps below, typing each command and pressing Enter.")
    print("-" * 60)
    
    print("\nLESSON 1: Learning 'red'")
    print("  - Hold a solid RED object in front of the camera.")
    print("  - Wait a few seconds for the AI to see it.")
    print("  - Type: bind red")
    print("  - Type: assess 0.7")

    print("\nLESSON 2: Learning 'cup'")
    print("  - Hold a CUP of any color in front of the camera.")
    print("  - Wait a few seconds.")
    print("  - Type: bind cup")
    print("  - Type: assess 0.7")

    print("\nLESSON 3: Associative Learning")
    print("  - Hold the RED CUP in front of the camera.")
    print("  - Wait a few seconds.")
    print("  - Type: assess 0.8")
    
    print("\nLESSON 4: Querying Knowledge")
    print("  - You can now test the AI's memory.")
    print("  - Type: query cup red")
    print("  (The result should be a high number, like 0.90 or more).")

    print("\nLESSON 5: Grounding an Action")
    print("  - Now, command the AI to speak.")
    print("  - Type: do cup action_speak")
    print("  - Type: do red action_speak")
    
    print("\n" + "="*60)
    print("Curriculum complete. You can continue experimenting or type 'quit'.")
    print("="*60 + "\n")
    
    # The main thread will now wait here until the AI thread finishes
    # (which happens when the user types 'quit').
    ai_thread.join()
    print("AI thread has finished. Exiting program.")

    # --- START OF CHANGE: The Main Thread Display Loop ---
    while ai.is_running:
        try:
            # Wait for a frame from the AI thread
            frame = ai.display_queue.get(timeout=1)
            if frame is None: # Sentinel value means the AI thread has exited
                break
            
            cv2.imshow('AI Perception', frame)
            
            # The waitKey is crucial for the window to update
            if cv2.waitKey(1) & 0xFF == ord('q'):
                ai.is_running = False # Signal the AI thread to stop
                break
                
        except queue.Empty:
            # If the queue is empty, it just means the AI is busy.
            # We can check if the thread is still alive.
            if not ai_thread.is_alive():
                break
            continue
    # --- END OF CHANGE ---

    # Cleanup
    ai.shutdown()
    cv2.destroyAllWindows()
    print("AI thread has finished. Exiting program.")
if __name__ == "__main__":
    run_learning_curriculum()