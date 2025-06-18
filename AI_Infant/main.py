# full, runnable code here
import sys
from src.infant_ai import InfantAI

def main():
    """
    Initializes and runs the InfantAI system.
    """
    try:
        ai = InfantAI()
        ai.live()
    except Exception as e:
        print(f"A fatal error occurred in the AI system: {e}")
        # Add any additional cleanup if necessary
    finally:
        print("\nProgram terminated.")
        sys.exit(0)

if __name__ == "__main__":
    main()