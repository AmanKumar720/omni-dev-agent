import sys
from .core.orchestration import Orchestrator

def main():
    print("Starting Omni-Dev Agent...")
    
    # Initialize the orchestrator
    orchestrator = Orchestrator()
    
    # Example usage
    sample_request = "Develop a web feedback form feature with backend API and frontend interface"
    print(f"\nProcessing request: {sample_request}")
    orchestrator.execute(sample_request)
    
    print("\nOmni-Dev Agent session completed.")

if __name__ == "__main__":
    main()
