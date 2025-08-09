"""
OmniAgent: Capable agent for omni-dev-agent project
Handles logging, problem detection, version issues, and project status using CentralMemory.
"""
from central_memory import CentralMemory

class OmniAgent:
    def __init__(self, name="OmniAgent"):
        self.name = name
        self.memory = CentralMemory()
        self.memory.track_contributor(self.name, "Agent initialized.")

    def handle_problem(self, description, root_cause):
        self.memory.track_poison(description, root_cause)
        print(f"Problem handled: {description} | Cause: {root_cause}")

    def handle_version_issue(self, package, current_version, required_version, auto_update_result):
        self.memory.track_version_problem(package, current_version, required_version, auto_update_result)
        print(f"Version issue: {package} {current_version} -> {required_version} | Result: {auto_update_result}")

    def update_project_status(self, status, details=None):
        self.memory.track_project_status(status, details)
        print(f"Project status updated: {status}")

    def get_memory_log(self):
        events = self.memory.get_events()
        for event in events:
            print(event)
        return events

# Example usage:
if __name__ == "__main__":
    agent = OmniAgent()
    agent.handle_problem("Dependency install failed", "ResolutionTooDeep error.")
    agent.handle_version_issue("torch", "2.7.1", "2.8.0", "Auto-update failed.")
    agent.update_project_status("install_failed", "See log for details.")
    agent.get_memory_log()
