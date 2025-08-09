"""
Central Memory Plan Module for omni-dev-agent
Tracks contributors, problems ("poison"), project status, version issues, and automatic update events.
"""
import os
import datetime
import json

MEMORY_LOG = "system_manager.log"

class CentralMemory:
    def __init__(self, log_file=MEMORY_LOG):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("")

    def log_event(self, who, what, why, capabilities=None, auto_action=None):
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "who": who,
            "what": what,
            "why": why,
            "capabilities": capabilities or [],
            "auto_action": auto_action or ""
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + "\n")

    def get_events(self):
        events = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
        return events

    def track_version_problem(self, package, current_version, required_version, auto_update_result):
        self.log_event(
            who="system",
            what=f"Version problem: {package} {current_version} -> {required_version}",
            why="Version conflict or outdated package detected.",
            capabilities=["detect_version_conflict", "auto_update_attempt"],
            auto_action=auto_update_result
        )

    def track_poison(self, description, root_cause):
        self.log_event(
            who="system",
            what=f"Poison detected: {description}",
            why=root_cause,
            capabilities=["problem_detection", "root_cause_analysis"]
        )

    def track_contributor(self, name, action):
        self.log_event(
            who=name,
            what=action,
            why="Contributor action logged."
        )

    def track_project_status(self, status, details=None):
        self.log_event(
            who="system",
            what=f"Project status: {status}",
            why="Status update.",
            capabilities=["status_tracking"],
            auto_action=details
        )

# Example usage:
if __name__ == "__main__":
    memory = CentralMemory()
    memory.track_contributor("Aman Kumar", "Initialized project memory plan.")
    memory.track_poison("Dependency resolution failed", "Conflicting torch/ultralytics versions.")
    memory.track_version_problem("torch", "2.7.1", "2.8.0", "Auto-update failed: ResolutionTooDeep.")
    memory.track_project_status("install_failed", "See log for details.")
    print(memory.get_events())
