from ..components.document_planner import DocumentPlanner

class Orchestrator:
    def __init__(self):
        self.components = {}
        self.planner = DocumentPlanner()
        self.load_components()

    def load_components(self):
        # Register or load available components
        # Example: self.components['component_name'] = ComponentClass()
        pass

    def parse_request(self, request):
        # Parse user requests
        print(f"Parsing request: {request}")
        return request

    def decompose_task(self, task):
        # Decompose task using Document Planner
        print(f"Generating plan for task: {task}")
        plan = self.planner.generate_plan(task)
        markdown_plan = self.planner.format_plan_as_markdown(plan)
        print("\nGenerated Plan in Markdown:")
        print(markdown_plan)
        return plan["phases"]

    def execute(self, request):
        # Main execution loop
        task = self.parse_request(request)
        sub_tasks = self.decompose_task(task)

        for phase in sub_tasks:
            self.handle_sub_task(phase)

    def handle_sub_task(self, phase):
        # Find and execute the appropriate component for the sub-task
        print(f"Handling phase: {phase['phase_name']}")
        for step in phase['steps']:
            print(f"Executing step: {step}")
        print("---")

# Usage example
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.execute("Develop a web feedback form feature with backend API and frontend interface")
