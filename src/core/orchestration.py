from ..components.document_planner import DocumentPlanner
from ..components.terminal_executor import TerminalExecutor
from ..components.code_development.debugger import CodeDebugger
from ..components.browser_testing.tester import BrowserTester
from ..components.documentation_generator.generator import DocGenerator
from ..components.document_planner_models import TaskPlan, Phase, Step
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class Orchestrator:
    def __init__(self):
        self.components = {}
        self.planner = DocumentPlanner()
        self.load_components()

    def load_components(self):
        self.components['terminal_executor'] = TerminalExecutor()
        self.components['code_debugger'] = CodeDebugger()
        self.components['browser_tester'] = BrowserTester()
        self.components['doc_generator'] = DocGenerator()

    def parse_request(self, request):
        logger.info(f"Parsing request: {request}")
        return request

    def decompose_task(self, task) -> List[Phase]:
        logger.info(f"Generating plan for task: {task}")
        task_plan = self.planner.generate_plan(task)
        markdown_plan = self.planner.format_plan_as_markdown(task_plan.dict())
        logger.info("\nGenerated Plan in Markdown:\n" + markdown_plan)
        return task_plan.phases

    def execute(self, request):
        task = self.parse_request(request)
        sub_tasks = self.decompose_task(task)

        for phase in sub_tasks:
            self.handle_sub_task(phase)

    def handle_sub_task(self, phase: Phase):
        logger.info(f"Handling phase: {phase.phase_name}")
        for step in phase.steps:
            logger.info(f"Executing step: {step.description}")
            step_lower = step.description.lower()

            if "run command" in step_lower:
                command = step.replace("run command ", "").strip()
                result = self.components['terminal_executor'].execute(command)
                logger.info(f"Command result: {result}")
            elif "lint" in step_lower and "file" in step_lower:
                file_path = step.split(" ")[-1] # Simple extraction, needs refinement
                result = self.components['code_debugger'].lint(file_path)
                logger.info(f"Lint result: {result}")
            elif "run tests" in step_lower or "write comprehensive unit tests" in step_lower or "write frontend unit and integration tests" in step_lower or "end-to-end testing" in step_lower:
                # For now, just log that tests are being run. Actual test path extraction needs more logic.
                logger.info(f"Performing: {step} (using CodeDebugger or BrowserTester)")
                # test_path = step.split(" ")[-1] # Needs more robust extraction
                # result = self.components['code_debugger'].run_tests(test_path)
                # logger.info(f"Test result: {result}")
            elif "navigate to url" in step_lower:
                url = step.replace("navigate to url ", "").strip()
                tester = self.components['browser_tester']
                tester.navigate_to_url(url)
                title = tester.get_page_title()
                logger.info(f"Navigated to {url}, page title: {title}")
                tester.close_browser()
            elif "generate documentation" in step_lower or "document api endpoints" in step_lower or "create user guides" in step_lower or "write technical specifications" in step_lower or "create deployment and maintenance guides" in step_lower:
                # Assuming source and output directories are known or can be inferred
                source_dir = "docs_source" # Placeholder
                output_dir = "docs_output" # Placeholder
                result = self.components['doc_generator'].generate_html(source_dir, output_dir)
                logger.info(f"Documentation generation result: {result}")
            elif "design api endpoints" in step_lower or \
                 "set up database schema" in step_lower or \
                 "implement api routes" in step_lower or \
                 "add authentication and authorization" in step_lower or \
                 "set up api documentation" in step_lower or \
                 "create wireframes" in step_lower or \
                 "set up frontend project structure" in step_lower or \
                 "implement core ui components" in step_lower or \
                 "add form validation" in step_lower or \
                 "implement responsive design" in step_lower or \
                 "connect frontend to backend" in step_lower or \
                 "implement error handling and loading states" in step_lower or \
                 "performance testing and optimization" in step_lower or \
                 "cross-browser compatibility testing" in step_lower:
                logger.info(f"Performing: {step}")
            else:
                logger.warning(f"Unrecognized step: {step}")
        logger.info("---")

# Usage example
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.execute("Develop a web feedback form feature with backend API and frontend interface")
