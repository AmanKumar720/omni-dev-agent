from ..components.document_planner import DocumentPlanner
from ..components.terminal_executor import TerminalExecutor
from ..components.code_development.debugger import CodeDebugger
from ..components.browser_testing.tester import BrowserTester
from ..components.documentation_generator.generator import DocGenerator
from ..components.document_planner_models import TaskPlan, Phase, Step
from ..utils.logging_config import get_logger
from .background_worker import BackgroundTaskManager, TaskPriority
from ..config.background_config import auto_load_config

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, use_background_processing: bool = True):
        self.components = {}
        self.planner = DocumentPlanner()
        self.use_background_processing = use_background_processing
        
        # Initialize background task manager if enabled
        if self.use_background_processing:
            config = auto_load_config()
            worker_config = config.threaded_worker.__dict__ if config.worker_type == "threaded" else {}
            self.task_manager = BackgroundTaskManager(
                worker_type=config.worker_type,
                monitor_enabled=config.monitoring.enabled,
                **worker_config
            )
        else:
            self.task_manager = None
        
        self.load_components()

    def load_components(self):
        self.components["terminal_executor"] = TerminalExecutor()
        self.components["code_debugger"] = CodeDebugger()
        self.components["browser_tester"] = BrowserTester()
        self.components["doc_generator"] = DocGenerator()

    def parse_request(self, request):
        logger.info(f"Parsing request: {request}")
        return request

    def decompose_task(self, task) -> List[Phase]:
        logger.info(f"Generating plan for task: {task}")
        task_plan = self.planner.generate_plan(task)
        markdown_plan = self.planner.format_plan_as_markdown(task_plan.dict())
        logger.info("\nGenerated Plan in Markdown:\n" + markdown_plan)
        return task_plan.phases

    def execute(self, request, use_background: bool = None):
        """Execute a request with optional background processing."""
        task = self.parse_request(request)
        sub_tasks = self.decompose_task(task)
        
        # Determine if we should use background processing
        use_bg = use_background if use_background is not None else self.use_background_processing
        
        if use_bg and self.task_manager:
            # Submit phases as background tasks
            task_ids = []
            for phase in sub_tasks:
                task_id = self.submit_phase_as_background_task(phase)
                task_ids.append(task_id)
            
            logger.info(f"Submitted {len(task_ids)} phases as background tasks: {task_ids}")
            return task_ids
        else:
            # Execute synchronously
            for phase in sub_tasks:
                self.handle_sub_task(phase)
            return None

    def handle_sub_task(self, phase: Phase):
        logger.info(f"Handling phase: {phase.phase_name}")
        for step in phase.steps:
            logger.info(f"Executing step: {step.description}")
            step_lower = step.description.lower()

            if "run command" in step_lower:
                command = step.replace("run command ", "").strip()
                result = self.components["terminal_executor"].execute(command)
                logger.info(f"Command result: {result}")
            elif "lint" in step_lower and "file" in step_lower:
                file_path = step.split(" ")[-1]  # Simple extraction, needs refinement
                result = self.components["code_debugger"].lint(file_path)
                logger.info(f"Lint result: {result}")
            elif (
                "run tests" in step_lower
                or "write comprehensive unit tests" in step_lower
                or "write frontend unit and integration tests" in step_lower
                or "end-to-end testing" in step_lower
            ):
                # For now, just log that tests are being run. Actual test path extraction needs more logic.
                logger.info(f"Performing: {step} (using CodeDebugger or BrowserTester)")
                # test_path = step.split(" ")[-1] # Needs more robust extraction
                # result = self.components['code_debugger'].run_tests(test_path)
                # logger.info(f"Test result: {result}")
            elif "navigate to url" in step_lower:
                url = step.replace("navigate to url ", "").strip()
                tester = self.components["browser_tester"]
                tester.navigate_to_url(url)
                title = tester.get_page_title()
                logger.info(f"Navigated to {url}, page title: {title}")
                tester.close_browser()
            elif (
                "generate documentation" in step_lower
                or "document api endpoints" in step_lower
                or "create user guides" in step_lower
                or "write technical specifications" in step_lower
                or "create deployment and maintenance guides" in step_lower
            ):
                # Assuming source and output directories are known or can be inferred
                source_dir = "docs_source"  # Placeholder
                output_dir = "docs_output"  # Placeholder
                result = self.components["doc_generator"].generate_html(
                    source_dir, output_dir
                )
                logger.info(f"Documentation generation result: {result}")
            elif (
                "design api endpoints" in step_lower
                or "set up database schema" in step_lower
                or "implement api routes" in step_lower
                or "add authentication and authorization" in step_lower
                or "set up api documentation" in step_lower
                or "create wireframes" in step_lower
                or "set up frontend project structure" in step_lower
                or "implement core ui components" in step_lower
                or "add form validation" in step_lower
                or "implement responsive design" in step_lower
                or "connect frontend to backend" in step_lower
                or "implement error handling and loading states" in step_lower
                or "performance testing and optimization" in step_lower
                or "cross-browser compatibility testing" in step_lower
            ):
                logger.info(f"Performing: {step}")
            else:
                logger.warning(f"Unrecognized step: {step}")
        logger.info("---")
    
    def submit_phase_as_background_task(self, phase: Phase) -> str:
        """Submit a phase as a background task."""
        if not self.task_manager:
            raise RuntimeError("Background task manager not available")
        
        # Determine priority based on phase content
        priority = self._determine_phase_priority(phase)
        
        task_id = self.task_manager.submit_task(
            func=self.handle_sub_task,
            phase,
            name=f"phase_{phase.phase_name}",
            priority=priority,
            timeout=None,  # Let long-running tasks complete
            tags=["orchestrator", "phase", phase.phase_name]
        )
        
        logger.info(f"Submitted phase '{phase.phase_name}' as background task {task_id}")
        return task_id
    
    def _determine_phase_priority(self, phase: Phase) -> TaskPriority:
        """Determine the priority of a phase based on its content."""
        phase_name_lower = phase.phase_name.lower()
        
        # High priority for critical operations
        if any(keyword in phase_name_lower for keyword in ['security', 'auth', 'critical', 'urgent']):
            return TaskPriority.HIGH
        
        # Normal priority for most operations
        if any(keyword in phase_name_lower for keyword in ['development', 'implementation', 'testing']):
            return TaskPriority.NORMAL
        
        # Low priority for documentation and cleanup
        if any(keyword in phase_name_lower for keyword in ['documentation', 'cleanup', 'maintenance']):
            return TaskPriority.LOW
        
        return TaskPriority.NORMAL
    
    def get_background_task_status(self, task_id: str):
        """Get the status of a background task."""
        if not self.task_manager:
            return None
        
        return self.task_manager.get_task_status(task_id)
    
    def get_background_task_result(self, task_id: str):
        """Get the result of a background task."""
        if not self.task_manager:
            return None
        
        task = self.task_manager.get_task(task_id)
        return task.result if task else None
    
    def cancel_background_task(self, task_id: str) -> bool:
        """Cancel a background task."""
        if not self.task_manager:
            return False
        
        return self.task_manager.cancel_task(task_id)
    
    def get_background_stats(self):
        """Get background processing statistics."""
        if not self.task_manager:
            return None
        
        return self.task_manager.get_stats()
    
    def shutdown(self):
        """Shutdown the orchestrator and background processing."""
        logger.info("Shutting down Orchestrator...")
        
        if self.task_manager:
            self.task_manager.shutdown()
        
        logger.info("Orchestrator shutdown complete")


# Usage example
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.execute(
        "Develop a web feedback form feature with backend API and frontend interface"
    )
