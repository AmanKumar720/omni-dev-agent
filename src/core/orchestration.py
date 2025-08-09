from ..components.document_planner import DocumentPlanner
from ..components.terminal_executor import TerminalExecutor
from ..components.code_development.debugger import CodeDebugger
from ..components.browser_testing.tester import BrowserTester
from ..components.documentation_generator.generator import DocGenerator
from ..components.document_planner_models import TaskPlan, Phase, Step
from ..components.ai_vision.object_detection import ObjectDetectionTask, create_detector
from ..components.ai_vision.computer_vision_analytics import ComputerVisionAnalyticsTask, analytics_agent
from ..utils.logging_config import get_logger
from typing import List, Dict, Any, Optional
import asyncio
import json

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self):
        self.components = {}
        self.planner = DocumentPlanner()
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
    
    async def handle_request(self, type: str, task: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Handle different types of requests including vision tasks
        
        Args:
            type: Request type (e.g., 'vision', 'code', 'documentation')
            task: Specific task within the type (e.g., 'object_detection', 'analytics')
            payload: Task-specific data payload
            **kwargs: Additional parameters
            
        Returns:
            Standardized response dictionary with status, data, and metadata
        """
        logger.info(f"Handling request - Type: {type}, Task: {task}")
        
        try:
            if type == 'vision':
                return await self._handle_vision_request(task, payload, **kwargs)
            elif type == 'code':
                return await self._handle_code_request(task, payload, **kwargs)
            elif type == 'documentation':
                return await self._handle_documentation_request(task, payload, **kwargs)
            else:
                return {
                    'status': 'error',
                    'error': f'Unsupported request type: {type}',
                    'data': None,
                    'metadata': {
                        'request_type': type,
                        'task': task,
                        'timestamp': self._get_timestamp()
                    }
                }
        
        except Exception as e:
            logger.error(f"Error handling request {type}:{task} - {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'data': None,
                'metadata': {
                    'request_type': type,
                    'task': task,
                    'timestamp': self._get_timestamp()
                }
            }
    
    async def _handle_vision_request(self, task: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Handle vision-related requests
        
        Args:
            task: Vision task type ('object_detection', 'analytics', etc.)
            payload: Vision task payload
            **kwargs: Additional parameters
            
        Returns:
            Standardized vision response
        """
        logger.info(f"Processing vision task: {task}")
        
        if task == 'object_detection':
            return await self._handle_object_detection(payload, **kwargs)
        elif task == 'analytics':
            return await self._handle_computer_vision_analytics(payload, **kwargs)
        else:
            return {
                'status': 'error',
                'error': f'Unsupported vision task: {task}',
                'data': None,
                'metadata': {
                    'vision_task': task,
                    'timestamp': self._get_timestamp()
                }
            }
    
    async def _handle_object_detection(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Handle object detection requests
        
        Args:
            payload: Object detection payload containing:
                - frame: Single frame for detection (optional)
                - batch_frames: Multiple frames for batch detection (optional)
                - conf_threshold: Confidence threshold (default: 0.25)
                - model_name: YOLOv8 model name (default: 'yolov8n')
            **kwargs: Additional parameters
            
        Returns:
            Object detection results in standardized format
        """
        try:
            # Extract parameters
            model_name = payload.get('model_name', 'yolov8n')
            conf_threshold = payload.get('conf_threshold', 0.25)
            frame = payload.get('frame')
            batch_frames = payload.get('batch_frames')
            
            # Validate input
            if frame is None and batch_frames is None:
                return {
                    'status': 'error',
                    'error': 'Either frame or batch_frames must be provided',
                    'data': None,
                    'metadata': {
                        'vision_task': 'object_detection',
                        'timestamp': self._get_timestamp()
                    }
                }
            
            # Create and execute object detection task
            task_id = f"obj_det_{self._get_timestamp()}"
            detection_task = ObjectDetectionTask(task_id, model_name=model_name)
            
            # Prepare input data
            input_data = {
                'conf_threshold': conf_threshold
            }
            
            if frame is not None:
                input_data['frame'] = frame
            if batch_frames is not None:
                input_data['batch_frames'] = batch_frames
            
            # Execute task
            logger.info(f"Executing object detection task {task_id}")
            result = await detection_task.execute(input_data)
            
            # Return standardized response
            if result.status.value == 'completed':
                return {
                    'status': 'success',
                    'data': result.data,
                    'confidence': result.confidence,
                    'metadata': {
                        'task_id': task_id,
                        'vision_task': 'object_detection',
                        'model_name': model_name,
                        'conf_threshold': conf_threshold,
                        'processing_type': result.metadata.get('processing_type', 'unknown'),
                        'timestamp': self._get_timestamp()
                    }
                }
            else:
                return {
                    'status': 'error',
                    'error': result.error_message or 'Object detection failed',
                    'data': result.data,
                    'metadata': {
                        'task_id': task_id,
                        'vision_task': 'object_detection',
                        'timestamp': self._get_timestamp()
                    }
                }
        
        except Exception as e:
            logger.error(f"Object detection error: {str(e)}")
            return {
                'status': 'error',
                'error': f'Object detection failed: {str(e)}',
                'data': None,
                'metadata': {
                    'vision_task': 'object_detection',
                    'timestamp': self._get_timestamp()
                }
            }
    
    async def _handle_computer_vision_analytics(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Handle computer vision analytics requests
        
        Args:
            payload: Analytics payload containing:
                - input_data: Video file path, frame, or frame sequence
                - analytics_config: Configuration for analytics processing
            **kwargs: Additional parameters
            
        Returns:
            Computer vision analytics results in standardized format
        """
        try:
            # Extract parameters
            input_data = payload.get('input_data')
            analytics_config = payload.get('analytics_config', {})
            
            if input_data is None:
                return {
                    'status': 'error',
                    'error': 'input_data must be provided for analytics',
                    'data': None,
                    'metadata': {
                        'vision_task': 'analytics',
                        'timestamp': self._get_timestamp()
                    }
                }
            
            # Create analytics task
            task_id = analytics_agent.create_analytics_task(analytics_config)
            
            # Execute task
            logger.info(f"Executing computer vision analytics task {task_id}")
            result = await analytics_agent.execute_task(task_id, input_data)
            
            # Clean up task
            analytics_agent.unregister_task(task_id)
            
            # Return standardized response
            if result and result.status.value == 'completed':
                return {
                    'status': 'success',
                    'data': result.data,
                    'confidence': result.confidence,
                    'metadata': {
                        'task_id': task_id,
                        'vision_task': 'analytics',
                        'analytics_config': analytics_config,
                        'timestamp': self._get_timestamp()
                    }
                }
            else:
                return {
                    'status': 'error',
                    'error': result.error_message if result else 'Analytics task failed',
                    'data': result.data if result else None,
                    'metadata': {
                        'task_id': task_id,
                        'vision_task': 'analytics',
                        'timestamp': self._get_timestamp()
                    }
                }
        
        except Exception as e:
            logger.error(f"Computer vision analytics error: {str(e)}")
            return {
                'status': 'error',
                'error': f'Analytics failed: {str(e)}',
                'data': None,
                'metadata': {
                    'vision_task': 'analytics',
                    'timestamp': self._get_timestamp()
                }
            }
    
    async def _handle_code_request(self, task: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Handle code-related requests (placeholder for future implementation)
        """
        return {
            'status': 'error',
            'error': 'Code requests not yet implemented',
            'data': None,
            'metadata': {
                'request_type': 'code',
                'task': task,
                'timestamp': self._get_timestamp()
            }
        }
    
    async def _handle_documentation_request(self, task: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Handle documentation-related requests (placeholder for future implementation)
        """
        return {
            'status': 'error',
            'error': 'Documentation requests not yet implemented',
            'data': None,
            'metadata': {
                'request_type': 'documentation',
                'task': task,
                'timestamp': self._get_timestamp()
            }
        }
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp as ISO string
        """
        from datetime import datetime
        return datetime.now().isoformat()


# Usage example
if __name__ == "__main__":
    # Traditional orchestrator usage
    orchestrator = Orchestrator()
    orchestrator.execute(
        "Develop a web feedback form feature with backend API and frontend interface"
    )
    
    # Vision task routing example
    async def vision_example():
        import numpy as np
        
        # Create sample image data
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Object detection example
        detection_result = await orchestrator.handle_request(
            type='vision',
            task='object_detection',
            payload={
                'frame': sample_frame,
                'conf_threshold': 0.25,
                'model_name': 'yolov8n'
            }
        )
        
        print("Object Detection Result:")
        print(json.dumps(detection_result, indent=2, default=str))
        
        # Analytics example
        analytics_result = await orchestrator.handle_request(
            type='vision',
            task='analytics',
            payload={
                'input_data': sample_frame,
                'analytics_config': {
                    'motion_config': {'threshold': 25},
                    'reasoning_config': {
                        'activity_detection': {'enabled': True}
                    }
                }
            }
        )
        
        print("\nAnalytics Result:")
        print(json.dumps(analytics_result, indent=2, default=str))
    
    # Uncomment to run vision examples
    # asyncio.run(vision_example())
