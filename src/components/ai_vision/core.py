# src/components/ai_vision/core.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Enumeration of possible task statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VisionResult:
    """Container for vision task results"""
    task_id: str
    status: TaskStatus
    data: Any
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class VisionTask(ABC):
    """Abstract base class for vision tasks"""
    
    def __init__(self, task_id: str, task_type: str, **kwargs):
        self.task_id = task_id
        self.task_type = task_type
        self.status = TaskStatus.PENDING
        self.result: Optional[VisionResult] = None
        self.metadata = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(self, input_data: Any) -> VisionResult:
        """
        Execute the vision task
        
        Args:
            input_data: Input data for the task (image, video, etc.)
            
        Returns:
            VisionResult: Result of the task execution
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for the task
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        pass
    
    def get_status(self) -> TaskStatus:
        """Get current task status"""
        return self.status
    
    def get_result(self) -> Optional[VisionResult]:
        """Get task result"""
        return self.result


class AIVisionAgent:
    """Base class for AI Vision agents that manage vision tasks"""
    
    def __init__(self, agent_id: str, name: str, **config):
        self.agent_id = agent_id
        self.name = name
        self.config = config
        self.tasks: Dict[str, VisionTask] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent with configuration"""
        self.logger.info(f"Initializing AI Vision Agent: {self.name} (ID: {self.agent_id})")
        
        # Set up default configurations
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
        self.task_timeout = self.config.get('task_timeout', 300)  # 5 minutes default
        self.enable_caching = self.config.get('enable_caching', True)
        
        self.logger.info(f"Agent initialized with max_concurrent_tasks={self.max_concurrent_tasks}")
    
    def register_task(self, task: VisionTask) -> bool:
        """
        Register a vision task with the agent
        
        Args:
            task: VisionTask instance to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            if task.task_id in self.tasks:
                self.logger.warning(f"Task {task.task_id} already registered")
                return False
            
            if len(self.tasks) >= self.max_concurrent_tasks:
                self.logger.error(f"Maximum concurrent tasks ({self.max_concurrent_tasks}) reached")
                return False
            
            self.tasks[task.task_id] = task
            self.logger.info(f"Task {task.task_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register task {task.task_id}: {str(e)}")
            return False
    
    def unregister_task(self, task_id: str) -> bool:
        """
        Unregister a task from the agent
        
        Args:
            task_id: ID of the task to unregister
            
        Returns:
            bool: True if unregistration successful, False otherwise
        """
        try:
            if task_id not in self.tasks:
                self.logger.warning(f"Task {task_id} not found")
                return False
            
            del self.tasks[task_id]
            self.logger.info(f"Task {task_id} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister task {task_id}: {str(e)}")
            return False
    
    def get_task(self, task_id: str) -> Optional[VisionTask]:
        """
        Get a registered task by ID
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Optional[VisionTask]: The task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        List all registered tasks with their status
        
        Returns:
            List[Dict[str, Any]]: List of task information
        """
        task_info = []
        for task_id, task in self.tasks.items():
            task_info.append({
                'task_id': task_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'metadata': task.metadata
            })
        return task_info
    
    async def execute_task(self, task_id: str, input_data: Any) -> Optional[VisionResult]:
        """
        Execute a registered task
        
        Args:
            task_id: ID of the task to execute
            input_data: Input data for the task
            
        Returns:
            Optional[VisionResult]: Task result if successful, None otherwise
        """
        task = self.get_task(task_id)
        if not task:
            self.logger.error(f"Task {task_id} not found")
            return None
        
        try:
            self.logger.info(f"Executing task {task_id}")
            task.status = TaskStatus.IN_PROGRESS
            
            # Validate input
            if not task.validate_input(input_data):
                task.status = TaskStatus.FAILED
                error_result = VisionResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    data=None,
                    confidence=0.0,
                    error_message="Input validation failed"
                )
                task.result = error_result
                return error_result
            
            # Execute task
            result = await task.execute(input_data)
            task.result = result
            task.status = result.status
            
            self.logger.info(f"Task {task_id} completed with status: {result.status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task_id} execution failed: {str(e)}")
            task.status = TaskStatus.FAILED
            error_result = VisionResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                data=None,
                confidence=0.0,
                error_message=str(e)
            )
            task.result = error_result
            return error_result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information and status
        
        Returns:
            Dict[str, Any]: Agent information
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'active_tasks': len(self.tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'config': self.config
        }
