# src/components/ai_vision/pipelines/__init__.py

"""
AI Vision Pipelines Module

This module contains processing pipelines that chain together multiple vision tasks:
- End-to-end image processing pipelines
- Multi-stage object detection pipelines
- Image enhancement and restoration pipelines
- Real-time video processing pipelines
- Batch processing pipelines
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import logging
from ..core import VisionTask, VisionResult

logger = logging.getLogger(__name__)


class VisionPipeline(ABC):
    """Abstract base class for vision processing pipelines"""
    
    def __init__(self, pipeline_id: str, name: str, **config):
        self.pipeline_id = pipeline_id
        self.name = name
        self.config = config
        self.stages: List[VisionTask] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(self, input_data: Any) -> VisionResult:
        """
        Execute the complete pipeline
        
        Args:
            input_data: Input data for the pipeline
            
        Returns:
            VisionResult: Final result of the pipeline
        """
        pass
    
    def add_stage(self, stage: VisionTask) -> None:
        """
        Add a processing stage to the pipeline
        
        Args:
            stage: VisionTask to add as a pipeline stage
        """
        self.stages.append(stage)
        self.logger.info(f"Added stage {stage.task_id} to pipeline {self.pipeline_id}")
    
    def remove_stage(self, task_id: str) -> bool:
        """
        Remove a stage from the pipeline
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            bool: True if stage was removed, False if not found
        """
        for i, stage in enumerate(self.stages):
            if stage.task_id == task_id:
                del self.stages[i]
                self.logger.info(f"Removed stage {task_id} from pipeline {self.pipeline_id}")
                return True
        return False
    
    def get_stages(self) -> List[str]:
        """
        Get list of stage IDs in the pipeline
        
        Returns:
            List of task IDs representing pipeline stages
        """
        return [stage.task_id for stage in self.stages]


class SequentialPipeline(VisionPipeline):
    """Pipeline that executes stages sequentially"""
    
    async def execute(self, input_data: Any) -> VisionResult:
        """
        Execute all stages sequentially
        
        Args:
            input_data: Initial input data
            
        Returns:
            VisionResult: Result from the final stage
        """
        current_data = input_data
        results = []
        
        for stage in self.stages:
            try:
                self.logger.info(f"Executing stage {stage.task_id}")
                result = await stage.execute(current_data)
                results.append(result)
                
                if result.status.value != "completed":
                    self.logger.error(f"Stage {stage.task_id} failed: {result.error_message}")
                    return result
                
                # Use the output of current stage as input for next stage
                current_data = result.data
                
            except Exception as e:
                self.logger.error(f"Exception in stage {stage.task_id}: {str(e)}")
                from ..core import TaskStatus
                return VisionResult(
                    task_id=self.pipeline_id,
                    status=TaskStatus.FAILED,
                    data=None,
                    confidence=0.0,
                    error_message=f"Pipeline failed at stage {stage.task_id}: {str(e)}"
                )
        
        # Return result from final stage
        if results:
            final_result = results[-1]
            final_result.metadata = final_result.metadata or {}
            final_result.metadata['pipeline_results'] = results
            return final_result
        
        from ..core import TaskStatus
        return VisionResult(
            task_id=self.pipeline_id,
            status=TaskStatus.COMPLETED,
            data=input_data,
            confidence=1.0,
            metadata={'message': 'Empty pipeline - no stages to execute'}
        )


# Pipeline registry for dynamic loading
PIPELINE_REGISTRY: Dict[str, Any] = {
    'sequential': SequentialPipeline
}

def register_pipeline(name: str, pipeline_class: Any) -> None:
    """
    Register a pipeline class in the global registry
    
    Args:
        name: Pipeline name/identifier
        pipeline_class: Pipeline class
    """
    PIPELINE_REGISTRY[name] = pipeline_class
    logger.info(f"Registered pipeline: {name}")

def get_pipeline_class(name: str) -> Optional[Any]:
    """
    Get a pipeline class from the registry
    
    Args:
        name: Pipeline name
        
    Returns:
        Pipeline class if found, None otherwise
    """
    return PIPELINE_REGISTRY.get(name)

def list_pipelines() -> List[str]:
    """
    List all registered pipeline types
    
    Returns:
        List of pipeline names
    """
    return list(PIPELINE_REGISTRY.keys())

__all__ = [
    'VisionPipeline',
    'SequentialPipeline',
    'PIPELINE_REGISTRY',
    'register_pipeline',
    'get_pipeline_class',
    'list_pipelines'
]
