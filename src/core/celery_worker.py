from celery import Celery, Task
from celery.result import AsyncResult
from celery.exceptions import Retry
import os
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from .background_worker import (
    BackgroundWorker, BackgroundTask, TaskStatus, TaskPriority
)

logger = logging.getLogger(__name__)


class CeleryConfig:
    """Celery configuration class."""
    
    # Broker and backend configuration
    broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Task configuration
    task_serializer = 'json'
    accept_content = ['json']
    result_serializer = 'json'
    timezone = 'UTC'
    enable_utc = True
    
    # Worker configuration
    worker_prefetch_multiplier = 1
    task_acks_late = True
    worker_max_tasks_per_child = 1000
    
    # Result configuration
    result_expires = 3600  # 1 hour
    
    # Task routing
    task_routes = {
        'omni_dev.tasks.*': {'queue': 'omni_dev'},
        'omni_dev.priority.*': {'queue': 'priority'},
        'omni_dev.scheduled.*': {'queue': 'scheduled'},
    }
    
    # Beat scheduler configuration (for scheduled tasks)
    beat_schedule = {}


# Create Celery app
celery_app = Celery('omni_dev_agent')
celery_app.config_from_object(CeleryConfig)


class CustomTask(Task):
    """Custom Celery task class with enhanced error handling and metadata."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        
        # Update task status in our tracking system
        if hasattr(self, 'worker_instance'):
            self.worker_instance._update_task_status(
                task_id, TaskStatus.FAILED, error=str(exc)
            )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {task_id} completed successfully")
        
        # Update task status in our tracking system
        if hasattr(self, 'worker_instance'):
            self.worker_instance._update_task_status(
                task_id, TaskStatus.COMPLETED, result=retval
            )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.info(f"Task {task_id} retrying: {exc}")
        
        # Update task status in our tracking system
        if hasattr(self, 'worker_instance'):
            self.worker_instance._update_task_status(
                task_id, TaskStatus.RETRYING, error=str(exc)
            )


@celery_app.task(base=CustomTask, bind=True)
def execute_background_task(self, task_data: dict):
    """
    Execute a background task with the given function and parameters.
    
    Args:
        task_data: Dictionary containing task information including
                  function name, args, kwargs, and metadata
    """
    task_id = self.request.id
    func_name = task_data.get('func_name')
    module_name = task_data.get('module_name')
    args = task_data.get('args', [])
    kwargs = task_data.get('kwargs', {})
    timeout = task_data.get('timeout')
    
    logger.info(f"Executing task {task_id}: {func_name}")
    
    try:
        # Import and execute the function
        if module_name:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
        else:
            # If no module specified, assume it's in globals
            func = globals().get(func_name)
            if not func:
                raise ValueError(f"Function {func_name} not found")
        
        # Execute with timeout if specified
        if timeout:
            # Celery handles timeout automatically if configured
            result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        return result
        
    except Exception as exc:
        logger.error(f"Task {task_id} failed: {exc}")
        raise


@celery_app.task(base=CustomTask)
def execute_scheduled_task(task_data: dict):
    """Execute a scheduled background task."""
    return execute_background_task.apply_async(args=[task_data])


class CeleryBackgroundWorker(BackgroundWorker):
    """Celery-based background worker for distributed task execution."""
    
    def __init__(self, app: Optional[Celery] = None):
        self.app = app or celery_app
        self.tasks: Dict[str, BackgroundTask] = {}
        self.celery_results: Dict[str, AsyncResult] = {}
        
        # Set worker instance reference for task callbacks
        execute_background_task.worker_instance = self
        execute_scheduled_task.worker_instance = self
        
        logger.info("CeleryBackgroundWorker initialized")
    
    def submit_task(self, task: BackgroundTask) -> str:
        """Submit a task for background execution via Celery."""
        
        # Store task metadata
        self.tasks[task.id] = task
        
        # Prepare task data for Celery
        task_data = {
            'task_id': task.id,
            'func_name': task.func.__name__ if hasattr(task.func, '__name__') else str(task.func),
            'module_name': getattr(task.func, '__module__', None),
            'args': task.args,
            'kwargs': task.kwargs,
            'timeout': task.timeout,
            'priority': task.priority.value,
            'max_retries': task.max_retries,
            'tags': task.tags,
            'metadata': task.metadata
        }
        
        # Choose queue based on priority
        queue = self._get_queue_for_priority(task.priority)
        
        # Submit to Celery
        celery_result = execute_background_task.apply_async(
            args=[task_data],
            task_id=task.id,
            queue=queue,
            retry=task.max_retries > 0,
            retry_policy={
                'max_retries': task.max_retries,
                'interval_start': 1,
                'interval_step': 2,
                'interval_max': 60,
            } if task.max_retries > 0 else None
        )
        
        # Store Celery result for status tracking
        self.celery_results[task.id] = celery_result
        
        # Update task status
        task.status = TaskStatus.PENDING
        
        logger.info(f"Task {task.id} ({task.name}) submitted to Celery queue: {queue}")
        return task.id
    
    def _get_queue_for_priority(self, priority: TaskPriority) -> str:
        """Get the appropriate queue based on task priority."""
        if priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            return 'priority'
        return 'omni_dev'
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        # Update status from Celery if available
        if task_id in self.celery_results:
            celery_result = self.celery_results[task_id]
            celery_status = celery_result.status
            
            # Map Celery status to our TaskStatus
            status_mapping = {
                'PENDING': TaskStatus.PENDING,
                'STARTED': TaskStatus.RUNNING,
                'SUCCESS': TaskStatus.COMPLETED,
                'FAILURE': TaskStatus.FAILED,
                'RETRY': TaskStatus.RETRYING,
                'REVOKED': TaskStatus.CANCELLED,
            }
            
            mapped_status = status_mapping.get(celery_status, TaskStatus.PENDING)
            
            # Update our task status if it changed
            if task.status != mapped_status:
                task.status = mapped_status
                
                if mapped_status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.now()
                    task.result = celery_result.result
                elif mapped_status == TaskStatus.FAILED:
                    task.completed_at = datetime.now()
                    task.error = str(celery_result.info)
                elif mapped_status == TaskStatus.RUNNING and not task.started_at:
                    task.started_at = datetime.now()
        
        return task.status
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        # Update status before returning
        self.get_task_status(task_id)
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # Revoke the Celery task
        if task_id in self.celery_results:
            celery_result = self.celery_results[task_id]
            celery_result.revoke(terminate=True)
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def _update_task_status(self, task_id: str, status: TaskStatus, 
                           result: Any = None, error: str = None):
        """Update task status (called from Celery task callbacks)."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            
            if status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now()
                task.result = result
            elif status == TaskStatus.FAILED:
                task.completed_at = datetime.now()
                task.error = error
            elif status == TaskStatus.RUNNING and not task.started_at:
                task.started_at = datetime.now()
    
    def get_task_stats(self) -> Dict[str, int]:
        """Get statistics about tasks."""
        stats = {status.value: 0 for status in TaskStatus}
        
        # Update all task statuses first
        for task_id in list(self.tasks.keys()):
            self.get_task_status(task_id)
        
        # Count tasks by status
        for task in self.tasks.values():
            stats[task.status.value] += 1
        
        stats['total'] = len(self.tasks)
        
        # Add Celery-specific stats
        try:
            inspect = self.app.control.inspect()
            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()
            
            if active_tasks:
                stats['celery_active'] = sum(len(tasks) for tasks in active_tasks.values())
            if scheduled_tasks:
                stats['celery_scheduled'] = sum(len(tasks) for tasks in scheduled_tasks.values())
                
        except Exception as e:
            logger.warning(f"Could not get Celery stats: {e}")
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the worker."""
        logger.info("Shutting down CeleryBackgroundWorker...")
        
        # Cancel all pending tasks
        for task_id in list(self.celery_results.keys()):
            self.cancel_task(task_id)
        
        logger.info("CeleryBackgroundWorker shutdown complete")


def setup_periodic_tasks():
    """Setup periodic tasks using Celery Beat."""
    from celery.schedules import crontab
    
    # Example periodic task configurations
    celery_app.conf.beat_schedule = {
        'cleanup-completed-tasks': {
            'task': 'omni_dev.tasks.cleanup_completed_tasks',
            'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        },
        'health-check': {
            'task': 'omni_dev.tasks.health_check',
            'schedule': 30.0,  # Every 30 seconds
        },
    }


@celery_app.task
def cleanup_completed_tasks():
    """Periodic task to cleanup old completed tasks."""
    logger.info("Running periodic cleanup of completed tasks")
    # Implementation would clean up old task records
    return "Cleanup completed"


@celery_app.task
def health_check():
    """Periodic health check task."""
    logger.info("Health check - worker is alive")
    return "OK"


# Initialize periodic tasks
setup_periodic_tasks()


def create_celery_worker(**kwargs) -> CeleryBackgroundWorker:
    """Factory function to create a Celery background worker."""
    return CeleryBackgroundWorker(**kwargs)