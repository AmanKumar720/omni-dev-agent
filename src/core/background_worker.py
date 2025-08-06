import time
import threading
import queue
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BackgroundTask:
    """Represents a background task with metadata and execution context."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert task to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'tags': self.tags,
            'metadata': self.metadata
        }


class BackgroundWorker(ABC):
    """Abstract base class for background workers."""
    
    @abstractmethod
    def submit_task(self, task: BackgroundTask) -> str:
        """Submit a task for background execution."""
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        pass
    
    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the worker."""
        pass


class ThreadedBackgroundWorker(BackgroundWorker):
    """Thread-based background worker for local task execution."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.task_queue = queue.PriorityQueue(maxsize=queue_size)
        self.tasks: Dict[str, BackgroundTask] = {}
        self.futures: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.worker_threads = []
        
        # Start worker threads
        for i in range(max_workers):
            thread = threading.Thread(target=self._worker_loop, name=f"BackgroundWorker-{i}")
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"ThreadedBackgroundWorker started with {max_workers} workers")
    
    def submit_task(self, task: BackgroundTask) -> str:
        """Submit a task for background execution."""
        if not self.running:
            raise RuntimeError("Worker is shutting down")
        
        try:
            # Store task
            self.tasks[task.id] = task
            
            # Add to priority queue (lower priority value = higher priority)
            priority = -task.priority.value  # Negate for correct ordering
            self.task_queue.put((priority, time.time(), task.id), timeout=1.0)
            
            logger.info(f"Task {task.id} ({task.name}) submitted to queue")
            return task.id
            
        except queue.Full:
            raise RuntimeError("Task queue is full")
    
    def _worker_loop(self):
        """Main worker loop that processes tasks from the queue."""
        while self.running:
            try:
                # Get task from queue with timeout
                priority, timestamp, task_id = self.task_queue.get(timeout=1.0)
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
    
    def _execute_task(self, task: BackgroundTask):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            logger.info(f"Executing task {task.id} ({task.name})")
            
            # Execute the task function
            if task.timeout:
                future = self.executor.submit(task.func, *task.args, **task.kwargs)
                self.futures[task.id] = future
                task.result = future.result(timeout=task.timeout)
            else:
                task.result = task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            logger.error(f"Task {task.id} failed: {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Re-queue for retry
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1})")
                priority = -task.priority.value
                try:
                    self.task_queue.put((priority, time.time(), task.id), timeout=1.0)
                except queue.Full:
                    logger.error(f"Failed to requeue task {task.id} for retry")
        
        finally:
            # Clean up future if it exists
            if task.id in self.futures:
                del self.futures[task.id]
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # Cancel future if running
        if task_id in self.futures:
            future = self.futures[task_id]
            if future.cancel():
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"Task {task_id} cancelled")
                return True
        
        # If not yet running, mark as cancelled
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.task_queue.qsize()
    
    def get_task_stats(self) -> Dict[str, int]:
        """Get statistics about tasks."""
        stats = {status.value: 0 for status in TaskStatus}
        
        for task in self.tasks.values():
            stats[task.status.value] += 1
        
        stats['total'] = len(self.tasks)
        stats['queue_size'] = self.get_queue_size()
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the worker."""
        logger.info("Shutting down ThreadedBackgroundWorker...")
        self.running = False
        
        # Cancel all pending futures
        for future in self.futures.values():
            future.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ThreadedBackgroundWorker shutdown complete")


class ScheduledTask:
    """Represents a scheduled task that can be executed at specific times or intervals."""
    
    def __init__(self, task: BackgroundTask, schedule_type: str, 
                 schedule_value: Union[datetime, timedelta, str]):
        self.task = task
        self.schedule_type = schedule_type  # 'once', 'interval', 'cron'
        self.schedule_value = schedule_value
        self.next_run = self._calculate_next_run()
    
    def _calculate_next_run(self) -> datetime:
        """Calculate the next run time based on schedule."""
        if self.schedule_type == 'once':
            return self.schedule_value
        elif self.schedule_type == 'interval':
            return datetime.now() + self.schedule_value
        # TODO: Add cron support
        return datetime.now()
    
    def should_run(self) -> bool:
        """Check if the task should run now."""
        return datetime.now() >= self.next_run
    
    def update_next_run(self):
        """Update the next run time after execution."""
        if self.schedule_type == 'interval':
            self.next_run = datetime.now() + self.schedule_value


class TaskScheduler:
    """Scheduler for managing scheduled and recurring tasks."""
    
    def __init__(self, worker: BackgroundWorker):
        self.worker = worker
        self.scheduled_tasks: List[ScheduledTask] = []
        self.running = False
        self.scheduler_thread = None
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("TaskScheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        logger.info("TaskScheduler stopped")
    
    def schedule_once(self, task: BackgroundTask, run_at: datetime) -> str:
        """Schedule a task to run once at a specific time."""
        scheduled_task = ScheduledTask(task, 'once', run_at)
        self.scheduled_tasks.append(scheduled_task)
        logger.info(f"Task {task.id} scheduled to run once at {run_at}")
        return task.id
    
    def schedule_interval(self, task: BackgroundTask, interval: timedelta) -> str:
        """Schedule a task to run at regular intervals."""
        scheduled_task = ScheduledTask(task, 'interval', interval)
        self.scheduled_tasks.append(scheduled_task)
        logger.info(f"Task {task.id} scheduled to run every {interval}")
        return task.id
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                current_time = datetime.now()
                
                for scheduled_task in self.scheduled_tasks[:]:  # Copy list to avoid modification issues
                    if scheduled_task.should_run():
                        # Submit task to worker
                        self.worker.submit_task(scheduled_task.task)
                        
                        # Update next run or remove if one-time
                        if scheduled_task.schedule_type == 'once':
                            self.scheduled_tasks.remove(scheduled_task)
                        else:
                            scheduled_task.update_next_run()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)


class BackgroundTaskManager:
    """Main manager for coordinating background tasks and workers."""
    
    def __init__(self, worker_type: str = "threaded", monitor_enabled: bool = True, **worker_kwargs):
        # Initialize monitor if enabled
        if monitor_enabled:
            from .task_monitor import get_task_monitor
            self.monitor = get_task_monitor()
        else:
            self.monitor = None
        
        # Create worker based on type
        if worker_type == "threaded":
            self.worker = ThreadedBackgroundWorker(**worker_kwargs)
        elif worker_type == "celery":
            from .celery_worker import CeleryBackgroundWorker
            self.worker = CeleryBackgroundWorker(**worker_kwargs)
        else:
            raise ValueError(f"Unknown worker type: {worker_type}")
        
        self.scheduler = TaskScheduler(self.worker)
        self.scheduler.start()
        
        logger.info(f"BackgroundTaskManager initialized with {worker_type} worker")
    
    def submit_task(self, func: Callable, *args, name: str = "", 
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[int] = None, 
                   max_retries: int = 3,
                   tags: List[str] = None,
                   **kwargs) -> str:
        """Submit a task for immediate background execution."""
        task = BackgroundTask(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            tags=tags or []
        )
        
        # Record task submission in monitor
        if self.monitor:
            self.monitor.record_task_submission(task)
        
        return self.worker.submit_task(task)
    
    def schedule_task_once(self, func: Callable, run_at: datetime, *args, 
                          name: str = "", **kwargs) -> str:
        """Schedule a task to run once at a specific time."""
        task = BackgroundTask(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        return self.scheduler.schedule_once(task, run_at)
    
    def schedule_task_interval(self, func: Callable, interval: timedelta, *args,
                              name: str = "", **kwargs) -> str:
        """Schedule a task to run at regular intervals."""
        task = BackgroundTask(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        return self.scheduler.schedule_interval(task, interval)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        return self.worker.get_task_status(task_id)
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        if hasattr(self.worker, 'get_task'):
            return self.worker.get_task(task_id)
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return self.worker.cancel_task(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker and task statistics."""
        stats = {
            'worker_type': type(self.worker).__name__,
            'scheduled_tasks': len(self.scheduler.scheduled_tasks)
        }
        
        if hasattr(self.worker, 'get_task_stats'):
            stats.update(self.worker.get_task_stats())
        
        return stats
    
    def shutdown(self):
        """Shutdown the task manager."""
        logger.info("Shutting down BackgroundTaskManager...")
        self.scheduler.stop()
        self.worker.shutdown()
        logger.info("BackgroundTaskManager shutdown complete")


# Global task manager instance
_task_manager = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


def submit_background_task(func: Callable, *args, **kwargs) -> str:
    """Convenience function to submit a background task."""
    return get_task_manager().submit_task(func, *args, **kwargs)


def schedule_task_once(func: Callable, run_at: datetime, *args, **kwargs) -> str:
    """Convenience function to schedule a one-time task."""
    return get_task_manager().schedule_task_once(func, run_at, *args, **kwargs)


def schedule_task_interval(func: Callable, interval: timedelta, *args, **kwargs) -> str:
    """Convenience function to schedule a recurring task."""
    return get_task_manager().schedule_task_interval(func, interval, *args, **kwargs)