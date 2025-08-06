import time
import threading
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from .background_worker import BackgroundTask, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for a specific task or task type."""
    
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_rate: float = 0.0
    
    def update_execution(self, task: BackgroundTask):
        """Update metrics based on a completed task."""
        self.total_executions += 1
        self.last_execution = task.completed_at or datetime.now()
        
        if task.status == TaskStatus.COMPLETED:
            self.successful_executions += 1
            self.last_success = self.last_execution
        elif task.status == TaskStatus.FAILED:
            self.failed_executions += 1
            self.last_failure = self.last_execution
        elif task.status == TaskStatus.CANCELLED:
            self.cancelled_executions += 1
        
        # Calculate execution time
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.total_execution_time += execution_time
            self.average_execution_time = self.total_execution_time / self.total_executions
            self.min_execution_time = min(self.min_execution_time, execution_time)
            self.max_execution_time = max(self.max_execution_time, execution_time)
        
        # Calculate error rate
        self.error_rate = self.failed_executions / self.total_executions if self.total_executions > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System-wide metrics for background task processing."""
    
    total_tasks_submitted: int = 0
    tasks_currently_running: int = 0
    tasks_in_queue: int = 0
    worker_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert system metrics to dictionary."""
        return asdict(self)


class TaskMonitor:
    """Monitor and track background task performance and metrics."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.task_metrics: Dict[str, TaskMetrics] = defaultdict(TaskMetrics)
        self.system_metrics = SystemMetrics()
        self.task_history: deque = deque(maxlen=max_history_size)
        self.performance_history: deque = deque(maxlen=max_history_size)
        self.start_time = datetime.now()
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.alert_callbacks: List[Callable] = []
        self.alert_thresholds = {
            'error_rate': 0.1,  # 10% error rate threshold
            'avg_execution_time': 300,  # 5 minutes threshold
            'queue_size': 100,  # Queue size threshold
        }
        
        # Performance tracking
        self.performance_samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("TaskMonitor initialized")
    
    def record_task_submission(self, task: BackgroundTask):
        """Record when a task is submitted."""
        if not self.monitoring_enabled:
            return
        
        self.system_metrics.total_tasks_submitted += 1
        
        # Record in history
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'submitted',
            'task_id': task.id,
            'task_name': task.name,
            'priority': task.priority.value,
            'tags': task.tags
        })
        
        logger.debug(f"Recorded task submission: {task.id}")
    
    def record_task_start(self, task: BackgroundTask):
        """Record when a task starts execution."""
        if not self.monitoring_enabled:
            return
        
        self.system_metrics.tasks_currently_running += 1
        
        # Record in history
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'started',
            'task_id': task.id,
            'task_name': task.name
        })
        
        logger.debug(f"Recorded task start: {task.id}")
    
    def record_task_completion(self, task: BackgroundTask):
        """Record when a task completes (success, failure, or cancellation)."""
        if not self.monitoring_enabled:
            return
        
        self.system_metrics.tasks_currently_running = max(0, self.system_metrics.tasks_currently_running - 1)
        
        # Update task-specific metrics
        task_name = task.name or 'unnamed'
        metrics = self.task_metrics[task_name]
        metrics.update_execution(task)
        
        # Record in history
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'completed',
            'task_id': task.id,
            'task_name': task.name,
            'status': task.status.value,
            'execution_time': (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else None,
            'error': task.error
        })
        
        # Record performance sample
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.performance_samples[task_name].append({
                'timestamp': datetime.now(),
                'execution_time': execution_time,
                'status': task.status.value
            })
        
        # Check for alerts
        self._check_alerts(task_name, metrics)
        
        logger.debug(f"Recorded task completion: {task.id} ({task.status.value})")
    
    def update_system_metrics(self, stats: Dict[str, Any]):
        """Update system-wide metrics."""
        if not self.monitoring_enabled:
            return
        
        # Update from worker stats
        self.system_metrics.tasks_in_queue = stats.get('queue_size', 0)
        self.system_metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Record performance history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.system_metrics.to_dict(),
            'task_counts': {status: stats.get(status, 0) for status in ['pending', 'running', 'completed', 'failed', 'cancelled']}
        })
    
    def get_task_metrics(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a specific task or all tasks."""
        if task_name:
            return self.task_metrics.get(task_name, TaskMetrics()).to_dict()
        
        return {name: metrics.to_dict() for name, metrics in self.task_metrics.items()}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self.system_metrics.to_dict()
    
    def get_performance_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get a performance summary for a given time window."""
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        
        # Filter recent task history
        recent_history = [
            entry for entry in self.task_history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
        ]
        
        # Calculate summary statistics
        summary = {
            'time_window': str(time_window),
            'total_tasks': len([e for e in recent_history if e['event'] == 'completed']),
            'successful_tasks': len([e for e in recent_history if e['event'] == 'completed' and e.get('status') == 'completed']),
            'failed_tasks': len([e for e in recent_history if e['event'] == 'completed' and e.get('status') == 'failed']),
            'average_execution_time': 0.0,
            'task_breakdown': defaultdict(int)
        }
        
        execution_times = []
        for entry in recent_history:
            if entry['event'] == 'completed' and entry.get('execution_time'):
                execution_times.append(entry['execution_time'])
                summary['task_breakdown'][entry.get('task_name', 'unnamed')] += 1
        
        if execution_times:
            summary['average_execution_time'] = sum(execution_times) / len(execution_times)
            summary['min_execution_time'] = min(execution_times)
            summary['max_execution_time'] = max(execution_times)
        
        return summary
    
    def get_task_trends(self, task_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for a specific task."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        samples = self.performance_samples.get(task_name, deque())
        recent_samples = [
            sample for sample in samples
            if sample['timestamp'] >= cutoff_time
        ]
        
        if not recent_samples:
            return {'task_name': task_name, 'no_data': True}
        
        # Calculate trends
        execution_times = [s['execution_time'] for s in recent_samples]
        success_count = len([s for s in recent_samples if s['status'] == 'completed'])
        
        return {
            'task_name': task_name,
            'sample_count': len(recent_samples),
            'success_rate': success_count / len(recent_samples) if recent_samples else 0,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'execution_time_trend': execution_times[-10:],  # Last 10 samples
            'time_window_hours': hours
        }
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, task_name: str, metrics: TaskMetrics):
        """Check if any alert conditions are met."""
        alerts = []
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'task_name': task_name,
                'current_value': metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate'],
                'message': f"High error rate for task '{task_name}': {metrics.error_rate:.2%}"
            })
        
        # Execution time alert
        if metrics.average_execution_time > self.alert_thresholds['avg_execution_time']:
            alerts.append({
                'type': 'slow_execution',
                'task_name': task_name,
                'current_value': metrics.average_execution_time,
                'threshold': self.alert_thresholds['avg_execution_time'],
                'message': f"Slow execution for task '{task_name}': {metrics.average_execution_time:.2f}s"
            })
        
        # Queue size alert
        if self.system_metrics.tasks_in_queue > self.alert_thresholds['queue_size']:
            alerts.append({
                'type': 'large_queue',
                'current_value': self.system_metrics.tasks_in_queue,
                'threshold': self.alert_thresholds['queue_size'],
                'message': f"Large task queue: {self.system_metrics.tasks_in_queue} tasks"
            })
        
        # Send alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert['message']}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert['type'], alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export all metrics in the specified format."""
        data = {
            'system_metrics': self.get_system_metrics(),
            'task_metrics': self.get_task_metrics(),
            'performance_summary': self.get_performance_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_metrics(self):
        """Clear all collected metrics and history."""
        self.task_metrics.clear()
        self.task_history.clear()
        self.performance_history.clear()
        self.performance_samples.clear()
        self.system_metrics = SystemMetrics()
        self.start_time = datetime.now()
        logger.info("All metrics cleared")
    
    def set_monitoring_enabled(self, enabled: bool):
        """Enable or disable monitoring."""
        self.monitoring_enabled = enabled
        logger.info(f"Task monitoring {'enabled' if enabled else 'disabled'}")


# Default alert handlers
def log_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
    """Default alert handler that logs alerts."""
    logger.warning(f"TASK ALERT [{alert_type}]: {alert_data.get('message', 'Unknown alert')}")


def email_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
    """Alert handler that sends email notifications (placeholder)."""
    # This would integrate with an email service
    logger.info(f"Would send email alert: {alert_data.get('message')}")


# Global monitor instance
_task_monitor = None


def get_task_monitor() -> TaskMonitor:
    """Get the global task monitor instance."""
    global _task_monitor
    if _task_monitor is None:
        _task_monitor = TaskMonitor()
        # Add default alert handler
        _task_monitor.add_alert_callback(log_alert_handler)
    return _task_monitor