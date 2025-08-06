# Background Processing Guide for Omni-Dev Agent

## Overview

The Omni-Dev Agent now includes comprehensive background processing capabilities that allow for:

- **Asynchronous task execution** with both threaded and distributed (Celery) workers
- **Task scheduling** with one-time and recurring tasks
- **Task monitoring** with metrics, alerts, and performance tracking
- **Priority-based processing** with automatic retry mechanisms
- **Seamless integration** with the orchestrator for long-running operations

## Architecture

### Core Components

1. **BackgroundTaskManager** - Main coordinator for task processing
2. **Workers** - Threaded and Celery-based task executors
3. **TaskScheduler** - Handles scheduled and recurring tasks
4. **TaskMonitor** - Tracks performance metrics and alerts
5. **Configuration System** - Flexible configuration management

### Worker Types

#### Threaded Worker
- **Use case**: Single-machine processing, development, lightweight tasks
- **Benefits**: Simple setup, no external dependencies
- **Configuration**: Max workers, queue size

#### Celery Worker  
- **Use case**: Distributed processing, production environments, scalable workloads
- **Benefits**: Horizontal scaling, fault tolerance, advanced scheduling
- **Requirements**: Redis/RabbitMQ broker

## Quick Start

### Basic Usage

```python
from src.core.background_worker import BackgroundTaskManager, TaskPriority

# Initialize task manager
task_manager = BackgroundTaskManager(worker_type="threaded")

# Submit a task
def my_task(x, y):
    return x + y

task_id = task_manager.submit_task(
    my_task, 10, 20,
    name="calculation",
    priority=TaskPriority.NORMAL
)

# Check status
status = task_manager.get_task_status(task_id)
print(f"Task status: {status}")

# Get result
task = task_manager.get_task(task_id)
print(f"Result: {task.result}")
```

### Scheduling Tasks

```python
from datetime import datetime, timedelta

# Schedule one-time task
future_time = datetime.now() + timedelta(minutes=5)
task_id = task_manager.schedule_task_once(my_task, future_time, 5, 10)

# Schedule recurring task
task_id = task_manager.schedule_task_interval(
    my_task, 
    timedelta(hours=1),  # Run every hour
    15, 25
)
```

### Using with Orchestrator

```python
from src.core.orchestration import Orchestrator

# Create orchestrator with background processing
orchestrator = Orchestrator(use_background_processing=True)

# Execute request in background
task_ids = orchestrator.execute(
    "Develop a web API with authentication",
    use_background=True
)

# Monitor progress
for task_id in task_ids:
    status = orchestrator.get_background_task_status(task_id)
    print(f"Phase status: {status}")
```

## Configuration

### Configuration File (YAML)

```yaml
# omni_dev_bg_config.yaml
worker_type: "threaded"  # or "celery"

threaded_worker:
  max_workers: 4
  queue_size: 100

celery_worker:
  broker_url: "redis://localhost:6379/0"
  result_backend: "redis://localhost:6379/0"
  default_queue: "omni_dev"
  priority_queue: "priority"

monitoring:
  enabled: true
  max_history_size: 1000
  alert_enabled: true
  error_rate_threshold: 0.1
  avg_execution_time_threshold: 300

scheduler:
  enabled: true
  check_interval_seconds: 1.0

log_level: "INFO"
shutdown_timeout_seconds: 30
```

### Environment Variables

```bash
# Worker configuration
export BG_WORKER_TYPE="threaded"
export BG_THREADED_MAX_WORKERS=4
export BG_THREADED_QUEUE_SIZE=100

# Celery configuration
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Monitoring
export BG_MONITORING_ENABLED=true
export BG_MONITORING_ERROR_THRESHOLD=0.1

# Use environment config
export OMNI_DEV_BG_USE_ENV=true
```

### Loading Configuration

```python
from src.config.background_config import auto_load_config, load_config

# Auto-load from default locations
config = auto_load_config()

# Load from specific file
config = load_config("/path/to/config.yaml")

# Load from environment
config = load_config("env")

# Use with task manager
task_manager = BackgroundTaskManager(
    worker_type=config.worker_type,
    **config.threaded_worker.__dict__
)
```

## Task Monitoring

### Metrics Available

- **Task-specific metrics**: Execution count, success rate, average time, error rate
- **System metrics**: Queue size, active tasks, worker utilization
- **Performance trends**: Historical execution times, success rates over time

### Using the Monitor

```python
from src.core.task_monitor import get_task_monitor

monitor = get_task_monitor()

# Get task metrics
metrics = monitor.get_task_metrics("my_task")
print(f"Success rate: {metrics['successful_executions'] / metrics['total_executions']:.2%}")

# Get system metrics
system_metrics = monitor.get_system_metrics()
print(f"Tasks in queue: {system_metrics['tasks_in_queue']}")

# Performance summary
summary = monitor.get_performance_summary(timedelta(hours=1))
print(f"Tasks last hour: {summary['total_tasks']}")

# Export metrics
json_metrics = monitor.export_metrics(format='json')
with open('metrics.json', 'w') as f:
    f.write(json_metrics)
```

### Alerts

```python
def custom_alert_handler(alert_type, alert_data):
    print(f"ALERT: {alert_data['message']}")
    # Send to monitoring system, email, etc.

monitor.add_alert_callback(custom_alert_handler)
```

## Celery Setup

### Install Dependencies

```bash
pip install celery redis
```

### Start Redis (Docker)

```bash
docker run -d -p 6379:6379 redis:alpine
```

### Start Celery Worker

```bash
# In project directory
celery -A src.core.celery_worker.celery_app worker --loglevel=info
```

### Start Celery Beat (for scheduled tasks)

```bash
celery -A src.core.celery_worker.celery_app beat --loglevel=info
```

### Monitor Celery

```bash
celery -A src.core.celery_worker.celery_app flower
# Access http://localhost:5555
```

## Error Handling

### Automatic Retries

```python
task_id = task_manager.submit_task(
    unreliable_function,
    name="might_fail",
    max_retries=3,  # Retry up to 3 times
    timeout=30      # 30 second timeout
)
```

### Error Monitoring

```python
# Tasks that fail frequently will trigger alerts
# Monitor error rates and patterns
task_trends = monitor.get_task_trends("problematic_task", hours=24)
print(f"Error rate last 24h: {1 - task_trends['success_rate']:.2%}")
```

## Production Deployment

### Recommended Architecture

```
Load Balancer
    ↓
Web Servers (Omni-Dev Agent instances)
    ↓
Redis Cluster (Celery broker/backend)
    ↓
Celery Workers (Multiple machines)
    ↓
Monitoring Dashboard
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  omni-dev:
    build: .
    environment:
      - BG_WORKER_TYPE=celery
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  celery-worker:
    build: .
    command: celery -A src.core.celery_worker.celery_app worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  celery-beat:
    build: .
    command: celery -A src.core.celery_worker.celery_app beat
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
```

## Performance Tuning

### Threaded Worker

```python
# Adjust based on CPU cores and task I/O characteristics
task_manager = BackgroundTaskManager(
    worker_type="threaded",
    max_workers=8,      # 2x CPU cores for I/O bound tasks
    queue_size=1000     # Larger queue for burst handling
)
```

### Celery Worker

```bash
# CPU-bound tasks
celery -A src.core.celery_worker.celery_app worker --concurrency=4

# I/O-bound tasks
celery -A src.core.celery_worker.celery_app worker --concurrency=20 --pool=eventlet
```

### Memory Management

```python
# Limit task history to prevent memory growth
config = BackgroundTaskConfig(
    monitoring=MonitoringConfig(
        max_history_size=5000,
        export_enabled=True,
        export_interval_seconds=1800  # Export and clear every 30 mins
    )
)
```

## Troubleshooting

### Common Issues

1. **Tasks not executing**
   - Check worker status: `task_manager.get_stats()`
   - Verify configuration: `auto_load_config()`
   - Check logs for errors

2. **High memory usage**
   - Reduce `max_history_size`
   - Enable periodic metrics export
   - Clear completed tasks regularly

3. **Celery connection errors**
   - Verify Redis is running
   - Check broker URL configuration
   - Test network connectivity

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('src.core').setLevel(logging.DEBUG)

# Disable monitoring for performance testing
task_manager = BackgroundTaskManager(monitor_enabled=False)
```

## Integration Examples

### With Flask

```python
from flask import Flask, request, jsonify
from src.core.background_worker import get_task_manager

app = Flask(__name__)

@app.route('/submit_task', methods=['POST'])
def submit_task():
    data = request.json
    task_manager = get_task_manager()
    
    task_id = task_manager.submit_task(
        process_data, data,
        name="api_task",
        priority=TaskPriority.HIGH
    )
    
    return jsonify({"task_id": task_id})

@app.route('/task_status/<task_id>')
def task_status(task_id):
    task_manager = get_task_manager()
    status = task_manager.get_task_status(task_id)
    task = task_manager.get_task(task_id)
    
    return jsonify({
        "status": status.value,
        "result": task.result if task else None
    })
```

### With Django

```python
# Django view
from django.http import JsonResponse
from src.core.background_worker import submit_background_task

def submit_task_view(request):
    task_id = submit_background_task(
        long_running_function,
        request.POST.get('data'),
        name="django_task"
    )
    
    return JsonResponse({"task_id": task_id})
```

## API Reference

### BackgroundTaskManager

```python
class BackgroundTaskManager:
    def submit_task(self, func, *args, **kwargs) -> str
    def schedule_task_once(self, func, run_at, *args, **kwargs) -> str
    def schedule_task_interval(self, func, interval, *args, **kwargs) -> str
    def get_task_status(self, task_id) -> TaskStatus
    def get_task(self, task_id) -> BackgroundTask
    def cancel_task(self, task_id) -> bool
    def get_stats(self) -> Dict[str, Any]
    def shutdown(self)
```

### TaskMonitor

```python
class TaskMonitor:
    def get_task_metrics(self, task_name=None) -> Dict[str, Any]
    def get_system_metrics(self) -> Dict[str, Any]
    def get_performance_summary(self, time_window) -> Dict[str, Any]
    def get_task_trends(self, task_name, hours) -> Dict[str, Any]
    def export_metrics(self, format='json') -> str
    def add_alert_callback(self, callback)
```

### Configuration Classes

```python
@dataclass
class BackgroundTaskConfig:
    worker_type: str = "threaded"
    threaded_worker: ThreadedWorkerConfig
    celery_worker: CeleryWorkerConfig
    monitoring: MonitoringConfig
    scheduler: SchedulerConfig
    log_level: str = "INFO"
```

## Demo Script

Run the comprehensive demo:

```bash
python demo_background_processing.py
```

This demonstrates all features including:
- Basic threaded worker usage
- Task priorities and retries
- Scheduling capabilities
- Monitoring and metrics
- Orchestrator integration
- Configuration management

## Next Steps

1. **Production Deployment**: Set up Redis cluster and multiple Celery workers
2. **Monitoring Dashboard**: Integrate with Grafana/Prometheus for visualization
3. **Custom Queues**: Configure specialized queues for different task types
4. **Load Testing**: Test performance under expected production loads
5. **Security**: Implement authentication for distributed setups

## Support

For questions or issues with background processing:

1. Check the logs: `logging.getLogger('src.core').setLevel(logging.DEBUG)`
2. Review configuration: `auto_load_config().to_dict()`
3. Monitor metrics: `get_task_monitor().get_system_metrics()`
4. Test connectivity: Redis for Celery, threading for local workers