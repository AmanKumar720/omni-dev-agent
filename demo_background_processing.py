#!/usr/bin/env python3
"""
Demo script for Omni-Dev Agent Background Processing Capabilities

This script demonstrates:
1. Threaded background workers
2. Celery distributed workers (if available)
3. Task scheduling and recurring tasks
4. Task monitoring and metrics
5. Integration with the orchestrator
6. Configuration management
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our background processing components
from src.core.background_worker import (
    BackgroundTaskManager, TaskPriority, TaskStatus,
    submit_background_task, schedule_task_once, schedule_task_interval
)
from src.core.task_monitor import get_task_monitor
from src.core.orchestration import Orchestrator
from src.config.background_config import (
    auto_load_config, BackgroundTaskConfig, 
    ThreadedWorkerConfig, MonitoringConfig
)


# Demo task functions
def simple_calculation(x: int, y: int) -> int:
    """A simple calculation task."""
    logger.info(f"Calculating {x} + {y}")
    result = x + y
    logger.info(f"Result: {result}")
    return result


def long_running_task(duration: int = 5) -> str:
    """A long-running task that simulates work."""
    logger.info(f"Starting long-running task (duration: {duration}s)")
    
    for i in range(duration):
        time.sleep(1)
        logger.info(f"Long-running task progress: {i+1}/{duration}")
    
    result = f"Long-running task completed after {duration} seconds"
    logger.info(result)
    return result


def error_prone_task(should_fail: bool = False) -> str:
    """A task that may fail for testing error handling."""
    if should_fail:
        raise ValueError("This task was designed to fail")
    
    return "Task completed successfully"


def data_processing_task(data_size: int = 1000) -> Dict[str, Any]:
    """Simulate processing a dataset."""
    logger.info(f"Processing dataset of size {data_size}")
    
    # Simulate some processing time
    time.sleep(2)
    
    result = {
        'processed_items': data_size,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }
    
    logger.info(f"Processed {data_size} items")
    return result


def scheduled_maintenance() -> str:
    """A maintenance task that runs on schedule."""
    logger.info("Running scheduled maintenance...")
    time.sleep(1)
    return "Maintenance completed"


def demo_basic_threaded_worker():
    """Demo 1: Basic threaded background worker."""
    print("\n🚀 DEMO 1: Basic Threaded Background Worker")
    print("=" * 60)
    
    # Create a task manager with threaded worker
    task_manager = BackgroundTaskManager(
        worker_type="threaded",
        max_workers=2,
        queue_size=10
    )
    
    # Submit some tasks
    print("📝 Submitting tasks...")
    task_ids = []
    
    # Simple calculation tasks
    for i in range(3):
        task_id = task_manager.submit_task(
            simple_calculation, 
            i * 10, 
            i * 5,
            name=f"calculation_{i}",
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
        print(f"   Submitted calculation task {i}: {task_id}")
    
    # Long-running task
    long_task_id = task_manager.submit_task(
        long_running_task,
        3,  # duration
        name="long_task",
        priority=TaskPriority.HIGH
    )
    task_ids.append(long_task_id)
    print(f"   Submitted long-running task: {long_task_id}")
    
    # Monitor task progress
    print("\n⏳ Monitoring task progress...")
    completed_tasks = set()
    
    while len(completed_tasks) < len(task_ids):
        for task_id in task_ids:
            if task_id not in completed_tasks:
                status = task_manager.get_task_status(task_id)
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task = task_manager.get_task(task_id)
                    print(f"   ✅ Task {task_id} ({task.name}): {status.value}")
                    if task.result:
                        print(f"      Result: {task.result}")
                    completed_tasks.add(task_id)
        
        time.sleep(0.5)
    
    # Show final statistics
    stats = task_manager.get_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    task_manager.shutdown()
    print("✅ Demo 1 completed")


def demo_task_priorities_and_retries():
    """Demo 2: Task priorities and retry mechanisms."""
    print("\n🎯 DEMO 2: Task Priorities and Retry Mechanisms")
    print("=" * 60)
    
    task_manager = BackgroundTaskManager(worker_type="threaded", max_workers=1)
    
    # Submit tasks with different priorities
    print("📝 Submitting tasks with different priorities...")
    
    # Low priority task
    low_task = task_manager.submit_task(
        long_running_task, 2,
        name="low_priority_task",
        priority=TaskPriority.LOW
    )
    
    # High priority task (should run first)
    high_task = task_manager.submit_task(
        simple_calculation, 100, 200,
        name="high_priority_task", 
        priority=TaskPriority.HIGH
    )
    
    # Task that will fail and retry
    retry_task = task_manager.submit_task(
        error_prone_task, True,  # should_fail=True
        name="retry_task",
        priority=TaskPriority.NORMAL,
        max_retries=2
    )
    
    print(f"   Low priority task: {low_task}")
    print(f"   High priority task: {high_task}")
    print(f"   Retry task: {retry_task}")
    
    # Monitor execution order and retries
    print("\n⏳ Monitoring execution order...")
    time.sleep(8)  # Let tasks complete
    
    # Check results
    high_task_obj = task_manager.get_task(high_task)
    retry_task_obj = task_manager.get_task(retry_task)
    
    print(f"\n📊 Results:")
    print(f"   High priority task status: {high_task_obj.status.value}")
    print(f"   Retry task status: {retry_task_obj.status.value}")
    print(f"   Retry task retry count: {retry_task_obj.retry_count}")
    
    task_manager.shutdown()
    print("✅ Demo 2 completed")


def demo_task_scheduling():
    """Demo 3: Task scheduling capabilities."""
    print("\n⏰ DEMO 3: Task Scheduling")
    print("=" * 60)
    
    task_manager = BackgroundTaskManager(worker_type="threaded")
    
    # Schedule a one-time task
    future_time = datetime.now() + timedelta(seconds=3)
    scheduled_task = task_manager.schedule_task_once(
        simple_calculation, future_time, 50, 75,
        name="scheduled_calculation"
    )
    
    print(f"📅 Scheduled one-time task for {future_time}: {scheduled_task}")
    
    # Schedule a recurring task
    recurring_task = task_manager.schedule_task_interval(
        scheduled_maintenance,
        timedelta(seconds=2),
        name="maintenance_task"
    )
    
    print(f"🔄 Scheduled recurring task (every 2s): {recurring_task}")
    
    # Let tasks run for a bit
    print("\n⏳ Letting scheduled tasks run...")
    time.sleep(10)
    
    # Show scheduler statistics
    stats = task_manager.get_stats()
    print(f"\n📊 Scheduler Statistics:")
    print(f"   Scheduled tasks: {stats.get('scheduled_tasks', 0)}")
    
    task_manager.shutdown()
    print("✅ Demo 3 completed")


def demo_task_monitoring():
    """Demo 4: Task monitoring and metrics."""
    print("\n📈 DEMO 4: Task Monitoring and Metrics")
    print("=" * 60)
    
    # Create task manager with monitoring enabled
    task_manager = BackgroundTaskManager(worker_type="threaded", monitor_enabled=True)
    monitor = get_task_monitor()
    
    # Submit various tasks to generate metrics
    print("📝 Submitting tasks to generate metrics...")
    
    task_ids = []
    for i in range(5):
        # Mix of successful and failed tasks
        should_fail = i % 3 == 0  # Every 3rd task fails
        
        task_id = task_manager.submit_task(
            error_prone_task, should_fail,
            name=f"monitored_task_{i}",
            priority=TaskPriority.NORMAL,
            max_retries=0  # No retries for cleaner metrics
        )
        task_ids.append(task_id)
    
    # Add some data processing tasks
    for i in range(3):
        task_id = task_manager.submit_task(
            data_processing_task, (i + 1) * 500,
            name="data_processing",
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
    
    # Wait for tasks to complete
    print("⏳ Waiting for tasks to complete...")
    time.sleep(5)
    
    # Display metrics
    print("\n📊 Task Metrics:")
    task_metrics = monitor.get_task_metrics()
    
    for task_name, metrics in task_metrics.items():
        print(f"\n   Task: {task_name}")
        print(f"      Total executions: {metrics['total_executions']}")
        print(f"      Success rate: {metrics['successful_executions'] / max(1, metrics['total_executions']):.2%}")
        print(f"      Average execution time: {metrics['average_execution_time']:.2f}s")
        print(f"      Error rate: {metrics['error_rate']:.2%}")
    
    # System metrics
    print(f"\n📊 System Metrics:")
    system_metrics = monitor.get_system_metrics()
    for key, value in system_metrics.items():
        print(f"   {key}: {value}")
    
    # Performance summary
    print(f"\n📊 Performance Summary (last hour):")
    summary = monitor.get_performance_summary(timedelta(hours=1))
    print(f"   Total tasks: {summary['total_tasks']}")
    print(f"   Success rate: {summary['successful_tasks'] / max(1, summary['total_tasks']):.2%}")
    print(f"   Average execution time: {summary['average_execution_time']:.2f}s")
    
    task_manager.shutdown()
    print("✅ Demo 4 completed")


def demo_orchestrator_integration():
    """Demo 5: Integration with the orchestrator."""
    print("\n🎭 DEMO 5: Orchestrator Integration")
    print("=" * 60)
    
    # Create orchestrator with background processing enabled
    orchestrator = Orchestrator(use_background_processing=True)
    
    print("📝 Executing request with background processing...")
    
    # Execute a request in the background
    request = "Develop a simple web API with authentication and database integration"
    task_ids = orchestrator.execute(request, use_background=True)
    
    if task_ids:
        print(f"   Submitted {len(task_ids)} phases as background tasks")
        
        # Monitor progress
        print("\n⏳ Monitoring orchestrator task progress...")
        for i, task_id in enumerate(task_ids):
            status = orchestrator.get_background_task_status(task_id)
            print(f"   Phase {i+1} ({task_id}): {status.value if status else 'Unknown'}")
        
        # Let some tasks run
        time.sleep(3)
        
        # Check final status
        print(f"\n📊 Final status:")
        for i, task_id in enumerate(task_ids):
            status = orchestrator.get_background_task_status(task_id)
            print(f"   Phase {i+1}: {status.value if status else 'Unknown'}")
        
        # Show background processing stats
        bg_stats = orchestrator.get_background_stats()
        if bg_stats:
            print(f"\n📊 Background Processing Stats:")
            for key, value in bg_stats.items():
                print(f"   {key}: {value}")
    
    orchestrator.shutdown()
    print("✅ Demo 5 completed")


def demo_configuration_management():
    """Demo 6: Configuration management."""
    print("\n⚙️ DEMO 6: Configuration Management")
    print("=" * 60)
    
    # Load default configuration
    print("📄 Loading default configuration...")
    config = auto_load_config()
    
    print(f"   Worker type: {config.worker_type}")
    print(f"   Monitoring enabled: {config.monitoring.enabled}")
    print(f"   Max workers: {config.threaded_worker.max_workers}")
    print(f"   Log level: {config.log_level}")
    
    # Create custom configuration
    print("\n📝 Creating custom configuration...")
    custom_config = BackgroundTaskConfig(
        worker_type="threaded",
        threaded_worker=ThreadedWorkerConfig(max_workers=8, queue_size=200),
        monitoring=MonitoringConfig(enabled=True, max_history_size=2000),
        log_level="DEBUG"
    )
    
    # Save configuration to file
    config_file = "/tmp/omni_dev_custom_config.yaml"
    custom_config.save_to_file(config_file)
    print(f"   Saved custom configuration to: {config_file}")
    
    # Load configuration from file
    loaded_config = BackgroundTaskConfig.from_file(config_file)
    print(f"   Loaded configuration from file")
    print(f"   Max workers: {loaded_config.threaded_worker.max_workers}")
    print(f"   Queue size: {loaded_config.threaded_worker.queue_size}")
    
    print("✅ Demo 6 completed")


def main():
    """Run all demos."""
    print("🤖 Omni-Dev Agent Background Processing Capabilities Demo")
    print("=" * 70)
    print("This demo showcases the comprehensive background processing system")
    print("including threaded workers, scheduling, monitoring, and orchestration.")
    print("=" * 70)
    
    try:
        # Run demos
        demo_basic_threaded_worker()
        demo_task_priorities_and_retries()
        demo_task_scheduling()
        demo_task_monitoring()
        demo_orchestrator_integration()
        demo_configuration_management()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Multi-threaded background task execution")
        print("✅ Task prioritization and retry mechanisms")
        print("✅ Scheduled and recurring tasks")
        print("✅ Comprehensive task monitoring and metrics")
        print("✅ Integration with the orchestrator")
        print("✅ Flexible configuration management")
        print("✅ Real-time task status tracking")
        print("✅ Error handling and recovery")
        
        print("\nNext Steps:")
        print("🔧 Set up Redis and enable Celery for distributed processing")
        print("📊 Integrate with monitoring dashboards")
        print("🔔 Configure alert notifications")
        print("📝 Customize task queues and priorities for your use case")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()