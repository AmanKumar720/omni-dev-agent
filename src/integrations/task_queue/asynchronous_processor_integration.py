# This file provides integration for Task Queue/Asynchronous Processor
# Enhanced with comprehensive background processing capabilities

import logging
from typing import Dict, Any, Optional

from ...core.background_worker import BackgroundTaskManager, get_task_manager
from ...core.celery_worker import create_celery_worker
from ...config.background_config import auto_load_config, get_config_manager

logger = logging.getLogger(__name__)


def initialize_task_queue_asynchronous_processor_integration(
    worker_type: str = "threaded",
    custom_config: Optional[Dict[str, Any]] = None
) -> BackgroundTaskManager:
    """
    Initialize the Task Queue/Asynchronous Processor integration with enhanced capabilities.
    
    Args:
        worker_type: Type of worker to use ("threaded" or "celery")
        custom_config: Optional custom configuration
    
    Returns:
        BackgroundTaskManager: Initialized task manager
    """
    print("Initializing enhanced Task Queue/Asynchronous Processor integration...")
    
    try:
        # Load configuration
        if custom_config:
            config_manager = get_config_manager()
            config_manager.load_from_dict(custom_config)
            config = config_manager.config
        else:
            config = auto_load_config()
        
        # Override worker type if specified
        if worker_type != config.worker_type:
            config.worker_type = worker_type
        
        # Create appropriate worker configuration
        if config.worker_type == "threaded":
            worker_config = {
                'max_workers': config.threaded_worker.max_workers,
                'queue_size': config.threaded_worker.queue_size
            }
        elif config.worker_type == "celery":
            # Celery worker uses its own configuration
            worker_config = {}
        else:
            raise ValueError(f"Unsupported worker type: {config.worker_type}")
        
        # Initialize task manager
        task_manager = BackgroundTaskManager(
            worker_type=config.worker_type,
            monitor_enabled=config.monitoring.enabled,
            **worker_config
        )
        
        logger.info(f"Task Queue integration initialized with {config.worker_type} worker")
        print(f"✅ Integration successful - {config.worker_type} worker ready")
        
        return task_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize task queue integration: {e}")
        print(f"❌ Integration failed: {e}")
        raise


def get_integration_status() -> Dict[str, Any]:
    """Get the current status of the task queue integration."""
    try:
        task_manager = get_task_manager()
        config = auto_load_config()
        
        status = {
            'worker_type': config.worker_type,
            'monitoring_enabled': config.monitoring.enabled,
            'scheduler_enabled': config.scheduler.enabled,
            'task_stats': task_manager.get_stats() if hasattr(task_manager, 'get_stats') else None,
            'status': 'active'
        }
        
        return status
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def shutdown_integration():
    """Shutdown the task queue integration."""
    try:
        task_manager = get_task_manager()
        if hasattr(task_manager, 'shutdown'):
            task_manager.shutdown()
        
        logger.info("Task queue integration shutdown complete")
        print("✅ Task queue integration shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        print(f"❌ Shutdown error: {e}")


# Legacy compatibility
def initialize_task_queue_asynchronous_processor_integration_legacy():
    """Legacy initialization method for backward compatibility."""
    return initialize_task_queue_asynchronous_processor_integration()
