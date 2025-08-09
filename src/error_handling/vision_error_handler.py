"""
Vision-specific error handling and retry mechanisms
Provides comprehensive error handling for all vision operations with retry logic.
"""

import functools
import time
import logging
import traceback
from typing import Any, Optional, Callable, Dict, List, Type, Union
from dataclasses import dataclass
from .error_manager import ErrorManager, ErrorSeverity, global_error_manager
from .vision_exceptions import (
    VisionError, ModelLoadError, CameraTimeoutError, CameraConnectionError,
    CameraNotFoundError, ImageProcessingError, ModelInferenceError,
    ModelValidationError, InsufficientMemoryError, GPUError, ConfigurationError,
    DataFormatError, DependencyError, NetworkError, PermissionError, CalibrationError
)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_exceptions: tuple = (VisionError,)


class VisionErrorHandler:
    """Comprehensive error handler for vision operations."""
    
    def __init__(self, error_manager: Optional[ErrorManager] = None):
        self.error_manager = error_manager or global_error_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup recovery strategies for different error types."""
        self.error_manager.register_recovery_strategy("ModelLoadError", self._handle_model_load_error)
        self.error_manager.register_recovery_strategy("CameraTimeoutError", self._handle_camera_timeout_error)
        self.error_manager.register_recovery_strategy("CameraConnectionError", self._handle_camera_connection_error)
        self.error_manager.register_recovery_strategy("GPUError", self._handle_gpu_error)
        self.error_manager.register_recovery_strategy("InsufficientMemoryError", self._handle_memory_error)
        self.error_manager.register_recovery_strategy("NetworkError", self._handle_network_error)
    
    def _handle_model_load_error(self, error_context):
        """Recovery strategy for model loading errors."""
        self.logger.info(f"Attempting recovery for model load error: {error_context.message}")
        # Could implement fallback to CPU, different model variant, etc.
        
    def _handle_camera_timeout_error(self, error_context):
        """Recovery strategy for camera timeout errors."""
        self.logger.info(f"Attempting recovery for camera timeout: {error_context.message}")
        # Could implement camera reconnection, timeout adjustment, etc.
        
    def _handle_camera_connection_error(self, error_context):
        """Recovery strategy for camera connection errors."""
        self.logger.info(f"Attempting recovery for camera connection error: {error_context.message}")
        # Could implement alternative connection methods, camera discovery, etc.
        
    def _handle_gpu_error(self, error_context):
        """Recovery strategy for GPU errors."""
        self.logger.info(f"Attempting recovery for GPU error: {error_context.message}")
        # Could implement fallback to CPU, different GPU, memory cleanup, etc.
        
    def _handle_memory_error(self, error_context):
        """Recovery strategy for memory errors."""
        self.logger.info(f"Attempting recovery for memory error: {error_context.message}")
        # Could implement memory cleanup, batch size reduction, etc.
        
    def _handle_network_error(self, error_context):
        """Recovery strategy for network errors."""
        self.logger.info(f"Attempting recovery for network error: {error_context.message}")
        # Could implement retry with different endpoints, offline mode, etc.
    
    def handle_vision_error(self, error: Exception, context: Dict[str, Any] = None):
        """Handle vision-specific errors with appropriate context."""
        if context is None:
            context = {}
        
        # Add vision-specific context
        context.update({
            'error_category': 'vision',
            'timestamp': time.time(),
        })
        
        # Determine severity based on error type
        if isinstance(error, (ModelLoadError, GPUError, ConfigurationError)):
            context['severity'] = ErrorSeverity.HIGH
        elif isinstance(error, (CameraTimeoutError, CameraConnectionError, NetworkError)):
            context['severity'] = ErrorSeverity.MEDIUM
        elif isinstance(error, (InsufficientMemoryError, CalibrationError)):
            context['severity'] = ErrorSeverity.CRITICAL
        else:
            context['severity'] = ErrorSeverity.MEDIUM
        
        return self.error_manager.capture(error, context)


def with_vision_error_handling(
    component: str,
    operation: str,
    error_handler: Optional[VisionErrorHandler] = None,
    retry_config: Optional[RetryConfig] = None,
    context_data: Optional[Dict[str, Any]] = None
):
    """Decorator for comprehensive vision error handling with retry logic."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or VisionErrorHandler()
            retry = retry_config or RetryConfig()
            context = context_data or {}
            context.update({'component': component, 'operation': operation})
            
            last_error = None
            
            for attempt in range(retry.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except retry.retry_on_exceptions as e:
                    last_error = e
                    
                    # Handle the error
                    error_context = handler.handle_vision_error(e, context.copy())
                    
                    # If this was the last attempt, re-raise
                    if attempt == retry.max_attempts - 1:
                        break
                    
                    # Calculate delay for next attempt
                    delay = retry.base_delay
                    if retry.exponential_backoff:
                        delay *= (2 ** attempt)
                    
                    if delay > retry.max_delay:
                        delay = retry.max_delay
                    
                    if retry.jitter:
                        import random
                        delay *= (0.5 + random.random())
                    
                    handler.logger.info(
                        f"Retrying {component}.{operation} in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{retry.max_attempts})"
                    )
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Handle non-retryable exceptions
                    handler.handle_vision_error(e, context)
                    raise
            
            # Re-raise the last error if all retries failed
            if last_error:
                raise last_error
        
        return wrapper
    return decorator


def with_camera_error_handling(
    camera_id: str,
    operation: str,
    timeout: float = 30.0,
    retry_config: Optional[RetryConfig] = None
):
    """Specialized decorator for camera operations."""
    camera_retry_config = retry_config or RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        retry_on_exceptions=(CameraTimeoutError, CameraConnectionError, NetworkError)
    )
    
    return with_vision_error_handling(
        component=f"camera_{camera_id}",
        operation=operation,
        retry_config=camera_retry_config,
        context_data={'camera_id': camera_id, 'timeout': timeout}
    )


def with_model_error_handling(
    model_name: str,
    operation: str,
    retry_config: Optional[RetryConfig] = None
):
    """Specialized decorator for model operations."""
    model_retry_config = retry_config or RetryConfig(
        max_attempts=2,
        base_delay=1.0,
        retry_on_exceptions=(ModelLoadError, ModelInferenceError, GPUError, InsufficientMemoryError)
    )
    
    return with_vision_error_handling(
        component=f"model_{model_name}",
        operation=operation,
        retry_config=model_retry_config,
        context_data={'model_name': model_name}
    )


def with_image_processing_error_handling(
    operation: str,
    retry_config: Optional[RetryConfig] = None
):
    """Specialized decorator for image processing operations."""
    processing_retry_config = retry_config or RetryConfig(
        max_attempts=2,
        base_delay=0.5,
        retry_on_exceptions=(ImageProcessingError, DataFormatError)
    )
    
    return with_vision_error_handling(
        component="image_processing",
        operation=operation,
        retry_config=processing_retry_config
    )


# Convenience functions for common error scenarios
def safe_model_load(model_loader: Callable, model_name: str, **kwargs):
    """Safely load a model with error handling."""
    @with_model_error_handling(model_name, "load")
    def _load():
        return model_loader(**kwargs)
    return _load()


def safe_camera_capture(camera_capture: Callable, camera_id: str, **kwargs):
    """Safely capture from camera with error handling."""
    @with_camera_error_handling(camera_id, "capture")
    def _capture():
        return camera_capture(**kwargs)
    return _capture()


def safe_image_process(processor: Callable, operation_name: str, **kwargs):
    """Safely process image with error handling."""
    @with_image_processing_error_handling(operation_name)
    def _process():
        return processor(**kwargs)
    return _process()


# Global vision error handler instance
vision_error_handler = VisionErrorHandler()
