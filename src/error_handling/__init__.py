# Error handling and recovery package

from .error_manager import ErrorManager, ErrorSeverity, global_error_manager
from .vision_exceptions import *
from .vision_error_handler import (
    VisionErrorHandler, RetryConfig, vision_error_handler,
    with_vision_error_handling, with_camera_error_handling, 
    with_model_error_handling, with_image_processing_error_handling,
    safe_model_load, safe_camera_capture, safe_image_process
)

# Import LLM overload handling components
try:
    from .llm_overload_handler import (
        LLMOverloadHandler, 
        OverloadConfig, 
        APILimits,
        OverloadType,
        OverloadSeverity,
        with_llm_overload_protection
    )
    from .llm_config_loader import (
        LLMConfigLoader,
        get_config_loader,
        create_configured_handler
    )
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import logging
    logging.getLogger(__name__).warning(f"LLM overload handling not available: {e}")
    
    # Define dummy classes to prevent import errors
    class LLMOverloadHandler:
        def __init__(self, *args, **kwargs):
            pass
    
    class OverloadConfig:
        def __init__(self, *args, **kwargs):
            pass
    
    def with_llm_overload_protection(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
