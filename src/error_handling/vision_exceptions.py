"""
Custom exceptions for AI Vision operations
Provides specific exception types for different vision error scenarios.
"""

class VisionError(Exception):
    """Base class for all vision-related errors."""
    def __init__(self, message: str, component: str = None, operation: str = None, details: dict = None):
        self.component = component
        self.operation = operation
        self.details = details or {}
        super().__init__(message)


class ModelLoadError(VisionError):
    """Raised when model loading fails."""
    def __init__(self, model_name: str, message: str = None, **kwargs):
        self.model_name = model_name
        msg = message or f"Failed to load model '{model_name}'"
        super().__init__(msg, **kwargs)


class CameraTimeoutError(VisionError):
    """Raised when camera operations timeout."""
    def __init__(self, camera_id: str, timeout_duration: float, message: str = None, **kwargs):
        self.camera_id = camera_id
        self.timeout_duration = timeout_duration
        msg = message or f"Camera '{camera_id}' operation timed out after {timeout_duration}s"
        super().__init__(msg, **kwargs)


class CameraConnectionError(VisionError):
    """Raised when camera connection fails."""
    def __init__(self, camera_id: str, message: str = None, **kwargs):
        self.camera_id = camera_id
        msg = message or f"Failed to connect to camera '{camera_id}'"
        super().__init__(msg, **kwargs)


class CameraNotFoundError(VisionError):
    """Raised when specified camera is not found."""
    def __init__(self, camera_id: str, message: str = None, **kwargs):
        self.camera_id = camera_id
        msg = message or f"Camera '{camera_id}' not found"
        super().__init__(msg, **kwargs)


class ImageProcessingError(VisionError):
    """Raised when image processing operations fail."""
    def __init__(self, operation: str, image_info: str = None, message: str = None, **kwargs):
        self.image_info = image_info
        msg = message or f"Image processing failed during '{operation}'"
        if image_info:
            msg += f" for image: {image_info}"
        super().__init__(msg, **kwargs)


class ModelInferenceError(VisionError):
    """Raised when model inference fails."""
    def __init__(self, model_name: str, input_shape: tuple = None, message: str = None, **kwargs):
        self.model_name = model_name
        self.input_shape = input_shape
        msg = message or f"Model inference failed for '{model_name}'"
        if input_shape:
            msg += f" with input shape {input_shape}"
        super().__init__(msg, **kwargs)


class ModelValidationError(VisionError):
    """Raised when model validation fails."""
    def __init__(self, model_name: str, validation_type: str, message: str = None, **kwargs):
        self.model_name = model_name
        self.validation_type = validation_type
        msg = message or f"Model '{model_name}' failed {validation_type} validation"
        super().__init__(msg, **kwargs)


class InsufficientMemoryError(VisionError):
    """Raised when there's insufficient memory for vision operations."""
    def __init__(self, required_mb: float = None, available_mb: float = None, message: str = None, **kwargs):
        self.required_mb = required_mb
        self.available_mb = available_mb
        msg = message or "Insufficient memory for vision operation"
        if required_mb and available_mb:
            msg += f" (required: {required_mb}MB, available: {available_mb}MB)"
        super().__init__(msg, **kwargs)


class GPUError(VisionError):
    """Raised when GPU-related operations fail."""
    def __init__(self, gpu_id: int = None, message: str = None, **kwargs):
        self.gpu_id = gpu_id
        msg = message or "GPU operation failed"
        if gpu_id is not None:
            msg += f" on GPU {gpu_id}"
        super().__init__(msg, **kwargs)


class ConfigurationError(VisionError):
    """Raised when vision system configuration is invalid."""
    def __init__(self, config_key: str, message: str = None, **kwargs):
        self.config_key = config_key
        msg = message or f"Invalid configuration for '{config_key}'"
        super().__init__(msg, **kwargs)


class DataFormatError(VisionError):
    """Raised when input data format is incorrect."""
    def __init__(self, expected_format: str, actual_format: str = None, message: str = None, **kwargs):
        self.expected_format = expected_format
        self.actual_format = actual_format
        msg = message or f"Data format error: expected '{expected_format}'"
        if actual_format:
            msg += f", got '{actual_format}'"
        super().__init__(msg, **kwargs)


class DependencyError(VisionError):
    """Raised when required dependencies are missing or incompatible."""
    def __init__(self, dependency: str, message: str = None, **kwargs):
        self.dependency = dependency
        msg = message or f"Dependency error: '{dependency}' is missing or incompatible"
        super().__init__(msg, **kwargs)


class NetworkError(VisionError):
    """Raised when network operations fail during vision processing."""
    def __init__(self, url: str = None, status_code: int = None, message: str = None, **kwargs):
        self.url = url
        self.status_code = status_code
        msg = message or "Network operation failed"
        if url:
            msg += f" for URL: {url}"
        if status_code:
            msg += f" (status code: {status_code})"
        super().__init__(msg, **kwargs)


class PermissionError(VisionError):
    """Raised when permission-related errors occur."""
    def __init__(self, resource: str, message: str = None, **kwargs):
        self.resource = resource
        msg = message or f"Permission denied for resource: '{resource}'"
        super().__init__(msg, **kwargs)


class CalibrationError(VisionError):
    """Raised when camera calibration fails."""
    def __init__(self, camera_id: str, calibration_type: str, message: str = None, **kwargs):
        self.camera_id = camera_id
        self.calibration_type = calibration_type
        msg = message or f"Camera '{camera_id}' {calibration_type} calibration failed"
        super().__init__(msg, **kwargs)
