# Vision Error Handling Integration

This document provides comprehensive documentation for the integrated error handling system implemented across all vision operations in the Omni-Dev Agent.

## Overview

The vision error handling system provides:

- **Custom Exceptions**: Specific exception types for different vision error scenarios
- **Automatic Retry Logic**: Configurable retry mechanisms with exponential backoff
- **Comprehensive Logging**: Detailed error tracking and analysis
- **Recovery Strategies**: Automatic recovery attempts for common error types
- **Context Capture**: Rich context information for debugging and analysis

## Architecture

### Core Components

1. **ErrorManager** (`error_handling/error_manager.py`)
   - Central error management and logging
   - Recovery strategy registration
   - Error pattern analysis

2. **Vision Exceptions** (`error_handling/vision_exceptions.py`)
   - Custom exception hierarchy for vision operations
   - Specific error types with context information

3. **Vision Error Handler** (`error_handling/vision_error_handler.py`)
   - Vision-specific error handling logic
   - Retry configuration and decorators
   - Recovery strategy implementations

## Custom Exceptions

### Base Exception
- `VisionError`: Base class for all vision-related errors

### Model-Related Exceptions
- `ModelLoadError`: Model loading failures
- `ModelInferenceError`: Model inference failures
- `ModelValidationError`: Model validation failures
- `DependencyError`: Missing or incompatible dependencies

### Camera-Related Exceptions
- `CameraTimeoutError`: Camera operation timeouts
- `CameraConnectionError`: Camera connection failures
- `CameraNotFoundError`: Camera not found or accessible
- `CalibrationError`: Camera calibration failures

### Processing-Related Exceptions
- `ImageProcessingError`: Image processing failures
- `DataFormatError`: Incorrect input data formats
- `InsufficientMemoryError`: Memory-related issues
- `GPUError`: GPU operation failures

### Infrastructure-Related Exceptions
- `NetworkError`: Network operation failures
- `ConfigurationError`: Configuration issues
- `PermissionError`: Permission-related errors

## Error Handling Decorators

### General Vision Operations
```python
@with_vision_error_handling(
    component="ComponentName",
    operation="operation_name",
    retry_config=RetryConfig(max_attempts=3, base_delay=1.0)
)
def vision_function():
    # Your vision code here
    pass
```

### Model Operations
```python
@with_model_error_handling(
    model_name="yolov8n",
    operation="load",
    retry_config=RetryConfig(max_attempts=2, base_delay=5.0)
)
def load_model():
    # Model loading code
    pass
```

### Camera Operations
```python
@with_camera_error_handling(
    camera_id="camera_1",
    operation="capture",
    timeout=30.0,
    retry_config=RetryConfig(max_attempts=3, base_delay=2.0)
)
def capture_frame():
    # Camera capture code
    pass
```

### Image Processing Operations
```python
@with_image_processing_error_handling(
    operation="detect_objects",
    retry_config=RetryConfig(max_attempts=2, base_delay=0.5)
)
def process_image():
    # Image processing code
    pass
```

## Retry Configuration

The `RetryConfig` class allows fine-tuning of retry behavior:

```python
RetryConfig(
    max_attempts=3,        # Maximum number of retry attempts
    base_delay=1.0,        # Base delay between retries (seconds)
    max_delay=30.0,        # Maximum delay between retries
    exponential_backoff=True,  # Use exponential backoff
    jitter=True,           # Add random jitter to delays
    retry_on_exceptions=(VisionError,)  # Exception types to retry on
)
```

## Usage Examples

### Model Hub with Error Handling

The ModelHub class demonstrates comprehensive error handling:

```python
from error_handling import vision_error_handler, ModelLoadError

class ModelHub:
    async def download_model(self, model_name: str) -> bool:
        """Download a model with error handling."""
        try:
            # Download logic here
            model_path = await self.downloader.download_model(metadata)
            return True
        except Exception as e:
            vision_error_handler.handle_vision_error(e, {
                'component': 'ModelHub',
                'operation': 'download_model',
                'model_name': model_name
            })
            raise
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a model with error handling."""
        try:
            # Load model logic
            return loaded_model
        except Exception as e:
            if not isinstance(e, ModelLoadError):
                error = ModelLoadError(model_name, str(e))
                vision_error_handler.handle_vision_error(error, {
                    'component': 'ModelHub',
                    'operation': 'load_model',
                    'model_name': model_name
                })
                raise error
            raise
```

### Camera Manager with Error Handling

The CameraManager demonstrates camera-specific error handling:

```python
from error_handling import CameraConnectionError, CameraTimeoutError

class CPPlusCameraController:
    @with_camera_error_handling("camera_stream", "start_video_stream", timeout=10.0)
    def start_video_stream(self) -> cv2.VideoCapture:
        """Start video stream with error handling."""
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                raise CameraConnectionError(
                    self.credentials.ip_address,
                    "Failed to open video stream"
                )
            return cap
        except Exception as e:
            # Error handling logic
            raise
```

### Manual Error Capture

For operations not using decorators, manual error capture is available:

```python
from error_handling import vision_error_handler

try:
    # Your vision operation
    result = some_vision_operation()
except Exception as e:
    # Capture error with context
    vision_error_handler.error_manager.capture(e, {
        'component': 'YourComponent',
        'operation': 'your_operation',
        'additional_context': 'value'
    })
    # Re-raise or handle as needed
    raise
```

## Error Analysis and Monitoring

### Pattern Analysis

The error manager provides built-in error pattern analysis:

```python
analysis = vision_error_handler.error_manager.analyze_error_patterns()
print(f"Total errors: {analysis['total_errors']}")
print(f"Most common error: {analysis['most_common_error']}")
print(f"Most affected component: {analysis['most_affected_component']}")
```

### Recommendations

Get automated recommendations based on error patterns:

```python
recommendations = vision_error_handler.error_manager.get_error_recommendations()
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

## Recovery Strategies

The system includes built-in recovery strategies for common errors:

### Model Loading Recovery
- Fallback to CPU if GPU fails
- Alternative model variants
- Dependency checking and installation

### Camera Connection Recovery
- Alternative connection methods
- Automatic camera discovery
- Timeout adjustment

### Memory Management Recovery
- Automatic memory cleanup
- Batch size reduction
- Model optimization

### Network Recovery
- Retry with different endpoints
- Offline mode activation
- Connection pooling

## Integration in Existing Code

### Wrapping Existing Functions

To add error handling to existing vision functions:

1. **Import the decorators**:
```python
from error_handling import with_vision_error_handling
```

2. **Add the decorator**:
```python
@with_vision_error_handling("YourComponent", "your_operation")
def existing_vision_function():
    # Your existing code
    pass
```

3. **Handle specific exceptions**:
```python
from error_handling import ModelLoadError, CameraTimeoutError

try:
    result = your_vision_function()
except ModelLoadError as e:
    # Handle model loading error
    pass
except CameraTimeoutError as e:
    # Handle camera timeout
    pass
```

## Configuration

### Environment Variables

You can configure error handling behavior through environment variables:

```bash
# Error logging level
VISION_ERROR_LOG_LEVEL=INFO

# Default retry attempts
VISION_DEFAULT_RETRY_ATTEMPTS=3

# Default retry delay
VISION_DEFAULT_RETRY_DELAY=1.0

# Enable recovery strategies
VISION_ENABLE_RECOVERY=true
```

### Configuration File

Create a `vision_error_config.yaml` for advanced configuration:

```yaml
error_handling:
  log_level: INFO
  log_file: vision_errors.log
  
  retry:
    default_max_attempts: 3
    default_base_delay: 1.0
    default_max_delay: 30.0
    exponential_backoff: true
    jitter: true
  
  recovery:
    enabled: true
    strategies:
      - model_load_fallback
      - camera_reconnect
      - memory_cleanup
  
  monitoring:
    error_analysis_interval: 300  # 5 minutes
    alert_threshold: 10  # errors per interval
```

## Best Practices

### 1. Use Specific Exception Types
```python
# Good
raise ModelLoadError("yolov8n", "Model file corrupted")

# Avoid
raise Exception("Model loading failed")
```

### 2. Provide Rich Context
```python
vision_error_handler.handle_vision_error(error, {
    'component': 'ObjectDetector',
    'operation': 'detect',
    'model_name': 'yolov8n',
    'image_shape': image.shape,
    'confidence_threshold': 0.5,
    'device': 'cuda:0'
})
```

### 3. Configure Appropriate Retry Logic
```python
# For network operations - more retries
RetryConfig(max_attempts=5, base_delay=2.0)

# For heavy computations - fewer retries
RetryConfig(max_attempts=2, base_delay=5.0)

# For quick operations - fast retries
RetryConfig(max_attempts=3, base_delay=0.5)
```

### 4. Implement Graceful Degradation
```python
try:
    # Try primary vision algorithm
    result = advanced_detection(image)
except ModelInferenceError:
    try:
        # Fallback to simpler algorithm
        result = basic_detection(image)
    except Exception:
        # Last resort - return empty result
        result = []
```

## Testing Error Handling

### Unit Tests

Create tests for error scenarios:

```python
import pytest
from error_handling import ModelLoadError

def test_model_load_error_handling():
    with pytest.raises(ModelLoadError) as exc_info:
        # Trigger model load error
        load_invalid_model()
    
    assert exc_info.value.model_name == "invalid_model"
    assert "corrupted" in str(exc_info.value)
```

### Integration Tests

Test error handling in realistic scenarios:

```python
def test_camera_timeout_recovery():
    # Simulate camera timeout
    with patch('cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = False
        
        # Should retry and eventually fail gracefully
        result = camera_manager.start_video_stream()
        assert result is None
```

## Demo and Examples

Run the comprehensive demo to see error handling in action:

```bash
cd examples
python vision_error_handling_demo.py
```

This demo demonstrates:
- Model loading with retries
- Camera connection error handling
- Image processing error recovery
- Error pattern analysis
- Recovery strategy execution

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure error_handling package is in PYTHONPATH
   - Check relative import paths

2. **Decorator Not Working**
   - Verify decorator parameters
   - Check if function is async (use appropriate decorator variant)

3. **Retries Not Happening**
   - Check exception type matches retry_on_exceptions
   - Verify RetryConfig settings

4. **Memory Issues**
   - Monitor error log file size
   - Implement log rotation
   - Tune error history retention

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.getLogger('error_handling').setLevel(logging.DEBUG)
```

## Future Enhancements

Planned improvements include:

1. **Metrics Integration**: Prometheus/Grafana dashboards
2. **Alert System**: Email/Slack notifications for critical errors  
3. **Adaptive Retry**: ML-based retry strategy optimization
4. **Distributed Tracing**: Integration with OpenTelemetry
5. **Error Prediction**: Proactive error prevention

## Conclusion

The integrated error handling system provides robust, production-ready error management for all vision operations. It ensures system reliability, aids in debugging, and enables graceful degradation under error conditions.

For additional support or feature requests, please refer to the project documentation or create an issue in the project repository.
