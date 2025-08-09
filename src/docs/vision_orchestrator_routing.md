# Vision Task Routing in Orchestrator

This document describes the new vision task routing functionality added to the Orchestrator class, which enables handling of computer vision tasks including object detection and analytics processing.

## Overview

The Orchestrator now supports vision task routing through the new `handle_request` method, which provides a standardized interface for executing vision tasks with consistent response formats and error handling.

## Features

- **Object Detection**: YOLOv8-based object detection for single frames and batch processing
- **Computer Vision Analytics**: Motion detection, scene segmentation, and visual reasoning
- **Standardized Response Format**: Consistent JSON responses with status, data, confidence, and metadata
- **Error Handling**: Comprehensive error handling with detailed error messages
- **Async Support**: Full async/await support for non-blocking execution

## API Reference

### Main Method

```python
async def handle_request(
    self, 
    type: str, 
    task: str, 
    payload: Dict[str, Any], 
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `type`: Request type (currently supports 'vision')
- `task`: Specific task within the type ('object_detection', 'analytics')
- `payload`: Task-specific data payload
- `**kwargs`: Additional parameters

**Returns:** Standardized response dictionary with `status`, `data`, `confidence`, and `metadata`

### Supported Vision Tasks

#### 1. Object Detection

Performs YOLOv8-based object detection on images.

**Task Name:** `'object_detection'`

**Payload Parameters:**
- `frame` (optional): Single image as numpy array (H, W, C)
- `batch_frames` (optional): List of images for batch processing
- `conf_threshold` (default: 0.25): Confidence threshold for detections
- `model_name` (default: 'yolov8n'): YOLOv8 model variant

**Example:**
```python
result = await orchestrator.handle_request(
    type='vision',
    task='object_detection',
    payload={
        'frame': image_array,
        'conf_threshold': 0.25,
        'model_name': 'yolov8n'
    }
)
```

#### 2. Computer Vision Analytics

Performs motion detection, scene analysis, and visual reasoning.

**Task Name:** `'analytics'`

**Payload Parameters:**
- `input_data` (required): Video file path, single frame, or frame sequence
- `analytics_config` (optional): Configuration for analytics processing

**Example:**
```python
result = await orchestrator.handle_request(
    type='vision',
    task='analytics',
    payload={
        'input_data': video_path_or_frame,
        'analytics_config': {
            'motion_config': {'threshold': 25},
            'reasoning_config': {
                'activity_detection': {'enabled': True}
            }
        }
    }
)
```

## Response Format

All responses follow a standardized format:

### Success Response

```json
{
    "status": "success",
    "data": {
        // Task-specific result data
    },
    "confidence": 0.85,
    "metadata": {
        "task_id": "obj_det_2024-01-01T12:00:00",
        "vision_task": "object_detection",
        "timestamp": "2024-01-01T12:00:00",
        // Additional task-specific metadata
    }
}
```

### Error Response

```json
{
    "status": "error",
    "error": "Error description",
    "data": null,
    "metadata": {
        "vision_task": "object_detection",
        "timestamp": "2024-01-01T12:00:00",
        // Additional context information
    }
}
```

## Usage Examples

### Basic Object Detection

```python
import numpy as np
from src.core.orchestration import Orchestrator

async def detect_objects_example():
    orchestrator = Orchestrator()
    
    # Create or load image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Perform object detection
    result = await orchestrator.handle_request(
        type='vision',
        task='object_detection',
        payload={
            'frame': image,
            'conf_threshold': 0.25
        }
    )
    
    if result['status'] == 'success':
        detections = result['data']
        print(f"Found {len(detections)} objects")
        for detection in detections:
            print(f"- {detection['class_name']}: {detection['confidence']:.3f}")
    else:
        print(f"Detection failed: {result['error']}")

# Run example
import asyncio
asyncio.run(detect_objects_example())
```

### Batch Object Detection

```python
async def batch_detection_example():
    orchestrator = Orchestrator()
    
    # Create batch of images
    batch_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    result = await orchestrator.handle_request(
        type='vision',
        task='object_detection',
        payload={
            'batch_frames': batch_images,
            'conf_threshold': 0.3,
            'model_name': 'yolov8s'
        }
    )
    
    if result['status'] == 'success':
        batch_data = result['data']
        print(f"Processed {batch_data['batch_size']} frames")
        print(f"Total processing time: {batch_data['processing_time']:.3f}s")
```

### Computer Vision Analytics

```python
async def analytics_example():
    orchestrator = Orchestrator()
    
    # Single frame analytics
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = await orchestrator.handle_request(
        type='vision',
        task='analytics',
        payload={
            'input_data': frame,
            'analytics_config': {
                'motion_config': {
                    'threshold': 25,
                    'min_area': 500
                },
                'reasoning_config': {
                    'activity_detection': {
                        'enabled': True,
                        'motion_threshold': 0.05
                    }
                }
            }
        }
    )
    
    if result['status'] == 'success':
        data = result['data']
        print(f"Motion detected: {data['motion_result']['motion_detected']}")
        print(f"Events generated: {len(data['events'])}")
```

## Error Handling

The system provides comprehensive error handling:

```python
async def error_handling_example():
    orchestrator = Orchestrator()
    
    # Example: Missing required parameters
    result = await orchestrator.handle_request(
        type='vision',
        task='object_detection',
        payload={}  # Missing frame or batch_frames
    )
    
    if result['status'] == 'error':
        print(f"Error: {result['error']}")
        print(f"Task: {result['metadata']['vision_task']}")
        print(f"Timestamp: {result['metadata']['timestamp']}")
```

## Dependencies

The vision task routing requires the following packages:

```bash
pip install ultralytics torch torchvision opencv-python numpy
```

For analytics functionality, additional dependencies from the learning engine are required.

## Implementation Details

### Architecture

The vision task routing is implemented through several layers:

1. **Orchestrator.handle_request()** - Main entry point and request routing
2. **Vision-specific handlers** - `_handle_object_detection()`, `_handle_computer_vision_analytics()`
3. **Task execution** - Uses `ObjectDetectionTask` and `ComputerVisionAnalyticsTask`
4. **Response standardization** - Consistent response format across all tasks

### Task Lifecycle

1. Request validation and parameter extraction
2. Task instantiation with configuration
3. Input data validation
4. Task execution (async)
5. Response standardization
6. Cleanup (for analytics tasks)

### Integration with Existing Components

The vision routing integrates with:
- **AI Vision Components**: Object detection and analytics modules
- **Learning Engine**: Analytics events are sent to the learning engine
- **Model Hub**: Centralized model management and caching
- **Logging System**: Comprehensive logging throughout execution

## Testing

### Running Tests

```bash
# Run the test suite
python tests/test_vision_orchestrator.py

# Run the demo
python examples/vision_orchestrator_demo.py
```

### Test Coverage

The test suite covers:
- Request routing and validation
- Error handling scenarios
- Response format validation
- Mocked task execution
- Integration tests (when dependencies available)

## Performance Considerations

### Optimization Tips

1. **Model Caching**: Models are cached after first load for better performance
2. **Batch Processing**: Use batch detection for multiple images
3. **Confidence Thresholds**: Adjust thresholds based on use case
4. **Resource Management**: Tasks are properly cleaned up after execution

### Memory Management

- Object detection tasks manage model memory automatically
- Analytics tasks clean up resources after execution
- Batch processing optimizes memory usage

## Future Enhancements

### Planned Features

1. **Additional Vision Tasks**:
   - Image classification
   - Face recognition
   - OCR (Optical Character Recognition)
   - Custom model support

2. **Streaming Support**:
   - Real-time video stream processing
   - WebRTC integration
   - Live analytics dashboards

3. **Advanced Analytics**:
   - Object tracking
   - Anomaly detection
   - Behavior analysis

4. **Performance Improvements**:
   - GPU optimization
   - Model quantization
   - Distributed processing

### Extension Points

The system is designed for easy extension:

```python
# Add new vision task handler
async def _handle_custom_task(self, payload: Dict[str, Any], **kwargs):
    # Implement custom vision task
    pass

# Register in _handle_vision_request
elif task == 'custom_task':
    return await self._handle_custom_task(payload, **kwargs)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Loading Failures**: Check internet connection for model downloads
3. **Memory Issues**: Reduce batch sizes or use smaller models
4. **Performance Issues**: Enable GPU acceleration if available

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To contribute to the vision task routing system:

1. Follow the existing code patterns and response formats
2. Add comprehensive tests for new functionality
3. Update documentation for new features
4. Ensure backward compatibility

## License

This functionality is part of the larger omni-dev-agent project and follows the same license terms.
