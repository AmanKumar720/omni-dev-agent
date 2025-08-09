# YOLOv8 Object Detection Module

This module provides comprehensive object detection capabilities using Ultralytics YOLOv8 with support for single frame detection, batch processing, and real-time video streaming.

## Features

- **YOLOv8 Integration**: Full support for YOLOv8 models (nano, small, medium, large, extra-large)
- **Flexible Input Handling**: Support for numpy arrays, PIL images, and OpenCV images
- **Batch Processing**: Efficient processing of multiple images simultaneously
- **Video Streaming**: Real-time object detection with OpenCV VideoCapture
- **GPU Acceleration**: Automatic CUDA support when available
- **Configurable Confidence**: Adjustable confidence thresholds
- **Model Hub Integration**: Seamless integration with the existing model hub system
- **Lazy Loading**: Efficient memory usage with lazy import of heavy dependencies

## Installation

### Basic Installation

```bash
pip install -r requirements_object_detection.txt
```

### Manual Installation

```bash
# Core dependencies
pip install ultralytics torch torchvision opencv-python numpy

# Additional dependencies for async support
pip install aiohttp aiofiles
```

### GPU Support (Optional)

For CUDA GPU acceleration, ensure you have the appropriate PyTorch version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Basic Usage

```python
import asyncio
import numpy as np
from object_detection import create_detector

async def main():
    # Create and initialize detector
    detector = await create_detector(model_name="yolov8n", device="auto")
    
    # Load your image (replace with actual image loading)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect objects
    detections = detector.detect_objects(image, conf_threshold=0.25)
    
    # Process results
    for detection in detections:
        print(f"Found {detection.class_name} with confidence {detection.confidence:.3f}")
        print(f"Bounding box: {detection.bbox}")

# Run the example
asyncio.run(main())
```

### Convenience Function

```python
from object_detection import detect_objects

# If you have a pre-loaded model
# detections = detect_objects(image, model, conf_threshold=0.25)
```

## API Reference

### Classes

#### `ObjectDetector`

Main class for object detection operations.

```python
class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n", device: str = "auto", model_hub: Optional[ModelHub] = None)
    async def ensure_model_ready(self) -> bool
    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[DetectionResult]
    def detect_objects_batch(self, frames: List[np.ndarray], conf_threshold: float = 0.25) -> BatchDetectionResult
```

**Parameters:**
- `model_name`: YOLOv8 model variant ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x")
- `device`: Computation device ("auto", "cpu", "cuda")
- `model_hub`: Optional ModelHub instance for model management

#### `VideoStreamDetector`

Class for real-time video stream object detection.

```python
class VideoStreamDetector:
    def __init__(self, detector: ObjectDetector, source: Union[int, str] = 0)
    def start_stream(self) -> bool
    def stop_stream(self) -> bool
    def detect_stream(self, conf_threshold: float = 0.25, 
                     frame_callback: Optional[Callable] = None) -> Iterator[Tuple[np.ndarray, List[DetectionResult]]]
```

**Parameters:**
- `detector`: Initialized ObjectDetector instance
- `source`: Video source (0 for webcam, or path to video file)
- `conf_threshold`: Confidence threshold for detections
- `frame_callback`: Optional callback function for each processed frame

#### `DetectionResult`

Container for individual object detection results.

```python
@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    
    def to_dict(self) -> Dict[str, Any]
```

#### `BatchDetectionResult`

Container for batch processing results.

```python
@dataclass
class BatchDetectionResult:
    detections: List[List[DetectionResult]]  # List of detections per image
    processing_time: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]
```

### Functions

#### `create_detector(model_name: str = "yolov8n", device: str = "auto") -> ObjectDetector`

Convenience function to create and initialize an object detector.

#### `detect_objects(frame: np.ndarray, model: Any, conf_threshold: float = 0.25) -> List[DetectionResult]`

Convenience function for single frame detection with pre-loaded model.

#### `detect_objects_batch(frames: List[np.ndarray], model: Any, conf_threshold: float = 0.25) -> BatchDetectionResult`

Convenience function for batch detection with pre-loaded model.

## Usage Examples

### Single Frame Detection

```python
import asyncio
import cv2
from object_detection import create_detector

async def detect_single_frame():
    # Initialize detector
    detector = await create_detector("yolov8n")
    
    # Load image
    image = cv2.imread("path/to/your/image.jpg")
    
    # Detect objects
    detections = detector.detect_objects(image, conf_threshold=0.5)
    
    # Draw results
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{detection.class_name}: {detection.confidence:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite("result.jpg", image)

asyncio.run(detect_single_frame())
```

### Batch Processing

```python
import asyncio
import cv2
from object_detection import create_detector

async def detect_batch():
    detector = await create_detector("yolov8s")
    
    # Load multiple images
    images = []
    for i in range(5):
        img = cv2.imread(f"image_{i}.jpg")
        images.append(img)
    
    # Process batch
    batch_result = detector.detect_objects_batch(images, conf_threshold=0.3)
    
    print(f"Processed {batch_result.batch_size} images in {batch_result.processing_time:.2f} seconds")
    
    # Process results
    for i, detections in enumerate(batch_result.detections):
        print(f"Image {i}: {len(detections)} objects detected")
        for detection in detections:
            print(f"  - {detection.class_name}: {detection.confidence:.3f}")

asyncio.run(detect_batch())
```

### Video Stream Detection

```python
import asyncio
import cv2
from object_detection import create_detector, VideoStreamDetector

async def detect_video_stream():
    # Initialize detector
    detector = await create_detector("yolov8n")
    
    # Create stream detector
    stream_detector = VideoStreamDetector(detector, source=0)  # Use webcam
    
    # Start stream
    if stream_detector.start_stream():
        print("Starting detection. Press 'q' to quit.")
        
        # Process frames
        for frame, detections in stream_detector.detect_stream(conf_threshold=0.25):
            # Draw detections
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection.class_name}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Object Detection", frame)
            
            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

asyncio.run(detect_video_stream())
```

### Using with Vision Task Framework

```python
from object_detection import ObjectDetectionTask
from core import AIVisionAgent

# Create vision agent
agent = AIVisionAgent("detector_agent", "Object Detection Agent")

# Create detection task
task = ObjectDetectionTask("detect_001", model_name="yolov8n")

# Register task
agent.register_task(task)

# Execute task
import numpy as np
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = await agent.execute_task("detect_001", {
    'frame': test_image,
    'conf_threshold': 0.25
})

print(f"Detection result: {result.data}")
```

## Model Variants

YOLOv8 offers different model sizes optimized for different use cases:

| Model | Size (MB) | Speed | Accuracy | Use Case |
|-------|-----------|--------|----------|----------|
| yolov8n | 6.2 | Fastest | Lowest | Real-time applications, mobile devices |
| yolov8s | 21.5 | Fast | Good | Balanced performance |
| yolov8m | 49.7 | Medium | Better | Higher accuracy requirements |
| yolov8l | 83.7 | Slow | High | High accuracy applications |
| yolov8x | 136.7 | Slowest | Highest | Maximum accuracy requirements |

## Supported Object Classes

The module detects 80 COCO dataset classes:

- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Traffic**: traffic light, fire hydrant, stop sign, parking meter
- **Furniture**: bench, chair, couch, bed, dining table, toilet
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Sports**: sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- **Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster
- **Household**: sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
- **Other**: umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, backpack, potted plant

## Performance Optimization

### GPU Usage

```python
# Force GPU usage
detector = await create_detector("yolov8n", device="cuda")

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name()}")
else:
    print("Using CPU")
```

### Batch Size Optimization

```python
# Process larger batches for better GPU utilization
batch_size = 8
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    results = detector.detect_objects_batch(batch)
```

### Model Caching

The module automatically handles model caching through the ModelHub system:

```python
# Models are automatically cached after first download
detector1 = await create_detector("yolov8n")  # Downloads model
detector2 = await create_detector("yolov8n")  # Uses cached model
```

## Error Handling

```python
import asyncio
from object_detection import create_detector

async def safe_detection():
    try:
        detector = await create_detector("yolov8n")
        
        # Your detection code here
        detections = detector.detect_objects(image)
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
    except RuntimeError as e:
        print(f"Model loading failed: {e}")
    except Exception as e:
        print(f"Detection failed: {e}")

asyncio.run(safe_detection())
```

## Integration with Model Hub

The object detection module seamlessly integrates with the existing model hub system:

```python
from model_hub import get_model_hub

# Get model hub instance
hub = get_model_hub()

# Check available YOLOv8 models
yolo_models = hub.list_available_models(model_type=ModelType.YOLO_V8)
print(f"Available YOLO models: {[m['name'] for m in yolo_models]}")

# Manual model management
await hub.download_model("yolov8n")
model = hub.load_model("yolov8n", device=DeviceType.AUTO)
```

## Troubleshooting

### Common Issues

1. **Import Error: ultralytics not found**
   ```bash
   pip install ultralytics
   ```

2. **CUDA Out of Memory**
   - Use smaller model (yolov8n instead of yolov8x)
   - Reduce batch size
   - Use CPU: `device="cpu"`

3. **OpenCV Error: Cannot open camera**
   - Check camera permissions
   - Try different camera index: `source=1`
   - Use video file instead: `source="path/to/video.mp4"`

4. **Model Download Issues**
   - Check internet connection
   - Manually download model to cache directory
   - Clear model cache: `hub.remove_model("yolov8n")`

### Performance Issues

1. **Slow Detection Speed**
   - Use GPU: `device="cuda"`
   - Use smaller model: `model_name="yolov8n"`
   - Reduce image resolution
   - Increase confidence threshold

2. **Memory Usage**
   - Use batch processing instead of individual frames
   - Unload models when not needed: `detector.model = None`
   - Clear model cache periodically

## License

This module integrates with YOLOv8 which is licensed under AGPL-3.0. Please ensure compliance with Ultralytics licensing terms for commercial use.

## Contributing

When contributing to the object detection module:

1. Follow the existing code style and patterns
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update this README with new functionality
5. Test with multiple YOLOv8 model variants

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the example scripts
3. Check Ultralytics documentation: https://docs.ultralytics.com/
4. File issues with detailed error messages and system information
