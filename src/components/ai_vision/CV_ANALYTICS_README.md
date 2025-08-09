# Computer Vision Analytics Module

A comprehensive computer vision analytics system that implements motion detection via frame differencing, scene segmentation, and basic visual reasoning with analytics event emission for Learning Engine integration.

## Features

### 🎯 Core Capabilities

- **Motion Detection**: Frame differencing-based motion detection with configurable sensitivity
- **Scene Segmentation**: K-means clustering for color-based scene segmentation
- **Visual Reasoning**: Rule-based reasoning engine with customizable rules
- **Event Emission**: Analytics events emitted to Learning Engine for continuous improvement
- **Real-time Processing**: Async processing support for real-time video streams
- **Extensible Rules**: Easy-to-add custom visual reasoning rules

### 📊 Analytics Events

The module generates various types of analytics events:

- `MOTION_DETECTED`: When motion is detected in the scene
- `SCENE_CHANGE`: When significant scene changes occur
- `DOOR_STATE_CHANGE`: When door open/closed state changes
- `ACTIVITY_DETECTED`: When general activity is detected
- `RULE_TRIGGERED`: When custom visual reasoning rules are triggered
- `ANOMALY_DETECTED`: When anomalies are detected (extensible)

### 🔧 Built-in Visual Rules

1. **Door Open Detection**: Detects if a door is open based on edge density and motion
2. **Activity Detection**: Monitors general activity levels based on motion patterns
3. **Custom Rules**: Framework for adding domain-specific rules

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements_cv_analytics.txt
```

### Core Dependencies

- `opencv-python >= 4.8.0`: Computer vision operations
- `numpy >= 1.21.0`: Numerical computations
- `scikit-learn >= 1.0.0`: Machine learning algorithms
- `asyncio`: Asynchronous processing

## Quick Start

### Basic Usage

```python
import asyncio
import cv2
import numpy as np
from computer_vision_analytics import ComputerVisionAnalyticsAgent

async def main():
    # Create analytics agent
    agent = ComputerVisionAnalyticsAgent(
        agent_id="my_analytics_agent"
    )
    
    # Create and execute analytics task
    task_id = agent.create_analytics_task()
    
    # Load or create a frame
    frame = cv2.imread("test_image.jpg")
    
    # Process the frame
    result = await agent.execute_task(task_id, frame)
    
    # Check results
    if result and result.data:
        print(f"Motion detected: {result.data['motion_result']['motion_detected']}")
        print(f"Events generated: {len(result.data['events'])}")
        
        # Print events
        for event in result.data['events']:
            print(f"Event: {event['event_type']} (confidence: {event['confidence']:.2f})")

# Run the example
asyncio.run(main())
```

### Video Processing

```python
async def process_video(video_path):
    agent = ComputerVisionAnalyticsAgent()
    task_id = agent.create_analytics_task()
    
    # Process entire video file
    result = await agent.execute_task(task_id, video_path)
    
    print(f"Processed {result.data['frames_processed']} frames")
    print(f"Total events: {len(result.data['events_generated'])}")
    
    # Get analytics summary
    summary = result.data['analytics_summary']
    print(f"Most common event: {summary['most_common_event']}")
```

### Custom Event Handling

```python
from computer_vision_analytics import AnalyticsEvent, AnalyticsEventType

def my_event_handler(event: AnalyticsEvent):
    if event.event_type == AnalyticsEventType.DOOR_STATE_CHANGE:
        door_open = event.metadata.get('door_open', False)
        print(f"Door is {'open' if door_open else 'closed'}")
    elif event.event_type == AnalyticsEventType.MOTION_DETECTED:
        motion_area = event.metadata.get('motion_area', 0)
        print(f"Motion detected with area: {motion_area} pixels")

# Add event handler to agent
agent = ComputerVisionAnalyticsAgent()
agent.event_emitter.add_event_handler(my_event_handler)
```

## Configuration

### Default Configuration

```python
DEFAULT_ANALYTICS_CONFIG = {
    'motion_config': {
        'threshold': 25,           # Motion detection threshold
        'min_area': 500,          # Minimum area for motion regions
        'blur_size': 21           # Gaussian blur kernel size
    },
    'segmentation_config': {
        'num_clusters': 8,        # Number of K-means clusters
        'resize_factor': 0.25     # Resize factor for performance
    },
    'reasoning_config': {
        'door_detection': {
            'enabled': True,
            'region': {'x': 100, 'y': 50, 'width': 200, 'height': 300},
            'threshold': 0.3      # Edge density threshold
        },
        'activity_detection': {
            'enabled': True,
            'motion_threshold': 0.05  # Activity threshold
        }
    }
}
```

### Custom Configuration

```python
# Create custom configuration
custom_config = {
    'motion_config': {
        'threshold': 15,          # More sensitive motion detection
        'min_area': 200,         # Smaller minimum area
    },
    'reasoning_config': {
        'door_detection': {
            'enabled': True,
            'region': {'x': 200, 'y': 100, 'width': 300, 'height': 400},
            'threshold': 0.25    # Lower threshold for door detection
        }
    }
}

# Create agent with custom config
agent = ComputerVisionAnalyticsAgent(
    agent_id="custom_agent",
    analytics_config=custom_config
)
```

## Advanced Usage

### Creating Custom Visual Rules

```python
from computer_vision_analytics import VisualRule

class RedObjectRule(VisualRule):
    def __init__(self, red_threshold=0.01):
        super().__init__("red_object", "Red Object Detection", 
                        "Detects red objects in the scene")
        self.red_threshold = red_threshold
    
    def evaluate(self, frame, motion_result, segmentation_result):
        try:
            # Convert to HSV for better red detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Red color range
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # Calculate red area ratio
            red_area = np.sum(mask > 0)
            frame_area = frame.shape[0] * frame.shape[1]
            red_ratio = red_area / frame_area if frame_area > 0 else 0
            
            triggered = red_ratio > self.red_threshold
            confidence = min(red_ratio * 10, 1.0)
            
            return {
                'triggered': triggered,
                'red_ratio': red_ratio,
                'confidence': confidence
            }
        except Exception as e:
            return {
                'triggered': False,
                'error': str(e),
                'confidence': 0.0
            }

# Add custom rule to visual reasoner
task = ComputerVisionAnalyticsTask("custom_task", config)
custom_rule = RedObjectRule(red_threshold=0.02)
task.visual_reasoner.add_rule(custom_rule)
```

### Processing Multiple Video Sources

```python
async def multi_source_processing(video_sources):
    agents = []
    tasks = []
    
    # Create agent for each source
    for i, source in enumerate(video_sources):
        agent = ComputerVisionAnalyticsAgent(
            agent_id=f"agent_{i}",
            analytics_config=custom_config_for_source(source)
        )
        agents.append(agent)
        
        # Create task
        task_id = agent.create_analytics_task()
        task = agent.execute_task(task_id, source)
        tasks.append(task)
    
    # Process all sources concurrently
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    total_events = sum(len(r.data.get('events_generated', [])) for r in results if r)
    print(f"Total events across all sources: {total_events}")
    
    return results
```

### Learning Engine Integration

The module automatically integrates with the Learning Engine:

```python
from src.learning.learning_engine import global_learning_engine

# Process some frames to generate learning data
agent = ComputerVisionAnalyticsAgent()
# ... process frames ...

# Check learning insights
insights = global_learning_engine.get_learning_insights()
print(f"Total experiences: {insights['total_experiences']}")
print(f"Success rate: {insights['overall_success_rate']:.2f}")

# Get recommendations for improvement
recommendations = insights.get('recommendations', [])
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

## API Reference

### Classes

#### `ComputerVisionAnalyticsAgent`
Main agent class for managing computer vision analytics tasks.

**Methods:**
- `create_analytics_task(analytics_config=None) -> str`: Create new analytics task
- `execute_task(task_id: str, input_data: Any) -> VisionResult`: Execute analytics task
- `get_analytics_summary() -> Dict`: Get processing summary

#### `ComputerVisionAnalyticsTask`
Task class for executing computer vision analytics on input data.

**Methods:**
- `execute(input_data: Any) -> VisionResult`: Execute analytics processing
- `validate_input(input_data: Any) -> bool`: Validate input data

#### `MotionDetector`
Frame differencing-based motion detection.

**Methods:**
- `detect_motion(frame: np.ndarray) -> MotionResult`: Detect motion in frame

#### `SceneSegmenter`
K-means based scene segmentation.

**Methods:**
- `segment_scene(frame: np.ndarray) -> SegmentationResult`: Segment scene

#### `VisualReasoner`
Rule-based visual reasoning engine.

**Methods:**
- `add_rule(rule: VisualRule)`: Add custom visual rule
- `evaluate_rules(...) -> ReasoningResult`: Evaluate all rules

#### `AnalyticsEventEmitter`
Event emission system for Learning Engine integration.

**Methods:**
- `emit_event(event: AnalyticsEvent)`: Emit analytics event
- `add_event_handler(handler: callable)`: Add custom event handler

### Data Classes

#### `AnalyticsEvent`
```python
@dataclass
class AnalyticsEvent:
    event_id: str
    timestamp: datetime
    event_type: AnalyticsEventType
    confidence: float
    location: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    frame_info: Optional[Dict[str, Any]] = None
```

#### `MotionResult`
```python
@dataclass
class MotionResult:
    motion_detected: bool
    motion_area: float
    motion_regions: List[Dict[str, Any]]
    confidence: float
    frame_diff: Optional[np.ndarray] = None
```

## Testing

Run the test suite:

```bash
# Basic tests
python test_cv_analytics_basic.py

# Or with pytest
pytest test_cv_analytics_basic.py -v

# Run example script
python example_cv_analytics.py
```

## Performance Considerations

### Optimization Tips

1. **Frame Resizing**: Use `resize_factor` in segmentation config for better performance
2. **Motion Sensitivity**: Adjust `threshold` and `min_area` for optimal motion detection
3. **Concurrent Processing**: Use multiple agents for parallel video stream processing
4. **Memory Management**: Process frames in batches for large video files

### Memory Usage

- Scene segmentation uses resized frames (25% by default)
- Motion detection maintains only previous frame in memory
- Event queue processes events in background thread

### Performance Benchmarks

Typical processing speeds (on modern hardware):
- Single frame (480x640): ~10-50ms
- Motion detection: ~5-15ms
- Scene segmentation: ~20-100ms
- Visual reasoning: ~1-5ms

## Troubleshooting

### Common Issues

1. **OpenCV Import Error**:
   ```bash
   pip install opencv-python
   ```

2. **Memory Issues with Large Videos**:
   - Process frames in smaller batches
   - Reduce segmentation resize_factor
   - Use frame skipping for real-time processing

3. **Low Motion Detection Sensitivity**:
   - Reduce `threshold` value
   - Reduce `min_area` value
   - Adjust `blur_size` for noise reduction

4. **Custom Rules Not Triggering**:
   - Check rule evaluation logic
   - Verify confidence calculation
   - Debug with print statements in rule evaluation

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all module operations will be logged
agent = ComputerVisionAnalyticsAgent()
```

## Examples

See `example_cv_analytics.py` for comprehensive usage examples including:

- Single frame processing
- Video sequence processing
- Custom visual rules
- Event handling
- Learning Engine integration

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure Learning Engine integration works
5. Add configuration options where appropriate

## License

This module is part of the Omni-Dev Agent project and follows the same licensing terms.
