# AI Vision Model Hub

A comprehensive model management utility for AI vision models with download, checksum validation, versioning, GPU/CPU selection, and lazy loading capabilities.

## Features

### Core Functionality
- **Model Registry**: Centralized registry of available models with metadata
- **Download Management**: Asynchronous model downloading with progress tracking
- **Checksum Validation**: SHA256 integrity verification for downloaded models
- **Version Control**: Support for multiple model versions
- **Device Management**: Automatic GPU/CPU device selection and optimization
- **Lazy Loading**: Models loaded on-demand with memory-efficient caching
- **Cache Management**: LRU-based cache with configurable size limits
- **Model Types**: Support for YOLOv8, SSD, ResNet, MobileNet, OCR, and Face Recognition models

### Supported Models

#### Object Detection
- **YOLOv8**: Ultra-fast real-time object detection
  - `yolov8n` - Nano version (6.2MB, fastest)
  - `yolov8s` - Small version (21.5MB, balanced)
- **SSD MobileNet v2**: Efficient object detection (67MB)

#### Image Classification  
- **ResNet-50**: Deep residual network (98MB)
- **MobileNet v2**: Mobile-optimized architecture (14MB)

#### OCR (Optical Character Recognition)
- **Tesseract English**: Text recognition model (5.2MB)

#### Face Recognition
- **HOG-based Face Recognition**: Face detection and recognition (3.1MB)

## Installation

### Dependencies

The model hub requires different dependencies based on which models you plan to use:

```bash
# Core dependencies
pip install aiohttp pathlib

# For YOLO models
pip install ultralytics torch torchvision

# For SSD/TensorFlow models  
pip install tensorflow opencv-python

# For PyTorch models (ResNet, MobileNet)
pip install torch torchvision

# For OCR models
pip install pytesseract pillow

# For Face Recognition
pip install face_recognition dlib
```

### Optional Dependencies
```bash
# For GPU acceleration
pip install torch[cuda] # or appropriate CUDA version

# For additional image processing
pip install opencv-python
```

## Usage

### Basic Usage

```python
import asyncio
from model_hub import get_model_hub, DeviceType, ModelType

# Initialize the model hub
hub = get_model_hub()

# List available models
models = hub.list_available_models()
for model in models:
    print(f"{model['name']}: {model['description']}")

# Get device information
device_info = hub.get_device_info()
print(f"Recommended device: {device_info['recommended_device']}")
```

### Download and Load Models

```python
async def setup_yolo_model():
    hub = get_model_hub()
    
    # Download model with progress tracking
    def progress_callback(progress):
        print(f"Download: {progress:.1%}")
    
    success = await hub.download_model("yolov8n", progress_callback)
    if success:
        print("Model downloaded successfully")
        
        # Load model for inference
        model = hub.load_model("yolov8n", DeviceType.AUTO)
        return model
    
    return None

# Run async function
model = asyncio.run(setup_yolo_model())
```

### One-Step Model Setup

```python
async def quick_setup():
    hub = get_model_hub()
    
    # Ensure model is downloaded and loaded in one call
    model = await hub.ensure_model_ready("mobilenet_v2", DeviceType.GPU)
    
    if model:
        print("Model ready for inference!")
        # Use model for predictions here
    
    return model

model = asyncio.run(quick_setup())
```

### Model Management

```python
hub = get_model_hub()

# Get cache information
cache_info = hub.get_cache_info()
print(f"Cache usage: {cache_info['utilization']:.1%}")
print(f"Total models: {cache_info['total_models']}")

# Validate model integrity
is_valid = hub.validate_model("yolov8n")
print(f"Model validation: {'PASSED' if is_valid else 'FAILED'}")

# Unload model from memory
hub.unload_model("yolov8n", DeviceType.AUTO)

# Remove model from cache
hub.remove_model("yolov8n")

# Clean up invalid cache entries
cleanup_stats = hub.cleanup_cache()
print(f"Cleaned up {cleanup_stats['removed_count']} invalid models")
```

### Custom Model Registration

```python
from datetime import datetime
from model_hub import ModelMetadata, ModelType, DeviceType

# Define custom model metadata
custom_model = ModelMetadata(
    name="my_custom_yolo",
    model_type=ModelType.YOLO_V8,
    version="1.0.0",
    size_mb=25.0,
    checksum_sha256="your-sha256-checksum-here",
    download_url="https://yourserver.com/model.pt",
    description="Custom trained YOLOv8 for specific domain",
    supported_devices=[DeviceType.CPU, DeviceType.GPU],
    dependencies=["ultralytics", "torch"],
    last_updated=datetime.now(),
    license="MIT",
    tags=["custom", "domain-specific"]
)

# Register with model hub
hub = get_model_hub()
success = hub.register_custom_model(custom_model)
print(f"Registration: {'SUCCESS' if success else 'FAILED'}")
```

## Configuration

### Cache Directory
By default, models are cached in `~/.omni_dev/models`. You can customize this:

```python
from pathlib import Path

# Custom cache directory
custom_cache = Path("/path/to/your/cache")
hub = ModelHub(cache_dir=custom_cache)
```

### Cache Size Limits
Configure maximum cache size (default: 10GB):

```python
# Set 5GB cache limit
hub = ModelHub(max_cache_size_gb=5.0)
```

### Device Selection
```python
# Force CPU usage
model = hub.load_model("yolov8n", DeviceType.CPU)

# Force GPU usage (falls back to CPU if unavailable)
model = hub.load_model("yolov8n", DeviceType.GPU)

# Automatic selection (default)
model = hub.load_model("yolov8n", DeviceType.AUTO)
```

## Architecture

### Core Components

1. **ModelRegistry**: Manages available model metadata and custom registrations
2. **DeviceManager**: Handles device selection and optimization
3. **CacheManager**: Implements LRU cache with size limits and persistence
4. **ModelDownloader**: Asynchronous downloading with progress tracking
5. **LazyModelLoader**: On-demand model loading with weak references
6. **ChecksumValidator**: SHA256 integrity verification

### Lazy Loading
Models are loaded only when needed and cached using weak references for automatic memory management:

```python
# First call loads the model
model1 = hub.load_model("yolov8n")

# Second call returns cached instance
model2 = hub.load_model("yolov8n")  # Same object as model1

# When references are released, model is automatically unloaded
del model1, model2  # Model garbage collected automatically
```

### Thread Safety
All operations are thread-safe with appropriate locking mechanisms for concurrent access.

## API Reference

### ModelHub Class

#### Methods

- `list_available_models(model_type=None)` → List[Dict]: List available models
- `download_model(model_name, progress_callback=None)` → bool: Download a model  
- `load_model(model_name, device=AUTO, version=None)` → Optional[Any]: Load model for inference
- `ensure_model_ready(model_name, device=AUTO)` → Optional[Any]: Download and load in one call
- `unload_model(model_name, device=AUTO, version=None)` → bool: Unload from memory
- `remove_model(model_name, version=None)` → bool: Remove from cache
- `validate_model(model_name, version=None)` → bool: Validate integrity
- `register_custom_model(metadata)` → bool: Register custom model
- `get_model_info(model_name)` → Optional[Dict]: Get detailed model information
- `get_cache_info()` → Dict: Get cache statistics
- `get_device_info()` → Dict: Get device information
- `cleanup_cache()` → Dict: Clean up invalid cache entries

### Enumerations

- `ModelType`: YOLO_V8, SSD, RESNET, MOBILENET, OCR, FACE_RECOGNITION  
- `DeviceType`: CPU, GPU, AUTO
- `ModelStatus`: NOT_DOWNLOADED, DOWNLOADING, VALIDATING, READY, ERROR, LOADING, LOADED

## Testing

Run the test suite:

```bash
cd src/components/ai_vision
python -m pytest tests/test_model_hub.py -v
```

Or run individual test classes:

```bash
python tests/test_model_hub.py
```

## Example Scripts

### Complete Demo
Run the comprehensive demo:

```bash
python model_hub_example.py
```

This demonstrates:
- Model listing and information
- Download with progress tracking
- Loading and device selection
- Cache management
- Custom model registration
- Memory management
- Cache cleanup

## Performance Considerations

### Memory Usage
- Models use lazy loading - only loaded when needed
- Weak references allow automatic garbage collection
- LRU cache eviction prevents memory bloat

### Network Usage  
- Resume interrupted downloads
- Progress tracking for large models
- Checksum validation prevents corrupted downloads

### Disk Usage
- Configurable cache size limits
- LRU eviction of least-used models
- Automatic cleanup of invalid models

## Error Handling

The model hub implements comprehensive error handling:

- **Download failures**: Automatic cleanup of partial downloads
- **Checksum mismatches**: Model rejected and removed  
- **Loading errors**: Graceful fallback with error reporting
- **Device unavailability**: Automatic fallback to CPU
- **Cache corruption**: Automatic detection and cleanup

## Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Model hub operations will now be logged
hub = get_model_hub()
```

## Contributing

When adding new model types:

1. Add model type to `ModelType` enum
2. Implement loader in `LazyModelLoader._load_model_by_type()`
3. Add model metadata to `ModelRegistry._initialize_model_registry()`
4. Update dependencies in requirements
5. Add tests for new functionality

## License

This model hub is part of the omni-dev-agent project. Individual models may have their own licenses as specified in their metadata.
