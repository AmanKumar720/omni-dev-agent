# Performance Optimization System

A comprehensive performance optimization system with GPU acceleration, CUDA support, auto-fallback to CPU, half-precision inference, batch aggregation, and asynchronous queue for video frames.

## Features

### 🚀 **GPU Acceleration & CUDA Support**
- **Automatic Device Detection**: Intelligently detects and selects optimal compute device (CUDA GPU, Apple Silicon MPS, or CPU)
- **CUDA Memory Management**: Configurable memory fraction and automatic cache management
- **Multi-GPU Support**: Automatic selection of best available GPU based on memory and compute capability
- **Graceful Fallback**: Seamless fallback to CPU when GPU is unavailable

### ⚡ **Half-Precision Inference**
- **Automatic Precision Selection**: Chooses optimal precision based on device capabilities
- **FP16 Support**: Hardware-accelerated half-precision for compatible GPUs (Pascal architecture and newer)
- **Mixed Precision**: Supports both FP16 and FP32 with automatic casting
- **Performance Boost**: Up to 2x speedup with FP16 on supported hardware

### 📦 **Batch Aggregation**
- **Dynamic Batching**: Intelligent frame aggregation with configurable batch sizes
- **Timeout-Based Processing**: Processes partial batches to prevent delays
- **Memory Efficient**: Optimized memory usage with configurable buffer sizes
- **Priority Queuing**: Support for frame priority scheduling

### 🔄 **Asynchronous Processing**
- **Non-Blocking Operations**: Fully asynchronous frame processing pipeline
- **Queue Management**: Configurable queue sizes with overflow protection
- **Concurrent Processing**: Multiple worker threads for optimal throughput
- **Real-Time Streaming**: Designed for live video processing applications

### 📊 **Performance Monitoring**
- **Real-Time Metrics**: Live GPU/CPU utilization, memory usage, and throughput monitoring
- **Detailed Profiling**: Optional performance profiling with timing breakdowns
- **System Statistics**: Comprehensive system resource monitoring
- **Benchmarking Tools**: Built-in benchmarking suite for performance analysis

## Installation

### Quick Setup

```bash
# Clone or download the performance optimization components
# Run the automated setup script
python setup_performance_optimization.py --install-cuda 11.8 --test --benchmark
```

### Manual Installation

1. **Install PyTorch with CUDA support:**

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only
pip install torch torchvision torchaudio
```

2. **Install requirements:**

```bash
pip install -r requirements_performance_optimization.txt
```

### System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 or higher
- **CUDA** (optional): 11.8 or 12.1 for GPU acceleration
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **GPU** (optional): NVIDIA GPU with Compute Capability 6.0+ for optimal performance

## Quick Start

### Basic Usage

```python
import asyncio
from components.ai_vision.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceConfig,
    DeviceType,
    PrecisionType,
    OptimizationLevel
)

async def main():
    # Create configuration
    config = PerformanceConfig(
        device=DeviceType.AUTO,          # Auto-detect best device
        precision=PrecisionType.AUTO,    # Auto-select optimal precision
        optimization_level=OptimizationLevel.AGGRESSIVE,
        batch_size=8,                    # Process 8 frames per batch
        async_processing=True            # Enable async processing
    )
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(config)
    await optimizer.initialize()
    
    # Your model (example)
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 80)
    )
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, "my_model")
    
    # Process frames (example)
    import numpy as np
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Submit frame for async processing
    frame_id = await optimizer.process_frame_async(
        frame, 
        "my_model", 
        lambda output, frame, meta: {"prediction": output.cpu().numpy()}
    )
    
    # Process batches
    result = await optimizer.process_batch_async(lambda output, frame, meta: {
        "prediction": output.cpu().numpy()
    })
    
    if result and not result.error:
        print(f"Processed {len(result.results)} frames in {result.processing_time:.4f}s")
        print(f"Device used: {result.device_used}")
        print(f"Precision used: {result.precision_used}")
    
    # Get performance stats
    stats = optimizer.get_performance_stats()
    print(f"Current device: {stats['device_info']['device']}")
    print(f"GPU utilization: {stats.get('gpu_stats', {}).get('load', 0):.1f}%")
    
    # Cleanup
    await optimizer.shutdown()

# Run the example
asyncio.run(main())
```

### Integration with Camera Streams

```python
from example_performance_optimization_integration import OptimizedVideoProcessor

async def camera_example():
    processor = OptimizedVideoProcessor()
    await processor.initialize()
    
    # Process live camera for 30 seconds
    await processor.process_camera_stream(camera_id=0, duration_seconds=30)
    
    # Process video file
    await processor.process_video_file("input_video.mp4")
    
    await processor.shutdown()

asyncio.run(camera_example())
```

## Configuration Options

### Performance Configuration

```python
config = PerformanceConfig(
    device=DeviceType.AUTO,              # AUTO, CUDA, CPU, MPS
    precision=PrecisionType.AUTO,        # AUTO, FULL, HALF, MIXED
    optimization_level=OptimizationLevel.AGGRESSIVE,  # NONE, BASIC, AGGRESSIVE, MAXIMUM
    batch_size=8,                        # Frames per batch (1-32)
    max_queue_size=100,                  # Maximum async queue size
    max_workers=4,                       # Number of worker threads
    memory_fraction=0.8,                 # GPU memory fraction (0.1-1.0)
    async_processing=True,               # Enable async processing
    enable_profiling=True,               # Enable performance profiling
    enable_tensorrt=False,               # Enable TensorRT optimization (NVIDIA only)
    enable_onnx=False                    # Enable ONNX optimization
)
```

### Configuration File (JSON)

```json
{
  "device": "AUTO",
  "precision": "AUTO",
  "optimization_level": "AGGRESSIVE",
  "batch_size": 8,
  "max_queue_size": 100,
  "async_processing": true,
  "enable_profiling": true
}
```

## Benchmarking

### Run Comprehensive Benchmarks

```bash
# Full benchmark suite
python benchmark_performance_optimization.py

# Quick benchmark
python benchmark_performance_optimization.py --quick

# Specific scenarios
python benchmark_performance_optimization.py --scenarios baseline_cpu optimized_gpu

# Custom configuration
python benchmark_performance_optimization.py --config benchmark_config.json --output results.json
```

### Benchmark Scenarios

The benchmarking system tests multiple optimization scenarios:

1. **baseline_cpu**: Unoptimized CPU processing
2. **optimized_cpu**: CPU with aggressive optimizations
3. **baseline_gpu**: Basic GPU processing
4. **optimized_gpu**: GPU with half-precision and batching
5. **maximum_optimization**: All optimizations enabled

### Sample Benchmark Results

```
BENCHMARK RESULTS SUMMARY
============================================================
Scenarios completed: 5
Scenarios failed: 0

Throughput Comparison:
  baseline_cpu              :    12.45 FPS (1.00x)
  optimized_cpu             :    18.23 FPS (1.46x)
  baseline_gpu              :    45.67 FPS (3.67x)
  optimized_gpu             :    89.34 FPS (7.18x)
  maximum_optimization      :   123.56 FPS (9.93x)

Best performing scenario: maximum_optimization (123.56 FPS)
```

## Performance Tips

### GPU Optimization

1. **Use Appropriate Batch Sizes**: 
   - Start with batch_size=8 for most GPUs
   - Increase for high-end GPUs (16-32)
   - Reduce for limited VRAM (2-4)

2. **Enable Half-Precision**:
   ```python
   config.precision = PrecisionType.HALF  # Or AUTO for automatic detection
   ```

3. **Memory Management**:
   ```python
   config.memory_fraction = 0.8  # Reserve 80% of GPU memory
   ```

### CPU Optimization

1. **Worker Threads**:
   ```python
   config.max_workers = 4  # Match CPU cores
   ```

2. **Batch Processing**:
   ```python
   config.batch_size = 4  # Smaller batches for CPU
   ```

### General Optimization

1. **Async Processing**: Always enable for real-time applications
2. **Profiling**: Enable during development, disable in production
3. **Queue Size**: Increase for bursty workloads, reduce for memory-limited systems

## Architecture Overview

### Component Structure

```
PerformanceOptimizer
├── DeviceManager          # Device detection and configuration
├── PrecisionManager       # Precision optimization
├── ModelOptimizer        # Model compilation and optimization
├── AsyncFrameQueue       # Asynchronous frame processing
├── PerformanceProfiler   # Performance monitoring
└── BatchProcessor        # Batch aggregation and processing
```

### Processing Pipeline

```
Input Frame → AsyncFrameQueue → BatchAggregator → ModelInference → ResultProcessing → Output
     ↓              ↓                 ↓               ↓              ↓
DeviceDetection → Preprocessing → BatchForming → GPUProcessing → PostProcessing
```

## Advanced Features

### Custom Inference Functions

```python
def custom_inference(model_output, frame, metadata):
    """Custom processing for model outputs"""
    # Your custom logic here
    predictions = torch.softmax(model_output, dim=-1)
    top_classes = torch.topk(predictions, k=5)
    
    return {
        'predictions': predictions.cpu().numpy(),
        'top_5_classes': top_classes.indices.cpu().numpy(),
        'confidences': top_classes.values.cpu().numpy(),
        'processing_time': time.time() - metadata.get('timestamp', 0)
    }

# Use with optimizer
frame_id = await optimizer.process_frame_async(frame, "model", custom_inference)
```

### Performance Monitoring

```python
# Get real-time performance statistics
stats = optimizer.get_performance_stats()

print(f"Device: {stats['device_info']['name']}")
print(f"GPU Memory: {stats['gpu_stats']['memory_percent']:.1f}%")
print(f"CPU Usage: {stats['cpu_stats']['usage_percent']:.1f}%")
print(f"Precision: {stats['precision']}")

# Access profiler metrics
if optimizer.config.enable_profiling:
    metrics = stats['profiler_metrics']
    for operation, timing in metrics.items():
        if 'duration' in timing:
            print(f"{operation}: {timing['duration']:.4f}s")
```

### Model Optimization Levels

#### BASIC (Default)
- Model evaluation mode
- Gradient computation disabled
- JIT scripting (if possible)

#### AGGRESSIVE
- All BASIC optimizations
- torch.compile (PyTorch 2.0+)
- Memory optimization

#### MAXIMUM
- All AGGRESSIVE optimizations
- TensorRT compilation (NVIDIA GPUs)
- ONNX optimization
- Maximum compiler optimizations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   config.memory_fraction = 0.6  # Reduce memory usage
   config.batch_size = 4         # Smaller batches
   ```

2. **Slow CPU Performance**:
   ```python
   config.max_workers = mp.cpu_count()  # Use all CPU cores
   config.optimization_level = OptimizationLevel.AGGRESSIVE
   ```

3. **High Memory Usage**:
   ```python
   config.max_queue_size = 50    # Reduce queue size
   config.batch_size = 4         # Smaller batches
   ```

4. **Import Errors**:
   ```bash
   # Reinstall dependencies
   python setup_performance_optimization.py --test
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed profiling
config.enable_profiling = True

# Check system compatibility
python setup_performance_optimization.py --test --skip-install
```

## Examples

See the following example files:

- `example_performance_optimization_integration.py` - Complete integration example
- `benchmark_performance_optimization.py` - Benchmarking suite
- `setup_performance_optimization.py` - Setup and testing script

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the benchmark suite
5. Submit a pull request

## License

This performance optimization system is provided under the MIT License. See the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Run the diagnostic script: `python setup_performance_optimization.py --test`
3. Review the benchmark results for performance comparisons
4. Enable debug logging for detailed error information

---

**Performance Optimization System v1.0** - Accelerating AI vision workloads with intelligent optimization
