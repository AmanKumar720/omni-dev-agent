#!/usr/bin/env python3
# src/components/ai_vision/model_hub_example.py

"""
Example usage of the ModelHub for managing AI vision models.

This script demonstrates how to:
1. Initialize the model hub
2. List available models
3. Download models with progress tracking
4. Load models with device selection
5. Use models for inference
6. Manage cache and cleanup
"""

import asyncio
import logging
from pathlib import Path
from model_hub import (
    ModelHub, 
    ModelType, 
    DeviceType, 
    ModelMetadata,
    get_model_hub
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def progress_callback(progress: float):
    """Progress callback for model downloads"""
    print(f"Download progress: {progress:.1%}")


async def demo_basic_usage():
    """Demonstrate basic model hub usage"""
    logger.info("=== Basic Model Hub Demo ===")
    
    # Initialize model hub (uses ~/.omni_dev/models by default)
    hub = get_model_hub()
    
    # List available models
    logger.info("Available models:")
    models = hub.list_available_models()
    for model in models[:3]:  # Show first 3 models
        print(f"  - {model['name']} ({model['type']}) - {model['description']}")
        print(f"    Version: {model['version']}, Size: {model['size_mb']:.1f}MB")
        print(f"    Status: {model['status']}")
        print()
    
    # Get device information
    device_info = hub.get_device_info()
    logger.info(f"Device info: {device_info}")
    
    return hub


async def demo_model_download_and_load():
    """Demonstrate downloading and loading models"""
    logger.info("=== Model Download and Load Demo ===")
    
    hub = get_model_hub()
    
    # Choose a lightweight model for demo
    model_name = "yolov8n"  # YOLOv8 nano - smallest version
    
    # Check model status
    status = hub.get_model_status(model_name)
    logger.info(f"Model {model_name} status: {status}")
    
    # Download model if not already downloaded
    if status == "not_downloaded":
        logger.info(f"Downloading {model_name}...")
        success = await hub.download_model(model_name, progress_callback)
        if success:
            logger.info(f"Successfully downloaded {model_name}")
        else:
            logger.error(f"Failed to download {model_name}")
            return None
    
    # Load model (with automatic device selection)
    logger.info(f"Loading {model_name}...")
    model = hub.load_model(model_name, DeviceType.AUTO)
    
    if model:
        logger.info(f"Successfully loaded {model_name}")
        logger.info(f"Model type: {type(model)}")
        
        # Get model info
        model_info = hub.get_model_info(model_name)
        logger.info(f"Model info: {model_info}")
        
        return model
    else:
        logger.error(f"Failed to load {model_name}")
        return None


async def demo_model_management():
    """Demonstrate model management features"""
    logger.info("=== Model Management Demo ===")
    
    hub = get_model_hub()
    
    # Get cache information
    cache_info = hub.get_cache_info()
    logger.info("Cache information:")
    print(f"  Total models: {cache_info['total_models']}")
    print(f"  Total size: {cache_info['total_size_gb']:.2f} GB")
    print(f"  Max size: {cache_info['max_size_gb']:.2f} GB")
    print(f"  Utilization: {cache_info['utilization']:.1%}")
    
    # List models by type
    logger.info("YOLOv8 models:")
    yolo_models = hub.list_available_models(ModelType.YOLO_V8)
    for model in yolo_models:
        print(f"  - {model['name']}: {model['status']}")
    
    # Validate models (check integrity)
    logger.info("Validating cached models...")
    for model_name in ["yolov8n", "yolov8s"]:
        if hub.get_model_status(model_name) != "not_downloaded":
            is_valid = hub.validate_model(model_name)
            logger.info(f"Model {model_name} validation: {'PASSED' if is_valid else 'FAILED'}")


async def demo_ensure_model_ready():
    """Demonstrate the ensure_model_ready convenience method"""
    logger.info("=== Ensure Model Ready Demo ===")
    
    hub = get_model_hub()
    
    # This method will download if needed, then load
    model_name = "mobilenet_v2"
    logger.info(f"Ensuring {model_name} is ready...")
    
    model = await hub.ensure_model_ready(model_name, DeviceType.CPU)
    
    if model:
        logger.info(f"Model {model_name} is ready for use!")
        logger.info(f"Model type: {type(model)}")
    else:
        logger.error(f"Failed to ensure {model_name} is ready")


def demo_custom_model_registration():
    """Demonstrate registering a custom model"""
    logger.info("=== Custom Model Registration Demo ===")
    
    hub = get_model_hub()
    
    # Create custom model metadata
    custom_metadata = ModelMetadata(
        name="my_custom_yolo",
        model_type=ModelType.YOLO_V8,
        version="1.0.0",
        size_mb=15.5,
        checksum_sha256="abc123def456789",  # This would be the actual checksum
        download_url="https://my-server.com/models/my_custom_yolo.pt",
        description="Custom trained YOLOv8 model for specific use case",
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        dependencies=["ultralytics", "torch"],
        last_updated=datetime.now(),
        license="MIT",
        tags=["custom", "specialized"]
    )
    
    # Register the model
    success = hub.register_custom_model(custom_metadata)
    logger.info(f"Custom model registration: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        # List models to show it was added
        custom_models = [m for m in hub.list_available_models() if m['name'] == 'my_custom_yolo']
        logger.info(f"Registered custom model: {custom_models[0] if custom_models else 'Not found'}")


async def demo_cache_cleanup():
    """Demonstrate cache cleanup functionality"""
    logger.info("=== Cache Cleanup Demo ===")
    
    hub = get_model_hub()
    
    # Get cache stats before cleanup
    cache_before = hub.get_cache_info()
    logger.info(f"Cache before cleanup: {cache_before['total_models']} models, "
                f"{cache_before['total_size_gb']:.2f} GB")
    
    # Cleanup cache (removes invalid/corrupted models)
    cleanup_stats = hub.cleanup_cache()
    logger.info(f"Cleanup results: Removed {cleanup_stats['removed_count']} models, "
                f"Reclaimed {cleanup_stats['reclaimed_bytes'] / (1024**2):.1f} MB")
    
    # Get cache stats after cleanup
    cache_after = hub.get_cache_info()
    logger.info(f"Cache after cleanup: {cache_after['total_models']} models, "
                f"{cache_after['total_size_gb']:.2f} GB")


async def demo_memory_management():
    """Demonstrate model memory management"""
    logger.info("=== Memory Management Demo ===")
    
    hub = get_model_hub()
    
    # Load a model
    model_name = "yolov8n"
    model = await hub.ensure_model_ready(model_name, DeviceType.AUTO)
    
    if model:
        logger.info(f"Model {model_name} loaded into memory")
        
        # Check loaded models
        loaded_models = hub.loader.get_loaded_models()
        logger.info(f"Currently loaded models: {loaded_models}")
        
        # Unload the model
        success = hub.unload_model(model_name, DeviceType.AUTO)
        logger.info(f"Model unload: {'SUCCESS' if success else 'FAILED'}")
        
        # Check loaded models again
        loaded_models = hub.loader.get_loaded_models()
        logger.info(f"Currently loaded models after unload: {loaded_models}")


async def run_full_demo():
    """Run the complete demonstration"""
    logger.info("Starting ModelHub demonstration...")
    
    try:
        # Basic usage
        await demo_basic_usage()
        await asyncio.sleep(1)
        
        # Download and load
        await demo_model_download_and_load()
        await asyncio.sleep(1)
        
        # Model management
        await demo_model_management()
        await asyncio.sleep(1)
        
        # Ensure model ready
        await demo_ensure_model_ready()
        await asyncio.sleep(1)
        
        # Custom model registration
        demo_custom_model_registration()
        await asyncio.sleep(1)
        
        # Memory management
        await demo_memory_management()
        await asyncio.sleep(1)
        
        # Cache cleanup
        await demo_cache_cleanup()
        
        logger.info("ModelHub demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    # Fix datetime import for custom model demo
    from datetime import datetime
    
    # Run the demonstration
    asyncio.run(run_full_demo())
