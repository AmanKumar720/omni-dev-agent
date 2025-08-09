#!/usr/bin/env python3
# src/components/ai_vision/test_model_hub_basic.py

"""
Basic test for model hub functionality without heavy dependencies.
This tests the core logic and structure without requiring ML libraries.
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# Temporarily mock the missing dependencies to test the structure
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()
    
    def __gt__(self, other):
        return False
    
    def __lt__(self, other):
        return False
    
    def __eq__(self, other):
        return True
    
    def __bool__(self):
        return False
    
    def __int__(self):
        return 0

# Mock heavy dependencies
sys.modules['aiohttp'] = MockModule()
sys.modules['torch'] = MockModule()
sys.modules['ultralytics'] = MockModule()
sys.modules['tensorflow'] = MockModule()
sys.modules['cv2'] = MockModule()
sys.modules['face_recognition'] = MockModule()
sys.modules['pytesseract'] = MockModule()

# Now import our model hub
try:
    from model_hub import (
        ModelHub, ModelRegistry, DeviceManager, CacheManager,
        ChecksumValidator, ModelType, DeviceType, ModelStatus,
        ModelMetadata, get_model_hub
    )
    print("✅ Successfully imported model hub components")
except ImportError as e:
    print(f"❌ Failed to import model hub: {e}")
    sys.exit(1)

def test_model_registry():
    """Test the model registry functionality"""
    print("Testing ModelRegistry...")
    
    registry = ModelRegistry()
    
    # Test getting existing models
    yolo_model = registry.get_model("yolov8n")
    assert yolo_model is not None, "Should find yolov8n model"
    assert yolo_model.name == "yolov8n", "Model name should match"
    assert yolo_model.model_type == ModelType.YOLO_V8, "Should be YOLO type"
    
    # Test getting non-existent model
    missing_model = registry.get_model("nonexistent")
    assert missing_model is None, "Should return None for missing model"
    
    # Test listing models
    all_models = registry.list_models()
    assert len(all_models) > 0, "Should have some models"
    
    # Test listing by type
    yolo_models = registry.list_models(ModelType.YOLO_V8)
    assert len(yolo_models) >= 2, "Should have at least 2 YOLO models"
    
    # Test custom model registration
    custom_metadata = ModelMetadata(
        name="test_model",
        model_type=ModelType.RESNET,
        version="1.0.0",
        size_mb=50.0,
        checksum_sha256="abc123",
        download_url="https://example.com",
        description="Test model",
        supported_devices=[DeviceType.CPU],
        dependencies=["torch"],
        last_updated=datetime.now()
    )
    
    success = registry.register_model(custom_metadata)
    assert success, "Should register custom model successfully"
    
    retrieved = registry.get_model("test_model")
    assert retrieved is not None, "Should retrieve registered model"
    assert retrieved.name == "test_model", "Retrieved model should match"
    
    print("✅ ModelRegistry tests passed")

def test_device_manager():
    """Test device management functionality"""
    print("Testing DeviceManager...")
    
    device_manager = DeviceManager()
    
    # Test CPU device selection
    cpu_device = device_manager.get_optimal_device(DeviceType.CPU)
    assert cpu_device == "cpu", "CPU device should return 'cpu'"
    
    # Test device info
    device_info = device_manager.get_device_info()
    assert "cpu_available" in device_info, "Should include cpu_available"
    assert "gpu_available" in device_info, "Should include gpu_available"
    assert "recommended_device" in device_info, "Should include recommended_device"
    assert device_info["cpu_available"] == True, "CPU should always be available"
    
    print("✅ DeviceManager tests passed")

def test_checksum_validator():
    """Test checksum validation"""
    print("Testing ChecksumValidator...")
    
    # Create temporary file
    temp_file = Path(tempfile.mktemp())
    try:
        temp_file.write_text("test content")
        
        # Calculate checksum
        checksum = ChecksumValidator.calculate_sha256(temp_file)
        assert isinstance(checksum, str), "Checksum should be string"
        assert len(checksum) == 64, "SHA256 should be 64 characters"
        
        # Validate correct checksum
        assert ChecksumValidator.validate_checksum(temp_file, checksum), "Should validate correct checksum"
        
        # Validate incorrect checksum
        wrong_checksum = "a" * 64
        assert not ChecksumValidator.validate_checksum(temp_file, wrong_checksum), "Should reject wrong checksum"
        
    finally:
        if temp_file.exists():
            temp_file.unlink()
    
    print("✅ ChecksumValidator tests passed")

def test_cache_manager():
    """Test cache management"""
    print("Testing CacheManager...")
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        cache_manager = CacheManager(temp_dir, max_cache_size_gb=1.0)
        
        # Test cache directory creation
        assert temp_dir.exists(), "Cache directory should exist"
        
        # Create dummy model file
        model_path = temp_dir / "test_model"
        model_path.write_text("dummy model data")
        
        # Create metadata
        metadata = ModelMetadata(
            name="cache_test_model",
            model_type=ModelType.RESNET,
            version="1.0.0",
            size_mb=1.0,
            checksum_sha256="def456",
            download_url="https://example.com",
            description="Cache test",
            supported_devices=[DeviceType.CPU],
            dependencies=[],
            last_updated=datetime.now()
        )
        
        # Add to cache
        cache_manager.add_model("cache_test_model", model_path, metadata)
        
        # Retrieve from cache
        entry = cache_manager.get_model("cache_test_model", "1.0.0")
        assert entry is not None, "Should retrieve cached model"
        assert entry.metadata.name == "cache_test_model", "Model name should match"
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        assert "total_models" in stats, "Stats should include total_models"
        assert stats["total_models"] >= 1, "Should have at least 1 model"
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("✅ CacheManager tests passed")

def test_model_hub():
    """Test main ModelHub functionality"""
    print("Testing ModelHub...")
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        hub = ModelHub(cache_dir=temp_dir, max_cache_size_gb=1.0)
        
        # Test initialization
        assert hub.cache_dir == temp_dir, "Cache directory should match"
        assert hub.registry is not None, "Should have registry"
        assert hub.device_manager is not None, "Should have device manager"
        
        # Test listing available models
        models = hub.list_available_models()
        assert len(models) > 0, "Should have available models"
        
        # Check model structure
        for model in models[:3]:
            assert "name" in model, "Model should have name"
            assert "type" in model, "Model should have type"
            assert "version" in model, "Model should have version"
            assert "status" in model, "Model should have status"
        
        # Test model status
        status = hub.get_model_status("yolov8n")
        assert status in [s.value for s in ModelStatus], "Status should be valid ModelStatus"
        
        # Test device info
        device_info = hub.get_device_info()
        assert isinstance(device_info, dict), "Device info should be dict"
        
        # Test cache info
        cache_info = hub.get_cache_info()
        assert isinstance(cache_info, dict), "Cache info should be dict"
        assert "total_models" in cache_info, "Should include total_models"
        
        # Test model info
        model_info = hub.get_model_info("yolov8n")
        assert model_info is not None, "Should get model info"
        assert model_info["name"] == "yolov8n", "Model name should match"
        
        # Test custom model registration
        custom_metadata = ModelMetadata(
            name="hub_test_model",
            model_type=ModelType.MOBILENET,
            version="1.0.0",
            size_mb=25.0,
            checksum_sha256="ghi789",
            download_url="https://example.com",
            description="Hub test model",
            supported_devices=[DeviceType.CPU, DeviceType.GPU],
            dependencies=["torch"],
            last_updated=datetime.now()
        )
        
        success = hub.register_custom_model(custom_metadata)
        assert success, "Should register custom model"
        
        # Verify it appears in listings
        all_models = hub.list_available_models()
        model_names = [m["name"] for m in all_models]
        assert "hub_test_model" in model_names, "Custom model should appear in listings"
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("✅ ModelHub tests passed")

def test_singleton():
    """Test singleton functionality"""
    print("Testing singleton...")
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Clear global instance first
        import model_hub
        model_hub._global_model_hub = None
        
        hub1 = get_model_hub(temp_dir)
        hub2 = get_model_hub(temp_dir)
        
        assert hub1 is hub2, "Should return same instance (singleton)"
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("✅ Singleton tests passed")

def main():
    """Run all tests"""
    print("🚀 Starting Model Hub Basic Tests...")
    print("=" * 50)
    
    try:
        test_model_registry()
        test_device_manager()
        test_checksum_validator()
        test_cache_manager()
        test_model_hub()
        test_singleton()
        
        print("=" * 50)
        print("🎉 All tests passed! Model Hub implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
