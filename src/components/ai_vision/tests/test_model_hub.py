# src/components/ai_vision/tests/test_model_hub.py

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_hub import (
    ModelHub, ModelRegistry, DeviceManager, CacheManager, 
    ModelDownloader, LazyModelLoader, ChecksumValidator,
    ModelType, DeviceType, ModelStatus, ModelMetadata
)


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry"""
    
    def setUp(self):
        self.registry = ModelRegistry()
    
    def test_get_model_existing(self):
        """Test getting an existing model"""
        model = self.registry.get_model("yolov8n")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "yolov8n")
        self.assertEqual(model.model_type, ModelType.YOLO_V8)
    
    def test_get_model_nonexistent(self):
        """Test getting a non-existent model"""
        model = self.registry.get_model("nonexistent_model")
        self.assertIsNone(model)
    
    def test_list_models(self):
        """Test listing all models"""
        models = self.registry.list_models()
        self.assertGreater(len(models), 0)
        
        # Check that all models have required attributes
        for model in models:
            self.assertIsInstance(model.name, str)
            self.assertIsInstance(model.model_type, ModelType)
            self.assertIsInstance(model.version, str)
    
    def test_list_models_by_type(self):
        """Test listing models by type"""
        yolo_models = self.registry.list_models(ModelType.YOLO_V8)
        
        # Should have at least 2 YOLO models (yolov8n, yolov8s)
        self.assertGreaterEqual(len(yolo_models), 2)
        
        # All should be YOLO models
        for model in yolo_models:
            self.assertEqual(model.model_type, ModelType.YOLO_V8)
    
    def test_register_custom_model(self):
        """Test registering a custom model"""
        custom_metadata = ModelMetadata(
            name="test_model",
            model_type=ModelType.RESNET,
            version="1.0.0",
            size_mb=50.0,
            checksum_sha256="abc123",
            download_url="https://example.com/model.pth",
            description="Test model",
            supported_devices=[DeviceType.CPU],
            dependencies=["torch"],
            last_updated=datetime.now()
        )
        
        success = self.registry.register_model(custom_metadata)
        self.assertTrue(success)
        
        # Verify it can be retrieved
        retrieved = self.registry.get_model("test_model")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test_model")


class TestDeviceManager(unittest.TestCase):
    """Test cases for DeviceManager"""
    
    def setUp(self):
        self.device_manager = DeviceManager()
    
    def test_get_optimal_device_cpu(self):
        """Test getting CPU device"""
        device = self.device_manager.get_optimal_device(DeviceType.CPU)
        self.assertEqual(device, "cpu")
    
    @patch('model_hub._torch')
    def test_get_optimal_device_auto_no_cuda(self, mock_torch):
        """Test auto device selection when CUDA is not available"""
        mock_torch.cuda.is_available.return_value = False
        
        device = self.device_manager.get_optimal_device(DeviceType.AUTO)
        self.assertEqual(device, "cpu")
    
    def test_get_device_info(self):
        """Test getting device information"""
        info = self.device_manager.get_device_info()
        
        self.assertIn("cpu_available", info)
        self.assertIn("gpu_available", info)
        self.assertIn("cuda_available", info)
        self.assertIn("recommended_device", info)
        
        self.assertTrue(info["cpu_available"])


class TestChecksumValidator(unittest.TestCase):
    """Test cases for ChecksumValidator"""
    
    def test_calculate_and_validate_checksum(self):
        """Test checksum calculation and validation"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = Path(temp_file.name)
        
        try:
            # Calculate checksum
            checksum = ChecksumValidator.calculate_sha256(temp_path)
            self.assertIsInstance(checksum, str)
            self.assertEqual(len(checksum), 64)  # SHA256 is 64 hex characters
            
            # Validate with correct checksum
            self.assertTrue(ChecksumValidator.validate_checksum(temp_path, checksum))
            
            # Validate with incorrect checksum
            wrong_checksum = "a" * 64
            self.assertFalse(ChecksumValidator.validate_checksum(temp_path, wrong_checksum))
            
        finally:
            temp_path.unlink()


class TestCacheManager(unittest.TestCase):
    """Test cases for CacheManager"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_manager = CacheManager(self.temp_dir, max_cache_size_gb=1.0)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created"""
        self.assertTrue(self.temp_dir.exists())
        self.assertTrue(self.temp_dir.is_dir())
    
    def test_add_and_get_model(self):
        """Test adding and retrieving a model from cache"""
        # Create a dummy model file
        model_path = self.temp_dir / "test_model"
        model_path.write_text("dummy model data")
        
        # Create metadata
        metadata = ModelMetadata(
            name="test_model",
            model_type=ModelType.RESNET,
            version="1.0.0",
            size_mb=1.0,
            checksum_sha256="abc123",
            download_url="https://example.com",
            description="Test",
            supported_devices=[DeviceType.CPU],
            dependencies=[],
            last_updated=datetime.now()
        )
        
        # Add to cache
        self.cache_manager.add_model("test_model", model_path, metadata)
        
        # Retrieve from cache
        entry = self.cache_manager.get_model("test_model", "1.0.0")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.metadata.name, "test_model")
        self.assertEqual(entry.access_count, 1)
    
    def test_cache_stats(self):
        """Test getting cache statistics"""
        stats = self.cache_manager.get_cache_stats()
        
        self.assertIn("total_models", stats)
        self.assertIn("total_size_bytes", stats)
        self.assertIn("total_size_gb", stats)
        self.assertIn("max_size_gb", stats)
        self.assertIn("utilization", stats)
        self.assertIn("models", stats)


class TestModelHub(unittest.TestCase):
    """Test cases for ModelHub"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.hub = ModelHub(cache_dir=self.temp_dir, max_cache_size_gb=1.0)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelHub initialization"""
        self.assertEqual(self.hub.cache_dir, self.temp_dir)
        self.assertIsNotNone(self.hub.registry)
        self.assertIsNotNone(self.hub.device_manager)
        self.assertIsNotNone(self.hub.cache_manager)
        self.assertIsNotNone(self.hub.downloader)
        self.assertIsNotNone(self.hub.loader)
    
    def test_list_available_models(self):
        """Test listing available models"""
        models = self.hub.list_available_models()
        self.assertGreater(len(models), 0)
        
        # Check structure of returned model info
        for model in models:
            self.assertIn("name", model)
            self.assertIn("type", model)
            self.assertIn("version", model)
            self.assertIn("size_mb", model)
            self.assertIn("description", model)
            self.assertIn("status", model)
    
    def test_get_model_status(self):
        """Test getting model status"""
        status = self.hub.get_model_status("yolov8n")
        self.assertIn(status, [s.value for s in ModelStatus])
    
    def test_get_cache_info(self):
        """Test getting cache information"""
        info = self.hub.get_cache_info()
        self.assertIsInstance(info, dict)
        self.assertIn("total_models", info)
    
    def test_get_device_info(self):
        """Test getting device information"""
        info = self.hub.get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn("cpu_available", info)
    
    def test_get_model_info(self):
        """Test getting detailed model information"""
        info = self.hub.get_model_info("yolov8n")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "yolov8n")
        self.assertEqual(info["type"], "yolo_v8")
        self.assertIn("version", info)
        self.assertIn("description", info)
        self.assertIn("status", info)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_model_hub_singleton(self):
        """Test that get_model_hub returns singleton"""
        from model_hub import get_model_hub, _global_model_hub
        
        # Clear global instance
        import model_hub
        model_hub._global_model_hub = None
        
        hub1 = get_model_hub(self.temp_dir)
        hub2 = get_model_hub(self.temp_dir)
        
        self.assertIs(hub1, hub2)
    
    def test_model_registration_and_retrieval(self):
        """Test the complete flow of registering and retrieving a model"""
        hub = ModelHub(cache_dir=self.temp_dir)
        
        # Create and register custom model
        metadata = ModelMetadata(
            name="integration_test_model",
            model_type=ModelType.MOBILENET,
            version="1.0.0",
            size_mb=25.0,
            checksum_sha256="def456",
            download_url="https://example.com/model.pth",
            description="Integration test model",
            supported_devices=[DeviceType.CPU, DeviceType.GPU],
            dependencies=["torch"],
            last_updated=datetime.now()
        )
        
        # Register
        success = hub.register_custom_model(metadata)
        self.assertTrue(success)
        
        # Verify it appears in listings
        all_models = hub.list_available_models()
        model_names = [m["name"] for m in all_models]
        self.assertIn("integration_test_model", model_names)
        
        # Get detailed info
        info = hub.get_model_info("integration_test_model")
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "integration_test_model")


if __name__ == '__main__':
    # Run the tests
    unittest.main()
