#!/usr/bin/env python3
# src/components/ai_vision/test_object_detection_basic.py

"""
Basic test script for YOLOv8 Object Detection Module

This script performs basic functionality tests without requiring actual model downloads
or heavy dependencies. Use this for quick validation of the module structure.
"""

import logging
import sys
import traceback
import numpy as np
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test if the object detection module can be imported"""
    logger.info("Testing imports...")
    
    try:
        # Test core imports
        from object_detection import (
            ObjectDetector, 
            VideoStreamDetector, 
            ObjectDetectionTask,
            DetectionResult,
            BatchDetectionResult,
            detect_objects,
            detect_objects_batch,
            create_detector
        )
        logger.info("✓ All imports successful")
        return True
    
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_data_structures():
    """Test data structure creation and conversion"""
    logger.info("Testing data structures...")
    
    try:
        from object_detection import DetectionResult, BatchDetectionResult
        
        # Test DetectionResult
        detection = DetectionResult(
            bbox=(100, 200, 300, 400),
            confidence=0.85,
            class_id=1,
            class_name="person"
        )
        
        detection_dict = detection.to_dict()
        expected_keys = {'bbox', 'confidence', 'class_id', 'class_name'}
        if set(detection_dict.keys()) != expected_keys:
            raise ValueError(f"DetectionResult.to_dict() missing keys: {expected_keys - set(detection_dict.keys())}")
        
        # Test BatchDetectionResult
        batch_result = BatchDetectionResult(
            detections=[[detection], [detection, detection]],
            processing_time=1.23,
            batch_size=2
        )
        
        batch_dict = batch_result.to_dict()
        expected_keys = {'detections', 'processing_time', 'batch_size'}
        if set(batch_dict.keys()) != expected_keys:
            raise ValueError(f"BatchDetectionResult.to_dict() missing keys: {expected_keys - set(batch_dict.keys())}")
        
        logger.info("✓ Data structures working correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ Data structure test failed: {e}")
        traceback.print_exc()
        return False


def test_object_detector_creation():
    """Test ObjectDetector class instantiation"""
    logger.info("Testing ObjectDetector creation...")
    
    try:
        from object_detection import ObjectDetector
        
        # Test with different parameters
        detector1 = ObjectDetector()  # Default parameters
        detector2 = ObjectDetector(model_name="yolov8s", device="cpu")
        detector3 = ObjectDetector(model_name="yolov8n", device="auto")
        
        # Check attributes
        assert detector1.model_name == "yolov8n"
        assert detector2.model_name == "yolov8s"
        assert detector3.device_type.value in ["auto", "cpu", "cuda"]
        
        # Check class names are loaded
        assert len(detector1.class_names) == 80
        assert "person" in detector1.class_names
        assert "car" in detector1.class_names
        
        logger.info("✓ ObjectDetector creation successful")
        return True
    
    except Exception as e:
        logger.error(f"✗ ObjectDetector creation failed: {e}")
        traceback.print_exc()
        return False


def test_video_stream_detector_creation():
    """Test VideoStreamDetector class instantiation"""
    logger.info("Testing VideoStreamDetector creation...")
    
    try:
        from object_detection import ObjectDetector, VideoStreamDetector
        
        # Create mock detector
        detector = ObjectDetector()
        
        # Create stream detector
        stream_detector = VideoStreamDetector(detector, source=0)
        
        # Check attributes
        assert stream_detector.detector == detector
        assert stream_detector.source == 0
        assert not stream_detector.is_streaming
        
        logger.info("✓ VideoStreamDetector creation successful")
        return True
    
    except Exception as e:
        logger.error(f"✗ VideoStreamDetector creation failed: {e}")
        traceback.print_exc()
        return False


def test_vision_task_creation():
    """Test ObjectDetectionTask creation"""
    logger.info("Testing ObjectDetectionTask creation...")
    
    try:
        from object_detection import ObjectDetectionTask
        
        # Create task
        task = ObjectDetectionTask("test_task_001", model_name="yolov8n")
        
        # Check attributes
        assert task.task_id == "test_task_001"
        assert task.task_type == "object_detection"
        assert task.model_name == "yolov8n"
        
        # Test input validation
        valid_input = {
            'frame': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'conf_threshold': 0.5
        }
        assert task.validate_input(valid_input)
        
        invalid_input = {'invalid_key': 'invalid_value'}
        assert not task.validate_input(invalid_input)
        
        logger.info("✓ ObjectDetectionTask creation successful")
        return True
    
    except Exception as e:
        logger.error(f"✗ ObjectDetectionTask creation failed: {e}")
        traceback.print_exc()
        return False


def test_lazy_imports():
    """Test that lazy imports don't break the module loading"""
    logger.info("Testing lazy imports...")
    
    try:
        from object_detection import _lazy_import_cv2, _lazy_import_torch, _lazy_import_ultralytics
        
        # These should raise ImportError if packages aren't installed
        # but shouldn't break the module import
        try:
            _lazy_import_cv2()
            logger.info("  OpenCV available")
        except ImportError:
            logger.info("  OpenCV not available (expected in test environment)")
        
        try:
            _lazy_import_torch()
            logger.info("  PyTorch available")
        except ImportError:
            logger.info("  PyTorch not available (expected in test environment)")
        
        try:
            _lazy_import_ultralytics()
            logger.info("  Ultralytics available")
        except ImportError:
            logger.info("  Ultralytics not available (expected in test environment)")
        
        logger.info("✓ Lazy imports working correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ Lazy import test failed: {e}")
        traceback.print_exc()
        return False


def test_convenience_functions():
    """Test convenience function signatures"""
    logger.info("Testing convenience functions...")
    
    try:
        from object_detection import detect_objects, detect_objects_batch
        
        # Create mock data
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_frames = [mock_frame, mock_frame]
        
        # Test that functions exist and can be called with None model
        # (should raise ValueError as expected)
        try:
            detect_objects(mock_frame, model=None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be provided" in str(e)
        
        try:
            detect_objects_batch(mock_frames, model=None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be provided" in str(e)
        
        logger.info("✓ Convenience functions working correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ Convenience function test failed: {e}")
        traceback.print_exc()
        return False


def test_model_integration():
    """Test integration with model registry"""
    logger.info("Testing model registry integration...")
    
    try:
        # Test model registry
        from models import MODEL_REGISTRY, register_model, get_model, list_models
        from object_detection import ObjectDetector
        
        # Check if yolov8 model is registered
        models = list_models()
        assert 'yolov8_object_detection' in models
        assert 'object_detection' in models['yolov8_object_detection']
        
        # Test registration
        register_model('test_detector', ObjectDetector, ['object_detection', 'test'])
        assert 'test_detector' in MODEL_REGISTRY
        
        logger.info("✓ Model registry integration successful")
        return True
    
    except Exception as e:
        logger.error(f"✗ Model registry integration failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all basic tests"""
    logger.info("="*60)
    logger.info("Running YOLOv8 Object Detection Basic Tests")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("ObjectDetector Creation", test_object_detector_creation),
        ("VideoStreamDetector Creation", test_video_stream_detector_creation),
        ("ObjectDetectionTask Creation", test_vision_task_creation),
        ("Lazy Imports", test_lazy_imports),
        ("Convenience Functions", test_convenience_functions),
        ("Model Integration", test_model_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Results Summary")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Module is ready for use.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
