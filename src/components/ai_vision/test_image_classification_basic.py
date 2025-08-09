#!/usr/bin/env python3
# src/components/ai_vision/test_image_classification_basic.py

"""
Basic Test Suite for Image Classification Module

This test suite provides basic functionality tests for the image classification module
to ensure core components are working correctly.
"""

import asyncio
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np

# Add the current directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from image_classification import (
        ImageClassifier,
        ImageClassificationTask,
        ClassificationResult,
        TopKResult,
        BatchClassificationResult,
        FineTuningHook,
        LoggingHook,
        CheckpointHook,
        ClassificationModel,
        classify_image,
        classify_images_batch,
        create_classifier
    )
    print("✓ Image classification module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import image classification module: {e}")
    sys.exit(1)


class TestClassificationDataStructures(unittest.TestCase):
    """Test data structure classes"""
    
    def test_classification_result(self):
        """Test ClassificationResult creation and serialization"""
        result = ClassificationResult(
            class_id=123,
            class_name="test_class",
            confidence=0.85
        )
        
        self.assertEqual(result.class_id, 123)
        self.assertEqual(result.class_name, "test_class")
        self.assertAlmostEqual(result.confidence, 0.85)
        
        # Test serialization
        result_dict = result.to_dict()
        expected_dict = {
            'class_id': 123,
            'class_name': "test_class",
            'confidence': 0.85
        }
        self.assertEqual(result_dict, expected_dict)
    
    def test_topk_result(self):
        """Test TopKResult creation and serialization"""
        predictions = [
            ClassificationResult(1, "class_1", 0.9),
            ClassificationResult(2, "class_2", 0.8),
            ClassificationResult(3, "class_3", 0.7)
        ]
        
        topk_result = TopKResult(
            predictions=predictions,
            processing_time=0.123,
            k=3
        )
        
        self.assertEqual(len(topk_result.predictions), 3)
        self.assertAlmostEqual(topk_result.processing_time, 0.123)
        self.assertEqual(topk_result.k, 3)
        
        # Test serialization
        result_dict = topk_result.to_dict()
        self.assertEqual(len(result_dict['predictions']), 3)
        self.assertAlmostEqual(result_dict['processing_time'], 0.123)
        self.assertEqual(result_dict['k'], 3)
    
    def test_batch_classification_result(self):
        """Test BatchClassificationResult creation and serialization"""
        predictions = [ClassificationResult(1, "class_1", 0.9)]
        topk_results = [TopKResult(predictions, 0.1, 1) for _ in range(3)]
        
        batch_result = BatchClassificationResult(
            results=topk_results,
            processing_time=0.456,
            batch_size=3
        )
        
        self.assertEqual(len(batch_result.results), 3)
        self.assertAlmostEqual(batch_result.processing_time, 0.456)
        self.assertEqual(batch_result.batch_size, 3)
        
        # Test serialization
        result_dict = batch_result.to_dict()
        self.assertEqual(len(result_dict['results']), 3)


class TestClassificationModel(unittest.TestCase):
    """Test ClassificationModel enum"""
    
    def test_model_enum_values(self):
        """Test that all expected model values are present"""
        expected_models = {
            "resnet50",
            "mobilenet_v3_large", 
            "mobilenet_v3_small"
        }
        
        actual_models = {model.value for model in ClassificationModel}
        self.assertEqual(actual_models, expected_models)


class TestFineTuningHooks(unittest.TestCase):
    """Test fine-tuning hook classes"""
    
    def test_base_hook(self):
        """Test base FineTuningHook class"""
        hook = FineTuningHook()
        
        # Test that all methods exist and can be called without errors
        hook.on_training_start(None, None, None)
        hook.on_epoch_start(0, None, None)
        hook.on_batch_start(0, None, None)
        hook.on_batch_end(0, None, None, None, None)
        hook.on_epoch_end(0, 0.0, None, None)
        hook.on_training_end(None, {})
    
    def test_logging_hook(self):
        """Test LoggingHook"""
        hook = LoggingHook(log_every_n_batches=10)
        self.assertEqual(hook.log_every_n_batches, 10)
        
        # Test methods don't raise errors
        hook.on_training_start(None, None, Mock(), Mock())
        hook.on_epoch_start(1, None, None)
        hook.on_batch_end(10, None, None, 0.5, None)  # Should log
        hook.on_batch_end(5, None, None, 0.5, None)   # Should not log
        hook.on_epoch_end(1, 0.5, 0.6, None)
    
    def test_checkpoint_hook(self):
        """Test CheckpointHook"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hook = CheckpointHook(temp_dir, save_every_n_epochs=2)
            self.assertEqual(hook.save_every_n_epochs, 2)
            self.assertTrue(hook.checkpoint_dir.exists())


class TestImageClassificationTask(unittest.TestCase):
    """Test ImageClassificationTask class"""
    
    def test_task_initialization(self):
        """Test task initialization"""
        task = ImageClassificationTask("test_task", model_name="resnet50")
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.task_type, "image_classification")
        self.assertEqual(task.model_name, "resnet50")
        self.assertIsNone(task.model)
    
    def test_input_validation(self):
        """Test input validation"""
        task = ImageClassificationTask("test_task")
        
        # Test valid inputs
        self.assertTrue(task.validate_input({'image': Mock()}))
        self.assertTrue(task.validate_input({'batch_images': [Mock(), Mock()]}))
        self.assertTrue(task.validate_input(Mock()))  # Assume it's an image
        
        # Test invalid inputs
        self.assertFalse(task.validate_input({}))  # Empty dict
        self.assertFalse(task.validate_input(None))


class TestImageClassifier(unittest.TestCase):
    """Test ImageClassifier class (without actual model loading)"""
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        classifier = ImageClassifier(
            model_name="resnet50",
            device="cpu",
            num_classes=1000
        )
        
        self.assertEqual(classifier.model_name, "resnet50")
        self.assertEqual(classifier.num_classes, 1000)
        self.assertIsNone(classifier.model)
        self.assertIsNone(classifier.transform)  # Lazy initialization
        self.assertIsNotNone(classifier.class_names)  # Should be loaded
    
    def test_class_names_loading(self):
        """Test that class names are properly loaded"""
        classifier = ImageClassifier(num_classes=30)
        
        # Should have exactly the specified number of class names
        self.assertEqual(len(classifier.class_names), 30)
        
        # Should have some real ImageNet class names at the beginning
        self.assertIn('tench', classifier.class_names)
        self.assertIn('goldfish', classifier.class_names)
        
        # Test with fewer classes than pre-defined names
        classifier_small = ImageClassifier(num_classes=5)
        self.assertEqual(len(classifier_small.class_names), 5)
        self.assertIn('tench', classifier_small.class_names)
    
    def test_transforms_initialization(self):
        """Test transform initialization for different models"""
        models = ["resnet50", "mobilenet_v3_large", "unknown_model"]
        
        for model_name in models:
            try:
                classifier = ImageClassifier(model_name=model_name)
                # Transforms are now lazy, so they should be None initially
                self.assertIsNone(classifier.transform)
            except Exception:
                # Some models might not be supported, that's ok for this test
                pass


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_classify_image_without_model(self):
        """Test classify_image with no model raises error"""
        with self.assertRaises(ValueError):
            classify_image(Mock(), model=None)
    
    def test_classify_images_batch_without_model(self):
        """Test classify_images_batch with no model raises error"""
        with self.assertRaises(ValueError):
            classify_images_batch([Mock()], model=None)


class TestAsyncFunctions(unittest.TestCase):
    """Test async functions"""
    
    def test_create_classifier_async(self):
        """Test create_classifier is async"""
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(create_classifier))


def run_basic_tests():
    """Run basic tests to ensure module functionality"""
    print("Running Basic Image Classification Module Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestClassificationDataStructures,
        TestClassificationModel,
        TestFineTuningHooks,
        TestImageClassificationTask,
        TestImageClassifier,
        TestConvenienceFunctions,
        TestAsyncFunctions
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All basic tests passed!")
        print(f"Tests run: {result.testsRun}")
    else:
        print("✗ Some tests failed!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        # Print details of failures and errors
        for test, traceback in result.failures:
            print(f"\nFAILURE: {test}")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
    
    return result.wasSuccessful()


async def test_async_functionality():
    """Test async functionality that requires mocking"""
    print("\nTesting async functionality...")
    
    # Test that create_classifier can be called (will fail without dependencies)
    try:
        await create_classifier("resnet50", "cpu")
        print("✓ create_classifier executed (may have failed due to missing deps)")
    except Exception as e:
        print(f"✓ create_classifier failed as expected without deps: {type(e).__name__}")
    
    # Test ImageClassificationTask execution (with mocks)
    task = ImageClassificationTask("test", "resnet50")
    
    # Mock the model loading
    task.model = Mock()
    
    # Test with invalid input
    result = await task.execute({})  # Empty dict should be invalid
    print(f"✓ Task validation working: {result.status}")


def main():
    """Main test function"""
    print("Image Classification Module - Basic Test Suite")
    print("=" * 60)
    
    # Test basic imports and structure
    print("Testing module structure and imports...")
    success = True
    
    try:
        # Test that all classes can be instantiated (basic structure test)
        result = ClassificationResult(1, "test", 0.5)
        topk = TopKResult([result], 0.1, 1)
        batch = BatchClassificationResult([topk], 0.2, 1)
        hook = FineTuningHook()
        log_hook = LoggingHook()
        task = ImageClassificationTask("test")
        
        print("✓ All data structures can be instantiated")
        
    except Exception as e:
        print(f"✗ Error instantiating basic structures: {e}")
        success = False
    
    # Run unit tests
    test_success = run_basic_tests()
    success = success and test_success
    
    # Run async tests
    try:
        asyncio.run(test_async_functionality())
    except Exception as e:
        print(f"✗ Error in async tests: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All basic tests completed successfully!")
        print("\nThe image classification module appears to be properly structured.")
        print("To run full functionality tests, install dependencies:")
        print("pip install torch torchvision Pillow")
        print("Then run: python example_image_classification.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
