#!/usr/bin/env python3
"""
Unit tests for Vision Task Routing in Orchestrator

Tests the new handle_request method for vision tasks including:
- Object detection routing
- Computer vision analytics routing
- Error handling
- Response format validation
"""

import sys
import os
import unittest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.orchestration import Orchestrator
    from src.components.ai_vision.core import VisionResult, TaskStatus
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class TestVisionOrchestrator(unittest.TestCase):
    """Test cases for vision task routing in Orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = Orchestrator()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_batch_frames = [self.test_frame.copy() for _ in range(3)]
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly"""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNotNone(self.orchestrator.components)
        self.assertIsNotNone(self.orchestrator.planner)
    
    async def test_unsupported_request_type(self):
        """Test handling of unsupported request types"""
        result = await self.orchestrator.handle_request(
            type='unsupported_type',
            task='some_task',
            payload={}
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Unsupported request type', result['error'])
        self.assertIsNone(result['data'])
        self.assertIn('metadata', result)
        self.assertIn('timestamp', result['metadata'])
    
    async def test_unsupported_vision_task(self):
        """Test handling of unsupported vision tasks"""
        result = await self.orchestrator.handle_request(
            type='vision',
            task='unsupported_vision_task',
            payload={}
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Unsupported vision task', result['error'])
        self.assertIsNone(result['data'])
        self.assertIn('vision_task', result['metadata'])
    
    async def test_object_detection_missing_input(self):
        """Test object detection with missing input data"""
        result = await self.orchestrator.handle_request(
            type='vision',
            task='object_detection',
            payload={
                'conf_threshold': 0.25
                # Missing frame or batch_frames
            }
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Either frame or batch_frames must be provided', result['error'])
        self.assertIn('vision_task', result['metadata'])
        self.assertEqual(result['metadata']['vision_task'], 'object_detection')
    
    async def test_analytics_missing_input(self):
        """Test analytics with missing input data"""
        result = await self.orchestrator.handle_request(
            type='vision',
            task='analytics',
            payload={
                'analytics_config': {}
                # Missing input_data
            }
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('input_data must be provided', result['error'])
        self.assertIn('vision_task', result['metadata'])
        self.assertEqual(result['metadata']['vision_task'], 'analytics')
    
    @patch('src.core.orchestration.ObjectDetectionTask')
    async def test_object_detection_success_mock(self, mock_detection_task):
        """Test successful object detection with mocked task"""
        # Mock successful detection result
        mock_result = VisionResult(
            task_id='test_task',
            status=TaskStatus.COMPLETED,
            data=[{
                'bbox': [100, 100, 200, 200],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'person'
            }],
            confidence=0.85,
            metadata={'processing_type': 'single_frame'}
        )
        
        mock_task_instance = Mock()
        mock_task_instance.execute = AsyncMock(return_value=mock_result)
        mock_detection_task.return_value = mock_task_instance
        
        result = await self.orchestrator.handle_request(
            type='vision',
            task='object_detection',
            payload={
                'frame': self.test_frame,
                'conf_threshold': 0.25,
                'model_name': 'yolov8n'
            }
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['confidence'], 0.85)
        self.assertIsNotNone(result['data'])
        self.assertIn('task_id', result['metadata'])
        self.assertEqual(result['metadata']['vision_task'], 'object_detection')
        self.assertEqual(result['metadata']['model_name'], 'yolov8n')
        self.assertEqual(result['metadata']['conf_threshold'], 0.25)
    
    @patch('src.core.orchestration.ObjectDetectionTask')
    async def test_object_detection_failure_mock(self, mock_detection_task):
        """Test failed object detection with mocked task"""
        # Mock failed detection result
        mock_result = VisionResult(
            task_id='test_task',
            status=TaskStatus.FAILED,
            data=None,
            confidence=0.0,
            error_message='Model loading failed'
        )
        
        mock_task_instance = Mock()
        mock_task_instance.execute = AsyncMock(return_value=mock_result)
        mock_detection_task.return_value = mock_task_instance
        
        result = await self.orchestrator.handle_request(
            type='vision',
            task='object_detection',
            payload={
                'frame': self.test_frame,
                'conf_threshold': 0.25
            }
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Model loading failed', result['error'])
        self.assertIn('task_id', result['metadata'])
    
    @patch('src.core.orchestration.analytics_agent')
    async def test_analytics_success_mock(self, mock_analytics_agent):
        """Test successful analytics with mocked agent"""
        # Mock successful analytics result
        mock_result = VisionResult(
            task_id='analytics_task',
            status=TaskStatus.COMPLETED,
            data={
                'motion_result': {'motion_detected': True, 'motion_area': 1500},
                'events': [],
                'confidence': 0.75
            },
            confidence=0.75
        )
        
        mock_analytics_agent.create_analytics_task.return_value = 'test_task_id'
        mock_analytics_agent.execute_task = AsyncMock(return_value=mock_result)
        mock_analytics_agent.unregister_task.return_value = True
        
        result = await self.orchestrator.handle_request(
            type='vision',
            task='analytics',
            payload={
                'input_data': self.test_frame,
                'analytics_config': {
                    'motion_config': {'threshold': 25}
                }
            }
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['confidence'], 0.75)
        self.assertIsNotNone(result['data'])
        self.assertIn('task_id', result['metadata'])
        self.assertEqual(result['metadata']['vision_task'], 'analytics')
        
        # Verify agent methods were called
        mock_analytics_agent.create_analytics_task.assert_called_once()
        mock_analytics_agent.execute_task.assert_called_once()
        mock_analytics_agent.unregister_task.assert_called_once()
    
    async def test_response_format_validation(self):
        """Test that all responses have the correct format"""
        # Test error response format
        result = await self.orchestrator.handle_request(
            type='vision',
            task='object_detection',
            payload={}  # Missing required data
        )
        
        # Check required fields for error response
        self.assertIn('status', result)
        self.assertIn('error', result)
        self.assertIn('data', result)
        self.assertIn('metadata', result)
        self.assertIn('timestamp', result['metadata'])
        
        # Verify field types
        self.assertIsInstance(result['status'], str)
        self.assertIsInstance(result['error'], str)
        self.assertIsInstance(result['metadata'], dict)
    
    def test_timestamp_generation(self):
        """Test timestamp generation utility"""
        timestamp = self.orchestrator._get_timestamp()
        self.assertIsInstance(timestamp, str)
        # Check basic ISO format (contains T and colons)
        self.assertIn('T', timestamp)
        self.assertIn(':', timestamp)
    
    async def test_exception_handling(self):
        """Test that exceptions are properly caught and returned as errors"""
        # This test simulates an exception during request processing
        with patch.object(self.orchestrator, '_handle_vision_request', 
                         side_effect=Exception('Simulated error')):
            result = await self.orchestrator.handle_request(
                type='vision',
                task='object_detection',
                payload={'frame': self.test_frame}
            )
            
            self.assertEqual(result['status'], 'error')
            self.assertIn('Simulated error', result['error'])
            self.assertIn('metadata', result)


class TestVisionOrchestratorIntegration(unittest.TestCase):
    """Integration tests (requires actual dependencies)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = Orchestrator()
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @unittest.skipUnless(
        self._check_dependencies(), 
        "Vision dependencies not available"
    )
    async def test_real_object_detection_integration(self):
        """Integration test with real object detection (if dependencies available)"""
        try:
            result = await self.orchestrator.handle_request(
                type='vision',
                task='object_detection',
                payload={
                    'frame': self.test_frame,
                    'conf_threshold': 0.5,
                    'model_name': 'yolov8n'
                }
            )
            
            # Should either succeed or fail gracefully
            self.assertIn(result['status'], ['success', 'error'])
            self.assertIn('metadata', result)
            
        except Exception as e:
            self.fail(f"Integration test should not raise exceptions: {e}")
    
    @staticmethod
    def _check_dependencies():
        """Check if vision dependencies are available"""
        try:
            import cv2
            import torch
            import ultralytics
            return True
        except ImportError:
            return False


def run_async_tests():
    """Helper to run async test methods"""
    async def run_test_suite():
        # Create test suite for sync tests
        test_suite = unittest.TestSuite()
        
        # Add sync tests
        for test_class in [TestVisionOrchestrator]:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run sync tests
        runner = unittest.TextTestRunner(verbosity=2)
        sync_result = runner.run(test_suite)
        
        # Run async tests manually
        print("\n" + "="*50)
        print("Running Async Tests")
        print("="*50)
        
        test_instance = TestVisionOrchestrator()
        test_instance.setUp()
        
        async_tests = [
            ('test_unsupported_request_type', test_instance.test_unsupported_request_type),
            ('test_unsupported_vision_task', test_instance.test_unsupported_vision_task),
            ('test_object_detection_missing_input', test_instance.test_object_detection_missing_input),
            ('test_analytics_missing_input', test_instance.test_analytics_missing_input),
            ('test_object_detection_success_mock', test_instance.test_object_detection_success_mock),
            ('test_object_detection_failure_mock', test_instance.test_object_detection_failure_mock),
            ('test_analytics_success_mock', test_instance.test_analytics_success_mock),
            ('test_response_format_validation', test_instance.test_response_format_validation),
            ('test_exception_handling', test_instance.test_exception_handling),
        ]
        
        async_results = []
        for test_name, test_method in async_tests:
            print(f"\nRunning {test_name}...")
            try:
                await test_method()
                print(f"✓ {test_name} PASSED")
                async_results.append((test_name, True, None))
            except Exception as e:
                print(f"✗ {test_name} FAILED: {e}")
                async_results.append((test_name, False, str(e)))
        
        # Summary
        print("\n" + "="*50)
        print("Async Test Results Summary")
        print("="*50)
        
        passed = sum(1 for _, success, _ in async_results if success)
        total = len(async_results)
        
        for test_name, success, error in async_results:
            status = "PASS" if success else "FAIL"
            print(f"{test_name}: {status}")
            if error:
                print(f"  Error: {error}")
        
        print(f"\nAsync Tests: {passed}/{total} passed")
        print(f"Sync Tests: {sync_result.testsRun - sync_result.failures - sync_result.errors}/{sync_result.testsRun} passed")
        
        return passed == total and sync_result.wasSuccessful()
    
    return asyncio.run(run_test_suite())


if __name__ == '__main__':
    print("Vision Orchestrator Test Suite")
    print("="*50)
    
    success = run_async_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
