#!/usr/bin/env python3
"""
Test suite for DeviceManager streaming functionality

This tests all the new streaming capabilities added to DeviceManager:
- Camera stream creation and management
- Callback subscription/unsubscription
- Frame processing and annotation
- Multiple stream handling
- Error handling and cleanup

Usage:
    python test_device_manager_streaming.py
"""

import unittest
import threading
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from device_manager import (
    DeviceManager, CameraStream, StreamStatus, StreamConfig, 
    FrameData
)


class TestDeviceManagerStreaming(unittest.TestCase):
    """Test cases for DeviceManager streaming functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device_manager = DeviceManager()
        self.received_frames = []
        self.received_statuses = []
    
    def tearDown(self):
        """Clean up after tests"""
        self.device_manager.cleanup_all_streams()
        self.received_frames.clear()
        self.received_statuses.clear()
    
    def frame_callback_test(self, frame_data: FrameData):
        """Test callback for receiving frames"""
        self.received_frames.append(frame_data)
    
    def status_callback_test(self, status: StreamStatus, message: str):
        """Test callback for receiving status updates"""
        self.received_statuses.append((status, message))
    
    @patch('cv2.VideoCapture')
    def test_camera_stream_creation(self, mock_video_capture):
        """Test creating a camera stream"""
        # Mock successful camera opening
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        # Test stream creation
        stream_id = self.device_manager.get_camera_stream(
            camera_id=0,
            resolution=(640, 480),
            fps=30
        )
        
        self.assertIsNotNone(stream_id)
        self.assertTrue(stream_id.startswith("camera_0"))
        
        # Verify the stream is active
        active_streams = self.device_manager.get_active_streams()
        self.assertIn(stream_id, active_streams)
        self.assertEqual(active_streams[stream_id]['camera_id'], 0)
        self.assertEqual(active_streams[stream_id]['resolution'], (640, 480))
        self.assertEqual(active_streams[stream_id]['fps'], 30)
    
    @patch('cv2.VideoCapture')
    def test_stream_subscription(self, mock_video_capture):
        """Test subscribing to a stream"""
        # Mock successful camera opening
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        # Create stream
        stream_id = self.device_manager.get_camera_stream(0, (640, 480), 30)
        
        # Subscribe to stream
        success = self.device_manager.subscribe_to_stream(
            stream_id, 
            self.frame_callback_test, 
            self.status_callback_test
        )
        
        self.assertTrue(success)
        
        # Verify subscription
        stream = self.device_manager.active_streams[stream_id]
        self.assertEqual(len(stream.frame_callbacks), 1)
        self.assertEqual(len(stream.status_callbacks), 1)
    
    @patch('cv2.VideoCapture')
    def test_stream_unsubscription(self, mock_video_capture):
        """Test unsubscribing from a stream"""
        # Setup
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        stream_id = self.device_manager.get_camera_stream(0, (640, 480), 30)
        self.device_manager.subscribe_to_stream(
            stream_id, self.frame_callback_test, self.status_callback_test
        )
        
        # Test unsubscription
        success = self.device_manager.unsubscribe_from_stream(
            stream_id, self.frame_callback_test, self.status_callback_test
        )
        
        self.assertTrue(success)
        
        # Verify unsubscription
        stream = self.device_manager.active_streams[stream_id]
        self.assertEqual(len(stream.frame_callbacks), 0)
        self.assertEqual(len(stream.status_callbacks), 0)
    
    @patch('cv2.VideoCapture')
    def test_stream_stopping(self, mock_video_capture):
        """Test stopping a camera stream"""
        # Setup
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        stream_id = self.device_manager.get_camera_stream(0, (640, 480), 30)
        
        # Stop stream
        success = self.device_manager.stop_camera_stream(stream_id)
        self.assertTrue(success)
        
        # Verify stream is removed
        active_streams = self.device_manager.get_active_streams()
        self.assertNotIn(stream_id, active_streams)
    
    @patch('cv2.VideoCapture')
    def test_multiple_streams(self, mock_video_capture):
        """Test handling multiple streams simultaneously"""
        # Mock successful camera opening
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        # Create multiple streams
        stream_ids = []
        for i in range(3):
            stream_id = self.device_manager.get_camera_stream(
                camera_id=i,
                resolution=(640, 480),
                fps=30
            )
            stream_ids.append(stream_id)
        
        # Verify all streams are active
        active_streams = self.device_manager.get_active_streams()
        self.assertEqual(len(active_streams), 3)
        
        for stream_id in stream_ids:
            self.assertIn(stream_id, active_streams)
        
        # Subscribe to all streams
        for stream_id in stream_ids:
            success = self.device_manager.subscribe_to_stream(
                stream_id, self.frame_callback_test
            )
            self.assertTrue(success)
    
    @patch('cv2.VideoCapture')
    def test_annotated_frame_push(self, mock_video_capture):
        """Test pushing annotated frames back to stream"""
        # Setup
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        stream_id = self.device_manager.get_camera_stream(0, (640, 480), 30)
        self.device_manager.subscribe_to_stream(stream_id, self.frame_callback_test)
        
        # Create mock annotated frame
        annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metadata = {'annotation': 'test_annotation'}
        
        # Push annotated frame
        success = self.device_manager.push_annotated_frame(
            stream_id, annotated_frame, metadata
        )
        
        self.assertTrue(success)
    
    @patch('cv2.VideoCapture')
    def test_error_handling_camera_not_available(self, mock_video_capture):
        """Test error handling when camera is not available"""
        # Mock failed camera opening
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        # Try to create stream
        stream_id = self.device_manager.get_camera_stream(0, (640, 480), 30)
        
        # Should return None for failed stream creation
        self.assertIsNone(stream_id)
    
    def test_invalid_stream_operations(self):
        """Test operations on invalid/non-existent streams"""
        fake_stream_id = "nonexistent_stream"
        
        # Test subscribing to non-existent stream
        success = self.device_manager.subscribe_to_stream(
            fake_stream_id, self.frame_callback_test
        )
        self.assertFalse(success)
        
        # Test unsubscribing from non-existent stream
        success = self.device_manager.unsubscribe_from_stream(
            fake_stream_id, self.frame_callback_test
        )
        self.assertFalse(success)
        
        # Test stopping non-existent stream
        success = self.device_manager.stop_camera_stream(fake_stream_id)
        self.assertFalse(success)
        
        # Test pushing to non-existent stream
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        success = self.device_manager.push_annotated_frame(
            fake_stream_id, fake_frame
        )
        self.assertFalse(success)
    
    @patch('cv2.VideoCapture')
    def test_stream_cleanup_all(self, mock_video_capture):
        """Test cleanup of all streams"""
        # Setup multiple streams
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        stream_ids = []
        for i in range(3):
            stream_id = self.device_manager.get_camera_stream(i, (640, 480), 30)
            stream_ids.append(stream_id)
        
        # Verify streams are active
        self.assertEqual(len(self.device_manager.get_active_streams()), 3)
        
        # Cleanup all streams
        self.device_manager.cleanup_all_streams()
        
        # Verify all streams are removed
        self.assertEqual(len(self.device_manager.get_active_streams()), 0)
    
    def test_stream_config(self):
        """Test StreamConfig dataclass"""
        config = StreamConfig(
            camera_id=0,
            resolution=(1920, 1080),
            fps=60,
            buffer_size=20,
            auto_restart=False
        )
        
        self.assertEqual(config.camera_id, 0)
        self.assertEqual(config.resolution, (1920, 1080))
        self.assertEqual(config.fps, 60)
        self.assertEqual(config.buffer_size, 20)
        self.assertFalse(config.auto_restart)
    
    def test_frame_data(self):
        """Test FrameData dataclass"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = time.time()
        metadata = {'test': 'data'}
        
        frame_data = FrameData(
            frame=frame,
            timestamp=timestamp,
            frame_number=42,
            camera_id=0,
            metadata=metadata
        )
        
        self.assertEqual(frame_data.frame_number, 42)
        self.assertEqual(frame_data.camera_id, 0)
        self.assertEqual(frame_data.timestamp, timestamp)
        self.assertEqual(frame_data.metadata, metadata)
        np.testing.assert_array_equal(frame_data.frame, frame)


class TestCameraStream(unittest.TestCase):
    """Test cases for CameraStream class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = StreamConfig(
            camera_id=0,
            resolution=(640, 480),
            fps=30
        )
        self.received_frames = []
        self.received_statuses = []
    
    def frame_callback_test(self, frame_data: FrameData):
        """Test callback for receiving frames"""
        self.received_frames.append(frame_data)
    
    def status_callback_test(self, status: StreamStatus, message: str):
        """Test callback for receiving status updates"""
        self.received_statuses.append((status, message))
    
    def test_stream_initialization(self):
        """Test CameraStream initialization"""
        stream = CameraStream(self.config)
        
        self.assertEqual(stream.config, self.config)
        self.assertEqual(stream.status, StreamStatus.INACTIVE)
        self.assertEqual(stream.frame_number, 0)
        self.assertIsNone(stream.last_frame)
        self.assertEqual(len(stream.frame_callbacks), 0)
        self.assertEqual(len(stream.status_callbacks), 0)
    
    def test_callback_management(self):
        """Test adding and removing callbacks"""
        stream = CameraStream(self.config)
        
        # Add callbacks
        stream.add_frame_callback(self.frame_callback_test)
        stream.add_status_callback(self.status_callback_test)
        
        self.assertEqual(len(stream.frame_callbacks), 1)
        self.assertEqual(len(stream.status_callbacks), 1)
        
        # Remove callbacks
        stream.remove_frame_callback(self.frame_callback_test)
        stream.remove_status_callback(self.status_callback_test)
        
        self.assertEqual(len(stream.frame_callbacks), 0)
        self.assertEqual(len(stream.status_callbacks), 0)
    
    def test_duplicate_callback_prevention(self):
        """Test that duplicate callbacks are not added"""
        stream = CameraStream(self.config)
        
        # Add same callback multiple times
        stream.add_frame_callback(self.frame_callback_test)
        stream.add_frame_callback(self.frame_callback_test)
        stream.add_frame_callback(self.frame_callback_test)
        
        # Should only have one instance
        self.assertEqual(len(stream.frame_callbacks), 1)


def run_integration_test():
    """
    Integration test that requires actual camera hardware.
    Only runs if a camera is available.
    """
    print("\n" + "="*50)
    print("INTEGRATION TEST (requires camera)")
    print("="*50)
    
    dm = DeviceManager()
    cameras = dm.get_local_cameras()
    
    if not cameras:
        print("No cameras found. Skipping integration test.")
        return
    
    print(f"Found {len(cameras)} camera(s). Using first camera for test.")
    
    # Track received data
    frames_received = []
    statuses_received = []
    
    def frame_callback(frame_data):
        frames_received.append(frame_data)
        print(f"Received frame {frame_data.frame_number} from camera {frame_data.camera_id}")
    
    def status_callback(status, message):
        statuses_received.append((status, message))
        print(f"Status: {status.value} - {message}")
    
    # Create and test stream
    camera_id = cameras[0]['index']
    stream_id = dm.get_camera_stream(camera_id, (640, 480), 10)
    
    if stream_id:
        print(f"Created stream: {stream_id}")
        
        # Subscribe and run for a few seconds
        dm.subscribe_to_stream(stream_id, frame_callback, status_callback)
        print("Streaming for 5 seconds...")
        time.sleep(5)
        
        # Test annotated frame push
        if frames_received:
            last_frame = frames_received[-1]
            annotated_frame = last_frame.frame.copy()
            # Add simple annotation
            import cv2
            cv2.putText(annotated_frame, "TEST ANNOTATION", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            success = dm.push_annotated_frame(stream_id, annotated_frame, 
                                            {'annotation': 'integration_test'})
            print(f"Pushed annotated frame: {success}")
        
        # Show results
        print(f"Total frames received: {len(frames_received)}")
        print(f"Status updates received: {len(statuses_received)}")
        
        # Cleanup
        dm.cleanup_all_streams()
        print("Integration test completed successfully!")
    else:
        print("Failed to create camera stream")


def main():
    """Run all tests"""
    print("DeviceManager Streaming Test Suite")
    print("="*50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test if requested
    try:
        import sys
        if '--integration' in sys.argv:
            run_integration_test()
    except Exception as e:
        print(f"Integration test failed: {e}")


if __name__ == "__main__":
    main()
