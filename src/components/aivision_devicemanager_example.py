#!/usr/bin/env python3
"""
Example integration between AIVisionAgent and enhanced DeviceManager

This demonstrates how the AIVisionAgent can:
1. Subscribe to live camera streams
2. Process frames in real-time
3. Push annotated frames back to the stream
4. Handle multiple streams simultaneously

Usage:
    python aivision_devicemanager_example.py
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional
from device_manager import DeviceManager, FrameData, StreamStatus
from ai_vision.core import AIVisionAgent, VisionResult, TaskStatus

class StreamingVisionAgent(AIVisionAgent):
    """
    Extended AIVisionAgent with streaming capabilities
    """
    
    def __init__(self, agent_id: str, name: str, device_manager: DeviceManager, **config):
        super().__init__(agent_id, name, **config)
        self.device_manager = device_manager
        self.active_subscriptions: Dict[str, List] = {}  # stream_id -> [callbacks]
        
    def subscribe_to_camera_stream(self, camera_id: int, resolution: tuple = (640, 480), 
                                 fps: int = 30, processing_enabled: bool = True) -> Optional[str]:
        """
        Subscribe to a camera stream for real-time processing.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fps: Frames per second
            processing_enabled: Whether to enable AI processing
            
        Returns:
            str: Stream ID if successful, None otherwise
        """
        # Create or get the camera stream
        stream_id = self.device_manager.get_camera_stream(camera_id, resolution, fps)
        if not stream_id:
            self.logger.error(f"Failed to create camera stream for camera {camera_id}")
            return None
        
        # Define callbacks for this stream
        def frame_callback(frame_data: FrameData):
            self._process_frame(stream_id, frame_data, processing_enabled)
        
        def status_callback(status: StreamStatus, message: str):
            self.logger.info(f"Stream {stream_id} status: {status.value} - {message}")
        
        # Subscribe to the stream
        success = self.device_manager.subscribe_to_stream(
            stream_id, frame_callback, status_callback
        )
        
        if success:
            self.active_subscriptions[stream_id] = [frame_callback, status_callback]
            self.logger.info(f"Successfully subscribed to stream {stream_id}")
            return stream_id
        else:
            self.logger.error(f"Failed to subscribe to stream {stream_id}")
            return None
    
    def unsubscribe_from_stream(self, stream_id: str) -> bool:
        """
        Unsubscribe from a camera stream.
        
        Args:
            stream_id: ID of the stream to unsubscribe from
            
        Returns:
            bool: True if successful, False otherwise
        """
        if stream_id not in self.active_subscriptions:
            return False
        
        callbacks = self.active_subscriptions[stream_id]
        frame_callback, status_callback = callbacks
        
        # Unsubscribe from the stream
        success = self.device_manager.unsubscribe_from_stream(
            stream_id, frame_callback, status_callback
        )
        
        if success:
            del self.active_subscriptions[stream_id]
            self.logger.info(f"Successfully unsubscribed from stream {stream_id}")
        
        return success
    
    def stop_stream(self, stream_id: str) -> bool:
        """
        Stop a camera stream completely.
        
        Args:
            stream_id: ID of the stream to stop
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First unsubscribe
        self.unsubscribe_from_stream(stream_id)
        
        # Then stop the stream
        return self.device_manager.stop_camera_stream(stream_id)
    
    def _process_frame(self, stream_id: str, frame_data: FrameData, processing_enabled: bool):
        """
        Process a frame from the camera stream.
        
        Args:
            stream_id: ID of the stream
            frame_data: Frame data to process
            processing_enabled: Whether to apply AI processing
        """
        try:
            if not processing_enabled:
                return
            
            # Simulate AI processing (face detection, object detection, etc.)
            processed_frame = self._apply_ai_processing(frame_data.frame)
            
            # Push the processed frame back to the stream
            self.device_manager.push_annotated_frame(
                stream_id, 
                processed_frame,
                {
                    'processing_timestamp': time.time(),
                    'agent_id': self.agent_id,
                    'processing_type': 'face_detection'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame from stream {stream_id}: {e}")
    
    def _apply_ai_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply AI processing to a frame (mock implementation).
        
        Args:
            frame: Input frame
            
        Returns:
            np.ndarray: Processed frame
        """
        # Mock AI processing - just add some annotations
        processed_frame = frame.copy()
        
        # Add timestamp
        timestamp_text = f"Processed: {time.strftime('%H:%M:%S')}"
        cv2.putText(processed_frame, timestamp_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add agent info
        agent_text = f"Agent: {self.name}"
        cv2.putText(processed_frame, agent_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Mock face detection (just draw a rectangle)
        h, w = processed_frame.shape[:2]
        cv2.rectangle(processed_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 255), 2)
        cv2.putText(processed_frame, "FACE DETECTED", (w//4, h//4-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return processed_frame
    
    def get_stream_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all active streams.
        
        Returns:
            Dict: Stream statistics
        """
        stats = {}
        active_streams = self.device_manager.get_active_streams()
        
        for stream_id in self.active_subscriptions.keys():
            if stream_id in active_streams:
                stream_info = active_streams[stream_id]
                stats[stream_id] = {
                    'camera_id': stream_info['camera_id'],
                    'resolution': stream_info['resolution'],
                    'fps': stream_info['fps'],
                    'status': stream_info['status'],
                    'frame_count': stream_info['frame_count'],
                    'subscribed': True
                }
            else:
                stats[stream_id] = {
                    'status': 'inactive',
                    'subscribed': True
                }
        
        return stats


def main():
    """
    Demo function showing AIVisionAgent integration with DeviceManager
    """
    print("AIVisionAgent + DeviceManager Integration Demo")
    print("=" * 50)
    
    # Initialize DeviceManager
    device_manager = DeviceManager()
    
    # Initialize AIVisionAgent with streaming capabilities
    agent = StreamingVisionAgent(
        agent_id="streaming_agent_001",
        name="StreamingVisionAgent",
        device_manager=device_manager,
        max_concurrent_tasks=3
    )
    
    # Find available cameras
    cameras = device_manager.get_local_cameras()
    if not cameras:
        print("No cameras found. Please connect a camera and try again.")
        return
    
    print(f"Found {len(cameras)} camera(s)")
    for cam in cameras:
        print(f"  - {cam['name']} (index: {cam['index']})")
    
    # Subscribe to the first camera
    camera_id = cameras[0]['index']
    stream_id = agent.subscribe_to_camera_stream(
        camera_id=camera_id,
        resolution=(640, 480),
        fps=15,  # Lower FPS for demo
        processing_enabled=True
    )
    
    if stream_id:
        print(f"\nSuccessfully subscribed to camera stream: {stream_id}")
        
        # Let it run for a few seconds
        print("Processing frames for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            # Show stream statistics every 2 seconds
            if int(time.time() - start_time) % 2 == 0:
                stats = agent.get_stream_statistics()
                for sid, stat in stats.items():
                    print(f"Stream {sid}: {stat['frame_count']} frames, status: {stat['status']}")
            
            time.sleep(0.5)
        
        # Final statistics
        print("\nFinal stream statistics:")
        final_stats = agent.get_stream_statistics()
        for sid, stat in final_stats.items():
            print(f"  {sid}: {stat}")
        
        # Cleanup
        print("\nCleaning up...")
        agent.stop_stream(stream_id)
        device_manager.cleanup_all_streams()
        print("Demo completed successfully!")
        
    else:
        print("Failed to subscribe to camera stream")


if __name__ == "__main__":
    main()
