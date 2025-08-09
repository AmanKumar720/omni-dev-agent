import platform
import sounddevice as sd
import cv2
import pyaudio
import socket
import nmap
import threading
import time
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class StreamStatus(Enum):
    """Enumeration of camera stream statuses"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for camera stream"""
    camera_id: int
    resolution: Tuple[int, int]  # (width, height)
    fps: int
    buffer_size: int = 10
    auto_restart: bool = True


@dataclass
class FrameData:
    """Container for frame data with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    camera_id: int
    metadata: Optional[Dict[str, Any]] = None


class CameraStream:
    """Manages live camera streaming with callback support"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.status = StreamStatus.INACTIVE
        self.capture = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_callbacks: List[Callable[[FrameData], None]] = []
        self.status_callbacks: List[Callable[[StreamStatus, str], None]] = []
        self.frame_number = 0
        self.last_frame: Optional[FrameData] = None
        self._lock = threading.Lock()
    
    def add_frame_callback(self, callback: Callable[[FrameData], None]) -> None:
        """Add a callback function to receive frame data"""
        with self._lock:
            if callback not in self.frame_callbacks:
                self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[FrameData], None]) -> None:
        """Remove a frame callback function"""
        with self._lock:
            if callback in self.frame_callbacks:
                self.frame_callbacks.remove(callback)
    
    def add_status_callback(self, callback: Callable[[StreamStatus, str], None]) -> None:
        """Add a callback function to receive status updates"""
        with self._lock:
            if callback not in self.status_callbacks:
                self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[StreamStatus, str], None]) -> None:
        """Remove a status callback function"""
        with self._lock:
            if callback in self.status_callbacks:
                self.status_callbacks.remove(callback)
    
    def _notify_status_callbacks(self, status: StreamStatus, message: str = "") -> None:
        """Notify all status callbacks about status changes"""
        with self._lock:
            callbacks = self.status_callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(status, message)
            except Exception as e:
                print(f"Error in status callback: {e}")
    
    def _notify_frame_callbacks(self, frame_data: FrameData) -> None:
        """Notify all frame callbacks about new frames"""
        with self._lock:
            callbacks = self.frame_callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(frame_data)
            except Exception as e:
                print(f"Error in frame callback: {e}")
    
    def start(self) -> bool:
        """Start the camera stream"""
        if self.status != StreamStatus.INACTIVE:
            return False
        
        self.status = StreamStatus.STARTING
        self._notify_status_callbacks(self.status, "Starting camera stream")
        
        try:
            self.capture = cv2.VideoCapture(self.config.camera_id)
            if not self.capture.isOpened():
                self.status = StreamStatus.ERROR
                self._notify_status_callbacks(self.status, f"Failed to open camera {self.config.camera_id}")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Start streaming thread
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            
            self.status = StreamStatus.ACTIVE
            self._notify_status_callbacks(self.status, "Camera stream started")
            return True
            
        except Exception as e:
            self.status = StreamStatus.ERROR
            self._notify_status_callbacks(self.status, f"Error starting stream: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the camera stream"""
        if self.status not in [StreamStatus.ACTIVE, StreamStatus.STARTING]:
            return False
        
        self.status = StreamStatus.STOPPING
        self._notify_status_callbacks(self.status, "Stopping camera stream")
        
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        self.status = StreamStatus.INACTIVE
        self._notify_status_callbacks(self.status, "Camera stream stopped")
        return True
    
    def _stream_loop(self) -> None:
        """Main streaming loop (runs in separate thread)"""
        frame_interval = 1.0 / self.config.fps
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    self.frame_number += 1
                    frame_data = FrameData(
                        frame=frame,
                        timestamp=time.time(),
                        frame_number=self.frame_number,
                        camera_id=self.config.camera_id,
                        metadata={
                            'resolution': self.config.resolution,
                            'fps': self.config.fps
                        }
                    )
                    
                    self.last_frame = frame_data
                    self._notify_frame_callbacks(frame_data)
                else:
                    if self.config.auto_restart:
                        self._notify_status_callbacks(StreamStatus.ERROR, "Lost camera connection, attempting restart")
                        time.sleep(1.0)  # Brief pause before restart attempt
                        continue
                    else:
                        self.status = StreamStatus.ERROR
                        self._notify_status_callbacks(self.status, "Lost camera connection")
                        break
            else:
                self.status = StreamStatus.ERROR
                self._notify_status_callbacks(self.status, "Camera capture is not available")
                break
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    
    def get_last_frame(self) -> Optional[FrameData]:
        """Get the most recent frame"""
        return self.last_frame
    
    def get_status(self) -> StreamStatus:
        """Get current stream status"""
        return self.status
    
    def is_active(self) -> bool:
        """Check if stream is currently active"""
        return self.status == StreamStatus.ACTIVE

class DeviceManager:
    def __init__(self):
        self.nm = nmap.PortScanner()
        self.active_streams: Dict[str, CameraStream] = {}  # stream_id -> CameraStream
        self._stream_lock = threading.Lock()
    
    def get_camera_stream(self, camera_id: int, resolution: Tuple[int, int], fps: int, 
                         stream_id: Optional[str] = None, auto_restart: bool = True) -> Optional[str]:
        """
        Create or get a camera stream with specified parameters.
        
        Args:
            camera_id: Camera device ID
            resolution: Tuple of (width, height)
            fps: Frames per second
            stream_id: Optional stream identifier. If None, auto-generated.
            auto_restart: Whether to auto-restart on connection loss
            
        Returns:
            str: Stream ID if successful, None if failed
        """
        if stream_id is None:
            stream_id = f"camera_{camera_id}_{resolution[0]}x{resolution[1]}_{fps}fps"
        
        with self._stream_lock:
            # Check if stream already exists
            if stream_id in self.active_streams:
                existing_stream = self.active_streams[stream_id]
                if existing_stream.is_active():
                    return stream_id
                else:
                    # Remove inactive stream
                    self._cleanup_stream(stream_id)
            
            # Create new stream configuration
            config = StreamConfig(
                camera_id=camera_id,
                resolution=resolution,
                fps=fps,
                auto_restart=auto_restart
            )
            
            # Create and start stream
            stream = CameraStream(config)
            if stream.start():
                self.active_streams[stream_id] = stream
                return stream_id
            else:
                return None
    
    def subscribe_to_stream(self, stream_id: str, 
                           frame_callback: Optional[Callable[[FrameData], None]] = None,
                           status_callback: Optional[Callable[[StreamStatus, str], None]] = None) -> bool:
        """
        Subscribe to a camera stream with callbacks.
        
        Args:
            stream_id: ID of the stream to subscribe to
            frame_callback: Callback for receiving frame data
            status_callback: Callback for receiving status updates
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        with self._stream_lock:
            stream = self.active_streams.get(stream_id)
            if not stream:
                return False
            
            if frame_callback:
                stream.add_frame_callback(frame_callback)
            if status_callback:
                stream.add_status_callback(status_callback)
            
            return True
    
    def unsubscribe_from_stream(self, stream_id: str,
                               frame_callback: Optional[Callable[[FrameData], None]] = None,
                               status_callback: Optional[Callable[[StreamStatus, str], None]] = None) -> bool:
        """
        Unsubscribe from a camera stream.
        
        Args:
            stream_id: ID of the stream to unsubscribe from
            frame_callback: Frame callback to remove
            status_callback: Status callback to remove
            
        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        with self._stream_lock:
            stream = self.active_streams.get(stream_id)
            if not stream:
                return False
            
            if frame_callback:
                stream.remove_frame_callback(frame_callback)
            if status_callback:
                stream.remove_status_callback(status_callback)
            
            return True
    
    def stop_camera_stream(self, stream_id: str) -> bool:
        """
        Stop a camera stream.
        
        Args:
            stream_id: ID of the stream to stop
            
        Returns:
            bool: True if stop successful, False otherwise
        """
        with self._stream_lock:
            return self._cleanup_stream(stream_id)
    
    def _cleanup_stream(self, stream_id: str) -> bool:
        """
        Internal method to cleanup and stop a stream.
        Must be called with _stream_lock held.
        """
        stream = self.active_streams.get(stream_id)
        if not stream:
            return False
        
        stream.stop()
        del self.active_streams[stream_id]
        return True
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active streams.
        
        Returns:
            Dict: Stream ID -> stream info
        """
        with self._stream_lock:
            stream_info = {}
            for stream_id, stream in self.active_streams.items():
                stream_info[stream_id] = {
                    'camera_id': stream.config.camera_id,
                    'resolution': stream.config.resolution,
                    'fps': stream.config.fps,
                    'status': stream.get_status().value,
                    'frame_count': stream.frame_number,
                    'is_active': stream.is_active()
                }
            return stream_info
    
    def get_stream_last_frame(self, stream_id: str) -> Optional[FrameData]:
        """
        Get the last frame from a specific stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Optional[FrameData]: Last frame data if available
        """
        with self._stream_lock:
            stream = self.active_streams.get(stream_id)
            if stream:
                return stream.get_last_frame()
            return None
    
    def push_annotated_frame(self, stream_id: str, annotated_frame: np.ndarray, 
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Push an annotated frame back to stream subscribers.
        This allows AIVisionAgent to send processed frames back through the stream.
        
        Args:
            stream_id: ID of the stream
            annotated_frame: Processed/annotated frame
            metadata: Additional metadata for the annotated frame
            
        Returns:
            bool: True if push successful, False otherwise
        """
        with self._stream_lock:
            stream = self.active_streams.get(stream_id)
            if not stream or not stream.is_active():
                return False
            
            # Create annotated frame data
            annotated_frame_data = FrameData(
                frame=annotated_frame,
                timestamp=time.time(),
                frame_number=stream.frame_number,  # Keep sync with original stream
                camera_id=stream.config.camera_id,
                metadata={
                    'annotated': True,
                    'original_metadata': metadata or {},
                    'resolution': stream.config.resolution,
                    'fps': stream.config.fps
                }
            )
            
            # Notify callbacks with annotated frame
            stream._notify_frame_callbacks(annotated_frame_data)
            return True
    
    def cleanup_all_streams(self) -> None:
        """
        Stop and cleanup all active streams.
        Useful for graceful shutdown.
        """
        with self._stream_lock:
            stream_ids = list(self.active_streams.keys())
            for stream_id in stream_ids:
                self._cleanup_stream(stream_id)

    def get_local_audio_devices(self) -> List[Dict]:
        devices = []
        # Microphones and speakers
        for idx, dev in enumerate(sd.query_devices()):
            devices.append({
                'index': idx,
                'name': dev['name'],
                'type': 'input' if dev['max_input_channels'] > 0 else 'output',
                'max_input_channels': dev['max_input_channels'],
                'max_output_channels': dev['max_output_channels'],
                'default_samplerate': dev['default_samplerate'],
            })
        return devices

    def get_local_cameras(self, max_test=5) -> List[Dict]:
        cameras = []
        for idx in range(max_test):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                cameras.append({'index': idx, 'name': f'Camera {idx}', 'type': 'camera'})
                cap.release()
        return cameras

    def discover_network_devices(self, network_range=None) -> List[Dict]:
        if not network_range:
            local_ip = self.get_local_ip()
            network_range = f"{'.'.join(local_ip.split('.')[:-1])}.0/24"
        self.nm.scan(hosts=network_range, arguments='-sn')
        hosts_list = [(x, self.nm[x]['status']['state']) for x in self.nm.all_hosts()]
        return [{'ip': host, 'status': status} for host, status in hosts_list]

    def scan_device_ports(self, ip_address, ports=[80, 554, 8080, 8554]) -> Dict:
        self.nm.scan(hosts=ip_address, arguments=f'-p {" ".join(map(str, ports))} --open')
        return self.nm[ip_address]

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def unified_device_list(self) -> List[Dict]:
        devices = []
        devices.extend(self.get_local_audio_devices())
        devices.extend(self.get_local_cameras())
        # Optionally add network devices
        devices.extend(self.discover_network_devices())
        return devices

    def control_device(self, device: Dict, action: str) -> str:
        # Robustly handle devices with or without 'type' key
        device_type = device.get('type')
        if not device_type:
            return f"Device {device.get('name', device.get('ip', 'Unknown'))} does not support action '{action}' (no type info)"
        if device_type == 'input' and action == 'mute':
            return f"Microphone {device['name']} muted (simulated)"
        if device_type == 'camera' and action == 'on':
            return f"Camera {device['name']} turned on (simulated)"
        return f"Action {action} not implemented for {device_type}"

    def capture_camera_frame(self, camera_index: int = 0, save_path: str = None):
        """
        Capture a single frame from the specified camera. Optionally save to file.
        Returns the frame (numpy array) or None if failed.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Camera {camera_index} could not be opened.")
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to capture frame from camera {camera_index}.")
            return None
        if save_path:
            cv2.imwrite(save_path, frame)
        return frame

    def list_camera_capabilities(self, camera_index: int = 0) -> Dict:
        """
        List basic capabilities of the specified camera (resolutions, FPS, etc.).
        Returns a dict of properties.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {"error": f"Camera {camera_index} could not be opened."}
        capabilities = {
            "frame_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "frame_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": cap.get(cv2.CAP_PROP_FOURCC),
            "format": cap.get(cv2.CAP_PROP_FORMAT),
        }
        cap.release()
        return capabilities

if __name__ == "__main__":
    dm = DeviceManager()
    devices = dm.unified_device_list()
    print("Unified device list:", devices)
    for device in devices:
        print(dm.control_device(device, 'mute' if device.get('type') == 'input' else 'on'))
    
    # Vision mode demo: capture a frame from the first camera (if available)
    cameras = [d for d in devices if d.get('type') == 'camera']
    if cameras:
        cam_idx = cameras[0]['index']
        frame = dm.capture_camera_frame(cam_idx, save_path=f'camera_{cam_idx}_frame.jpg')
        if frame is not None:
            print(f"Captured frame from camera {cam_idx} and saved to camera_{cam_idx}_frame.jpg")
        else:
            print(f"Failed to capture frame from camera {cam_idx}")
        print("Camera capabilities:", dm.list_camera_capabilities(cam_idx))
        
        # Streaming demo
        print("\nStreaming demo:")
        
        # Define callback functions for AIVisionAgent integration
        def frame_callback(frame_data: FrameData):
            print(f"Received frame {frame_data.frame_number} from camera {frame_data.camera_id} at {frame_data.timestamp}")
            # This is where AIVisionAgent would process the frame
            # For demo, just add a simple annotation
            annotated_frame = frame_data.frame.copy()
            cv2.putText(annotated_frame, f"Frame {frame_data.frame_number}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Push the annotated frame back
            dm.push_annotated_frame(stream_id, annotated_frame, {'annotation': 'frame_counter'})
        
        def status_callback(status: StreamStatus, message: str):
            print(f"Stream status: {status.value} - {message}")
        
        # Create a camera stream
        stream_id = dm.get_camera_stream(cam_idx, (640, 480), 10)
        if stream_id:
            print(f"Created stream: {stream_id}")
            
            # Subscribe to the stream (this is what AIVisionAgent would do)
            success = dm.subscribe_to_stream(stream_id, frame_callback, status_callback)
            if success:
                print("Successfully subscribed to stream")
                
                # Let it run for a bit
                import time
                time.sleep(3)
                
                # Show active streams
                print("Active streams:", dm.get_active_streams())
                
                # Cleanup
                dm.unsubscribe_from_stream(stream_id, frame_callback, status_callback)
                dm.stop_camera_stream(stream_id)
                print("Cleaned up streaming demo")
            else:
                print("Failed to subscribe to stream")
        else:
            print("Failed to create camera stream")
    else:
        print("No cameras found for vision mode demo.")
