# components/camera_integration/camera_manager.py

import cv2
import numpy as np
import requests
import socket
import threading
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule
import os
from pathlib import Path

# Import vision components
try:
    from ..ai_vision.object_detection import detect_objects
    from ..ai_vision.face_recognition import create_face_recognizer
    from ..ai_vision.image_classification import classify_image
except ImportError:
    # Fallback for direct execution
    from components.ai_vision.object_detection import detect_objects
    from components.ai_vision.face_recognition import create_face_recognizer
    from components.ai_vision.image_classification import classify_image

logger = logging.getLogger(__name__)

# Import error handling
try:
    from ...error_handling import (
        with_camera_error_handling, with_vision_error_handling, vision_error_handler,
        CameraTimeoutError, CameraConnectionError, CameraNotFoundError, 
        ImageProcessingError, NetworkError
    )
except ImportError:
    try:
        from ..error_handling import (
            with_camera_error_handling, with_vision_error_handling, vision_error_handler,
            CameraTimeoutError, CameraConnectionError, CameraNotFoundError, 
            ImageProcessingError, NetworkError
        )
    except ImportError:
        from error_handling import (
            with_camera_error_handling, with_vision_error_handling, vision_error_handler,
            CameraTimeoutError, CameraConnectionError, CameraNotFoundError, 
            ImageProcessingError, NetworkError
        )

@dataclass
class CameraCredentials:
    """Camera authentication credentials"""
    username: str
    password: str
    ip_address: str
    port: int = 80
    rtsp_port: int = 554
    http_port: int = 80

@dataclass
class CameraCapabilities:
    """Camera feature capabilities"""
    has_ptz: bool = False
    has_360_view: bool = False
    has_motion_detection: bool = False
    has_motion_tracking: bool = False
    has_privacy_mode: bool = False
    has_ir_lights: bool = False
    has_night_vision: bool = False
    max_resolution: Tuple[int, int] = (1920, 1080)
    supported_formats: List[str] = field(default_factory=lambda: ['H264', 'MJPEG'])

@dataclass
class MotionEvent:
    """Motion detection event data"""
    timestamp: datetime
    camera_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame: np.ndarray
    objects_detected: List[Dict[str, Any]] = field(default_factory=list)

@dataclass 
class CameraConfig:
    """Camera configuration settings"""
    camera_id: str
    name: str
    credentials: CameraCredentials
    capabilities: CameraCapabilities
    recording_schedule: Dict[str, Any] = field(default_factory=dict)
    motion_detection_enabled: bool = True
    privacy_mode_schedule: Dict[str, Any] = field(default_factory=dict)
    ai_analysis_enabled: bool = True
    notification_settings: Dict[str, Any] = field(default_factory=dict)

class NetworkScanner:
    """Network scanner to discover CP Plus cameras"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NetworkScanner")
        
    def get_network_range(self) -> str:
        """Get local network range"""
        try:
            # Get default gateway
            result = subprocess.run(['route', 'print'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if '0.0.0.0' in line and 'Gateway' not in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        gateway = parts[2]
                        # Convert to network range (assuming /24)
                        network_parts = gateway.split('.')
                        network_base = '.'.join(network_parts[:3])
                        return f"{network_base}.0/24"
            
            # Fallback to common ranges
            return "192.168.1.0/24"
            
        except Exception as e:
            self.logger.error(f"Failed to detect network range: {e}")
            return "192.168.1.0/24"
    
    def scan_for_cameras(self, network_range: str = None, ports: List[int] = None) -> List[Dict[str, Any]]:
        """Scan network for CP Plus cameras"""
        if network_range is None:
            network_range = self.get_network_range()
        
        if ports is None:
            ports = [80, 8080, 554, 8554, 443]  # Common camera ports
        
        self.logger.info(f"Scanning network range: {network_range}")
        
        # Generate IP range
        base_ip = network_range.split('/')[0].rsplit('.', 1)[0]
        ip_range = [f"{base_ip}.{i}" for i in range(1, 255)]
        
        cameras_found = []
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            # Submit scan tasks for all IP/port combinations
            future_to_ip_port = {}
            for ip in ip_range:
                for port in ports:
                    future = executor.submit(self._scan_ip_port, ip, port)
                    future_to_ip_port[future] = (ip, port)
            
            # Collect results
            for future in as_completed(future_to_ip_port):
                ip, port = future_to_ip_port[future]
                try:
                    result = future.result(timeout=2)
                    if result:
                        cameras_found.append(result)
                except Exception as e:
                    pass  # Ignore timeouts and connection errors
        
        self.logger.info(f"Found {len(cameras_found)} potential cameras")
        return cameras_found
    
    def _scan_ip_port(self, ip: str, port: int) -> Optional[Dict[str, Any]]:
        """Scan specific IP and port for camera"""
        try:
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                # Port is open, try to identify if it's a camera
                return self._identify_camera(ip, port)
                
        except Exception:
            pass
        
        return None
    
    def _identify_camera(self, ip: str, port: int) -> Optional[Dict[str, Any]]:
        """Try to identify if device is a CP Plus camera"""
        try:
            # Try common camera HTTP endpoints
            endpoints_to_try = [
                f"http://{ip}:{port}/",
                f"http://{ip}:{port}/cgi-bin/hi3510/param.cgi?cmd=getserverinfo",
                f"http://{ip}:{port}/onvif/device_service",
                f"http://{ip}:{port}/ISAPI/System/deviceInfo",
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    response = requests.get(endpoint, timeout=2)
                    content = response.text.lower()
                    
                    # Check for CP Plus indicators
                    cp_plus_indicators = ['cp-plus', 'cpplus', 'cp plus', 'cp-e305a']
                    
                    if any(indicator in content for indicator in cp_plus_indicators):
                        return {
                            'ip': ip,
                            'port': port,
                            'manufacturer': 'CP Plus',
                            'model': 'CP-E305A',
                            'endpoint': endpoint,
                            'response_sample': content[:200]
                        }
                    
                    # Check for generic camera indicators
                    camera_indicators = ['camera', 'ipcam', 'webcam', 'rtsp', 'onvif']
                    if any(indicator in content for indicator in camera_indicators):
                        return {
                            'ip': ip,
                            'port': port,
                            'manufacturer': 'Unknown',
                            'model': 'Generic Camera',
                            'endpoint': endpoint,
                            'response_sample': content[:200]
                        }
                        
                except requests.RequestException:
                    continue
                    
        except Exception as e:
            pass
        
        return None

class CPPlusCameraController:
    """Controller for CP Plus CP-E305A camera"""
    
    def __init__(self, credentials: CameraCredentials):
        self.credentials = credentials
        self.logger = logging.getLogger(f"{__name__}.CPPlusCameraController")
        self.session = requests.Session()
        
        # CP Plus specific endpoints
        self.base_url = f"http://{credentials.ip_address}:{credentials.http_port}"
        self.rtsp_url = f"rtsp://{credentials.username}:{credentials.password}@{credentials.ip_address}:{credentials.rtsp_port}/cam/realmonitor?channel=1&subtype=0"
        
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with camera"""
        try:
            # Try basic auth
            self.session.auth = (self.credentials.username, self.credentials.password)
            
            # Test authentication with device info request
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=getserverinfo", timeout=5)
            
            if response.status_code == 200:
                self.logger.info(f"Successfully authenticated with camera at {self.credentials.ip_address}")
                return True
            else:
                self.logger.warning(f"Authentication failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get camera device information"""
        try:
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=getserverinfo")
            if response.status_code == 200:
                # Parse the response (format may vary)
                return {
                    'status': 'success',
                    'model': 'CP-E305A',
                    'manufacturer': 'CP Plus',
                    'response': response.text
                }
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
        
        return {'status': 'error', 'message': 'Failed to retrieve device info'}
    
    @with_camera_error_handling("camera_stream", "start_video_stream", timeout=10.0)
    def start_video_stream(self) -> cv2.VideoCapture:
        """Start video stream from camera"""
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            if cap.isOpened():
                self.logger.info("Video stream started successfully")
                return cap
            else:
                error = CameraConnectionError(
                    self.credentials.ip_address, 
                    "Failed to open video stream - check RTSP URL and credentials"
                )
                vision_error_handler.handle_vision_error(error, {
                    'component': 'CPPlusCameraController',
                    'operation': 'start_video_stream',
                    'camera_ip': self.credentials.ip_address,
                    'rtsp_url': self.rtsp_url
                })
                raise error
        except Exception as e:
            if not isinstance(e, CameraConnectionError):
                error = CameraConnectionError(
                    self.credentials.ip_address,
                    f"Error starting video stream: {str(e)}"
                )
                vision_error_handler.handle_vision_error(error, {
                    'component': 'CPPlusCameraController',
                    'operation': 'start_video_stream',
                    'camera_ip': self.credentials.ip_address,
                    'original_error': str(e)
                })
                raise error
            raise
    
    def enable_motion_detection(self) -> bool:
        """Enable motion detection on camera"""
        try:
            # CP Plus specific motion detection command
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=setmotiondetect&-enable=1")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to enable motion detection: {e}")
            return False
    
    def disable_motion_detection(self) -> bool:
        """Disable motion detection on camera"""
        try:
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=setmotiondetect&-enable=0")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to disable motion detection: {e}")
            return False
    
    def enable_privacy_mode(self) -> bool:
        """Enable privacy mode (lens cover)"""
        try:
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=setlenscover&-enable=1")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to enable privacy mode: {e}")
            return False
    
    def disable_privacy_mode(self) -> bool:
        """Disable privacy mode"""
        try:
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=setlenscover&-enable=0")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to disable privacy mode: {e}")
            return False
    
    def set_ir_lights(self, enabled: bool) -> bool:
        """Control IR lights for night vision"""
        try:
            state = 1 if enabled else 0
            response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/param.cgi?cmd=setinfrared&-enable={state}")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to set IR lights: {e}")
            return False
    
    def ptz_control(self, action: str, speed: int = 5) -> bool:
        """Control pan/tilt/zoom (360 view control)"""
        try:
            # CP Plus PTZ commands
            ptz_commands = {
                'up': 'tiltup',
                'down': 'tiltdown', 
                'left': 'panleft',
                'right': 'panright',
                'zoom_in': 'zoomin',
                'zoom_out': 'zoomout',
                'stop': 'ptzstop'
            }
            
            if action in ptz_commands:
                cmd = ptz_commands[action]
                response = self.session.get(f"{self.base_url}/cgi-bin/hi3510/ptzctrl.cgi?-step=0&-act={cmd}&-speed={speed}")
                return response.status_code == 200
            
            return False
        except Exception as e:
            self.logger.error(f"PTZ control failed: {e}")
            return False

class CameraManager:
    """Main camera management system"""
    
    def __init__(self, storage_path: str = "camera_data"):
        self.logger = logging.getLogger(f"{__name__}.CameraManager")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.cameras: Dict[str, CPPlusCameraController] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
        self.vision_processors = {
            'face_recognizer': create_face_recognizer(),
            'object_detector': None,  # Will be initialized when needed
            'classifier': None
        }
        
        self.recording_threads: Dict[str, threading.Thread] = {}
        self.motion_detection_threads: Dict[str, threading.Thread] = {}
        self.privacy_schedules: Dict[str, Any] = {}
        
        self.network_scanner = NetworkScanner()
        self._setup_scheduling()
    
    def _setup_scheduling(self):
        """Setup automated scheduling"""
        schedule.every(1).minutes.do(self._check_privacy_schedules)
        schedule.every(5).minutes.do(self._check_recording_schedules)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run scheduled tasks"""
        while True:
            schedule.run_pending()
            time.sleep(30)
    
    def discover_cameras(self) -> List[Dict[str, Any]]:
        """Discover CP Plus cameras on network"""
        return self.network_scanner.scan_for_cameras()
    
    def add_camera(self, camera_config: CameraConfig) -> bool:
        """Add and configure a camera"""
        try:
            camera_id = camera_config.camera_id
            
            # Create camera controller
            controller = CPPlusCameraController(camera_config.credentials)
            
            # Test connection
            device_info = controller.get_device_info()
            if device_info.get('status') != 'success':
                self.logger.error(f"Failed to connect to camera {camera_id}")
                return False
            
            self.cameras[camera_id] = controller
            self.camera_configs[camera_id] = camera_config
            
            # Setup camera features
            if camera_config.motion_detection_enabled:
                controller.enable_motion_detection()
                self.start_motion_detection(camera_id)
            
            # Setup AI analysis
            if camera_config.ai_analysis_enabled:
                self.start_ai_monitoring(camera_id)
            
            self.logger.info(f"Camera {camera_id} added successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add camera {camera_config.camera_id}: {e}")
            return False
    
    def start_motion_detection(self, camera_id: str):
        """Start motion detection for a camera"""
        if camera_id not in self.cameras:
            return False
        
        def motion_detection_worker():
            controller = self.cameras[camera_id]
            cap = controller.start_video_stream()
            
            if not cap:
                return
            
            # Initialize background subtractor
            backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply background subtraction
                fgMask = backSub.apply(frame)
                
                # Find contours
                contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area
                motion_detected = False
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum area threshold
                        motion_detected = True
                        
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Create motion event
                        motion_event = MotionEvent(
                            timestamp=datetime.now(),
                            camera_id=camera_id,
                            bbox=(x, y, w, h),
                            confidence=min(area / 10000, 1.0),
                            frame=frame.copy()
                        )
                        
                        # Process with AI if enabled
                        if self.camera_configs[camera_id].ai_analysis_enabled:
                            self._process_motion_with_ai(motion_event)
                        
                        # Send notifications
                        self._send_motion_notification(motion_event)
                        
                        break
                
                time.sleep(0.1)  # Control processing rate
            
            cap.release()
        
        thread = threading.Thread(target=motion_detection_worker, daemon=True)
        thread.start()
        self.motion_detection_threads[camera_id] = thread
    
    def _process_motion_with_ai(self, motion_event: MotionEvent):
        """Process motion event with AI vision"""
        try:
            frame = motion_event.frame
            
            # Object detection
            detections = detect_objects(frame, confidence_threshold=0.5)
            
            objects_detected = []
            for det in detections:
                objects_detected.append({
                    'class_name': det.class_name,
                    'confidence': float(det.confidence),
                    'bbox': [int(x) for x in det.bbox]
                })
            
            motion_event.objects_detected = objects_detected
            
            # Face recognition if people detected
            person_detected = any(obj['class_name'] == 'person' for obj in objects_detected)
            if person_detected and self.vision_processors['face_recognizer']:
                faces = self.vision_processors['face_recognizer'].recognize_faces(frame)
                
                face_data = []
                for face in faces:
                    face_data.append({
                        'name': face.name or 'Unknown',
                        'confidence': float(face.confidence)
                    })
                
                motion_event.metadata = {'faces': face_data}
            
        except Exception as e:
            self.logger.error(f"AI processing failed for motion event: {e}")
    
    def start_ai_monitoring(self, camera_id: str):
        """Start continuous AI monitoring of camera feed"""
        if camera_id not in self.cameras:
            return False
        
        def ai_monitoring_worker():
            controller = self.cameras[camera_id]
            cap = controller.start_video_stream()
            
            if not cap:
                return
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame to reduce load
                frame_count += 1
                if frame_count % 10 != 0:
                    continue
                
                try:
                    # Object detection
                    detections = detect_objects(frame, confidence_threshold=0.3)
                    
                    # Log interesting objects
                    interesting_objects = ['person', 'car', 'truck', 'bicycle', 'motorcycle']
                    for det in detections:
                        if det.class_name in interesting_objects:
                            self.logger.info(f"Camera {camera_id} detected: {det.class_name} ({det.confidence:.2f})")
                    
                    # Face recognition for people
                    person_detected = any(det.class_name == 'person' for det in detections)
                    if person_detected:
                        faces = self.vision_processors['face_recognizer'].recognize_faces(frame)
                        for face in faces:
                            if face.name:
                                self.logger.info(f"Camera {camera_id} recognized: {face.name} ({face.confidence:.2f})")
                
                except Exception as e:
                    pass  # Continue monitoring even if AI processing fails
                
                time.sleep(1)  # Process roughly once per second
            
            cap.release()
        
        thread = threading.Thread(target=ai_monitoring_worker, daemon=True)
        thread.start()
    
    def _send_motion_notification(self, motion_event: MotionEvent):
        """Send motion detection notification"""
        try:
            # Save motion frame
            timestamp_str = motion_event.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"motion_{motion_event.camera_id}_{timestamp_str}.jpg"
            filepath = self.storage_path / filename
            
            cv2.imwrite(str(filepath), motion_event.frame)
            
            # Create notification data
            notification = {
                'type': 'motion_detected',
                'camera_id': motion_event.camera_id,
                'timestamp': motion_event.timestamp.isoformat(),
                'confidence': motion_event.confidence,
                'bbox': motion_event.bbox,
                'objects_detected': motion_event.objects_detected,
                'image_path': str(filepath)
            }
            
            # Log notification (could be extended to send to notification service)
            self.logger.info(f"Motion notification: {json.dumps(notification, indent=2)}")
            
            # TODO: Send to OS notification system, email, webhook, etc.
            
        except Exception as e:
            self.logger.error(f"Failed to send motion notification: {e}")
    
    def set_privacy_schedule(self, camera_id: str, schedule_config: Dict[str, Any]):
        """Set privacy mode schedule for a camera"""
        self.privacy_schedules[camera_id] = schedule_config
    
    def _check_privacy_schedules(self):
        """Check and apply privacy schedules"""
        current_time = datetime.now()
        
        for camera_id, schedule_config in self.privacy_schedules.items():
            if camera_id not in self.cameras:
                continue
            
            controller = self.cameras[camera_id]
            
            # Check if current time falls within privacy hours
            privacy_enabled = self._is_privacy_time(current_time, schedule_config)
            
            # Apply privacy mode
            if privacy_enabled:
                controller.enable_privacy_mode()
            else:
                controller.disable_privacy_mode()
    
    def _is_privacy_time(self, current_time: datetime, schedule_config: Dict[str, Any]) -> bool:
        """Check if current time requires privacy mode"""
        # Implement privacy schedule logic based on configuration
        # This is a simplified version - can be extended for complex schedules
        
        privacy_hours = schedule_config.get('privacy_hours', [])
        current_hour = current_time.hour
        
        return current_hour in privacy_hours
    
    def _check_recording_schedules(self):
        """Check and manage recording schedules"""
        # TODO: Implement scheduled recording functionality
        pass
    
    def get_camera_status(self, camera_id: str) -> Dict[str, Any]:
        """Get current status of a camera"""
        if camera_id not in self.cameras:
            return {'error': 'Camera not found'}
        
        controller = self.cameras[camera_id]
        config = self.camera_configs[camera_id]
        
        return {
            'camera_id': camera_id,
            'name': config.name,
            'ip_address': config.credentials.ip_address,
            'status': 'online',  # Could check actual connectivity
            'motion_detection_enabled': config.motion_detection_enabled,
            'ai_analysis_enabled': config.ai_analysis_enabled,
            'privacy_mode_active': False,  # Could check actual state
            'capabilities': config.capabilities
        }
    
    def control_camera(self, camera_id: str, action: str, **kwargs) -> bool:
        """Control camera functions"""
        if camera_id not in self.cameras:
            return False
        
        controller = self.cameras[camera_id]
        
        if action == 'enable_privacy_mode':
            return controller.enable_privacy_mode()
        elif action == 'disable_privacy_mode':
            return controller.disable_privacy_mode()
        elif action == 'enable_motion_detection':
            return controller.enable_motion_detection()
        elif action == 'disable_motion_detection':
            return controller.disable_motion_detection()
        elif action == 'set_ir_lights':
            enabled = kwargs.get('enabled', True)
            return controller.set_ir_lights(enabled)
        elif action == 'ptz_control':
            direction = kwargs.get('direction', 'stop')
            speed = kwargs.get('speed', 5)
            return controller.ptz_control(direction, speed)
        
        return False

# Example usage and setup
def setup_camera_integration():
    """Setup example for CP Plus camera integration"""
    
    # Initialize camera manager
    manager = CameraManager(storage_path="camera_recordings")
    
    # Discover cameras on network
    discovered_cameras = manager.discover_cameras()
    print(f"Discovered cameras: {json.dumps(discovered_cameras, indent=2)}")
    
    # If you found your camera and know credentials, add it
    if discovered_cameras:
        camera_ip = discovered_cameras[0]['ip']  # Use first found camera
        
        # Create camera configuration
        credentials = CameraCredentials(
            username="admin",  # Try common defaults
            password="admin",  # You'll need to find/set the actual password
            ip_address=camera_ip
        )
        
        capabilities = CameraCapabilities(
            has_360_view=True,
            has_motion_detection=True,
            has_motion_tracking=True,
            has_privacy_mode=True,
            has_ir_lights=True,
            has_night_vision=True,
            max_resolution=(1920, 1080)
        )
        
        config = CameraConfig(
            camera_id="cp_plus_main",
            name="CP Plus Main Camera",
            credentials=credentials,
            capabilities=capabilities,
            motion_detection_enabled=True,
            ai_analysis_enabled=True
        )
        
        # Add camera to manager
        success = manager.add_camera(config)
        if success:
            print("Camera added successfully!")
            
            # Set privacy schedule (privacy mode during specific hours)
            privacy_schedule = {
                'privacy_hours': [22, 23, 0, 1, 2, 3, 4, 5, 6]  # 10 PM to 6 AM
            }
            manager.set_privacy_schedule("cp_plus_main", privacy_schedule)
            
        else:
            print("Failed to add camera - check credentials and connectivity")
    
    return manager

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run camera integration setup
    camera_manager = setup_camera_integration()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
            print("Camera integration running...")
    except KeyboardInterrupt:
        print("Shutting down camera integration...")
