"""
Comprehensive Vision Error Handling Integration Demo

This example demonstrates the complete integration of error handling
throughout the vision processing pipeline with retry logic and 
custom exceptions.
"""

import asyncio
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import error handling components
from error_handling import (
    vision_error_handler, with_vision_error_handling, with_model_error_handling,
    with_camera_error_handling, with_image_processing_error_handling,
    ModelLoadError, CameraConnectionError, CameraTimeoutError,
    ImageProcessingError, ModelInferenceError, RetryConfig
)

# Import vision components
from components.ai_vision.model_hub import ModelHub, DeviceType
from components.camera_integration.camera_manager import (
    CameraManager, CameraConfig, CameraCredentials, CameraCapabilities
)
from components.ai_vision.object_detection import detect_objects
from components.ai_vision.face_recognition import create_face_recognizer
from components.ai_vision.image_classification import classify_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionPipelineDemo:
    """Demonstration of comprehensive vision pipeline with error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_hub = None
        self.camera_manager = None
        
        # Initialize with error handling
        self._initialize_components()
    
    @with_vision_error_handling(
        component="VisionPipelineDemo",
        operation="initialization",
        retry_config=RetryConfig(max_attempts=3, base_delay=2.0)
    )
    def _initialize_components(self):
        """Initialize vision components with error handling."""
        try:
            # Initialize model hub
            self.logger.info("Initializing Model Hub...")
            self.model_hub = ModelHub(
                cache_dir=Path("./models_cache"),
                max_cache_size_gb=5.0
            )
            
            # Initialize camera manager
            self.logger.info("Initializing Camera Manager...")
            self.camera_manager = CameraManager(storage_path="./camera_data")
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    @with_model_error_handling(
        model_name="yolov8n",
        operation="download_and_load",
        retry_config=RetryConfig(max_attempts=2, base_delay=5.0)
    )
    async def prepare_models(self):
        """Download and load required models with error handling."""
        try:
            self.logger.info("Downloading YOLOv8 model...")
            
            # Download model with error handling
            success = await self.model_hub.download_model("yolov8n")
            if not success:
                raise ModelLoadError("yolov8n", "Failed to download model")
            
            # Load model with error handling
            self.logger.info("Loading YOLOv8 model...")
            model = self.model_hub.load_model("yolov8n", DeviceType.AUTO)
            if model is None:
                raise ModelLoadError("yolov8n", "Failed to load model")
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Model preparation failed: {e}")
            raise
    
    @with_camera_error_handling(
        camera_id="demo_camera",
        operation="setup",
        timeout=30.0,
        retry_config=RetryConfig(max_attempts=3, base_delay=2.0)
    )
    def setup_demo_camera(self, camera_ip: str = "192.168.1.100") -> bool:
        """Setup demo camera with comprehensive error handling."""
        try:
            # Create camera configuration
            credentials = CameraCredentials(
                username="admin",
                password="admin123",
                ip_address=camera_ip
            )
            
            capabilities = CameraCapabilities(
                has_motion_detection=True,
                has_privacy_mode=True,
                has_ir_lights=True,
                max_resolution=(1920, 1080)
            )
            
            config = CameraConfig(
                camera_id="demo_camera",
                name="Demo Camera",
                credentials=credentials,
                capabilities=capabilities,
                motion_detection_enabled=True,
                ai_analysis_enabled=True
            )
            
            # Add camera with error handling
            success = self.camera_manager.add_camera(config)
            if not success:
                raise CameraConnectionError(
                    camera_ip,
                    "Failed to add camera to manager"
                )
            
            self.logger.info(f"Camera {camera_ip} setup successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera setup failed: {e}")
            raise
    
    @with_image_processing_error_handling(
        operation="process_frame",
        retry_config=RetryConfig(max_attempts=2, base_delay=1.0)
    )
    def process_camera_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process camera frame with comprehensive error handling."""
        try:
            if frame is None or frame.size == 0:
                raise ImageProcessingError(
                    "process_frame",
                    "empty_frame",
                    "Received empty or None frame"
                )
            
            results = {}
            
            # Object detection with error handling
            try:
                self.logger.debug("Running object detection...")
                detections = detect_objects(frame, confidence_threshold=0.5)
                
                results['objects'] = []
                for det in detections:
                    results['objects'].append({
                        'class_name': det.class_name,
                        'confidence': float(det.confidence),
                        'bbox': [int(x) for x in det.bbox]
                    })
                
                self.logger.debug(f"Detected {len(results['objects'])} objects")
                
            except Exception as e:
                vision_error_handler.handle_vision_error(e, {
                    'component': 'VisionPipelineDemo',
                    'operation': 'object_detection',
                    'frame_shape': frame.shape
                })
                results['objects'] = []
                results['object_detection_error'] = str(e)
            
            # Face recognition with error handling
            try:
                self.logger.debug("Running face recognition...")
                face_recognizer = create_face_recognizer()
                
                if face_recognizer:
                    faces = face_recognizer.recognize_faces(frame)
                    
                    results['faces'] = []
                    for face in faces:
                        results['faces'].append({
                            'name': face.name or 'Unknown',
                            'confidence': float(face.confidence)
                        })
                    
                    self.logger.debug(f"Detected {len(results['faces'])} faces")
                else:
                    results['faces'] = []
                    results['face_recognition_error'] = "Face recognizer not available"
                
            except Exception as e:
                vision_error_handler.handle_vision_error(e, {
                    'component': 'VisionPipelineDemo',
                    'operation': 'face_recognition',
                    'frame_shape': frame.shape
                })
                results['faces'] = []
                results['face_recognition_error'] = str(e)
            
            # Image classification with error handling
            try:
                self.logger.debug("Running image classification...")
                classification = classify_image(frame, top_k=3)
                
                if classification:
                    results['classification'] = {
                        'predictions': [
                            {
                                'class_name': pred.class_name,
                                'confidence': float(pred.confidence)
                            }
                            for pred in classification[:3]
                        ]
                    }
                else:
                    results['classification'] = {'predictions': []}
                
            except Exception as e:
                vision_error_handler.handle_vision_error(e, {
                    'component': 'VisionPipelineDemo',
                    'operation': 'image_classification',
                    'frame_shape': frame.shape
                })
                results['classification'] = {'predictions': []}
                results['classification_error'] = str(e)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            raise
    
    @with_vision_error_handling(
        component="VisionPipelineDemo",
        operation="simulate_camera_timeout",
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0)
    )
    def simulate_camera_timeout_scenario(self):
        """Simulate camera timeout for demonstration."""
        import time
        import random
        
        # Simulate random timeout
        if random.random() < 0.3:  # 30% chance of timeout
            time.sleep(12)  # Simulate long operation
            raise CameraTimeoutError(
                "demo_camera",
                10.0,
                "Simulated camera timeout for demonstration"
            )
        
        return "Camera operation completed successfully"
    
    @with_vision_error_handling(
        component="VisionPipelineDemo", 
        operation="simulate_model_error",
        retry_config=RetryConfig(max_attempts=2, base_delay=2.0)
    )
    def simulate_model_error_scenario(self):
        """Simulate model inference error for demonstration."""
        import random
        
        # Simulate random model error
        if random.random() < 0.4:  # 40% chance of error
            raise ModelInferenceError(
                "demo_model",
                (1, 3, 640, 640),
                "Simulated model inference error for demonstration"
            )
        
        return "Model inference completed successfully"
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of error handling."""
        self.logger.info("Starting comprehensive vision error handling demo")
        
        try:
            # Test model operations
            self.logger.info("=== Testing Model Operations ===")
            asyncio.run(self.prepare_models())
            
            # Test camera operations
            self.logger.info("=== Testing Camera Operations ===")
            try:
                self.setup_demo_camera()
            except CameraConnectionError as e:
                self.logger.warning(f"Camera setup failed (expected): {e}")
            
            # Test image processing with sample frame
            self.logger.info("=== Testing Image Processing ===")
            sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            try:
                results = self.process_camera_frame(sample_frame)
                self.logger.info(f"Processing results: {results}")
            except ImageProcessingError as e:
                self.logger.warning(f"Image processing error (handled): {e}")
            
            # Test error simulation scenarios
            self.logger.info("=== Testing Error Scenarios ===")
            
            # Simulate camera timeout
            for i in range(3):
                try:
                    result = self.simulate_camera_timeout_scenario()
                    self.logger.info(f"Camera timeout test {i+1}: {result}")
                except CameraTimeoutError as e:
                    self.logger.warning(f"Camera timeout handled: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")
            
            # Simulate model errors
            for i in range(3):
                try:
                    result = self.simulate_model_error_scenario()
                    self.logger.info(f"Model error test {i+1}: {result}")
                except ModelInferenceError as e:
                    self.logger.warning(f"Model error handled: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")
            
            # Show error analysis
            self.logger.info("=== Error Analysis ===")
            analysis = vision_error_handler.error_manager.analyze_error_patterns()
            self.logger.info(f"Error patterns: {analysis}")
            
            recommendations = vision_error_handler.error_manager.get_error_recommendations()
            self.logger.info(f"Recommendations: {recommendations}")
            
        except Exception as e:
            self.logger.error(f"Demo failed with error: {e}")
            vision_error_handler.handle_vision_error(e, {
                'component': 'VisionPipelineDemo',
                'operation': 'run_comprehensive_demo'
            })
        
        self.logger.info("Demo completed")
    
    def test_empty_frame_handling(self):
        """Test handling of empty frames."""
        self.logger.info("=== Testing Empty Frame Handling ===")
        
        test_frames = [
            None,
            np.array([]),
            np.zeros((0, 0, 3)),
            np.zeros((100, 100, 3))  # Valid frame
        ]
        
        for i, frame in enumerate(test_frames):
            try:
                results = self.process_camera_frame(frame)
                self.logger.info(f"Frame {i} processed successfully: {len(results)} results")
            except ImageProcessingError as e:
                self.logger.warning(f"Frame {i} processing failed (handled): {e}")
            except Exception as e:
                self.logger.error(f"Frame {i} unexpected error: {e}")


def main():
    """Main function to run the comprehensive demo."""
    try:
        demo = VisionPipelineDemo()
        
        # Run comprehensive error handling demonstration
        demo.run_comprehensive_demo()
        
        # Test specific error scenarios
        demo.test_empty_frame_handling()
        
        print("\n" + "="*50)
        print("Vision Error Handling Demo completed successfully!")
        print("Check the logs above to see how errors were handled.")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Demo initialization failed: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
