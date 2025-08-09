# src/components/ai_vision/object_detection.py

"""
Object Detection Module using Ultralytics YOLOv8

This module provides object detection capabilities using YOLOv8 models with support for:
- Single frame detection
- Batch image processing  
- Video streaming with OpenCV VideoCapture
- Configurable confidence thresholds
- GPU acceleration when available
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Iterator, Callable
from dataclasses import dataclass
import time
import numpy as np

# Core imports - handle both relative and absolute imports
try:
    from .core import VisionTask, VisionResult, TaskStatus
    from .model_hub import ModelHub, get_model_hub, ModelType, DeviceType
except ImportError:
    # Fallback for standalone execution
    try:
        from core import VisionTask, VisionResult, TaskStatus
        from model_hub import ModelHub, get_model_hub, ModelType, DeviceType
    except ImportError:
        # Create minimal fallback classes for standalone testing
        from typing import Any, Optional
        from abc import ABC, abstractmethod
        from enum import Enum
        
        class TaskStatus(Enum):
            PENDING = "pending"
            IN_PROGRESS = "in_progress" 
            COMPLETED = "completed"
            FAILED = "failed"
        
        class VisionResult:
            def __init__(self, task_id: str, status: TaskStatus, data: Any, confidence: float, 
                        metadata: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None):
                self.task_id = task_id
                self.status = status
                self.data = data
                self.confidence = confidence
                self.metadata = metadata
                self.error_message = error_message
        
        class VisionTask(ABC):
            def __init__(self, task_id: str, task_type: str, **kwargs):
                self.task_id = task_id
                self.task_type = task_type
                self.status = TaskStatus.PENDING
                self.result = None
                self.metadata = kwargs
                import logging
                self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            
            @abstractmethod
            async def execute(self, input_data: Any) -> VisionResult:
                pass
            
            @abstractmethod
            def validate_input(self, input_data: Any) -> bool:
                pass
        
        class DeviceType(Enum):
            CPU = "cpu"
            GPU = "cuda"
            AUTO = "auto"
        
        class ModelType(Enum):
            YOLO_V8 = "yolo_v8"
        
        class ModelHub:
            def __init__(self, cache_dir=None, max_cache_size_gb=10.0):
                pass
            
            async def ensure_model_ready(self, model_name: str, device_type=None):
                return None
        
        def get_model_hub():
            return ModelHub()

# Lazy imports for heavy dependencies
_cv2 = None
_torch = None
_ultralytics = None


def _lazy_import_cv2():
    """Lazy import OpenCV"""
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as _cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required for object detection. "
                "Install with: pip install opencv-python"
            )
    return _cv2


def _lazy_import_torch():
    """Lazy import PyTorch"""
    global _torch
    if _torch is None:
        try:
            import torch as _torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for YOLOv8. "
                "Install with: pip install torch torchvision"
            )
    return _torch


def _lazy_import_ultralytics():
    """Lazy import Ultralytics"""
    global _ultralytics
    if _ultralytics is None:
        try:
            import ultralytics as _ultralytics
            from ultralytics import YOLO
            _ultralytics.YOLO = YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is required for YOLOv8. "
                "Install with: pip install ultralytics"
            )
    return _ultralytics


@dataclass
class DetectionResult:
    """Container for object detection results"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }


@dataclass
class BatchDetectionResult:
    """Container for batch detection results"""
    detections: List[List[DetectionResult]]  # List of detections per image
    processing_time: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'detections': [[det.to_dict() for det in img_dets] for img_dets in self.detections],
            'processing_time': self.processing_time,
            'batch_size': self.batch_size
        }


class ObjectDetectionTask(VisionTask):
    """Vision task implementation for object detection"""
    
    def __init__(self, task_id: str, model_name: str = "yolov8n", **kwargs):
        super().__init__(task_id, "object_detection", **kwargs)
        self.model_name = model_name
        self.model = None
        self.model_hub = get_model_hub()
        
    async def execute(self, input_data: Any) -> VisionResult:
        """Execute object detection task"""
        try:
            if not self.model:
                await self._ensure_model_loaded()
            
            # Handle different input types
            if isinstance(input_data, dict):
                frame = input_data.get('frame')
                conf_threshold = input_data.get('conf_threshold', 0.25)
                batch_frames = input_data.get('batch_frames')
                
                if batch_frames is not None:
                    # Batch processing
                    detections = detect_objects_batch(batch_frames, self.model, conf_threshold)
                    return VisionResult(
                        task_id=self.task_id,
                        status=TaskStatus.COMPLETED,
                        data=detections.to_dict(),
                        confidence=1.0,
                        metadata={'processing_type': 'batch'}
                    )
                elif frame is not None:
                    # Single frame processing
                    detections = detect_objects(frame, self.model, conf_threshold)
                    return VisionResult(
                        task_id=self.task_id,
                        status=TaskStatus.COMPLETED,
                        data=[det.to_dict() for det in detections],
                        confidence=max([det.confidence for det in detections]) if detections else 0.0,
                        metadata={'processing_type': 'single_frame'}
                    )
            
            raise ValueError("Invalid input data format")
            
        except Exception as e:
            self.logger.error(f"Object detection task failed: {e}")
            return VisionResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                data=None,
                confidence=0.0,
                error_message=str(e)
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if not isinstance(input_data, dict):
            return False
        
        frame = input_data.get('frame')
        batch_frames = input_data.get('batch_frames')
        
        # Must have either frame or batch_frames
        if frame is None and batch_frames is None:
            return False
        
        # Validate frame shape if present
        if frame is not None:
            if not hasattr(frame, 'shape') or len(frame.shape) != 3:
                return False
        
        # Validate batch frames if present
        if batch_frames is not None:
            if not isinstance(batch_frames, (list, tuple)):
                return False
            for f in batch_frames:
                if not hasattr(f, 'shape') or len(f.shape) != 3:
                    return False
        
        return True
    
    async def _ensure_model_loaded(self):
        """Ensure the model is loaded"""
        try:
            self.model = await self.model_hub.ensure_model_ready(self.model_name)
            if self.model is None:
                raise RuntimeError(f"Failed to load model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise


class ObjectDetector:
    """Main object detection class using YOLOv8"""
    
    def __init__(self, model_name: str = "yolov8n", device: str = "auto", model_hub: Optional[ModelHub] = None):
        """
        Initialize object detector
        
        Args:
            model_name: YOLOv8 model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: Device to run on ('cpu', 'cuda', 'auto')
            model_hub: Optional ModelHub instance
        """
        self.model_name = model_name
        self.device_type = DeviceType.AUTO if device == "auto" else DeviceType.GPU if device == "cuda" else DeviceType.CPU
        self.model_hub = model_hub or get_model_hub()
        self.model = None
        self._model_lock = threading.Lock()
        self.logger = logging.getLogger(__name__ + ".ObjectDetector")
        
        # COCO class names (YOLOv8 default)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    async def ensure_model_ready(self) -> bool:
        """Ensure model is downloaded and loaded"""
        try:
            with self._model_lock:
                if self.model is None:
                    self.logger.info(f"Loading YOLOv8 model: {self.model_name}")
                    self.model = await self.model_hub.ensure_model_ready(self.model_name, self.device_type)
                    if self.model is None:
                        self.logger.error(f"Failed to load model {self.model_name}")
                        return False
                    self.logger.info(f"Model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to ensure model ready: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[DetectionResult]:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input image as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call ensure_model_ready() first.")
        
        try:
            # Run inference
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # First (and only) image
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # Extract data
                    if hasattr(boxes, 'xyxy'):
                        coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                    else:
                        coords = []
                    
                    if hasattr(boxes, 'conf'):
                        confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                    else:
                        confidences = []
                    
                    if hasattr(boxes, 'cls'):
                        class_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                    else:
                        class_ids = []
                    
                    # Create detection results
                    for i in range(len(coords)):
                        x1, y1, x2, y2 = map(int, coords[i])
                        confidence = float(confidences[i])
                        class_id = int(class_ids[i])
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detections.append(DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        ))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise
    
    def detect_objects_batch(self, frames: List[np.ndarray], conf_threshold: float = 0.25) -> BatchDetectionResult:
        """
        Detect objects in a batch of frames
        
        Args:
            frames: List of input images as numpy arrays
            conf_threshold: Confidence threshold for detections
            
        Returns:
            BatchDetectionResult object
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call ensure_model_ready() first.")
        
        start_time = time.time()
        batch_detections = []
        
        try:
            # Process batch
            results = self.model(frames, conf=conf_threshold, verbose=False)
            
            # Parse results for each image
            for i, result in enumerate(results):
                detections = []
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # Extract data
                    if hasattr(boxes, 'xyxy'):
                        coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                    else:
                        coords = []
                    
                    if hasattr(boxes, 'conf'):
                        confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                    else:
                        confidences = []
                    
                    if hasattr(boxes, 'cls'):
                        class_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                    else:
                        class_ids = []
                    
                    # Create detection results
                    for j in range(len(coords)):
                        x1, y1, x2, y2 = map(int, coords[j])
                        confidence = float(confidences[j])
                        class_id = int(class_ids[j])
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detections.append(DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        ))
                
                batch_detections.append(detections)
            
            processing_time = time.time() - start_time
            
            return BatchDetectionResult(
                detections=batch_detections,
                processing_time=processing_time,
                batch_size=len(frames)
            )
            
        except Exception as e:
            self.logger.error(f"Batch detection failed: {e}")
            raise


class VideoStreamDetector:
    """Video stream object detection with OpenCV VideoCapture"""
    
    def __init__(self, detector: ObjectDetector, source: Union[int, str] = 0):
        """
        Initialize video stream detector
        
        Args:
            detector: ObjectDetector instance
            source: Video source (camera index or video file path)
        """
        self.detector = detector
        self.source = source
        self.cap = None
        self.is_streaming = False
        self._stream_lock = threading.Lock()
        self.logger = logging.getLogger(__name__ + ".VideoStreamDetector")
    
    def start_stream(self) -> bool:
        """Start video stream"""
        cv2 = _lazy_import_cv2()
        
        try:
            with self._stream_lock:
                if self.is_streaming:
                    self.logger.warning("Stream already started")
                    return True
                
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    self.logger.error(f"Failed to open video source: {self.source}")
                    return False
                
                self.is_streaming = True
                self.logger.info(f"Video stream started from source: {self.source}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")
            return False
    
    def stop_stream(self) -> bool:
        """Stop video stream"""
        try:
            with self._stream_lock:
                if not self.is_streaming:
                    return True
                
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                self.is_streaming = False
                self.logger.info("Video stream stopped")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to stop stream: {e}")
            return False
    
    def detect_stream(self, conf_threshold: float = 0.25, 
                     frame_callback: Optional[Callable[[np.ndarray, List[DetectionResult]], None]] = None) -> Iterator[Tuple[np.ndarray, List[DetectionResult]]]:
        """
        Detect objects in video stream
        
        Args:
            conf_threshold: Confidence threshold for detections
            frame_callback: Optional callback function called for each frame
            
        Yields:
            Tuple of (frame, detections) for each frame
        """
        if not self.is_streaming:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        try:
            while self.is_streaming:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    break
                
                # Detect objects
                detections = self.detector.detect_objects(frame, conf_threshold)
                
                # Call callback if provided
                if frame_callback:
                    frame_callback(frame, detections)
                
                yield frame, detections
                
        except Exception as e:
            self.logger.error(f"Stream detection failed: {e}")
            raise
        finally:
            self.stop_stream()


# Convenience functions
async def create_detector(model_name: str = "yolov8n", device: str = "auto") -> ObjectDetector:
    """
    Create and initialize an object detector
    
    Args:
        model_name: YOLOv8 model name
        device: Device to run on
        
    Returns:
        Initialized ObjectDetector instance
    """
    detector = ObjectDetector(model_name=model_name, device=device)
    success = await detector.ensure_model_ready()
    if not success:
        raise RuntimeError(f"Failed to initialize detector with model {model_name}")
    return detector


def detect_objects(frame: np.ndarray, model: Any = None, conf_threshold: float = 0.25) -> List[DetectionResult]:
    """
    Convenience function for single frame object detection
    
    Args:
        frame: Input image as numpy array
        model: Optional pre-loaded YOLOv8 model (if None, uses global detector)
        conf_threshold: Confidence threshold
        
    Returns:
        List of DetectionResult objects
    """
    if model is None:
        raise ValueError("Model must be provided or use ObjectDetector class")
    
    # Create temporary detector with the model
    detector = ObjectDetector()
    detector.model = model
    
    return detector.detect_objects(frame, conf_threshold)


def detect_objects_batch(frames: List[np.ndarray], model: Any = None, conf_threshold: float = 0.25) -> BatchDetectionResult:
    """
    Convenience function for batch object detection
    
    Args:
        frames: List of input images as numpy arrays
        model: Optional pre-loaded YOLOv8 model (if None, uses global detector)
        conf_threshold: Confidence threshold
        
    Returns:
        BatchDetectionResult object
    """
    if model is None:
        raise ValueError("Model must be provided or use ObjectDetector class")
    
    # Create temporary detector with the model
    detector = ObjectDetector()
    detector.model = model
    
    return detector.detect_objects_batch(frames, conf_threshold)


# Export main classes and functions
__all__ = [
    'ObjectDetector',
    'VideoStreamDetector',
    'ObjectDetectionTask',
    'DetectionResult',
    'BatchDetectionResult',
    'detect_objects',
    'detect_objects_batch',
    'create_detector'
]
