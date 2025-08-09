"""
Unit tests for Object Detection module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

from components.ai_vision.object_detection import (
    DetectionResult,
    BatchDetectionResult,
    ObjectDetectionTask,
    ObjectDetector,
    VideoStreamDetector,
    detect_objects,
    detect_objects_batch,
    create_detector
)
from components.ai_vision.core import TaskStatus


@pytest.mark.unit
@pytest.mark.detection
class TestDetectionResult:
    """Test DetectionResult dataclass"""
    
    def test_detection_result_creation(self):
        """Test creating a DetectionResult"""
        bbox = (10, 20, 100, 150)  # x1, y1, x2, y2
        result = DetectionResult(
            bbox=bbox,
            confidence=0.87,
            class_id=1,
            class_name="person"
        )
        
        assert result.bbox == bbox
        assert result.confidence == 0.87
        assert result.class_id == 1
        assert result.class_name == "person"
    
    def test_detection_result_to_dict(self):
        """Test converting DetectionResult to dictionary"""
        bbox = (15, 25, 105, 155)
        result = DetectionResult(
            bbox=bbox,
            confidence=0.92,
            class_id=2,
            class_name="car"
        )
        
        result_dict = result.to_dict()
        expected = {
            'bbox': bbox,
            'confidence': 0.92,
            'class_id': 2,
            'class_name': 'car'
        }
        
        assert result_dict == expected


@pytest.mark.unit
@pytest.mark.detection
class TestBatchDetectionResult:
    """Test BatchDetectionResult dataclass"""
    
    def test_batch_detection_result_creation(self):
        """Test creating a BatchDetectionResult"""
        detections1 = [DetectionResult((10, 20, 100, 150), 0.9, 0, "person")]
        detections2 = [DetectionResult((50, 60, 200, 250), 0.8, 1, "car")]
        
        batch_result = BatchDetectionResult(
            detections=[detections1, detections2],
            processing_time=0.25,
            batch_size=2
        )
        
        assert len(batch_result.detections) == 2
        assert batch_result.processing_time == 0.25
        assert batch_result.batch_size == 2
        assert batch_result.detections[0][0].class_name == "person"
        assert batch_result.detections[1][0].class_name == "car"
    
    def test_batch_detection_result_to_dict(self):
        """Test converting BatchDetectionResult to dictionary"""
        detections1 = [DetectionResult((10, 20, 100, 150), 0.9, 0, "person")]
        detections2 = [DetectionResult((50, 60, 200, 250), 0.8, 1, "car")]
        
        batch_result = BatchDetectionResult(
            detections=[detections1, detections2],
            processing_time=0.25,
            batch_size=2
        )
        
        result_dict = batch_result.to_dict()
        
        assert "detections" in result_dict
        assert "processing_time" in result_dict
        assert "batch_size" in result_dict
        assert len(result_dict["detections"]) == 2
        assert result_dict["detections"][0][0]["class_name"] == "person"


@pytest.mark.unit
@pytest.mark.detection
class TestObjectDetectionTask:
    """Test ObjectDetectionTask"""
    
    def test_task_initialization(self, mock_model_hub):
        """Test task initialization"""
        task = ObjectDetectionTask("task_001", model_name="yolov8n")
        
        assert task.task_id == "task_001"
        assert task.task_type == "object_detection"
        assert task.model_name == "yolov8n"
        assert task.model is None
    
    def test_input_validation_success(self, mock_model_hub):
        """Test successful input validation"""
        task = ObjectDetectionTask("task_001")
        
        # Test valid inputs
        assert task.validate_input({"frame": "test_frame"}) is True
        assert task.validate_input({"batch_frames": ["frame1", "frame2"]}) is True
        assert task.validate_input("direct_frame") is True
    
    def test_input_validation_failure(self, mock_model_hub):
        """Test input validation failure"""
        task = ObjectDetectionTask("task_001")
        
        # Test invalid inputs
        assert task.validate_input(None) is False
        assert task.validate_input({}) is False  # Neither frame nor batch_frames
    
    @pytest.mark.asyncio
    async def test_execute_single_frame(self, mock_model_hub, sample_image_array):
        """Test executing detection on single frame"""
        with patch('components.ai_vision.object_detection.detect_objects') as mock_detect:
            # Setup mock
            mock_detections = [DetectionResult((10, 20, 100, 150), 0.9, 0, "person")]
            mock_detect.return_value = mock_detections
            
            task = ObjectDetectionTask("task_001")
            task.model = Mock()  # Mock loaded model
            
            input_data = {"frame": sample_image_array, "conf_threshold": 0.5}
            result = await task.execute(input_data)
            
            assert result.task_id == "task_001"
            assert result.status == TaskStatus.COMPLETED
            assert len(result.data["detections"]) == 1
            assert result.data["detections"][0]["class_name"] == "person"
    
    @pytest.mark.asyncio
    async def test_execute_batch_frames(self, mock_model_hub, sample_image_array):
        """Test executing detection on batch of frames"""
        with patch('components.ai_vision.object_detection.detect_objects_batch') as mock_batch:
            # Setup mock
            mock_result = BatchDetectionResult(
                detections=[[DetectionResult((10, 20, 100, 150), 0.9, 0, "person")]],
                processing_time=0.1,
                batch_size=1
            )
            mock_batch.return_value = mock_result
            
            task = ObjectDetectionTask("task_001")
            task.model = Mock()  # Mock loaded model
            
            input_data = {"batch_frames": [sample_image_array], "conf_threshold": 0.5}
            result = await task.execute(input_data)
            
            assert result.task_id == "task_001"
            assert result.status == TaskStatus.COMPLETED
            assert result.metadata["batch_size"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_model_loading_error(self, mock_model_hub):
        """Test execution when model loading fails"""
        mock_model_hub.ensure_model_ready.side_effect = Exception("Model load failed")
        
        task = ObjectDetectionTask("task_001")
        task.model_hub = mock_model_hub
        
        input_data = {"frame": "test_frame"}
        result = await task.execute(input_data)
        
        assert result.status == TaskStatus.FAILED
        assert "Model load failed" in result.error_message


@pytest.mark.unit
@pytest.mark.detection
class TestObjectDetector:
    """Test ObjectDetector class"""
    
    def test_detector_initialization(self, mock_model_hub, mock_ultralytics):
        """Test detector initialization"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            detector = ObjectDetector(
                model_name="yolov8s",
                device="cpu",
                conf_threshold=0.3
            )
            
            assert detector.model_name == "yolov8s"
            assert detector.conf_threshold == 0.3
            assert detector.model is None
    
    @pytest.mark.asyncio
    async def test_ensure_model_ready_success(self, mock_model_hub, mock_ultralytics):
        """Test successful model loading"""
        mock_model = Mock()
        mock_model_hub.ensure_model_ready.return_value = mock_model
        
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                detector = ObjectDetector(model_name="yolov8n")
                result = await detector.ensure_model_ready()
                
                assert result is True
                assert detector.model is not None
    
    @pytest.mark.asyncio
    async def test_ensure_model_ready_failure(self, mock_model_hub):
        """Test model loading failure"""
        mock_model_hub.ensure_model_ready.side_effect = Exception("Load failed")
        
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            detector = ObjectDetector()
            result = await detector.ensure_model_ready()
            
            assert result is False
    
    def test_detect_objects_without_model(self, mock_model_hub):
        """Test detection without loaded model"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            detector = ObjectDetector()
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                detector.detect_objects("test_frame")
    
    def test_detect_objects_with_model(self, mock_model_hub, mock_ultralytics, sample_image_array):
        """Test object detection with loaded model"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                # Setup mock YOLO model and results
                mock_results = Mock()
                mock_results.boxes = Mock()
                mock_results.boxes.data = np.array([
                    [10, 20, 100, 150, 0.9, 0],  # x1,y1,x2,y2,conf,class
                    [50, 60, 200, 250, 0.8, 1]
                ])
                mock_results.names = {0: "person", 1: "car"}
                
                mock_model = Mock()
                mock_model.return_value = [mock_results]
                mock_model.names = {0: "person", 1: "car"}
                
                detector = ObjectDetector()
                detector.model = mock_model
                
                detections = detector.detect_objects(sample_image_array)
                
                assert len(detections) == 2
                assert detections[0].class_name == "person"
                assert detections[0].confidence == 0.9
                assert detections[1].class_name == "car"
                assert detections[1].confidence == 0.8
    
    def test_detect_objects_batch_without_model(self, mock_model_hub):
        """Test batch detection without loaded model"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            detector = ObjectDetector()
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                detector.detect_objects_batch(["test_frame"])
    
    def test_detect_objects_batch_with_model(self, mock_model_hub, mock_ultralytics, sample_image_array):
        """Test batch object detection with loaded model"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                # Setup mock YOLO model and results
                mock_results1 = Mock()
                mock_results1.boxes = Mock()
                mock_results1.boxes.data = np.array([[10, 20, 100, 150, 0.9, 0]])
                mock_results1.names = {0: "person"}
                
                mock_results2 = Mock()
                mock_results2.boxes = Mock()
                mock_results2.boxes.data = np.array([[50, 60, 200, 250, 0.8, 1]])
                mock_results2.names = {1: "car"}
                
                mock_model = Mock()
                mock_model.return_value = [mock_results1, mock_results2]
                mock_model.names = {0: "person", 1: "car"}
                
                detector = ObjectDetector()
                detector.model = mock_model
                
                frames = [sample_image_array, sample_image_array]
                batch_result = detector.detect_objects_batch(frames)
                
                assert batch_result.batch_size == 2
                assert len(batch_result.detections) == 2
                assert len(batch_result.detections[0]) == 1  # First frame has 1 detection
                assert len(batch_result.detections[1]) == 1  # Second frame has 1 detection
    
    def test_confidence_filtering(self, mock_model_hub, mock_ultralytics, sample_image_array):
        """Test confidence threshold filtering"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                # Setup mock results with different confidence values
                mock_results = Mock()
                mock_results.boxes = Mock()
                mock_results.boxes.data = np.array([
                    [10, 20, 100, 150, 0.9, 0],   # High confidence - should be included
                    [50, 60, 200, 250, 0.3, 1],   # Low confidence - should be filtered out
                    [100, 120, 300, 350, 0.8, 2]  # Medium confidence - should be included
                ])
                mock_results.names = {0: "person", 1: "car", 2: "dog"}
                
                mock_model = Mock()
                mock_model.return_value = [mock_results]
                mock_model.names = {0: "person", 1: "car", 2: "dog"}
                
                detector = ObjectDetector(conf_threshold=0.5)  # Filter out confidence < 0.5
                detector.model = mock_model
                
                detections = detector.detect_objects(sample_image_array)
                
                # Should only get 2 detections (confidence >= 0.5)
                assert len(detections) == 2
                assert detections[0].confidence == 0.9
                assert detections[1].confidence == 0.8
                assert all(d.confidence >= 0.5 for d in detections)


@pytest.mark.unit
@pytest.mark.detection
class TestVideoStreamDetector:
    """Test VideoStreamDetector class"""
    
    def test_video_detector_initialization(self, mock_model_hub, mock_cv2):
        """Test video stream detector initialization"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_cv2', return_value=mock_cv2):
                detector = VideoStreamDetector(
                    source=0,  # Default webcam
                    model_name="yolov8n"
                )
                
                assert detector.source == 0
                assert detector.model_name == "yolov8n"
    
    def test_video_detector_with_file_source(self, mock_model_hub, mock_cv2):
        """Test video detector with file source"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_cv2', return_value=mock_cv2):
                detector = VideoStreamDetector(
                    source="path/to/video.mp4",
                    model_name="yolov8s"
                )
                
                assert detector.source == "path/to/video.mp4"
                assert detector.model_name == "yolov8s"
    
    def test_process_frame(self, mock_model_hub, mock_cv2, mock_ultralytics, sample_image_array):
        """Test processing a single frame from video stream"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_cv2', return_value=mock_cv2):
                with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                    # Setup mock model and results
                    mock_results = Mock()
                    mock_results.boxes = Mock()
                    mock_results.boxes.data = np.array([[10, 20, 100, 150, 0.9, 0]])
                    mock_results.names = {0: "person"}
                    
                    mock_model = Mock()
                    mock_model.return_value = [mock_results]
                    mock_model.names = {0: "person"}
                    
                    detector = VideoStreamDetector(source=0)
                    detector.model = mock_model
                    
                    detections = detector.process_frame(sample_image_array)
                    
                    assert len(detections) == 1
                    assert detections[0].class_name == "person"
                    assert detections[0].confidence == 0.9
    
    def test_start_stream_without_model(self, mock_model_hub, mock_cv2):
        """Test starting stream without loaded model"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_cv2', return_value=mock_cv2):
                detector = VideoStreamDetector(source=0)
                
                with pytest.raises(RuntimeError, match="Model not loaded"):
                    for _ in detector.stream_detections():
                        break


@pytest.mark.unit
@pytest.mark.detection
class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_detect_objects_without_model(self):
        """Test detect_objects function without model"""
        with pytest.raises(ValueError, match="Model must be provided"):
            detect_objects("test_frame", model=None)
    
    def test_detect_objects_with_model(self, sample_image_array):
        """Test detect_objects function with model"""
        mock_model = Mock()
        
        with patch('components.ai_vision.object_detection.ObjectDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_detections = [DetectionResult((10, 20, 100, 150), 0.9, 0, "person")]
            mock_detector.detect_objects.return_value = mock_detections
            mock_detector_class.return_value = mock_detector
            
            result = detect_objects(sample_image_array, mock_model, conf_threshold=0.5)
            
            assert result == mock_detections
            mock_detector.detect_objects.assert_called_once_with(sample_image_array)
    
    def test_detect_objects_batch_without_model(self):
        """Test detect_objects_batch function without model"""
        with pytest.raises(ValueError, match="Model must be provided"):
            detect_objects_batch(["test_frame"], model=None)
    
    def test_detect_objects_batch_with_model(self, sample_image_array):
        """Test detect_objects_batch function with model"""
        mock_model = Mock()
        frames = [sample_image_array]
        
        with patch('components.ai_vision.object_detection.ObjectDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_result = BatchDetectionResult(
                [[DetectionResult((10, 20, 100, 150), 0.9, 0, "person")]],
                0.1, 1
            )
            mock_detector.detect_objects_batch.return_value = mock_result
            mock_detector_class.return_value = mock_detector
            
            result = detect_objects_batch(frames, mock_model, conf_threshold=0.5)
            
            assert result == mock_result
            mock_detector.detect_objects_batch.assert_called_once_with(frames)
    
    @pytest.mark.asyncio
    async def test_create_detector_success(self, mock_model_hub):
        """Test create_detector function success"""
        with patch('components.ai_vision.object_detection.ObjectDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_detector.ensure_model_ready = AsyncMock(return_value=True)
            mock_detector_class.return_value = mock_detector
            
            result = await create_detector("yolov8n", "cpu")
            
            assert result == mock_detector
            mock_detector.ensure_model_ready.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_detector_failure(self, mock_model_hub):
        """Test create_detector function failure"""
        with patch('components.ai_vision.object_detection.ObjectDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_detector.ensure_model_ready = AsyncMock(return_value=False)
            mock_detector_class.return_value = mock_detector
            
            with pytest.raises(RuntimeError, match="Failed to initialize detector"):
                await create_detector("yolov8n", "cpu")


@pytest.mark.unit
@pytest.mark.detection
class TestDetectionErrorHandling:
    """Test error handling in object detection"""
    
    def test_invalid_frame_handling(self, mock_model_hub, mock_ultralytics):
        """Test handling of invalid frames"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                mock_model = Mock()
                mock_model.side_effect = Exception("Invalid frame")
                
                detector = ObjectDetector()
                detector.model = mock_model
                
                with pytest.raises(Exception, match="Invalid frame"):
                    detector.detect_objects("invalid_frame")
    
    def test_model_inference_error(self, mock_model_hub, mock_ultralytics, sample_image_array):
        """Test handling of model inference errors"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                mock_model = Mock()
                mock_model.side_effect = RuntimeError("CUDA out of memory")
                
                detector = ObjectDetector()
                detector.model = mock_model
                
                with pytest.raises(RuntimeError, match="CUDA out of memory"):
                    detector.detect_objects(sample_image_array)
    
    def test_empty_detection_results(self, mock_model_hub, mock_ultralytics, sample_image_array):
        """Test handling when no objects are detected"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            with patch('components.ai_vision.object_detection._lazy_import_ultralytics', return_value=mock_ultralytics):
                # Setup mock to return empty results
                mock_results = Mock()
                mock_results.boxes = Mock()
                mock_results.boxes.data = np.array([]).reshape(0, 6)  # Empty detections
                mock_results.names = {0: "person", 1: "car"}
                
                mock_model = Mock()
                mock_model.return_value = [mock_results]
                mock_model.names = {0: "person", 1: "car"}
                
                detector = ObjectDetector()
                detector.model = mock_model
                
                detections = detector.detect_objects(sample_image_array)
                
                assert len(detections) == 0
                assert isinstance(detections, list)
    
    def test_invalid_confidence_threshold(self, mock_model_hub):
        """Test handling of invalid confidence thresholds"""
        with patch('components.ai_vision.object_detection.get_model_hub', return_value=mock_model_hub):
            # Test negative confidence threshold
            detector = ObjectDetector(conf_threshold=-0.1)
            assert detector.conf_threshold >= 0.0  # Should be clamped
            
            # Test confidence threshold > 1.0
            detector = ObjectDetector(conf_threshold=1.5)
            assert detector.conf_threshold <= 1.0  # Should be clamped
