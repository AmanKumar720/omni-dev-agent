"""
Pytest configuration and fixtures for AI Vision tests
"""

import pytest
import asyncio
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Generator, Optional

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_torch():
    """Mock PyTorch to avoid requiring installation for basic tests."""
    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.return_value = "cpu"
    mock_torch.no_grad.return_value = Mock(__enter__=Mock(), __exit__=Mock())
    mock_torch.nn = Mock()
    mock_torch.nn.functional = Mock()
    mock_torch.nn.functional.softmax.return_value = Mock()
    mock_torch.topk.return_value = (Mock(), Mock())
    mock_torch.save = Mock()
    mock_torch.load = Mock()
    
    with patch.dict("sys.modules", {"torch": mock_torch}):
        yield mock_torch

@pytest.fixture
def mock_cv2():
    """Mock OpenCV for tests that don't need actual computer vision."""
    mock_cv2 = Mock()
    mock_cv2.VideoCapture.return_value = Mock()
    mock_cv2.imread.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    mock_cv2.resize.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        yield mock_cv2

@pytest.fixture
def mock_ultralytics():
    """Mock Ultralytics YOLO for object detection tests."""
    mock_results = Mock()
    mock_results.boxes = Mock()
    mock_results.boxes.data = np.array([[10, 10, 100, 100, 0.9, 0]])  # x1,y1,x2,y2,conf,class
    mock_results.names = {0: "person", 1: "car", 2: "dog"}
    
    mock_yolo = Mock()
    mock_yolo.return_value.return_value = [mock_results]
    mock_yolo.return_value.names = {0: "person", 1: "car", 2: "dog"}
    
    mock_ultralytics = Mock()
    mock_ultralytics.YOLO = mock_yolo
    
    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
        yield mock_ultralytics

@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array for testing."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def sample_image_pil():
    """Create a sample PIL image for testing."""
    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(array)

@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    return np.random.randint(0, 255, (224, 224), dtype=np.uint8)

@pytest.fixture
def sample_video_frames():
    """Create sample video frames for testing."""
    frames = []
    for i in range(5):  # 5 sample frames
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames

@pytest.fixture
def sample_text_image():
    """Create an image with text for OCR testing."""
    # Create white background
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add some text using OpenCV (if available) or return simple pattern
    try:
        cv2.putText(img, "HELLO WORLD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Test OCR Text", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    except:
        # If cv2 not available, create simple black rectangles as text placeholders
        img[40:60, 10:150] = 0  # "HELLO WORLD"
        img[70:85, 10:120] = 0  # "Test OCR Text"
    
    return img

@pytest.fixture
def mock_face_recognition():
    """Mock face_recognition library."""
    mock_face_rec = Mock()
    mock_face_rec.face_locations.return_value = [(0, 100, 100, 0)]  # top, right, bottom, left
    mock_face_rec.face_encodings.return_value = [np.random.random(128)]  # 128D face encoding
    mock_face_rec.compare_faces.return_value = [True]
    mock_face_rec.face_distance.return_value = [0.3]
    
    with patch.dict("sys.modules", {"face_recognition": mock_face_rec}):
        yield mock_face_rec

@pytest.fixture
def mock_pytesseract():
    """Mock pytesseract for OCR tests."""
    mock_tess = Mock()
    mock_tess.image_to_string.return_value = "HELLO WORLD\nTest OCR Text"
    mock_tess.image_to_data.return_value = {
        'text': ['', '', 'HELLO', 'WORLD', '', 'Test', 'OCR', 'Text'],
        'conf': ['-1', '-1', '95', '89', '-1', '92', '88', '91'],
        'left': [0, 0, 10, 70, 0, 10, 50, 90],
        'top': [0, 0, 40, 40, 0, 70, 70, 70],
        'width': [300, 300, 50, 60, 0, 35, 30, 40],
        'height': [100, 100, 20, 20, 0, 15, 15, 15]
    }
    
    with patch.dict("sys.modules", {"pytesseract": mock_tess}):
        yield mock_tess

@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_vision"
    test_dir.mkdir()
    return test_dir

@pytest.fixture
def mock_model_hub():
    """Mock model hub for testing."""
    mock_hub = Mock()
    mock_hub.ensure_model_ready.return_value = Mock()  # Mock model
    mock_hub.download_model.return_value = True
    mock_hub.get_model_path.return_value = "/mock/path/to/model"
    return mock_hub

@pytest.fixture
def vision_agent_config():
    """Default configuration for vision agent testing."""
    return {
        "max_concurrent_tasks": 3,
        "task_timeout": 300,
        "enable_caching": True,
        "cache_size": 100
    }

@pytest.fixture
def mock_sqlite_db(tmp_path):
    """Create a mock SQLite database for testing."""
    db_path = tmp_path / "test_faces.db"
    return str(db_path)

@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    
    # Reduce log level for tests
    logging.getLogger("components.ai_vision").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    
    yield
    
    # Reset after tests
    logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def mock_transformers():
    """Mock transformers library for advanced vision models."""
    mock_pipeline = Mock()
    mock_pipeline.return_value = [{"label": "cat", "score": 0.95}]
    
    mock_transformers = Mock()
    mock_transformers.pipeline = Mock(return_value=mock_pipeline)
    
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        yield mock_transformers

@pytest.fixture(params=[
    "resnet50",
    "mobilenet_v3_large", 
    "mobilenet_v3_small"
])
def classification_model_name(request):
    """Parametrize classification model names."""
    return request.param

@pytest.fixture(params=[
    "yolov8n",
    "yolov8s", 
    "yolov8m"
])
def detection_model_name(request):
    """Parametrize detection model names."""
    return request.param

@pytest.fixture
def large_batch_images():
    """Create a large batch of images for stress testing."""
    batch_size = 50
    images = []
    for i in range(batch_size):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(img))
    return images

@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability."""
    with patch("torch.cuda.is_available", return_value=True):
        yield

@pytest.fixture
def mock_no_gpu():
    """Mock no GPU availability."""
    with patch("torch.cuda.is_available", return_value=False):
        yield

def create_test_video(path: Path, frames: int = 10, fps: int = 30):
    """Helper function to create a test video file."""
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
        
        for i in range(frames):
            # Create frame with changing content
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add frame number as text
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        return True
    except Exception:
        return False

@pytest.fixture
def sample_test_video(temp_test_dir):
    """Create a sample test video file."""
    video_path = temp_test_dir / "test_video.mp4"
    if create_test_video(video_path):
        return video_path
    else:
        # Return None if video creation failed (e.g., no cv2)
        return None

# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu
pytest.mark.model = pytest.mark.model

# Skip markers for missing dependencies
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "requires_torch: mark test to run only if torch is available")
    config.addinivalue_line("markers", "requires_cv2: mark test to run only if cv2 is available")
    config.addinivalue_line("markers", "requires_ultralytics: mark test to run only if ultralytics is available")

def pytest_runtest_setup(item):
    """Setup function to skip tests based on missing dependencies."""
    if "requires_torch" in item.keywords:
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    
    if "requires_cv2" in item.keywords:
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")
    
    if "requires_ultralytics" in item.keywords:
        try:
            import ultralytics
        except ImportError:
            pytest.skip("Ultralytics not available")
