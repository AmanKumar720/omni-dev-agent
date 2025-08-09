"""
Integration tests for AI Vision modules with sample images and videos
"""

import pytest
import numpy as np
import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch, AsyncMock
import json

from components.ai_vision import (
    AIVisionAgent,
    ImageClassifier,
    ImageClassificationTask,
    ObjectDetector, 
    ObjectDetectionTask,
    OCREngine,
    FaceRecognizer,
    FaceRecognitionTask,
    ComputerVisionAnalyticsAgent
)


@pytest.fixture
def sample_classification_image():
    """Create a sample image that looks like it could contain a cat"""
    # Create a simple image with cat-like features
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple cat-like shape
    # Body (oval)
    draw.ellipse([50, 100, 174, 180], fill='orange', outline='black')
    
    # Head (circle)
    draw.ellipse([80, 50, 144, 120], fill='orange', outline='black')
    
    # Ears (triangles)
    draw.polygon([(85, 60), (95, 40), (105, 60)], fill='orange', outline='black')
    draw.polygon([(119, 60), (129, 40), (139, 60)], fill='orange', outline='black')
    
    # Eyes
    draw.ellipse([92, 70, 100, 78], fill='black')
    draw.ellipse([124, 70, 132, 78], fill='black')
    
    # Nose
    draw.polygon([(110, 85), (115, 90), (105, 90)], fill='pink')
    
    # Convert to numpy array
    return np.array(img)


@pytest.fixture
def sample_detection_image():
    """Create a sample image for object detection"""
    # Create image with person-like and car-like objects
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a person-like figure
    # Head
    draw.ellipse([100, 50, 140, 90], fill='peachpuff', outline='black')
    # Body
    draw.rectangle([110, 90, 130, 150], fill='blue', outline='black')
    # Arms
    draw.rectangle([90, 100, 110, 120], fill='peachpuff', outline='black')
    draw.rectangle([130, 100, 150, 120], fill='peachpuff', outline='black')
    # Legs
    draw.rectangle([110, 150, 120, 200], fill='black', outline='black')
    draw.rectangle([120, 150, 130, 200], fill='black', outline='black')
    
    # Draw a car-like object
    # Car body
    draw.rectangle([300, 250, 500, 320], fill='red', outline='black')
    # Wheels
    draw.ellipse([320, 310, 360, 350], fill='black', outline='gray')
    draw.ellipse([440, 310, 480, 350], fill='black', outline='gray')
    # Windows
    draw.rectangle([320, 260, 480, 300], fill='lightblue', outline='black')
    
    return np.array(img)


@pytest.fixture  
def sample_ocr_image():
    """Create a sample image with clear text for OCR"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to built-in if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw some text
    draw.text((20, 20), "SAMPLE TEXT", fill='black', font=font)
    draw.text((20, 60), "For OCR Testing", fill='black', font=font)
    draw.text((20, 100), "Line 3 with numbers: 12345", fill='black', font=font)
    draw.text((20, 140), "Special chars: @#$%", fill='black', font=font)
    
    return np.array(img)


@pytest.fixture
def sample_face_image():
    """Create a simple face-like image"""
    img = Image.new('RGB', (200, 200), color='peachpuff')
    draw = ImageDraw.Draw(img)
    
    # Face outline
    draw.ellipse([50, 40, 150, 160], fill='peachpuff', outline='brown', width=2)
    
    # Eyes
    draw.ellipse([70, 80, 85, 95], fill='white', outline='black')
    draw.ellipse([115, 80, 130, 95], fill='white', outline='black')
    draw.ellipse([75, 85, 80, 90], fill='black')  # Pupils
    draw.ellipse([120, 85, 125, 90], fill='black')
    
    # Nose
    draw.polygon([(95, 105), (105, 105), (100, 115)], fill='brown', outline='brown')
    
    # Mouth
    draw.arc([80, 125, 120, 145], start=0, end=180, fill='red', width=3)
    
    return np.array(img)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.vision
class TestImageClassificationIntegration:
    """Integration tests for image classification"""
    
    def test_create_sample_images_for_classification(self, temp_test_dir):
        """Test creating and saving sample classification images"""
        # Create different types of images
        images = {
            'cat_like.jpg': self._create_cat_image(),
            'dog_like.jpg': self._create_dog_image(),
            'car_like.jpg': self._create_car_image(),
            'natural_scene.jpg': self._create_natural_scene()
        }
        
        for filename, img_array in images.items():
            img_path = temp_test_dir / filename
            img = Image.fromarray(img_array)
            img.save(img_path)
            
            assert img_path.exists()
            assert img_path.stat().st_size > 0
    
    def _create_cat_image(self):
        """Create a cat-like image"""
        img = Image.new('RGB', (224, 224), color=(245, 245, 220))  # Beige background
        draw = ImageDraw.Draw(img)
        
        # Cat silhouette
        draw.ellipse([40, 80, 180, 180], fill=(255, 140, 0), outline='black')  # Orange body
        draw.ellipse([70, 40, 150, 120], fill=(255, 140, 0), outline='black')  # Head
        
        # Ears
        draw.polygon([(75, 50), (85, 20), (95, 50)], fill=(255, 140, 0))
        draw.polygon([(125, 50), (135, 20), (145, 50)], fill=(255, 140, 0))
        
        # Cat features
        draw.ellipse([90, 70, 100, 80], fill='green')  # Left eye
        draw.ellipse([120, 70, 130, 80], fill='green')  # Right eye
        draw.polygon([(105, 85), (115, 95), (95, 95)], fill='pink')  # Nose
        
        return np.array(img)
    
    def _create_dog_image(self):
        """Create a dog-like image"""
        img = Image.new('RGB', (224, 224), color=(135, 206, 235))  # Sky blue background
        draw = ImageDraw.Draw(img)
        
        # Dog silhouette
        draw.ellipse([30, 90, 190, 190], fill=(139, 69, 19), outline='black')  # Brown body
        draw.ellipse([60, 50, 160, 130], fill=(139, 69, 19), outline='black')  # Head
        
        # Floppy ears
        draw.ellipse([45, 60, 75, 110], fill=(101, 67, 33))  # Left ear
        draw.ellipse([145, 60, 175, 110], fill=(101, 67, 33))  # Right ear
        
        # Dog features
        draw.ellipse([85, 80, 95, 90], fill='black')  # Left eye
        draw.ellipse([125, 80, 135, 90], fill='black')  # Right eye
        draw.ellipse([105, 95, 115, 105], fill='black')  # Nose
        
        # Tongue
        draw.ellipse([100, 105, 120, 120], fill='pink')
        
        return np.array(img)
    
    def _create_car_image(self):
        """Create a car-like image"""
        img = Image.new('RGB', (224, 224), color=(192, 192, 192))  # Gray background (road)
        draw = ImageDraw.Draw(img)
        
        # Car body
        draw.rectangle([40, 100, 180, 160], fill='red', outline='darkred', width=2)
        
        # Car roof
        draw.rectangle([60, 80, 160, 100], fill='red', outline='darkred', width=2)
        
        # Wheels
        draw.ellipse([50, 150, 80, 180], fill='black', outline='gray', width=2)
        draw.ellipse([140, 150, 170, 180], fill='black', outline='gray', width=2)
        
        # Wheel centers
        draw.ellipse([60, 160, 70, 170], fill='silver')
        draw.ellipse([150, 160, 160, 170], fill='silver')
        
        # Windows
        draw.rectangle([70, 85, 150, 95], fill='lightblue', outline='blue')
        
        # Headlights
        draw.ellipse([175, 110, 185, 130], fill='yellow', outline='orange')
        draw.ellipse([175, 130, 185, 150], fill='yellow', outline='orange')
        
        return np.array(img)
    
    def _create_natural_scene(self):
        """Create a natural scene image"""
        img = Image.new('RGB', (224, 224), color=(135, 206, 235))  # Sky
        draw = ImageDraw.Draw(img)
        
        # Ground
        draw.rectangle([0, 150, 224, 224], fill=(34, 139, 34))  # Green ground
        
        # Tree trunk
        draw.rectangle([90, 100, 110, 150], fill=(139, 69, 19))  # Brown trunk
        
        # Tree crown
        draw.ellipse([60, 60, 140, 120], fill=(0, 100, 0), outline='darkgreen')
        
        # Sun
        draw.ellipse([170, 20, 200, 50], fill='yellow', outline='orange')
        
        # Clouds
        draw.ellipse([20, 30, 60, 50], fill='white', outline='lightgray')
        draw.ellipse([30, 25, 70, 45], fill='white', outline='lightgray')
        
        # Flowers
        for i in range(5):
            x = 20 + i * 30
            y = 160 + np.random.randint(-10, 10)
            draw.ellipse([x, y, x+10, y+10], fill='red', outline='darkred')
        
        return np.array(img)
    
    @pytest.mark.asyncio
    async def test_classification_with_mock_model(self, sample_classification_image, mock_torch):
        """Test image classification with mocked model"""
        from components.ai_vision.image_classification import TopKResult, ClassificationResult
        
        with patch('components.ai_vision.image_classification.ImageClassifier') as MockClassifier:
            # Create mock classifier with expected results
            mock_classifier = Mock()
            
            # Mock classification results
            predictions = [
                ClassificationResult(class_id=281, class_name="tabby_cat", confidence=0.87),
                ClassificationResult(class_id=282, class_name="tiger_cat", confidence=0.76),
                ClassificationResult(class_id=285, class_name="egyptian_cat", confidence=0.65)
            ]
            
            mock_result = TopKResult(
                predictions=predictions,
                processing_time=0.125,
                k=3
            )
            
            mock_classifier.classify_image.return_value = mock_result
            mock_classifier.ensure_model_ready = AsyncMock(return_value=True)
            MockClassifier.return_value = mock_classifier
            
            # Test classification
            classifier = MockClassifier()
            await classifier.ensure_model_ready()
            
            result = classifier.classify_image(Image.fromarray(sample_classification_image), k=3)
            
            assert len(result.predictions) == 3
            assert result.predictions[0].class_name == "tabby_cat"
            assert result.predictions[0].confidence == 0.87
            assert result.k == 3
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_batch_classification(self, temp_test_dir, mock_torch):
        """Test batch image classification"""
        from components.ai_vision.image_classification import BatchClassificationResult, TopKResult, ClassificationResult
        
        # Create multiple test images
        images = []
        for i in range(3):
            if i == 0:
                img_array = self._create_cat_image()
            elif i == 1:
                img_array = self._create_dog_image()
            else:
                img_array = self._create_car_image()
            
            images.append(Image.fromarray(img_array))
        
        with patch('components.ai_vision.image_classification.ImageClassifier') as MockClassifier:
            mock_classifier = Mock()
            
            # Mock batch results
            batch_results = []
            for i, expected_class in enumerate(['cat', 'dog', 'car']):
                predictions = [ClassificationResult(i, expected_class, 0.85)]
                batch_results.append(TopKResult(predictions, 0.1, 1))
            
            mock_batch_result = BatchClassificationResult(
                results=batch_results,
                processing_time=0.35,
                batch_size=3
            )
            
            mock_classifier.classify_images_batch.return_value = mock_batch_result
            mock_classifier.ensure_model_ready = AsyncMock(return_value=True)
            MockClassifier.return_value = mock_classifier
            
            # Test batch classification
            classifier = MockClassifier()
            await classifier.ensure_model_ready()
            
            result = classifier.classify_images_batch(images, k=1)
            
            assert result.batch_size == 3
            assert len(result.results) == 3
            assert result.results[0].predictions[0].class_name == 'cat'
            assert result.results[1].predictions[0].class_name == 'dog'
            assert result.results[2].predictions[0].class_name == 'car'


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.vision
class TestObjectDetectionIntegration:
    """Integration tests for object detection"""
    
    @pytest.mark.asyncio
    async def test_detection_with_mock_model(self, sample_detection_image, mock_ultralytics):
        """Test object detection with mocked YOLO model"""
        from components.ai_vision.object_detection import DetectionResult
        
        with patch('components.ai_vision.object_detection.ObjectDetector') as MockDetector:
            # Setup mock detector
            mock_detector = Mock()
            
            # Mock detection results - person and car
            detections = [
                DetectionResult(
                    bbox=(100, 50, 150, 200),  # Person bounding box
                    confidence=0.92,
                    class_id=0,
                    class_name="person"
                ),
                DetectionResult(
                    bbox=(300, 250, 500, 350),  # Car bounding box
                    confidence=0.87,
                    class_id=2,
                    class_name="car"
                )
            ]
            
            mock_detector.detect_objects.return_value = detections
            mock_detector.ensure_model_ready = AsyncMock(return_value=True)
            MockDetector.return_value = mock_detector
            
            # Test detection
            detector = MockDetector()
            await detector.ensure_model_ready()
            
            results = detector.detect_objects(sample_detection_image)
            
            assert len(results) == 2
            assert results[0].class_name == "person"
            assert results[0].confidence == 0.92
            assert results[1].class_name == "car" 
            assert results[1].confidence == 0.87
            
            # Verify bounding boxes are reasonable
            person_bbox = results[0].bbox
            car_bbox = results[1].bbox
            assert person_bbox[2] > person_bbox[0]  # width > 0
            assert person_bbox[3] > person_bbox[1]  # height > 0
            assert car_bbox[2] > car_bbox[0]  # width > 0
            assert car_bbox[3] > car_bbox[1]  # height > 0
    
    @pytest.mark.asyncio
    async def test_video_frame_processing(self, mock_ultralytics, mock_cv2):
        """Test processing multiple video frames"""
        from components.ai_vision.object_detection import VideoStreamDetector, DetectionResult
        
        # Create sequence of frames with moving objects
        frames = []
        for i in range(5):
            frame = self._create_frame_with_moving_objects(i)
            frames.append(frame)
        
        with patch('components.ai_vision.object_detection.VideoStreamDetector') as MockVideoDetector:
            mock_detector = Mock()
            
            # Mock detection results that change over time
            def mock_process_frame(frame):
                frame_idx = len(frames) - 5 + 1  # Simulate frame index
                return [
                    DetectionResult(
                        bbox=(100 + frame_idx * 10, 100, 150 + frame_idx * 10, 150),
                        confidence=0.85,
                        class_id=0,
                        class_name="person"
                    )
                ]
            
            mock_detector.process_frame.side_effect = mock_process_frame
            mock_detector.ensure_model_ready = AsyncMock(return_value=True)
            MockVideoDetector.return_value = mock_detector
            
            # Test video processing
            detector = MockVideoDetector()
            await detector.ensure_model_ready()
            
            all_detections = []
            for frame in frames:
                detections = detector.process_frame(frame)
                all_detections.append(detections)
            
            assert len(all_detections) == 5
            assert all(len(dets) == 1 for dets in all_detections)
            assert all(dets[0].class_name == "person" for dets in all_detections)
    
    def _create_frame_with_moving_objects(self, frame_idx):
        """Create a frame with objects at different positions"""
        img = Image.new('RGB', (640, 480), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Moving person (shifts right over time)
        x_offset = frame_idx * 20
        person_x = 100 + x_offset
        
        # Person
        draw.ellipse([person_x, 50, person_x + 40, 90], fill='peachpuff', outline='black')
        draw.rectangle([person_x + 10, 90, person_x + 30, 150], fill='blue', outline='black')
        
        # Stationary car
        draw.rectangle([400, 300, 550, 350], fill='red', outline='black')
        draw.ellipse([410, 340, 440, 370], fill='black')
        draw.ellipse([520, 340, 550, 370], fill='black')
        
        return np.array(img)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ocr
class TestOCRIntegration:
    """Integration tests for OCR"""
    
    def test_ocr_with_mock_tesseract(self, sample_ocr_image, mock_pytesseract):
        """Test OCR with mocked tesseract"""
        from components.ai_vision.ocr import OCRResult, OCRPage, OCRBlock, OCRLine, OCRWord, BoundingBox
        
        # Mock tesseract responses
        mock_pytesseract.image_to_string.return_value = (
            "SAMPLE TEXT\\n"
            "For OCR Testing\\n" 
            "Line 3 with numbers: 12345\\n"
            "Special chars: @#$%"
        )
        
        mock_data = {
            'text': ['', '', 'SAMPLE', 'TEXT', '', 'For', 'OCR', 'Testing', '', 'Line', '3', 'with', 'numbers:', '12345', '', 'Special', 'chars:', '@#$%'],
            'conf': ['-1', '-1', '95', '92', '-1', '90', '88', '91', '-1', '87', '89', '85', '84', '90', '-1', '86', '83', '82'],
            'left': [0, 0, 20, 90, 0, 20, 50, 90, 0, 20, 50, 70, 110, 170, 0, 20, 70, 120],
            'top': [0, 0, 20, 20, 0, 60, 60, 60, 0, 100, 100, 100, 100, 100, 0, 140, 140, 140],
            'width': [400, 400, 60, 40, 0, 25, 35, 50, 0, 25, 15, 30, 50, 40, 0, 40, 45, 30],
            'height': [200, 200, 30, 30, 0, 25, 25, 25, 0, 25, 25, 25, 25, 25, 0, 25, 25, 25]
        }
        mock_pytesseract.image_to_data.return_value = mock_data
        
        with patch('components.ai_vision.ocr.OCREngine') as MockOCREngine:
            mock_engine = Mock()
            
            # Mock simple text extraction
            mock_engine.extract_text.return_value = mock_pytesseract.image_to_string.return_value
            
            # Mock structured extraction
            def mock_extract_structured(image, page_num=1):
                words = []
                for i, text in enumerate(mock_data['text']):
                    if text.strip() and mock_data['conf'][i] != '-1':
                        bbox = BoundingBox(
                            left=mock_data['left'][i],
                            top=mock_data['top'][i], 
                            width=mock_data['width'][i],
                            height=mock_data['height'][i]
                        )
                        confidence = float(mock_data['conf'][i]) / 100.0
                        word = OCRWord(text=text, confidence=confidence, bounding_box=bbox)
                        words.append(word)
                
                # Group words into lines (simplified)
                lines = []
                current_line_words = []
                current_y = None
                
                for word in words:
                    if current_y is None or abs(word.bounding_box.top - current_y) < 10:
                        current_line_words.append(word)
                        current_y = word.bounding_box.top
                    else:
                        if current_line_words:
                            line_text = ' '.join(w.text for w in current_line_words)
                            line_conf = sum(w.confidence for w in current_line_words) / len(current_line_words)
                            line_bbox = BoundingBox(0, current_y, 400, 25)
                            lines.append(OCRLine(line_text, line_conf, line_bbox, current_line_words))
                        current_line_words = [word]
                        current_y = word.bounding_box.top
                
                # Add last line
                if current_line_words:
                    line_text = ' '.join(w.text for w in current_line_words)
                    line_conf = sum(w.confidence for w in current_line_words) / len(current_line_words)
                    line_bbox = BoundingBox(0, current_y, 400, 25)
                    lines.append(OCRLine(line_text, line_conf, line_bbox, current_line_words))
                
                # Create block
                block_text = '\\n'.join(line.text for line in lines)
                block_conf = sum(line.confidence for line in lines) / len(lines) if lines else 0
                block = OCRBlock(block_text, block_conf, BoundingBox(0, 0, 400, 200), lines)
                
                # Create page
                page = OCRPage(
                    page_number=page_num,
                    text=block_text,
                    confidence=block_conf,
                    bounding_box=BoundingBox(0, 0, 400, 200),
                    blocks=[block],
                    language="en",
                    processing_time=0.2
                )
                return page
            
            mock_engine.extract_structured_data.side_effect = mock_extract_structured
            MockOCREngine.return_value = mock_engine
            
            # Test OCR
            engine = MockOCREngine()
            
            # Test simple text extraction
            text = engine.extract_text(sample_ocr_image)
            assert "SAMPLE TEXT" in text
            assert "For OCR Testing" in text
            assert "12345" in text
            assert "@#$%" in text
            
            # Test structured extraction
            page = engine.extract_structured_data(sample_ocr_image)
            assert page.page_number == 1
            assert page.language == "en"
            assert len(page.blocks) == 1
            assert len(page.blocks[0].lines) > 0
            
            # Check that we extracted meaningful words
            all_words = []
            for line in page.blocks[0].lines:
                all_words.extend(line.words)
            
            word_texts = [w.text for w in all_words]
            assert "SAMPLE" in word_texts
            assert "TEXT" in word_texts
            assert "12345" in word_texts
    
    def test_multilingual_ocr(self, mock_pytesseract):
        """Test OCR with multiple languages"""
        # Create image with mixed text
        img = Image.new('RGB', (400, 150), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((20, 20), "Hello World", fill='black', font=font)
        draw.text((20, 50), "Bonjour le monde", fill='black', font=font)  # French
        draw.text((20, 80), "Hola mundo", fill='black', font=font)  # Spanish
        draw.text((20, 110), "你好世界", fill='black', font=font)  # Chinese (if font supports)
        
        img_array = np.array(img)
        
        # Mock multilingual detection
        mock_pytesseract.image_to_string.return_value = (
            "Hello World\\n"
            "Bonjour le monde\\n"
            "Hola mundo\\n"
            "你好世界"
        )
        
        with patch('components.ai_vision.ocr.OCREngine') as MockOCREngine:
            mock_engine = Mock()
            mock_engine.extract_text.return_value = mock_pytesseract.image_to_string.return_value
            mock_engine.detect_language.return_value = ["en", "fr", "es", "zh"]
            MockOCREngine.return_value = mock_engine
            
            engine = MockOCREngine()
            
            # Test text extraction
            text = engine.extract_text(img_array)
            assert "Hello World" in text
            assert "Bonjour le monde" in text
            assert "Hola mundo" in text
            
            # Test language detection
            languages = engine.detect_language(img_array)
            assert "en" in languages
            assert "fr" in languages
            assert "es" in languages


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.face
class TestFaceRecognitionIntegration:
    """Integration tests for face recognition"""
    
    def test_face_detection_and_recognition(self, sample_face_image, mock_face_recognition, mock_sqlite_db):
        """Test face detection and recognition pipeline"""
        from components.ai_vision.face_recognition import (
            FaceRecognitionResult, FaceDetection, BoundingBox, Identity
        )
        from datetime import datetime
        
        # Mock face recognition results
        mock_face_recognition.face_locations.return_value = [(40, 150, 160, 50)]  # top, right, bottom, left
        mock_face_recognition.face_encodings.return_value = [np.random.random(128)]
        mock_face_recognition.compare_faces.return_value = [True]
        mock_face_recognition.face_distance.return_value = [0.25]
        
        with patch('components.ai_vision.face_recognition.FaceRecognizer') as MockFaceRecognizer:
            mock_recognizer = Mock()
            
            # Mock recognition results
            detection = FaceDetection(
                bbox=BoundingBox(x=50, y=40, width=100, height=120),
                confidence=0.95,
                landmarks={'left_eye': (70, 80), 'right_eye': (115, 80), 'nose': (100, 105)}
            )
            
            result = FaceRecognitionResult(
                face_id="person_001",
                name="John Doe",
                confidence=0.85,
                embedding=np.random.random(128),
                detection=detection,
                metadata={'timestamp': datetime.now().isoformat()}
            )
            
            mock_recognizer.recognize_faces.return_value = [result]
            mock_recognizer.detect_faces.return_value = [detection]
            MockFaceRecognizer.return_value = mock_recognizer
            
            # Test face recognition
            recognizer = MockFaceRecognizer()
            
            results = recognizer.recognize_faces(sample_face_image)
            assert len(results) == 1
            assert results[0].name == "John Doe"
            assert results[0].confidence == 0.85
            assert results[0].face_id == "person_001"
            
            # Test face detection only
            detections = recognizer.detect_faces(sample_face_image)
            assert len(detections) == 1
            assert detections[0].confidence == 0.95
            assert detections[0].bbox.width == 100
            assert detections[0].bbox.height == 120
    
    def test_face_enrollment_and_recognition(self, mock_face_recognition, mock_sqlite_db):
        """Test enrolling a new person and recognizing them"""
        from components.ai_vision.face_recognition import FaceDatabase, Identity
        from datetime import datetime
        
        # Mock multiple face images for the same person
        person_images = [
            self._create_face_image("frontal"),
            self._create_face_image("slight_left"),
            self._create_face_image("slight_right")
        ]
        
        # Mock face encodings for enrollment
        mock_encodings = [np.random.random(128) for _ in range(3)]
        mock_face_recognition.face_encodings.return_value = mock_encodings
        mock_face_recognition.face_locations.return_value = [(40, 150, 160, 50)]  # Same for all
        
        with patch('components.ai_vision.face_recognition.FaceRecognizer') as MockFaceRecognizer:
            with patch('components.ai_vision.face_recognition.FaceDatabase') as MockFaceDatabase:
                # Setup mocks
                mock_recognizer = Mock()
                mock_db = Mock()
                
                # Mock enrollment
                identity = Identity(
                    id="test_person_001",
                    name="Test Person",
                    embeddings=mock_encodings,
                    metadata={"enrolled_date": datetime.now().isoformat()},
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                mock_recognizer.enroll_person.return_value = identity
                mock_db.add_identity.return_value = True
                mock_db.get_identity.return_value = identity
                
                MockFaceRecognizer.return_value = mock_recognizer
                MockFaceDatabase.return_value = mock_db
                
                # Test enrollment
                recognizer = MockFaceRecognizer()
                db = MockFaceDatabase()
                
                enrolled_identity = recognizer.enroll_person("Test Person", person_images)
                assert enrolled_identity.name == "Test Person"
                assert len(enrolled_identity.embeddings) == 3
                
                # Test database storage
                success = db.add_identity(enrolled_identity)
                assert success is True
                
                # Test retrieval
                retrieved = db.get_identity("test_person_001")
                assert retrieved.name == "Test Person"
    
    def _create_face_image(self, pose="frontal"):
        """Create face images with different poses"""
        img = Image.new('RGB', (200, 200), color='peachpuff')
        draw = ImageDraw.Draw(img)
        
        if pose == "frontal":
            # Centered face
            draw.ellipse([50, 40, 150, 160], fill='peachpuff', outline='brown', width=2)
            draw.ellipse([70, 80, 85, 95], fill='white', outline='black')
            draw.ellipse([115, 80, 130, 95], fill='white', outline='black')
        elif pose == "slight_left":
            # Face turned slightly left
            draw.ellipse([45, 40, 145, 160], fill='peachpuff', outline='brown', width=2)
            draw.ellipse([65, 80, 80, 95], fill='white', outline='black')
            draw.ellipse([110, 80, 125, 95], fill='white', outline='black')
        elif pose == "slight_right":
            # Face turned slightly right  
            draw.ellipse([55, 40, 155, 160], fill='peachpuff', outline='brown', width=2)
            draw.ellipse([75, 80, 90, 95], fill='white', outline='black')
            draw.ellipse([120, 80, 135, 95], fill='white', outline='black')
        
        # Common features
        draw.ellipse([95, 105, 105, 115], fill='brown')  # Nose
        draw.arc([80, 125, 120, 145], start=0, end=180, fill='red', width=3)  # Mouth
        
        return np.array(img)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.analytics
class TestVisionAnalyticsIntegration:
    """Integration tests for computer vision analytics"""
    
    @pytest.mark.asyncio
    async def test_motion_detection_sequence(self):
        """Test motion detection across a sequence of frames"""
        from components.ai_vision.computer_vision_analytics import MotionDetector, MotionResult
        
        # Create sequence showing motion
        frames = []
        for i in range(6):
            frame = self._create_motion_frame(i)
            frames.append(frame)
        
        with patch('components.ai_vision.computer_vision_analytics.MotionDetector') as MockMotionDetector:
            mock_detector = Mock()
            
            # Mock motion detection results
            def mock_detect_motion(current_frame, previous_frame=None):
                if previous_frame is None:
                    return MotionResult(
                        has_motion=False,
                        motion_areas=[],
                        motion_percentage=0.0,
                        confidence=1.0,
                        metadata={}
                    )
                else:
                    # Simulate detecting motion
                    return MotionResult(
                        has_motion=True,
                        motion_areas=[(100, 100, 200, 200)],  # Motion area
                        motion_percentage=15.5,
                        confidence=0.92,
                        metadata={'frame_diff_pixels': 2500}
                    )
            
            mock_detector.detect_motion.side_effect = mock_detect_motion
            MockMotionDetector.return_value = mock_detector
            
            # Test motion detection
            detector = MockMotionDetector()
            
            previous_frame = None
            motion_results = []
            
            for frame in frames:
                result = detector.detect_motion(frame, previous_frame)
                motion_results.append(result)
                previous_frame = frame
            
            # First frame should show no motion (no previous frame)
            assert motion_results[0].has_motion is False
            
            # Subsequent frames should show motion
            for i in range(1, len(motion_results)):
                assert motion_results[i].has_motion is True
                assert motion_results[i].motion_percentage > 0
                assert len(motion_results[i].motion_areas) > 0
    
    def _create_motion_frame(self, frame_idx):
        """Create frames with moving objects for motion detection"""
        img = Image.new('RGB', (320, 240), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Static background elements
        draw.rectangle([0, 200, 320, 240], fill='green')  # Ground
        draw.rectangle([20, 150, 60, 200], fill='brown')  # Tree trunk
        draw.ellipse([10, 100, 70, 160], fill='darkgreen')  # Tree crown
        
        # Moving object (ball)
        ball_x = 50 + frame_idx * 30  # Moves right
        ball_y = 180 - abs(frame_idx - 3) * 10  # Bounces
        draw.ellipse([ball_x, ball_y, ball_x + 20, ball_y + 20], fill='red', outline='darkred')
        
        # Moving person
        person_x = 100 + frame_idx * 15
        draw.ellipse([person_x, 160, person_x + 20, 180], fill='peachpuff')  # Head
        draw.rectangle([person_x + 5, 180, person_x + 15, 210], fill='blue')  # Body
        
        return np.array(img)
    
    @pytest.mark.asyncio
    async def test_scene_analytics_pipeline(self):
        """Test complete scene analytics pipeline"""
        from components.ai_vision.computer_vision_analytics import (
            ComputerVisionAnalyticsAgent, AnalyticsEvent, AnalyticsEventType
        )
        
        # Create scene with multiple events
        scene_frames = [
            self._create_scene_frame("empty_room"),
            self._create_scene_frame("person_enters"),
            self._create_scene_frame("person_moves"),
            self._create_scene_frame("door_opens"),
            self._create_scene_frame("person_exits")
        ]
        
        with patch('components.ai_vision.computer_vision_analytics.ComputerVisionAnalyticsAgent') as MockAgent:
            mock_agent = Mock()
            
            # Mock analytics events
            events_sequence = [
                [],  # Empty room - no events
                [AnalyticsEvent(AnalyticsEventType.PERSON_DETECTED, 0.95, {'person_id': 'p001'})],
                [AnalyticsEvent(AnalyticsEventType.MOTION_DETECTED, 0.88, {'motion_area': [100, 100, 200, 200]})],
                [AnalyticsEvent(AnalyticsEventType.DOOR_OPENED, 0.92, {'door_id': 'main_entrance'})],
                [AnalyticsEvent(AnalyticsEventType.PERSON_LEFT, 0.90, {'person_id': 'p001'})]
            ]
            
            def mock_analyze_frame(frame):
                frame_idx = len([f for f in scene_frames if np.array_equal(f, frame)])
                if frame_idx < len(events_sequence):
                    return events_sequence[frame_idx]
                return []
            
            mock_agent.analyze_frame.side_effect = mock_analyze_frame
            MockAgent.return_value = mock_agent
            
            # Test analytics pipeline
            agent = MockAgent()
            
            all_events = []
            for frame in scene_frames:
                events = agent.analyze_frame(frame)
                all_events.extend(events)
            
            # Verify event sequence
            event_types = [event.event_type for event in all_events]
            assert AnalyticsEventType.PERSON_DETECTED in event_types
            assert AnalyticsEventType.MOTION_DETECTED in event_types
            assert AnalyticsEventType.DOOR_OPENED in event_types
            assert AnalyticsEventType.PERSON_LEFT in event_types
            
            # Verify event confidences
            assert all(event.confidence > 0.8 for event in all_events)
    
    def _create_scene_frame(self, scene_type):
        """Create different scene frames for analytics testing"""
        img = Image.new('RGB', (640, 480), color='white')
        draw = ImageDraw.Draw(img)
        
        # Common room elements
        draw.rectangle([0, 0, 640, 480], fill='lightgray')  # Floor
        draw.rectangle([580, 100, 640, 300], fill='brown')  # Door
        
        if scene_type == "empty_room":
            # Just the room
            pass
        elif scene_type == "person_enters":
            # Person near door
            draw.ellipse([570, 150, 590, 170], fill='peachpuff')  # Head
            draw.rectangle([575, 170, 585, 220], fill='blue')  # Body
        elif scene_type == "person_moves":
            # Person in middle of room
            draw.ellipse([320, 200, 340, 220], fill='peachpuff')  # Head
            draw.rectangle([325, 220, 335, 270], fill='blue')  # Body
        elif scene_type == "door_opens":
            # Door appears open, person nearby
            draw.rectangle([540, 100, 580, 300], fill='darkgray')  # Open door
            draw.rectangle([580, 100, 640, 300], fill='brown')  # Door frame
            draw.ellipse([520, 150, 540, 170], fill='peachpuff')  # Head
            draw.rectangle([525, 170, 535, 220], fill='blue')  # Body
        elif scene_type == "person_exits":
            # Door open, no person visible
            draw.rectangle([540, 100, 580, 300], fill='darkgray')  # Open door
        
        return np.array(img)
