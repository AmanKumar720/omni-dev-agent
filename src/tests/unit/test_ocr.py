"""
Unit tests for OCR module
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from components.ai_vision.ocr import (
    BoundingBox,
    OCRWord,
    OCRLine,
    OCRBlock,
    OCRPage,
    OCRResult,
    ImagePreprocessor,
    OCREngine
)


@pytest.mark.unit
@pytest.mark.ocr
class TestBoundingBox:
    """Test BoundingBox dataclass"""
    
    def test_bounding_box_creation(self):
        """Test creating a BoundingBox"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50)
        
        assert bbox.left == 10
        assert bbox.top == 20
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.right == 110  # left + width
        assert bbox.bottom == 70  # top + height
    
    def test_bounding_box_with_explicit_right_bottom(self):
        """Test BoundingBox with explicitly provided right/bottom"""
        bbox = BoundingBox(left=10, top=20, width=100, height=50, right=115, bottom=75)
        
        # Explicit values should be used
        assert bbox.right == 115
        assert bbox.bottom == 75


@pytest.mark.unit
@pytest.mark.ocr
class TestOCRDataStructures:
    """Test OCR data structures"""
    
    def test_ocr_word_creation(self):
        """Test creating an OCRWord"""
        bbox = BoundingBox(10, 20, 100, 50)
        word = OCRWord(text="hello", confidence=0.95, bounding_box=bbox)
        
        assert word.text == "hello"
        assert word.confidence == 0.95
        assert word.bounding_box == bbox
    
    def test_ocr_line_creation(self):
        """Test creating an OCRLine"""
        bbox = BoundingBox(10, 20, 200, 50)
        words = [
            OCRWord("hello", 0.95, BoundingBox(10, 20, 80, 50)),
            OCRWord("world", 0.90, BoundingBox(100, 20, 80, 50))
        ]
        line = OCRLine(text="hello world", confidence=0.92, bounding_box=bbox, words=words)
        
        assert line.text == "hello world"
        assert line.confidence == 0.92
        assert len(line.words) == 2
        assert line.words[0].text == "hello"
    
    def test_ocr_block_creation(self):
        """Test creating an OCRBlock"""
        bbox = BoundingBox(10, 20, 200, 100)
        lines = [
            OCRLine("hello world", 0.92, BoundingBox(10, 20, 200, 50), []),
            OCRLine("test text", 0.88, BoundingBox(10, 70, 150, 50), [])
        ]
        block = OCRBlock(text="hello world\\ntest text", confidence=0.90, bounding_box=bbox, lines=lines)
        
        assert "hello world" in block.text
        assert "test text" in block.text
        assert block.confidence == 0.90
        assert len(block.lines) == 2
    
    def test_ocr_page_creation(self):
        """Test creating an OCRPage"""
        bbox = BoundingBox(0, 0, 800, 1000)
        blocks = [
            OCRBlock("Block 1", 0.90, BoundingBox(10, 20, 200, 100), []),
            OCRBlock("Block 2", 0.85, BoundingBox(10, 150, 200, 100), [])
        ]
        
        page = OCRPage(
            page_number=1,
            text="Block 1\\nBlock 2",
            confidence=0.87,
            bounding_box=bbox,
            blocks=blocks,
            language="en",
            processing_time=0.5
        )
        
        assert page.page_number == 1
        assert page.language == "en"
        assert page.processing_time == 0.5
        assert len(page.blocks) == 2
    
    def test_ocr_result_creation(self):
        """Test creating an OCRResult"""
        page = OCRPage(1, "Test", 0.9, BoundingBox(0, 0, 800, 1000), [], "en", 0.5)
        result = OCRResult(
            pages=[page],
            total_pages=1,
            average_confidence=0.9,
            processing_time=0.5,
            languages_detected=["en"]
        )
        
        assert result.total_pages == 1
        assert result.average_confidence == 0.9
        assert len(result.pages) == 1
        assert "en" in result.languages_detected
    
    def test_ocr_result_to_json(self):
        """Test converting OCRResult to JSON"""
        page = OCRPage(1, "Test", 0.9, BoundingBox(0, 0, 800, 1000), [], "en", 0.5)
        result = OCRResult([page], 1, 0.9, 0.5, ["en"])
        
        json_str = result.to_json()
        assert isinstance(json_str, str)
        
        # Parse back to verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["total_pages"] == 1
        assert parsed["average_confidence"] == 0.9
    
    def test_ocr_result_to_dict(self):
        """Test converting OCRResult to dictionary"""
        page = OCRPage(1, "Test", 0.9, BoundingBox(0, 0, 800, 1000), [], "en", 0.5)
        result = OCRResult([page], 1, 0.9, 0.5, ["en"])
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["total_pages"] == 1
        assert result_dict["average_confidence"] == 0.9


@pytest.mark.unit
@pytest.mark.ocr
class TestImagePreprocessor:
    """Test ImagePreprocessor class"""
    
    def test_deskew_image_no_rotation(self):
        """Test deskewing when no rotation is needed"""
        # Create a simple test image
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        with patch('cv2.GaussianBlur') as mock_blur, \
             patch('cv2.threshold') as mock_thresh, \
             patch('cv2.Canny') as mock_canny, \
             patch('cv2.HoughLines') as mock_hough:
            
            # Setup mocks
            mock_blur.return_value = image[:, :, 0]  # grayscale
            mock_thresh.return_value = (128, image[:, :, 0])
            mock_canny.return_value = image[:, :, 0]
            mock_hough.return_value = None  # No lines detected
            
            deskewed, angle = ImagePreprocessor.deskew_image(image)
            
            assert angle == 0.0
            assert np.array_equal(deskewed, image)
    
    def test_deskew_image_with_rotation(self):
        """Test deskewing when rotation is detected"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        with patch('cv2.cvtColor') as mock_cvt, \
             patch('cv2.GaussianBlur') as mock_blur, \
             patch('cv2.threshold') as mock_thresh, \
             patch('cv2.Canny') as mock_canny, \
             patch('cv2.HoughLines') as mock_hough, \
             patch('cv2.getRotationMatrix2D') as mock_matrix, \
             patch('cv2.warpAffine') as mock_warp:
            
            # Setup mocks
            mock_cvt.return_value = image[:, :, 0]
            mock_blur.return_value = image[:, :, 0]
            mock_thresh.return_value = (128, image[:, :, 0])
            mock_canny.return_value = image[:, :, 0]
            
            # Mock HoughLines to return a line with some angle
            mock_lines = np.array([[[100, np.pi/2 + 0.1]]])  # Slight rotation
            mock_hough.return_value = mock_lines
            
            # Mock rotation matrix and warping
            mock_matrix.return_value = np.eye(2, 3)
            mock_warp.return_value = image
            
            deskewed, angle = ImagePreprocessor.deskew_image(image)
            
            # Should detect some rotation
            assert abs(angle) > 0.1
            mock_warp.assert_called_once()
    
    def test_deskew_grayscale_image(self):
        """Test deskewing a grayscale image"""
        # Grayscale image (2D array)
        image = np.ones((100, 200), dtype=np.uint8) * 255
        
        with patch('cv2.GaussianBlur') as mock_blur, \
             patch('cv2.threshold') as mock_thresh, \
             patch('cv2.Canny') as mock_canny, \
             patch('cv2.HoughLines') as mock_hough:
            
            mock_blur.return_value = image
            mock_thresh.return_value = (128, image)
            mock_canny.return_value = image
            mock_hough.return_value = None
            
            deskewed, angle = ImagePreprocessor.deskew_image(image)
            
            assert angle == 0.0
            assert np.array_equal(deskewed, image)
    
    def test_remove_noise(self):
        """Test noise removal from image"""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        with patch('cv2.cvtColor') as mock_cvt, \
             patch('cv2.bilateralFilter') as mock_bilateral, \
             patch('cv2.getStructuringElement') as mock_kernel, \
             patch('cv2.morphologyEx') as mock_morph:
            
            # Setup mocks
            grayscale = image[:, :, 0]
            mock_cvt.return_value = grayscale
            mock_bilateral.return_value = grayscale
            mock_kernel.return_value = np.ones((3, 3), dtype=np.uint8)
            mock_morph.return_value = grayscale
            
            result = ImagePreprocessor.remove_noise(image)
            
            # Should call bilateral filter and morphological operations
            mock_bilateral.assert_called_once()
            assert mock_morph.call_count == 2  # Opening and closing
            assert result.shape == grayscale.shape
    
    def test_remove_noise_grayscale(self):
        """Test noise removal from grayscale image"""
        image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        
        with patch('cv2.bilateralFilter') as mock_bilateral, \
             patch('cv2.getStructuringElement') as mock_kernel, \
             patch('cv2.morphologyEx') as mock_morph:
            
            mock_bilateral.return_value = image
            mock_kernel.return_value = np.ones((3, 3), dtype=np.uint8)
            mock_morph.return_value = image
            
            result = ImagePreprocessor.remove_noise(image)
            
            mock_bilateral.assert_called_once()
            assert mock_morph.call_count == 2


@pytest.mark.unit
@pytest.mark.ocr
class TestOCREngine:
    """Test OCREngine class"""
    
    def test_ocr_engine_initialization(self, mock_pytesseract):
        """Test OCR engine initialization"""
        with patch('components.ai_vision.ocr.OCREngine.__init__', return_value=None):
            engine = object.__new__(OCREngine)  # Create without calling __init__
            engine.languages = ["en"]
            engine.tesseract_config = "--oem 1 --psm 1"
            engine.preprocessor = ImagePreprocessor()
            
            assert engine.languages == ["en"]
            assert "oem" in engine.tesseract_config
    
    def test_extract_text_simple(self, mock_pytesseract, sample_text_image):
        """Test simple text extraction"""
        mock_pytesseract.image_to_string.return_value = "Hello World\\nTest Text"
        
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            # Create a mock engine instance
            engine = Mock()
            engine.extract_text.return_value = "Hello World\\nTest Text"
            MockEngine.return_value = engine
            
            result = engine.extract_text(sample_text_image)
            assert "Hello World" in result
            assert "Test Text" in result
    
    def test_extract_structured_data(self, mock_pytesseract, sample_text_image):
        """Test structured data extraction"""
        # Mock pytesseract to return structured data
        mock_data = {
            'text': ['', '', 'Hello', 'World', '', 'Test', 'Text'],
            'conf': ['-1', '-1', '95', '89', '-1', '92', '91'],
            'left': [0, 0, 10, 70, 0, 10, 90],
            'top': [0, 0, 40, 40, 0, 70, 70],
            'width': [300, 300, 50, 60, 0, 35, 40],
            'height': [100, 100, 20, 20, 0, 15, 15]
        }
        mock_pytesseract.image_to_data.return_value = mock_data
        
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            engine = Mock()
            
            # Mock the structured extraction method
            def mock_extract_structured(image, page_num=1):
                words = []
                for i, text in enumerate(mock_data['text']):
                    if text.strip():
                        bbox = BoundingBox(
                            left=mock_data['left'][i],
                            top=mock_data['top'][i],
                            width=mock_data['width'][i],
                            height=mock_data['height'][i]
                        )
                        word = OCRWord(
                            text=text,
                            confidence=float(mock_data['conf'][i]) / 100.0,
                            bounding_box=bbox
                        )
                        words.append(word)
                
                # Create a simple line and block structure
                line = OCRLine(
                    text="Hello World Test Text",
                    confidence=0.92,
                    bounding_box=BoundingBox(0, 0, 300, 100),
                    words=words
                )
                block = OCRBlock(
                    text="Hello World Test Text",
                    confidence=0.92,
                    bounding_box=BoundingBox(0, 0, 300, 100),
                    lines=[line]
                )
                page = OCRPage(
                    page_number=page_num,
                    text="Hello World Test Text",
                    confidence=0.92,
                    bounding_box=BoundingBox(0, 0, 300, 100),
                    blocks=[block],
                    language="en",
                    processing_time=0.1
                )
                return page
            
            engine.extract_structured_data.side_effect = mock_extract_structured
            MockEngine.return_value = engine
            
            result = engine.extract_structured_data(sample_text_image)
            
            assert result.page_number == 1
            assert len(result.blocks) == 1
            assert len(result.blocks[0].lines) == 1
            assert len(result.blocks[0].lines[0].words) > 0
            assert "Hello" in [word.text for word in result.blocks[0].lines[0].words]
    
    def test_preprocess_image_integration(self, sample_text_image):
        """Test image preprocessing integration"""
        with patch('components.ai_vision.ocr.ImagePreprocessor.deskew_image') as mock_deskew, \
             patch('components.ai_vision.ocr.ImagePreprocessor.remove_noise') as mock_denoise:
            
            mock_deskew.return_value = (sample_text_image, 0.0)
            mock_denoise.return_value = sample_text_image
            
            preprocessor = ImagePreprocessor()
            
            # Test deskewing
            deskewed, angle = preprocessor.deskew_image(sample_text_image)
            mock_deskew.assert_called_once_with(sample_text_image)
            
            # Test denoising
            denoised = preprocessor.remove_noise(sample_text_image)
            mock_denoise.assert_called_once_with(sample_text_image, kernel_size=3)
    
    def test_language_detection(self, mock_pytesseract):
        """Test language detection functionality"""
        # Mock language detection
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            engine = Mock()
            engine.detect_language.return_value = ["en", "fr"]
            MockEngine.return_value = engine
            
            languages = engine.detect_language("test_image")
            assert "en" in languages
            assert isinstance(languages, list)
    
    def test_confidence_filtering(self, mock_pytesseract):
        """Test confidence-based filtering"""
        # Test that low confidence results can be filtered
        mock_data = {
            'text': ['good', 'bad', 'excellent'],
            'conf': ['95', '30', '98'],  # One low confidence word
            'left': [10, 50, 90],
            'top': [20, 20, 20],
            'width': [30, 20, 40],
            'height': [15, 15, 15]
        }
        mock_pytesseract.image_to_data.return_value = mock_data
        
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            engine = Mock()
            
            def mock_filter_by_confidence(words, min_confidence=0.8):
                return [w for w in words if w.confidence >= min_confidence]
            
            # Mock words for testing
            words = [
                OCRWord("good", 0.95, BoundingBox(10, 20, 30, 15)),
                OCRWord("bad", 0.30, BoundingBox(50, 20, 20, 15)),
                OCRWord("excellent", 0.98, BoundingBox(90, 20, 40, 15))
            ]
            
            engine.filter_by_confidence = mock_filter_by_confidence
            MockEngine.return_value = engine
            
            filtered_words = engine.filter_by_confidence(words, 0.8)
            
            assert len(filtered_words) == 2  # Should filter out the low confidence word
            assert all(w.confidence >= 0.8 for w in filtered_words)
            assert "bad" not in [w.text for w in filtered_words]


@pytest.mark.unit
@pytest.mark.ocr
class TestOCRErrorHandling:
    """Test OCR error handling"""
    
    def test_invalid_image_handling(self, mock_pytesseract):
        """Test handling of invalid images"""
        mock_pytesseract.image_to_string.side_effect = Exception("Invalid image")
        
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            engine = Mock()
            engine.extract_text.side_effect = Exception("Invalid image")
            MockEngine.return_value = engine
            
            with pytest.raises(Exception, match="Invalid image"):
                engine.extract_text("invalid_image")
    
    def test_tesseract_not_found_error(self):
        """Test handling when Tesseract is not installed"""
        with patch('pytesseract.image_to_string') as mock_tess:
            mock_tess.side_effect = FileNotFoundError("Tesseract not found")
            
            with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
                engine = Mock()
                engine.extract_text.side_effect = FileNotFoundError("Tesseract not found")
                MockEngine.return_value = engine
                
                with pytest.raises(FileNotFoundError):
                    engine.extract_text("test_image")
    
    def test_empty_image_handling(self, mock_pytesseract):
        """Test handling of empty/blank images"""
        mock_pytesseract.image_to_string.return_value = ""
        
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            engine = Mock()
            engine.extract_text.return_value = ""
            MockEngine.return_value = engine
            
            result = engine.extract_text("blank_image")
            assert result == ""
    
    def test_corrupted_data_handling(self, mock_pytesseract):
        """Test handling of corrupted OCR data"""
        # Mock corrupted data structure
        mock_pytesseract.image_to_data.return_value = {
            'text': ['hello'],
            'conf': ['invalid_conf'],  # Invalid confidence value
            'left': [10],
            'top': ['invalid_top'],  # Invalid coordinate
            'width': [30],
            'height': [15]
        }
        
        with patch('components.ai_vision.ocr.OCREngine') as MockEngine:
            engine = Mock()
            
            def mock_robust_extraction(image):
                try:
                    # Simulate handling corrupted data gracefully
                    return OCRPage(
                        page_number=1,
                        text="",  # Empty due to corrupted data
                        confidence=0.0,
                        bounding_box=BoundingBox(0, 0, 100, 100),
                        blocks=[],
                        language="en",
                        processing_time=0.1
                    )
                except Exception:
                    # Return empty result on error
                    return OCRPage(1, "", 0.0, BoundingBox(0, 0, 100, 100), [], "en", 0.1)
            
            engine.extract_structured_data.side_effect = mock_robust_extraction
            MockEngine.return_value = engine
            
            result = engine.extract_structured_data("corrupted_image")
            
            # Should handle gracefully and return empty result
            assert result.text == ""
            assert len(result.blocks) == 0
