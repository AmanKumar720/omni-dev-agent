"""
OCR Module with Pre- and Post-processing

This module provides a comprehensive wrapper around pytesseract with:
- Image pre-processing (de-skew, noise reduction)
- Language selection
- Structured JSON output (text, bounding boxes, confidence, page numbers)
- Multi-page document support
"""

import json
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from scipy import ndimage
from skimage import filters
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    left: int
    top: int
    width: int
    height: int
    right: int = None
    bottom: int = None
    
    def __post_init__(self):
        if self.right is None:
            self.right = self.left + self.width
        if self.bottom is None:
            self.bottom = self.top + self.height


@dataclass
class OCRWord:
    """Represents a single OCR word result."""
    text: str
    confidence: float
    bounding_box: BoundingBox


@dataclass
class OCRLine:
    """Represents a line of OCR text."""
    text: str
    confidence: float
    bounding_box: BoundingBox
    words: List[OCRWord]


@dataclass
class OCRBlock:
    """Represents a block of OCR text."""
    text: str
    confidence: float
    bounding_box: BoundingBox
    lines: List[OCRLine]


@dataclass
class OCRPage:
    """Represents a complete OCR page result."""
    page_number: int
    text: str
    confidence: float
    bounding_box: BoundingBox
    blocks: List[OCRBlock]
    language: str
    processing_time: float


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    pages: List[OCRPage]
    total_pages: int
    average_confidence: float
    processing_time: float
    languages_detected: List[str]
    
    def to_json(self) -> str:
        """Convert OCR result to JSON string."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict:
        """Convert OCR result to dictionary."""
        return asdict(self)


class ImagePreprocessor:
    """Handles image preprocessing for better OCR results."""
    
    @staticmethod
    def deskew_image(image: np.ndarray, angle_range: int = 45) -> Tuple[np.ndarray, float]:
        """
        Deskew an image by detecting and correcting text rotation.
        
        Args:
            image: Input image as numpy array
            angle_range: Range of angles to check for rotation
            
        Returns:
            Tuple of (deskewed_image, detected_angle)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Hough line transform to detect dominant angles
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if abs(angle) <= angle_range:
                    angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # Calculate the median angle
        median_angle = np.median(angles)
        
        # Rotate the image
        if abs(median_angle) > 0.1:  # Only rotate if angle is significant
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            # Calculate new dimensions
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            
            # Adjust rotation matrix for new dimensions
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            deskewed = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed, median_angle
        
        return image, 0.0
    
    @staticmethod
    def remove_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Remove noise from image using morphological operations.
        
        Args:
            image: Input image
            kernel_size: Size of morphological kernel
            
        Returns:
            Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Opening to remove small noise
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    @staticmethod
    def preprocess_image(image: Union[np.ndarray, str, Path], 
                        deskew: bool = True,
                        denoise: bool = True, 
                        enhance_contrast: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply complete preprocessing pipeline to an image.
        
        Args:
            image: Input image (array, path, or PIL Image)
            deskew: Whether to apply deskewing
            denoise: Whether to apply denoising
            enhance_contrast: Whether to enhance contrast
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        processing_info = {
            'deskew_angle': 0.0,
            'denoised': False,
            'contrast_enhanced': False,
            'original_shape': None,
            'final_shape': None
        }
        
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        processing_info['original_shape'] = img.shape
        
        # Apply deskewing
        if deskew:
            img, angle = ImagePreprocessor.deskew_image(img)
            processing_info['deskew_angle'] = angle
            logger.info(f"Deskewed image by {angle:.2f} degrees")
        
        # Apply denoising
        if denoise:
            img = ImagePreprocessor.remove_noise(img)
            processing_info['denoised'] = True
            logger.info("Applied noise reduction")
        
        # Enhance contrast
        if enhance_contrast:
            img = ImagePreprocessor.enhance_contrast(img)
            processing_info['contrast_enhanced'] = True
            logger.info("Enhanced image contrast")
        
        processing_info['final_shape'] = img.shape
        
        return img, processing_info


class OCREngine:
    """Main OCR engine with pytesseract wrapper and advanced features."""
    
    SUPPORTED_LANGUAGES = {
        'eng': 'English',
        'spa': 'Spanish', 
        'fra': 'French',
        'deu': 'German',
        'ita': 'Italian',
        'por': 'Portuguese',
        'rus': 'Russian',
        'jpn': 'Japanese',
        'chi_sim': 'Chinese Simplified',
        'chi_tra': 'Chinese Traditional',
        'ara': 'Arabic',
        'hin': 'Hindi',
        'kor': 'Korean'
    }
    
    def __init__(self, 
                 languages: List[str] = None,
                 config: str = None,
                 preprocess: bool = True):
        """
        Initialize OCR Engine.
        
        Args:
            languages: List of language codes to use for OCR
            config: Custom tesseract config string
            preprocess: Whether to apply preprocessing by default
        """
        self.languages = languages or ['eng']
        self.config = config or '--oem 3 --psm 6'
        self.preprocess = preprocess
        self.preprocessor = ImagePreprocessor()
        
        # Validate languages
        for lang in self.languages:
            if lang not in self.SUPPORTED_LANGUAGES:
                logger.warning(f"Language '{lang}' may not be supported")
        
        logger.info(f"OCR Engine initialized with languages: {self.languages}")
    
    def _create_bounding_box(self, left: int, top: int, width: int, height: int) -> BoundingBox:
        """Create a BoundingBox object from coordinates."""
        return BoundingBox(left=left, top=top, width=width, height=height)
    
    def _extract_detailed_data(self, image: np.ndarray, page_num: int = 1) -> OCRPage:
        """
        Extract detailed OCR data with hierarchical structure.
        
        Args:
            image: Preprocessed image
            page_num: Page number
            
        Returns:
            OCRPage object with detailed results
        """
        import time
        start_time = time.time()
        
        # Get language string for tesseract
        lang_string = '+'.join(self.languages)
        
        # Extract data with pytesseract
        custom_config = f'-l {lang_string} {self.config}'
        
        # Get detailed data
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Get full text for the page
        page_text = pytesseract.image_to_string(image, config=custom_config)
        
        # Process hierarchical data
        blocks = []
        current_block = None
        current_line = None
        
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            level = data['level'][i]
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if conf < 0:  # Skip invalid detections
                continue
                
            left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            bbox = self._create_bounding_box(left, top, width, height)
            
            if level == 2:  # Block level
                if current_block:
                    blocks.append(current_block)
                current_block = OCRBlock(
                    text="",
                    confidence=0.0,
                    bounding_box=bbox,
                    lines=[]
                )
                current_line = None
                
            elif level == 4:  # Line level
                if current_block and text:
                    if current_line:
                        current_block.lines.append(current_line)
                    current_line = OCRLine(
                        text="",
                        confidence=0.0,
                        bounding_box=bbox,
                        words=[]
                    )
                    
            elif level == 5 and text:  # Word level
                if current_line:
                    word = OCRWord(
                        text=text,
                        confidence=conf,
                        bounding_box=bbox
                    )
                    current_line.words.append(word)
        
        # Add the last line and block
        if current_line and current_block:
            current_block.lines.append(current_line)
        if current_block:
            blocks.append(current_block)
        
        # Calculate aggregated data for lines and blocks
        for block in blocks:
            block_text_parts = []
            block_confidences = []
            
            for line in block.lines:
                line_text_parts = []
                line_confidences = []
                
                for word in line.words:
                    line_text_parts.append(word.text)
                    line_confidences.append(word.confidence)
                
                line.text = ' '.join(line_text_parts)
                line.confidence = np.mean(line_confidences) if line_confidences else 0.0
                
                if line.text.strip():
                    block_text_parts.append(line.text)
                    block_confidences.append(line.confidence)
            
            block.text = '\n'.join(block_text_parts)
            block.confidence = np.mean(block_confidences) if block_confidences else 0.0
        
        # Calculate page-level statistics
        all_confidences = []
        for block in blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence > 0:
                        all_confidences.append(word.confidence)
        
        page_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        # Create page bounding box (entire image)
        height, width = image.shape[:2]
        page_bbox = self._create_bounding_box(0, 0, width, height)
        
        processing_time = time.time() - start_time
        
        return OCRPage(
            page_number=page_num,
            text=page_text.strip(),
            confidence=page_confidence,
            bounding_box=page_bbox,
            blocks=blocks,
            language='+'.join(self.languages),
            processing_time=processing_time
        )
    
    def extract_text(self, 
                    image_path: Union[str, Path, np.ndarray, Image.Image],
                    preprocess: Optional[bool] = None,
                    return_json: bool = False) -> Union[OCRResult, str]:
        """
        Extract text from image with comprehensive processing.
        
        Args:
            image_path: Path to image file, numpy array, or PIL Image
            preprocess: Override preprocessing setting
            return_json: Whether to return JSON string instead of OCRResult object
            
        Returns:
            OCRResult object or JSON string
        """
        import time
        start_time = time.time()
        
        # Determine preprocessing setting
        should_preprocess = preprocess if preprocess is not None else self.preprocess
        
        # Load and preprocess image
        if should_preprocess:
            processed_image, processing_info = self.preprocessor.preprocess_image(image_path)
            logger.info(f"Preprocessing completed: {processing_info}")
        else:
            # Load image without preprocessing
            if isinstance(image_path, (str, Path)):
                processed_image = cv2.imread(str(image_path))
                if processed_image is None:
                    raise ValueError(f"Could not load image from {image_path}")
            elif isinstance(image_path, Image.Image):
                processed_image = np.array(image_path)
                if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            else:
                processed_image = image_path
        
        # Extract OCR data
        page_result = self._extract_detailed_data(processed_image, page_num=1)
        
        # Create final result
        total_time = time.time() - start_time
        
        result = OCRResult(
            pages=[page_result],
            total_pages=1,
            average_confidence=page_result.confidence,
            processing_time=total_time,
            languages_detected=self.languages
        )
        
        logger.info(f"OCR completed in {total_time:.2f}s with confidence {page_result.confidence:.1f}%")
        
        if return_json:
            return result.to_json()
        
        return result
    
    def extract_text_batch(self,
                          image_paths: List[Union[str, Path]],
                          preprocess: Optional[bool] = None,
                          return_json: bool = False) -> Union[List[OCRResult], str]:
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image paths
            preprocess: Override preprocessing setting
            return_json: Whether to return JSON string
            
        Returns:
            List of OCRResult objects or JSON string
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            try:
                result = self.extract_text(image_path, preprocess=preprocess)
                # Update page number for batch processing
                result.pages[0].page_number = i + 1
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        if return_json:
            return json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False)
        
        return results


def create_ocr_engine(languages: List[str] = None,
                     config: str = None,
                     preprocess: bool = True) -> OCREngine:
    """
    Factory function to create OCR engine instance.
    
    Args:
        languages: List of language codes
        config: Custom tesseract configuration
        preprocess: Whether to enable preprocessing
        
    Returns:
        OCREngine instance
    """
    return OCREngine(languages=languages, config=config, preprocess=preprocess)


# Convenience functions
def extract_text_from_image(image_path: Union[str, Path],
                           languages: List[str] = None,
                           preprocess: bool = True,
                           return_json: bool = False) -> Union[OCRResult, str]:
    """
    Convenience function to extract text from a single image.
    
    Args:
        image_path: Path to the image
        languages: List of language codes to use
        preprocess: Whether to apply preprocessing
        return_json: Whether to return JSON string
        
    Returns:
        OCRResult object or JSON string
    """
    engine = create_ocr_engine(languages=languages, preprocess=preprocess)
    return engine.extract_text(image_path, return_json=return_json)


def extract_text_from_images(image_paths: List[Union[str, Path]],
                           languages: List[str] = None,
                           preprocess: bool = True,
                           return_json: bool = False) -> Union[List[OCRResult], str]:
    """
    Convenience function to extract text from multiple images.
    
    Args:
        image_paths: List of image paths
        languages: List of language codes to use
        preprocess: Whether to apply preprocessing
        return_json: Whether to return JSON string
        
    Returns:
        List of OCRResult objects or JSON string
    """
    engine = create_ocr_engine(languages=languages, preprocess=preprocess)
    return engine.extract_text_batch(image_paths, return_json=return_json)


if __name__ == "__main__":
    # Example usage
    logger.info("OCR Module loaded successfully")
    
    # Print supported languages
    print("Supported Languages:")
    for code, name in OCREngine.SUPPORTED_LANGUAGES.items():
        print(f"  {code}: {name}")
