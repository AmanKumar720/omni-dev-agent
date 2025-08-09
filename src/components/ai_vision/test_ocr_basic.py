#!/usr/bin/env python3
"""
Basic tests and examples for OCR module functionality.

This script demonstrates:
- OCR engine initialization
- Image preprocessing
- Text extraction with structured output
- Multi-language support
- Batch processing
- JSON output format
"""

import json
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path

# Import OCR module
from ocr import (
    OCREngine, 
    ImagePreprocessor, 
    create_ocr_engine,
    extract_text_from_image,
    extract_text_from_images
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_images():
    """Create sample test images for OCR testing."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Test image 1: Simple English text
    img1 = Image.new('RGB', (400, 200), color='white')
    draw1 = ImageDraw.Draw(img1)
    
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    draw1.text((50, 50), "Hello World!", fill='black', font=font)
    draw1.text((50, 80), "This is a test image", fill='black', font=font)
    draw1.text((50, 110), "for OCR processing.", fill='black', font=font)
    img1.save(test_dir / "test1_english.png")
    
    # Test image 2: Rotated text (for deskewing test)
    img2 = Image.new('RGB', (500, 300), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((100, 100), "Rotated Text Sample", fill='black', font=font)
    draw2.text((100, 130), "Testing deskew functionality", fill='black', font=font)
    rotated_img2 = img2.rotate(5, expand=True, fillcolor='white')
    rotated_img2.save(test_dir / "test2_rotated.png")
    
    # Test image 3: Text with noise (for noise reduction test)
    img3 = Image.new('RGB', (400, 200), color='white')
    draw3 = ImageDraw.Draw(img3)
    draw3.text((50, 70), "Noisy Text Example", fill='black', font=font)
    draw3.text((50, 100), "Noise reduction test", fill='black', font=font)
    
    # Add noise
    img3_array = np.array(img3)
    noise = np.random.randint(0, 50, img3_array.shape, dtype=np.uint8)
    noisy_array = np.clip(img3_array.astype(int) + noise, 0, 255).astype(np.uint8)
    img3_noisy = Image.fromarray(noisy_array)
    img3_noisy.save(test_dir / "test3_noisy.png")
    
    logger.info(f"Created test images in {test_dir}")
    return [
        test_dir / "test1_english.png",
        test_dir / "test2_rotated.png", 
        test_dir / "test3_noisy.png"
    ]


def test_basic_ocr():
    """Test basic OCR functionality."""
    logger.info("\\n" + "="*50)
    logger.info("Testing Basic OCR Functionality")
    logger.info("="*50)
    
    # Create test images
    image_paths = create_test_images()
    
    # Test 1: Basic OCR with default settings
    logger.info("\\n1. Basic OCR with default settings:")
    engine = create_ocr_engine(languages=['eng'], preprocess=True)
    
    for image_path in image_paths:
        logger.info(f"\\nProcessing: {image_path.name}")
        try:
            result = engine.extract_text(image_path)
            logger.info(f"Extracted text: '{result.pages[0].text.strip()}'")
            logger.info(f"Confidence: {result.pages[0].confidence:.1f}%")
            logger.info(f"Processing time: {result.processing_time:.2f}s")
            logger.info(f"Number of blocks: {len(result.pages[0].blocks)}")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")


def test_preprocessing():
    """Test individual preprocessing steps."""
    logger.info("\\n" + "="*50)
    logger.info("Testing Image Preprocessing")
    logger.info("="*50)
    
    image_paths = create_test_images()
    preprocessor = ImagePreprocessor()
    
    for image_path in image_paths:
        logger.info(f"\\nTesting preprocessing on: {image_path.name}")
        
        # Test deskewing
        img = cv2.imread(str(image_path))
        deskewed, angle = preprocessor.deskew_image(img)
        logger.info(f"Detected skew angle: {angle:.2f} degrees")
        
        # Test noise reduction
        denoised = preprocessor.remove_noise(img)
        logger.info(f"Applied noise reduction")
        
        # Test contrast enhancement
        enhanced = preprocessor.enhance_contrast(img)
        logger.info(f"Enhanced contrast")
        
        # Test full preprocessing pipeline
        processed, info = preprocessor.preprocess_image(image_path)
        logger.info(f"Full preprocessing info: {info}")


def test_structured_output():
    """Test structured JSON output."""
    logger.info("\\n" + "="*50)
    logger.info("Testing Structured JSON Output")
    logger.info("="*50)
    
    image_paths = create_test_images()
    engine = create_ocr_engine(languages=['eng'])
    
    # Test structured output for first image
    image_path = image_paths[0]
    logger.info(f"\\nTesting structured output for: {image_path.name}")
    
    result = engine.extract_text(image_path)
    
    # Display hierarchical structure
    logger.info(f"\\nDocument Structure:")
    logger.info(f"  Total pages: {result.total_pages}")
    logger.info(f"  Average confidence: {result.average_confidence:.1f}%")
    logger.info(f"  Languages: {result.languages_detected}")
    
    page = result.pages[0]
    logger.info(f"\\nPage {page.page_number}:")
    logger.info(f"  Text: '{page.text.strip()}'")
    logger.info(f"  Confidence: {page.confidence:.1f}%")
    logger.info(f"  Processing time: {page.processing_time:.2f}s")
    logger.info(f"  Blocks: {len(page.blocks)}")
    
    for i, block in enumerate(page.blocks):
        logger.info(f"\\n  Block {i+1}:")
        logger.info(f"    Text: '{block.text.strip()}'")
        logger.info(f"    Confidence: {block.confidence:.1f}%")
        logger.info(f"    BBox: ({block.bounding_box.left}, {block.bounding_box.top}) - "
                   f"({block.bounding_box.right}, {block.bounding_box.bottom})")
        logger.info(f"    Lines: {len(block.lines)}")
        
        for j, line in enumerate(block.lines):
            logger.info(f"      Line {j+1}: '{line.text.strip()}' (conf: {line.confidence:.1f}%)")
            logger.info(f"        Words: {len(line.words)}")
            for k, word in enumerate(line.words):
                logger.info(f"          Word {k+1}: '{word.text}' (conf: {word.confidence:.1f}%)")


def test_json_output():
    """Test JSON serialization."""
    logger.info("\\n" + "="*50)
    logger.info("Testing JSON Output")
    logger.info("="*50)
    
    image_paths = create_test_images()
    
    # Test JSON output
    logger.info("\\nTesting JSON output:")
    json_result = extract_text_from_image(image_paths[0], return_json=True)
    
    # Parse and pretty print JSON
    parsed_json = json.loads(json_result)
    logger.info(f"JSON structure keys: {list(parsed_json.keys())}")
    
    # Save JSON to file
    json_file = Path("test_ocr_output.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(json_result)
    logger.info(f"JSON output saved to: {json_file}")


def test_batch_processing():
    """Test batch processing functionality."""
    logger.info("\\n" + "="*50)
    logger.info("Testing Batch Processing")
    logger.info("="*50)
    
    image_paths = create_test_images()
    
    # Test batch processing
    logger.info(f"\\nProcessing {len(image_paths)} images in batch:")
    results = extract_text_from_images(image_paths, languages=['eng'])
    
    for i, result in enumerate(results):
        page = result.pages[0]
        logger.info(f"\\nImage {i+1} ({image_paths[i].name}):")
        logger.info(f"  Text: '{page.text.strip()}'")
        logger.info(f"  Confidence: {page.confidence:.1f}%")
        logger.info(f"  Processing time: {page.processing_time:.2f}s")
    
    # Test batch JSON output
    json_results = extract_text_from_images(image_paths, return_json=True)
    batch_json_file = Path("test_batch_ocr_output.json")
    with open(batch_json_file, 'w', encoding='utf-8') as f:
        f.write(json_results)
    logger.info(f"\\nBatch JSON output saved to: {batch_json_file}")


def test_language_support():
    """Test multi-language support."""
    logger.info("\\n" + "="*50)
    logger.info("Testing Language Support")
    logger.info("="*50)
    
    # Display supported languages
    logger.info("\\nSupported languages:")
    for code, name in OCREngine.SUPPORTED_LANGUAGES.items():
        logger.info(f"  {code}: {name}")
    
    # Test with multiple languages
    logger.info("\\nTesting multi-language engine (eng+spa):")
    try:
        engine = create_ocr_engine(languages=['eng', 'spa'])
        logger.info("Multi-language engine created successfully")
    except Exception as e:
        logger.error(f"Error creating multi-language engine: {e}")


def test_preprocessing_options():
    """Test different preprocessing options."""
    logger.info("\\n" + "="*50)
    logger.info("Testing Preprocessing Options")
    logger.info("="*50)
    
    image_paths = create_test_images()
    image_path = image_paths[1]  # Use rotated image
    
    # Test with different preprocessing options
    options = [
        {"preprocess": False, "name": "No preprocessing"},
        {"preprocess": True, "name": "Full preprocessing"},
    ]
    
    for option in options:
        logger.info(f"\\nTesting: {option['name']}")
        try:
            result = extract_text_from_image(
                image_path, 
                preprocess=option["preprocess"]
            )
            page = result.pages[0]
            logger.info(f"  Text: '{page.text.strip()}'")
            logger.info(f"  Confidence: {page.confidence:.1f}%")
            logger.info(f"  Processing time: {page.processing_time:.2f}s")
        except Exception as e:
            logger.error(f"  Error: {e}")


def main():
    """Run all OCR tests."""
    logger.info("Starting OCR Module Tests")
    logger.info("========================")
    
    try:
        # Run all tests
        test_basic_ocr()
        test_preprocessing()
        test_structured_output()
        test_json_output()
        test_batch_processing()
        test_language_support()
        test_preprocessing_options()
        
        logger.info("\\n" + "="*50)
        logger.info("All OCR tests completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
