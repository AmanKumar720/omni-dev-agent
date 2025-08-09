#!/usr/bin/env python3
"""
OCR Module Usage Examples

This script demonstrates practical usage scenarios for the OCR module:
- Document scanning
- Receipt processing
- Multi-language document processing
- Batch processing workflows
- Quality assessment
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import time

from ocr import (
    create_ocr_engine,
    extract_text_from_image,
    extract_text_from_images,
    OCREngine
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_document(image_path: str, languages: List[str] = None) -> Dict:
    """
    Scan a document image and extract structured text data.
    
    Args:
        image_path: Path to document image
        languages: List of languages to use for OCR
        
    Returns:
        Dictionary with extracted text and metadata
    """
    logger.info(f"Scanning document: {image_path}")
    
    # Default to English if no languages specified
    if languages is None:
        languages = ['eng']
    
    # Extract text with preprocessing
    result = extract_text_from_image(
        image_path, 
        languages=languages, 
        preprocess=True,
        return_json=False
    )
    
    # Process results into structured format
    page = result.pages[0]
    
    document_data = {
        'file_path': str(image_path),
        'extracted_text': page.text,
        'confidence': page.confidence,
        'processing_time': result.processing_time,
        'languages_used': result.languages_detected,
        'word_count': len(page.text.split()),
        'character_count': len(page.text),
        'blocks': []
    }
    
    # Extract block-level information
    for i, block in enumerate(page.blocks):
        block_data = {
            'block_id': i + 1,
            'text': block.text,
            'confidence': block.confidence,
            'bounding_box': {
                'left': block.bounding_box.left,
                'top': block.bounding_box.top,
                'width': block.bounding_box.width,
                'height': block.bounding_box.height
            },
            'line_count': len(block.lines)
        }
        document_data['blocks'].append(block_data)
    
    logger.info(f"Document scanned successfully - Confidence: {page.confidence:.1f}%")
    return document_data


def process_receipt(image_path: str) -> Dict:
    """
    Process a receipt image and extract key information.
    
    Args:
        image_path: Path to receipt image
        
    Returns:
        Dictionary with receipt data
    """
    logger.info(f"Processing receipt: {image_path}")
    
    # Use OCR optimized for receipts
    engine = create_ocr_engine(
        languages=['eng'],
        config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-$',
        preprocess=True
    )
    
    result = engine.extract_text(image_path)
    page = result.pages[0]
    
    # Extract receipt information using simple patterns
    text_lines = page.text.split('\\n')
    
    receipt_data = {
        'file_path': str(image_path),
        'raw_text': page.text,
        'confidence': page.confidence,
        'processing_time': result.processing_time,
        'lines': text_lines,
        'extracted_info': {}
    }
    
    # Simple extraction patterns (can be enhanced with regex)
    for line in text_lines:
        line = line.strip()
        if '$' in line:
            # Potential price line
            if 'total' in line.lower():
                receipt_data['extracted_info']['total'] = line
            elif 'tax' in line.lower():
                receipt_data['extracted_info']['tax'] = line
        elif any(word in line.lower() for word in ['store', 'market', 'shop']):
            # Potential store name
            receipt_data['extracted_info']['store_name'] = line
        elif len(line) > 10 and line.replace(' ', '').replace('-', '').isdigit():
            # Potential phone or receipt number
            receipt_data['extracted_info']['phone_or_receipt_id'] = line
    
    logger.info(f"Receipt processed - Found {len(receipt_data['extracted_info'])} key items")
    return receipt_data


def batch_document_processing(image_directory: str, output_file: str = None) -> List[Dict]:
    """
    Process multiple documents in a directory.
    
    Args:
        image_directory: Directory containing images to process
        output_file: Optional output JSON file path
        
    Returns:
        List of document processing results
    """
    logger.info(f"Starting batch processing of directory: {image_directory}")
    
    # Find image files
    image_dir = Path(image_directory)
    if not image_dir.exists():
        raise ValueError(f"Directory does not exist: {image_directory}")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    image_files = [
        f for f in image_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {image_directory}")
        return []
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Process documents
    results = []
    start_time = time.time()
    
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            doc_result = scan_document(str(image_file))
            doc_result['batch_index'] = i + 1
            results.append(doc_result)
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            # Add error entry
            error_result = {
                'file_path': str(image_file),
                'batch_index': i + 1,
                'error': str(e),
                'success': False
            }
            results.append(error_result)
    
    total_time = time.time() - start_time
    
    # Create summary
    successful = len([r for r in results if 'error' not in r])
    failed = len(results) - successful
    
    batch_summary = {
        'total_files': len(image_files),
        'successful': successful,
        'failed': failed,
        'total_processing_time': total_time,
        'average_time_per_file': total_time / len(image_files) if image_files else 0,
        'results': results
    }
    
    logger.info(f"Batch processing completed: {successful}/{len(image_files)} successful")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    
    return results


def multilingual_document_processing(image_path: str, languages: List[str]) -> Dict:
    """
    Process a document with multiple language support.
    
    Args:
        image_path: Path to document image
        languages: List of language codes to try
        
    Returns:
        Dictionary comparing results across languages
    """
    logger.info(f"Processing multilingual document: {image_path}")
    logger.info(f"Languages to try: {languages}")
    
    results = {}
    
    for lang in languages:
        logger.info(f"Processing with language: {lang}")
        
        try:
            result = extract_text_from_image(
                image_path,
                languages=[lang],
                preprocess=True
            )
            
            page = result.pages[0]
            results[lang] = {
                'language': lang,
                'language_name': OCREngine.SUPPORTED_LANGUAGES.get(lang, 'Unknown'),
                'text': page.text,
                'confidence': page.confidence,
                'processing_time': result.processing_time,
                'word_count': len(page.text.split()),
                'character_count': len(page.text)
            }
            
            logger.info(f"  {lang}: Confidence {page.confidence:.1f}%, "
                       f"{len(page.text.split())} words")
            
        except Exception as e:
            logger.error(f"Error processing with language {lang}: {e}")
            results[lang] = {
                'language': lang,
                'error': str(e),
                'success': False
            }
    
    # Find best result
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    if successful_results:
        best_lang = max(successful_results.keys(), 
                       key=lambda k: successful_results[k]['confidence'])
        results['best_result'] = {
            'language': best_lang,
            'confidence': successful_results[best_lang]['confidence'],
            'text': successful_results[best_lang]['text']
        }
        
        logger.info(f"Best result: {best_lang} with {successful_results[best_lang]['confidence']:.1f}% confidence")
    else:
        logger.warning("No successful language processing results")
    
    return results


def quality_assessment(image_path: str) -> Dict:
    """
    Assess OCR quality and provide recommendations.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with quality metrics and recommendations
    """
    logger.info(f"Assessing OCR quality for: {image_path}")
    
    # Test with and without preprocessing
    result_no_prep = extract_text_from_image(image_path, preprocess=False)
    result_with_prep = extract_text_from_image(image_path, preprocess=True)
    
    page_no_prep = result_no_prep.pages[0]
    page_with_prep = result_with_prep.pages[0]
    
    # Calculate quality metrics
    assessment = {
        'image_path': str(image_path),
        'without_preprocessing': {
            'confidence': page_no_prep.confidence,
            'text_length': len(page_no_prep.text),
            'word_count': len(page_no_prep.text.split()),
            'processing_time': result_no_prep.processing_time
        },
        'with_preprocessing': {
            'confidence': page_with_prep.confidence,
            'text_length': len(page_with_prep.text),
            'word_count': len(page_with_prep.text.split()),
            'processing_time': result_with_prep.processing_time
        },
        'improvement': {
            'confidence_gain': page_with_prep.confidence - page_no_prep.confidence,
            'text_length_change': len(page_with_prep.text) - len(page_no_prep.text),
            'word_count_change': len(page_with_prep.text.split()) - len(page_no_prep.text.split())
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if page_with_prep.confidence < 50:
        assessment['recommendations'].append("Low confidence detected. Consider higher resolution image or different preprocessing.")
    
    if page_with_prep.confidence > page_no_prep.confidence + 5:
        assessment['recommendations'].append("Preprocessing significantly improves results. Recommended for production use.")
    
    if page_with_prep.confidence > 90:
        assessment['recommendations'].append("Excellent OCR quality. Results should be highly reliable.")
    
    if len(page_with_prep.text) < 10:
        assessment['recommendations'].append("Very little text detected. Check if image contains readable text.")
    
    logger.info(f"Quality assessment complete - Best confidence: {max(page_no_prep.confidence, page_with_prep.confidence):.1f}%")
    return assessment


def main():
    """Example usage of OCR functions."""
    logger.info("OCR Module Usage Examples")
    logger.info("=" * 40)
    
    # Note: These examples assume you have test images
    # In practice, you would provide paths to your actual images
    
    try:
        # Example 1: Single document scanning
        logger.info("\\n1. Document Scanning Example:")
        logger.info("   This would scan a document image and extract structured text")
        # doc_result = scan_document("path/to/document.jpg", languages=['eng'])
        # print(f"   Extracted {doc_result['word_count']} words with {doc_result['confidence']:.1f}% confidence")
        
        # Example 2: Receipt processing
        logger.info("\\n2. Receipt Processing Example:")
        logger.info("   This would process a receipt and extract key information")
        # receipt_result = process_receipt("path/to/receipt.jpg")
        # print(f"   Found {len(receipt_result['extracted_info'])} key items")
        
        # Example 3: Batch processing
        logger.info("\\n3. Batch Processing Example:")
        logger.info("   This would process all images in a directory")
        # results = batch_document_processing("images/", "batch_results.json")
        # print(f"   Processed {len(results)} files")
        
        # Example 4: Multilingual processing
        logger.info("\\n4. Multilingual Processing Example:")
        logger.info("   This would try multiple languages and find the best result")
        # multi_result = multilingual_document_processing("path/to/document.jpg", ['eng', 'spa', 'fra'])
        # if 'best_result' in multi_result:
        #     print(f"   Best language: {multi_result['best_result']['language']}")
        
        # Example 5: Quality assessment
        logger.info("\\n5. Quality Assessment Example:")
        logger.info("   This would assess OCR quality and provide recommendations")
        # quality_result = quality_assessment("path/to/image.jpg")
        # print(f"   Recommendations: {len(quality_result['recommendations'])}")
        
        logger.info("\\nTo use these examples:")
        logger.info("1. Install required dependencies: pip install -r requirements_ocr.txt")
        logger.info("2. Install Tesseract OCR on your system")
        logger.info("3. Uncomment the example code and provide actual image paths")
        logger.info("4. Run this script")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")


if __name__ == "__main__":
    main()
