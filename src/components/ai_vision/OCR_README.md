# OCR Module

A comprehensive Optical Character Recognition module that wraps pytesseract with advanced pre-processing, post-processing, and structured JSON output capabilities.

## Features

### 🔍 Advanced OCR Processing
- **Pre-processing Pipeline**: De-skewing, noise reduction, contrast enhancement
- **Multi-language Support**: 13+ languages including English, Spanish, French, German, Chinese, Japanese, Arabic, and more
- **Structured Output**: Hierarchical text extraction with blocks, lines, and words
- **Confidence Scoring**: Per-word, per-line, per-block, and per-page confidence metrics
- **Bounding Box Detection**: Precise coordinate information for all text elements

### 📄 Output Formats
- **Structured Objects**: Rich dataclasses with comprehensive metadata
- **JSON Export**: Clean, structured JSON output with full text hierarchy
- **Batch Processing**: Process multiple documents with summary statistics

### 🛠 Image Preprocessing
- **Deskewing**: Automatic rotation correction using Hough line detection
- **Noise Reduction**: Morphological operations and bilateral filtering
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Flexible Pipeline**: Enable/disable individual preprocessing steps

## Installation

### Prerequisites

1. **Install Tesseract OCR**:
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`

2. **Install Language Packs** (optional):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr-[lang]
   
   # macOS
   brew install tesseract-lang
   ```

### Python Dependencies

```bash
pip install -r requirements_ocr.txt
```

**Requirements include**:
- pytesseract >= 0.3.10
- opencv-python >= 4.5.0
- Pillow >= 9.0.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-image >= 0.19.0

## Quick Start

### Basic Usage

```python
from ocr import extract_text_from_image

# Simple text extraction
result = extract_text_from_image("document.jpg")
print(f"Extracted text: {result.pages[0].text}")
print(f"Confidence: {result.pages[0].confidence:.1f}%")
```

### Advanced Usage

```python
from ocr import create_ocr_engine

# Create OCR engine with custom settings
engine = create_ocr_engine(
    languages=['eng', 'spa'],  # Multiple languages
    preprocess=True,           # Enable preprocessing
    config='--oem 3 --psm 6'   # Custom tesseract config
)

# Extract with structured output
result = engine.extract_text("document.jpg")

# Access hierarchical structure
for block in result.pages[0].blocks:
    print(f"Block: {block.text}")
    for line in block.lines:
        print(f"  Line: {line.text} (confidence: {line.confidence:.1f}%)")
        for word in line.words:
            print(f"    Word: {word.text} ({word.confidence:.1f}%)")
```

## API Reference

### Main Classes

#### `OCREngine`
The main OCR processing engine.

```python
engine = OCREngine(
    languages=['eng'],        # List of language codes
    config='--oem 3 --psm 6', # Tesseract configuration
    preprocess=True           # Enable preprocessing
)
```

**Methods**:
- `extract_text(image_path, preprocess=None, return_json=False)` - Extract text from single image
- `extract_text_batch(image_paths, preprocess=None, return_json=False)` - Process multiple images

#### `ImagePreprocessor`
Handles image preprocessing operations.

**Static Methods**:
- `deskew_image(image, angle_range=45)` - Correct image rotation
- `remove_noise(image, kernel_size=3)` - Apply noise reduction
- `enhance_contrast(image, clip_limit=2.0)` - Improve contrast
- `preprocess_image(image, deskew=True, denoise=True, enhance_contrast=True)` - Full pipeline

### Data Classes

#### `OCRResult`
Complete OCR result for a document.

```python
result = OCRResult(
    pages=[OCRPage],           # List of processed pages
    total_pages=1,             # Number of pages
    average_confidence=85.5,   # Overall confidence
    processing_time=1.23,      # Time taken in seconds
    languages_detected=['eng'] # Languages used
)
```

#### `OCRPage`
Represents a single page of OCR results.

```python
page = OCRPage(
    page_number=1,            # Page number
    text="Extracted text",    # Full page text
    confidence=85.5,          # Page confidence
    bounding_box=BoundingBox, # Page coordinates
    blocks=[OCRBlock],        # Text blocks
    language='eng',           # Language used
    processing_time=0.5       # Processing time
)
```

#### `OCRBlock`
Text block with lines and metadata.

#### `OCRLine`
Text line with words and metadata.

#### `OCRWord`
Individual word with confidence and coordinates.

#### `BoundingBox`
Coordinate information for text elements.

```python
bbox = BoundingBox(
    left=10,    # X coordinate
    top=20,     # Y coordinate  
    width=100,  # Width in pixels
    height=30,  # Height in pixels
    right=110,  # Calculated right edge
    bottom=50   # Calculated bottom edge
)
```

### Convenience Functions

#### `extract_text_from_image()`
```python
result = extract_text_from_image(
    image_path="document.jpg",
    languages=['eng', 'spa'],
    preprocess=True,
    return_json=False
)
```

#### `extract_text_from_images()`
```python
results = extract_text_from_images(
    image_paths=["doc1.jpg", "doc2.jpg"],
    languages=['eng'],
    preprocess=True,
    return_json=False
)
```

## Supported Languages

| Code | Language |
|------|----------|
| `eng` | English |
| `spa` | Spanish |
| `fra` | French |
| `deu` | German |
| `ita` | Italian |
| `por` | Portuguese |
| `rus` | Russian |
| `jpn` | Japanese |
| `chi_sim` | Chinese Simplified |
| `chi_tra` | Chinese Traditional |
| `ara` | Arabic |
| `hin` | Hindi |
| `kor` | Korean |

## Configuration Options

### Tesseract PSM Modes
Page Segmentation Modes (PSM) for different document types:

- `--psm 0`: Orientation and script detection (OSD) only
- `--psm 1`: Automatic page segmentation with OSD
- `--psm 3`: Fully automatic page segmentation (default)
- `--psm 6`: Uniform block of text (recommended for documents)
- `--psm 7`: Single text line
- `--psm 8`: Single word
- `--psm 13`: Raw line (useful for receipts)

### OCR Engine Modes
- `--oem 0`: Legacy engine only
- `--oem 1`: Neural nets LSTM engine only
- `--oem 2`: Legacy + LSTM engines
- `--oem 3`: Default (based on what's available)

### Custom Configuration Example
```python
# Receipt processing configuration
engine = create_ocr_engine(
    languages=['eng'],
    config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-$'
)

# Single line processing
engine = create_ocr_engine(
    config='--oem 1 --psm 7'
)
```

## Usage Examples

### Document Scanning
```python
def scan_document(image_path):
    engine = create_ocr_engine(languages=['eng'], preprocess=True)
    result = engine.extract_text(image_path)
    
    return {
        'text': result.pages[0].text,
        'confidence': result.pages[0].confidence,
        'word_count': len(result.pages[0].text.split()),
        'blocks': len(result.pages[0].blocks)
    }
```

### Batch Processing
```python
def process_directory(directory_path):
    from pathlib import Path
    
    image_files = list(Path(directory_path).glob('*.jpg'))
    results = extract_text_from_images(image_files, languages=['eng'])
    
    for i, result in enumerate(results):
        print(f"File {i+1}: {result.pages[0].confidence:.1f}% confidence")
    
    return results
```

### Multi-language Processing
```python
def detect_best_language(image_path, languages):
    results = {}
    
    for lang in languages:
        result = extract_text_from_image(image_path, languages=[lang])
        results[lang] = {
            'confidence': result.pages[0].confidence,
            'text': result.pages[0].text
        }
    
    # Find best result
    best_lang = max(results.keys(), key=lambda k: results[k]['confidence'])
    return best_lang, results[best_lang]
```

### JSON Output
```python
# Get structured JSON output
json_result = extract_text_from_image("document.jpg", return_json=True)

# Save to file
with open("ocr_result.json", "w") as f:
    f.write(json_result)

# Load and process
import json
data = json.loads(json_result)
print(f"Processed {data['total_pages']} pages")
```

## Performance Optimization

### Image Preprocessing
- **Enable preprocessing** for low-quality images
- **Disable preprocessing** for high-quality scanned documents
- **Custom preprocessing** for specific image types

### Tesseract Configuration  
- Use `--oem 1` for better accuracy with neural networks
- Use `--psm 6` for standard documents
- Use `--psm 7` or `--psm 8` for single lines/words
- Restrict character sets with `tessedit_char_whitelist`

### Batch Processing
- Process images in parallel for large batches
- Use appropriate image formats (PNG, TIFF preferred over JPEG)
- Resize very large images to reasonable dimensions

## Error Handling

The module includes comprehensive error handling:

```python
try:
    result = extract_text_from_image("document.jpg")
    if result.pages[0].confidence < 50:
        print("Warning: Low confidence OCR result")
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"OCR processing failed: {e}")
```

## Testing

Run the test suite to verify functionality:

```bash
# Run basic tests (creates test images)
python test_ocr_basic.py

# Run usage examples
python example_ocr_usage.py
```

## Troubleshooting

### Common Issues

1. **"pytesseract.TesseractNotFoundError"**
   - Ensure Tesseract is installed and in system PATH
   - On Windows, set `pytesseract.pytesseract.tesseract_cmd` path

2. **Poor OCR accuracy**
   - Enable preprocessing: `preprocess=True`
   - Try different PSM modes: `--psm 6`, `--psm 7`, etc.
   - Verify correct language is specified
   - Check image quality and resolution

3. **Language not found**
   - Install additional language packs
   - Verify language code is correct
   - Use `tesseract --list-langs` to check available languages

4. **Slow processing**
   - Disable unnecessary preprocessing steps
   - Resize large images
   - Use appropriate OEM mode

### Debug Information

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To contribute to this module:

1. Follow existing code style and patterns
2. Add comprehensive docstrings
3. Include test cases for new features
4. Update documentation
5. Handle edge cases and errors gracefully

## License

This module is part of the AI Vision component and follows the project's licensing terms.
