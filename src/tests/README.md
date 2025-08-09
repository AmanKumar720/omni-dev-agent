# AI Vision Testing Suite

This directory contains comprehensive tests for the AI Vision modules, including unit tests, integration tests, and test data generation.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests for individual modules
│   ├── test_core.py         # Core vision framework tests
│   ├── test_image_classification.py
│   ├── test_object_detection.py
│   ├── test_ocr.py
│   └── test_face_recognition.py
├── integration/             # Integration tests with real data
│   └── test_vision_integration.py
├── data/                    # Test data generation and samples
│   ├── generate_test_data.py
│   ├── images/              # Sample test images
│   └── videos/              # Sample test videos
└── fixtures/                # Test fixtures and utilities
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements_camera_vision.txt
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Test Categories

Tests are organized by markers:

- `unit` - Fast unit tests for individual components
- `integration` - Integration tests with sample data
- `slow` - Longer running tests (model downloads, etc.)
- `gpu` - Tests requiring GPU acceleration
- `model` - Tests requiring model downloads
- `vision`, `face`, `ocr`, `detection`, `classification`, `analytics` - Domain-specific markers

### Running Different Test Suites

**Quick unit tests (no model downloads):**
```bash
pytest tests/unit/ -m "unit and not slow"
```

**All unit tests:**
```bash
pytest tests/unit/ -v --cov=components
```

**Integration tests:**
```bash
pytest tests/integration/ -m integration
```

**Specific module tests:**
```bash
pytest tests/unit/test_image_classification.py -v
pytest tests/ -m classification
pytest tests/ -m "face or ocr"
```

**GPU tests (if GPU available):**
```bash
pytest tests/ -m gpu
```

**Model tests (downloads models):**
```bash
pytest tests/ -m model --timeout=1200
```

**All tests with coverage:**
```bash
pytest tests/ --cov=components --cov-report=html
```

## Test Data

### Generating Test Data

Generate sample images and videos for testing:
```bash
cd tests/data
python generate_test_data.py
```

This creates:
- Classification images (cat, dog, car, nature scenes)
- Detection scenes (multi-object, crowds, traffic)
- OCR documents (text, receipts, signage)  
- Face recognition images (frontal, profile, groups)
- Edge case images (blurry, low contrast, noisy)

### Using Custom Test Data

Place your own test images in `tests/data/images/` and videos in `tests/data/videos/`. The integration tests will automatically discover and use them.

## Continuous Integration

### GitHub Actions Workflow

The AI Vision test suite runs automatically on:
- Push to main/develop branches
- Pull requests
- Daily scheduled runs (2 AM UTC)

Features:
- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Python version matrix** (3.8, 3.9, 3.10, 3.11)
- **Model caching** to speed up CI runs
- **Headless OpenCV** for server environments
- **Coverage reporting** with Codecov integration
- **Performance benchmarking** (on schedule)
- **Security scanning** with Safety and Bandit

### Local CI Testing

Test the same configuration locally using Docker:
```bash
docker build -f .github/workflows/Dockerfile.test -t ai-vision-tests .
docker run --rm ai-vision-tests
```

## Mocking Strategy

### Dependencies

Heavy dependencies are mocked in unit tests to ensure fast, reliable testing:

- **PyTorch/TorchVision** - Mocked model loading and inference
- **Ultralytics YOLO** - Mocked detection results
- **OpenCV** - Mocked for non-vision operations
- **Face Recognition** - Mocked face detection/encoding
- **Tesseract** - Mocked OCR results

### Real Testing

Integration tests use:
- Real image processing (PIL, NumPy)
- Generated synthetic test data
- Headless OpenCV for computer vision
- Mock models with realistic outputs

## Test Configuration

### Pytest Configuration (pytest.ini)

Key settings:
- Async test support with `asyncio_mode = auto`
- Timeout protection (300s unit, 600s integration)
- Coverage reporting with XML/HTML output
- Warning filters for cleaner output
- Custom markers for test categorization

### Environment Variables

Test behavior controlled by environment variables:

- `OPENCV_HEADLESS=1` - Use headless OpenCV
- `MODELS_CACHE_DIR` - Cache directory for downloaded models
- `PYTEST_CURRENT_TEST` - Current test name (auto-set)
- `CI=true` - Enables CI-specific behavior

## Writing New Tests

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
@pytest.mark.classification
class TestImageClassifier:
    def test_classifier_initialization(self, mock_model_hub):
        """Test classifier initialization"""
        # Your test code here
        pass
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations"""
        # Your async test code here
        pass
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
class TestVisionPipeline:
    def test_with_real_image(self, sample_image_pil):
        """Test with generated sample image"""
        # Your integration test code here
        pass
```

### Test Fixtures

Common fixtures available in all tests:

- `sample_image_pil` - PIL Image for testing
- `sample_image_array` - NumPy array image
- `sample_text_image` - Image with text for OCR
- `sample_face_image` - Face image for recognition
- `temp_test_dir` - Temporary directory
- `mock_model_hub` - Mocked model hub
- Mock libraries: `mock_torch`, `mock_cv2`, `mock_ultralytics`, etc.

### Performance Testing

Benchmark critical operations:
```python
@pytest.mark.benchmark
def test_classification_performance(benchmark):
    """Benchmark image classification speed"""
    result = benchmark(classifier.classify_image, test_image)
    assert result is not None
```

## Debugging Tests

### Common Issues

1. **Import errors** - Check module paths and dependencies
2. **Async test failures** - Ensure proper `@pytest.mark.asyncio` usage
3. **Mock setup** - Verify mocks match actual API signatures
4. **Timeout issues** - Increase timeout for slow operations
5. **File path issues** - Use `Path` objects and `tmp_path` fixture

### Debugging Commands

```bash
# Verbose output with no capture
pytest tests/unit/test_core.py -v -s

# Drop into debugger on failure
pytest tests/unit/test_core.py --pdb

# Run last failed tests only
pytest --lf

# Show test coverage gaps
pytest --cov=components --cov-report=term-missing
```

### Test Isolation

Each test runs in isolation with:
- Fresh mock objects
- Temporary directories cleaned up
- No shared state between tests
- Predictable random seeds where needed

## Contributing

### Before Submitting

1. **Run the full test suite:**
   ```bash
   pytest tests/ --cov=components
   ```

2. **Check for new test data needs:**
   ```bash
   python tests/data/generate_test_data.py
   ```

3. **Verify CI compatibility:**
   ```bash
   # Test with headless OpenCV
   OPENCV_HEADLESS=1 pytest tests/unit/
   ```

4. **Add appropriate test markers:**
   ```python
   @pytest.mark.unit
   @pytest.mark.classification
   @pytest.mark.slow  # if test takes >5 seconds
   ```

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>_<condition>`
- Async tests: Include `async` in the method name

### Documentation

Include docstrings for:
- Test classes describing the component under test
- Complex test methods explaining the test scenario
- Fixtures describing what they provide

## Troubleshooting

### Common Solutions

**Tests hanging:**
- Check for missing `await` in async tests
- Verify mock setup doesn't create infinite loops
- Use `--timeout` option to catch hanging tests

**Import errors in CI:**
- Ensure all dependencies in requirements files
- Check for platform-specific imports
- Use conditional imports with try/except

**Model loading failures:**
- Models are mocked in unit tests
- Integration tests may need internet for downloads
- Use model caching to avoid repeated downloads

**OpenCV display errors:**
- Set `OPENCV_HEADLESS=1` environment variable
- Use `cv2.imread` instead of `cv2.imshow`
- Mock display-related functions

For more help, see the individual test files for examples or check the CI logs for common failure patterns.
