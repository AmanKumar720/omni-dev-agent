# Testing Guide for Omni-Dev Agent

This document provides comprehensive instructions for running tests in the Omni-Dev Agent project.

## Quick Start

### Option 1: Using the Test Runner Script (Recommended)
```bash
# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py tests/test_documentation_ingestor.py

# Run with additional pytest options
python run_tests.py --verbose --tb=short
```

### Option 2: Using Windows Batch File
```cmd
# Double-click run_tests.bat or run from command prompt
run_tests.bat

# Run specific test
run_tests.bat tests/test_documentation_ingestor.py
```

## Manual Test Execution

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install pytest transformers sentence-transformers spacy scikit-learn nltk networkx astor black beautifulsoup4 lxml commonmark PyPDF2 pdfplumber pytesseract markdown docutils
```

### Running Tests Manually

#### Method 1: Using PYTHONPATH Environment Variable
```bash
# Windows PowerShell
$env:PYTHONPATH="."; python -m pytest tests/ -v

# Windows Command Prompt
set PYTHONPATH=. && python -m pytest tests/ -v

# Linux/macOS
PYTHONPATH=. python -m pytest tests/ -v
```

#### Method 2: Using Python Path Configuration
```bash
python -m pytest tests/ -v
```

## Project Structure for Testing

The project follows this structure to enable proper imports:
```
omni-dev-agent/
├── src/                          # Source code
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   └── documentation_analyzer/
│   │       ├── __init__.py
│   │       └── documentation_ingestor.py
│   └── ...
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_documentation_ingestor.py
│   └── ...
├── pyproject.toml               # Project configuration
├── pytest.ini                  # Pytest configuration
├── run_tests.py                # Test runner script
└── run_tests.bat               # Windows batch file
```

## Configuration Files

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_paths = .
addopts = -v --tb=short
pythonpath = .
```

### pyproject.toml
Contains project metadata and dependency information. Key sections for testing:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_paths = ["."]
addopts = "-v --tb=short"
```

## Common Issues and Solutions

### Issue: ModuleNotFoundError: No module named 'src'
**Solution**: Use one of the provided test runners or set PYTHONPATH:
```bash
$env:PYTHONPATH="."; python -m pytest tests/ -v
```

### Issue: Missing dependencies
**Solution**: Install required packages:
```bash
pip install -r requirements.txt
# or install specific missing packages
pip install pytest markdown docutils
```

### Issue: Python version conflicts
**Solution**: Use specific Python version:
```bash
# Use Python 3.12 specifically
py -3.12 -m pytest tests/ -v

# Or use the version that has all dependencies installed
python --version  # Check current version
```

## Writing New Tests

### Test File Structure
```python
import pytest
from unittest.mock import patch, MagicMock
from src.components.your_module import YourClass

class TestYourClass:
    def setup_method(self):
        self.instance = YourClass()
    
    def test_your_method(self):
        result = self.instance.your_method("input")
        assert result == "expected_output"
    
    @patch('src.components.your_module.external_dependency')
    def test_with_mock(self, mock_dependency):
        mock_dependency.return_value = "mocked_result"
        result = self.instance.method_using_dependency()
        assert result == "expected_result"
        mock_dependency.assert_called_once()
```

### Best Practices
1. Use descriptive test method names
2. Test both success and failure cases
3. Mock external dependencies
4. Use setup_method for test initialization
5. Include assertions for expected behavior
6. Group related tests in classes

## Test Coverage

To run tests with coverage reporting:
```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

## Continuous Integration

The project is configured to work with CI/CD pipelines. The test runner scripts ensure consistent execution across different environments.

## Troubleshooting

### Debug Mode
Run tests in debug mode for more information:
```bash
python -m pytest tests/ -v -s --tb=long
```

### Specific Test Selection
```bash
# Run specific test method
python -m pytest tests/test_documentation_ingestor.py::TestDocumentationIngestor::test_format_code -v

# Run tests matching pattern
python -m pytest tests/ -k "test_format" -v
```

### Environment Verification
Check if the environment is set up correctly:
```python
import sys
print(sys.path)  # Should include project root
```

## Support

If you encounter issues not covered in this guide:
1. Check the project's GitHub issues
2. Verify all dependencies are installed
3. Ensure you're using a supported Python version (3.8+)
4. Try using the provided test runner scripts
