import pytest
from unittest.mock import patch, MagicMock
from src.components.documentation_analyzer.documentation_ingestor import (
    DocumentationIngestor,
)


class TestDocumentationIngestor:
    def setup_method(self):
        self.ingestor = DocumentationIngestor()

    def test_format_code(self):
        code = "def foo():\n    return 1"
        formatted = self.ingestor.format_code(code)
        assert "def foo()" in formatted

    @patch("subprocess.run")
    def test_format_code_subprocess_called(self, mock_subprocess):
        code = "def foo():\n    return 1"
        # Mock the subprocess call to black
        mock_subprocess.return_value = MagicMock()
        result = self.ingestor.format_code(code)
        mock_subprocess.assert_called()
        assert isinstance(result, str)

    def test_add_error_handling(self):
        code = "def foo():\n    return 1"
        result = self.ingestor.add_error_handling(code)
        assert "try:" in result or "except" in result

    def test_check_security(self):
        safe_code = "def foo():\n    return 1"
        unsafe_code = "eval('malicious code')"

        safe_result = self.ingestor.check_security(safe_code)
        unsafe_result = self.ingestor.check_security(unsafe_code)

        assert isinstance(safe_result, bool)
        assert safe_result is True
        assert unsafe_result is False

    def test_is_idempotent(self):
        code = "def foo():\n    return 1"
        snippet = "return 1"
        result = self.ingestor.is_idempotent(code, snippet)
        assert isinstance(result, bool)
        assert result is True

    def test_context_aware_insert(self):
        code = "def foo():\n    return 1"
        snippet = "print('hello')"
        result = self.ingestor.context_aware_insert(code, snippet)
        assert isinstance(result, str)
        assert snippet in result or code == result  # Either inserted or idempotent

    def test_insert_code_snippet(self):
        code = "def foo():\n    return 1"
        snippet = "print('hello')"

        # Test end insertion
        result_end = self.ingestor.insert_code_snippet(code, snippet, "end")
        assert snippet in result_end

        # Test start insertion
        result_start = self.ingestor.insert_code_snippet(code, snippet, "start")
        assert result_start.startswith(snippet)

    def test_regex_replace_code(self):
        code = "def foo():\n    return 1"
        pattern = r"return 1"
        replacement = "return 2"
        result = self.ingestor.regex_replace_code(code, pattern, replacement)
        assert "return 2" in result
        assert "return 1" not in result
