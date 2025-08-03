
import unittest
import os
import shutil
from src.components.documentation_generator.generator import DocGenerator

class TestDocGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = DocGenerator()
        self.source_dir = "test_docs_source"
        self.output_dir = "test_docs_output"
        os.makedirs(self.source_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.source_dir):
            shutil.rmtree(self.source_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_generate_html_success(self):
        """Test successful HTML generation."""
        result = self.generator.generate_html(self.source_dir, self.output_dir)
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "index.html")))

    def test_generate_html_no_source(self):
        """Test HTML generation with a non-existent source directory."""
        shutil.rmtree(self.source_dir)
        result = self.generator.generate_html(self.source_dir, self.output_dir)
        self.assertEqual(result["status"], "error")
        self.assertIn("Source directory not found", result["errors"])

if __name__ == '__main__':
    unittest.main()

