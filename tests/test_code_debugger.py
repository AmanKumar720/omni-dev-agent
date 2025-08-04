import unittest
import os
from src.components.code_development.debugger import CodeDebugger


class TestCodeDebugger(unittest.TestCase):

    def setUp(self):
        self.debugger = CodeDebugger()
        self.test_file_path = "test_code_to_lint.py"
        # Use single quotes for the outer string to avoid conflict with the docstring
        file_content = '''
import os

def my_function():
    """This is a docstring."""
    unused_variable = 1
    print("Hello, world!")
'''
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_lint(self):
        """Test the lint method."""
        result = self.debugger.lint(self.test_file_path)
        self.assertIn("Your code has been rated", result["report"])
        self.assertIn("unused-import", result["report"])

    def test_run_tests(self):
        """Test the run_tests method."""
        test_file_content = """
import unittest

class MyTest(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
"""
        test_file_name = "test_to_run.py"
        with open(test_file_name, "w", encoding="utf-8") as f:
            f.write(test_file_content)

        result = self.debugger.run_tests(test_file_name)
        self.assertIn("Ran 1 test", result["errors"])
        self.assertIn("OK", result["errors"])

        os.remove(test_file_name)


if __name__ == "__main__":
    unittest.main()
