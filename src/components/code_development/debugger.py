import subprocess


class CodeDebugger:
    """
    A component for developing and debugging code.
    """

    def lint(self, file_path: str):
        """
        Analyzes a Python file with pylint.

        Args:
            file_path: The path to the Python file to lint.

        Returns:
            A dictionary containing the pylint report.
        """
        try:
            result = subprocess.run(
                ["pylint", file_path], capture_output=True, text=True, encoding="utf-8"
            )
            return {"report": result.stdout, "errors": result.stderr}
        except FileNotFoundError:
            return {
                "report": "",
                "errors": 'pylint is not installed. Please install it with "pip install pylint"',
            }

    def run_tests(self, test_path: str):
        """
        Runs a Python test file using the unittest framework.

        Args:
            test_path: The path to the test file.

        Returns:
            A dictionary containing the test results.
        """
        try:
            result = subprocess.run(
                ["python", "-m", "unittest", test_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return {"output": result.stdout, "errors": result.stderr}
        except Exception as e:
            return {"output": "", "errors": str(e)}
