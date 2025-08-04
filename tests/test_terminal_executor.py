import unittest
from src.components.terminal_executor import TerminalExecutor


class TestTerminalExecutor(unittest.TestCase):

    def setUp(self):
        self.executor = TerminalExecutor()

    def test_execute_success(self):
        """Test successful command execution."""
        result = self.executor.execute('echo "hello world"')
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello world", result["stdout"])
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["status"], "completed")

    def test_execute_error(self):
        """Test command execution with an error."""
        result = self.executor.execute("some_non_existent_command")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("not recognized", result["stderr"])
        self.assertEqual(result["status"], "error")

    def test_execute_timeout(self):
        """Test command execution timeout."""
        result = self.executor.execute(
            "ping -n 5 127.0.0.1", timeout=2
        )  # Windows specific ping
        self.assertEqual(result["exit_code"], -1)
        self.assertEqual(result["status"], "timeout expired after 2 seconds")


if __name__ == "__main__":
    unittest.main()
