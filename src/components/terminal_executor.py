
import subprocess
import threading
import queue

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class TerminalExecutor:
    """
    Interacts with the operating system's command-line interface to execute shell commands.
    """

    def execute(self, command: str, timeout: int = 60):
        """
        Executes a shell command and captures its output.

        Args:
            command: The command to execute.
            timeout: The timeout for the command in seconds.

        Returns:
            A dictionary containing the stdout, stderr, exit code, and a status message.
        """
        logger.info(f"Executing command: {command}")
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )

            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode

            status = "completed"
            if exit_code != 0:
                status = "error"

            logger.info(f"Command finished with exit code {exit_code}")
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "status": status
            }

        except subprocess.TimeoutExpired as e:
            process.kill()
            stdout, stderr = process.communicate()
            logger.warning(f"Command timed out after {timeout} seconds: {command}")
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": -1,
                "status": f"timeout expired after {timeout} seconds"
            }
        except Exception as e:
            logger.error(f"An error occurred while executing command: {command}", exc_info=True)
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "status": "error"
            }

