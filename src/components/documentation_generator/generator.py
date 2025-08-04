import subprocess
import os
import shutil


class DocGenerator:
    """
    Generates documentation using Sphinx.
    """

    def generate_html(self, source_dir: str, output_dir: str):
        """
        Generates HTML documentation from a source directory.

        Args:
            source_dir: The directory containing the Sphinx source files.
            output_dir: The directory to output the generated HTML.

        Returns:
            A dictionary containing the status and any errors.
        """
        if not os.path.isdir(source_dir):
            return {
                "status": "error",
                "errors": f"Source directory not found: {source_dir}",
            }

        # Basic conf.py for testing if it doesn't exist
        conf_py_path = os.path.join(source_dir, "conf.py")
        if not os.path.exists(conf_py_path):
            with open(conf_py_path, "w", encoding="utf-8") as f:
                f.write("project = 'Test Project'\n")
                f.write("extensions = []\n")
                f.write("templates_path = ['_templates']\n")
                f.write("exclude_patterns = []\n")

        # Basic index.rst for testing if it doesn't exist
        index_rst_path = os.path.join(source_dir, "index.rst")
        if not os.path.exists(index_rst_path):
            with open(index_rst_path, "w", encoding="utf-8") as f:
                f.write(
                    "Test Documentation\n==================\n\n.. toctree:\n   :maxdepth: 2\n   :caption: Contents:\n\n"
                )

        try:
            result = subprocess.run(
                ["sphinx-build", "-b", "html", source_dir, output_dir],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            if result.returncode == 0:
                return {"status": "success", "output": result.stdout}
            else:
                return {"status": "error", "errors": result.stderr}

        except FileNotFoundError:
            return {
                "status": "error",
                "errors": 'Sphinx is not installed. Please install it with "pip install Sphinx"',
            }
        except Exception as e:
            return {"status": "error", "errors": str(e)}
