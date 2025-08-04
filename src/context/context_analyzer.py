"""
Omni-Dev Agent Context Awareness Module
Extracts and understands project architecture and coding conventions.
"""

import ast
from pathlib import Path
from typing import List, Dict, Any

class ContextAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.project_structure = {}
        self.coding_conventions = {}

    def analyze_structure(self) -> Dict[str, List[str]]:
        """Analyze the project directory structure."""
        for path in self.project_root.rglob("*.py"):
            module_name = str(path.relative_to(self.project_root)).replace('/', '.').replace('\\', '.').replace('.py', '')
            self.project_structure[module_name] = self._get_dependencies(path)
        return self.project_structure

    def _get_dependencies(self, file_path: Path) -> List[str]:
        """Extracts import dependencies from a Python file."""
        with open(file_path, "r") as file:
            tree = ast.parse(file.read(), filename=str(file_path))
            return [node.module for node in ast.walk(tree) if isinstance(node, ast.Import)]

    def analyze_conventions(self) -> Dict[str, Any]:
        """Analyze coding conventions from the codebase."""
        self.coding_conventions['indentation'] = self._check_indentation()
        # Add more convention checks as needed.
        return self.coding_conventions

    def _check_indentation(self) -> str:
        """Check the indentation style used in the project."""
        for path in self.project_root.rglob("*.py"):
            with open(path, "r") as file:
                for line in file:
                    if line.startswith(' '):
                        return 'spaces'
                    elif line.startswith('\t'):
                        return 'tabs'
        return 'unknown'
    
    def report(self):
        """Prints a report of project structure and conventions."""
        print("Project Structure:")
        for module, deps in self.project_structure.items():
            print(f"- {module}: {', '.join(deps)}")

        print("\nCoding Conventions:")
        for convention, value in self.coding_conventions.items():
            print(f"- {convention}: {value}")
