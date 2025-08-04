"""
Comprehensive Testing Framework for Omni-Dev Agent
Implements various testing strategies including unit, integration, e2e, and regression tests.
"""

import os
import sys
import json
import time
import logging
import subprocess
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pytest
import ast
from unittest.mock import Mock, MagicMock, patch

class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "e2e"
    REGRESSION = "regression"
    STATIC_ANALYSIS = "static"

class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1-10, 10 being highest priority

@dataclass
class TestSuite:
    name: str
    test_type: TestType
    tests: List[Callable]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    priority: int = 1

class TestFramework:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_results: List[TestResult] = []
        self.test_suites: Dict[TestType, List[TestSuite]] = {
            test_type: [] for test_type in TestType
        }
        self.logger = self._setup_logging()
        self.rollback_points: List[Dict[str, Any]] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the test framework."""
        logger = logging.getLogger("TestFramework")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "test_framework.log")
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def register_test_suite(self, suite: TestSuite):
        """Register a test suite with the framework."""
        self.test_suites[suite.test_type].append(suite)
        self.logger.info(f"Registered {suite.test_type.value} test suite: {suite.name}")

    def create_rollback_point(self, description: str) -> int:
        """Create a rollback point before running tests."""
        rollback_point = {
            "id": len(self.rollback_points),
            "timestamp": time.time(),
            "description": description,
            "git_commit": self._get_current_git_commit(),
            "file_checksums": self._calculate_file_checksums()
        }
        self.rollback_points.append(rollback_point)
        self.logger.info(f"Created rollback point {rollback_point['id']}: {description}")
        return rollback_point["id"]

    def _get_current_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _calculate_file_checksums(self) -> Dict[str, str]:
        """Calculate checksums for important files."""
        import hashlib
        checksums = {}
        
        # Check Python files in src directory
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'rb') as f:
                        checksums[str(py_file.relative_to(self.project_root))] = \
                            hashlib.md5(f.read()).hexdigest()
                except Exception as e:
                    self.logger.warning(f"Could not checksum {py_file}: {e}")
        
        return checksums

    def rollback_to_point(self, rollback_id: int) -> bool:
        """Rollback to a specific rollback point."""
        if rollback_id >= len(self.rollback_points):
            self.logger.error(f"Invalid rollback point ID: {rollback_id}")
            return False
        
        rollback_point = self.rollback_points[rollback_id]
        self.logger.info(f"Rolling back to point {rollback_id}: {rollback_point['description']}")
        
        # Attempt git rollback if commit is available
        if rollback_point.get("git_commit"):
            try:
                subprocess.run(
                    ["git", "reset", "--hard", rollback_point["git_commit"]], 
                    cwd=self.project_root, 
                    check=True
                )
                self.logger.info("Git rollback successful")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Git rollback failed: {e}")
        
        return False

    def run_unit_tests(self, component_path: Optional[str] = None) -> List[TestResult]:
        """Run unit tests for specific component or all components."""
        self.logger.info("Running unit tests...")
        results = []
        
        for suite in self.test_suites[TestType.UNIT]:
            if suite.setup_func:
                suite.setup_func()
            
            for test_func in suite.tests:
                result = self._run_single_test(test_func, TestType.UNIT, suite.name)
                results.append(result)
            
            if suite.teardown_func:
                suite.teardown_func()
        
        # Also run pytest-based unit tests
        pytest_results = self._run_pytest_tests("tests/unit/", TestType.UNIT)
        results.extend(pytest_results)
        
        self.test_results.extend(results)
        return results

    def run_integration_tests(self, components: List[str] = None) -> List[TestResult]:
        """Run integration tests between components."""
        self.logger.info("Running integration tests...")
        results = []
        
        for suite in self.test_suites[TestType.INTEGRATION]:
            if suite.setup_func:
                suite.setup_func()
            
            for test_func in suite.tests:
                result = self._run_single_test(test_func, TestType.INTEGRATION, suite.name)
                results.append(result)
            
            if suite.teardown_func:
                suite.teardown_func()
        
        # Run pytest-based integration tests
        pytest_results = self._run_pytest_tests("tests/integration/", TestType.INTEGRATION)
        results.extend(pytest_results)
        
        self.test_results.extend(results)
        return results

    def run_e2e_tests(self) -> List[TestResult]:
        """Run end-to-end tests."""
        self.logger.info("Running end-to-end tests...")
        results = []
        
        for suite in self.test_suites[TestType.END_TO_END]:
            if suite.setup_func:
                suite.setup_func()
            
            for test_func in suite.tests:
                result = self._run_single_test(test_func, TestType.END_TO_END, suite.name)
                results.append(result)
            
            if suite.teardown_func:
                suite.teardown_func()
        
        # Run pytest-based e2e tests
        pytest_results = self._run_pytest_tests("tests/e2e/", TestType.END_TO_END)
        results.extend(pytest_results)
        
        self.test_results.extend(results)
        return results

    def run_regression_tests(self) -> List[TestResult]:
        """Run regression tests to ensure no existing functionality is broken."""
        self.logger.info("Running regression tests...")
        results = []
        
        # Run all existing tests as regression tests
        regression_results = self._run_pytest_tests("tests/", TestType.REGRESSION)
        results.extend(regression_results)
        
        self.test_results.extend(results)
        return results

    def run_static_analysis(self) -> List[TestResult]:
        """Run static analysis tools."""
        self.logger.info("Running static analysis...")
        results = []
        
        # Run pylint
        pylint_result = self._run_pylint()
        if pylint_result:
            results.append(pylint_result)
        
        # Run bandit for security analysis
        bandit_result = self._run_bandit()
        if bandit_result:
            results.append(bandit_result)
        
        # Run mypy for type checking
        mypy_result = self._run_mypy()
        if mypy_result:
            results.append(mypy_result)
        
        self.test_results.extend(results)
        return results

    def _run_single_test(self, test_func: Callable, test_type: TestType, suite_name: str) -> TestResult:
        """Run a single test function and return the result."""
        start_time = time.time()
        test_name = f"{suite_name}.{test_func.__name__}"
        
        try:
            test_func()
            status = TestStatus.PASSED
            error_message = None
        except Exception as e:
            status = TestStatus.FAILED
            error_message = str(e)
            self.logger.error(f"Test {test_name} failed: {error_message}")
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name=test_name,
            test_type=test_type,
            status=status,
            duration=duration,
            error_message=error_message
        )

    def _run_pytest_tests(self, test_dir: str, test_type: TestType) -> List[TestResult]:
        """Run pytest tests in a specific directory."""
        test_path = self.project_root / test_dir
        if not test_path.exists():
            return []
        
        try:
            # Run pytest with JSON output
            cmd = [
                sys.executable, "-m", "pytest", 
                str(test_path), 
                "--json-report", 
                "--json-report-file", 
                str(self.project_root / f"test_report_{test_type.value}.json"),
                "-v"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # Parse JSON report
            report_file = self.project_root / f"test_report_{test_type.value}.json"
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)
                
                results = []
                for test in report_data.get("tests", []):
                    test_result = TestResult(
                        test_name=test["nodeid"],
                        test_type=test_type,
                        status=TestStatus(test["outcome"]) if test["outcome"] in [s.value for s in TestStatus] else TestStatus.ERROR,
                        duration=test.get("duration", 0),
                        error_message=test.get("call", {}).get("longrepr", None)
                    )
                    results.append(test_result)
                
                return results
            
        except Exception as e:
            self.logger.error(f"Failed to run pytest tests in {test_dir}: {e}")
        
        return []

    def _run_pylint(self) -> Optional[TestResult]:
        """Run pylint static analysis."""
        try:
            cmd = [sys.executable, "-m", "pylint", "src/", "--output-format=json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # Parse pylint output
            issues = []
            if result.stdout:
                try:
                    pylint_data = json.loads(result.stdout)
                    issues = [issue for issue in pylint_data if issue.get("type") in ["error", "warning"]]
                except json.JSONDecodeError:
                    pass
            
            status = TestStatus.PASSED if len(issues) == 0 else TestStatus.FAILED
            error_message = f"Found {len(issues)} pylint issues" if issues else None
            
            return TestResult(
                test_name="pylint_analysis",
                test_type=TestType.STATIC_ANALYSIS,
                status=status,
                duration=0,
                error_message=error_message,
                details={"issues": issues}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to run pylint: {e}")
            return None

    def _run_bandit(self) -> Optional[TestResult]:
        """Run bandit security analysis."""
        try:
            cmd = [sys.executable, "-m", "bandit", "-r", "src/", "-f", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # Parse bandit output
            issues = []
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get("results", [])
                except json.JSONDecodeError:
                    pass
            
            status = TestStatus.PASSED if len(issues) == 0 else TestStatus.FAILED
            error_message = f"Found {len(issues)} security issues" if issues else None
            
            return TestResult(
                test_name="bandit_security_analysis",
                test_type=TestType.STATIC_ANALYSIS,
                status=status,
                duration=0,
                error_message=error_message,
                details={"issues": issues}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to run bandit: {e}")
            return None

    def _run_mypy(self) -> Optional[TestResult]:
        """Run mypy type checking."""
        try:
            cmd = [sys.executable, "-m", "mypy", "src/", "--json-report", "mypy_report"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # Parse mypy output
            issues = []
            mypy_report_dir = self.project_root / "mypy_report"
            if mypy_report_dir.exists():
                try:
                    with open(mypy_report_dir / "index.txt") as f:
                        mypy_output = f.read()
                        # Simple parsing - count error lines
                        issues = [line for line in mypy_output.split('\n') if 'error:' in line]
                except Exception:
                    pass
            
            status = TestStatus.PASSED if len(issues) == 0 else TestStatus.FAILED
            error_message = f"Found {len(issues)} type errors" if issues else None
            
            return TestResult(
                test_name="mypy_type_checking",
                test_type=TestType.STATIC_ANALYSIS,
                status=status,
                duration=0,
                error_message=error_message,
                details={"issues": issues}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to run mypy: {e}")
            return None

    def auto_generate_tests(self, component_path: str) -> List[str]:
        """Automatically generate tests based on component API."""
        self.logger.info(f"Auto-generating tests for {component_path}")
        
        try:
            # Parse the component file
            with open(component_path) as f:
                tree = ast.parse(f.read())
            
            generated_tests = []
            
            # Find classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    test_code = self._generate_class_tests(node)
                    generated_tests.append(test_code)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    test_code = self._generate_function_tests(node)
                    generated_tests.append(test_code)
            
            return generated_tests
            
        except Exception as e:
            self.logger.error(f"Failed to auto-generate tests for {component_path}: {e}")
            return []

    def _generate_class_tests(self, class_node: ast.ClassDef) -> str:
        """Generate test code for a class."""
        class_name = class_node.name
        test_code = f"""
import pytest
from unittest.mock import Mock, patch
from {self._get_import_path()} import {class_name}

class Test{class_name}:
    def setup_method(self):
        self.instance = {class_name}()
    
    def test_{class_name.lower()}_initialization(self):
        assert self.instance is not None
        assert isinstance(self.instance, {class_name})
"""
        
        # Generate tests for public methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                method_test = self._generate_method_test(node, class_name)
                test_code += method_test
        
        return test_code

    def _generate_function_tests(self, func_node: ast.FunctionDef) -> str:
        """Generate test code for a function."""
        func_name = func_node.name
        test_code = f"""
import pytest
from unittest.mock import Mock, patch
from {self._get_import_path()} import {func_name}

def test_{func_name}():
    # Test with valid inputs
    result = {func_name}()
    assert result is not None
    
def test_{func_name}_with_invalid_input():
    # Test with invalid inputs
    with pytest.raises(Exception):
        {func_name}(invalid_param="invalid")
"""
        return test_code

    def _generate_method_test(self, method_node: ast.FunctionDef, class_name: str) -> str:
        """Generate test code for a class method."""
        method_name = method_node.name
        test_code = f"""
    
    def test_{method_name}(self):
        result = self.instance.{method_name}()
        # Add specific assertions based on method signature and expected behavior
        assert result is not None
    
    def test_{method_name}_error_handling(self):
        # Test error handling
        try:
            self.instance.{method_name}()
        except Exception as e:
            pytest.fail(f"Method {method_name} raised unexpected exception: {{e}}")
"""
        return test_code

    def _get_import_path(self) -> str:
        """Get the import path for the component being tested."""
        # This would need to be customized based on project structure
        return "src.components.your_component"

    def select_relevant_tests(self, changed_files: List[str]) -> List[str]:
        """Select relevant tests based on changed files."""
        relevant_tests = []
        
        # Simple strategy: if a source file changed, run its corresponding test
        for file_path in changed_files:
            if file_path.endswith('.py') and 'src/' in file_path:
                # Convert src/components/example.py to tests/test_example.py
                relative_path = file_path.replace('src/', '').replace('.py', '')
                test_file = f"tests/test_{relative_path.replace('/', '_')}.py"
                relevant_tests.append(test_file)
        
        return relevant_tests

    def prioritize_tests(self, test_results: List[TestResult]) -> List[TestResult]:
        """Prioritize tests based on importance and failure likelihood."""
        # Sort by priority (highest first), then by previous failure rate
        prioritized = sorted(
            test_results, 
            key=lambda x: (x.priority, x.status == TestStatus.FAILED), 
            reverse=True
        )
        return prioritized

    def analyze_test_results(self) -> Dict[str, Any]:
        """Analyze test results and provide insights."""
        total_tests = len(self.test_results)
        if total_tests == 0:
            return {"message": "No tests run"}
        
        passed = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        skipped = len([r for r in self.test_results if r.status == TestStatus.SKIPPED])
        errors = len([r for r in self.test_results if r.status == TestStatus.ERROR])
        
        analysis = {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "pass_rate": (passed / total_tests) * 100,
            "average_duration": sum(r.duration for r in self.test_results) / total_tests,
            "failed_tests": [r.test_name for r in self.test_results if r.status == TestStatus.FAILED],
            "recommendations": self._generate_recommendations()
        }
        
        return analysis

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if r.status == TestStatus.FAILED]
        if failed_tests:
            recommendations.append(f"Fix {len(failed_tests)} failed tests before deployment")
        
        long_tests = [r for r in self.test_results if r.duration > 10]  # 10 seconds
        if long_tests:
            recommendations.append(f"Optimize {len(long_tests)} slow tests for better performance")
        
        static_issues = [r for r in self.test_results if r.test_type == TestType.STATIC_ANALYSIS and r.status == TestStatus.FAILED]
        if static_issues:
            recommendations.append("Address static analysis issues for code quality")
        
        return recommendations

    def generate_test_report(self, output_file: str = "test_report.html"):
        """Generate a comprehensive test report."""
        analysis = self.analyze_test_results()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - Omni-Dev Agent</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; background: #e8f4f8; border-radius: 5px; }}
        .passed {{ background: #d4edda; }}
        .failed {{ background: #f8d7da; }}
        .test-details {{ margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Report - Omni-Dev Agent</h1>
        <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="metric passed">
            <h3>{analysis['passed']}</h3>
            <p>Passed</p>
        </div>
        <div class="metric failed">
            <h3>{analysis['failed']}</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3>{analysis['pass_rate']:.1f}%</h3>
            <p>Pass Rate</p>
        </div>
        <div class="metric">
            <h3>{analysis['average_duration']:.2f}s</h3>
            <p>Avg Duration</p>
        </div>
    </div>
    
    <div class="test-details">
        <h2>Test Details</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Error</th>
            </tr>
"""
        
        for result in self.test_results:
            status_class = result.status.value
            html_content += f"""
            <tr class="{status_class}">
                <td>{result.test_name}</td>
                <td>{result.test_type.value}</td>
                <td>{result.status.value}</td>
                <td>{result.duration:.3f}s</td>
                <td>{result.error_message or ''}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
        
        for rec in analysis.get('recommendations', []):
            html_content += f"<li>{rec}</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
        
        with open(self.project_root / output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Test report generated: {output_file}")

    def get_feedback_loop_data(self) -> Dict[str, Any]:
        """Generate feedback data for the agent to learn from."""
        return {
            "test_results": [asdict(result) for result in self.test_results],
            "analysis": self.analyze_test_results(),
            "patterns": self._identify_failure_patterns(),
            "suggestions": self._generate_improvement_suggestions()
        }

    def _identify_failure_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in test failures."""
        patterns = []
        failed_tests = [r for r in self.test_results if r.status == TestStatus.FAILED]
        
        # Group by error message
        error_groups = {}
        for test in failed_tests:
            if test.error_message:
                key = test.error_message[:50]  # First 50 chars
                if key not in error_groups:
                    error_groups[key] = []
                error_groups[key].append(test.test_name)
        
        for error, tests in error_groups.items():
            if len(tests) > 1:  # Pattern if multiple tests have same error
                patterns.append({
                    "error_pattern": error,
                    "affected_tests": tests,
                    "frequency": len(tests)
                })
        
        return patterns

    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate suggestions for improving the integration process."""
        suggestions = []
        
        analysis = self.analyze_test_results()
        
        if analysis["pass_rate"] < 80:
            suggestions.append("Consider improving test coverage and code quality")
        
        if analysis["failed"] > 0:
            suggestions.append("Review failed tests and fix underlying issues")
        
        if analysis["average_duration"] > 5:
            suggestions.append("Optimize test performance to reduce execution time")
        
        return suggestions
