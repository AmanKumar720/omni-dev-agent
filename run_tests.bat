@echo off
REM Test runner batch file for omni-dev-agent project on Windows
REM Sets up Python path and runs tests

echo Setting up environment...
set PYTHONPATH=%~dp0

echo Running tests...
python run_tests.py %*

if %ERRORLEVEL% NEQ 0 (
    echo Tests failed with error code %ERRORLEVEL%
    pause
) else (
    echo All tests passed successfully!
)
