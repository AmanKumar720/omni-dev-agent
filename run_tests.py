#!/usr/bin/env python3
"""
Test runner script for omni-dev-agent project.
Handles Python path configuration and runs tests with appropriate environment setup.
"""

import sys
import os
import subprocess

def main():
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env['PYTHONPATH'] = current_dir
    
    # Run pytest with verbose output
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-v']
    
    if len(sys.argv) > 1:
        # Allow passing specific test files or options
        cmd.extend(sys.argv[1:])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, env=env, cwd=current_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
