
import sys
import os
import runpy

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add the src directory to the Python path as well
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"Project Root: {project_root}")
print(f"Src Path: {src_path}")
print(f"Sys Path: {sys.path}")

# Run the desktop application
runpy.run_module('desktop_app', run_name='__main__')
