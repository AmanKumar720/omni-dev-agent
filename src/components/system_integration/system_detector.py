"""
System Integration Detector for Omni-Dev Agent
Detects and integrates with various software applications on the operating system
"""

import os
import sys
import subprocess
import platform
import json
import winreg
import glob
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SoftwareInfo:
    """Information about installed software"""
    name: str
    version: str
    path: str
    executable: str
    category: str
    installed: bool
    launch_command: str
    description: str

@dataclass
class SystemInfo:
    """System information"""
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    total_software: int
    categories: List[str]

class SystemDetector:
    """Detects and manages system software integration"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.system_info = self._get_system_info()
        self.software_registry = self._load_software_registry()
        self.detected_software = self._detect_installed_software()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for system detector"""
        logger = logging.getLogger("SystemDetector")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("system_integration.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _get_system_info(self) -> SystemInfo:
        """Get system information"""
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            architecture=platform.architecture()[0],
            python_version=sys.version,
            total_software=0,
            categories=[]
        )
    
    def _load_software_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load software registry with known applications"""
        return {
            "development": {
                "python": {
                    "name": "Python",
                    "executable": "python",
                    "category": "development",
                    "description": "Python programming language"
                },
                "nodejs": {
                    "name": "Node.js",
                    "executable": "node",
                    "category": "development",
                    "description": "JavaScript runtime"
                },
                "git": {
                    "name": "Git",
                    "executable": "git",
                    "category": "development",
                    "description": "Version control system"
                },
                "docker": {
                    "name": "Docker",
                    "executable": "docker",
                    "category": "development",
                    "description": "Container platform"
                },
                "postgresql": {
                    "name": "PostgreSQL",
                    "executable": "psql",
                    "category": "development",
                    "description": "Database management system"
                },
                "mysql": {
                    "name": "MySQL",
                    "executable": "mysql",
                    "category": "development",
                    "description": "Database management system"
                }
            },
            "ides": {
                "cursor": {
                    "name": "Cursor IDE",
                    "executable": "cursor",
                    "category": "ides",
                    "description": "AI-powered code editor"
                },
                "vscode": {
                    "name": "Visual Studio Code",
                    "executable": "code",
                    "category": "ides",
                    "description": "Popular code editor"
                },
                "pycharm": {
                    "name": "PyCharm",
                    "executable": "pycharm",
                    "category": "ides",
                    "description": "Python IDE"
                },
                "intellij": {
                    "name": "IntelliJ IDEA",
                    "executable": "idea",
                    "category": "ides",
                    "description": "Java IDE"
                },
                "sublime": {
                    "name": "Sublime Text",
                    "executable": "subl",
                    "category": "ides",
                    "description": "Text editor"
                }
            },
            "browsers": {
                "chrome": {
                    "name": "Google Chrome",
                    "executable": "chrome",
                    "category": "browsers",
                    "description": "Web browser"
                },
                "firefox": {
                    "name": "Mozilla Firefox",
                    "executable": "firefox",
                    "category": "browsers",
                    "description": "Web browser"
                },
                "edge": {
                    "name": "Microsoft Edge",
                    "executable": "msedge",
                    "category": "browsers",
                    "description": "Web browser"
                }
            },
            "cloud": {
                "aws": {
                    "name": "AWS CLI",
                    "executable": "aws",
                    "category": "cloud",
                    "description": "Amazon Web Services CLI"
                },
                "azure": {
                    "name": "Azure CLI",
                    "executable": "az",
                    "category": "cloud",
                    "description": "Microsoft Azure CLI"
                },
                "gcloud": {
                    "name": "Google Cloud CLI",
                    "executable": "gcloud",
                    "category": "cloud",
                    "description": "Google Cloud Platform CLI"
                }
            },
            "utilities": {
                "notepad": {
                    "name": "Notepad",
                    "executable": "notepad",
                    "category": "utilities",
                    "description": "Text editor"
                },
                "calc": {
                    "name": "Calculator",
                    "executable": "calc",
                    "category": "utilities",
                    "description": "Windows calculator"
                },
                "paint": {
                    "name": "Paint",
                    "executable": "mspaint",
                    "category": "utilities",
                    "description": "Image editor"
                }
            }
        }
    
    def _detect_installed_software(self) -> Dict[str, SoftwareInfo]:
        """Detect installed software on the system"""
        detected = {}
        
        # Detect software from registry
        for category, software_dict in self.software_registry.items():
            for software_id, software_info in software_dict.items():
                software = self._check_software_installation(software_id, software_info)
                if software.installed:
                    detected[software_id] = software
        
        # Detect additional software from system
        additional_software = self._detect_additional_software()
        detected.update(additional_software)
        
        # Update system info
        self.system_info.total_software = len(detected)
        self.system_info.categories = list(set([sw.category for sw in detected.values()]))
        
        return detected
    
    def _check_software_installation(self, software_id: str, software_info: Dict[str, Any]) -> SoftwareInfo:
        """Check if a specific software is installed"""
        executable = software_info["executable"]
        installed = False
        version = "Unknown"
        path = ""
        launch_command = executable
        
        try:
            # Try to run the executable to check if it's installed
            result = subprocess.run([executable, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                installed = True
                version = result.stdout.strip().split('\n')[0]
                path = self._find_executable_path(executable)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Try alternative methods for Windows
            if platform.system() == "Windows":
                installed, version, path = self._check_windows_software(software_id, software_info)
        
        return SoftwareInfo(
            name=software_info["name"],
            version=version,
            path=path,
            executable=executable,
            category=software_info["category"],
            installed=installed,
            launch_command=launch_command,
            description=software_info["description"]
        )
    
    def _check_windows_software(self, software_id: str, software_info: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Check software installation on Windows using registry"""
        installed = False
        version = "Unknown"
        path = ""
        
        # Common registry paths for software
        registry_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
        ]
        
        try:
            for reg_path in registry_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                        # Try to find the software in registry
                        for i in range(winreg.QueryInfoKey(key)[0]):
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                with winreg.OpenKey(key, subkey_name) as subkey:
                                    if software_id.lower() in subkey_name.lower():
                                        installed = True
                                        try:
                                            path, _ = winreg.QueryValueEx(subkey, "")
                                            version = subkey_name
                                        except:
                                            pass
                                        break
                            except:
                                continue
                except:
                    continue
        except:
            pass
        
        return installed, version, path
    
    def _find_executable_path(self, executable: str) -> str:
        """Find the full path of an executable"""
        try:
            result = subprocess.run(["where", executable] if platform.system() == "Windows" else ["which", executable],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return ""
    
    def _detect_additional_software(self) -> Dict[str, SoftwareInfo]:
        """Detect additional software not in the registry"""
        additional = {}
        
        # Common installation directories
        common_paths = []
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files",
                r"C:\Program Files (x86)",
                r"C:\Users\%USERNAME%\AppData\Local\Programs",
                r"C:\Users\%USERNAME%\AppData\Roaming"
            ]
        else:
            common_paths = [
                "/usr/bin",
                "/usr/local/bin",
                "/opt",
                "/Applications"
            ]
        
        # Scan for additional software
        for base_path in common_paths:
            expanded_path = os.path.expandvars(base_path)
            if os.path.exists(expanded_path):
                for item in os.listdir(expanded_path):
                    item_path = os.path.join(expanded_path, item)
                    if os.path.isdir(item_path):
                        # Look for executables
                        for ext in [".exe", ".app", ""]:
                            exe_path = os.path.join(item_path, f"{item}{ext}")
                            if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                                software_id = item.lower().replace(" ", "_")
                                additional[software_id] = SoftwareInfo(
                                    name=item,
                                    version="Unknown",
                                    path=item_path,
                                    executable=item,
                                    category="additional",
                                    installed=True,
                                    launch_command=exe_path,
                                    description=f"Additional software: {item}"
                                )
                                break
        
        return additional
    
    def get_system_info(self) -> SystemInfo:
        """Get system information"""
        return self.system_info
    
    def get_installed_software(self, category: str = None) -> Dict[str, SoftwareInfo]:
        """Get installed software, optionally filtered by category"""
        if category:
            return {k: v for k, v in self.detected_software.items() if v.category == category}
        return self.detected_software
    
    def launch_software(self, software_id: str, args: List[str] = None) -> Dict[str, Any]:
        """Launch a software application"""
        if software_id not in self.detected_software:
            return {"success": False, "error": f"Software not found: {software_id}"}
        
        software = self.detected_software[software_id]
        if not software.installed:
            return {"success": False, "error": f"Software not installed: {software_id}"}
        
        try:
            cmd = [software.launch_command]
            if args:
                cmd.extend(args)
            
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            self.logger.info(f"Launched {software.name}")
            return {
                "success": True,
                "software": software.name,
                "command": " ".join(cmd)
            }
        except Exception as e:
            self.logger.error(f"Failed to launch {software.name}: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_software_command(self, software_id: str, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute a command with a specific software"""
        if software_id not in self.detected_software:
            return {"success": False, "error": f"Software not found: {software_id}"}
        
        software = self.detected_software[software_id]
        if not software.installed:
            return {"success": False, "error": f"Software not installed: {software_id}"}
        
        try:
            cmd = [software.executable, command]
            if args:
                cmd.extend(args)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            return {
                "success": result.returncode == 0,
                "software": software.name,
                "command": " ".join(cmd),
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_software_categories(self) -> List[str]:
        """Get all software categories"""
        return list(set([sw.category for sw in self.detected_software.values()]))
    
    def search_software(self, query: str) -> List[SoftwareInfo]:
        """Search for software by name or description"""
        query_lower = query.lower()
        results = []
        
        for software in self.detected_software.values():
            if (query_lower in software.name.lower() or 
                query_lower in software.description.lower() or
                query_lower in software.category.lower()):
                results.append(software)
        
        return results
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get a summary of system integration capabilities"""
        return {
            "system_info": {
                "os_name": self.system_info.os_name,
                "os_version": self.system_info.os_version,
                "architecture": self.system_info.architecture,
                "python_version": self.system_info.python_version
            },
            "software_summary": {
                "total_installed": self.system_info.total_software,
                "categories": self.system_info.categories,
                "categories_count": len(self.system_info.categories)
            },
            "integration_features": [
                "Software detection and monitoring",
                "Application launching",
                "Command execution",
                "System information gathering",
                "Cross-platform compatibility",
                "Registry integration (Windows)",
                "File system scanning",
                "Process management"
            ]
        } 