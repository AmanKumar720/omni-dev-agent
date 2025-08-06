"""
System Integration Manager for Omni-Dev Agent
Orchestrates integrations with any software on the operating system
"""

import json
import os
import subprocess
import platform
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

from .system_detector import SystemDetector, SoftwareInfo, SystemInfo

@dataclass
class IntegrationTask:
    """Represents a system integration task"""
    task_name: str
    description: str
    required_software: List[str]
    commands: List[str]
    category: str

@dataclass
class IntegrationResult:
    """Result of a system integration task"""
    task_name: str
    success: bool
    software_used: List[str]
    commands_executed: List[str]
    output: Dict[str, Any]
    timestamp: datetime

class SystemManager:
    """Manages system-wide software integrations"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.system_detector = SystemDetector()
        self.integration_tasks = self._load_integration_tasks()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for system manager"""
        logger = logging.getLogger("SystemManager")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("system_manager.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_integration_tasks(self) -> Dict[str, IntegrationTask]:
        """Load predefined integration tasks"""
        return {
            "web_development": IntegrationTask(
                task_name="Web Development Setup",
                description="Set up a complete web development environment",
                required_software=["python", "nodejs", "git", "vscode"],
                commands=[
                    "python --version",
                    "node --version",
                    "git --version",
                    "code --version"
                ],
                category="development"
            ),
            "database_management": IntegrationTask(
                task_name="Database Management",
                description="Set up and manage databases",
                required_software=["postgresql", "mysql"],
                commands=[
                    "psql --version",
                    "mysql --version"
                ],
                category="database"
            ),
            "cloud_development": IntegrationTask(
                task_name="Cloud Development",
                description="Set up cloud development tools",
                required_software=["aws", "azure", "gcloud", "docker"],
                commands=[
                    "aws --version",
                    "az --version",
                    "gcloud --version",
                    "docker --version"
                ],
                category="cloud"
            ),
            "ai_development": IntegrationTask(
                task_name="AI Development",
                description="Set up AI development environment",
                required_software=["python", "cursor", "git"],
                commands=[
                    "python -c 'import torch; print(torch.__version__)'",
                    "python -c 'import tensorflow; print(tensorflow.__version__)'",
                    "cursor --version"
                ],
                category="ai"
            ),
            "system_utilities": IntegrationTask(
                task_name="System Utilities",
                description="Access system utilities",
                required_software=["notepad", "calc", "paint"],
                commands=[
                    "notepad",
                    "calc",
                    "mspaint"
                ],
                category="utilities"
            )
        }
    
    def get_system_info(self) -> SystemInfo:
        """Get system information"""
        return self.system_detector.get_system_info()
    
    def get_installed_software(self, category: str = None) -> Dict[str, SoftwareInfo]:
        """Get installed software"""
        return self.system_detector.get_installed_software(category)
    
    def launch_software(self, software_id: str, args: List[str] = None) -> Dict[str, Any]:
        """Launch a software application"""
        return self.system_detector.launch_software(software_id, args)
    
    def execute_software_command(self, software_id: str, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute a command with a specific software"""
        return self.system_detector.execute_software_command(software_id, command, args)
    
    def search_software(self, query: str) -> List[SoftwareInfo]:
        """Search for software"""
        return self.system_detector.search_software(query)
    
    def get_software_categories(self) -> List[str]:
        """Get all software categories"""
        return self.system_detector.get_software_categories()
    
    def execute_integration_task(self, task_name: str) -> IntegrationResult:
        """Execute a predefined integration task"""
        if task_name not in self.integration_tasks:
            return IntegrationResult(
                task_name=task_name,
                success=False,
                software_used=[],
                commands_executed=[],
                output={"error": f"Task not found: {task_name}"},
                timestamp=datetime.now()
            )
        
        task = self.integration_tasks[task_name]
        installed_software = self.system_detector.get_installed_software()
        
        # Check if required software is installed
        missing_software = []
        for required in task.required_software:
            if required not in installed_software or not installed_software[required].installed:
                missing_software.append(required)
        
        if missing_software:
            return IntegrationResult(
                task_name=task_name,
                success=False,
                software_used=[],
                commands_executed=[],
                output={"error": f"Missing software: {', '.join(missing_software)}"},
                timestamp=datetime.now()
            )
        
        # Execute commands
        results = {}
        commands_executed = []
        software_used = []
        
        for command in task.commands:
            try:
                # Try to execute the command
                result = subprocess.run(command.split(), capture_output=True, text=True, timeout=30)
                results[command] = {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                commands_executed.append(command)
                
                # Determine which software was used
                for software_id, software in installed_software.items():
                    if software.executable in command:
                        software_used.append(software_id)
                
            except Exception as e:
                results[command] = {
                    "success": False,
                    "error": str(e)
                }
        
        success = all(result.get("success", False) for result in results.values())
        
        return IntegrationResult(
            task_name=task_name,
            success=success,
            software_used=list(set(software_used)),
            commands_executed=commands_executed,
            output=results,
            timestamp=datetime.now()
        )
    
    def create_custom_integration(self, task_name: str, description: str, 
                                required_software: List[str], commands: List[str], 
                                category: str = "custom") -> bool:
        """Create a custom integration task"""
        try:
            self.integration_tasks[task_name] = IntegrationTask(
                task_name=task_name,
                description=description,
                required_software=required_software,
                commands=commands,
                category=category
            )
            self.logger.info(f"Created custom integration task: {task_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create custom integration: {e}")
            return False
    
    def get_available_tasks(self) -> Dict[str, IntegrationTask]:
        """Get all available integration tasks"""
        return self.integration_tasks
    
    def get_tasks_by_category(self, category: str) -> Dict[str, IntegrationTask]:
        """Get tasks filtered by category"""
        return {name: task for name, task in self.integration_tasks.items() 
                if task.category == category}
    
    def integrate_with_omni_agent(self, agent_task: str) -> Dict[str, Any]:
        """Integrate Omni-Dev Agent tasks with system software"""
        integration_result = {
            "agent_task": agent_task,
            "system_info": self.system_detector.get_system_info(),
            "recommended_software": [],
            "recommended_tasks": [],
            "available_software": self.system_detector.get_installed_software(),
            "timestamp": datetime.now().isoformat()
        }
        
        task_lower = agent_task.lower()
        installed_software = self.system_detector.get_installed_software()
        
        # Recommend software based on task
        if "web" in task_lower or "frontend" in task_lower:
            integration_result["recommended_software"].extend([
                "vscode", "nodejs", "git", "chrome"
            ])
            integration_result["recommended_tasks"].append("web_development")
        
        if "database" in task_lower or "sql" in task_lower:
            integration_result["recommended_software"].extend([
                "postgresql", "mysql"
            ])
            integration_result["recommended_tasks"].append("database_management")
        
        if "cloud" in task_lower or "aws" in task_lower or "azure" in task_lower:
            integration_result["recommended_software"].extend([
                "aws", "azure", "gcloud", "docker"
            ])
            integration_result["recommended_tasks"].append("cloud_development")
        
        if "ai" in task_lower or "machine learning" in task_lower:
            integration_result["recommended_software"].extend([
                "python", "cursor", "git"
            ])
            integration_result["recommended_tasks"].append("ai_development")
        
        if "code" in task_lower or "development" in task_lower:
            integration_result["recommended_software"].extend([
                "python", "vscode", "cursor", "git"
            ])
            integration_result["recommended_tasks"].append("web_development")
        
        # Filter to only installed software
        integration_result["recommended_software"] = [
            sw for sw in integration_result["recommended_software"]
            if sw in installed_software and installed_software[sw].installed
        ]
        
        return integration_result
    
    def run_system_scan(self) -> Dict[str, Any]:
        """Run a comprehensive system scan"""
        scan_result = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_detector.get_system_info(),
            "installed_software": {},
            "software_categories": {},
            "integration_capabilities": [],
            "recommendations": []
        }
        
        # Get all installed software
        installed_software = self.system_detector.get_installed_software()
        scan_result["installed_software"] = {
            sw_id: {
                "name": sw.name,
                "version": sw.version,
                "category": sw.category,
                "path": sw.path
            }
            for sw_id, sw in installed_software.items()
        }
        
        # Group by categories
        for software in installed_software.values():
            if software.category not in scan_result["software_categories"]:
                scan_result["software_categories"][software.category] = []
            scan_result["software_categories"][software.category].append(software.name)
        
        # Determine integration capabilities
        if "python" in installed_software:
            scan_result["integration_capabilities"].append("Python Development")
        if "nodejs" in installed_software:
            scan_result["integration_capabilities"].append("JavaScript Development")
        if "git" in installed_software:
            scan_result["integration_capabilities"].append("Version Control")
        if "docker" in installed_software:
            scan_result["integration_capabilities"].append("Container Management")
        if "aws" in installed_software or "azure" in installed_software:
            scan_result["integration_capabilities"].append("Cloud Development")
        if "cursor" in installed_software or "vscode" in installed_software:
            scan_result["integration_capabilities"].append("IDE Integration")
        
        # Generate recommendations
        if "python" in installed_software and "git" in installed_software:
            scan_result["recommendations"].append("Ready for Python development with version control")
        if "nodejs" in installed_software and "git" in installed_software:
            scan_result["recommendations"].append("Ready for JavaScript development with version control")
        if "docker" in installed_software:
            scan_result["recommendations"].append("Ready for containerized development")
        if "aws" in installed_software:
            scan_result["recommendations"].append("Ready for AWS cloud development")
        
        return scan_result
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get a summary of system integration capabilities"""
        return {
            "system_info": self.system_detector.get_system_info(),
            "software_summary": {
                "total_installed": len(self.system_detector.get_installed_software()),
                "categories": self.system_detector.get_software_categories(),
                "available_tasks": len(self.integration_tasks)
            },
            "integration_features": [
                "Software detection and monitoring",
                "Application launching and control",
                "Command execution and automation",
                "System-wide task orchestration",
                "Cross-platform compatibility",
                "Custom integration creation",
                "Real-time system scanning",
                "Intelligent software recommendations"
            ]
        } 