"""
IDE Integration Manager for Omni-Dev Agent
Provides integration with multiple IDEs for enhanced development experience
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

from .cursor_integration import CursorIntegration

@dataclass
class IDEInfo:
    """Information about an IDE"""
    name: str
    executable: str
    version: str
    installed: bool
    workspace_support: bool
    extension_support: bool

@dataclass
class IDECommand:
    """Represents an IDE command"""
    ide: str
    command: str
    description: str
    category: str
    shortcut: Optional[str] = None

class IDEIntegration(ABC):
    """Abstract base class for IDE integrations"""
    
    @abstractmethod
    def is_installed(self) -> bool:
        """Check if the IDE is installed"""
        pass
    
    @abstractmethod
    def open_workspace(self, path: str) -> bool:
        """Open workspace in the IDE"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute a command in the IDE"""
        pass
    
    @abstractmethod
    def get_available_commands(self) -> List[IDECommand]:
        """Get available commands for the IDE"""
        pass

class VSCodeIntegration(IDEIntegration):
    """Integration with Visual Studio Code"""
    
    def __init__(self):
        self.name = "Visual Studio Code"
        self.executable = "code"
    
    def is_installed(self) -> bool:
        try:
            result = subprocess.run([self.executable, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def open_workspace(self, path: str) -> bool:
        try:
            subprocess.Popen([self.executable, path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    
    def execute_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        try:
            cmd_args = [self.executable, "--command", command]
            if args:
                cmd_args.extend(args)
            
            result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=10)
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_available_commands(self) -> List[IDECommand]:
        return [
            IDECommand("vscode", "workbench.action.quickOpen", "Quick Open", "Navigation", "Ctrl+P"),
            IDECommand("vscode", "workbench.action.showCommands", "Command Palette", "Navigation", "Ctrl+Shift+P"),
            IDECommand("vscode", "workbench.action.terminal.new", "New Terminal", "Terminal", "Ctrl+`"),
            IDECommand("vscode", "workbench.action.debug.start", "Start Debugging", "Debug", "F5"),
            IDECommand("vscode", "workbench.action.tasks.runTask", "Run Task", "Tasks", "Ctrl+Shift+P")
        ]

class PyCharmIntegration(IDEIntegration):
    """Integration with PyCharm"""
    
    def __init__(self):
        self.name = "PyCharm"
        self.executable = "pycharm"
    
    def is_installed(self) -> bool:
        # Check common PyCharm installation paths
        pycharm_paths = [
            r"C:\Program Files\JetBrains\PyCharm Community Edition\bin\pycharm64.exe",
            r"C:\Program Files\JetBrains\PyCharm Professional\bin\pycharm64.exe",
            "/Applications/PyCharm CE.app/Contents/MacOS/pycharm",
            "/Applications/PyCharm.app/Contents/MacOS/pycharm"
        ]
        
        for path in pycharm_paths:
            if os.path.exists(path):
                self.executable = path
                return True
        
        # Try command line
        try:
            result = subprocess.run([self.executable, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def open_workspace(self, path: str) -> bool:
        try:
            subprocess.Popen([self.executable, path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    
    def execute_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        # PyCharm doesn't have a direct command line API like VSCode
        return {"success": False, "error": "PyCharm command execution not supported via CLI"}
    
    def get_available_commands(self) -> List[IDECommand]:
        return [
            IDECommand("pycharm", "FindAction", "Find Action", "Navigation", "Ctrl+Shift+A"),
            IDECommand("pycharm", "GotoClass", "Go to Class", "Navigation", "Ctrl+N"),
            IDECommand("pycharm", "GotoFile", "Go to File", "Navigation", "Ctrl+Shift+N"),
            IDECommand("pycharm", "RunClass", "Run Class", "Run", "Ctrl+Shift+F10"),
            IDECommand("pycharm", "DebugClass", "Debug Class", "Debug", "Ctrl+Shift+D")
        ]

class IDEManager:
    """Manager for multiple IDE integrations"""
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.logger = self._setup_logging()
        
        # Initialize IDE integrations
        self.integrations = {
            "cursor": CursorIntegration(str(self.workspace_path)),
            "vscode": VSCodeIntegration(),
            "pycharm": PyCharmIntegration()
        }
        
        self.available_ides = self._detect_available_ides()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for IDE manager"""
        logger = logging.getLogger("IDEManager")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("ide_integration.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _detect_available_ides(self) -> Dict[str, IDEInfo]:
        """Detect which IDEs are available on the system"""
        available_ides = {}
        
        for ide_name, integration in self.integrations.items():
            installed = integration.is_installed()
            version = "Unknown"
            
            if installed:
                # Try to get version
                try:
                    if ide_name == "cursor":
                        result = subprocess.run(["cursor", "--version"], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            version = result.stdout.strip()
                    elif ide_name == "vscode":
                        result = subprocess.run(["code", "--version"], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            version = result.stdout.strip().split('\n')[0]
                except Exception:
                    pass
            
            available_ides[ide_name] = IDEInfo(
                name=integration.name if hasattr(integration, 'name') else ide_name.title(),
                executable=getattr(integration, 'executable', ide_name),
                version=version,
                installed=installed,
                workspace_support=True,  # All modern IDEs support workspaces
                extension_support=ide_name in ["cursor", "vscode"]
            )
        
        return available_ides
    
    def get_available_ides(self) -> Dict[str, IDEInfo]:
        """Get information about available IDEs"""
        return self.available_ides
    
    def open_in_ide(self, ide_name: str, path: str = None) -> bool:
        """Open workspace in specified IDE"""
        if ide_name not in self.integrations:
            self.logger.error(f"IDE not supported: {ide_name}")
            return False
        
        if not self.available_ides[ide_name].installed:
            self.logger.error(f"IDE not installed: {ide_name}")
            return False
        
        target_path = path if path else str(self.workspace_path)
        return self.integrations[ide_name].open_workspace(target_path)
    
    def execute_ide_command(self, ide_name: str, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute a command in the specified IDE"""
        if ide_name not in self.integrations:
            return {"success": False, "error": f"IDE not supported: {ide_name}"}
        
        if not self.available_ides[ide_name].installed:
            return {"success": False, "error": f"IDE not installed: {ide_name}"}
        
        return self.integrations[ide_name].execute_command(command, args)
    
    def get_ide_commands(self, ide_name: str) -> List[IDECommand]:
        """Get available commands for the specified IDE"""
        if ide_name not in self.integrations:
            return []
        
        return self.integrations[ide_name].get_available_commands()
    
    def integrate_with_omni_agent(self, agent_task: str, preferred_ide: str = None) -> Dict[str, Any]:
        """Integrate Omni-Dev Agent tasks with available IDEs"""
        integration_result = {
            "task": agent_task,
            "available_ides": self.available_ides,
            "ide_suggestions": {},
            "recommended_ide": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine recommended IDE
        if preferred_ide and preferred_ide in self.available_ides and self.available_ides[preferred_ide].installed:
            integration_result["recommended_ide"] = preferred_ide
        else:
            # Auto-select best IDE based on task
            task_lower = agent_task.lower()
            
            if "ai" in task_lower or "generate" in task_lower:
                # Prefer Cursor for AI tasks
                if "cursor" in self.available_ides and self.available_ides["cursor"].installed:
                    integration_result["recommended_ide"] = "cursor"
                elif "vscode" in self.available_ides and self.available_ides["vscode"].installed:
                    integration_result["recommended_ide"] = "vscode"
            elif "python" in task_lower or "debug" in task_lower:
                # Prefer PyCharm for Python tasks
                if "pycharm" in self.available_ides and self.available_ides["pycharm"].installed:
                    integration_result["recommended_ide"] = "pycharm"
                elif "vscode" in self.available_ides and self.available_ides["vscode"].installed:
                    integration_result["recommended_ide"] = "vscode"
            else:
                # Default to first available IDE
                for ide_name, ide_info in self.available_ides.items():
                    if ide_info.installed:
                        integration_result["recommended_ide"] = ide_name
                        break
        
        # Generate IDE-specific suggestions
        for ide_name, ide_info in self.available_ides.items():
            if ide_info.installed:
                if ide_name == "cursor":
                    # Use Cursor's specific integration
                    cursor_integration = self.integrations["cursor"]
                    ide_suggestions = cursor_integration.integrate_with_omni_agent(agent_task)
                    integration_result["ide_suggestions"][ide_name] = ide_suggestions
                else:
                    # Generic suggestions for other IDEs
                    integration_result["ide_suggestions"][ide_name] = {
                        "commands": self.get_ide_commands(ide_name),
                        "suggestions": [
                            f"Use {ide_name.title()} for development",
                            f"Open workspace in {ide_name.title()}",
                            f"Use {ide_name.title()} extensions for enhanced functionality"
                        ]
                    }
        
        return integration_result
    
    def create_ide_extension(self, ide_name: str, extension_name: str, features: List[str]) -> str:
        """Create an extension for the specified IDE"""
        if ide_name not in self.integrations:
            raise ValueError(f"IDE not supported: {ide_name}")
        
        if ide_name == "cursor":
            # Use Cursor's extension creation
            cursor_integration = self.integrations["cursor"]
            return cursor_integration.create_cursor_extension(extension_name, features)
        elif ide_name == "vscode":
            # Create VSCode extension
            return self._create_vscode_extension(extension_name, features)
        else:
            raise ValueError(f"Extension creation not supported for {ide_name}")
    
    def _create_vscode_extension(self, extension_name: str, features: List[str]) -> str:
        """Create a VSCode extension"""
        extension_dir = self.workspace_path / "vscode-extensions" / extension_name
        extension_dir.mkdir(parents=True, exist_ok=True)
        
        # Create package.json
        package_json = {
            "name": f"omni-dev-agent-{extension_name}",
            "displayName": f"Omni-Dev Agent {extension_name.title()}",
            "description": f"Integration with Omni-Dev Agent for {extension_name}",
            "version": "1.0.0",
            "publisher": "omni-dev-agent",
            "engines": {
                "vscode": "^1.60.0"
            },
            "categories": ["Other"],
            "activationEvents": [
                "onCommand:omni-dev-agent.activate"
            ],
            "main": "./out/extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "omni-dev-agent.activate",
                        "title": "Activate Omni-Dev Agent"
                    }
                ],
                "menus": {
                    "commandPalette": [
                        {
                            "command": "omni-dev-agent.activate"
                        }
                    ]
                }
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "tsc -p ./",
                "watch": "tsc -watch -p ./"
            },
            "devDependencies": {
                "@types/vscode": "^1.60.0",
                "@types/node": "^16.0.0",
                "typescript": "^4.5.0"
            }
        }
        
        with open(extension_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Create extension.ts
        extension_ts = f'''
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {{
    console.log('Omni-Dev Agent {extension_name} extension is now active!');
    
    let disposable = vscode.commands.registerCommand('omni-dev-agent.activate', () => {{
        vscode.window.showInformationMessage('Omni-Dev Agent {extension_name} activated!');
        
        // Integrate with Omni-Dev Agent features: {", ".join(features)}
        vscode.window.showInformationMessage('Features available: {", ".join(features)}');
    }});
    
    context.subscriptions.push(disposable);
}}

export function deactivate() {{}}
'''
        
        (extension_dir / "src").mkdir(exist_ok=True)
        with open(extension_dir / "src" / "extension.ts", 'w') as f:
            f.write(extension_ts)
        
        self.logger.info(f"Created VSCode extension: {extension_dir}")
        return str(extension_dir)
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get a summary of IDE integration capabilities"""
        return {
            "total_ides": len(self.available_ides),
            "installed_ides": [name for name, info in self.available_ides.items() if info.installed],
            "available_ides": self.available_ides,
            "workspace_path": str(self.workspace_path),
            "integration_features": [
                "Workspace opening",
                "Command execution",
                "Extension creation",
                "AI task integration",
                "Multi-IDE support"
            ]
        } 