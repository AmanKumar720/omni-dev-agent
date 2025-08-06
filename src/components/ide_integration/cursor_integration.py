"""
Cursor IDE Integration for Omni-Dev Agent
Provides seamless integration with Cursor IDE for enhanced development experience
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

# Import IDE integration base classes
try:
    from .ide_manager import IDEIntegration, IDECommand
except ImportError:
    # Fallback for when ide_manager is not available
    class IDEIntegration:
        pass
    class IDECommand:
        pass

@dataclass
class CursorCommand:
    """Represents a Cursor IDE command"""
    command: str
    description: str
    category: str
    shortcut: Optional[str] = None

@dataclass
class CursorWorkspace:
    """Represents a Cursor workspace"""
    path: str
    name: str
    files: List[str]
    extensions: List[str]
    settings: Dict[str, Any]

class CursorIntegration(IDEIntegration):
    """Integration with Cursor IDE"""
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.logger = self._setup_logging()
        self.cursor_commands = self._load_cursor_commands()
        self.workspace = self._analyze_workspace()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for Cursor integration"""
        logger = logging.getLogger("CursorIntegration")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("cursor_integration.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_cursor_commands(self) -> List[CursorCommand]:
        """Load available Cursor commands"""
        return [
            CursorCommand(
                command="cursor.command.palette",
                description="Open Command Palette",
                category="Navigation",
                shortcut="Ctrl+Shift+P"
            ),
            CursorCommand(
                command="cursor.chat.open",
                description="Open AI Chat",
                category="AI",
                shortcut="Ctrl+L"
            ),
            CursorCommand(
                command="cursor.chat.focus",
                description="Focus AI Chat",
                category="AI",
                shortcut="Ctrl+I"
            ),
            CursorCommand(
                command="cursor.generate",
                description="Generate Code",
                category="AI",
                shortcut="Ctrl+K"
            ),
            CursorCommand(
                command="cursor.explain",
                description="Explain Code",
                category="AI",
                shortcut="Ctrl+Shift+E"
            ),
            CursorCommand(
                command="cursor.fix",
                description="Fix Code Issues",
                category="AI",
                shortcut="Ctrl+Shift+F"
            ),
            CursorCommand(
                command="cursor.test",
                description="Generate Tests",
                category="AI",
                shortcut="Ctrl+Shift+T"
            ),
            CursorCommand(
                command="cursor.document",
                description="Generate Documentation",
                category="AI",
                shortcut="Ctrl+Shift+D"
            ),
            CursorCommand(
                command="cursor.refactor",
                description="Refactor Code",
                category="AI",
                shortcut="Ctrl+Shift+R"
            ),
            CursorCommand(
                command="cursor.debug",
                description="Debug with AI",
                category="AI",
                shortcut="Ctrl+Shift+B"
            )
        ]
    
    def _analyze_workspace(self) -> CursorWorkspace:
        """Analyze the current Cursor workspace"""
        files = []
        extensions = set()
        
        # Scan workspace for files
        for file_path in self.workspace_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                files.append(str(file_path.relative_to(self.workspace_path)))
                if file_path.suffix:
                    extensions.add(file_path.suffix)
        
        # Load workspace settings
        settings_file = self.workspace_path / ".vscode" / "settings.json"
        settings = {}
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load workspace settings: {e}")
        
        return CursorWorkspace(
            path=str(self.workspace_path),
            name=self.workspace_path.name,
            files=files,
            extensions=list(extensions),
            settings=settings
        )
    
    def is_installed(self) -> bool:
        """Check if Cursor is installed on the system"""
        try:
            # Try to find Cursor executable
            if sys.platform == "win32":
                cursor_paths = [
                    r"C:\Users\%USERNAME%\AppData\Local\Programs\cursor\Cursor.exe",
                    r"C:\Program Files\Cursor\Cursor.exe"
                ]
            elif sys.platform == "darwin":
                cursor_paths = [
                    "/Applications/Cursor.app/Contents/MacOS/Cursor"
                ]
            else:  # Linux
                cursor_paths = [
                    "/usr/bin/cursor",
                    "/usr/local/bin/cursor"
                ]
            
            for path in cursor_paths:
                if os.path.exists(os.path.expandvars(path)):
                    return True
            
            # Try to run cursor command
            result = subprocess.run(["cursor", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def open_workspace(self, path: str) -> bool:
        """Open file or workspace in Cursor"""
        try:
            target_path = Path(path)
            
            if not target_path.exists():
                self.logger.error(f"Path does not exist: {target_path}")
                return False
            
            # Open in Cursor
            subprocess.Popen(["cursor", str(target_path)], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            self.logger.info(f"Opened {target_path} in Cursor")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open in Cursor: {e}")
            return False
    
    def execute_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute a Cursor command"""
        try:
            # This would integrate with Cursor's command API
            # For now, we'll simulate command execution
            cmd = self._find_command(command)
            if not cmd:
                return {"success": False, "error": f"Command not found: {command}"}
            
            self.logger.info(f"Executing Cursor command: {cmd.command}")
            
            # Simulate command execution
            result = {
                "success": True,
                "command": cmd.command,
                "description": cmd.description,
                "category": cmd.category,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute command {command}: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_command(self, command: str) -> Optional[CursorCommand]:
        """Find a Cursor command by name or description"""
        for cmd in self.cursor_commands:
            if (command.lower() in cmd.command.lower() or 
                command.lower() in cmd.description.lower()):
                return cmd
        return None
    
    def get_available_commands(self) -> List[IDECommand]:
        """Get available commands for Cursor"""
        commands = []
        for cmd in self.cursor_commands:
            commands.append(IDECommand(
                ide="cursor",
                command=cmd.command,
                description=cmd.description,
                category=cmd.category,
                shortcut=cmd.shortcut
            ))
        return commands
    
    def get_ai_suggestions(self, context: str, file_path: str = None) -> List[str]:
        """Get AI suggestions for the current context"""
        suggestions = []
        
        # Analyze context and provide suggestions
        if "error" in context.lower() or "exception" in context.lower():
            suggestions.extend([
                "Use 'cursor.fix' to automatically fix code issues",
                "Use 'cursor.explain' to understand the error",
                "Use 'cursor.debug' to get AI-powered debugging help"
            ])
        
        if "test" in context.lower():
            suggestions.extend([
                "Use 'cursor.test' to generate comprehensive tests",
                "Use 'cursor.explain' to understand test requirements"
            ])
        
        if "documentation" in context.lower() or "comment" in context.lower():
            suggestions.extend([
                "Use 'cursor.document' to generate documentation",
                "Use 'cursor.explain' to add inline comments"
            ])
        
        if "refactor" in context.lower() or "optimize" in context.lower():
            suggestions.extend([
                "Use 'cursor.refactor' to improve code structure",
                "Use 'cursor.generate' to create optimized versions"
            ])
        
        # Default suggestions
        if not suggestions:
            suggestions.extend([
                "Use 'cursor.generate' to create new code",
                "Use 'cursor.explain' to understand existing code",
                "Use 'cursor.chat.open' for general AI assistance"
            ])
        
        return suggestions
    
    def integrate_with_omni_agent(self, agent_task: str) -> Dict[str, Any]:
        """Integrate Omni-Dev Agent tasks with Cursor"""
        integration_result = {
            "task": agent_task,
            "cursor_commands": [],
            "suggestions": [],
            "workspace_analysis": self.workspace,
            "timestamp": datetime.now().isoformat()
        }
        
        # Map agent tasks to Cursor commands
        task_lower = agent_task.lower()
        
        if "free tier" in task_lower or "service" in task_lower:
            integration_result["cursor_commands"].append({
                "command": "cursor.chat.open",
                "description": "Open AI chat to discuss service integration",
                "shortcut": "Ctrl+L"
            })
            integration_result["suggestions"].append(
                "Use Cursor's AI chat to research and analyze cloud services"
            )
        
        if "code" in task_lower and "generate" in task_lower:
            integration_result["cursor_commands"].append({
                "command": "cursor.generate",
                "description": "Generate code based on requirements",
                "shortcut": "Ctrl+K"
            })
        
        if "test" in task_lower:
            integration_result["cursor_commands"].append({
                "command": "cursor.test",
                "description": "Generate tests for your code",
                "shortcut": "Ctrl+Shift+T"
            })
        
        if "document" in task_lower or "comment" in task_lower:
            integration_result["cursor_commands"].append({
                "command": "cursor.document",
                "description": "Generate documentation",
                "shortcut": "Ctrl+Shift+D"
            })
        
        if "debug" in task_lower or "fix" in task_lower:
            integration_result["cursor_commands"].extend([
                {
                    "command": "cursor.fix",
                    "description": "Fix code issues automatically",
                    "shortcut": "Ctrl+Shift+F"
                },
                {
                    "command": "cursor.debug",
                    "description": "Get AI-powered debugging help",
                    "shortcut": "Ctrl+Shift+B"
                }
            ])
        
        return integration_result
    
    def create_cursor_extension(self, extension_name: str, features: List[str]) -> str:
        """Create a Cursor extension for Omni-Dev Agent integration"""
        extension_dir = self.workspace_path / "cursor-extensions" / extension_name
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
        
        with open(extension_dir / "src" / "extension.ts", 'w') as f:
            f.write(extension_ts)
        
        # Create README
        readme_content = f'''
# Omni-Dev Agent {extension_name.title()} Extension

This Cursor extension provides integration with Omni-Dev Agent for {extension_name} functionality.

## Features

{chr(10).join(f"- {feature}" for feature in features)}

## Usage

1. Install the extension
2. Use Command Palette (Ctrl+Shift+P)
3. Run "Activate Omni-Dev Agent"

## Configuration

Add to your workspace settings:

```json
{{
    "omni-dev-agent.enabled": true,
    "omni-dev-agent.features": {json.dumps(features)}
}}
```
'''
        
        with open(extension_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Created Cursor extension: {extension_dir}")
        return str(extension_dir)
    
    def get_workspace_stats(self) -> Dict[str, Any]:
        """Get workspace statistics"""
        return {
            "workspace_name": self.workspace.name,
            "total_files": len(self.workspace.files),
            "file_extensions": self.workspace.extensions,
            "workspace_path": self.workspace.path,
            "cursor_installed": self.is_cursor_installed(),
            "available_commands": len(self.cursor_commands)
        } 