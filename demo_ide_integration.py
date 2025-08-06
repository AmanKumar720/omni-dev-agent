#!/usr/bin/env python3
"""
Demo script for Omni-Dev Agent IDE Integration
Shows how the agent can integrate with various IDEs including Cursor
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from components.ide_integration.ide_manager import IDEManager
from components.ide_integration.cursor_integration import CursorIntegration

def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🤖 Omni-Dev Agent - IDE Integration Demo")
    print("=" * 80)
    print("This demo shows how the agent can integrate with:")
    print("✅ Cursor IDE - AI-powered development")
    print("✅ Visual Studio Code - Popular code editor")
    print("✅ PyCharm - Python-focused IDE")
    print("✅ Any other IDE with custom integrations")
    print("=" * 80)

def demo_ide_detection():
    """Demo IDE detection capabilities"""
    print("\n🔍 DEMO 1: IDE Detection")
    print("-" * 40)
    
    ide_manager = IDEManager()
    available_ides = ide_manager.get_available_ides()
    
    print("Detecting available IDEs on your system...")
    
    for ide_name, ide_info in available_ides.items():
        status = "✅ Installed" if ide_info.installed else "❌ Not Installed"
        print(f"\n{ide_name.upper()}:")
        print(f"   Name: {ide_info.name}")
        print(f"   Status: {status}")
        print(f"   Version: {ide_info.version}")
        print(f"   Executable: {ide_info.executable}")
        print(f"   Workspace Support: {'✅ Yes' if ide_info.workspace_support else '❌ No'}")
        print(f"   Extension Support: {'✅ Yes' if ide_info.extension_support else '❌ No'}")
    
    installed_count = sum(1 for info in available_ides.values() if info.installed)
    print(f"\n📊 Summary: {installed_count}/{len(available_ides)} IDEs detected")

def demo_cursor_integration():
    """Demo Cursor-specific integration"""
    print("\n🎯 DEMO 2: Cursor IDE Integration")
    print("-" * 40)
    
    cursor_integration = CursorIntegration()
    
    # Check if Cursor is installed
    cursor_installed = cursor_integration.is_installed()
    print(f"Cursor Installed: {'✅ Yes' if cursor_installed else '❌ No'}")
    
    if cursor_installed:
        # Get workspace stats
        workspace_stats = cursor_integration.get_workspace_stats()
        print(f"\n📁 Workspace Analysis:")
        print(f"   Workspace Name: {workspace_stats['workspace_name']}")
        print(f"   Total Files: {workspace_stats['total_files']}")
        print(f"   File Extensions: {', '.join(workspace_stats['file_extensions'][:5])}...")
        print(f"   Available Commands: {workspace_stats['available_commands']}")
        
        # Show available Cursor commands
        print(f"\n🎮 Available Cursor Commands:")
        for cmd in cursor_integration.cursor_commands:
            shortcut_info = f" ({cmd.shortcut})" if cmd.shortcut else ""
            print(f"   {cmd.command}: {cmd.description}{shortcut_info}")
        
        # Demo AI suggestions
        print(f"\n🤖 AI Suggestions for different contexts:")
        
        contexts = [
            "I have an error in my code",
            "I need to write tests",
            "I want to add documentation",
            "I need to refactor this code"
        ]
        
        for context in contexts:
            suggestions = cursor_integration.get_ai_suggestions(context)
            print(f"\n   Context: '{context}'")
            for suggestion in suggestions[:2]:  # Show first 2 suggestions
                print(f"     💡 {suggestion}")

def demo_omni_agent_integration():
    """Demo integration with Omni-Dev Agent tasks"""
    print("\n🚀 DEMO 3: Omni-Dev Agent Task Integration")
    print("-" * 40)
    
    ide_manager = IDEManager()
    
    # Define different agent tasks
    agent_tasks = [
        "Research and analyze free tier cloud services",
        "Generate Python code for web API",
        "Debug and fix code issues",
        "Create comprehensive test suite",
        "Generate documentation for the project"
    ]
    
    for task in agent_tasks:
        print(f"\n📋 Task: {task}")
        
        # Get IDE integration suggestions
        integration_result = ide_manager.integrate_with_omni_agent(task)
        
        print(f"   Recommended IDE: {integration_result['recommended_ide'] or 'None'}")
        
        # Show IDE-specific suggestions
        for ide_name, suggestions in integration_result['ide_suggestions'].items():
            if ide_name in integration_result['available_ides'] and integration_result['available_ides'][ide_name].installed:
                print(f"   {ide_name.upper()} Suggestions:")
                if isinstance(suggestions, dict) and 'suggestions' in suggestions:
                    for suggestion in suggestions['suggestions'][:2]:
                        print(f"     💡 {suggestion}")
                elif isinstance(suggestions, dict) and 'cursor_commands' in suggestions:
                    for cmd in suggestions['cursor_commands'][:2]:
                        print(f"     🎮 {cmd['description']} ({cmd['shortcut']})")

def demo_extension_creation():
    """Demo IDE extension creation"""
    print("\n🔧 DEMO 4: IDE Extension Creation")
    print("-" * 40)
    
    ide_manager = IDEManager()
    
    # Create extensions for different features
    extensions_to_create = [
        {
            "name": "free-tier-analyzer",
            "features": ["Service Research", "Free Tier Analysis", "Cost Monitoring", "Integration Planning"]
        },
        {
            "name": "code-generator",
            "features": ["Code Generation", "Test Creation", "Documentation", "Refactoring"]
        }
    ]
    
    for extension in extensions_to_create:
        print(f"\n🔧 Creating {extension['name']} extension...")
        
        try:
            # Try to create Cursor extension
            if "cursor" in ide_manager.get_available_ides() and ide_manager.get_available_ides()["cursor"].installed:
                extension_path = ide_manager.create_ide_extension(
                    "cursor", 
                    extension["name"], 
                    extension["features"]
                )
                print(f"   ✅ Cursor extension created: {extension_path}")
            
            # Try to create VSCode extension
            if "vscode" in ide_manager.get_available_ides() and ide_manager.get_available_ides()["vscode"].installed:
                extension_path = ide_manager.create_ide_extension(
                    "vscode", 
                    extension["name"], 
                    extension["features"]
                )
                print(f"   ✅ VSCode extension created: {extension_path}")
                
        except Exception as e:
            print(f"   ❌ Failed to create extension: {e}")

def demo_workspace_operations():
    """Demo workspace operations"""
    print("\n📁 DEMO 5: Workspace Operations")
    print("-" * 40)
    
    ide_manager = IDEManager()
    
    # Get integration summary
    summary = ide_manager.get_integration_summary()
    
    print("📊 Integration Summary:")
    print(f"   Total IDEs Supported: {summary['total_ides']}")
    print(f"   Installed IDEs: {', '.join(summary['installed_ides']) if summary['installed_ides'] else 'None'}")
    print(f"   Workspace Path: {summary['workspace_path']}")
    
    print(f"\n🔧 Integration Features:")
    for feature in summary['integration_features']:
        print(f"   ✅ {feature}")
    
    # Demo opening workspace in different IDEs
    if summary['installed_ides']:
        print(f"\n🚀 Opening workspace in available IDEs:")
        for ide_name in summary['installed_ides']:
            success = ide_manager.open_in_ide(ide_name)
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {ide_name.upper()}: {status}")

def demo_custom_integration():
    """Demo custom IDE integration capabilities"""
    print("\n🎨 DEMO 6: Custom Integration Capabilities")
    print("-" * 40)
    
    print("The Omni-Dev Agent can integrate with ANY IDE through:")
    
    integration_methods = [
        "Command Line Interface (CLI) integration",
        "Extension/Plugin development",
        "API integration (if available)",
        "File system monitoring",
        "Workspace configuration management",
        "Custom command execution",
        "Project template generation",
        "Automated setup and configuration"
    ]
    
    for method in integration_methods:
        print(f"   ✅ {method}")
    
    print(f"\n🔧 Supported Integration Types:")
    integration_types = [
        "Direct IDE Integration (Cursor, VSCode, PyCharm)",
        "Extension Development (Custom plugins)",
        "CLI Integration (Command-line tools)",
        "API Integration (REST APIs, Webhooks)",
        "File-based Integration (Configuration files)",
        "Process Integration (Subprocess execution)"
    ]
    
    for integration_type in integration_types:
        print(f"   🎯 {integration_type}")

def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Run all demos
        demo_ide_detection()
        demo_cursor_integration()
        demo_omni_agent_integration()
        demo_extension_creation()
        demo_workspace_operations()
        demo_custom_integration()
        
        print("\n" + "=" * 80)
        print("🎉 IDE INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The Omni-Dev Agent can now integrate with:")
        print("✅ Cursor IDE - Full AI-powered integration")
        print("✅ Visual Studio Code - Extension and CLI support")
        print("✅ PyCharm - Workspace and project integration")
        print("✅ Any other IDE - Custom integration capabilities")
        print("✅ Multiple IDEs simultaneously")
        print("✅ Automatic IDE detection and configuration")
        print("✅ Custom extension creation for any IDE")
        print("=" * 80)
        
        print("\n🚀 Next Steps:")
        print("1. Install your preferred IDE")
        print("2. Run the agent to detect and configure integration")
        print("3. Use the agent's IDE commands for enhanced development")
        print("4. Create custom extensions for your specific needs")
        print("5. Enjoy seamless development with AI assistance!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 