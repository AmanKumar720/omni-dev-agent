#!/usr/bin/env python3
"""
Demo script for Omni-Dev Agent System Integration
Shows how the agent can integrate with any software on your operating system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from components.system_integration.system_manager import SystemManager
from components.system_integration.system_detector import SystemDetector

def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🤖 Omni-Dev Agent - System Integration Demo")
    print("=" * 80)
    print("This demo shows how the agent can integrate with:")
    print("✅ ANY software on your operating system")
    print("✅ Development tools and IDEs")
    print("✅ Cloud platforms and databases")
    print("✅ System utilities and applications")
    print("✅ Custom software and applications")
    print("=" * 80)

def demo_system_detection():
    """Demo system detection capabilities"""
    print("\n🔍 DEMO 1: System Detection")
    print("-" * 40)
    
    system_manager = SystemManager()
    system_info = system_manager.get_system_info()
    
    print("📊 System Information:")
    print(f"   Operating System: {system_info.os_name}")
    print(f"   OS Version: {system_info.os_version}")
    print(f"   Architecture: {system_info.architecture}")
    print(f"   Python Version: {system_info.python_version}")
    
    # Get installed software
    installed_software = system_manager.get_installed_software()
    
    print(f"\n📦 Installed Software ({len(installed_software)} found):")
    for software_id, software in installed_software.items():
        status = "✅ Installed" if software.installed else "❌ Not Installed"
        print(f"   {software.name}: {status}")
        if software.installed:
            print(f"     Version: {software.version}")
            print(f"     Category: {software.category}")
            print(f"     Path: {software.path}")

def demo_software_categories():
    """Demo software categorization"""
    print("\n📂 DEMO 2: Software Categories")
    print("-" * 40)
    
    system_manager = SystemManager()
    categories = system_manager.get_software_categories()
    
    print("📂 Software Categories:")
    for category in categories:
        software_in_category = system_manager.get_installed_software(category)
        print(f"\n   {category.upper()}:")
        for software_id, software in software_in_category.items():
            if software.installed:
                print(f"     ✅ {software.name} ({software.version})")

def demo_software_search():
    """Demo software search capabilities"""
    print("\n🔍 DEMO 3: Software Search")
    print("-" * 40)
    
    system_manager = SystemManager()
    
    # Search for different types of software
    search_queries = ["python", "git", "editor", "browser", "database"]
    
    for query in search_queries:
        print(f"\n🔍 Searching for '{query}':")
        results = system_manager.search_software(query)
        
        if results:
            for software in results:
                status = "✅ Installed" if software.installed else "❌ Not Installed"
                print(f"   {software.name}: {status}")
                if software.installed:
                    print(f"     Version: {software.version}")
                    print(f"     Category: {software.category}")
        else:
            print(f"   No software found matching '{query}'")

def demo_software_launching():
    """Demo software launching capabilities"""
    print("\n🚀 DEMO 4: Software Launching")
    print("-" * 40)
    
    system_manager = SystemManager()
    installed_software = system_manager.get_installed_software()
    
    # Try to launch some common software
    software_to_launch = ["notepad", "calc", "python"]
    
    for software_id in software_to_launch:
        if software_id in installed_software and installed_software[software_id].installed:
            print(f"\n🚀 Launching {installed_software[software_id].name}...")
            result = system_manager.launch_software(software_id)
            
            if result["success"]:
                print(f"   ✅ Successfully launched {result['software']}")
                print(f"   Command: {result['command']}")
            else:
                print(f"   ❌ Failed to launch: {result['error']}")
        else:
            print(f"\n❌ {software_id} not installed or not found")

def demo_command_execution():
    """Demo command execution with software"""
    print("\n⚡ DEMO 5: Command Execution")
    print("-" * 40)
    
    system_manager = SystemManager()
    installed_software = system_manager.get_installed_software()
    
    # Execute commands with different software
    commands_to_execute = [
        ("python", "--version", "Check Python version"),
        ("git", "--version", "Check Git version"),
        ("node", "--version", "Check Node.js version")
    ]
    
    for software_id, command, description in commands_to_execute:
        if software_id in installed_software and installed_software[software_id].installed:
            print(f"\n⚡ {description}...")
            result = system_manager.execute_software_command(software_id, command)
            
            if result["success"]:
                print(f"   ✅ Command executed successfully")
                print(f"   Output: {result['output'].strip()}")
            else:
                print(f"   ❌ Command failed: {result['error']}")
        else:
            print(f"\n❌ {software_id} not available for command execution")

def demo_integration_tasks():
    """Demo predefined integration tasks"""
    print("\n🎯 DEMO 6: Integration Tasks")
    print("-" * 40)
    
    system_manager = SystemManager()
    available_tasks = system_manager.get_available_tasks()
    
    print("📋 Available Integration Tasks:")
    for task_name, task in available_tasks.items():
        print(f"\n   {task.task_name}:")
        print(f"     Description: {task.description}")
        print(f"     Category: {task.category}")
        print(f"     Required Software: {', '.join(task.required_software)}")
        print(f"     Commands: {len(task.commands)} commands")
    
    # Execute a simple task
    print(f"\n🎯 Executing 'system_utilities' task...")
    result = system_manager.execute_integration_task("system_utilities")
    
    if result.success:
        print(f"   ✅ Task completed successfully")
        print(f"   Software used: {', '.join(result.software_used)}")
        print(f"   Commands executed: {len(result.commands_executed)}")
    else:
        print(f"   ❌ Task failed: {result.output.get('error', 'Unknown error')}")

def demo_omni_agent_integration():
    """Demo integration with Omni-Dev Agent tasks"""
    print("\n🤖 DEMO 7: Omni-Dev Agent Integration")
    print("-" * 40)
    
    system_manager = SystemManager()
    
    # Define different agent tasks
    agent_tasks = [
        "Set up a web development environment",
        "Configure database management tools",
        "Set up cloud development environment",
        "Create AI development workspace",
        "Launch system utilities for file editing"
    ]
    
    for task in agent_tasks:
        print(f"\n📋 Agent Task: {task}")
        
        # Get integration recommendations
        integration = system_manager.integrate_with_omni_agent(task)
        
        print(f"   Recommended Software: {', '.join(integration['recommended_software'])}")
        print(f"   Recommended Tasks: {', '.join(integration['recommended_tasks'])}")
        
        # Show available software
        available_software = integration['available_software']
        installed_count = sum(1 for sw in available_software.values() if sw.installed)
        print(f"   Available Software: {installed_count}/{len(available_software)} installed")

def demo_system_scan():
    """Demo comprehensive system scan"""
    print("\n🔍 DEMO 8: Comprehensive System Scan")
    print("-" * 40)
    
    system_manager = SystemManager()
    
    print("🔍 Running comprehensive system scan...")
    scan_result = system_manager.run_system_scan()
    
    print(f"\n📊 Scan Results:")
    print(f"   Total Software: {len(scan_result['installed_software'])}")
    print(f"   Categories: {len(scan_result['software_categories'])}")
    print(f"   Integration Capabilities: {len(scan_result['integration_capabilities'])}")
    
    print(f"\n📂 Software Categories:")
    for category, software_list in scan_result['software_categories'].items():
        print(f"   {category}: {len(software_list)} software")
        for software in software_list[:3]:  # Show first 3
            print(f"     - {software}")
        if len(software_list) > 3:
            print(f"     ... and {len(software_list) - 3} more")
    
    print(f"\n🚀 Integration Capabilities:")
    for capability in scan_result['integration_capabilities']:
        print(f"   ✅ {capability}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in scan_result['recommendations']:
        print(f"   💡 {recommendation}")

def demo_custom_integration():
    """Demo custom integration creation"""
    print("\n🎨 DEMO 9: Custom Integration Creation")
    print("-" * 40)
    
    system_manager = SystemManager()
    
    # Create a custom integration task
    custom_task_name = "custom_python_analysis"
    custom_description = "Analyze Python code and generate reports"
    custom_required_software = ["python", "git"]
    custom_commands = [
        "python --version",
        "git --version",
        "python -c 'import ast; print(\"Python AST module available\")'"
    ]
    
    print(f"🎨 Creating custom integration task: {custom_task_name}")
    success = system_manager.create_custom_integration(
        custom_task_name,
        custom_description,
        custom_required_software,
        custom_commands,
        "custom"
    )
    
    if success:
        print(f"   ✅ Custom integration created successfully")
        
        # Execute the custom task
        print(f"   🎯 Executing custom task...")
        result = system_manager.execute_integration_task(custom_task_name)
        
        if result.success:
            print(f"   ✅ Custom task completed successfully")
            print(f"   Software used: {', '.join(result.software_used)}")
        else:
            print(f"   ❌ Custom task failed: {result.output.get('error', 'Unknown error')}")
    else:
        print(f"   ❌ Failed to create custom integration")

def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Run all demos
        demo_system_detection()
        demo_software_categories()
        demo_software_search()
        demo_software_launching()
        demo_command_execution()
        demo_integration_tasks()
        demo_omni_agent_integration()
        demo_system_scan()
        demo_custom_integration()
        
        print("\n" + "=" * 80)
        print("🎉 SYSTEM INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The Omni-Dev Agent can now integrate with:")
        print("✅ ANY software on your operating system")
        print("✅ Development tools and IDEs")
        print("✅ Cloud platforms and databases")
        print("✅ System utilities and applications")
        print("✅ Custom software and applications")
        print("✅ Cross-platform compatibility")
        print("✅ Automated software detection")
        print("✅ Intelligent task orchestration")
        print("=" * 80)
        
        print("\n🚀 Key Capabilities:")
        print("1. **Universal Software Detection**: Finds any installed software")
        print("2. **Application Launching**: Launch any software automatically")
        print("3. **Command Execution**: Execute commands with any software")
        print("4. **Task Orchestration**: Coordinate multiple software tools")
        print("5. **Custom Integrations**: Create custom integration tasks")
        print("6. **System Scanning**: Comprehensive system analysis")
        print("7. **Intelligent Recommendations**: Suggest optimal software combinations")
        print("8. **Cross-Platform Support**: Works on Windows, macOS, Linux")
        
        print("\n🎯 Next Steps:")
        print("1. Install your preferred software")
        print("2. Run the agent to detect and configure integrations")
        print("3. Create custom integration tasks for your workflow")
        print("4. Automate complex multi-software workflows")
        print("5. Enjoy seamless system-wide automation!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 