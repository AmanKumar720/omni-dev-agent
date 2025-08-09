
# src/components/component_manager/component_manager.py

from ..documentation_analyzer.analyzer import analyze_documentation
from ..code_integrator.integrator import integrate_component_code
from .package_manager import PackageManager
from ..project_analyzer.requirements_analyzer import analyze_requirements_document
from ..dependency_monitor.monitor import check_for_new_dependency_versions, check_for_security_vulnerabilities
from ..component_registry.registry import get_component_metadata
import importlib
import logging

package_manager = PackageManager()
logger = logging.getLogger(__name__)

# Registry of loaded components for hot-plug support
LOADED_COMPONENTS = {}

def load_component_dynamically(component_name: str):
    """
    Dynamically load a component using hot-plug loading.
    
    Args:
        component_name (str): The name of the component to load
        
    Returns:
        The loaded component class or None if loading failed
    """
    try:
        component_metadata = get_component_metadata(component_name)
        
        if not component_metadata:
            logger.error(f"Component {component_name} not found in registry")
            return None
        
        # Check if component supports hot-plugging
        if not component_metadata.get('hot_pluggable', False):
            logger.warning(f"Component {component_name} does not support hot-plug loading")
            return None
        
        # Check if already loaded
        if component_name in LOADED_COMPONENTS:
            logger.info(f"Component {component_name} already loaded")
            return LOADED_COMPONENTS[component_name]
        
        # Get component path and main class
        component_path = component_metadata.get('component_path')
        main_class = component_metadata.get('main_class')
        
        if not component_path or not main_class:
            logger.error(f"Component {component_name} missing path or main_class in metadata")
            return None
        
        # Dynamically import the component
        logger.info(f"Loading component {component_name} from {component_path}")
        module = importlib.import_module(component_path)
        
        # Get the main class from the module
        component_class = getattr(module, main_class, None)
        
        if not component_class:
            logger.error(f"Main class {main_class} not found in {component_path}")
            return None
        
        # Cache the loaded component
        LOADED_COMPONENTS[component_name] = component_class
        
        logger.info(f"Successfully loaded component {component_name}")
        return component_class
        
    except ImportError as e:
        logger.error(f"Failed to import component {component_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading component {component_name}: {str(e)}")
        return None

def unload_component(component_name: str) -> bool:
    """
    Unload a dynamically loaded component.
    
    Args:
        component_name (str): Name of the component to unload
        
    Returns:
        bool: True if unloaded successfully, False otherwise
    """
    try:
        if component_name in LOADED_COMPONENTS:
            del LOADED_COMPONENTS[component_name]
            logger.info(f"Component {component_name} unloaded successfully")
            return True
        else:
            logger.warning(f"Component {component_name} not currently loaded")
            return False
    except Exception as e:
        logger.error(f"Error unloading component {component_name}: {str(e)}")
        return False

def list_loaded_components() -> list:
    """
    List all currently loaded components.
    
    Returns:
        list: List of loaded component names
    """
    return list(LOADED_COMPONENTS.keys())

def reload_component(component_name: str):
    """
    Reload a component (unload and load again).
    
    Args:
        component_name (str): Name of the component to reload
        
    Returns:
        The reloaded component class or None if failed
    """
    logger.info(f"Reloading component {component_name}")
    
    # Unload first
    unload_component(component_name)
    
    # Force module reload by removing from sys.modules if present
    component_metadata = get_component_metadata(component_name)
    if component_metadata:
        component_path = component_metadata.get('component_path')
        if component_path:
            import sys
            if component_path in sys.modules:
                del sys.modules[component_path]
                logger.debug(f"Removed {component_path} from sys.modules")
    
    # Load again
    return load_component_dynamically(component_name)

def create_component_instance(component_name: str, *args, **kwargs):
    """
    Create an instance of a dynamically loaded component.
    
    Args:
        component_name (str): Name of the component
        *args: Positional arguments for component constructor
        **kwargs: Keyword arguments for component constructor
        
    Returns:
        Component instance or None if failed
    """
    try:
        component_class = load_component_dynamically(component_name)
        
        if not component_class:
            logger.error(f"Failed to load component class for {component_name}")
            return None
        
        # Create instance
        instance = component_class(*args, **kwargs)
        logger.info(f"Created instance of {component_name}")
        
        return instance
        
    except Exception as e:
        logger.error(f"Failed to create instance of {component_name}: {str(e)}")
        return None

def request_component_integration(component_name: str, functionality_description: str, documentation_source: str = None):
    """
    Simulates a user requesting the integration of a new component.
    In a real scenario, this would trigger the agent's component identification,
    documentation analysis, and implementation process.

    Args:
        component_name (str): The name of the component to integrate (e.g., 'Supabase', 'Redis').
        functionality_description (str): A brief description of the functionality
                                         this component is expected to provide.
        documentation_source (str, optional): A URL or file path to the documentation.
                                            If None, the agent would attempt to search.
    """
    print(f"User requested integration of component: {component_name}")
    print(f"Expected functionality: {functionality_description}")
    print("\nInitiating component identification and analysis...")

    component_metadata = get_component_metadata(component_name)
    if component_metadata:
        print(f"Found component metadata for {component_name}: {component_metadata.get('description')}")
        # If documentation_source is not provided, try to get it from metadata
        if not documentation_source and "documentation_url" in component_metadata:
            documentation_source = component_metadata["documentation_url"]
            print(f"Using documentation URL from registry: {documentation_source}")
    else:
        print(f"No metadata found for {component_name} in the registry.")

    doc_content = None
    if documentation_source:
        if documentation_source.startswith("http"): # Assume URL
            print(f"Attempting to fetch documentation from URL: {documentation_source}")
            try:
                fetch_result = default_api.web_fetch(prompt=f"Get content from {documentation_source}")
                if fetch_result and 'output' in fetch_result:
                    doc_content = fetch_result['output']
                    print("Documentation fetched successfully (first 200 chars):", doc_content[:200])
                else:
                    print("Failed to fetch documentation from URL.")
            except Exception as e:
                print(f"Error fetching documentation: {e}")
        else: # Assume local file path
            print(f"Attempting to read documentation from local file: {documentation_source}")
            try:
                read_result = default_api.read_file(absolute_path=documentation_source)
                if read_result and 'content' in read_result:
                    doc_content = read_result['content']
                    print("Documentation read successfully (first 200 chars):", doc_content[:200])
                else:
                    print("Failed to read documentation from local file.")
            except Exception as e:
                print(f"Error reading documentation: {e}")
    else:
        print("No documentation source provided. Analysis will be limited.")

    analysis_result = analyze_documentation(component_name, doc_content)
    print("\nDocumentation Analysis Result:", analysis_result)

    # Now, proceed to code integration based on the analysis
    integration_status = integrate_component_code(component_name, analysis_result)
    print("\nCode Integration Status:", integration_status)

    # Simulate package installation using the PackageManager
    install_result = package_manager.install_component(component_name)
    print("Package Installation Status:", install_result)




def analyze_project_requirements(document_content: str):
    """
    Analyzes a project requirements document to identify potential component needs
    and triggers their integration process.

    Args:
        document_content (str): The content of the requirements document.
    """
    print("\n--- Analyzing Project Requirements ---")
    analysis_result = analyze_requirements_document(document_content)
    print("\nProject Requirements Analysis Summary:", analysis_result["analysis_summary"])

    identified_needs = analysis_result.get("identified_component_needs", [])
    if identified_needs:
        print("Identified component needs from requirements:")
        for need in identified_needs:
            print(f"  - Component: {need['name']}, Rationale: {need['rationale']}")
            # Trigger the component integration process for each identified need
            request_component_integration(need['name'], need['rationale'])
    else:
        print("No specific component needs identified from the requirements document.")

def monitor_external_events(requirements_file_path: str):
    """
    Monitors external events like new dependency versions or security vulnerabilities
    and triggers component integration/modification if needed.

    Args:
        requirements_file_path (str): Path to the project's requirements.txt file.
    """
    print("\n--- Monitoring External Events (Dependencies) ---")
    dependencies = {}
    try:
        with open(requirements_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Basic parsing for 'package>=version' or 'package==version'
                    if '>=' in line:
                        name, version = line.split('>=', 1)
                    elif '==' in line:
                        name, version = line.split('==', 1)
                    else:
                        name = line
                        version = "unknown" # Or fetch latest if possible
                    dependencies[name.strip()] = version.strip()
    except FileNotFoundError:
        print(f"Error: requirements.txt not found at {requirements_file_path}")
        return

    print("Current Dependencies:", dependencies)

    # Check for new versions
    new_versions = check_for_new_dependency_versions(dependencies)
    for dep_info in new_versions:
        component_name = dep_info['name']
        old_version = dep_info['current_version']
        new_version = dep_info['new_version']
        rationale = f"New version {new_version} available for {component_name} (from {old_version})."
        print(f"\nDetected: {rationale}")
        request_component_integration(
            component_name=f"{component_name} Upgrade",
            functionality_description=rationale,
            documentation_content=f"Search for {component_name} {new_version} release notes"
        )

    # Check for security vulnerabilities
    vulnerabilities = check_for_security_vulnerabilities(dependencies)
    for dep_info in vulnerabilities:
        component_name = dep_info['name']
        version = dep_info['version']
        vuln_id = dep_info['vulnerability_id']
        rationale = f"Security vulnerability {vuln_id} found in {component_name}@{version}. Requires patching/replacement."
        print(f"\nDetected: {rationale}")
        request_component_integration(
            component_name=f"{component_name} Security Fix",
            functionality_description=rationale,
            documentation_content=f"Search for {component_name} {vuln_id} patch/fix"
        )

    if not new_versions and not vulnerabilities:
        print("No new dependency versions or security vulnerabilities detected.")




if __name__ == "__main__":
    # Example 1: User-initiated component request
    request_component_integration(
        component_name="Supabase",
        functionality_description="Provide a backend database, authentication, and real-time capabilities.",
        documentation_content="Example Supabase documentation content."
    )

    print("\n=====================================")
    print("\n=====================================")

    # Example 2: Agent-initiated component identification from requirements document
    mock_srs_content = """
    1. Functional Requirements:
        1.1. Users shall be able to register and log in.
        1.2. User profiles shall be stored securely (data storage).
        1.3. The system shall provide real-time updates on user activity.
        1.4. Background tasks will process user data asynchronously.
    2. Non-Functional Requirements:
        2.1. The system must be highly available.
    """
    analyze_project_requirements(mock_srs_content)

    print("\n=====================================")
    print("\n=====================================")

    # Example 3: Agent-initiated component identification from external events
    monitor_external_events("c:/Users/Aman kumar/omni-dev-agent/requirements.txt")





