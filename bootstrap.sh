#!/bin/bash

# bootstrap.sh - Package orchestration runner for GitHub repository setup
# Creates Python virtualenv, installs dependencies, executes setup_repo.py, and outputs summary

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly VENV_DIR=".venv-setup"
readonly MANIFEST_FILE="${1:-repos.yaml}"
readonly LOG_FILE="bootstrap.log"
readonly SUMMARY_FILE="setup_summary.txt"
readonly ENV_FILE=".github-config.env"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        INFO)  echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE" ;;
        DEBUG) echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE" ;;
    esac
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Error handler
error_exit() {
    local line_number="$1"
    local error_code="$2"
    log ERROR "Script failed at line $line_number with exit code $error_code"
    log ERROR "Check $LOG_FILE for detailed error information"
    exit "$error_code"
}

# Set error trap
trap 'error_exit $LINENO $?' ERR

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "======================================================"
    echo "    GitHub Repository Setup - Bootstrap Script"
    echo "======================================================"
    echo -e "${NC}"
}

# Detect OS and set appropriate commands
detect_os() {
    log INFO "Detecting operating system..."
    
    case "$(uname -s)" in
        Linux*)     OS="linux";;
        Darwin*)    OS="macos";;
        CYGWIN*|MINGW*|MSYS*) OS="windows";;
        *)          OS="unknown";;
    esac
    
    log INFO "Detected OS: $OS"
    
    # Set Python command based on OS
    if [ "$OS" = "windows" ]; then
        # On Windows, prefer 'python' over 'python3'
        if command -v python &> /dev/null; then
            PYTHON_CMD="python"
        elif command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            log ERROR "Python is not installed or not in PATH"
            exit 1
        fi
    else
        # On Unix-like systems, prefer 'python3' over 'python'
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            log ERROR "Python is not installed or not in PATH"
            exit 1
        fi
    fi
    
    log INFO "Using Python command: $PYTHON_CMD"
}

# Check prerequisites
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    # Check Python version (3.7+)
    local python_version_output=$($PYTHON_CMD --version 2>&1)
    local python_version=$(echo "$python_version_output" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    if [ -z "$python_version" ]; then
        log ERROR "Could not determine Python version from: $python_version_output"
        exit 1
    fi
    
    local major_version=$(echo "$python_version" | cut -d'.' -f1)
    local minor_version=$(echo "$python_version" | cut -d'.' -f2)
    
    if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 7 ]); then
        log ERROR "Python 3.7 or higher is required (found: $python_version)"
        exit 1
    fi
    
    log INFO "Python version: $python_version ✓"
    
    # Check if pip is available
    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        log ERROR "pip is not available"
        exit 1
    fi
    
    # Check if virtualenv is available
    if ! $PYTHON_CMD -m venv --help &> /dev/null; then
        log ERROR "Python venv module is not available"
        exit 1
    fi
    
    # Check if manifest file exists
    if [ ! -f "$MANIFEST_FILE" ]; then
        log ERROR "Manifest file '$MANIFEST_FILE' not found"
        log INFO "Usage: $0 [manifest_file.yaml]"
        exit 1
    fi
    
    log INFO "Prerequisites check completed ✓"
}

# Setup GitHub token
setup_github_token() {
    log INFO "Setting up GitHub token..."
    
    # Check if GITHUB_TOKEN is already set
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        log INFO "GITHUB_TOKEN environment variable is already set ✓"
        return 0
    fi
    
    # Check if .github-config.env exists and load it
    if [ -f "$ENV_FILE" ]; then
        log INFO "Loading environment variables from $ENV_FILE"
        # shellcheck source=/dev/null
        source "$ENV_FILE"
    fi
    
    # Check if GITHUB_TOKEN is now available
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        log INFO "GITHUB_TOKEN loaded from $ENV_FILE ✓"
        return 0
    fi
    
    # Prompt user for GitHub token
    log WARN "GITHUB_TOKEN not found in environment or $ENV_FILE"
    echo -e "${YELLOW}Please enter your GitHub Personal Access Token:${NC}"
    echo -e "${BLUE}Required scopes: repo, workflow, admin:org${NC}"
    echo -e "${BLUE}You can create one at: https://github.com/settings/tokens${NC}"
    echo ""
    read -r -s GITHUB_TOKEN
    
    if [ -z "$GITHUB_TOKEN" ]; then
        log ERROR "GitHub token is required to proceed"
        exit 1
    fi
    
    # Validate token format (basic check)
    if [[ ! "$GITHUB_TOKEN" =~ ^ghp_[A-Za-z0-9]{36}$ ]] && [[ ! "$GITHUB_TOKEN" =~ ^github_pat_[A-Za-z0-9_]{82}$ ]]; then
        log WARN "Token format doesn't match expected GitHub token patterns"
        log WARN "Continuing anyway, but token validation may fail"
    fi
    
    export GITHUB_TOKEN
    log INFO "GitHub token set successfully ✓"
}

# Create and setup virtual environment
setup_virtualenv() {
    log INFO "Setting up Python virtual environment..."
    
    # Remove existing virtual environment if it exists
    if [ -d "$VENV_DIR" ]; then
        log DEBUG "Removing existing virtual environment"
        rm -rf "$VENV_DIR"
    fi
    
    # Create new virtual environment
    log DEBUG "Creating virtual environment in $VENV_DIR"
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    # Activate virtual environment based on OS
    if [ "$OS" = "windows" ]; then
        # shellcheck source=/dev/null
        source "$VENV_DIR/Scripts/activate"
    else
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
    fi
    
    # Upgrade pip
    log DEBUG "Upgrading pip"
    python -m pip install --upgrade pip
    
    log INFO "Virtual environment created and activated ✓"
}

# Install dependencies
install_dependencies() {
    log INFO "Installing dependencies..."
    
    # Install setup dependencies
    if [ -f "requirements-setup.txt" ]; then
        log DEBUG "Installing setup dependencies from requirements-setup.txt"
        python -m pip install -r requirements-setup.txt
    else
        log DEBUG "Installing setup dependencies individually"
        python -m pip install "PyGithub>=1.59.0" "PyYAML>=6.0" "requests>=2.31.0" "PyNaCl>=1.5.0"
    fi
    
    # Verify key packages are installed
    log DEBUG "Verifying installed packages"
    python -c "import github, yaml, requests, nacl; print('All dependencies verified')"
    
    log INFO "Dependencies installed successfully ✓"
}

# Validate manifest file
validate_manifest() {
    log INFO "Validating manifest file: $MANIFEST_FILE"
    
    # Basic YAML validation
    python -c "
import yaml
import sys
try:
    with open('$MANIFEST_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError('Manifest must be a dictionary')
    
    if 'repositories' not in config:
        raise ValueError('Manifest must contain a repositories key')
    
    if not isinstance(config['repositories'], list):
        raise ValueError('repositories must be a list')
    
    if len(config['repositories']) == 0:
        raise ValueError('repositories list cannot be empty')
    
    for i, repo in enumerate(config['repositories']):
        if not isinstance(repo, dict):
            raise ValueError(f'Repository {i} must be a dictionary')
        if 'name' not in repo:
            raise ValueError(f'Repository {i} must have a name')
    
    print(f'Manifest validation successful: {len(config[\"repositories\"])} repositories found')

except Exception as e:
    print(f'Manifest validation failed: {e}', file=sys.stderr)
    sys.exit(1)
    "
    
    log INFO "Manifest file validation completed ✓"
}

# Execute setup_repo.py
execute_setup() {
    log INFO "Executing repository setup..."
    
    # Record start time
    local start_time=$(date +%s)
    
    # Execute setup_repo.py with manifest file
    log DEBUG "Running: python setup_repo.py $MANIFEST_FILE"
    python setup_repo.py "$MANIFEST_FILE" 2>&1 | tee -a "$LOG_FILE"
    
    # Record end time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log INFO "Repository setup completed in ${duration}s ✓"
}

# Generate summary table
generate_summary() {
    log INFO "Generating summary table..."
    
    # Create summary using Python
    python << 'EOF'
import yaml
import sys
import os
from datetime import datetime

def load_manifest(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def generate_summary_table(manifest, manifest_file):
    repositories = manifest.get('repositories', [])
    
    # Header
    print("\n" + "="*80)
    print(" " * 20 + "REPOSITORY SETUP SUMMARY")
    print("="*80)
    print(f"Manifest File: {manifest_file}")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Repositories: {len(repositories)}")
    print("="*80)
    
    # Table header
    print(f"{'Repository':<25} {'Visibility':<12} {'Topics':<15} {'Protections':<25}")
    print("-" * 80)
    
    for repo in repositories:
        name = repo.get('name', 'Unknown')[:24]
        visibility = repo.get('visibility', 'private')[:11]
        topics = str(len(repo.get('topics', [])))
        
        # Count protections
        protections = []
        if repo.get('secrets_and_environments', {}).get('repository_secrets'):
            protections.append("Repo Secrets")
        if repo.get('secrets_and_environments', {}).get('organization_secrets'):
            protections.append("Org Secrets")
        if repo.get('secrets_and_environments', {}).get('environments'):
            env_count = len(repo['secrets_and_environments']['environments'])
            protections.append(f"{env_count} Env(s)")
        
        protections_str = ", ".join(protections)[:24] if protections else "None"
        
        print(f"{name:<25} {visibility:<12} {topics:<15} {protections_str:<25}")
    
    print("-" * 80)
    
    # Detailed protection summary
    print("\nDETAILED PROTECTION SUMMARY")
    print("-" * 40)
    
    total_repo_secrets = 0
    total_org_secrets = 0
    total_environments = 0
    
    for repo in repositories:
        secrets_config = repo.get('secrets_and_environments', {})
        
        repo_secrets = len(secrets_config.get('repository_secrets', {}))
        org_secrets = len(secrets_config.get('organization_secrets', {}))
        environments = len(secrets_config.get('environments', {}))
        
        total_repo_secrets += repo_secrets
        total_org_secrets += org_secrets
        total_environments += environments
        
        if repo_secrets > 0 or org_secrets > 0 or environments > 0:
            print(f"\n{repo['name']}:")
            if repo_secrets > 0:
                print(f"  - Repository Secrets: {repo_secrets}")
            if org_secrets > 0:
                print(f"  - Organization Secrets: {org_secrets}")
            if environments > 0:
                print(f"  - Protected Environments: {environments}")
                for env_name, env_config in secrets_config['environments'].items():
                    reviewers = len(env_config.get('reviewers', []))
                    wait_time = env_config.get('wait_timer', 0)
                    env_secrets = len(env_config.get('secrets', {}))
                    print(f"    * {env_name}: {reviewers} reviewers, {wait_time}s wait, {env_secrets} secrets")
    
    print(f"\nTOTAL CONFIGURED PROTECTIONS:")
    print(f"- Repository Secrets: {total_repo_secrets}")
    print(f"- Organization Secrets: {total_org_secrets}")
    print(f"- Protected Environments: {total_environments}")
    print("="*80)

if __name__ == "__main__":
    try:
        manifest_file = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('MANIFEST_FILE', 'repos.yaml')
        manifest = load_manifest(manifest_file)
        generate_summary_table(manifest, manifest_file)
    except Exception as e:
        print(f"Error generating summary: {e}", file=sys.stderr)
        sys.exit(1)
EOF

    # Save summary to file as well
    python << 'EOF' > "$SUMMARY_FILE"
import yaml
import sys
import os
from datetime import datetime

def load_manifest(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def generate_summary_table(manifest, manifest_file):
    repositories = manifest.get('repositories', [])
    
    # Header
    print("\n" + "="*80)
    print(" " * 20 + "REPOSITORY SETUP SUMMARY")
    print("="*80)
    print(f"Manifest File: {manifest_file}")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Repositories: {len(repositories)}")
    print("="*80)
    
    # Table header
    print(f"{'Repository':<25} {'Visibility':<12} {'Topics':<15} {'Protections':<25}")
    print("-" * 80)
    
    for repo in repositories:
        name = repo.get('name', 'Unknown')[:24]
        visibility = repo.get('visibility', 'private')[:11]
        topics = str(len(repo.get('topics', [])))
        
        # Count protections
        protections = []
        if repo.get('secrets_and_environments', {}).get('repository_secrets'):
            protections.append("Repo Secrets")
        if repo.get('secrets_and_environments', {}).get('organization_secrets'):
            protections.append("Org Secrets")
        if repo.get('secrets_and_environments', {}).get('environments'):
            env_count = len(repo['secrets_and_environments']['environments'])
            protections.append(f"{env_count} Env(s)")
        
        protections_str = ", ".join(protections)[:24] if protections else "None"
        
        print(f"{name:<25} {visibility:<12} {topics:<15} {protections_str:<25}")
    
    print("-" * 80)

if __name__ == "__main__":
    try:
        manifest_file = os.environ.get('MANIFEST_FILE', 'repos.yaml')
        manifest = load_manifest(manifest_file)
        generate_summary_table(manifest, manifest_file)
    except Exception as e:
        print(f"Error generating summary: {e}", file=sys.stderr)
        sys.exit(1)
EOF

    log INFO "Summary table generated and saved to $SUMMARY_FILE ✓"
}

# Cleanup function
cleanup() {
    if [ -d "$VENV_DIR" ]; then
        log DEBUG "Cleaning up virtual environment"
        rm -rf "$VENV_DIR"
    fi
}

# Main execution function
main() {
    print_banner
    
    # Initialize log file
    echo "Bootstrap script started at $(date)" > "$LOG_FILE"
    
    log INFO "Starting GitHub repository setup bootstrap"
    log INFO "Manifest file: $MANIFEST_FILE"
    log INFO "Log file: $LOG_FILE"
    
    # Execute all steps
    detect_os
    check_prerequisites
    setup_github_token
    validate_manifest
    setup_virtualenv
    install_dependencies
    
    # Set environment variable for summary generation
    export MANIFEST_FILE="$MANIFEST_FILE"
    
    execute_setup
    generate_summary
    
    log INFO "Bootstrap process completed successfully!"
    log INFO "Check $LOG_FILE for detailed logs"
    log INFO "Check $SUMMARY_FILE for setup summary"
    
    # Cleanup
    cleanup
    
    echo -e "${GREEN}"
    echo "======================================================"
    echo "           BOOTSTRAP COMPLETED SUCCESSFULLY!"
    echo "======================================================"
    echo -e "${NC}"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
