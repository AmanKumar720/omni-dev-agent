# bootstrap.ps1 - Package orchestration runner for GitHub repository setup
# Creates Python virtualenv, installs dependencies, executes setup_repo.py, and outputs summary

param(
    [string]$ManifestFile = "repos.yaml"
)

# Configuration
$VenvDir = ".venv-setup"
$LogFile = "bootstrap.log"
$SummaryFile = "setup_summary.txt"

# Color functions
function Write-Info($message) {
    Write-Host "[INFO] $message" -ForegroundColor Green
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [INFO] $message"
}

function Write-Error-Custom($message) {
    Write-Host "[ERROR] $message" -ForegroundColor Red
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [ERROR] $message"
}

function Write-Debug-Custom($message) {
    Write-Host "[DEBUG] $message" -ForegroundColor Blue
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [DEBUG] $message"
}

# Banner
function Print-Banner {
    Write-Host "======================================================" -ForegroundColor Blue
    Write-Host "    GitHub Repository Setup - Bootstrap Script" -ForegroundColor Blue
    Write-Host "======================================================" -ForegroundColor Blue
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Python is not installed or not in PATH"
            exit 1
        }
        Write-Info "Python version: $pythonVersion ✓"
    }
    catch {
        Write-Error-Custom "Python is not available"
        exit 1
    }
    
    # Check pip
    try {
        python -m pip --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "pip is not available"
            exit 1
        }
    }
    catch {
        Write-Error-Custom "pip is not available"
        exit 1
    }
    
    # Check manifest file
    if (-not (Test-Path $ManifestFile)) {
        Write-Error-Custom "Manifest file '$ManifestFile' not found"
        Write-Info "Usage: .\bootstrap.ps1 [manifest_file.yaml]"
        exit 1
    }
    
    # Check GITHUB_TOKEN
    if (-not $env:GITHUB_TOKEN) {
        Write-Error-Custom "GITHUB_TOKEN environment variable is not set"
        Write-Info "Please set GITHUB_TOKEN with a valid GitHub personal access token"
        exit 1
    }
    
    Write-Info "Prerequisites check completed ✓"
}

# Setup virtual environment
function Setup-VirtualEnv {
    Write-Info "Setting up Python virtual environment..."
    
    # Remove existing virtual environment
    if (Test-Path $VenvDir) {
        Write-Debug-Custom "Removing existing virtual environment"
        Remove-Item -Recurse -Force $VenvDir
    }
    
    # Create new virtual environment
    Write-Debug-Custom "Creating virtual environment in $VenvDir"
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to create virtual environment"
        exit 1
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    } else {
        Write-Error-Custom "Failed to find virtual environment activation script"
        exit 1
    }
    
    # Upgrade pip
    Write-Debug-Custom "Upgrading pip"
    python -m pip install --upgrade pip
    
    Write-Info "Virtual environment created and activated ✓"
}

# Install dependencies
function Install-Dependencies {
    Write-Info "Installing dependencies..."
    
    # Install setup dependencies
    if (Test-Path "requirements-setup.txt") {
        Write-Debug-Custom "Installing setup dependencies from requirements-setup.txt"
        python -m pip install -r requirements-setup.txt
    } else {
        Write-Debug-Custom "Installing setup dependencies individually"
        python -m pip install "PyGithub>=1.59.0" "PyYAML>=6.0" "requests>=2.31.0" "PyNaCl>=1.5.0"
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to install dependencies"
        exit 1
    }
    
    # Verify packages
    Write-Debug-Custom "Verifying installed packages"
    python -c "import github, yaml, requests, nacl; print('All dependencies verified')"
    
    Write-Info "Dependencies installed successfully ✓"
}

# Validate manifest
function Test-Manifest {
    Write-Info "Validating manifest file: $ManifestFile"
    
    $validationScript = @"
import yaml
import sys
try:
    with open('$ManifestFile', 'r') as f:
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
    
    print(f'Manifest validation successful: {len(config["repositories"])} repositories found')

except Exception as e:
    print(f'Manifest validation failed: {e}', file=sys.stderr)
    sys.exit(1)
"@
    
    python -c $validationScript
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Manifest validation failed"
        exit 1
    }
    
    Write-Info "Manifest file validation completed ✓"
}

# Execute setup
function Invoke-Setup {
    Write-Info "Executing repository setup..."
    
    $startTime = Get-Date
    
    Write-Debug-Custom "Running: python setup_repo.py $ManifestFile"
    python setup_repo.py $ManifestFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Repository setup failed"
        exit 1
    }
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Info "Repository setup completed in ${duration}s ✓"
}

# Generate summary
function New-Summary {
    Write-Info "Generating summary table..."
    
$summaryScript = "@
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
        manifest_file = '$ManifestFile'
        manifest = load_manifest(manifest_file)
        generate_summary_table(manifest, manifest_file)
    except Exception as e:
        print(f"Error generating summary: {e}", file=sys.stderr)
        sys.exit(1)
"@
    
    # Display summary
    python -c $summaryScript
    
    # Save summary to file
    python -c $summaryScript > $SummaryFile
    
    Write-Info "Summary table generated and saved to $SummaryFile ✓"
}

# Cleanup
function Remove-VirtualEnv {
    if (Test-Path $VenvDir) {
        Write-Debug-Custom "Cleaning up virtual environment"
        Remove-Item -Recurse -Force $VenvDir
    }
}

# Main function
function Main {
    Print-Banner
    
    # Initialize log file
    "Bootstrap script started at $(Get-Date)" | Out-File -FilePath $LogFile -Encoding UTF8
    
    Write-Info "Starting GitHub repository setup bootstrap"
    Write-Info "Manifest file: $ManifestFile"
    Write-Info "Log file: $LogFile"
    
try {
    Test-Prerequisites
    Test-Manifest
    Setup-VirtualEnv
    Install-Dependencies
    Invoke-Setup
    New-Summary
}
        
        Write-Info "Bootstrap process completed successfully!"
        Write-Info "Check $LogFile for detailed logs"
        Write-Info "Check $SummaryFile for setup summary"
        
        Write-Host "======================================================" -ForegroundColor Green
        Write-Host "           BOOTSTRAP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
        Write-Host "======================================================" -ForegroundColor Green
    }
    catch {
        Write-Error-Custom "Bootstrap process failed: $_"
        exit 1
    }
    finally {
        Remove-VirtualEnv
    }
}

# Entry point
Main
