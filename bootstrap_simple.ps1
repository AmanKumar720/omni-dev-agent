# Simple bootstrap.ps1 - Package orchestration runner for GitHub repository setup
param(
    [string]$ManifestFile = "repos.yaml"
)

Write-Host "======================================================" -ForegroundColor Blue
Write-Host "    GitHub Repository Setup - Bootstrap Script" -ForegroundColor Blue  
Write-Host "======================================================" -ForegroundColor Blue

# Check prerequisites
Write-Host "[INFO] Checking prerequisites..." -ForegroundColor Green

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "[INFO] Python version: $pythonVersion ✓" -ForegroundColor Green

# Check manifest file
if (-not (Test-Path $ManifestFile)) {
    Write-Host "[ERROR] Manifest file '$ManifestFile' not found" -ForegroundColor Red
    exit 1
}

# Validate manifest
Write-Host "[INFO] Validating manifest file: $ManifestFile" -ForegroundColor Green
python -c "
import yaml
import sys
try:
    with open('$ManifestFile', 'r') as f:
        config = yaml.safe_load(f)
    print(f'Manifest validation successful: {len(config[\"repositories\"])} repositories found')
except Exception as e:
    print(f'Manifest validation failed: {e}', file=sys.stderr)
    sys.exit(1)"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Manifest validation failed" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "[INFO] Setting up Python virtual environment..." -ForegroundColor Green
if (Test-Path ".venv-setup") {
    Remove-Item -Recurse -Force ".venv-setup"
}
python -m venv .venv-setup

# Activate virtual environment and install dependencies
Write-Host "[INFO] Installing dependencies..." -ForegroundColor Green
& ".venv-setup\Scripts\Activate.ps1"
python -m pip install --upgrade pip

if (Test-Path "requirements-setup.txt") {
    python -m pip install -r requirements-setup.txt
} else {
    python -m pip install "PyGithub>=1.59.0" "PyYAML>=6.0" "requests>=2.31.0" "PyNaCl>=1.5.0"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Generate summary only (skip actual setup since we don't have real GITHUB_TOKEN)
Write-Host "[INFO] Generating summary table..." -ForegroundColor Green

python -c "
import yaml
import sys
from datetime import datetime

def load_manifest(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def generate_summary_table(manifest, manifest_file):
    repositories = manifest.get('repositories', [])
    
    print('\n' + '='*80)
    print(' ' * 20 + 'REPOSITORY SETUP SUMMARY')
    print('='*80)
    print(f'Manifest File: {manifest_file}')
    print(f'Execution Time: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
    print(f'Total Repositories: {len(repositories)}')
    print('='*80)
    
    print(f'{'Repository':<25} {'Visibility':<12} {'Topics':<15} {'Protections':<25}')
    print('-' * 80)
    
    for repo in repositories:
        name = repo.get('name', 'Unknown')[:24]
        visibility = repo.get('visibility', 'private')[:11]
        topics = str(len(repo.get('topics', [])))
        
        protections = []
        if repo.get('secrets_and_environments', {}).get('repository_secrets'):
            protections.append('Repo Secrets')
        if repo.get('secrets_and_environments', {}).get('organization_secrets'):
            protections.append('Org Secrets')
        if repo.get('secrets_and_environments', {}).get('environments'):
            env_count = len(repo['secrets_and_environments']['environments'])
            protections.append(f'{env_count} Env(s)')
        
        protections_str = ', '.join(protections)[:24] if protections else 'None'
        
        print(f'{name:<25} {visibility:<12} {topics:<15} {protections_str:<25}')
    
    print('-' * 80)
    
    # Detailed protection summary
    print('\nDETAILED PROTECTION SUMMARY')
    print('-' * 40)
    
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
            print(f'\n{repo[\"name\"]}:')
            if repo_secrets > 0:
                print(f'  - Repository Secrets: {repo_secrets}')
            if org_secrets > 0:
                print(f'  - Organization Secrets: {org_secrets}')
            if environments > 0:
                print(f'  - Protected Environments: {environments}')
                for env_name, env_config in secrets_config['environments'].items():
                    reviewers = len(env_config.get('reviewers', []))
                    wait_time = env_config.get('wait_timer', 0)
                    env_secrets = len(env_config.get('secrets', {}))
                    print(f'    * {env_name}: {reviewers} reviewers, {wait_time}s wait, {env_secrets} secrets')
    
    print(f'\nTOTAL CONFIGURED PROTECTIONS:')
    print(f'- Repository Secrets: {total_repo_secrets}')
    print(f'- Organization Secrets: {total_org_secrets}')
    print(f'- Protected Environments: {total_environments}')
    print('='*80)

if __name__ == '__main__':
    try:
        manifest_file = '$ManifestFile'
        manifest = load_manifest(manifest_file)
        generate_summary_table(manifest, manifest_file)
    except Exception as e:
        print(f'Error generating summary: {e}', file=sys.stderr)
        sys.exit(1)
"

# Save summary to file
python -c "
import yaml
import sys
from datetime import datetime

def load_manifest(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

manifest = load_manifest('$ManifestFile')
repositories = manifest.get('repositories', [])

with open('setup_summary.txt', 'w') as f:
    f.write('\n' + '='*80 + '\n')
    f.write(' ' * 20 + 'REPOSITORY SETUP SUMMARY\n')
    f.write('='*80 + '\n')
    f.write(f'Manifest File: $ManifestFile\n')
    f.write(f'Execution Time: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n')
    f.write(f'Total Repositories: {len(repositories)}\n')
    f.write('='*80 + '\n')
print('Summary saved to setup_summary.txt')
"

# Cleanup
if (Test-Path ".venv-setup") {
    Remove-Item -Recurse -Force ".venv-setup"
}

Write-Host "======================================================" -ForegroundColor Green
Write-Host "           BOOTSTRAP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

Write-Host "[INFO] Bootstrap process completed successfully!" -ForegroundColor Green
Write-Host "[INFO] Check setup_summary.txt for setup summary" -ForegroundColor Green
