import os
import json
import yaml
import base64
import requests
from github import Github, GithubException
from nacl import encoding, public
from typing import Dict, List, Optional

# Read the configuration manifest (either JSON or YAML)
def load_manifest(filename):
    with open(filename, 'r') as file:
        if filename.endswith('.json'):
            return json.load(file)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            return yaml.safe_load(file)
        else:
            raise ValueError('The manifest file must be a JSON or YAML file.')

# Get GitHub authentication token from environment
def get_github_client():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise EnvironmentError('GITHUB_TOKEN environment variable not set.')
    return Github(token)

# Create or clone a repository
def initialize_repo(gh_client, repo_config):
    user = gh_client.get_user()
    repo_name = repo_config['name']
    description = repo_config.get('description', '')
    private = repo_config.get('private', True)

    try:
        repo = user.get_repo(repo_name)
        print(f'Repository {repo_name} already exists. Cloning...')
        # Here, you clone the repository if needed
    except GithubException:
        print(f'Creating repository {repo_name}...')
        repo = user.create_repo(
            name=repo_name,
            description=description,
            private=private
        )
    return repo

# Push `.github` directory via a temporary branch and create a PR
def push_github_folder(repo):
    
    branch_name = "temporary-branch"
    try:
        default_branch = repo.default_branch

        # Create temporary branch
        source = repo.get_branch(default_branch)
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=source.commit.sha)

        # Add and commit .github directory
        # This part needs to add files and commit, can use local git commands
        # or scripts via os.system() or subprocess.

        # Create Pull Request
        repo.create_pull(
            title='Add .github folder',
            body='Automated Addition of .github Folder',
            head=branch_name,
            base=default_branch
        )
    except GithubException as e:
        print(f'Error pushing .github folder: {e}')

# Enable branch protection and settings
def configure_repo(repo):
    try:
        branch = repo.get_branch(repo.default_branch)
        branch.edit_protection(
            required_approving_review_count=1,
            enforce_admins=True,
            required_status_checks=None,
            restrictions=None
        )

        repo.edit(allow_auto_merge=True)

        # Enable security features
        repo.enable_vulnerability_alert()
        repo.enable_secret_scanning()
    except GithubException as e:
        print(f'Error in configuring repository: {e}')

# Set labels, topics, and visibility
def set_repo_metadata(repo, repo_config):
    try:
        topics = repo_config.get('topics', [])
        repo.replace_topics(topics)

        labels = repo_config.get('labels', [])
        for label in labels:
            try:
                repo.create_label(name=label['name'], color=label.get('color', 'FFFFFF'))
            except GithubException:
                print(f'Label {label['name']} already exists.')

        repo.edit(visibility=repo_config.get('visibility', 'private'))
    except GithubException as e:
        print(f'Error setting repository metadata: {e}')

# Encrypt a secret using the repository's public key
def encrypt_secret(public_key: str, secret_value: str) -> str:
    """Encrypt a secret using the repository's public key."""
    public_key_bytes = base64.b64decode(public_key)
    sealed_box = public.SealedBox(public.PublicKey(public_key_bytes))
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")

# Get repository public key for secret encryption
def get_repo_public_key(repo):
    """Get the repository's public key for secret encryption."""
    try:
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
                  'Accept': 'application/vnd.github.v3+json'}
        url = f'https://api.github.com/repos/{repo.full_name}/actions/secrets/public-key'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f'Error getting repository public key: {e}')
        return None

# Create or update a repository secret
def create_repo_secret(repo, secret_name: str, secret_value: str):
    """Create or update a repository secret."""
    try:
        public_key_data = get_repo_public_key(repo)
        if not public_key_data:
            return False
        
        encrypted_value = encrypt_secret(public_key_data['key'], secret_value)
        
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
                  'Accept': 'application/vnd.github.v3+json'}
        url = f'https://api.github.com/repos/{repo.full_name}/actions/secrets/{secret_name}'
        
        data = {
            'encrypted_value': encrypted_value,
            'key_id': public_key_data['key_id']
        }
        
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        print(f'Successfully created/updated repository secret: {secret_name}')
        return True
    except requests.RequestException as e:
        print(f'Error creating repository secret {secret_name}: {e}')
        return False

# Get organization public key for secret encryption
def get_org_public_key(org_name: str):
    """Get the organization's public key for secret encryption."""
    try:
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
                  'Accept': 'application/vnd.github.v3+json'}
        url = f'https://api.github.com/orgs/{org_name}/actions/secrets/public-key'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f'Error getting organization public key: {e}')
        return None

# Create or update an organization secret
def create_org_secret(org_name: str, secret_name: str, secret_value: str, selected_repository_ids: Optional[List[int]] = None):
    """Create or update an organization secret."""
    try:
        public_key_data = get_org_public_key(org_name)
        if not public_key_data:
            return False
        
        encrypted_value = encrypt_secret(public_key_data['key'], secret_value)
        
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
                  'Accept': 'application/vnd.github.v3+json'}
        url = f'https://api.github.com/orgs/{org_name}/actions/secrets/{secret_name}'
        
        data = {
            'encrypted_value': encrypted_value,
            'key_id': public_key_data['key_id'],
            'visibility': 'selected' if selected_repository_ids else 'all',
        }
        
        if selected_repository_ids:
            data['selected_repository_ids'] = selected_repository_ids
        
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        print(f'Successfully created/updated organization secret: {secret_name}')
        return True
    except requests.RequestException as e:
        print(f'Error creating organization secret {secret_name}: {e}')
        return False

# Create a protected environment
def create_environment(repo, env_name: str, reviewers: Optional[List[str]] = None, wait_timer: int = 0):
    """Create a protected environment with optional reviewers and wait timer."""
    try:
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
                  'Accept': 'application/vnd.github.v3+json'}
        url = f'https://api.github.com/repos/{repo.full_name}/environments/{env_name}'
        
        data = {}
        
        if reviewers or wait_timer > 0:
            protection_rules = []
            
            if reviewers:
                # Convert usernames to user IDs if needed
                reviewer_data = []
                for reviewer in reviewers:
                    if isinstance(reviewer, str):
                        # Assume it's a username, convert to user object
                        try:
                            user = repo._github.get_user(reviewer)
                            reviewer_data.append({'type': 'User', 'id': user.id})
                        except GithubException:
                            print(f'Warning: Could not find user {reviewer}')
                    else:
                        reviewer_data.append(reviewer)
                
                protection_rules.append({
                    'type': 'required_reviewers',
                    'reviewers': reviewer_data
                })
            
            if wait_timer > 0:
                protection_rules.append({
                    'type': 'wait_timer',
                    'wait_timer': wait_timer
                })
            
            data['protection_rules'] = protection_rules
        
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        print(f'Successfully created/updated environment: {env_name}')
        return True
    except requests.RequestException as e:
        print(f'Error creating environment {env_name}: {e}')
        return False

# Create environment-specific secrets
def create_environment_secret(repo, env_name: str, secret_name: str, secret_value: str):
    """Create or update an environment-specific secret."""
    try:
        public_key_data = get_repo_public_key(repo)
        if not public_key_data:
            return False
        
        encrypted_value = encrypt_secret(public_key_data['key'], secret_value)
        
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
                  'Accept': 'application/vnd.github.v3+json'}
        url = f'https://api.github.com/repos/{repo.full_name}/environments/{env_name}/secrets/{secret_name}'
        
        data = {
            'encrypted_value': encrypted_value,
            'key_id': public_key_data['key_id']
        }
        
        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        print(f'Successfully created/updated environment secret: {env_name}/{secret_name}')
        return True
    except requests.RequestException as e:
        print(f'Error creating environment secret {env_name}/{secret_name}: {e}')
        return False

# Setup secrets and environments based on configuration
def setup_secrets_and_environments(repo, secrets_config: Dict):
    """Setup repository secrets, organization secrets, and environments."""
    print(f'Setting up secrets and environments for {repo.name}...')
    
    # Repository secrets
    repo_secrets = secrets_config.get('repository_secrets', {})
    for secret_name, secret_value in repo_secrets.items():
        # Check if secret value is in environment variable
        if secret_value.startswith('${') and secret_value.endswith('}'):
            env_var = secret_value[2:-1]
            secret_value = os.getenv(env_var)
            if not secret_value:
                print(f'Warning: Environment variable {env_var} not set for secret {secret_name}')
                continue
        
        create_repo_secret(repo, secret_name, secret_value)
    
    # Organization secrets (if user has permissions)
    org_secrets = secrets_config.get('organization_secrets', {})
    if org_secrets:
        try:
            org_name = repo.organization.login if repo.organization else repo.owner.login
            for secret_name, config in org_secrets.items():
                secret_value = config.get('value', '')
                selected_repos = config.get('selected_repository_ids', None)
                
                # Check if secret value is in environment variable
                if secret_value.startswith('${') and secret_value.endswith('}'):
                    env_var = secret_value[2:-1]
                    secret_value = os.getenv(env_var)
                    if not secret_value:
                        print(f'Warning: Environment variable {env_var} not set for org secret {secret_name}')
                        continue
                
                create_org_secret(org_name, secret_name, secret_value, selected_repos)
        except Exception as e:
            print(f'Warning: Could not create organization secrets: {e}')
    
    # Environments
    environments = secrets_config.get('environments', {})
    for env_name, env_config in environments.items():
        reviewers = env_config.get('reviewers', [])
        wait_timer = env_config.get('wait_timer', 0)
        
        # Create the environment
        if create_environment(repo, env_name, reviewers, wait_timer):
            # Add environment-specific secrets
            env_secrets = env_config.get('secrets', {})
            for secret_name, secret_value in env_secrets.items():
                # Check if secret value is in environment variable
                if secret_value.startswith('${') and secret_value.endswith('}'):
                    env_var = secret_value[2:-1]
                    secret_value = os.getenv(env_var)
                    if not secret_value:
                        print(f'Warning: Environment variable {env_var} not set for env secret {secret_name}')
                        continue
                
                create_environment_secret(repo, env_name, secret_name, secret_value)

# Main function accepting manifest file
def main(manifest_file):
    gh_client = get_github_client()
    manifest = load_manifest(manifest_file)

    for repo_config in manifest['repositories']:
        repo = initialize_repo(gh_client, repo_config)
        push_github_folder(repo)
        configure_repo(repo)
        set_repo_metadata(repo, repo_config)
        
        # Setup secrets and environments if specified
        secrets_config = repo_config.get('secrets_and_environments', {})
        if secrets_config:
            setup_secrets_and_environments(repo, secrets_config)

# Entry point
def setup_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Setup GitHub Repositories')
    parser.add_argument('manifest_file', metavar='M', type=str, 
                        help='Path to the JSON/YAML manifest file describing repositories')

    args = parser.parse_args()
    main(args.manifest_file)

if __name__ == '__main__':
    setup_cli()
