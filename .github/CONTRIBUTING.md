# Contributing to {{ repo_name }}

Thank you for your interest in contributing to {{ repo_name }}! We welcome contributions from everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/{{ github_username }}/{{ repo_name }}.git
   cd {{ repo_name }}
   ```
3. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Python {{ python_version | default('3.8+') }}
{% if use_poetry %}
- Poetry for dependency management
{% else %}
- pip for dependency management
{% endif %}

### Installation
{% if use_poetry %}
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```
{% else %}
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```
{% endif %}

## Development Workflow

### Running Tests
{% if use_poetry %}
```bash
poetry run pytest tests/
```
{% else %}
```bash
pytest tests/
```
{% endif %}

### Code Quality Checks
{% if use_poetry %}
```bash
# Linting
poetry run pylint {{ package_name | default('src') }}

# Type checking
poetry run mypy {{ package_name | default('src') }}

# Security scanning
poetry run bandit -r {{ package_name | default('src') }}

# Code formatting
poetry run black {{ package_name | default('src') }} tests/
poetry run isort {{ package_name | default('src') }} tests/
```
{% else %}
```bash
# Linting
pylint {{ package_name | default('src') }}

# Type checking
mypy {{ package_name | default('src') }}

# Security scanning
bandit -r {{ package_name | default('src') }}

# Code formatting
black {{ package_name | default('src') }} tests/
isort {{ package_name | default('src') }} tests/
```
{% endif %}

### Pre-commit Hooks
We use pre-commit hooks to ensure code quality. Install them with:
```bash
pre-commit install
```

## Submitting Changes

1. Make sure all tests pass
2. Ensure your code follows our style guidelines
3. Update documentation if necessary
4. Add tests for new functionality
5. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a Pull Request on GitHub

## Pull Request Guidelines

- Fill out the PR template completely
- Link any related issues
- Ensure CI checks pass
- Keep PRs focused on a single feature or bug fix
- Write clear, descriptive commit messages
- Update documentation as needed

## Code Style

- Follow PEP 8 for Python code style
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep line length to 88 characters (Black's default)
- Use descriptive variable and function names

## Testing Guidelines

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies appropriately

## Documentation

- Update docstrings for any changes to public APIs
- Add examples for new features
- Update README.md if needed
- Consider adding or updating tutorials

## Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the appropriate issue template
3. Provide as much detail as possible
4. Include steps to reproduce for bugs

## Community Guidelines

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion on GitHub
- Create an issue with the "question" label
- Contact the maintainers at {{ contact_email | default('maintainers@example.com') }}

Thank you for contributing to {{ repo_name }}!
