# Contributing to Omni-Dev Agent

We love your input! We want to make contributing to Omni-Dev Agent as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Quality Standards

### Python Code Style
- We use [Black](https://github.com/psf/black) for code formatting
- We use [pylint](https://pylint.org/) for linting
- We use [mypy](http://mypy-lang.org/) for type checking
- We use [bandit](https://bandit.readthedocs.io/) for security analysis

### Before Submitting
Run these commands to ensure code quality:

```bash
# Format code
black src/ tests/

# Lint code
pylint src/

# Type checking
mypy src/

# Security analysis
bandit -r src/

# Run tests
python run_tests.py
```

## Testing

- Write tests for any new functionality
- Ensure all existing tests continue to pass
- Aim for high test coverage
- Use meaningful test names that describe what is being tested

### Test Types
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Static Analysis**: Code quality and security checks

## Pull Request Process

1. **Update Documentation**: Update the README.md with details of changes to the interface
2. **Add Tests**: Include appropriate tests for your changes
3. **Version Bumping**: You may merge the Pull Request once you have the sign-off of two other developers
4. **Code Review**: All submissions require review before being merged

## Issue Reporting

### Bug Reports

Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

### Feature Requests

We welcome feature requests! Please provide:

- A clear and detailed explanation of the feature
- Why this feature would be useful to most users
- How the feature should work
- Any relevant examples or mockups

## Development Setup

### Prerequisites
- Python 3.8+
- Git

### Setup Steps

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/omni-dev-agent.git
   cd omni-dev-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   python run_tests.py
   ```

## Coding Guidelines

### General Principles
- Keep it simple and readable
- Write self-documenting code with clear variable and function names
- Follow the existing code style and patterns
- Add docstrings to all public functions and classes
- Handle errors appropriately and provide meaningful error messages

### Architecture Guidelines
- Follow the modular architecture pattern
- Keep components loosely coupled
- Use dependency injection where appropriate
- Implement proper logging throughout your code
- Consider security implications of your changes

### Documentation
- Update docstrings for any modified functions
- Add comments for complex logic
- Update README.md if your changes affect usage
- Add examples for new features

## Community

### Code of Conduct
- Be respectful and inclusive
- Focus on what is best for the community
- Show empathy towards other community members
- Be collaborative
- Use welcoming and inclusive language

### Getting Help
- Check existing issues and documentation first
- Join our discussions on GitHub
- Ask questions in issues with the "question" label
- Be patient and respectful when asking for help

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- The project's README.md file
- Release notes for significant contributions
- The project's contributor page

Thank you for contributing to Omni-Dev Agent! 🚀
