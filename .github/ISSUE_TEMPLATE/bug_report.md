---
name: "🐛 Bug Report"
about: Report a bug to help us improve Omni-Dev Agent
title: "[BUG] "
labels: ["bug", "needs-triage"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report! Please provide as much detail as possible to help us reproduce and fix the issue.
  
  - type: checkboxes
    id: prerequisites
    attributes:
      label: Prerequisites
      description: Please confirm you have completed the following
      options:
        - label: I have searched for existing issues
          required: true
        - label: I have read the documentation
          required: true
        - label: I am using a supported Python version (3.8+)
          required: true
        - label: I have tested with the latest version of Omni-Dev Agent
          required: true

  - type: textarea
    id: bug-description
    attributes:
      label: Bug Description
      description: A clear and concise description of what the bug is
      placeholder: Describe the bug...
    validations:
      required: true

  - type: textarea
    id: reproduction-steps
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Go to '...'
        2. Run command '...'
        3. See error
    validations:
      required: true

  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected Behavior
      description: A clear description of what you expected to happen
      placeholder: What should have happened?
    validations:
      required: true

  - type: textarea
    id: actual-behavior
    attributes:
      label: Actual Behavior
      description: What actually happened instead
      placeholder: What went wrong?
    validations:
      required: true

  - type: dropdown
    id: bug-severity
    attributes:
      label: Bug Severity
      description: How severe is this bug?
      options:
        - Critical (System unusable)
        - High (Major functionality broken)
        - Medium (Some functionality affected)
        - Low (Minor issue or cosmetic)
      default: 2
    validations:
      required: true

  - type: dropdown
    id: component-affected
    attributes:
      label: Component Affected
      description: Which component is affected by this bug?
      options:
        - Context Analyzer
        - Package Manager
        - Learning Engine
        - Error Manager
        - Documentation Analyzer
        - Testing Framework
        - Core Orchestration
        - Integration Components
        - CLI Interface
        - Unknown/Other
    validations:
      required: true

  - type: dropdown
    id: operating-system
    attributes:
      label: Operating System
      description: What operating system are you using?
      options:
        - Windows
        - macOS
        - Linux (Ubuntu)
        - Linux (Debian)
        - Linux (CentOS/RHEL)
        - Linux (Other)
        - Other
    validations:
      required: true

  - type: dropdown
    id: python-version
    attributes:
      label: Python Version
      description: What Python version are you using?
      options:
        - "3.12"
        - "3.11"
        - "3.10"
        - "3.9"
        - "3.8"
        - Other
    validations:
      required: true

  - type: input
    id: omni-version
    attributes:
      label: Omni-Dev Agent Version
      description: What version of Omni-Dev Agent are you using?
      placeholder: "e.g., 0.1.0, main branch, commit hash"
    validations:
      required: true

  - type: textarea
    id: error-logs
    attributes:
      label: Error Logs
      description: Please paste any relevant error messages or logs
      placeholder: Paste error logs here...
      render: shell

  - type: textarea
    id: code-snippet
    attributes:
      label: Code Snippet
      description: If applicable, provide a minimal code example that reproduces the issue
      placeholder: |
        ```python
        # Your code here
        ```
      render: python

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context about the problem here (screenshots, config files, etc.)
      placeholder: Any additional information that might help...

  - type: checkboxes
    id: urgency
    attributes:
      label: Urgency
      description: Check all that apply
      options:
        - label: This is blocking my work
        - label: This affects production systems
        - label: This is a regression from a previous version
        - label: I can provide a fix/patch
        - label: I'm willing to help test the fix
