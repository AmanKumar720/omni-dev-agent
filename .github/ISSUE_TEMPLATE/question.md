---
name: "❓ Question"
about: Ask a question about Omni-Dev Agent
title: "[QUESTION] "
labels: ["question", "help-wanted"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for your question! Please provide as much detail as possible to help us give you the best answer.
  
  - type: checkboxes
    id: prerequisites
    attributes:
      label: Prerequisites
      description: Please confirm you have completed the following
      options:
        - label: I have searched existing issues and discussions
          required: true
        - label: I have checked the documentation
          required: true
        - label: My question is specific to Omni-Dev Agent
          required: true
        - label: This is not a bug report or feature request
          required: true

  - type: dropdown
    id: question-category
    attributes:
      label: Question Category
      description: What category does your question relate to?
      options:
        - Installation & Setup
        - Configuration & Usage
        - Context Analysis
        - Package Management
        - Learning Engine
        - Error Handling
        - Documentation Processing
        - Testing Framework
        - Integration & APIs
        - Performance & Optimization
        - Troubleshooting
        - Best Practices
        - Other
    validations:
      required: true

  - type: dropdown
    id: urgency-level
    attributes:
      label: Urgency Level
      description: How urgent is getting an answer to this question?
      options:
        - Urgent (Blocking my work)
        - High (Need answer soon)
        - Medium (Would be helpful)
        - Low (Just curious)
      default: 2
    validations:
      required: true

  - type: textarea
    id: question-details
    attributes:
      label: Your Question
      description: Please provide a clear and concise question
      placeholder: What would you like to know?
    validations:
      required: true

  - type: textarea
    id: context-background
    attributes:
      label: Context & Background
      description: Provide relevant context or background information
      placeholder: |
        - What are you trying to achieve?
        - What's your current setup?
        - What led to this question?
    validations:
      required: true

  - type: textarea
    id: attempted-solutions
    attributes:
      label: What You've Tried
      description: Describe what you've already tried or researched
      placeholder: |
        - I tried X but got Y
        - I looked at Z documentation
        - I searched for A but couldn't find B

  - type: dropdown
    id: operating-system
    attributes:
      label: Operating System (if relevant)
      description: What operating system are you using?
      options:
        - Not applicable
        - Windows
        - macOS
        - Linux (Ubuntu)
        - Linux (Debian)
        - Linux (CentOS/RHEL)
        - Linux (Other)
        - Other
      default: 0

  - type: dropdown
    id: python-version
    attributes:
      label: Python Version (if relevant)
      description: What Python version are you using?
      options:
        - Not applicable
        - "3.12"
        - "3.11"
        - "3.10"
        - "3.9"
        - "3.8"
        - Other
      default: 0

  - type: input
    id: omni-version
    attributes:
      label: Omni-Dev Agent Version (if relevant)
      description: What version are you using?
      placeholder: "e.g., 0.1.0, main branch, or 'not applicable'"

  - type: textarea
    id: code-examples
    attributes:
      label: Code Examples
      description: If applicable, provide relevant code snippets
      placeholder: |
        ```python
        # Your code here
        ```
      render: python

  - type: textarea
    id: error-messages
    attributes:
      label: Error Messages (if any)
      description: If you're getting errors, please include them
      placeholder: Paste any error messages here...
      render: shell

  - type: textarea
    id: expected-outcome
    attributes:
      label: Expected Outcome
      description: What outcome are you hoping to achieve?
      placeholder: What should happen or what result are you looking for?

  - type: textarea
    id: additional-info
    attributes:
      label: Additional Information
      description: Any other context, screenshots, or details that might help
      placeholder: Any other relevant information...

  - type: checkboxes
    id: help-preferences
    attributes:
      label: Help Preferences
      description: How would you prefer to receive help?
      options:
        - label: Detailed written explanation
        - label: Code examples
        - label: Links to relevant documentation
        - label: Step-by-step instructions
        - label: I'm willing to try solutions and provide feedback
