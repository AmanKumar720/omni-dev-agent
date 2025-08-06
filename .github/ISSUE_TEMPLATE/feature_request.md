---
name: "✨ Feature Request"
about: Suggest a new feature for Omni-Dev Agent
title: "[FEATURE] "
labels: ["enhancement", "needs-review"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature! Please provide detailed information to help us understand your request.
  
  - type: checkboxes
    id: prerequisites
    attributes:
      label: Prerequisites
      description: Please confirm you have completed the following
      options:
        - label: I have searched for existing feature requests
          required: true
        - label: I have read the project documentation
          required: true
        - label: This feature aligns with Omni-Dev Agent's goals
          required: true
        - label: This is not a bug report (use bug report template instead)
          required: true

  - type: dropdown
    id: feature-category
    attributes:
      label: Feature Category
      description: What category does this feature belong to?
      options:
        - Context Analysis Enhancement
        - Package Management
        - Learning & AI Capabilities
        - Error Handling & Recovery
        - Documentation Processing
        - Testing & Quality Assurance
        - Security & Compliance
        - Performance & Optimization
        - CLI & User Interface
        - Integration & API
        - Other
    validations:
      required: true

  - type: dropdown
    id: feature-priority
    attributes:
      label: Priority Level
      description: How important is this feature to you?
      options:
        - Critical (Essential for my workflow)
        - High (Would significantly improve my experience)
        - Medium (Nice to have)
        - Low (Minor enhancement)
      default: 2
    validations:
      required: true

  - type: textarea
    id: problem-description
    attributes:
      label: Problem Description
      description: Is your feature request related to a problem? Please describe.
      placeholder: "I'm always frustrated when..."
    validations:
      required: true

  - type: textarea
    id: proposed-solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
      placeholder: "I would like to see..."
    validations:
      required: true

  - type: textarea
    id: use-cases
    attributes:
      label: Use Cases
      description: Describe specific use cases and who would benefit
      placeholder: |
        1. As a developer, I want to...
        2. This would help users who...
        3. The benefit would be...
    validations:
      required: true

  - type: textarea
    id: alternatives-considered
    attributes:
      label: Alternatives Considered
      description: Describe alternative solutions or features you've considered
      placeholder: "I also considered..."

  - type: textarea
    id: implementation-ideas
    attributes:
      label: Implementation Ideas
      description: If you have ideas about how to implement this feature, please share them
      placeholder: |
        ```python
        # Potential implementation approach
        ```
      render: python

  - type: textarea
    id: acceptance-criteria
    attributes:
      label: Acceptance Criteria
      description: What would make this feature complete and successful?
      placeholder: |
        - [ ] Feature should do X
        - [ ] Feature should handle Y scenario
        - [ ] Feature should integrate with Z

  - type: checkboxes
    id: feature-scope
    attributes:
      label: Feature Scope
      description: Check all that apply
      options:
        - label: This is a breaking change
        - label: This requires new dependencies
        - label: This affects the CLI interface
        - label: This requires documentation updates
        - label: This needs new tests
        - label: This impacts performance
        - label: This affects security

  - type: checkboxes
    id: contribution-willingness
    attributes:
      label: Contribution
      description: Are you willing to help with this feature?
      options:
        - label: I can help with design/planning
        - label: I can help with implementation
        - label: I can help with testing
        - label: I can help with documentation
        - label: I can provide ongoing feedback

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context, mockups, or examples about the feature request
      placeholder: Any additional information, screenshots, or examples...
