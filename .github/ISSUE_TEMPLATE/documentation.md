---
name: "📚 Documentation Issue"
about: Report an issue with documentation or request documentation improvements
title: "[DOCS] "
labels: ["documentation", "needs-review"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for helping improve our documentation! Clear documentation is essential for a great developer experience.
  
  - type: checkboxes
    id: prerequisites
    attributes:
      label: Prerequisites
      description: Please confirm you have completed the following
      options:
        - label: I have searched for existing documentation issues
          required: true
        - label: I have checked multiple pages for this information
          required: true
        - label: This is specifically about documentation, not a feature request
          required: true

  - type: dropdown
    id: documentation-type
    attributes:
      label: Documentation Type
      description: What type of documentation issue is this?
      options:
        - Missing documentation
        - Incorrect/outdated information
        - Unclear/confusing content
        - Broken links or references
        - Code examples don't work
        - API documentation issues
        - Installation instructions
        - Configuration guides
        - Tutorial improvements
        - Reference documentation
        - Other
    validations:
      required: true

  - type: dropdown
    id: documentation-area
    attributes:
      label: Documentation Area
      description: Which area of documentation is affected?
      options:
        - README.md
        - API Documentation
        - Installation Guide
        - Configuration
        - Tutorials/Examples
        - Context Analysis
        - Package Management
        - Learning Engine
        - Error Handling
        - Testing Framework
        - Contributing Guidelines
        - Security Policy
        - Other
    validations:
      required: true

  - type: dropdown
    id: priority-level
    attributes:
      label: Priority Level
      description: How important is fixing this documentation issue?
      options:
        - Critical (Prevents users from getting started)
        - High (Significantly impacts user experience)
        - Medium (Would improve user experience)
        - Low (Minor improvement)
      default: 2
    validations:
      required: true

  - type: textarea
    id: issue-description
    attributes:
      label: Documentation Issue Description
      description: Clearly describe the documentation issue
      placeholder: What's wrong with the current documentation?
    validations:
      required: true

  - type: input
    id: documentation-location
    attributes:
      label: Documentation Location
      description: Where is this documentation located?
      placeholder: "e.g., README.md line 42, docs/api.md, wiki page URL"
    validations:
      required: true

  - type: textarea
    id: current-content
    attributes:
      label: Current Content (if applicable)
      description: Copy the current text that needs improvement
      placeholder: |
        ```
        Current documentation text that's problematic
        ```
      render: markdown

  - type: textarea
    id: expected-content
    attributes:
      label: Expected/Suggested Content
      description: What should the documentation say instead?
      placeholder: |
        Provide suggested improvements or what you expected to find
      render: markdown

  - type: textarea
    id: user-impact
    attributes:
      label: User Impact
      description: How does this documentation issue affect users?
      placeholder: |
        - Who is affected by this issue?
        - What problems does it cause?
        - How does it impact the user experience?
    validations:
      required: true

  - type: checkboxes
    id: documentation-issues
    attributes:
      label: Specific Issues (check all that apply)
      description: What specific problems exist?
      options:
        - label: Information is missing
        - label: Information is incorrect
        - label: Information is outdated
        - label: Steps don't work as described
        - label: Code examples are broken
        - label: Links are broken
        - label: Images/diagrams are missing or broken
        - label: Grammar/spelling errors
        - label: Formatting issues
        - label: Inconsistent terminology
        - label: Lacks examples
        - label: Too technical/complex
        - label: Not enough detail
        - label: Too much detail/verbose

  - type: textarea
    id: reproduction-context
    attributes:
      label: Context/Environment (if relevant)
      description: If this is about instructions that don't work, provide environment details
      placeholder: |
        - Operating System:
        - Python Version:
        - Omni-Dev Agent Version:
        - What you were trying to do:

  - type: textarea
    id: suggested-solution
    attributes:
      label: Suggested Solution
      description: How do you think this should be fixed?
      placeholder: |
        - Add section about X
        - Update the example to show Y
        - Clarify the explanation of Z
        - Add link to related resource

  - type: checkboxes
    id: contribution-offer
    attributes:
      label: Contribution Offer
      description: Would you like to help fix this documentation issue?
      options:
        - label: I can help write the improved content
        - label: I can help review proposed changes
        - label: I can provide additional examples
        - label: I can help test the instructions
        - label: I can provide technical expertise

  - type: dropdown
    id: target-audience
    attributes:
      label: Target Audience
      description: Who is the primary audience for this documentation?
      options:
        - New users/beginners
        - Experienced developers
        - System administrators
        - Contributors/maintainers
        - API users
        - All users
        - Other
      default: 5

  - type: textarea
    id: related-resources
    attributes:
      label: Related Resources
      description: Links to related documentation, issues, or external resources
      placeholder: |
        - Related issue: #123
        - Similar documentation: [link]
        - External reference: [link]

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Any other context about this documentation issue
      placeholder: Screenshots, additional details, or context that might help...
