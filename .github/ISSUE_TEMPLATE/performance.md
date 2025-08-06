---
name: "⚡ Performance Issue"
about: Report a performance problem with Omni-Dev Agent
title: "[PERFORMANCE] "
labels: ["performance", "needs-investigation"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting a performance issue! This helps us make Omni-Dev Agent faster and more efficient.
  
  - type: checkboxes
    id: prerequisites
    attributes:
      label: Prerequisites
      description: Please confirm you have completed the following
      options:
        - label: I have searched for existing performance issues
          required: true
        - label: I have profiled or measured the performance issue
          required: true
        - label: I am using the latest version of Omni-Dev Agent
          required: true
        - label: This is a performance issue, not a functional bug
          required: true

  - type: dropdown
    id: performance-category
    attributes:
      label: Performance Category
      description: What type of performance issue is this?
      options:
        - Slow startup/initialization
        - Memory usage/leaks
        - CPU usage/high load
        - Disk I/O performance
        - Network operations
        - Context analysis speed
        - Package management operations
        - Learning engine performance
        - Documentation processing
        - Test execution speed
        - Overall system responsiveness
      default: 0
    validations:
      required: true

  - type: dropdown
    id: severity-level
    attributes:
      label: Severity Level
      description: How severe is the performance impact?
      options:
        - Critical (System unusable)
        - High (Significantly impacts workflow)
        - Medium (Noticeable slowdown)
        - Low (Minor performance issue)
      default: 2
    validations:
      required: true

  - type: textarea
    id: performance-description
    attributes:
      label: Performance Issue Description
      description: Describe the performance problem in detail
      placeholder: What performance issue are you experiencing?
    validations:
      required: true

  - type: textarea
    id: expected-performance
    attributes:
      label: Expected Performance
      description: What performance did you expect?
      placeholder: How fast should this operation be?
    validations:
      required: true

  - type: textarea
    id: actual-performance
    attributes:
      label: Actual Performance
      description: What performance are you actually seeing?
      placeholder: How slow is it currently? Include specific timing if available.
    validations:
      required: true

  - type: textarea
    id: performance-measurements
    attributes:
      label: Performance Measurements
      description: Provide specific measurements (timing, memory usage, etc.)
      placeholder: |
        - Operation takes X seconds (expected Y seconds)
        - Memory usage: X MB (expected Y MB)
        - CPU usage: X% (expected Y%)
      render: text

  - type: textarea
    id: reproduction-scenario
    attributes:
      label: Reproduction Scenario
      description: How can we reproduce this performance issue?
      placeholder: |
        1. Set up environment with...
        2. Run operation...
        3. Measure performance...
    validations:
      required: true

  - type: dropdown
    id: operating-system
    attributes:
      label: Operating System
      description: What OS are you running on?
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
      description: What version are you using?
      placeholder: "e.g., 0.1.0, main branch, commit hash"
    validations:
      required: true

  - type: textarea
    id: system-specs
    attributes:
      label: System Specifications
      description: Provide relevant system specifications
      placeholder: |
        - CPU: 
        - RAM: 
        - Storage: 
        - Other relevant specs:

  - type: textarea
    id: dataset-size
    attributes:
      label: Dataset/Project Size
      description: Information about the size of data being processed
      placeholder: |
        - Number of files: 
        - Total project size: 
        - Lines of code: 
        - Dependencies count:

  - type: textarea
    id: profiling-data
    attributes:
      label: Profiling Data
      description: If you have profiling data, please include it
      placeholder: |
        ```
        # Profiling output, timing data, etc.
        ```
      render: text

  - type: textarea
    id: workarounds
    attributes:
      label: Workarounds
      description: Have you found any workarounds for this performance issue?
      placeholder: Describe any workarounds you've discovered...

  - type: checkboxes
    id: performance-impact
    attributes:
      label: Performance Impact
      description: Check all that apply
      options:
        - label: Affects daily workflow
        - label: Causes timeouts or failures
        - label: Increases resource costs
        - label: Impacts CI/CD pipeline
        - label: Affects user experience
        - label: Prevents scaling up usage

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Any other context about the performance issue
      placeholder: Any additional information, logs, or context...
