---
name: "🔒 Security Issue"
about: Report a security vulnerability in Omni-Dev Agent
security: true
title: "[SECURITY] "
labels: ["security", "confidential"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        ⚠️ **IMPORTANT: Security vulnerabilities must be reported privately!**
  
  - type: input
    id: contact-email
    attributes:
      label: Contact Email
      description: Please provide your email so we can follow up with you (only visible to maintainers)
    validations:
      required: true
  
  - type: markdown
    attributes:
      value: |
        Please report the security vulnerabilities by visiting [Security Policy](../SECURITY.md) and following the guidance there.

  - type: textarea
    id: security-description
    attributes:
      label: Vulnerability Description
      description: Describe the nature of the vulnerability
      placeholder: "Describe the security issue..."

  - type: textarea
    id: reproduction-steps
    attributes:
      label: Steps to Reproduce
      description: Clear steps to reproduce the vulnerability
      placeholder: "1. Do this
2. Then this
3. Vulnerability manifests"

  - type: textarea
    id: impact
    attributes:
      label: Potential Impact
      description: What could happen if the vulnerability is exploited?
      placeholder: "Explain the impact..."

  - type: textarea
    id: mitigation-suggestions
    attributes:
      label: Suggested Mitigation
      description: If you have ideas on how to fix or mitigate, please provide them
      placeholder: "Consider doing this..."

  - type: textarea
    id: additional-information
    attributes:
      label: Additional Information
      description: Any other related details or links
      placeholder: "Provide any more information..."
