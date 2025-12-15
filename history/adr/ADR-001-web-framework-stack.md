# ADR-001: Web Framework and Static Site Generator Stack

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-10
- **Feature:** Physical AI & Humanoid Robotics Book
- **Context:** Need to select a technology stack for creating a static site to host an educational book on Physical AI and Humanoid Robotics. The solution must support modular content, be highly accessible, and deployable to GitHub Pages with fast load times.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

Use Docusaurus as the static site generator with React and Node.js for the Physical AI & Humanoid Robotics educational book.

- Framework: Docusaurus 3.x
- Content Format: Markdown with MDX support
- Styling: Docusaurus built-in theme with custom CSS
- Deployment: GitHub Pages
- Search: Algolia DocSearch integration

## Consequences

### Positive

- Excellent documentation-focused features with built-in search, versioning, and navigation
- Strong Markdown/MDX support ideal for technical documentation with code examples
- Built-in accessibility features and compliance
- Active community and ecosystem for documentation sites
- SEO-friendly with static generation
- Integration with GitHub workflow for content management
- Support for interactive components and custom React elements

### Negative

- Additional build step and dependency on Node.js ecosystem
- Potential complexity for non-technical content authors
- Possible performance overhead compared to simpler static site generators
- Learning curve for custom theming beyond default options
- Dependency on Facebook/Meta's continued support for Docusaurus

## Alternatives Considered

Alternative Stack A: Jekyll + GitHub Pages (native integration)
- Why rejected: Less suitable for complex technical documentation with interactive elements, limited JavaScript support

Alternative Stack B: Hugo + Netlify
- Why rejected: Less suitable for Markdown-heavy technical content, steeper learning curve for non-developers

Alternative Stack C: VuePress
- Why rejected: Smaller ecosystem and community compared to Docusaurus, less proven for large-scale documentation projects

## References

- Feature Spec: ./specs/1-physical-ai-book/spec.md
- Implementation Plan: ./specs/1-physical-ai-book/plan.md
- Related ADRs:
- Evaluator Evidence: ./history/prompts/1-physical-ai-book/