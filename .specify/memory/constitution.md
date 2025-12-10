<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.0.0 (initial constitution)
- Added sections: Core Principles (6), Technical Standards, Development Workflow, Governance
- Templates requiring updates: N/A (initial creation)
- Follow-up TODOs: None
-->
# AI/Spec-Driven Book — Physical AI & Humanoid Robotics Constitution

## Core Principles

### Technical Accuracy and Educational Clarity
All robotics and AI concepts must be technically correct and current. Content must be clear and accessible for students and developers entering Physical AI, with step-by-step, hands-on educational structure that prioritizes understanding over complexity.

### Spec-Driven Development Excellence
Use Spec-Kit Plus and Claude Code as the primary authoring system for all content creation. Every chapter must begin with a specification that defines objectives, provides working examples (code, configs, or workflows), and includes a final exercise to reinforce learning.

### Ethical Responsibility and Safety First
All content must emphasize ethical considerations, safety protocols, and responsible AI practices. When covering robotics implementations, safety notes and failure modes must be explicitly addressed to ensure responsible development practices.

### Industry-Standard Implementation Patterns
ROS 2, Gazebo, Unity, and NVIDIA Isaac examples must be realistic and follow industry-grade patterns for voice, vision, and action pipelines. All technical implementations must reflect current best practices in robotics and AI engineering.

### Modular and Accessible Content Structure
Chapters must be modular and short for spec-driven generation, with clean, structured, and beginner-friendly writing. Content must be fully compatible with Docusaurus Markdown format and include alt+text for all images/figures.

### Originality and Quality Assurance
Maintain 0% plagiarism tolerance with only original content. Every technical module must include inputs, outputs, architectures, codes, failure modes, and safety notes to ensure comprehensive coverage of topics.

## Technical Standards

- Output format: Markdown (.md / .mdx) compatible with Docusaurus
- Deployment: GitHub Pages using Docusaurus
- Technology stack: ROS 2, Gazebo, Unity, NVIDIA Isaac for robotics examples
- Code examples: Follow industry-grade patterns for voice, vision, and action pipelines
- Images/figures: Must include alt+text for accessibility
- Navigation: Ensure no broken links or missing navigation elements

## Development Workflow

- Each chapter begins with specification defining clear objectives
- Provide working examples including code, configurations, or workflows
- Include final exercise for each chapter to reinforce learning
- Every technical module includes: inputs, outputs, architectures, codes, failure modes, safety notes
- Use Spec-Kit Plus + Claude Code as the primary authoring system
- Validate content against Docusaurus build requirements
- Test rendering on GitHub Pages environment

## Governance

This constitution supersedes all other development practices and standards within the project. All content creation, modifications, and reviews must verify compliance with these principles. Any deviation from these principles requires explicit justification and approval. Complexity must be justified with clear educational value. Use this constitution as the primary guidance document for all development decisions.

**Version**: 1.0.0 | **Ratified**: 2025-12-09 | **Last Amended**: 2025-12-09