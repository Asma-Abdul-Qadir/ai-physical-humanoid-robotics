# ADR-002: Simulation Environment Technology Stack

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-10
- **Feature:** Physical AI & Humanoid Robotics Book
- **Context:** Need to select simulation environments for demonstrating Physical AI and Humanoid Robotics concepts. The solution must support ROS 2 integration, physics simulation, and AI training environments for educational purposes.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

Use a multi-simulation environment approach with ROS 2 Humble Hawksbill, Gazebo Garden, Unity 2023.2 LTS, and NVIDIA Isaac Sim 2023.2 for the Physical AI & Humanoid Robotics educational content.

- ROS 2 Distribution: Humble Hawksbill (LTS version)
- Physics Simulation: Gazebo Garden for realistic physics and sensor simulation
- Visualization: Unity 2023.2 LTS for advanced graphics and visualization
- AI Training Environment: NVIDIA Isaac Sim 2023.2 for reinforcement learning and perception tasks
- Containerization: Docker for consistent development and deployment environments

## Consequences

### Positive

- Comprehensive coverage of industry-standard tools used in robotics research and development
- Different simulation environments serve specific purposes: Gazebo for physics, Unity for visualization, Isaac for NVIDIA-specific AI integration
- Support for real-world robotics development workflows
- Access to extensive documentation and community resources for each platform
- Ability to demonstrate different aspects of humanoid robotics in appropriate environments
- Integration with existing robotics frameworks and libraries

### Negative

- Increased complexity for students learning multiple simulation environments
- Higher system requirements and setup complexity
- Potential compatibility issues between different simulation platforms
- Need for multiple licensing considerations (Unity, NVIDIA Isaac Sim)
- Steep learning curve requiring familiarity with multiple tools
- Potential for fragmented learning experience if not properly integrated

## Alternatives Considered

Alternative Stack A: Single simulation environment (e.g., only Gazebo + ROS 2)
- Why rejected: Would limit educational scope and not cover industry-standard tools like Unity for visualization or NVIDIA Isaac for AI training

Alternative Stack B: Web-based simulation (e.g., Webots in browser)
- Why rejected: Would not provide the same level of realism and industry-relevant experience as native simulation tools

Alternative Stack C: Custom simulation framework
- Why rejected: Would require significant development effort and not leverage existing industry-standard tools and community support

## References

- Feature Spec: ./specs/1-physical-ai-book/spec.md
- Implementation Plan: ./specs/1-physical-ai-book/plan.md
- Related ADRs:
- Evaluator Evidence: ./history/prompts/1-physical-ai-book/