---
id: 137
title: "Update Docusaurus Config and Add Course Modules"
stage: general
date: 2025-12-16
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: Claude
command: /sp.git.commit_pr
labels: ["git", "docusaurus", "documentation", "modules", "configuration"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files: |
  - docusaurus.config.js
  - docs/modules/ai-robot-brain/*
  - docs/modules/capstone/*
  - docs/modules/digital-twin/*
  - docs/modules/foundations/*
  - docs/modules/ros2-urdf/*
  - .github/workflows/*
  - history/adr/*
  - src/components/*
  - src/pages/*
  - static/logo.png
tests: |
  - None
---

## Prompt Text

/sp.git.commit_pr

## Response Text

Successfully executed git workflow to commit and push changes to main branch:

- Updated docusaurus.config.js with dark mode support and new navigation routes
- Added comprehensive course modules (Foundations, ROS 2 + URDF, Digital Twin, AI Robot Brain)
- Included GitHub Actions workflows for accessibility, build, deploy, and link checking
- Added Architectural Decision Records (ADRs)
- Created source code components and static assets
- Removed unnecessary image files
- Fixed the 'nul' file issue on Windows
- Committed with descriptive message and pushed to remote main branch

## Outcome

The repository has been successfully updated with the new course content and configuration changes. All changes were committed and pushed to the main branch.