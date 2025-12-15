---
id: "125"
title: "Redesign Landing Page Navbar Black Red Cyber UI"
stage: "tasks"
date: "2025-12-13"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "1-physical-ai-book"
user: "unknown"
command: "/sp.phr"
labels:
  - "ui-redesign"
  - "landing-page"
  - "navbar"
  - "docusaurus"
  - "css-modules"
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "specs/1-physical-ai-book/tasks.md"
tests: []
---

# Redesign Landing Page & Navbar (Black & Red Cyber UI) - PHR

## PROMPT TEXT

```
id: T125 title: Redesign Landing Page & Navbar (Black & Red Cyber UI)
context:
  project: Physical AI & Humanoid Robotics Book
  framework: Docusaurus (React)
  existingTasksFile: tasks.md
  phase: Phase 14 – Polish & Cross-Cutting Concerns
  note: This task refines UI only; no content or backend changes.
scope:
  modifyOnly:
    - src/components/LandingPage.tsx
    - src/components/LandingPage.module.css
    - Docusaurus navbar styling/config
  doNotTouch:
    - docs content
    - sidebars.js structure
    - backend/auth logic
    - build & deployment pipelines
theme:
  primary: Black
  accent: Neon Red
  defaultMode: Dark
  optionalMode: Light (red accents only)
navbar:
  items:
    - Home
    - Modules
    - ThemeToggle
    - AuthButton
  style:
    glassmorphism: true
    background: rgba(0,0,0,0.55)
    blur: 14px
    sticky: true
    hoverGlow: red
  themeToggle:
    useDocusaurusColorMode: true
    icons: sun/moon
    animated: true
  authButton:
    uiOnly: true
    states:
      loggedOut: Sign In
      loggedIn: Logout

heroSection:
  height: 100vh
  layout: twoColumn
  left:
    image:
      source: static folder
      examplePath: /img/robot.png
      glow: red neon
  right:
    title:
      textGlow: red neon
    subtitle:
      subtleGlow: true
    button:
      text: Explore
      link: /docs/intro
      style: glass + red glow

modulesSection:
  sourceOfTruth: sidebars.js → tutorialSidebar → Modules
  displayOnly:
    - Foundations
    - ROS 2 + URDF
    - Digital Twin
    - AI Robot Brain
  exclude:
    - Capstone Project
  cards:
    style: glassmorphism
    hover:
      glow: red
      lift: true
    content:
      moduleName: exact
      chapters: exact (slug → human readable)
  grid:
    desktop: 2x2
    tablet: 2x1
    mobile: 1x4
  navigation:
    onClick: /docs/modules/<module-slug>/intro

rules:
  - React functional components only
  - CSS Modules only
  - No Tailwind
  - No UI libraries
  - No importing sidebars.js at runtime
  - No fake module or chapter names

verification:
  runLocal:
    command: npm run start
    url: http://localhost:3000
  confirm:
    - navbar glass + red glow
    - light/dark toggle works
    - auth button state changes visually
    - hero image loads from static folder
    - modules match sidebar
    - responsive layout works

deliverables:
  - Updated LandingPage.tsx
  - Updated LandingPage.module.css
  - Navbar override CSS/config
  - Confirmation localhost works
  - Short change summary
```

## RESPONSE TEXT

Successfully generated tasks.md file with 36 checklist-format tasks (T125-T160) for the landing page and navbar redesign with black and red cyber UI theme. The tasks cover all requirements including two-column layout, glassmorphism effects, red neon glow styling, responsive design, and proper module display.

## OUTCOME & IMPACT

Created comprehensive task list for UI redesign following the black and red cyber theme. Tasks are properly formatted with checklist format, include parallelization markers where appropriate, and cover all specified requirements from the user input. The tasks will enable systematic implementation of the new UI design.

## TESTS SUMMARY

none

## FILES SUMMARY

- specs/1-physical-ai-book/tasks.md

## NEXT PROMPTS

none

## REFLECTION NOTE

The task generation process successfully created detailed, checklist-format tasks that align with all specified requirements for the black and red cyber UI redesign. The parallelization markers [P] were appropriately added to tasks that can be worked on independently.

## FAILURE MODES OBSERVED

none

## NEXT EXPERIMENT TO IMPROVE PROMPT QUALITY

none