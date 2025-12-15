# Foundations Module Tests

This document outlines tests to validate the Foundations module learning objectives.

## Test 1: Hardware Requirements Understanding
**Objective**: Verify learner understands different hardware tiers and their use cases.

### Questions:
1. What are the three tiers of hardware requirements?
2. Which tier would be most appropriate for running NVIDIA Isaac Sim?
3. Why is Ubuntu 22.04 LTS the primary supported OS?

### Expected Answers:
1. Basic (8GB RAM, GTX 1660), Recommended (16GB RAM, RTX 3060), Optimal (32GB+ RAM, RTX 4080)
2. Recommended or Optimal tier
3. Because it's the primary development and testing platform with best compatibility

## Test 2: Software Prerequisites Setup
**Objective**: Verify learner can set up the required software environment.

### Tasks:
1. Install ROS 2 Humble Hawksbill
2. Install Gazebo Garden
3. Verify Node.js and NPM installation
4. Verify Docker installation

### Expected Results:
- ROS 2 installation verified with `ros2 topic list`
- Gazebo runs without errors
- Node.js and NPM display version numbers
- Docker runs hello-world container successfully

## Test 3: Basic Robotics Concepts
**Objective**: Verify understanding of fundamental robotics principles.

### Questions:
1. What are the three key characteristics of a robot?
2. Explain the difference between forward and inverse kinematics.
3. What are the three components of PID control?

### Expected Answers:
1. Sensing, processing, actuation
2. Forward: given joint angles, find end effector position; Inverse: given desired end effector position, find joint angles
3. Proportional, Integral, Derivative

## Test 4: Safety and Ethics Principles
**Objective**: Verify understanding of safety and ethical considerations.

### Questions:
1. What are the four core safety principles?
2. Name two ethical principles for robotics.
3. What does the acronym FMEA stand for?

### Expected Answers:
1. Inherently safe design, functional safety, risk assessment, safety standards compliance
2. Beneficence, non-maleficence, autonomy, justice
3. Failure Modes and Effects Analysis

## Test 5: Exercise Completion
**Objective**: Verify learner completed the module exercises successfully.

### Validation:
- [ ] Exercise 1 completed with all commands executing successfully
- [ ] Exercise 2 completed with correct coordinate transformations
- [ ] Exercise 3 completed with 5+ hazards identified and risk assessed
- [ ] Exercise 4 completed with successful ROS 2 publisher-subscriber communication
- [ ] Exercise 5 completed with reflective essay demonstrating concept connections

## Success Criteria
- [ ] Achieve 90% accuracy on all test questions
- [ ] Successfully complete all practical exercises
- [ ] Demonstrate understanding of safety considerations
- [ ] Show ability to connect foundational concepts to practical applications

## Assessment Rubric
- **Excellent (90-100%)**: All concepts understood, all exercises completed successfully, able to explain connections between concepts
- **Proficient (80-89%)**: Most concepts understood, nearly all exercises completed, good understanding of connections
- **Developing (70-79%)**: Core concepts understood, most exercises completed, basic understanding of connections
- **Beginning (Below 70%)**: Needs additional review and practice