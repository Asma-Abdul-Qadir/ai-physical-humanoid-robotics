# ROS 2 Module Tests

This document outlines tests to validate the ROS 2 + URDF module learning objectives with a 90% accuracy requirement.

## Module Overview

The ROS 2 + URDF module covers:
- ROS 2 installation and basic concepts
- Nodes, topics, and services communication
- URDF modeling for humanoid robots
- Launch files configuration
- Actions and parameters management
- Practical exercises and integration

## Test Structure

### Knowledge Assessment (40% of total grade)
- Multiple choice questions
- True/false questions
- Fill-in-the-blank questions

### Practical Assessment (40% of total grade)
- Code implementation tasks
- Configuration exercises
- Problem-solving scenarios

### Integration Assessment (20% of total grade)
- Complete system implementation
- Debugging and troubleshooting
- Performance evaluation

## Test 1: ROS 2 Fundamentals Knowledge

### Objective
Assess understanding of core ROS 2 concepts and architecture.

### Questions

1. **Multiple Choice**: What does ROS 2 stand for?
   a) Robot Operating System 2
   b) Robot Operation System 2
   c) Robotic Operating System 2
   d) Robot Operating Service 2

   **Answer**: a) Robot Operating System 2

2. **True/False**: ROS 2 uses a master-based architecture like ROS 1.
   **Answer**: False (ROS 2 uses a decentralized architecture)

3. **Multiple Choice**: Which DDS implementation is commonly used with ROS 2 Humble Hawksbill?
   a) OpenSplice
   b) Fast DDS
   c) RTI Connext
   d) All of the above

   **Answer**: d) All of the above

4. **Fill-in-the-blank**: The command to list all active ROS 2 nodes is `_______`.
   **Answer**: ros2 node list

5. **Multiple Choice**: What is the primary difference between a topic and a service in ROS 2?
   a) Topics are faster than services
   b) Topics provide asynchronous communication, services provide synchronous communication
   c) Topics can only send integers, services can send any data type
   d) There is no difference

   **Answer**: b) Topics provide asynchronous communication, services provide synchronous communication

### Scoring
- Each question: 5 points
- Total: 25 points
- Passing: 23 points (92%) for this section

## Test 2: Nodes, Topics, and Services Practical

### Objective
Assess ability to create and implement ROS 2 communication patterns.

### Tasks

1. **Node Creation (10 points)**
   Create a ROS 2 node in Python that:
   - Has a unique name
   - Logs a startup message
   - Runs continuously until interrupted

2. **Publisher Implementation (15 points)**
   Create a publisher that:
   - Publishes String messages to topic "my_topic"
   - Publishes at 1 Hz
   - Sends incrementing messages ("Message 1", "Message 2", etc.)

3. **Subscriber Implementation (15 points)**
   Create a subscriber that:
   - Subscribes to "my_topic"
   - Logs received messages
   - Handles messages properly

4. **Service Server (10 points)**
   Create a service server that:
   - Implements the AddTwoInts service
   - Returns the sum of two integers
   - Logs the request and response

5. **Service Client (10 points)**
   Create a service client that:
   - Calls the AddTwoInts service
   - Passes two integer values
   - Logs the result

### Scoring
- Total: 60 points
- Passing: 54 points (90%) for this section

## Test 3: URDF Modeling Assessment

### Objective
Assess understanding of URDF syntax and humanoid robot modeling.

### Tasks

1. **URDF Structure (10 points)**
   Identify and correct errors in the following URDF snippet:
   ```xml
   <robot name="test_robot">
     <link name="base_link">
       <visual>
         <geometry>
           <box size="1 1 1"/>
         </geometry>
       </visual>
     </link>
     <!-- Missing joint definition -->
   </robot>
   ```

2. **Link Definition (15 points)**
   Create a URDF link definition that includes:
   - Visual properties with a sphere geometry
   - Collision properties matching the visual
   - Inertial properties with mass and moments of inertia

3. **Joint Creation (15 points)**
   Define a revolute joint that connects two links with:
   - Proper parent-child relationship
   - Axis of rotation
   - Joint limits
   - Proper origin transformation

4. **Humanoid Arm (20 points)**
   Create a simple humanoid arm with:
   - Shoulder, elbow, and wrist joints
   - Proper kinematic chain
   - Appropriate link dimensions
   - Realistic joint limits

### Scoring
- Total: 60 points
- Passing: 54 points (90%) for this section

## Test 4: Launch Files Configuration

### Objective
Assess ability to create and use ROS 2 launch files.

### Tasks

1. **Basic Launch File (10 points)**
   Create a Python launch file that starts:
   - One publisher node
   - One subscriber node
   - Uses proper launch description structure

2. **Parameter Launch (15 points)**
   Create a launch file that:
   - Sets parameters for nodes
   - Uses a parameter YAML file
   - Passes parameters to nodes correctly

3. **Launch Arguments (10 points)**
   Create a launch file that:
   - Defines at least 2 launch arguments
   - Uses arguments to customize node behavior
   - Has proper default values

4. **Complex Launch System (25 points)**
   Create a launch file that:
   - Starts 3+ different types of nodes
   - Includes robot state publisher
   - Sets up visualization (RViz)
   - Uses proper parameter configuration
   - Includes conditional launching

### Scoring
- Total: 60 points
- Passing: 54 points (90%) for this section

## Test 5: Actions and Parameters Assessment

### Objective
Assess understanding of advanced ROS 2 communication patterns.

### Questions and Tasks

1. **Action Concepts (10 points)**
   **Short Answer**: Explain the difference between a service and an action in ROS 2. When would you use each?

2. **Action Server Implementation (20 points)**
   Create an action server that:
   - Implements a Fibonacci action
   - Provides feedback during execution
   - Handles cancellation requests
   - Returns proper results

3. **Action Client Implementation (15 points)**
   Create an action client that:
   - Sends a goal to the Fibonacci action
   - Handles feedback appropriately
   - Receives and processes results

4. **Parameter Management (15 points)**
   Create a node that:
   - Declares at least 3 different parameter types
   - Validates parameter changes
   - Responds to parameter updates

### Scoring
- Total: 60 points
- Passing: 54 points (90%) for this section

## Test 6: Integration Challenge

### Objective
Assess ability to integrate all ROS 2 concepts into a complete system.

### Scenario
Create a simulated robot system that includes:
- A URDF model of a simple robot
- A launch file that starts all required nodes
- A controller node that uses parameters for configuration
- A navigation system using actions
- Proper communication between all components

### Requirements
1. **URDF Model (15 points)**: Create a robot with at least 3 links and 2 joints
2. **Launch System (20 points)**: Create launch files to start the complete system
3. **Parameter Configuration (10 points)**: Use parameters for robot configuration
4. **Action Integration (15 points)**: Implement a navigation action
5. **Communication (10 points)**: Proper topic/service communication
6. **Documentation (5 points)**: Comment code and provide usage instructions

### Scoring
- Total: 75 points
- Passing: 68 points (90%) for this section

## Grading Rubric

### Overall Module Grade
- Knowledge Assessment: 40% (100 points possible, 92 points needed)
- Practical Assessment: 40% (120 points possible, 108 points needed)
- Integration Assessment: 20% (75 points possible, 68 points needed)
- **Total: 295 points possible, 268 points needed for 90%**

### Grade Scale
- A: 266-295 points (90-100%)
- B: 236-265 points (80-89%)
- C: 206-235 points (70-79%)
- D: 177-205 points (60-69%)
- F: Below 177 points (<60%)

## Sample Test Questions with Solutions

### Question 1: Topic Communication
**Question**: Write a Python publisher that sends Twist messages (geometry_msgs/msg/Twist) to the "/cmd_vel" topic at 10 Hz.

**Solution**:
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelPublisher(Node):

    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.5  # Move forward at 0.5 m/s
        msg.angular.z = 0.2  # Turn left at 0.2 rad/s
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher()
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Question 2: Parameter Validation
**Question**: Explain how to validate parameters in a ROS 2 node and provide an example that ensures a parameter is within a specific range.

**Solution**:
```python
def parameter_callback(self, params):
    result = SetParametersResult()
    result.successful = True

    for param in params:
        if param.name == 'max_velocity':
            if param.value < 0.0 or param.value > 5.0:
                result.successful = False
                result.reason = 'Max velocity must be between 0.0 and 5.0'
                return result

    return result

# Register the callback
self.add_on_set_parameters_callback(self.parameter_callback)
```

## Assessment Guidelines

### For Instructors
- Allow 3 hours for the complete assessment
- Provide access to ROS 2 documentation
- Students may use their own code from exercises as reference
- Focus on understanding rather than memorization

### For Students
- Review all module content before attempting assessment
- Practice with exercises provided in the module
- Understand concepts rather than memorizing code
- Test all implementations in a ROS 2 environment

## Remediation Plan

### For Failed Sections
If a student scores below 90% on any section:

1. **Knowledge Assessment**: Review core concepts and retake quiz
2. **Practical Assessment**: Complete additional hands-on exercises
3. **Integration Assessment**: Work through step-by-step tutorials

### Additional Resources
- ROS 2 official tutorials
- Community forums and documentation
- Practice exercises with solutions
- Video demonstrations of concepts

## Key Takeaways

- Students must demonstrate 90% proficiency across all areas
- Practical implementation is weighted heavily
- Integration of multiple concepts is essential
- Continuous learning and practice are required for mastery