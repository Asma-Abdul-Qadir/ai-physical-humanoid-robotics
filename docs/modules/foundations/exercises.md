---
sidebar_position: 6
---

# Foundations Module Exercises

This section provides exercises to reinforce the concepts covered in the Foundations module. These exercises are designed to be completed within 2-4 hours of focused work and should help validate your understanding of the material.

## Exercise 1: System Setup Verification

### Objective
Verify that your development environment is properly configured with all required tools.

### Steps
1. Check ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic list
   ```

2. Check Gazebo installation:
   ```bash
   gazebo --version
   gazebo
   ```

3. Check Node.js and NPM:
   ```bash
   node --version
   npm --version
   ```

4. Check Docker:
   ```bash
   docker run hello-world
   ```

5. Build the Docusaurus site:
   ```bash
   cd /path/to/physical-ai-humanoid-robotics-book
   npm install
   npm run build
   ```

### Expected Output
- ROS 2 should show an empty topic list (no errors)
- Gazebo should start without errors
- Node.js and NPM should display version numbers
- Docker should run the hello-world container successfully
- Docusaurus should build without errors

### Validation Criteria
- [ ] All commands execute without errors
- [ ] Docusaurus site builds successfully
- [ ] All required software versions match specifications

## Exercise 2: Coordinate System Understanding

### Objective
Understand how coordinate systems work in robotics and practice transforming between them.

### Background
In robotics, we use coordinate systems to describe positions and orientations of robot parts and objects in the environment.

### Steps
1. Create a simple ROS 2 package for coordinate transformations:
   ```bash
   mkdir -p ~/robotics_ws/src
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python coordinate_transform_demo
   cd coordinate_transform_demo
   ```

2. Create a simple Python script to demonstrate coordinate transformations:
   ```python
   #!/usr/bin/env python3
   import numpy as np
   from scipy.spatial.transform import Rotation as R

   def transform_point(point, translation, rotation_rpy):
       """Transform a point using translation and rotation (roll, pitch, yaw)"""
       # Create rotation matrix from roll, pitch, yaw
       rot = R.from_euler('xyz', rotation_rpy).as_matrix()

       # Apply rotation then translation
       transformed = np.dot(rot, point) + translation
       return transformed

   # Example: Transform a point from base frame to end-effector frame
   point_base = np.array([1.0, 0.0, 0.0])  # Point 1m in front of robot
   translation = np.array([0.1, 0.0, 0.5])  # End-effector is 0.1m right, 0.5m up from base
   rotation_rpy = np.array([0.0, 0.0, np.pi/2])  # Rotated 90 degrees around Z axis

   transformed_point = transform_point(point_base, translation, rotation_rpy)
   print(f"Original point: {point_base}")
   print(f"Transformed point: {transformed_point}")
   ```

3. Run the script and verify the transformation is correct.

### Expected Output
The script should output the transformed coordinates based on the rotation and translation.

### Validation Criteria
- [ ] Script runs without errors
- [ ] Transformation results match expectations
- [ ] Understanding of how rotation and translation affect coordinates

## Exercise 3: Safety Risk Assessment

### Objective
Practice identifying and assessing safety risks in a simple robotic system.

### Scenario
You are designing a simple humanoid robot that will operate in an office environment. The robot is 1.5m tall, weighs 50kg, has 2 DOF arms, and moves on wheels at up to 1 m/s.

### Steps
1. Identify at least 5 potential hazards:
   - Mechanical hazards (moving parts, pinch points)
   - Environmental hazards (falling, collisions)
   - Operational hazards (unexpected behaviors)

2. For each hazard, assess:
   - Probability (Low/Medium/High)
   - Severity (Minor/Moderate/Severe)
   - Risk level (Low/Medium/High)

3. Propose mitigation strategies for the highest risk hazards.

### Example Format
```
Hazard: [Description]
Probability: [Low/Medium/High]
Severity: [Minor/Moderate/Severe]
Risk Level: [Low/Medium/High]
Mitigation: [Strategy to reduce risk]
```

### Validation Criteria
- [ ] At least 5 hazards identified
- [ ] Risk assessment completed for each hazard
- [ ] Mitigation strategies proposed for high-risk hazards
- [ ] Consideration of both human and property safety

## Exercise 4: Basic ROS 2 Communication

### Objective
Create a simple publisher-subscriber pair to understand ROS 2 communication.

### Steps
1. Create a new ROS 2 package:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python simple_communication_demo
   cd simple_communication_demo/simple_communication_demo
   ```

2. Create a publisher script (`publisher.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class MinimalPublisher(Node):
       def __init__(self):
           super().__init__('minimal_publisher')
           self.publisher_ = self.create_publisher(String, 'robot_status', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Robot status: operational - {self.i}'
           self.publisher_.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1

   def main(args=None):
       rclpy.init(args=args)
       minimal_publisher = MinimalPublisher()
       rclpy.spin(minimal_publisher)
       minimal_publisher.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. Create a subscriber script (`subscriber.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class MinimalSubscriber(Node):
       def __init__(self):
           super().__init__('minimal_subscriber')
           self.subscription = self.create_subscription(
               String,
               'robot_status',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info(f'I heard: "{msg.data}"')

   def main(args=None):
       rclpy.init(args=args)
       minimal_subscriber = MinimalSubscriber()
       rclpy.spin(minimal_subscriber)
       minimal_subscriber.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. Update the `setup.py` file to make scripts executable:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'simple_communication_demo'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='your_name',
       maintainer_email='your_email@example.com',
       description='Simple ROS 2 publisher and subscriber demo',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'publisher = simple_communication_demo.publisher:main',
               'subscriber = simple_communication_demo.subscriber:main',
           ],
       },
   )
   ```

5. Build and run the publisher and subscriber in separate terminals:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select simple_communication_demo
   source install/setup.bash

   # Terminal 1
   ros2 run simple_communication_demo publisher

   # Terminal 2
   ros2 run simple_communication_demo subscriber
   ```

### Expected Output
The subscriber should receive and display messages published by the publisher.

### Validation Criteria
- [ ] Publisher and subscriber communicate successfully
- [ ] Messages are transmitted and received correctly
- [ ] Understanding of ROS 2 topic-based communication

## Exercise 5: Reflection and Integration

### Objective
Synthesize your understanding of the foundations module by connecting concepts.

### Steps
1. Write a brief essay (200-300 words) addressing:
   - How do the mathematical foundations (coordinate systems, transformations) enable safe robot operation?
   - What is the relationship between robot architecture (perception, planning, control) and safety considerations?
   - How do the software tools (ROS 2) facilitate the implementation of safety systems?

2. Identify which concepts from this module will be most important for the next module on ROS 2 and URDF.

### Validation Criteria
- [ ] Essay demonstrates understanding of connections between concepts
- [ ] Clear explanation of how mathematical foundations support safety
- [ ] Identification of key concepts for future learning

## Exercise Completion Checklist

After completing these exercises, you should be able to:
- [ ] Verify your development environment setup
- [ ] Understand coordinate system transformations
- [ ] Perform basic safety risk assessments
- [ ] Create simple ROS 2 publisher-subscriber communication
- [ ] Connect foundational concepts to practical applications

## Next Steps

Once you've successfully completed these exercises with at least 90% accuracy, you're ready to move on to the ROS 2 + URDF module, where you'll dive deeper into the Robot Operating System and robot description formats.