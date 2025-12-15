---
sidebar_position: 6
---

# ROS 2 Exercises

This section provides hands-on exercises to reinforce the ROS 2 concepts covered in this module. These exercises build upon each other and should help you gain practical experience with ROS 2 development.

## Exercise 1: Basic Publisher-Subscriber

### Objective
Create a simple publisher-subscriber pair to understand ROS 2 communication patterns.

### Steps
1. Create a new ROS 2 package:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python exercise_1_pub_sub --dependencies rclpy std_msgs
   ```

2. Create a publisher script (`exercise_1_pub_sub/exercise_1_pub_sub/publisher.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String


   class MinimalPublisher(Node):

       def __init__(self):
           super().__init__('minimal_publisher')
           self.publisher_ = self.create_publisher(String, 'chatter', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Hello World: {self.i}'
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

3. Create a subscriber script (`exercise_1_pub_sub/exercise_1_pub_sub/subscriber.py`):
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
               'chatter',
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

4. Update the package setup file (`exercise_1_pub_sub/setup.py`):
   ```python
   from setuptools import find_packages
   from setuptools import setup

   package_name = 'exercise_1_pub_sub'

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
       description='Exercise 1: Basic Publisher-Subscriber',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'publisher = exercise_1_pub_sub.publisher:main',
               'subscriber = exercise_1_pub_sub.subscriber:main',
           ],
       },
   )
   ```

5. Build and run the publisher and subscriber in separate terminals:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select exercise_1_pub_sub
   source install/setup.bash

   # Terminal 1
   ros2 run exercise_1_pub_sub publisher

   # Terminal 2
   ros2 run exercise_1_pub_sub subscriber
   ```

### Expected Output
- Publisher should print "Publishing: 'Hello World: X'" every 0.5 seconds
- Subscriber should print "I heard: 'Hello World: X'" in response

### Validation Criteria
- [ ] Publisher and subscriber communicate successfully
- [ ] Messages are transmitted and received correctly
- [ ] Both nodes run without errors

## Exercise 2: Service Client-Server

### Objective
Create a service server and client to understand request-response communication.

### Steps
1. Create a new ROS 2 package:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python exercise_2_service --dependencies rclpy example_interfaces
   ```

2. Create a service server (`exercise_2_service/exercise_2_service/service_server.py`):
   ```python
   #!/usr/bin/env python3

   from example_interfaces.srv import AddTwoInts
   import rclpy
   from rclpy.node import Node


   class MinimalService(Node):

       def __init__(self):
           super().__init__('minimal_service')
           self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

       def add_two_ints_callback(self, request, response):
           response.sum = request.a + request.b
           self.get_logger().info(f'Returning {response.sum}')
           return response


   def main(args=None):
       rclpy.init(args=args)
       minimal_service = MinimalService()
       rclpy.spin(minimal_service)
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. Create a service client (`exercise_2_service/exercise_2_service/service_client.py`):
   ```python
   #!/usr/bin/env python3

   from example_interfaces.srv import AddTwoInts
   import rclpy
   from rclpy.node import Node


   class MinimalClientAsync(Node):

       def __init__(self):
           super().__init__('minimal_client_async')
           self.cli = self.create_client(AddTwoInts, 'add_two_ints')
           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Service not available, waiting again...')
           self.req = AddTwoInts.Request()

       def send_request(self, a, b):
           self.req.a = a
           self.req.b = b
           self.future = self.cli.call_async(self.req)
           rclpy.spin_until_future_complete(self, self.future)
           return self.future.result()


   def main(args=None):
       rclpy.init(args=args)
       minimal_client = MinimalClientAsync()
       response = minimal_client.send_request(1, 2)
       minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
       minimal_client.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. Update the package setup file (`exercise_2_service/setup.py`):
   ```python
   from setuptools import find_packages
   from setuptools import setup

   package_name = 'exercise_2_service'

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
       description='Exercise 2: Service Client-Server',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'service_server = exercise_2_service.service_server:main',
               'service_client = exercise_2_service.service_client:main',
           ],
       },
   )
   ```

5. Build and run the service:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select exercise_2_service
   source install/setup.bash

   # Terminal 1
   ros2 run exercise_2_service service_server

   # Terminal 2
   ros2 run exercise_2_service service_client
   ```

### Expected Output
- Service server should respond to requests with the sum
- Service client should print the result of the addition

### Validation Criteria
- [ ] Service server runs without errors
- [ ] Service client successfully calls the service
- [ ] Correct sum is returned and displayed

## Exercise 3: URDF Robot Model

### Objective
Create a simple URDF robot model and visualize it in RViz.

### Steps
1. Create a URDF file (`simple_robot.urdf`):
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_robot">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <box size="0.5 0.3 0.2"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 0.8 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.5 0.3 0.2"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1"/>
         <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Wheel links -->
     <joint name="left_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="left_wheel"/>
       <origin xyz="0.2 -0.2 -0.1" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>

     <link name="left_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
       </inertial>
     </link>

     <joint name="right_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="right_wheel"/>
       <origin xyz="0.2 0.2 -0.1" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>

     <link name="right_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
       </inertial>
     </link>
   </robot>
   ```

2. Create a launch file to visualize the robot (`visualize_robot.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   import os
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Get the URDF file path
       urdf_file = os.path.join(
           get_package_share_directory('exercise_2_service'),  # Use any existing package
           'urdf',
           'simple_robot.urdf'
       )

       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation time if true'
       )

       # Robot state publisher node
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           parameters=[{
               'use_sim_time': LaunchConfiguration('use_sim_time'),
               'robot_description': open(urdf_file).read()
           }]
       )

       # Joint state publisher (for visualization)
       joint_state_publisher = Node(
           package='joint_state_publisher',
           executable='joint_state_publisher',
           name='joint_state_publisher',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
       )

       # RViz node
       rviz = Node(
           package='rviz2',
           executable='rviz2',
           name='rviz2'
       )

       return LaunchDescription([
           use_sim_time,
           robot_state_publisher,
           joint_state_publisher,
           rviz
       ])
   ```

3. Create a directory for URDF files and save the URDF:
   ```bash
   mkdir -p ~/robotics_ws/src/exercise_2_service/urdf
   # Save the URDF content to ~/robotics_ws/src/exercise_2_service/urdf/simple_robot.urdf
   ```

4. Launch the visualization:
   ```bash
   # Source your ROS 2 environment
   source /opt/ros/humble/setup.bash
   ros2 launch visualize_robot.launch.py
   ```

### Expected Output
- RViz should open and display the simple robot model
- Robot should appear with a base and two wheels

### Validation Criteria
- [ ] URDF file is properly formatted
- [ ] Robot model displays correctly in RViz
- [ ] All links and joints are visible

## Exercise 4: Launch File with Parameters

### Objective
Create a launch file that starts multiple nodes with configurable parameters.

### Steps
1. Create a parameter file (`config/robot_config.yaml`):
   ```yaml
   /**:
     ros__parameters:
       use_sim_time: false

   robot_controller:
     ros__parameters:
       max_linear_velocity: 1.0
       max_angular_velocity: 1.57
       wheel_radius: 0.05
       wheel_separation: 0.3
   ```

2. Create a launch file (`launch/robot_system.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   import os
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation time if true'
       )

       # Get parameter file path
       param_file = os.path.join(
           get_package_share_directory('exercise_2_service'),  # Use any existing package
           'config',
           'robot_config.yaml'
       )

       # Robot controller node
       robot_controller = Node(
           package='turtlebot3_node',  # Example package
           executable='turtlebot3_ros',  # Example executable
           name='robot_controller',
           parameters=[
               param_file,
               {'use_sim_time': LaunchConfiguration('use_sim_time')}
           ],
           remappings=[
               ('/cmd_vel', '/cmd_vel'),
               ('/odom', '/odom')
           ]
       )

       # Robot state publisher
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
       )

       # Joint state publisher
       joint_state_publisher = Node(
           package='joint_state_publisher',
           executable='joint_state_publisher',
           name='joint_state_publisher',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
       )

       return LaunchDescription([
           use_sim_time,
           robot_controller,
           robot_state_publisher,
           joint_state_publisher
       ])
   ```

3. Launch the system:
   ```bash
   # Create config directory and save the YAML file
   mkdir -p ~/robotics_ws/src/exercise_2_service/config
   # Save the YAML content to the config file

   # Launch the system
   ros2 launch robot_system.launch.py use_sim_time:=false
   ```

### Expected Output
- Multiple nodes should start simultaneously
- Parameters should be loaded from the YAML file
- System should run without errors

### Validation Criteria
- [ ] Launch file starts all specified nodes
- [ ] Parameters are loaded correctly from YAML file
- [ ] No errors during launch

## Exercise 5: Action Server and Client

### Objective
Implement a navigation action that moves a simulated robot to a goal pose.

### Steps
1. Create a custom action definition file (`action/NavigateToPose.action`):
   ```
   # Define goal
   geometry_msgs/PoseStamped pose
   float32 tolerance
   ---
   # Define result
   bool success
   string message
   nav_msgs/Path path
   ---
   # Define feedback
   geometry_msgs/PoseStamped current_pose
   float32 distance_remaining
   int32 percent_complete
   ```

2. Create an action server (`exercise_5_action/exercise_5_action/navigation_action_server.py`):
   ```python
   #!/usr/bin/env python3

   import time
   import rclpy
   from rclpy.action import ActionServer, GoalResponse, CancelResponse
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped
   from nav_msgs.msg import Path

   # Import your custom action (you would need to create this)
   # from custom_interfaces.action import NavigateToPose


   # For this example, we'll use the Fibonacci action as a placeholder
   from example_interfaces.action import Fibonacci


   class NavigationActionServer(Node):

       def __init__(self):
           super().__init__('navigation_action_server')

           # Create action server (using Fibonacci as placeholder)
           self._action_server = ActionServer(
               self,
               Fibonacci,
               'navigate_to_pose',  # Placeholder name
               execute_callback=self.execute_callback,
               goal_callback=self.goal_callback,
               cancel_callback=self.cancel_callback)

       def goal_callback(self, goal_request):
           """Accept or reject a client request to begin an action."""
           self.get_logger().info('Received navigation goal request')
           return GoalResponse.ACCEPT

       def cancel_callback(self, goal_handle):
           """Accept or reject a client request to cancel an action."""
           self.get_logger().info('Received cancel request for navigation')
           return CancelResponse.ACCEPT

       async def execute_callback(self, goal_handle):
           """Execute the navigation task."""
           self.get_logger().info('Executing navigation goal...')

           # Placeholder feedback
           feedback_msg = Fibonacci.Feedback()
           feedback_msg.sequence = [0, 1]

           # Simulate navigation progress
           for i in range(1, goal_handle.request.order):
               # Check if there's a cancel request
               if goal_handle.is_cancel_requested:
                   goal_handle.canceled()
                   self.get_logger().info('Navigation goal canceled')
                   return Fibonacci.Result()

               # Update feedback
               feedback_msg.sequence.append(
                   feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

               # Publish feedback
               goal_handle.publish_feedback(feedback_msg)
               self.get_logger().info(f'Navigation progress: {feedback_msg.sequence}')

               # Simulate processing time
               time.sleep(0.5)

           # Check if goal was canceled during execution
           if goal_handle.is_cancel_requested:
               goal_handle.canceled()
               self.get_logger().info('Navigation goal canceled')
               return Fibonacci.Result()

           # Set result and return
           goal_handle.succeed()
           result = Fibonacci.Result()
           result.sequence = feedback_msg.sequence
           self.get_logger().info(f'Navigation completed with result: {result.sequence}')

           return result


   def main(args=None):
       rclpy.init(args=args)
       navigation_action_server = NavigationActionServer()
       rclpy.spin(navigation_action_server)
       navigation_action_server.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. Create an action client (`exercise_5_action/exercise_5_action/navigation_action_client.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.action import ActionClient
   from rclpy.node import Node

   # Import your custom action (you would need to create this)
   # from custom_interfaces.action import NavigateToPose

   # For this example, we'll use the Fibonacci action as a placeholder
   from example_interfaces.action import Fibonacci


   class NavigationActionClient(Node):

       def __init__(self):
           super().__init__('navigation_action_client')
           self._action_client = ActionClient(self, Fibonacci, 'navigate_to_pose')

       def send_goal(self, order):
           # Wait for the action server to be available
           self._action_client.wait_for_server()

           # Create a goal message
           goal_msg = Fibonacci.Goal()
           goal_msg.order = order

           # Send the goal and register callbacks
           self._send_goal_future = self._action_client.send_goal_async(
               goal_msg,
               feedback_callback=self.feedback_callback)

           self._send_goal_future.add_done_callback(self.goal_response_callback)

       def goal_response_callback(self, future):
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Navigation goal rejected')
               return

           self.get_logger().info('Navigation goal accepted')

           # Request the result
           self._get_result_future = goal_handle.get_result_async()
           self._get_result_future.add_done_callback(self.get_result_callback)

       def feedback_callback(self, feedback_msg):
           feedback = feedback_msg.feedback
           self.get_logger().info(f'Navigation feedback: {feedback.sequence}')

       def get_result_callback(self, future):
           result = future.result().result
           self.get_logger().info(f'Navigation result: {result.sequence}')
           rclpy.shutdown()


   def main(args=None):
       rclpy.init(args=args)

       action_client = NavigationActionClient()

       # Send a navigation goal (order 10 as an example)
       action_client.send_goal(10)

       rclpy.spin(action_client)


   if __name__ == '__main__':
       main()
   ```

4. Update the package setup file and build:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select exercise_5_action
   source install/setup.bash

   # Terminal 1
   ros2 run exercise_5_action navigation_action_server

   # Terminal 2
   ros2 run exercise_5_action navigation_action_client
   ```

### Expected Output
- Action server should accept and execute goals
- Action client should receive feedback during execution
- Result should be returned upon completion

### Validation Criteria
- [ ] Action server runs without errors
- [ ] Action client successfully sends goal
- [ ] Feedback is received during execution
- [ ] Result is received upon completion

## Exercise 6: Parameter Management

### Objective
Create a node that uses parameters for configuration and responds to parameter changes.

### Steps
1. Create a parameter management node (`exercise_6_params/exercise_6_params/parameter_manager.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from rcl_interfaces.msg import ParameterDescriptor, ParameterType
   from rcl_interfaces.msg import SetParametersResult


   class ParameterManagerNode(Node):

       def __init__(self):
           super().__init__('parameter_manager_node')

           # Define parameter descriptors with constraints
           max_velocity_desc = ParameterDescriptor()
           max_velocity_desc.description = 'Maximum linear velocity in m/s'
           max_velocity_desc.type = ParameterType.PARAMETER_DOUBLE
           max_velocity_desc.floating_point_range = [
               {'from_value': 0.1, 'to_value': 5.0, 'step': 0.01}
           ]

           robot_name_desc = ParameterDescriptor()
           robot_name_desc.description = 'Name of the robot'
           robot_name_desc.type = ParameterType.PARAMETER_STRING

           # Declare parameters with descriptors
           self.declare_parameter('max_linear_velocity', 1.0, max_velocity_desc)
           self.declare_parameter('robot_name', 'default_robot', robot_name_desc)
           self.declare_parameter('enable_control', True)

           # Register parameter callback
           self.add_on_set_parameters_callback(self.validate_parameters)

           # Initialize properties
           self.update_properties()

           # Timer to periodically display current parameters
           self.timer = self.create_timer(2.0, self.display_parameters)

       def validate_parameters(self, params):
           """Validate parameter changes."""
           result = SetParametersResult()
           result.successful = True

           for param in params:
               if param.name == 'max_linear_velocity':
                   if param.value <= 0:
                       result.successful = False
                       result.reason = 'Max velocity must be positive'
                       return result
                   elif param.value > 5.0:
                       result.successful = False
                       result.reason = 'Max velocity must be <= 5.0'
                       return result

               elif param.name == 'robot_name' and not isinstance(param.value, str):
                   result.successful = False
                   result.reason = 'Robot name must be a string'
                   return result

           # If validation passes, update properties
           self.update_properties()
           return result

       def update_properties(self):
           """Update internal properties based on parameters."""
           self.max_velocity = self.get_parameter('max_linear_velocity').value
           self.robot_name = self.get_parameter('robot_name').value
           self.enable_control = self.get_parameter('enable_control').value

           self.get_logger().info(f'Updated configuration for {self.robot_name}')
           self.get_logger().info(f'  Max velocity: {self.max_velocity} m/s')
           self.get_logger().info(f'  Control enabled: {self.enable_control}')

       def display_parameters(self):
           """Display current parameter values."""
           self.get_logger().info(
               f'Current config - Robot: {self.robot_name}, '
               f'Max Vel: {self.max_velocity}, Control: {self.enable_control}'
           )


   def main(args=None):
       rclpy.init(args=args)
       node = ParameterManagerNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. Test parameter changes at runtime:
   ```bash
   # Terminal 1: Start the parameter manager
   ros2 run exercise_6_params parameter_manager

   # Terminal 2: Change parameters
   ros2 param set /parameter_manager_node max_linear_velocity 2.5
   ros2 param set /parameter_manager_node robot_name "TestRobot"
   ros2 param get /parameter_manager_node robot_name
   ```

### Expected Output
- Node should start with default parameter values
- Parameter changes should be validated and applied
- Node should respond appropriately to parameter updates

### Validation Criteria
- [ ] Node starts with default parameters
- [ ] Parameter validation works correctly
- [ ] Parameter changes are applied successfully
- [ ] Invalid parameter values are rejected

## Exercise Completion Checklist

After completing these exercises, you should be able to:
- [ ] Create and run basic publisher-subscriber nodes
- [ ] Implement service client-server communication
- [ ] Create and visualize URDF robot models
- [ ] Use launch files to start complex systems
- [ ] Implement action servers and clients
- [ ] Manage parameters in ROS 2 nodes

## Next Steps

Once you've successfully completed these exercises with at least 90% accuracy, you're ready to move on to the next module covering Digital Twin technologies (Gazebo, Unity, Isaac Sim). The exercises in this module have provided you with hands-on experience with the core ROS 2 concepts needed for humanoid robotics development.