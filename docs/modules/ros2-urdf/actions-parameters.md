---
sidebar_position: 5
---

# ROS 2 Actions and Parameters

This section covers advanced ROS 2 communication patterns: actions for goal-oriented tasks and parameters for configuration management. These concepts are essential for building robust robotic systems.

## Learning Objectives

By the end of this section, you will be able to:
- Implement and use ROS 2 actions for long-running tasks with feedback
- Manage node parameters for configuration and runtime adjustment
- Create action clients and servers for complex robot behaviors
- Understand the lifecycle of action goals and parameter updates
- Apply actions and parameters to humanoid robot control scenarios

## ROS 2 Actions

Actions are a communication pattern in ROS 2 designed for long-running tasks that require feedback, goal management, and cancellation. Unlike services, which are synchronous, actions allow for continuous feedback and can be cancelled during execution.

### When to Use Actions

Actions are appropriate for tasks that:
- Take a long time to complete
- Need to provide feedback during execution
- May need to be cancelled before completion
- Require goal monitoring and status reporting

### Action Characteristics

- **Goal**: Request sent to start an action
- **Feedback**: Continuous updates during execution
- **Result**: Final outcome when action completes
- **Status**: Current state of the action goal

### Action Message Types

Each action definition includes three message types:
- **Goal**: Defines the goal request
- **Feedback**: Provides ongoing feedback
- **Result**: Contains the final result

## Action Server Implementation

### Python Action Server

```python
#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')

        # Create action server with callback group for reentrancy
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Notify that the goal is executing
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        # Simulate the action execution
        for i in range(1, goal_handle.request.order):
            # Check if there's a cancel request
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            # Update feedback
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

            # Simulate processing time
            time.sleep(1)

        # Check if goal was canceled during execution
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled')
            return Fibonacci.Result()

        # Set result and return
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result


def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    executor = MultiThreadedExecutor()
    executor.add_node(fibonacci_action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### C++ Action Server

```cpp
#include <memory>
#include <thread>

#include "example_interfaces/action/fibonacci.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace ros2_examples {
class FibonacciActionServer : public rclcpp::Node
{
public:
  using Fibonacci = example_interfaces::action::Fibonacci;
  using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

  explicit FibonacciActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("fibonacci_action_server", options)
  {
    using namespace std::placeholders;

    // Create action server
    this->action_server_ = rclcpp_action::create_server<Fibonacci>(
      this->get_node_base_interface(),
      this->get_node_clock_interface(),
      this->get_node_logging_interface(),
      this->get_node_waitables_interface(),
      "fibonacci",
      std::bind(&FibonacciActionServer::handle_goal, this, _1, _2),
      std::bind(&FibonacciActionServer::handle_cancel, this, _1),
      std::bind(&FibonacciActionServer::handle_accepted, this, _1));
  }

private:
  rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const Fibonacci::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleFibonacci> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
  {
    using namespace std::placeholders;
    // This needs to return quickly to avoid blocking the executor
    std::thread{std::bind(&FibonacciActionServer::execute, this, _1), goal_handle}.detach();
  }

  void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal");

    // Create feedback and result messages
    auto feedback = std::make_shared<Fibonacci::Feedback>();
    auto result = std::make_shared<Fibonacci::Result>();

    // Initialize the sequence
    feedback->sequence = {0, 1};

    auto goal = goal_handle->get_goal();

    // Simulate the action execution
    for (int i = 1; i < goal->order; ++i) {
      // Check if there is a cancel request
      if (goal_handle->is_canceling()) {
        result->sequence = feedback->sequence;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        return;
      }

      // Update the sequence
      feedback->sequence.push_back(
        feedback->sequence[i] + feedback->sequence[i - 1]);

      // Publish feedback
      goal_handle->publish_feedback(feedback);
      RCLCPP_INFO(this->get_logger(), "Publishing feedback");

      // Simulate processing time
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    // Check if goal was canceled during execution
    if (goal_handle->is_canceling()) {
      result->sequence = feedback->sequence;
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "Goal canceled");
      return;
    }

    // Set result and succeed
    result->sequence = feedback->sequence;
    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Goal succeeded");
  }
};  // class FibonacciActionServer

}  // namespace ros2_examples

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_examples::FibonacciActionServer)
```

## Action Client Implementation

### Python Action Client

```python
#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

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
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Request the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    action_client = FibonacciActionClient()

    # Send a goal
    action_client.send_goal(10)

    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
```

## Practical Action Example: Robot Navigation

Here's a practical example of using actions for robot navigation:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from custom_interfaces.action import NavigateToPose  # Custom action


class NavigationActionServer(Node):

    def __init__(self):
        super().__init__('navigation_action_server')

        # Create action server for navigation
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_navigate_callback,
            goal_callback=self.goal_navigate_callback)

        # Publishers for visualization
        self.path_publisher = self.create_publisher(Path, 'planned_path', 10)
        self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)

    def goal_navigate_callback(self, goal_request):
        """Validate the navigation goal."""
        # Check if the goal is valid (e.g., not in an obstacle)
        if self.is_valid_goal(goal_request.pose):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def is_valid_goal(self, pose):
        """Check if the goal pose is valid."""
        # Implementation would check if the pose is in free space
        # This is a simplified example
        return True

    async def execute_navigate_callback(self, goal_handle):
        """Execute the navigation task."""
        self.get_logger().info('Executing navigation goal...')

        # Extract goal pose
        goal_pose = goal_request.pose

        # Plan path to goal
        path = self.plan_path_to_pose(goal_pose)
        if path is None:
            goal_handle.abort()
            return NavigateToPose.Result(success=False, message="Failed to plan path")

        # Publish planned path for visualization
        self.path_publisher.publish(path)

        # Initialize feedback
        feedback_msg = NavigateToPose.Feedback()
        feedback_msg.current_pose = self.get_current_pose()
        feedback_msg.distance_remaining = self.calculate_distance(
            feedback_msg.current_pose, goal_pose)

        # Execute navigation
        while feedback_msg.distance_remaining > 0.1:  # 10cm tolerance
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavigateToPose.Result(
                    success=False, message="Navigation cancelled")

            # Get current robot pose
            feedback_msg.current_pose = self.get_current_pose()

            # Calculate remaining distance
            feedback_msg.distance_remaining = self.calculate_distance(
                feedback_msg.current_pose, goal_pose)

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Move robot towards goal (simplified)
            self.move_robot_towards_goal(goal_pose)

            # Sleep to allow other callbacks to run
            await asyncio.sleep(0.1)

        # Navigation completed successfully
        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.success = True
        result.message = "Successfully reached goal"
        result.path = path

        return result

    def plan_path_to_pose(self, goal_pose):
        """Plan a path to the goal pose."""
        # Implementation would use path planning algorithms
        # This is a simplified placeholder
        path_msg = Path()
        path_msg.header.frame_id = "map"
        # Plan path implementation would go here
        return path_msg

    def get_current_pose(self):
        """Get the current robot pose."""
        # Implementation would get pose from localization
        # This is a simplified placeholder
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        return pose.pose

    def calculate_distance(self, pose1, pose2):
        """Calculate distance between two poses."""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return (dx*dx + dy*dy)**0.5

    def move_robot_towards_goal(self, goal_pose):
        """Move robot towards the goal."""
        # Implementation would send commands to base controller
        # This is a simplified placeholder
        pass


def main(args=None):
    rclpy.init(args=args)
    node = NavigationActionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## ROS 2 Parameters

Parameters in ROS 2 provide a way to configure nodes at runtime. They can be set at startup, changed during operation, and accessed by other nodes.

### Parameter Basics

- **Type safety**: Parameters have defined types (int, double, string, bool, lists)
- **Runtime modification**: Parameters can be changed while nodes are running
- **Declarative definition**: Parameters can be declared with default values and constraints
- **Parameter events**: Nodes can be notified when parameters change

### Declaring Parameters

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class ParameterExampleNode(Node):

    def __init__(self):
        super().__init__('parameter_example_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('use_camera', True)
        self.declare_parameter('sensor_topics', ['front_camera', 'rear_camera'])

        # Declare parameter with descriptor (for constraints)
        from rcl_interfaces.msg import ParameterDescriptor
        velocity_descriptor = ParameterDescriptor()
        velocity_descriptor.description = 'Maximum linear velocity of the robot'
        velocity_descriptor.floating_point_range = [0.0, 5.0]  # Min, Max
        self.declare_parameter('max_velocity_constrained', 1.0, velocity_descriptor)

        # Get parameter values
        robot_name = self.get_parameter('robot_name').value
        max_velocity = self.get_parameter('max_velocity').value

        self.get_logger().info(f'Robot name: {robot_name}')
        self.get_logger().info(f'Max velocity: {max_velocity}')

    def get_parameter_callback(self):
        """Callback to get current parameter values."""
        robot_name = self.get_parameter('robot_name').value
        max_velocity = self.get_parameter('max_velocity').value
        self.get_logger().info(f'Current robot name: {robot_name}')
        self.get_logger().info(f'Current max velocity: {max_velocity}')


def main(args=None):
    rclpy.init(args=args)
    node = ParameterExampleNode()

    # Example of changing parameter during runtime
    timer = node.create_timer(5.0, node.get_parameter_callback)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Parameter Callbacks

You can register callbacks to be notified when parameters change:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterEvent
from rcl_interfaces.msg import SetParametersResult


class ParameterCallbackNode(Node):

    def __init__(self):
        super().__init__('parameter_callback_node')

        # Declare parameters
        self.declare_parameter('target_velocity', 1.0)
        self.declare_parameter('control_mode', 'velocity')

        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Get initial values
        self.target_velocity = self.get_parameter('target_velocity').value
        self.control_mode = self.get_parameter('control_mode').value

    def parameter_callback(self, params):
        """Callback for parameter changes."""
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'target_velocity':
                if param.type == ParameterEvent.PARAMETER_DOUBLE:
                    if 0.0 <= param.value <= 5.0:  # Validate range
                        self.target_velocity = param.value
                        self.get_logger().info(f'New target velocity: {param.value}')
                    else:
                        result.successful = False
                        result.reason = 'Target velocity must be between 0.0 and 5.0'
                else:
                    result.successful = False
                    result.reason = 'Target velocity must be a double'

            elif param.name == 'control_mode':
                if param.type == ParameterEvent.PARAMETER_STRING:
                    if param.value in ['velocity', 'position', 'effort']:
                        self.control_mode = param.value
                        self.get_logger().info(f'New control mode: {param.value}')
                    else:
                        result.successful = False
                        result.reason = 'Control mode must be velocity, position, or effort'
                else:
                    result.successful = False
                    result.reason = 'Control mode must be a string'

        return result


def main(args=None):
    rclpy.init(args=args)
    node = ParameterCallbackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Practical Parameter Example: Robot Configuration

Here's a practical example of using parameters for robot configuration:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.msg import SetParametersResult


class RobotConfigurationNode(Node):

    def __init__(self):
        super().__init__('robot_configuration_node')

        # Define parameter descriptors with constraints
        wheel_radius_desc = ParameterDescriptor()
        wheel_radius_desc.description = 'Radius of the robot wheels in meters'
        wheel_radius_desc.type = ParameterType.PARAMETER_DOUBLE
        wheel_radius_desc.floating_point_range = [
            {'from_value': 0.01, 'to_value': 0.5, 'step': 0.001}
        ]

        wheel_separation_desc = ParameterDescriptor()
        wheel_separation_desc.description = 'Distance between left and right wheels in meters'
        wheel_separation_desc.type = ParameterType.PARAMETER_DOUBLE
        wheel_separation_desc.floating_point_range = [
            {'from_value': 0.1, 'to_value': 2.0, 'step': 0.01}
        ]

        max_linear_vel_desc = ParameterDescriptor()
        max_linear_vel_desc.description = 'Maximum linear velocity in m/s'
        max_linear_vel_desc.type = ParameterType.PARAMETER_DOUBLE
        max_linear_vel_desc.floating_point_range = [
            {'from_value': 0.1, 'to_value': 5.0, 'step': 0.01}
        ]

        # Declare parameters with descriptors
        self.declare_parameter('wheel_radius', 0.05, wheel_radius_desc)
        self.declare_parameter('wheel_separation', 0.3, wheel_separation_desc)
        self.declare_parameter('max_linear_velocity', 1.0, max_linear_vel_desc)
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('enable_odometry', True)

        # Register parameter callback
        self.add_on_set_parameters_callback(self.validate_parameters)

        # Initialize robot properties
        self.update_robot_properties()

        # Timer to periodically check for parameter changes
        self.timer = self.create_timer(1.0, self.check_parameters)

    def validate_parameters(self, params):
        """Validate parameter changes."""
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'wheel_radius' and param.value <= 0:
                result.successful = False
                result.reason = 'Wheel radius must be positive'
                return result
            elif param.name == 'wheel_separation' and param.value <= 0:
                result.successful = False
                result.reason = 'Wheel separation must be positive'
                return result
            elif param.name == 'max_linear_velocity' and param.value <= 0:
                result.successful = False
                result.reason = 'Max velocity must be positive'
                return result

        # If validation passes, update robot properties
        self.update_robot_properties()
        return result

    def update_robot_properties(self):
        """Update robot properties based on parameters."""
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.robot_name = self.get_parameter('robot_name').value
        self.enable_odometry = self.get_parameter('enable_odometry').value

        # Calculate derived properties
        self.wheel_circumference = 2 * 3.14159 * self.wheel_radius

        self.get_logger().info(f'Updated robot configuration for {self.robot_name}')
        self.get_logger().info(f'  Wheel radius: {self.wheel_radius} m')
        self.get_logger().info(f'  Wheel separation: {self.wheel_separation} m')
        self.get_logger().info(f'  Max velocity: {self.max_linear_velocity} m/s')

    def check_parameters(self):
        """Periodically check for parameter changes."""
        # This could be used to trigger recalculations or reconfigurations
        pass


def main(args=None):
    rclpy.init(args=args)
    node = RobotConfigurationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Parameter Management Commands

### Setting Parameters

```bash
# Set a parameter on a running node
ros2 param set /node_name parameter_name parameter_value

# Example
ros2 param set /robot_configuration_node max_linear_velocity 2.0
```

### Getting Parameters

```bash
# Get a parameter value
ros2 param get /node_name parameter_name

# List all parameters for a node
ros2 param list /node_name

# Example
ros2 param get /robot_configuration_node robot_name
ros2 param list /robot_configuration_node
```

### Loading Parameters from Files

```yaml
# config/robot_params.yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false

robot_configuration_node:  # Applies only to this node
  ros__parameters:
    wheel_radius: 0.05
    wheel_separation: 0.3
    max_linear_velocity: 1.0
    robot_name: "turtlebot4"
    enable_odometry: true
```

### Using Parameter Files in Launch

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to parameter file
    param_file = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    )

    robot_node = Node(
        package='my_robot_package',
        executable='robot_configuration_node',
        parameters=[param_file]
    )

    return LaunchDescription([robot_node])
```

## Best Practices for Actions and Parameters

### Actions Best Practices

- **Use appropriate feedback frequency**: Don't overwhelm the system with feedback messages
- **Handle cancellation gracefully**: Always check for cancellation requests
- **Provide meaningful status updates**: Make feedback informative for debugging
- **Set realistic timeouts**: Consider network latency and processing time

### Parameters Best Practices

- **Validate parameter values**: Use parameter callbacks to validate inputs
- **Provide sensible defaults**: Ensure nodes work with default parameter values
- **Use appropriate parameter types**: Match parameter types to expected values
- **Document parameters**: Clearly document what each parameter does
- **Group related parameters**: Organize parameters logically

## Common Issues and Troubleshooting

### Action Issues

- **Thread safety**: Use appropriate callback groups for thread safety
- **Cancellation handling**: Always check for cancellation during long operations
- **Feedback frequency**: Avoid sending feedback too frequently
- **Result handling**: Properly handle both success and failure cases

### Parameter Issues

- **Parameter declaration**: Always declare parameters before using them
- **Type mismatches**: Ensure parameter types match expectations
- **Node startup order**: Consider the order of node initialization
- **Parameter persistence**: Parameters are not automatically saved between runs

## Key Takeaways

- Actions are ideal for long-running tasks that need feedback and cancellation
- Parameters provide runtime configuration for nodes
- Action servers and clients follow a specific pattern with goals, feedback, and results
- Parameter validation helps maintain system stability
- Proper error handling is essential for robust action implementations
- Parameters enable flexible robot configuration without code changes

In the next section, we'll create exercises to reinforce these concepts.