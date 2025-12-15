---
sidebar_position: 1
---

# Nav2 Navigation: Path Planning for Humanoid Robots

Welcome to the Navigation module using Navigation2 (Nav2), the state-of-the-art navigation system for mobile robots in ROS 2. This chapter focuses on adapting Nav2 for humanoid robot navigation, including path planning, obstacle avoidance, and motion execution specifically tailored for bipedal locomotion.

## Learning Objectives

By the end of this section, you will be able to:
- Understand the Navigation2 architecture and its components
- Configure Nav2 for humanoid robot characteristics
- Implement path planning algorithms suitable for humanoid locomotion
- Customize obstacle avoidance for humanoid robot dynamics
- Integrate Nav2 with humanoid robot controllers
- Validate navigation performance in simulation environments
- Troubleshoot common navigation issues in humanoid robots

## Introduction to Navigation2 (Nav2)

Navigation2 is the next-generation navigation system for ROS 2, designed to be more robust, flexible, and maintainable than its predecessor. For humanoid robots, Nav2 provides the foundation for autonomous navigation while accounting for the unique characteristics of bipedal locomotion.

### Key Components of Nav2

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Behavior      │    │   Path Planner  │    │   Controller    │
│   Tree (BT)     │◄──►│   (Global)      │◄──►│   (Local)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Recovery      │    │   Costmap       │    │   Velocity      │
│   Behaviors     │    │   (2D/3D)       │    │   Smoothers     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Nav2 Architecture Overview

1. **Behavior Tree (BT) Executor**: Orchestrates navigation tasks using behavior trees
2. **Global Planner**: Computes optimal path from start to goal
3. **Local Planner**: Tracks path while avoiding obstacles
4. **Costmap**: Represents obstacles and navigation space
5. **Controllers**: Execute velocity commands
6. **Recovery Behaviors**: Handle navigation failures

## Nav2 for Humanoid Robots

### Unique Challenges for Humanoid Navigation

Humanoid robots face specific challenges that differ from wheeled mobile robots:

#### Bipedal Dynamics
- **Stability**: Maintaining balance during movement
- **Foot placement**: Precise footstep planning required
- **Center of Mass (CoM)**: Critical for stability
- **Zero Moment Point (ZMP)**: Key for balance control

#### Locomotion Characteristics
- **Discrete footsteps**: Unlike continuous wheeled motion
- **Limited turning**: Cannot rotate in place like differential drives
- **Step constraints**: Minimum/maximum step size and height
- **Dynamic balance**: Requires constant balance adjustments

#### Hardware Constraints
- **Joint limitations**: Range of motion constraints
- **Torque limitations**: Power constraints affect speed/force
- **Sensing limitations**: IMU, force/torque sensors for balance
- **Computational constraints**: Real-time balance requirements

### Nav2 Adaptations for Humanoids

#### 1. Footstep Planner Integration
Unlike traditional planners that output smooth paths, humanoid navigation requires discrete footstep plans:

```yaml
# humanoid_nav2_config.yaml
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    enable_bt_monitoring: False
    expected_planner_frequency: 1.0  # Lower frequency for footstep planning

    # Footstep planner specific parameters
    footstep_planning_enabled: True
    min_step_size: 0.1    # Minimum step size (m)
    max_step_size: 0.3    # Maximum step size (m)
    step_rotation_resolution: 0.2  # Rotation resolution (rad)

    # Behavior tree configuration
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_are_equal_poses_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transformer_node_bt_node
    - nav2_get_costmap_node_bt_node
    - nav2_if_decorator_bt_node
    - nav2_while_do_loop_node_bt_node
    - nav2_compute_path_footstep_action_bt_node  # Custom footstep planner
```

#### 2. Humanoid-Specific Costmap Configuration

```yaml
# costmap_config.yaml
global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  transform_tolerance: 0.5
  resolution: 0.05  # Higher resolution for precise footstep planning

  # Footprint adapted for humanoid
  footprint: [[-0.15, -0.1], [-0.15, 0.1], [0.15, 0.1], [0.15, -0.1]]
  footprint_padding: 0.02

  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}

  obstacle_layer:
    enabled: True
    observation_sources: scan
    scan:
      topic: /laser_scan
      max_obstacle_height: 2.0  # Humanoid height consideration
      clearing: True
      marking: True
      data_type: LaserScan
      obstacle_range: 3.0
      raytrace_range: 4.0

  inflation_layer:
    enabled: True
    cost_scaling_factor: 3.0  # Increased for humanoid safety
    inflation_radius: 0.5     # Larger safety margin for bipedal stability
    inflate_unknown: False

local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 10.0
  publish_frequency: 5.0
  transform_tolerance: 0.5
  resolution: 0.025  # Even higher resolution for local planning
  width: 5.0
  height: 5.0
  origin_x: -2.5
  origin_y: -2.5

  # Smaller footprint for local navigation
  footprint: [[-0.1, -0.05], [-0.1, 0.05], [0.1, 0.05], [0.1, -0.05]]
  footprint_padding: 0.01

  plugins:
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: voxel_layer, type: "nav2_costmap_2d::VoxelLayer"}  # 3D obstacles
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
```

## Path Planning for Humanoid Robots

### Global Path Planning Considerations

Traditional global planners like A* and Dijkstra work for humanoid robots, but with important adaptations:

#### 1. Kinodynamic Planning
Humanoid robots have complex kinodynamic constraints that must be considered:

```python
#!/usr/bin/env python3

import math
import numpy as np
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node


class HumanoidPathPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Footstep planning parameters
        self.min_step_size = 0.1
        self.max_step_size = 0.3
        self.max_turn_angle = math.radians(30)  # Max turn per step

        # Balance constraints
        self.zmp_margin = 0.05  # Safety margin for Zero Moment Point

        # Initialize action server
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        """Execute path planning with humanoid constraints"""
        self.get_logger().info('Received path planning request')

        start_pose = goal_handle.request.start
        goal_pose = goal_handle.request.goal.pose

        # Plan path considering humanoid constraints
        path = self.plan_humanoid_path(start_pose.pose, goal_pose)

        if path is None:
            self.get_logger().error('Failed to find valid path')
            goal_handle.abort()
            return ComputePathToPose.Result()

        # Convert to Nav2 Path message
        nav_path = Path()
        nav_path.header.frame_id = "map"
        nav_path.header.stamp = self.get_clock().now().to_msg()

        for pose in path.poses:
            nav_path.poses.append(pose)

        goal_handle.succeed()
        result = ComputePathToPose.Result()
        result.path = nav_path

        return result

    def plan_humanoid_path(self, start_pose, goal_pose):
        """Plan path with humanoid-specific constraints"""
        # Implement RRT* or other kinodynamic planning algorithm
        # that considers humanoid step constraints

        path = Path()
        path.poses = []

        # Calculate straight-line path first
        dx = goal_pose.position.x - start_pose.position.x
        dy = goal_pose.position.y - start_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate number of steps needed
        num_steps = int(distance / self.max_step_size) + 1

        # Generate waypoints respecting step size constraints
        for i in range(num_steps + 1):
            ratio = i / num_steps if num_steps > 0 else 0

            waypoint = PoseStamped()
            waypoint.header.frame_id = "map"

            # Interpolate position
            waypoint.pose.position.x = start_pose.position.x + dx * ratio
            waypoint.pose.position.y = start_pose.position.y + dy * ratio
            waypoint.pose.position.z = start_pose.position.z  # Maintain height

            # Interpolate orientation (simplified)
            # In practice, this would involve more complex orientation planning
            # considering humanoid balance and turning constraints
            waypoint.pose.orientation = start_pose.orientation

            path.poses.append(waypoint)

        # Add goal pose explicitly
        final_waypoint = PoseStamped()
        final_waypoint.header.frame_id = "map"
        final_waypoint.pose = goal_pose
        path.poses.append(final_waypoint)

        return path

    def validate_footstep(self, prev_pose, next_pose):
        """Validate that a footstep is feasible for humanoid"""
        # Check step size constraint
        dx = next_pose.position.x - prev_pose.position.x
        dy = next_pose.position.y - prev_pose.position.y
        step_distance = math.sqrt(dx*dx + dy*dy)

        if step_distance < self.min_step_size or step_distance > self.max_step_size:
            return False

        # Check turning constraint
        # (This would involve more complex orientation validation)

        # Check for obstacles at footstep location
        # (This would check the costmap at the step location)

        return True


def main(args=None):
    rclpy.init(args=args)
    node = HumanoidPathPlanner()

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

#### 2. Footstep Planning Algorithms

For humanoid robots, the path needs to be converted to discrete footsteps:

```python
#!/usr/bin/env python3

import numpy as np
import math
from geometry_msgs.msg import Pose, Point
from builtin_interfaces.msg import Duration


class FootstepPlanner:
    def __init__(self):
        self.step_constraints = {
            'min_step_size': 0.1,
            'max_step_size': 0.3,
            'max_turn_angle': math.radians(30),
            'step_height': 0.05,  # Clearance for stepping
            'stance_width': 0.2   # Distance between feet
        }

    def plan_footsteps(self, path, robot_state):
        """Convert continuous path to discrete footsteps"""
        footsteps = []

        # Start with current robot state
        current_left_foot = self.get_left_foot_pose(robot_state)
        current_right_foot = self.get_right_foot_pose(robot_state)

        # Determine leading foot based on current stance
        leading_foot = self.determine_leading_foot(current_left_foot, current_right_foot)

        path_poses = path.poses
        i = 0

        while i < len(path_poses) - 1:
            # Calculate desired step location
            desired_step = self.calculate_desired_step_location(
                path_poses, i, current_left_foot, current_right_foot, leading_foot)

            # Find feasible step location respecting constraints
            feasible_step = self.find_feasible_step_location(
                desired_step, current_left_foot, current_right_foot, leading_foot)

            if feasible_step is not None:
                # Add the step to the sequence
                footsteps.append({
                    'foot': leading_foot,
                    'pose': feasible_step,
                    'timestamp': self.calculate_step_timing(len(footsteps))
                })

                # Update current foot position
                if leading_foot == 'left':
                    current_left_foot = feasible_step
                else:
                    current_right_foot = feasible_step

                # Switch leading foot
                leading_foot = 'right' if leading_foot == 'left' else 'left'

                # Move along path
                i += self.calculate_path_advance(feasible_step, path_poses, i)
            else:
                # Need to replan or adjust path
                break

        return footsteps

    def calculate_desired_step_location(self, path, path_idx, left_foot, right_foot, leading_foot):
        """Calculate where we'd like to place the next foot"""
        # This is a simplified approach
        # In practice, this would use more sophisticated gait planning

        # Look ahead in the path to determine desired direction
        look_ahead = min(path_idx + 2, len(path) - 1)
        desired_direction = self.calculate_direction(path[path_idx].pose, path[look_ahead].pose)

        # Calculate step location based on current stance and desired direction
        current_support_foot = right_foot if leading_foot == 'left' else left_foot

        # Offset the step in the desired direction
        step_pose = Pose()
        step_pose.position.x = current_support_foot.position.x + desired_direction[0] * self.step_constraints['max_step_size']
        step_pose.position.y = current_support_foot.position.y + desired_direction[1] * self.step_constraints['max_step_size']
        step_pose.position.z = current_support_foot.position.z  # Maintain ground contact

        # Set appropriate orientation
        step_pose.orientation = path[path_idx].pose.orientation

        return step_pose

    def find_feasible_step_location(self, desired_pose, left_foot, right_foot, leading_foot):
        """Find a step location that respects all constraints"""
        # Check if desired pose is feasible
        if self.is_step_feasible(desired_pose, left_foot, right_foot, leading_foot):
            return desired_pose

        # If not feasible, search for nearby feasible location
        # This could use sampling-based approaches or optimization
        return self.search_feasible_location(desired_pose, left_foot, right_foot, leading_foot)

    def is_step_feasible(self, step_pose, left_foot, right_foot, leading_foot):
        """Check if a step is feasible given constraints"""
        # Check step size constraints
        support_foot = right_foot if leading_foot == 'left' else left_foot
        step_distance = self.calculate_distance(step_pose.position, support_foot.position)

        if step_distance < self.step_constraints['min_step_size'] or \
           step_distance > self.step_constraints['max_step_size']:
            return False

        # Check balance constraints (ZMP, etc.)
        if not self.check_balance_feasibility(step_pose, left_foot, right_foot, leading_foot):
            return False

        # Check for obstacles
        if not self.check_obstacle_feasibility(step_pose):
            return False

        return True

    def check_balance_feasibility(self, step_pose, left_foot, right_foot, leading_foot):
        """Check if step maintains robot balance"""
        # Calculate potential ZMP after step
        # This is a simplified check
        support_polygon = self.calculate_support_polygon(left_foot, right_foot, leading_foot)

        # Check if CoM projection is within support polygon
        # (Implementation would depend on specific balance controller)
        return True  # Simplified for example

    def calculate_direction(self, pose1, pose2):
        """Calculate normalized direction vector between two poses"""
        dx = pose2.position.x - pose1.position.x
        dy = pose2.position.y - pose1.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0:
            return [dx/distance, dy/distance]
        else:
            return [0, 1]  # Default direction

    def calculate_distance(self, point1, point2):
        """Calculate 2D distance between two points"""
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        return math.sqrt(dx*dx + dy*dy)
```

## Local Path Following and Control

### Humanoid-Specific Local Planner

The local planner for humanoid robots needs to handle the discrete nature of footsteps:

```python
#!/usr/bin/env python3

from nav2_msgs.action import FollowPath
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionServer
from rclpy.node import Node
import math


class HumanoidLocalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_local_planner')

        # Initialize action server
        self._action_server = ActionServer(
            self,
            FollowPath,
            'follow_path',
            self.execute_callback)

        # Humanoid-specific parameters
        self.step_frequency = 1.0  # Steps per second
        self.balance_margin = 0.05  # Safety margin for balance
        self.max_linear_speed = 0.3  # Conservative speed for stability

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Robot state
        self.current_pose = None
        self.current_velocity = None

    def execute_callback(self, goal_handle):
        """Execute path following with humanoid constraints"""
        self.get_logger().info('Following path with humanoid constraints')

        path = goal_handle.request.path
        self.follow_humanoid_path(path)

        goal_handle.succeed()
        result = FollowPath.Result()
        return result

    def follow_humanoid_path(self, path):
        """Follow path using humanoid-appropriate control"""
        # Convert path to footstep sequence
        footsteps = self.convert_path_to_footsteps(path)

        # Execute footsteps one by one
        for i, footstep in enumerate(footsteps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return

            # Execute individual footstep
            success = self.execute_footstep(footstep)

            if not success:
                # Try recovery behavior
                recovery_success = self.attempt_recovery(path, i)
                if not recovery_success:
                    break  # Navigation failed

        # Finalize path following
        self.stop_robot()

    def convert_path_to_footsteps(self, path):
        """Convert continuous path to discrete footsteps"""
        # Use the footstep planner from previous section
        planner = FootstepPlanner()
        robot_state = self.get_current_robot_state()
        footsteps = planner.plan_footsteps(path, robot_state)
        return footsteps

    def execute_footstep(self, footstep):
        """Execute a single footstep"""
        # This would interface with the robot's balance controller
        # and footstep execution system

        # For simulation purposes, we'll publish velocity commands
        # that approximate the desired stepping motion

        # Calculate required velocity to reach the footstep
        desired_velocity = self.calculate_footstep_velocity(footstep)

        # Execute for the appropriate duration
        duration = Duration()
        duration.sec = int(1.0 / self.step_frequency)  # Time per step
        duration.nanosec = int((1.0 / self.step_frequency - duration.sec) * 1e9)

        # Publish velocity command
        self.publish_velocity_command(desired_velocity, duration)

        # Wait for step completion
        self.wait_for_step_completion(duration)

        return True  # Simplified success

    def calculate_footstep_velocity(self, footstep):
        """Calculate velocity needed to execute a footstep"""
        # This is a simplified calculation
        # Real implementation would consider balance, dynamics, etc.

        cmd_vel = Twist()

        # Calculate linear velocity toward footstep
        current_pos = self.current_pose.position if self.current_pose else Point(x=0, y=0, z=0)
        dx = footstep['pose'].position.x - current_pos.x
        dy = footstep['pose'].position.y - current_pos.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Set conservative velocity
        cmd_vel.linear.x = min(self.max_linear_speed, distance * self.step_frequency)
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0

        # Calculate angular velocity for orientation
        # (simplified - real implementation would be more complex)
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0  # Would need to consider turning constraints

        return cmd_vel

    def publish_velocity_command(self, cmd_vel, duration):
        """Publish velocity command for specified duration"""
        # In real implementation, this would interface with
        # the robot's motion controller
        self.cmd_vel_pub.publish(cmd_vel)

    def wait_for_step_completion(self, duration):
        """Wait for step execution to complete"""
        # Simplified wait - real implementation would monitor actual execution
        self.get_clock().sleep_for(duration)

    def attempt_recovery(self, path, current_step):
        """Attempt to recover from navigation failure"""
        # Recovery strategies for humanoid robots
        # - Step in place to regain balance
        # - Small adjustment steps
        # - Abort and replan

        self.get_logger().info('Attempting humanoid navigation recovery')

        # For now, return success to continue
        return True

    def stop_robot(self):
        """Stop robot motion"""
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
```

## Integration with Navigation Stack

### Launch File for Humanoid Navigation

```xml
<!-- humanoid_navigation.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    # Paths
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')
    pkg_humanoid_nav = FindPackageShare('humanoid_navigation')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg_humanoid_nav, 'config', 'humanoid_nav2_params.yaml']),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack')

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition',
        default_value='False',
        description='Whether to use composed bringup')

    declare_use_respawn = DeclareLaunchArgument(
        'use_respawn',
        default_value='False',
        description='Whether to respawn if a node crashes')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='log level')

    # Include the main nav2 launch file
    nav2_bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            pkg_nav2_bringup,
            'launch',
            'navigation_launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': params_file,
            'use_composition': use_composition,
            'use_respawn': use_respawn,
            'log_level': log_level}.items())

    # Humanoid-specific nodes
    humanoid_controller = Node(
        package='humanoid_navigation',
        executable='humanoid_local_planner',
        name='humanoid_local_planner',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/cmd_vel', '/humanoid_cmd_vel')],
        output='screen'
    )

    balance_controller = Node(
        package='humanoid_control',
        executable='balance_controller',
        name='balance_controller',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_use_respawn)
    ld.add_action(declare_log_level_cmd)

    # Add nodes
    ld.add_action(nav2_bringup_cmd)
    ld.add_action(humanoid_controller)
    ld.add_action(balance_controller)

    return ld
```

## Practical Implementation Example

### Setting up Nav2 for a Humanoid Robot

Here's a complete example of configuring and using Nav2 for a humanoid robot:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import time


class HumanoidNavigator(Node):
    def __init__(self):
        super().__init__('humanoid_navigator')

        # Create action client for navigation
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # Timer to send navigation goals
        self.timer = self.create_timer(5.0, self.send_navigation_goal)
        self.goal_count = 0

    def send_navigation_goal(self):
        """Send a navigation goal to the humanoid robot"""
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Navigation action server not available')
            return

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set goal position (example: moving in a square pattern)
        positions = [
            (2.0, 0.0),   # Move forward
            (2.0, 2.0),   # Move right
            (0.0, 2.0),   # Move back
            (0.0, 0.0)    # Return to start
        ]

        if self.goal_count < len(positions):
            x, y = positions[self.goal_count]
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.orientation.w = 1.0  # No rotation

            # Send goal
            self.get_logger().info(f'Sending navigation goal to ({x}, {y})')
            future = self.nav_to_pose_client.send_goal_async(goal_msg)
            future.add_done_callback(self.goal_response_callback)

            self.goal_count += 1
        else:
            # Stop after completing the pattern
            self.timer.cancel()

    def goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

        # Continue with next goal after a delay
        time.sleep(2.0)


def main(args=None):
    rclpy.init(args=args)
    navigator = HumanoidNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Simulation and Testing

### Gazebo Simulation Setup

To test Nav2 with a humanoid robot in Gazebo, you'll need to set up proper controllers and sensors:

```xml
<!-- humanoid_with_nav2.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_nav2">

  <!-- Include base humanoid model -->
  <xacro:include filename="$(find humanoid_description)/urdf/humanoid.urdf.xacro"/>

  <!-- Navigation sensors -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.0 0.0 1.0" rpy="0 0 0"/>  <!-- Height of LIDAR on robot -->
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Gazebo plugin for LIDAR -->
  <gazebo reference="lidar_link">
    <sensor type="ray" name="humanoid_lidar">
      <pose>0 0 0 0 0 0</pose>
      <visualize>false</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <argument>~/out:=/laser_scan</argument>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU for balance sensing -->
  <joint name="imu_joint" type="fixed">
    <parent link="torso"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="imu_link"/>

  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
    </sensor>
  </gazebo>

  <!-- Odometry source -->
  <gazebo>
    <plugin name="ground_truth_odom" filename="libgazebo_ros_p3d.so">
      <ros>
        <argument>~/odom:=/odom</argument>
      </ros>
      <update_rate>50</update_rate>
      <body_name>base_link</body_name>
      <gaussian_noise>0.01</gaussian_noise>
      <frame_name>odom</frame_name>
      <child_frame_name>base_link</child_frame_name>
    </plugin>
  </gazebo>

</robot>
```

## Performance Optimization

### Tuning Parameters for Humanoid Navigation

Here are key parameters to tune for optimal humanoid navigation performance:

```yaml
# performance_tuning.yaml

# Global planner tuning
global_costmap:
  inflation_layer:
    cost_scaling_factor: 5.0  # Higher for humanoid safety
    inflation_radius: 0.6     # Larger for stability margin

# Local planner tuning
local_costmap:
  inflation_layer:
    cost_scaling_factor: 3.0  # Balance between safety and efficiency
    inflation_radius: 0.4     # Local safety margin

# Controller tuning
humanoid_local_planner:
  ros__parameters:
    controller_frequency: 10.0    # Conservative for stability
    min_x_velocity_threshold: 0.05  # Minimum movement threshold
    max_x_velocity: 0.3          # Conservative speed
    min_y_velocity_threshold: 0.0  # No lateral movement for simple humanoid
    max_y_velocity: 0.0
    min_theta_velocity_threshold: 0.01
    max_theta_velocity: 0.2      # Limited turning for humanoid

    # Footstep-specific parameters
    step_frequency: 1.0          # Steps per second
    balance_margin: 0.05         # Safety margin for ZMP
    recovery_step_size: 0.1      # Size of recovery steps

# Behavior tree tuning
bt_navigator:
  ros__parameters:
    # Increase timeouts for complex humanoid computations
    default_server_timeout: 30
    expected_planner_frequency: 0.5  # Lower due to complex planning
    expected_controller_frequency: 1.0  # Matches step frequency
```

## Troubleshooting Common Issues

### Navigation Instability

**Problem**: Robot becomes unstable during navigation
**Solutions**:
- Reduce navigation speed
- Increase safety margins in costmaps
- Improve balance controller parameters
- Use more conservative step sizes

### Path Planning Failures

**Problem**: Global planner cannot find valid path
**Solutions**:
- Adjust costmap resolution for better precision
- Modify inflation parameters
- Check if robot footprint is correctly configured
- Verify map quality and resolution

### Local Navigation Oscillation

**Problem**: Robot oscillates around obstacles
**Solutions**:
- Adjust local planner parameters
- Increase minimum turning radius
- Modify recovery behaviors
- Tune velocity profiles

## Integration with Other Systems

### Vision and Perception Integration

Nav2 can be enhanced with perception data for better navigation:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import numpy as np


class PerceptionAugmentedNavigation(Node):
    def __init__(self):
        super().__init__('perception_augmented_nav')

        # Subscriptions for perception data
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)

        # Publisher for augmented costmap
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray, '/perception_obstacles', 10)

        # Perception data storage
        self.latest_rgb = None
        self.latest_depth = None
        self.obstacle_points = []

        # Timer for processing perception data
        self.process_timer = self.create_timer(0.5, self.process_perception_data)

    def rgb_callback(self, msg):
        """Handle RGB camera data"""
        # Process RGB data for object detection
        # This would typically use a neural network
        self.latest_rgb = msg

    def depth_callback(self, msg):
        """Handle depth camera data"""
        # Process depth data for 3D obstacle detection
        self.latest_depth = msg

        # Extract obstacle points from depth data
        obstacle_points = self.extract_obstacles_from_depth(msg)
        self.obstacle_points = obstacle_points

    def pointcloud_callback(self, msg):
        """Handle point cloud data"""
        # Process point cloud for detailed obstacle mapping
        filtered_points = self.filter_ground_points(msg)
        self.update_costmap_with_pointcloud(filtered_points)

    def process_perception_data(self):
        """Process perception data and update navigation system"""
        if self.obstacle_points:
            # Create visualization markers for detected obstacles
            marker_array = self.create_obstacle_markers(self.obstacle_points)
            self.obstacle_marker_pub.publish(marker_array)

            # Update costmap with perception data
            self.update_dynamic_costmap(self.obstacle_points)

    def extract_obstacles_from_depth(self, depth_msg):
        """Extract obstacle points from depth image"""
        # This is a simplified implementation
        # Real implementation would use computer vision techniques

        obstacle_points = []

        # Convert depth message to usable format
        # (In practice, use cv_bridge to convert to OpenCV format)

        # Process depth data to identify obstacles
        # This would typically involve:
        # 1. Depth thresholding
        # 2. Clustering of obstacle points
        # 3. Filtering based on height/distance

        return obstacle_points

    def update_dynamic_costmap(self, obstacle_points):
        """Update costmap with dynamic obstacle information"""
        # This would interface with Nav2's costmap system
        # to add dynamic obstacles detected by perception
        pass

    def create_obstacle_markers(self, obstacle_points):
        """Create visualization markers for obstacles"""
        marker_array = MarkerArray()

        for i, point in enumerate(obstacle_points):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "perception_obstacles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionAugmentedNavigation()

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

## Best Practices for Humanoid Navigation

### 1. Safety-First Approach
- Always maintain safety margins in costmaps
- Use conservative velocity profiles
- Implement robust recovery behaviors
- Monitor balance and stability continuously

### 2. Gradual Complexity Increase
- Start with simple navigation tasks
- Progress to complex environments
- Test in simulation before real-world deployment
- Validate each component individually

### 3. Comprehensive Testing
- Test in various environments
- Validate edge cases and failure scenarios
- Verify integration with balance control
- Test with different walking gaits

### 4. Performance Monitoring
- Monitor navigation performance metrics
- Track success rates and failure modes
- Profile computational requirements
- Optimize for real-time operation

## Key Takeaways

- Nav2 provides a robust foundation for humanoid robot navigation
- Humanoid-specific adaptations are needed for footstep planning
- Balance and stability are critical considerations
- Integration with perception enhances navigation capabilities
- Careful parameter tuning is essential for stable operation
- Simulation testing is crucial before real-world deployment

In the next section, we'll explore Vision-Language-Action pipelines that integrate with the navigation system to enable more sophisticated humanoid robot behaviors.