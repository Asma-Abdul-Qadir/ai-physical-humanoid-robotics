---
sidebar_position: 4
---

# ROS 2 Launch Files

Launch files provide a way to start multiple nodes and configure parameters in a single command. This section covers how to create and use launch files for managing complex robot systems.

## Learning Objectives

By the end of this section, you will be able to:
- Create launch files using both XML and Python formats
- Launch multiple nodes with a single command
- Configure parameters for nodes in launch files
- Use launch arguments to make launch files flexible
- Include other launch files for modular organization

## What are Launch Files?

Launch files in ROS 2 are configuration files that allow you to start multiple nodes, set parameters, and configure the system in a single command. They replace the roslaunch functionality from ROS 1 and provide more flexibility and better integration with the ROS 2 architecture.

### Benefits of Launch Files

- **Convenience**: Start complex systems with a single command
- **Configuration**: Set parameters and configure nodes in one place
- **Modularity**: Include other launch files to build complex systems
- **Reusability**: Share and reuse launch configurations
- **Flexibility**: Use arguments to customize behavior

## Launch File Formats

ROS 2 supports multiple launch file formats:

- **Python**: Most flexible and powerful
- **XML**: Simpler syntax, easier for basic configurations
- **YAML**: Parameter files (used alongside launch files)

## Python Launch Files

Python launch files provide the most flexibility and are the recommended approach for complex configurations.

### Basic Python Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop'
        )
    ])
```

### Python Launch File with Parameters

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    param_file = LaunchConfiguration('param_file')
    robot_name = LaunchConfiguration('robot_name')

    # Declare the arguments
    param_file_arg = DeclareLaunchArgument(
        'param_file',
        default_value='config/default.yaml',
        description='Path to parameter file'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='robot1',
        description='Name of the robot'
    )

    # Create nodes
    robot_node = Node(
        package='my_robot_package',
        executable='robot_control',
        name=[robot_name, '_control'],
        parameters=[param_file],
        remappings=[
            ('/cmd_vel', [robot_name, '/cmd_vel']),
            ('/odom', [robot_name, '/odom'])
        ]
    )

    return LaunchDescription([
        param_file_arg,
        robot_name_arg,
        robot_node
    ])
```

### Advanced Python Launch File with Conditions

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    world_file = LaunchConfiguration('world')

    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz if true'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='World file to load in Gazebo'
    )

    # Nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_rviz_arg,
        world_arg,
        robot_state_publisher,
        joint_state_publisher,
        rviz_node
    ])
```

## XML Launch Files

XML launch files provide a simpler syntax for basic configurations.

### Basic XML Launch File

```xml
<launch>
  <node pkg="turtlesim" exec="turtlesim_node" name="sim"/>
  <node pkg="turtlesim" exec="turtle_teleop_key" name="teleop"/>
</launch>
```

### XML Launch File with Parameters

```xml
<launch>
  <!-- Declare arguments -->
  <arg name="robot_name" default="robot1"/>
  <arg name="use_sim_time" default="false"/>
  <arg name="param_file" default="config/default.yaml"/>

  <!-- Robot control node -->
  <node pkg="my_robot_package" exec="robot_control" name="$(var robot_name)_control">
    <param from="$(var param_file)"/>
    <param name="use_sim_time" value="$(var use_sim_time)"/>

    <!-- Remappings -->
    <remap from="/cmd_vel" to="$(var robot_name)/cmd_vel"/>
    <remap from="/odom" to="$(var robot_name)/odom"/>
  </node>

  <!-- Visualization -->
  <node pkg="rviz2" exec="rviz2" name="rviz2" if="$(var use_rviz)">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
</launch>
```

## Launch Arguments

Launch arguments allow you to customize launch files at runtime.

### Declaring Arguments in Python

```python
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare argument
    my_arg = DeclareLaunchArgument(
        'my_param',
        default_value='default_value',
        description='Description of the parameter'
    )

    # Use the argument
    param_value = LaunchConfiguration('my_param')

    # Use in node
    my_node = Node(
        package='my_package',
        executable='my_executable',
        parameters=[{'param_name': param_value}]
    )

    return LaunchDescription([
        my_arg,
        my_node
    ])
```

### Using Arguments

```bash
# Using default value
ros2 launch my_package my_launch.py

# Providing a value
ros2 launch my_package my_launch.py my_param:=custom_value

# Multiple arguments
ros2 launch my_package my_launch.py robot_name:=robot2 use_sim_time:=true
```

## Node Parameters in Launch Files

You can set parameters for nodes directly in launch files or load them from YAML files.

### Setting Individual Parameters

```python
from launch_ros.actions import Node

my_node = Node(
    package='my_package',
    executable='my_executable',
    parameters=[
        {'param1': 'value1'},
        {'param2': 42},
        {'param3': True},
        {'param4': [1.0, 2.0, 3.0]}
    ]
)
```

### Loading Parameters from YAML

```python
my_node = Node(
    package='my_package',
    executable='my_executable',
    parameters=[
        'config/my_config.yaml',  # Load from file
        {'additional_param': 'value'}  # Additional parameters
    ]
)
```

### YAML Parameter File Example

```yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    global_frame: "map"

my_robot_control_node:  # Applies only to this node
  ros__parameters:
    cmd_vel_topic: "/cmd_vel"
    odom_topic: "/odom"
    wheel_radius: 0.05
    wheel_separation: 0.3
    max_linear_velocity: 1.0
    max_angular_velocity: 1.57
```

## Node Remapping

Remappings allow you to change the topics and services that nodes subscribe to or provide.

### In Python Launch Files

```python
from launch_ros.actions import Node

my_node = Node(
    package='my_package',
    executable='my_executable',
    remappings=[
        ('original_topic', 'new_topic'),
        ('/tf', '/my_tf'),
        ('/tf_static', '/my_tf_static')
    ]
)
```

### In XML Launch Files

```xml
<node pkg="my_package" exec="my_executable">
  <remap from="original_topic" to="new_topic"/>
  <remap from="/tf" to="/my_tf"/>
</node>
```

## Including Other Launch Files

You can include other launch files to build complex systems from modular components.

### Including in Python

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Include another launch file
    included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('other_package'),
                'launch',
                'other_launch.py'
            ])
        ])
    )

    return LaunchDescription([
        included_launch,
        # Additional actions...
    ])
```

### Including in XML

```xml
<launch>
  <include file="$(find-pkg-share other_package)/launch/other_launch.py"/>
</launch>
```

## Practical Example: Humanoid Robot Launch

Here's a complete example of a launch file for a humanoid robot simulation:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    robot_description_path = LaunchConfiguration('robot_description_path')

    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if true'
    )

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz if true'
    )

    declare_robot_description_path = DeclareLaunchArgument(
        'robot_description_path',
        default_value=os.path.join(
            get_package_share_directory('my_humanoid_robot'),
            'urdf',
            'humanoid.urdf.xacro'
        ),
        description='Path to robot description file'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(robot_description_path.value).read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Joint state publisher GUI (for manual control during development)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(use_rviz)
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory('my_humanoid_robot'),
            'rviz',
            'humanoid.rviz'
        )],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )

    # Robot controller
    robot_controller = Node(
        package='my_humanoid_robot',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            os.path.join(
                get_package_share_directory('my_humanoid_robot'),
                'config',
                'controller_params.yaml'
            )
        ]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_use_rviz)
    ld.add_action(declare_robot_description_path)

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(joint_state_publisher_gui)
    ld.add_action(rviz)
    ld.add_action(robot_controller)

    return ld
```

## Launch File Best Practices

### Organization

- **Modular design**: Break complex systems into smaller, reusable launch files
- **Consistent naming**: Use clear, descriptive names for arguments and nodes
- **Documentation**: Comment complex launch files to explain the purpose

### Flexibility

- **Use arguments**: Make launch files configurable with launch arguments
- **Conditional execution**: Use conditions to enable/disable components
- **Parameter files**: Separate parameters from launch logic

### Performance

- **Efficient loading**: Only launch what's needed for the specific use case
- **Resource management**: Consider computational requirements of each node

## Common Launch File Issues

### Path Issues

- **Package paths**: Use `FindPackageShare` in Python or `$(find-pkg-share)` in XML
- **File paths**: Ensure files exist at specified locations
- **Relative vs absolute**: Be clear about path types

### Parameter Issues

- **Type mismatches**: Ensure parameter types match node expectations
- **Missing parameters**: Nodes may fail if required parameters are not provided
- **Scope**: Understand parameter namespaces and scoping

### Node Issues

- **Package not found**: Ensure packages are properly built and sourced
- **Executable not found**: Verify executable names match package contents
- **Dependency issues**: Check that all dependencies are available

## Launch File Commands

### Basic Launch Commands

```bash
# Launch a file
ros2 launch my_package my_launch.py

# Launch with arguments
ros2 launch my_package my_launch.py arg_name:=value

# Launch with multiple arguments
ros2 launch my_package my_launch.py arg1:=value1 arg2:=value2

# List available launch files
ros2 launch my_package
```

### Advanced Launch Commands

```bash
# Launch with verbose output
ros2 launch -d my_package my_launch.py

# Launch in simulation time mode
ros2 launch my_package my_launch.py use_sim_time:=true
```

## Key Takeaways

- Launch files allow starting multiple nodes with a single command
- Python launch files provide more flexibility for complex configurations
- XML launch files are simpler for basic setups
- Launch arguments make launch files customizable
- Parameter files separate configuration from launch logic
- Modular design improves reusability and maintainability

In the next section, we'll explore ROS 2 Actions and Parameters, advanced communication patterns for robotics applications.