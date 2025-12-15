---
sidebar_position: 1
---

# Gazebo Garden Introduction: Physics Simulation for Humanoid Robotics

Welcome to the Gazebo Garden module, which focuses on physics-based simulation for humanoid robotics. Gazebo Garden provides realistic physics simulation that's essential for testing and validating humanoid robot behaviors before deployment on real hardware.

## Learning Objectives

By the end of this section, you will be able to:
- Install and configure Gazebo Garden for humanoid robotics simulation
- Understand the physics engine and its parameters
- Create and modify simulation worlds
- Spawn and control humanoid robot models
- Implement physics-based sensors and actuators
- Debug simulation issues and optimize performance

## What is Gazebo Garden?

Gazebo Garden is the latest major release of the Gazebo robotics simulator, featuring the Ignition simulation framework. It provides high-fidelity physics simulation, realistic rendering, and a robust plugin system that makes it ideal for humanoid robotics development.

### Key Features of Gazebo Garden

- **High-fidelity physics**: Accurate simulation of rigid body dynamics, collisions, and contacts
- **Modular architecture**: Plugin-based system for extending functionality
- **Realistic rendering**: High-quality visual rendering with support for multiple render engines
- **Sensor simulation**: Accurate simulation of cameras, LIDAR, IMU, force/torque sensors
- **ROS 2 integration**: Native support for ROS 2 communication through Ignition bridges
- **Scalable simulation**: Support for single and multi-robot simulations

## Gazebo vs. Other Simulation Environments

### Gazebo vs. Webots
- **Gazebo**: Better physics accuracy, more ROS integration, larger community
- **Webots**: Easier to use, built-in robot models, web-based interface

### Gazebo vs. PyBullet
- **Gazebo**: More comprehensive simulation environment, better visualization
- **PyBullet**: Faster simulation, good for reinforcement learning applications

### Gazebo vs. MuJoCo
- **Gazebo**: Open-source, extensive robotics tools, ROS integration
- **MuJoCo**: Proprietary, very accurate physics, expensive licensing

## Installation and Setup

### System Requirements
- **OS**: Ubuntu 22.04 LTS (recommended) or other Linux distributions
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Dedicated GPU with OpenGL 3.3+ support
- **CPU**: Multi-core processor with good single-thread performance

### Installation Steps

1. **Add the OSRF repository**:
   ```bash
   sudo apt update && sudo apt install wget lsb-release gnupg
   sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
   ```

2. **Install Gazebo Garden**:
   ```bash
   sudo apt update
   sudo apt install gz-harmonic
   ```

3. **Install ROS 2 Gazebo packages**:
   ```bash
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
   ```

4. **Verify installation**:
   ```bash
   gz sim --version
   ```

### Environment Setup

Add Gazebo paths to your environment:
```bash
# Add to ~/.bashrc
export GZ_SIM_SYSTEM_PLUGIN_PATH=$AMENT_PREFIX_PATH/lib
export GZ_SIM_RESOURCE_PATH=$AMENT_PREFIX_PATH/share
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$AMENT_PREFIX_PATH/share/turtlebot3_gazebo/models
```

## Gazebo Architecture

### Core Components

1. **Gazebo Server (`gz sim`)**: Headless simulation engine
2. **Gazebo GUI (`gz sim -g`)**: Graphical user interface
3. **Ignition Transport**: Message passing system
4. **Plugins**: Extendable functionality through plugin system

### World Files

Gazebo uses SDF (Simulation Description Format) files to define simulation worlds:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include a model -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add a light source -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Add physics parameters -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

## Physics Engine Configuration

### Physics Parameters

The physics engine configuration affects simulation accuracy and performance:

```xml
<physics name="ode" type="ode">
  <!-- Time step settings -->
  <max_step_size>0.001</max_step_size>  <!-- Simulation time step (s) -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Steps per second -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time vs simulation speed -->

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- Solver settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Performance vs. Accuracy Trade-offs

- **Smaller time steps**: More accurate but slower simulation
- **Higher solver iterations**: More stable contacts but slower performance
- **Lower ERP/CFM**: More responsive but potentially unstable

## Working with Models

### Model Structure

A Gazebo model typically includes:

```
my_robot/
├── model.config          # Model metadata
├── model.sdf             # Model definition
└── meshes/               # 3D mesh files
    ├── link1.dae
    └── link2.stl
└── materials/            # Material definitions
    └── textures/
```

### Model Configuration File (model.config)

```xml
<?xml version="1.0"?>
<model>
  <name>My Humanoid Robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>

  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>

  <description>
    A simple humanoid robot for simulation
  </description>
</model>
```

### Basic Model Definition

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_humanoid">
    <!-- Torso -->
    <link name="torso">
      <pose>0 0 0.8 0 0 0</pose>
      <collision name="torso_collision">
        <geometry>
          <box>
            <size>0.3 0.2 0.6</size>
          </box>
        </geometry>
      </collision>
      <visual name="torso_visual">
        <geometry>
          <box>
            <size>0.3 0.2 0.6</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.5</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.6</iyy>
          <iyz>0.0</iyz>
          <izz>0.3</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Head -->
    <link name="head">
      <pose>0 0 0.3 0 0 0</pose>
      <collision name="head_collision">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
      </collision>
      <visual name="head_visual">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
      </visual>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.004</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.004</iyy>
          <iyz>0.0</iyz>
          <izz>0.004</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joint connecting torso and head -->
    <joint name="neck_joint" type="revolute">
      <parent>torso</parent>
      <child>head</child>
      <pose>0 0 0.3 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>10.0</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

## Sensors in Gazebo

### Camera Sensor

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <plugin filename="gz-sim-camera-system" name="gz::sim::systems::Camera">
    <topic>camera/image_raw</topic>
  </plugin>
</sensor>
```

### IMU Sensor

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>imu/data</topic>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu">
    <topic>imu/data</topic>
  </plugin>
</sensor>
```

### Force/Torque Sensor

```xml
<sensor name="ft_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>ft_sensor/wrench</topic>
</sensor>
```

## ROS 2 Integration

### Gazebo ROS Packages

Gazebo integrates with ROS 2 through several packages:

- `gazebo_ros_pkgs`: Core ROS 2 plugins for Gazebo
- `gazebo_ros2_control`: ROS 2 control interface
- `ros_gz_bridge`: Message bridge between ROS 2 and Ignition

### Launching with ROS 2

```xml
<launch>
  <!-- Start Gazebo server -->
  <node name="gazebo_server" pkg="gazebo_ros" exec="gzserver" args="$(find-pkg-share my_robot_gazebo)/worlds/my_world.sdf">
    <param name="use_sim_time" value="true"/>
  </node>

  <!-- Start Gazebo client -->
  <node name="gazebo_client" pkg="gazebo_ros" exec="gzclient" if="$(var use_gui)">
    <param name="use_sim_time" value="true"/>
  </node>
</launch>
```

## Practical Example: Humanoid Robot Simulation

Let's create a complete example of a simple humanoid robot in Gazebo:

### World File (humanoid_world.sdf)

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics -->
    <physics name="ode" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Humanoid robot -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Launch File

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')

    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if true'
    )

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            get_package_share_directory('my_robot_gazebo'),
            'worlds',
            'humanoid_world.sdf'
        ),
        description='SDF world file'
    )

    # Gazebo server
    gzserver = Node(
        package='gazebo_ros',
        executable='gzserver',
        arguments=[world],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Gazebo client
    gzclient = Node(
        package='gazebo_ros',
        executable='gzclient',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui'))
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world,
        gzserver,
        gzclient
    ])
```

## Simulation Optimization Tips

### Performance Optimization

1. **Adjust physics parameters**: Use appropriate time steps and solver settings
2. **Reduce visual complexity**: Simplify meshes for collision detection
3. **Limit update rates**: Set appropriate sensor update rates
4. **Use simpler collision geometries**: Use boxes and spheres instead of complex meshes

### Accuracy Optimization

1. **Fine-tune physics parameters**: Smaller time steps for better accuracy
2. **Proper inertia values**: Calculate accurate inertia tensors
3. **Realistic friction values**: Set appropriate friction coefficients
4. **Appropriate solver settings**: Increase iterations for stable contacts

## Debugging Common Issues

### Model Spawning Issues

- **Model not found**: Check GAZEBO_MODEL_PATH environment variable
- **Model falls through ground**: Verify collision geometries and physics properties
- **Joints behave strangely**: Check joint limits and axis alignment

### Physics Issues

- **Unstable simulation**: Increase solver iterations or decrease time step
- **Objects passing through each other**: Check collision geometries and surface layers
- **Excessive jittering**: Adjust ERP and CFM values

### Performance Issues

- **Slow simulation**: Reduce complexity or adjust physics parameters
- **High CPU usage**: Lower update rates or simplify models
- **Graphics issues**: Check GPU drivers and OpenGL support

## Key Takeaways

- Gazebo Garden provides high-fidelity physics simulation for humanoid robotics
- Proper physics configuration is crucial for realistic simulation
- Model structure includes visual, collision, and inertial properties
- ROS 2 integration enables seamless communication with simulation
- Performance and accuracy require careful balance of simulation parameters

In the next section, we'll explore Unity 2023.2 LTS for advanced visualization capabilities.