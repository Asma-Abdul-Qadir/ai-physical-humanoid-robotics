---
sidebar_position: 6
---

# Digital Twin Exercises: Multi-Environment Simulation

This section provides hands-on exercises to reinforce the concepts covered in the Digital Twin module. These exercises focus on integrating multiple simulation environments and leveraging their respective strengths.

## Exercise 1: Multi-Environment Robot Control

### Objective
Create a system that controls a robot in Gazebo while visualizing in Unity and collecting perception data in Isaac Sim.

### Prerequisites
- Gazebo Garden installed and configured
- Unity 2023.2 LTS with Robotics packages
- NVIDIA Isaac Sim 2023.2 with Omniverse
- ROS 2 Humble Hawksbill

### Setup Instructions

1. **Create a ROS 2 workspace**:
   ```bash
   mkdir -p ~/digital_twin_ws/src
   cd ~/digital_twin_ws
   colcon build
   source install/setup.bash
   ```

2. **Create the exercise package**:
   ```bash
   cd ~/digital_twin_ws/src
   ros2 pkg create --build-type ament_python multi_env_exercise --dependencies rclpy std_msgs geometry_msgs sensor_msgs
   ```

3. **Create the main controller script** (`multi_env_exercise/multi_env_exercise/controller.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import JointState, LaserScan
   from std_msgs.msg import Float32MultiArray
   import numpy as np


   class MultiEnvironmentController(Node):
       def __init__(self):
           super().__init__('multi_env_controller')

           # Publishers for different simulators
           self.gazebo_cmd_pub = self.create_publisher(Twist, '/gazebo/cmd_vel', 10)
           self.unity_cmd_pub = self.create_publisher(Twist, '/unity/cmd_vel', 10)
           self.isaac_cmd_pub = self.create_publisher(Twist, '/isaac/cmd_vel', 10)

           # Subscribers for feedback from different simulators
           self.gazebo_odom_sub = self.create_subscription(JointState, '/gazebo/odom', self.odom_callback, 10)
           self.unity_scan_sub = self.create_subscription(LaserScan, '/unity/lidar_scan', self.scan_callback, 10)
           self.isaac_image_sub = self.create_subscription(Float32MultiArray, '/isaac/camera_data', self.image_callback, 10)

           # Timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

           # Robot state
           self.robot_position = [0.0, 0.0]
           self.robot_orientation = 0.0
           self.laser_scan = None
           self.camera_data = None

           self.get_logger().info('Multi-environment controller initialized')

       def control_loop(self):
           """Main control loop"""
           # Simple navigation behavior
           cmd_vel = Twist()

           # If obstacles detected close ahead, turn
           if self.laser_scan and min(self.laser_scan.ranges) < 1.0:
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = 0.5  # Turn right
           else:
               cmd_vel.linear.x = 0.5  # Move forward
               cmd_vel.angular.z = 0.0

           # Publish to all simulators
           self.gazebo_cmd_pub.publish(cmd_vel)
           self.unity_cmd_pub.publish(cmd_vel)
           self.isaac_cmd_pub.publish(cmd_vel)

           self.get_logger().info(f'Published cmd_vel: linear.x={cmd_vel.linear.x}, angular.z={cmd_vel.angular.z}')

       def odom_callback(self, msg):
           """Handle odometry feedback"""
           # Update robot position based on joint states
           if len(msg.position) >= 2:
               self.robot_position[0] = msg.position[0]
               self.robot_position[1] = msg.position[1]

       def scan_callback(self, msg):
           """Handle laser scan feedback"""
           self.laser_scan = msg

       def image_callback(self, msg):
           """Handle camera data feedback"""
           self.camera_data = msg.data


   def main(args=None):
       rclpy.init(args=args)
       controller = MultiEnvironmentController()

       try:
           rclpy.spin(controller)
       except KeyboardInterrupt:
           pass
       finally:
           controller.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py** (`multi_env_exercise/setup.py`):
   ```python
   from setuptools import find_packages, setup

   package_name = 'multi_env_exercise'

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
       description='Multi-environment simulation exercise',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'controller = multi_env_exercise.controller:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/digital_twin_ws
   colcon build --packages-select multi_env_exercise
   source install/setup.bash
   ```

### Exercise Steps

1. **Launch Gazebo simulation**:
   ```bash
   # Terminal 1
   ros2 launch gazebo_ros empty_world.launch.py
   ```

2. **Launch Unity simulation** (in Unity editor):
   - Open Unity project
   - Load the robot scene
   - Run the Unity simulation with ROS TCP connection

3. **Launch Isaac Sim**:
   ```bash
   # Terminal 2
   # Start Isaac Sim with a robot scene
   ```

4. **Run the controller**:
   ```bash
   # Terminal 3
   ros2 run multi_env_exercise controller
   ```

5. **Monitor the synchronization**:
   ```bash
   # Terminal 4
   ros2 topic echo /gazebo/odom
   ros2 topic echo /unity/lidar_scan
   ros2 topic echo /isaac/camera_data
   ```

### Expected Behavior
- Robot should move forward in all three environments
- When obstacle is detected, robot should turn in all environments
- States should be synchronized across environments

### Validation Criteria
- [ ] Robot moves consistently across all three environments
- [ ] Obstacle avoidance behavior works in all simulators
- [ ] State synchronization is maintained
- [ ] No errors in ROS communication

## Exercise 2: Perception Pipeline Integration

### Objective
Create a perception pipeline that combines sensor data from different simulators for improved accuracy.

### Setup Instructions

1. **Create perception package**:
   ```bash
   cd ~/digital_twin_ws/src
   ros2 pkg create --build-type ament_python perception_fusion --dependencies rclpy sensor_msgs geometry_msgs std_msgs cv_bridge message_filters
   ```

2. **Create perception fusion node** (`perception_fusion/perception_fusion/fusion_node.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan, PointCloud2
   from geometry_msgs.msg import PoseArray
   from std_msgs.msg import String
   from cv_bridge import CvBridge
   import numpy as np
   import json


   class PerceptionFusionNode(Node):
       def __init__(self):
           super().__init__('perception_fusion')

           # Initialize CV bridge
           self.bridge = CvBridge()

           # Subscribers for different sensor modalities
           self.gazebo_camera_sub = self.create_subscription(
               Image, '/gazebo/camera/image_raw', self.camera_callback, 10)
           self.unity_lidar_sub = self.create_subscription(
               LaserScan, '/unity/lidar_scan', self.lidar_callback, 10)
           self.isaac_depth_sub = self.create_subscription(
               Image, '/isaac/depth/image_raw', self.depth_callback, 10)

           # Publisher for fused perception
           self.detection_pub = self.create_publisher(String, '/fused_detections', 10)
           self.object_map_pub = self.create_publisher(PoseArray, '/object_map', 10)

           # Storage for sensor data
           self.camera_image = None
           self.lidar_data = None
           self.depth_image = None

           # Timer for fusion processing
           self.fusion_timer = self.create_timer(0.1, self.fusion_process)  # 10 Hz

           self.get_logger().info('Perception fusion node initialized')

       def camera_callback(self, msg):
           """Handle camera data from Gazebo"""
           try:
               self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
           except Exception as e:
               self.get_logger().error(f'Camera callback error: {e}')

       def lidar_callback(self, msg):
           """Handle LIDAR data from Unity"""
           self.lidar_data = msg

       def depth_callback(self, msg):
           """Handle depth data from Isaac Sim"""
           try:
               self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
           except Exception as e:
               self.get_logger().error(f'Depth callback error: {e}')

       def fusion_process(self):
           """Process and fuse sensor data"""
           if not all([self.camera_image, self.lidar_data, self.depth_image]):
               return  # Wait for all data

           # Perform sensor fusion
           detections = self.fuse_camera_lidar()
           object_map = self.create_object_map(detections)

           # Publish fused results
           detection_msg = String()
           detection_msg.data = json.dumps(detections)
           self.detection_pub.publish(detection_msg)

           self.object_map_pub.publish(object_map)

           self.get_logger().info(f'Fused {len(detections)} detections')

       def fuse_camera_lidar(self):
           """Fuse camera and LIDAR detections"""
           detections = []

           # Simulate object detection in camera
           camera_detections = self.detect_objects_in_camera()
           lidar_detections = self.process_lidar_data()

           # Associate camera and LIDAR detections
           for cam_det in camera_detections:
               for lidar_det in lidar_detections:
                   # Simple association based on proximity
                   if self.associate_detections(cam_det, lidar_det):
                       fused_detection = {
                           'id': len(detections),
                           'class': cam_det['class'],
                           'confidence': (cam_det['confidence'] + lidar_det['confidence']) / 2,
                           'position': lidar_det['position'],
                           'bbox': cam_det['bbox'],
                           'method': 'camera_lidar_fusion'
                       }
                       detections.append(fused_detection)

           return detections

       def detect_objects_in_camera(self):
           """Simulate object detection in camera image"""
           # In a real implementation, this would run a neural network
           # For this exercise, we'll simulate detections
           detections = []

           # Simulate detecting 1-3 objects
           num_objects = np.random.randint(1, 4)
           for i in range(num_objects):
               detection = {
                   'class': 'object',
                   'confidence': np.random.uniform(0.7, 1.0),
                   'bbox': [np.random.randint(0, 640), np.random.randint(0, 480),
                           np.random.randint(50, 200), np.random.randint(50, 200)],
                   'center': [np.random.randint(0, 640), np.random.randint(0, 480)]
               }
               detections.append(detection)

           return detections

       def process_lidar_data(self):
           """Process LIDAR data for object detection"""
           if not self.lidar_data:
               return []

           # In a real implementation, this would run clustering algorithms
           # For this exercise, we'll simulate detections
           detections = []

           # Extract valid ranges
           ranges = [r for r in self.lidar_data.ranges if not np.isnan(r) and r < self.lidar_data.range_max]

           if len(ranges) > 10:  # Enough valid readings
               # Simulate detecting obstacles
               for i in range(min(5, len(ranges)//20)):  # Max 5 detections
                   idx = np.random.randint(0, len(ranges))
                   distance = ranges[idx]
                   angle = self.lidar_data.angle_min + idx * self.lidar_data.angle_increment

                   x = distance * np.cos(angle)
                   y = distance * np.sin(angle)

                   detection = {
                       'position': [x, y, 0.0],
                       'confidence': np.random.uniform(0.6, 0.9),
                       'distance': distance
                   }
                   detections.append(detection)

           return detections

       def associate_detections(self, cam_det, lidar_det):
           """Associate camera and LIDAR detections"""
           # Simple association based on geometric consistency
           # This is a simplified version - real implementation would be more sophisticated
           cam_center_x = cam_det['center'][0]
           cam_center_y = cam_det['center'][1]

           # Convert image coordinates to world coordinates approximately
           # (simplified for this exercise)
           expected_distance = lidar_det['distance']
           if expected_distance < 5.0:  # Reasonable distance
               return True

           return False

       def create_object_map(self, detections):
           """Create object map from detections"""
           pose_array = PoseArray()
           pose_array.header.stamp = self.get_clock().now().to_msg()
           pose_array.header.frame_id = "map"

           for det in detections:
               if det['confidence'] > 0.7:  # Confidence threshold
                   pose = Pose()
                   pose.position.x = det['position'][0] if 'position' in det else 0.0
                   pose.position.y = det['position'][1] if 'position' in det else 0.0
                   pose.position.z = det['position'][2] if 'position' in det else 0.0
                   pose.orientation.w = 1.0  # No rotation
                   pose_array.poses.append(pose)

           return pose_array


   def main(args=None):
       rclpy.init(args=args)
       fusion_node = PerceptionFusionNode()

       try:
           rclpy.spin(fusion_node)
       except KeyboardInterrupt:
           pass
       finally:
           fusion_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Update setup.py** for perception fusion package:
   ```python
   from setuptools import find_packages, setup

   package_name = 'perception_fusion'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools', 'opencv-python'],
       zip_safe=True,
       maintainer='your_name',
       maintainer_email='your_email@example.com',
       description='Perception fusion for multi-environment simulation',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'fusion_node = perception_fusion.fusion_node:main',
           ],
       },
   )
   ```

4. **Build the perception fusion package**:
   ```bash
   cd ~/digital_twin_ws
   colcon build --packages-select perception_fusion
   source install/setup.bash
   ```

### Exercise Steps

1. **Start all simulation environments**:
   ```bash
   # Terminal 1: Gazebo
   ros2 launch gazebo_ros empty_world.launch.py

   # Terminal 2: Unity (with ROS connection)
   # Launch Unity scene with robot and sensors

   # Terminal 3: Isaac Sim
   # Launch Isaac Sim with robot and depth camera
   ```

2. **Run the fusion node**:
   ```bash
   # Terminal 4
   ros2 run perception_fusion fusion_node
   ```

3. **Monitor the fusion results**:
   ```bash
   # Terminal 5
   ros2 topic echo /fused_detections
   ros2 topic echo /object_map
   ```

### Expected Behavior
- Sensor data from different environments should be fused
- More accurate detections should result from fusion
- Object map should show combined detections

### Validation Criteria
- [ ] Sensor data is received from all sources
- [ ] Fusion algorithm produces combined detections
- [ ] Confidence values are appropriately calculated
- [ ] Object map reflects fused detections

## Exercise 3: Performance Optimization Challenge

### Objective
Optimize a multi-environment simulation for maximum performance while maintaining acceptable accuracy.

### Setup Instructions

1. **Create optimization package**:
   ```bash
   cd ~/digital_twin_ws/src
   ros2 pkg create --build-type ament_python perf_optimization --dependencies rclpy std_msgs sensor_msgs
   ```

2. **Create performance monitor** (`perf_optimization/perf_optimization/monitor.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32, String
   import time
   import psutil
   import subprocess
   import json


   class PerformanceMonitor(Node):
       def __init__(self):
           super().__init__('performance_monitor')

           # Publishers for performance metrics
           self.cpu_pub = self.create_publisher(Float32, '/performance/cpu_usage', 10)
           self.memory_pub = self.create_publisher(Float32, '/performance/memory_usage', 10)
           self.gpu_pub = self.create_publisher(Float32, '/performance/gpu_usage', 10)
           self.fps_pub = self.create_publisher(Float32, '/performance/simulation_fps', 10)
           self.report_pub = self.create_publisher(String, '/performance/report', 10)

           # Timer for monitoring
           self.monitor_timer = self.create_timer(1.0, self.monitor_performance)

           # Performance tracking
           self.frame_times = []
           self.last_frame_time = None

           self.get_logger().info('Performance monitor initialized')

       def monitor_performance(self):
           """Monitor system performance"""
           # CPU usage
           cpu_percent = psutil.cpu_percent()
           cpu_msg = Float32()
           cpu_msg.data = float(cpu_percent)
           self.cpu_pub.publish(cpu_msg)

           # Memory usage
           memory_percent = psutil.virtual_memory().percent
           memory_msg = Float32()
           memory_msg.data = float(memory_percent)
           self.memory_pub.publish(memory_msg)

           # GPU usage (if available)
           gpu_percent = self.get_gpu_usage()
           if gpu_percent >= 0:
               gpu_msg = Float32()
               gpu_msg.data = float(gpu_percent)
               self.gpu_pub.publish(gpu_msg)

           # Calculate FPS if possible
           fps = self.calculate_current_fps()
           if fps >= 0:
               fps_msg = Float32()
               fps_msg.data = float(fps)
               self.fps_pub.publish(fps_msg)

           # Log performance
           self.get_logger().info(
               f'Performance: CPU={cpu_percent:.1f}%, '
               f'Memory={memory_percent:.1f}%, '
               f'GPU={gpu_percent:.1f}%, '
               f'FPS={fps:.1f}'
           )

           # Generate performance report if needed
           if cpu_percent > 80 or memory_percent > 80:
               self.generate_performance_report(cpu_percent, memory_percent, gpu_percent, fps)

       def get_gpu_usage(self):
           """Get GPU usage percentage using nvidia-smi"""
           try:
               result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True)
               if result.returncode == 0:
                   gpu_usage = float(result.stdout.strip())
                   return gpu_usage
           except Exception:
               pass
           return -1  # GPU not available or nvidia-smi not found

       def calculate_current_fps(self):
           """Calculate current simulation FPS"""
           # This would be more accurate with simulation-specific timing
           # For this exercise, we'll simulate FPS calculation
           return 60.0  # Simulated FPS

       def generate_performance_report(self, cpu, memory, gpu, fps):
           """Generate detailed performance report"""
           report = {
               'timestamp': time.time(),
               'cpu_usage': cpu,
               'memory_usage': memory,
               'gpu_usage': gpu,
               'fps': fps,
               'recommendations': []
           }

           # Add recommendations based on performance
           if cpu > 80:
               report['recommendations'].append('Consider reducing physics complexity or update rate')
           if memory > 80:
               report['recommendations'].append('Optimize asset loading or reduce scene complexity')
           if gpu > 80:
               report['recommendations'].append('Reduce rendering quality or resolution')
           if fps < 30:
               report['recommendations'].append('Increase time step or reduce simulation complexity')

           # Publish report
           report_msg = String()
           report_msg.data = json.dumps(report, indent=2)
           self.report_pub.publish(report_msg)


   def main(args=None):
       rclpy.init(args=args)
       monitor = PerformanceMonitor()

       try:
           rclpy.spin(monitor)
       except KeyboardInterrupt:
           pass
       finally:
           monitor.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create optimization controller** (`perf_optimization/perf_optimization/optimizer.py`):
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32, String
   import json


   class PerformanceOptimizer(Node):
       def __init__(self):
           super().__init__('performance_optimizer')

           # Subscriptions for performance metrics
           self.cpu_sub = self.create_subscription(Float32, '/performance/cpu_usage', self.cpu_callback, 10)
           self.memory_sub = self.create_subscription(Float32, '/performance/memory_usage', self.memory_callback, 10)
           self.gpu_sub = self.create_subscription(Float32, '/performance/gpu_usage', self.gpu_callback, 10)
           self.fps_sub = self.create_subscription(Float32, '/performance/simulation_fps', self.fps_callback, 10)

           # Publishers for optimization commands
           self.quality_pub = self.create_publisher(String, '/optimization/quality_setting', 10)
           self.physics_pub = self.create_publisher(String, '/optimization/physics_setting', 10)
           self.rendering_pub = self.create_publisher(String, '/optimization/rendering_setting', 10)

           # Performance thresholds
           self.cpu_threshold = 70.0
           self.memory_threshold = 70.0
           self.gpu_threshold = 75.0
           self.fps_threshold = 30.0

           # Current settings
           self.current_quality = "high"
           self.current_physics = "accurate"
           self.current_rendering = "photorealistic"

           self.get_logger().info('Performance optimizer initialized')

       def cpu_callback(self, msg):
           """Handle CPU usage updates"""
           cpu_usage = msg.data
           if cpu_usage > self.cpu_threshold:
               self.adjust_for_cpu_load(cpu_usage)

       def memory_callback(self, msg):
           """Handle memory usage updates"""
           memory_usage = msg.data
           if memory_usage > self.memory_threshold:
               self.adjust_for_memory_load(memory_usage)

       def gpu_callback(self, msg):
           """Handle GPU usage updates"""
           gpu_usage = msg.data
           if gpu_usage > self.gpu_threshold:
               self.adjust_for_gpu_load(gpu_usage)

       def fps_callback(self, msg):
           """Handle FPS updates"""
           fps = msg.data
           if fps < self.fps_threshold:
               self.adjust_for_performance(fps)

       def adjust_for_cpu_load(self, cpu_usage):
           """Adjust settings based on CPU load"""
           if cpu_usage > 90:
               # Significantly reduce complexity
               self.current_physics = "fast"
               self.publish_physics_setting("fast")
               self.get_logger().warn(f'High CPU load ({cpu_usage}%), switched to fast physics')
           elif cpu_usage > 75:
               # Moderately reduce complexity
               if self.current_physics == "accurate":
                   self.current_physics = "balanced"
                   self.publish_physics_setting("balanced")
                   self.get_logger().info(f'High CPU load ({cpu_usage}%), adjusted physics settings')

       def adjust_for_memory_load(self, memory_usage):
           """Adjust settings based on memory load"""
           if memory_usage > 85:
               # Reduce asset quality
               if self.current_quality == "high":
                   self.current_quality = "medium"
                   self.publish_quality_setting("medium")
                   self.get_logger().warn(f'High memory usage ({memory_usage}%), reduced quality')
           elif memory_usage > 75:
               # Moderate reduction
               if self.current_quality == "high":
                   self.current_quality = "medium-high"
                   self.publish_quality_setting("medium-high")
                   self.get_logger().info(f'Moderate memory usage ({memory_usage}%), adjusted quality')

       def adjust_for_gpu_load(self, gpu_usage):
           """Adjust settings based on GPU load"""
           if gpu_usage > 85:
               # Reduce rendering quality significantly
               self.current_rendering = "performance"
               self.publish_rendering_setting("performance")
               self.get_logger().warn(f'High GPU load ({gpu_usage}%), switched to performance rendering')
           elif gpu_usage > 75:
               # Moderate reduction
               if self.current_rendering == "photorealistic":
                   self.current_rendering = "balanced"
                   self.publish_rendering_setting("balanced")
                   self.get_logger().info(f'High GPU load ({gpu_usage}%), adjusted rendering')

       def adjust_for_performance(self, fps):
           """Adjust settings to improve FPS"""
           target_fps = 60.0
           ratio = fps / target_fps

           if ratio < 0.5:  # Less than half target
               # Major adjustments needed
               if self.current_rendering != "performance":
                   self.current_rendering = "performance"
                   self.publish_rendering_setting("performance")
               if self.current_physics == "accurate":
                   self.current_physics = "fast"
                   self.publish_physics_setting("fast")
               self.get_logger().warn(f'Very low FPS ({fps}), made major optimizations')
           elif ratio < 0.75:  # Below 75% of target
               # Moderate adjustments
               if self.current_rendering == "photorealistic":
                   self.current_rendering = "balanced"
                   self.publish_rendering_setting("balanced")
               self.get_logger().info(f'Low FPS ({fps}), adjusted settings for performance')

       def publish_quality_setting(self, setting):
           """Publish quality setting adjustment"""
           msg = String()
           msg.data = setting
           self.quality_pub.publish(msg)

       def publish_physics_setting(self, setting):
           """Publish physics setting adjustment"""
           msg = String()
           msg.data = setting
           self.physics_pub.publish(msg)

       def publish_rendering_setting(self, setting):
           """Publish rendering setting adjustment"""
           msg = String()
           msg.data = setting
           self.rendering_pub.publish(msg)


   def main(args=None):
       rclpy.init(args=args)
       optimizer = PerformanceOptimizer()

       try:
           rclpy.spin(optimizer)
       except KeyboardInterrupt:
           pass
       finally:
           optimizer.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py**:
   ```python
   from setuptools import find_packages, setup

   package_name = 'perf_optimization'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools', 'psutil'],
       zip_safe=True,
       maintainer='your_name',
       maintainer_email='your_email@example.com',
       description='Performance optimization for multi-environment simulation',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'monitor = perf_optimization.monitor:main',
               'optimizer = perf_optimization.optimizer:main',
           ],
       },
   )
   ```

5. **Build the optimization package**:
   ```bash
   cd ~/digital_twin_ws
   colcon build --packages-select perf_optimization
   source install/setup.bash
   ```

### Exercise Steps

1. **Start simulation environments with high complexity**:
   ```bash
   # Launch complex simulation scenarios in all three environments
   ```

2. **Run the performance monitor**:
   ```bash
   ros2 run perf_optimization monitor
   ```

3. **Run the performance optimizer**:
   ```bash
   ros2 run perf_optimization optimizer
   ```

4. **Monitor the optimization process**:
   ```bash
   ros2 topic echo /performance/report
   ros2 topic echo /optimization/quality_setting
   ros2 topic echo /optimization/physics_setting
   ros2 topic echo /optimization/rendering_setting
   ```

### Expected Behavior
- Performance monitor should detect high resource usage
- Optimizer should automatically adjust settings
- Simulation should maintain acceptable performance
- Resource usage should decrease after optimization

### Validation Criteria
- [ ] Performance monitor detects high resource usage
- [ ] Optimizer adjusts settings appropriately
- [ ] Simulation remains stable after optimization
- [ ] Resource usage decreases as expected

## Exercise 4: Cross-Environment Validation

### Objective
Validate that robot behaviors are consistent across all three simulation environments.

### Exercise Steps

1. **Create a simple navigation task**:
   - Define a start position in each environment
   - Define a goal position in each environment
   - Implement the same navigation algorithm in each environment

2. **Run the same navigation scenario in all environments**:
   - Same starting conditions
   - Same obstacles
   - Same navigation parameters

3. **Compare the results**:
   - Path taken should be similar
   - Time to goal should be comparable
   - Behavior should be consistent

4. **Document differences**:
   - Note any discrepancies in behavior
   - Identify sources of differences
   - Propose solutions for consistency

### Validation Criteria
- [ ] Robot behavior is consistent across environments
- [ ] Navigation paths are similar
- [ ] Performance metrics are comparable
- [ ] Differences are documented and explained

## Exercise 5: Hybrid AI Training Pipeline

### Objective
Create a training pipeline that uses Isaac Sim for data generation, Gazebo for physics validation, and Unity for visualization.

### Exercise Steps

1. **Set up Isaac Sim for synthetic data generation**:
   - Configure randomization for diverse scenarios
   - Set up sensor simulation
   - Create data collection pipeline

2. **Set up Gazebo for physics validation**:
   - Implement the trained model in Gazebo
   - Test physics accuracy
   - Validate robot dynamics

3. **Set up Unity for visualization**:
   - Visualize training progress
   - Display performance metrics
   - Create dashboards for monitoring

4. **Connect all environments via ROS 2**:
   - Share models and parameters
   - Transfer data between environments
   - Synchronize training and validation

### Validation Criteria
- [ ] Training data is successfully generated in Isaac Sim
- [ ] Model performs correctly in Gazebo
- [ ] Results are visualized in Unity
- [ ] Performance is consistent across environments

## Exercise Completion Checklist

After completing these exercises, you should be able to:
- [ ] Integrate multiple simulation environments using ROS 2
- [ ] Implement sensor fusion across different simulators
- [ ] Optimize simulation performance dynamically
- [ ] Validate consistency across simulation environments
- [ ] Create hybrid AI training pipelines

## Advanced Challenge

For advanced practitioners, try creating a complete digital twin system that:
1. Continuously synchronizes between physical robot and all simulators
2. Adapts simulation parameters based on real-world performance
3. Implements closed-loop learning between real and simulated environments
4. Provides real-time performance optimization across all platforms

## Key Takeaways

- Multi-environment simulation leverages the strengths of each platform
- Proper synchronization is essential for consistent behavior
- Performance optimization requires dynamic adaptation
- Cross-validation ensures simulation accuracy
- ROS 2 provides the backbone for integration

These exercises provide hands-on experience with the key concepts of digital twin simulation for humanoid robotics, preparing you to create complex, integrated simulation systems for your own projects.