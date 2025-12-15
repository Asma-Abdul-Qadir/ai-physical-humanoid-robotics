---
sidebar_position: 3
---

# Capstone Project Implementation: Step-by-Step Guide

This chapter provides a comprehensive step-by-step implementation guide for the capstone project, walking through the development of the integrated humanoid robotics system. Each step builds upon previous work, ensuring a systematic approach to creating the complete system.

## Implementation Overview

The implementation follows the planning framework established in the previous chapter, with each step carefully designed to build toward the final integrated system. This guide emphasizes practical implementation while maintaining focus on integration requirements and success criteria.

### Prerequisites Check

Before beginning implementation, verify the following prerequisites are met:

1. **Development Environment**: Ubuntu 22.04 LTS with ROS 2 Humble Hawksbill installed
2. **Dependencies**: All required packages and libraries installed as per project specifications
3. **Simulation Environment**: Gazebo Garden configured and tested
4. **Robot Model**: Humanoid robot model loaded and functional in simulation
5. **Basic Modules**: Individual modules from previous chapters are available

## Step 1: Project Setup and Workspace Configuration

### 1.1 Create ROS 2 Workspace

```bash
# Create workspace directory
mkdir -p ~/capstone_project/src
cd ~/capstone_project

# Initialize workspace
colcon build
source install/setup.bash
```

### 1.2 Create Capstone Project Packages

```bash
cd ~/capstone_project/src

# Create main capstone package
ros2 pkg create --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs moveit_ros_planning_interface -- python_pkg capstone_main

# Create perception package
ros2 pkg create --dependencies rclpy sensor_msgs cv_bridge message_filters -- python_pkg capstone_perception

# Create language package
ros2 pkg create --dependencies rclpy std_msgs audio_common_msgs -- python_pkg capstone_language

# Create action package
ros2 pkg create --dependencies rclpy geometry_msgs nav_msgs moveit_ros_planning_interface -- python_pkg capstone_action

# Create safety package
ros2 pkg create --dependencies rclpy sensor_msgs geometry_msgs -- python_pkg capstone_safety

# Create fusion package
ros2 pkg create --dependencies rclpy std_msgs sensor_msgs geometry_msgs -- python_pkg capstone_fusion
```

### 1.3 Configure Package Structure

Create the basic directory structure for each package:

```bash
# For each package, create necessary directories
for pkg in capstone_main capstone_perception capstone_language capstone_action capstone_safety capstone_fusion; do
    mkdir -p ~/capstone_project/src/$pkg/capstone_$pkg/nodes
    mkdir -p ~/capstone_project/src/$pkg/capstone_$pkg/utils
    mkdir -p ~/capstone_project/src/$pkg/test
done
```

## Step 2: Define Custom Message Types

### 2.1 Create Perception Messages

Create `~/capstone_project/src/capstone_perception/msg/ObjectDetection.msg`:

```
# ObjectDetection.msg
string class
float64 confidence
int32[4] bbox  # [x, y, width, height]
float64[3] position  # [x, y, z]
float64[4] orientation  # [x, y, z, w]
```

Create `~/capstone_project/src/capstone_perception/msg/SceneDescription.msg`:

```
# SceneDescription.msg
string room_type
string activity
ObjectDetection[] objects
float64[3] robot_position
float64 timestamp
```

### 2.2 Create Language Messages

Create `~/capstone_project/src/capstone_language/msg/Command.msg`:

```
# Command.msg
string intent
string[] entities
string original_text
float64 confidence
string[] parameters
```

Create `~/capstone_project/src/capstone_language/msg/Response.msg`:

```
# Response.msg
string text
string intent
float64 confidence
string[] suggestions
bool needs_clarification
```

### 2.3 Create Action Messages

Create `~/capstone_project/src/capstone_action/msg/ActionResult.msg`:

```
# ActionResult.msg
string action_type
bool success
string error_message
float64 execution_time
string details
```

Create `~/capstone_project/src/capstone_action/msg/Task.msg`:

```
# Task.msg
string type
string[] parameters
float64 priority
bool blocking
string id
```

### 2.4 Update Package.xml Files

Update the `package.xml` files to include message dependencies:

```xml
<!-- In each package.xml that uses custom messages -->
<depend>capstone_perception_msgs</depend>
<depend>capstone_language_msgs</depend>
<depend>capstone_action_msgs</depend>
```

Update the `setup.py` files to export message types:

```python
# In setup.py for packages that define messages
from setuptools import setup
from glob import glob
import os

package_name = 'capstone_perception'

setup(
    name=package_name,
    # ... other setup parameters ...
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Export message types
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    # ... other setup parameters ...
)
```

## Step 3: Implement Perception Module

### 3.1 Create Perception Node

Create `~/capstone_project/src/capstone_perception/capstone_perception/perception_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from capstone_perception.msg import ObjectDetection, SceneDescription
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
import message_filters


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_pub = self.create_publisher(ObjectDetection, 'object_detections', 10)
        self.scene_pub = self.create_publisher(SceneDescription, 'scene_description', 10)

        # Subscribers with message filters for synchronization
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronize image and depth
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

        # Object detection parameters
        self.confidence_threshold = 0.5
        self.object_classes = [
            'person', 'chair', 'table', 'cup', 'bottle',
            'phone', 'book', 'laptop', 'door', 'obstacle'
        ]

        # Initialize perception components
        self.object_detector = self.initialize_object_detector()
        self.scene_analyzer = SceneAnalyzer()

        self.get_logger().info('Perception node initialized')

    def initialize_object_detector(self):
        """Initialize object detection model (simulated for this example)"""
        # In a real implementation, this would load a YOLO, SSD, or similar model
        # For simulation, return a mock detector
        return MockObjectDetector()

    def sync_callback(self, image_msg, depth_msg):
        """Process synchronized image and depth data"""
        try:
            # Convert ROS images to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Perform object detection
            detections = self.object_detector.detect(cv_image, self.confidence_threshold)

            # Analyze scene
            scene_description = self.scene_analyzer.analyze(cv_image, detections)

            # Publish detections
            for detection in detections:
                detection_msg = ObjectDetection()
                detection_msg.class = detection['class']
                detection_msg.confidence = detection['confidence']
                detection_msg.bbox = detection['bbox']
                detection_msg.position = detection['position']
                detection_msg.orientation = [0.0, 0.0, 0.0, 1.0]  # Default orientation

                self.detection_pub.publish(detection_msg)

            # Publish scene description
            scene_msg = SceneDescription()
            scene_msg.room_type = scene_description['room_type']
            scene_msg.activity = scene_description['activity']
            scene_msg.objects = [self.create_object_detection_msg(det) for det in detections]
            scene_msg.robot_position = [0.0, 0.0, 0.0]  # Placeholder
            scene_msg.timestamp = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9

            self.scene_pub.publish(scene_msg)

        except Exception as e:
            self.get_logger().error(f'Error in perception callback: {e}')

    def create_object_detection_msg(self, detection_dict):
        """Convert detection dictionary to message"""
        msg = ObjectDetection()
        msg.class = detection_dict['class']
        msg.confidence = detection_dict['confidence']
        msg.bbox = detection_dict['bbox']
        msg.position = detection_dict['position']
        msg.orientation = [0.0, 0.0, 0.0, 1.0]
        return msg


class MockObjectDetector:
    """Mock object detector for simulation"""
    def __init__(self):
        self.classes = ['person', 'chair', 'table', 'cup', 'bottle']

    def detect(self, image, threshold=0.5):
        """Simulate object detection"""
        height, width = image.shape[:2]
        detections = []

        # Simulate detecting some objects
        for i in range(np.random.randint(1, 4)):
            class_name = np.random.choice(self.classes)
            x = np.random.randint(0, width // 2)
            y = np.random.randint(0, height // 2)
            w = np.random.randint(width // 8, width // 4)
            h = np.random.randint(height // 8, height // 4)
            confidence = np.random.uniform(threshold, 0.95)

            detection = {
                'class': class_name,
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(confidence),
                'position': [float(x + w/2), float(y + h/2), 0.0]  # Placeholder depth
            }

            detections.append(detection)

        return detections


class SceneAnalyzer:
    """Analyze scene context"""
    def __init__(self):
        self.room_types = ['kitchen', 'living_room', 'bedroom', 'office', 'hallway']

    def analyze(self, image, detections):
        """Analyze scene and return context"""
        room_type = np.random.choice(self.room_types)

        # Infer activity based on detected objects
        person_count = sum(1 for det in detections if det['class'] == 'person')
        if person_count > 0:
            activity = 'social_interaction' if len(detections) > 1 else 'presence_detected'
        else:
            activity = 'inactive'

        return {
            'room_type': room_type,
            'activity': activity,
            'object_count': len(detections)
        }


def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3.2 Create Perception Launch File

Create `~/capstone_project/src/capstone_perception/launch/perception_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[
                {'confidence_threshold': 0.5}
            ]
        )
    ])
```

### 3.3 Update Setup.py for Perception Package

Update `~/capstone_project/src/capstone_perception/setup.py`:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'capstone_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='capstone@todo.todo',
    description='Capstone Project Perception Module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = capstone_perception.perception_node:main',
        ],
    },
)
```

## Step 4: Implement Language Processing Module

### 4.1 Create Language Node

Create `~/capstone_project/src/capstone_language/capstone_language/language_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
from capstone_language.msg import Command, Response
from capstone_perception.msg import SceneDescription
import numpy as np
import threading
import queue
import time
import re


class LanguageNode(Node):
    def __init__(self):
        super().__init__('language_node')

        # Publishers
        self.command_pub = self.create_publisher(Command, 'parsed_commands', 10)
        self.response_pub = self.create_publisher(Response, 'responses', 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String, 'speech_recognition', self.speech_callback, 10)
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)

        # Internal state
        self.current_scene = None
        self.conversation_context = []

        # Initialize language processing components
        self.command_parser = CommandParser()
        self.response_generator = ResponseGenerator()

        self.get_logger().info('Language node initialized')

    def speech_callback(self, msg):
        """Process incoming speech recognition results"""
        text = msg.data.lower().strip()
        self.get_logger().info(f'Received speech: {text}')

        # Add to conversation context
        self.conversation_context.append({
            'type': 'user_input',
            'text': text,
            'timestamp': time.time()
        })

        # Parse the command
        parsed_command = self.command_parser.parse(text, self.current_scene)

        if parsed_command:
            # Publish parsed command
            command_msg = Command()
            command_msg.intent = parsed_command['intent']
            command_msg.entities = parsed_command['entities']
            command_msg.original_text = text
            command_msg.confidence = parsed_command['confidence']
            command_msg.parameters = parsed_command.get('parameters', [])

            self.command_pub.publish(command_msg)

            # Generate response
            response = self.response_generator.generate(parsed_command, self.current_scene)

            response_msg = Response()
            response_msg.text = response['text']
            response_msg.intent = parsed_command['intent']
            response_msg.confidence = response['confidence']
            response_msg.suggestions = response.get('suggestions', [])
            response_msg.needs_clarification = response.get('needs_clarification', False)

            self.response_pub.publish(response_msg)
        else:
            # Generate error response
            error_response = self.response_generator.generate_error_response(text)

            response_msg = Response()
            response_msg.text = error_response['text']
            response_msg.intent = 'unknown'
            response_msg.confidence = error_response['confidence']
            response_msg.suggestions = error_response.get('suggestions', [])
            response_msg.needs_clarification = True

            self.response_pub.publish(response_msg)

    def scene_callback(self, msg):
        """Update current scene context"""
        self.current_scene = {
            'room_type': msg.room_type,
            'activity': msg.activity,
            'objects': [obj.class for obj in msg.objects],
            'object_count': len(msg.objects)
        }


class CommandParser:
    def __init__(self):
        # Define command patterns and their corresponding intents
        self.command_patterns = {
            'navigation': [
                (r'go to (the )?(?P<destination>\w+)', 'navigate_to_destination'),
                (r'move to (the )?(?P<destination>\w+)', 'navigate_to_destination'),
                (r'walk to (the )?(?P<destination>\w+)', 'navigate_to_destination'),
                (r'go (forward|backward|left|right)', 'move_direction'),
                (r'come (here|to me|over)', 'come_to_user'),
                (r'follow me', 'follow_user'),
            ],
            'manipulation': [
                (r'get (me )?(the )?(?P<object>\w+)', 'fetch_object'),
                (r'bring (me )?(the )?(?P<object>\w+)', 'fetch_object'),
                (r'pick up (the )?(?P<object>\w+)', 'fetch_object'),
                (r'grab (the )?(?P<object>\w+)', 'fetch_object'),
                (r'hand me (the )?(?P<object>\w+)', 'hand_object'),
            ],
            'information': [
                (r'what (is|are) (there|in here)', 'describe_environment'),
                (r'tell me about (the )?surroundings', 'describe_environment'),
                (r'how many people', 'count_people'),
                (r'what time is it', 'get_time'),
            ],
            'social': [
                (r'hello|hi|hey', 'greet'),
                (r'goodbye|bye|see you', 'farewell'),
                (r'thank you|thanks', 'acknowledge_gratitude'),
            ]
        }

        # Define entity mappings
        self.entity_synonyms = {
            'kitchen': ['cooking', 'food', 'eat'],
            'living_room': ['sofa', 'couch', 'tv', 'relax'],
            'bedroom': ['sleep', 'bed', 'rest'],
            'office': ['work', 'computer', 'desk'],
            'cup': ['mug', 'glass', 'drink'],
            'bottle': ['water', 'drink', 'liquid'],
            'person': ['human', 'man', 'woman', 'people']
        }

    def parse(self, text, scene_context=None):
        """Parse natural language command"""
        text_lower = text.lower()

        # Try to match against each intent type
        for intent_type, patterns in self.command_patterns.items():
            for pattern, action in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract entities from the match
                    entities = match.groupdict()

                    # Enhance entities with context if available
                    if scene_context:
                        entities = self.enhance_entities_with_context(entities, scene_context)

                    # Calculate confidence based on match quality
                    confidence = self.calculate_confidence(pattern, match, text_lower)

                    return {
                        'intent': intent_type,
                        'entities': entities,
                        'confidence': confidence,
                        'action': action,
                        'original_text': text
                    }

        # If no pattern matches, return None
        return None

    def enhance_entities_with_context(self, entities, scene_context):
        """Enhance entities using scene context"""
        enhanced = entities.copy()

        # Map synonyms based on context
        for entity_key, entity_value in entities.items():
            for main_value, synonyms in self.entity_synonyms.items():
                if entity_value in synonyms:
                    enhanced[entity_key] = main_value
                    break

        return enhanced

    def calculate_confidence(self, pattern, match, text):
        """Calculate confidence score for the match"""
        base_confidence = 0.7

        # Boost confidence for longer, more specific matches
        match_length = len(match.group(0))
        text_length = len(text)

        if text_length > 0:
            length_ratio = match_length / text_length
            base_confidence += min(0.2, length_ratio * 0.1)

        # Additional boosts based on match quality
        if match.group(0).strip() == text.strip():
            # Exact match of the whole text
            base_confidence += 0.1

        return min(1.0, base_confidence)


class ResponseGenerator:
    def __init__(self):
        self.response_templates = {
            'navigation': [
                "Okay, I'll go to the {destination}.",
                "Navigating to the {destination} now.",
                "On my way to the {destination}."
            ],
            'manipulation': [
                "I'll get the {object} for you.",
                "Fetching the {object} now.",
                "Going to get the {object}."
            ],
            'information': {
                'describe_environment': "I can see several objects around us, including {objects}.",
                'count_people': "I see {count} people in the area.",
                'get_time': "The current time is {time}."
            },
            'social': {
                'greet': "Hello! How can I help you today?",
                'farewell': "Goodbye! It was nice talking with you.",
                'acknowledge_gratitude': "You're welcome! I'm happy to help."
            }
        }

    def generate(self, parsed_command, scene_context=None):
        """Generate appropriate response"""
        intent = parsed_command['intent']
        entities = parsed_command['entities']

        if intent in self.response_templates:
            if isinstance(self.response_templates[intent], list):
                # Use template list
                import random
                template = random.choice(self.response_templates[intent])
                response_text = template.format(**entities)
            elif isinstance(self.response_templates[intent], dict):
                # Use template dict with action mapping
                action = parsed_command['action']
                if action in self.response_templates[intent]:
                    template = self.response_templates[intent][action]
                    if '{objects}' in template and scene_context:
                        objects_str = ', '.join(scene_context.get('objects', []))
                        response_text = template.format(objects=objects_str)
                    elif '{count}' in template and scene_context:
                        count = scene_context.get('object_count', 0)
                        response_text = template.format(count=count)
                    elif '{time}' in template:
                        from datetime import datetime
                        current_time = datetime.now().strftime("%H:%M")
                        response_text = template.format(time=current_time)
                    else:
                        response_text = template
                else:
                    response_text = f"I understand you want me to {action.replace('_', ' ')}."
            else:
                response_text = f"I understand you want me to {intent}."
        else:
            response_text = "I understand your request."

        return {
            'text': response_text,
            'confidence': parsed_command['confidence'],
            'needs_clarification': False
        }

    def generate_error_response(self, text):
        """Generate response for unrecognized commands"""
        return {
            'text': f"I didn't understand '{text}'. Could you please rephrase?",
            'confidence': 0.1,
            'needs_clarification': True,
            'suggestions': [
                "Try commands like 'go to kitchen', 'get the cup', or 'what do you see'"
            ]
        }


def main(args=None):
    rclpy.init(args=args)
    language_node = LanguageNode()

    try:
        rclpy.spin(language_node)
    except KeyboardInterrupt:
        pass
    finally:
        language_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 4.2 Create Language Launch File

Create `~/capstone_project/src/capstone_language/launch/language_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_language',
            executable='language_node',
            name='language_node',
            output='screen'
        )
    ])
```

### 4.3 Update Setup.py for Language Package

Update `~/capstone_project/src/capstone_language/setup.py`:

```python
from setuptools import setup

package_name = 'capstone_language'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/language_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='capstone@todo.todo',
    description='Capstone Project Language Processing Module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'language_node = capstone_language.language_node:main',
        ],
    },
)
```

## Step 5: Implement Action Module

### 5.1 Create Action Node

Create `~/capstone_project/src/capstone_action/capstone_action/action_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Path
from capstone_action.msg import Command, ActionResult, Task
from capstone_perception.msg import SceneDescription
from std_msgs.msg import String
import numpy as np
import time
from enum import Enum
from typing import Dict, Any, Optional


class ActionState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionNode(Node):
    def __init__(self):
        super().__init__('action_node')

        # Publishers
        self.navigation_pub = self.create_publisher(Pose, '/move_base_simple/goal', 10)
        self.task_status_pub = self.create_publisher(ActionResult, 'action_results', 10)
        self.system_status_pub = self.create_publisher(String, 'system_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            Command, 'parsed_commands', self.command_callback, 10)
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)

        # Internal state
        self.current_scene = None
        self.current_task = None
        self.action_state = ActionState.IDLE
        self.robot_pose = Pose()

        # Initialize action components
        self.navigation_executor = NavigationExecutor(self)
        self.manipulation_executor = ManipulationExecutor(self)
        self.information_executor = InformationExecutor(self)

        self.get_logger().info('Action node initialized')

    def command_callback(self, msg):
        """Process incoming commands"""
        self.get_logger().info(f'Received command: {msg.intent} with entities {msg.entities}')

        # Set current task
        self.current_task = {
            'intent': msg.intent,
            'entities': msg.entities,
            'original_text': msg.original_text,
            'confidence': msg.confidence
        }

        # Execute based on intent
        if msg.intent == 'navigation':
            self.execute_navigation(msg)
        elif msg.intent == 'manipulation':
            self.execute_manipulation(msg)
        elif msg.intent == 'information':
            self.execute_information(msg)
        elif msg.intent == 'social':
            self.execute_social(msg)
        else:
            self.publish_action_result('unknown_intent', False, 'Unknown intent')

    def scene_callback(self, msg):
        """Update scene context"""
        self.current_scene = msg

    def execute_navigation(self, command_msg):
        """Execute navigation commands"""
        self.action_state = ActionState.EXECUTING

        destination = command_msg.entities.get('destination', 'unknown')

        # In a real implementation, this would look up the location
        # For simulation, we'll use predefined locations
        location_poses = {
            'kitchen': (2.0, 1.0, 0.0),
            'living_room': (0.0, 2.0, 1.57),
            'bedroom': (-1.0, -1.0, 3.14),
            'office': (3.0, -2.0, -1.57),
            'dining_room': (1.5, -1.5, 0.78),
            'bathroom': (-2.0, 0.0, 2.35)
        }

        if destination in location_poses:
            x, y, theta = location_poses[destination]

            # Create navigation pose
            nav_pose = Pose()
            nav_pose.position.x = float(x)
            nav_pose.position.y = float(y)
            nav_pose.position.z = 0.0

            # Convert theta to quaternion
            from math import sin, cos
            cy = cos(theta * 0.5)
            sy = sin(theta * 0.5)
            nav_pose.orientation.z = sy
            nav_pose.orientation.w = cy

            # Publish navigation goal
            self.navigation_pub.publish(nav_pose)

            # Simulate execution time
            time.sleep(2)  # Simulate navigation time

            # Publish result
            self.publish_action_result(
                'navigation', True, f'Navigated to {destination}',
                execution_time=2.0
            )
        else:
            self.publish_action_result(
                'navigation', False, f'Unknown destination: {destination}'
            )

        self.action_state = ActionState.IDLE

    def execute_manipulation(self, command_msg):
        """Execute manipulation commands"""
        self.action_state = ActionState.EXECUTING

        target_object = command_msg.entities.get('object', 'unknown')

        # Check if object is in current scene
        object_available = False
        if self.current_scene:
            object_available = target_object in [obj.class for obj in self.current_scene.objects]

        if object_available:
            # Simulate manipulation execution
            time.sleep(1)  # Simulate grasping time

            self.publish_action_result(
                'manipulation', True, f'Fetched {target_object}',
                execution_time=1.0
            )
        else:
            self.publish_action_result(
                'manipulation', False, f'{target_object} not found in current scene'
            )

        self.action_state = ActionState.IDLE

    def execute_information(self, command_msg):
        """Execute information commands"""
        self.action_state = ActionState.EXECUTING

        # For simulation, return some contextual information
        info_text = "I can see several objects in the area."
        if self.current_scene:
            info_text = f"I see {len(self.current_scene.objects)} objects in the {self.current_scene.room_type}."

        self.publish_action_result(
            'information', True, info_text,
            execution_time=0.5
        )

        self.action_state = ActionState.IDLE

    def execute_social(self, command_msg):
        """Execute social commands"""
        self.action_state = ActionState.EXECUTING

        # For simulation, just acknowledge social commands
        social_responses = {
            'greet': 'Hello! How can I help you?',
            'farewell': 'Goodbye! It was nice talking with you.',
            'acknowledge_gratitude': 'You\'re welcome! I\'m happy to help.'
        }

        action = command_msg.action
        response = social_responses.get(action.replace('social_', ''), 'Hello!')

        self.publish_action_result(
            'social', True, response,
            execution_time=0.3
        )

        self.action_state = ActionState.IDLE

    def publish_action_result(self, action_type, success, message, execution_time=0.0):
        """Publish action result"""
        result_msg = ActionResult()
        result_msg.action_type = action_type
        result_msg.success = success
        result_msg.error_message = '' if success else message
        result_msg.execution_time = execution_time
        result_msg.details = message

        self.task_status_pub.publish(result_msg)

        # Log the result
        status = "SUCCESS" if success else "FAILED"
        self.get_logger().info(f'Action {action_type}: {status} - {message}')


class NavigationExecutor:
    def __init__(self, node):
        self.node = node

    def execute(self, destination):
        """Execute navigation to destination"""
        # Implementation would go here
        pass


class ManipulationExecutor:
    def __init__(self, node):
        self.node = node

    def execute(self, object_info):
        """Execute manipulation of object"""
        # Implementation would go here
        pass


class InformationExecutor:
    def __init__(self, node):
        self.node = node

    def execute(self, query):
        """Execute information query"""
        # Implementation would go here
        pass


def main(args=None):
    rclpy.init(args=args)
    action_node = ActionNode()

    try:
        rclpy.spin(action_node)
    except KeyboardInterrupt:
        pass
    finally:
        action_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 5.2 Create Action Launch File

Create `~/capstone_project/src/capstone_action/launch/action_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_action',
            executable='action_node',
            name='action_node',
            output='screen'
        )
    ])
```

### 5.3 Update Setup.py for Action Package

Update `~/capstone_project/src/capstone_action/setup.py`:

```python
from setuptools import setup

package_name = 'capstone_action'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/action_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='capstone@todo.todo',
    description='Capstone Project Action Execution Module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_node = capstone_action.action_node:main',
        ],
    },
)
```

## Step 6: Implement Fusion Module

### 6.1 Create Fusion Node

Create `~/capstone_project/src/capstone_fusion/capstone_fusion/fusion_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from capstone_perception.msg import SceneDescription, ObjectDetection
from capstone_language.msg import Command, Response
from capstone_action.msg import ActionResult
from capstone_safety.msg import SafetyStatus
from std_msgs.msg import String
import numpy as np
import time
from typing import Dict, Any, List
from collections import defaultdict, deque


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Publishers
        self.decision_pub = self.create_publisher(String, 'multimodal_decision', 10)
        self.status_pub = self.create_publisher(String, 'fusion_status', 10)
        self.system_command_pub = self.create_publisher(String, 'system_commands', 10)

        # Subscribers
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)
        self.command_sub = self.create_subscription(
            Command, 'parsed_commands', self.command_callback, 10)
        self.response_sub = self.create_subscription(
            Response, 'responses', self.response_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)
        self.safety_sub = self.create_subscription(
            SafetyStatus, 'safety_status', self.safety_callback, 10)

        # Internal state
        self.current_scene = None
        self.current_command = None
        self.recent_responses = deque(maxlen=10)
        self.action_history = deque(maxlen=20)
        self.context = {
            'objects': {},
            'people': {},
            'locations': {},
            'tasks': [],
            'intentions': []
        }

        # Initialize fusion components
        self.decision_maker = DecisionMaker()
        self.context_manager = ContextManager()
        self.confidence_fuser = ConfidenceFuser()

        # Timers for periodic tasks
        self.decision_timer = self.create_timer(0.5, self.make_decisions)  # 2Hz decision making
        self.context_update_timer = self.create_timer(1.0, self.update_context)  # 1Hz context updates

        self.get_logger().info('Fusion node initialized')

    def scene_callback(self, msg):
        """Update scene context"""
        self.current_scene = msg

        # Update context with scene information
        for obj in msg.objects:
            self.context['objects'][obj.class] = {
                'confidence': obj.confidence,
                'position': obj.position,
                'timestamp': time.time()
            }

        self.context['current_room'] = msg.room_type
        self.context['current_activity'] = msg.activity

    def command_callback(self, msg):
        """Process commands for context and decision making"""
        self.current_command = msg

        # Add to context
        self.context['intentions'].append({
            'intent': msg.intent,
            'entities': dict(msg.entities),
            'confidence': msg.confidence,
            'timestamp': time.time()
        })

        # Keep only recent intentions (last 5 minutes)
        self.context['intentions'] = [
            intent for intent in self.context['intentions']
            if time.time() - intent['timestamp'] < 300
        ]

    def response_callback(self, msg):
        """Process responses for context"""
        self.recent_responses.append({
            'text': msg.text,
            'intent': msg.intent,
            'confidence': msg.confidence,
            'timestamp': time.time()
        })

    def action_result_callback(self, msg):
        """Process action results for context and learning"""
        self.action_history.append({
            'action_type': msg.action_type,
            'success': msg.success,
            'execution_time': msg.execution_time,
            'timestamp': time.time()
        })

        # Update success metrics
        self.update_success_metrics(msg)

    def safety_callback(self, msg):
        """Process safety status"""
        # Safety status affects decision making
        self.context['safety_status'] = {
            'safe_to_proceed': msg.safe_to_proceed,
            'risk_level': msg.risk_level,
            'hazards': msg.hazards
        }

    def make_decisions(self):
        """Make decisions based on multimodal context"""
        if not self.current_command:
            return  # No command to process

        # Fuse information from all modalities
        fused_context = self.fuse_multimodal_context()

        # Make decision
        decision = self.decision_maker.make_decision(
            self.current_command,
            fused_context,
            self.current_scene
        )

        if decision:
            # Publish decision
            decision_msg = String()
            decision_msg.data = str(decision)
            self.decision_pub.publish(decision_msg)

            self.get_logger().info(f'Multimodal decision: {decision}')

    def fuse_multimodal_context(self) -> Dict[str, Any]:
        """Fuse information from all modalities"""
        fused_context = {
            'scene_confidence': self.get_scene_confidence(),
            'command_confidence': self.get_command_confidence(),
            'temporal_consistency': self.get_temporal_consistency(),
            'spatial_alignment': self.get_spatial_alignment(),
            'context_relevance': self.get_context_relevance()
        }

        # Calculate overall confidence
        confidences = [
            fused_context['scene_confidence'],
            fused_context['command_confidence']
        ]
        fused_context['overall_confidence'] = np.mean(confidences) if confidences else 0.5

        return fused_context

    def get_scene_confidence(self) -> float:
        """Calculate confidence in scene understanding"""
        if not self.current_scene:
            return 0.0

        # Confidence based on number of detected objects and their confidence
        if self.current_scene.objects:
            avg_confidence = np.mean([obj.confidence for obj in self.current_scene.objects])
            object_count_factor = min(1.0, len(self.current_scene.objects) / 5.0)  # Scale by object count
            return avg_confidence * object_count_factor

        return 0.3  # Low confidence if no objects detected

    def get_command_confidence(self) -> float:
        """Calculate confidence in command understanding"""
        if not self.current_command:
            return 0.0

        return min(1.0, self.current_command.confidence)

    def get_temporal_consistency(self) -> float:
        """Calculate temporal consistency of information"""
        # Check if recent information is consistent
        recent_actions = [action for action in self.action_history
                         if time.time() - action['timestamp'] < 10]  # Last 10 seconds

        if len(recent_actions) < 2:
            return 0.8  # Assume consistency if insufficient data

        # Calculate success rate of recent actions
        success_rate = sum(1 for action in recent_actions if action['success']) / len(recent_actions)
        return success_rate

    def get_spatial_alignment(self) -> float:
        """Calculate spatial alignment between modalities"""
        # In a real implementation, this would check alignment between
        # visual objects and audio sources, etc.
        return 0.9  # Assume good alignment for now

    def get_context_relevance(self) -> float:
        """Calculate relevance of context to current command"""
        if not self.current_command or not self.current_scene:
            return 0.5

        # Check if command entities match scene objects
        command_entities = self.current_command.entities
        scene_objects = [obj.class for obj in self.current_scene.objects] if self.current_scene.objects else []

        if not command_entities or not scene_objects:
            return 0.6  # Moderate relevance if no entities or objects

        # Calculate overlap between command entities and scene objects
        entity_overlap = sum(1 for entity in command_entities.values()
                           if entity in scene_objects)
        relevance_score = entity_overlap / len(command_entities) if command_entities else 0.0

        return min(1.0, relevance_score + 0.3)  # Add base relevance

    def update_context(self):
        """Update context and publish status"""
        # Clean up old context elements
        current_time = time.time()

        # Update status
        status_msg = String()
        status_msg.data = f"Fusion active - Scene: {getattr(self.current_scene, 'room_type', 'unknown')}, " \
                         f"Commands processed: {len(self.context['intentions'])}, " \
                         f"Actions completed: {len(self.action_history)}"
        self.status_pub.publish(status_msg)

    def update_success_metrics(self, action_result):
        """Update success metrics based on action results"""
        # This could be used for learning and adaptation
        pass


class DecisionMaker:
    def __init__(self):
        self.confidence_threshold = 0.6
        self.priorities = {
            'safety': 10.0,
            'navigation': 5.0,
            'manipulation': 4.0,
            'information': 3.0,
            'social': 2.0
        }

    def make_decision(self, command, fused_context, scene):
        """Make decision based on fused context"""
        if fused_context['overall_confidence'] < self.confidence_threshold:
            return {
                'action': 'request_clarification',
                'reason': 'Low confidence in understanding',
                'confidence': fused_context['overall_confidence']
            }

        # Check safety first
        if scene and hasattr(scene, 'room_type') and scene.room_type == 'unknown':
            return {
                'action': 'wait_for_localization',
                'reason': 'Need to determine location',
                'confidence': 0.8
            }

        # Process command based on intent
        decision = {
            'action': f"execute_{command.intent}",
            'entities': dict(command.entities),
            'original_command': command.original_text,
            'confidence': fused_context['overall_confidence'],
            'timestamp': time.time()
        }

        return decision


class ContextManager:
    def __init__(self):
        self.context_history = deque(maxlen=100)

    def update_context(self, new_context):
        """Update context with new information"""
        self.context_history.append({
            'context': new_context,
            'timestamp': time.time()
        })

    def get_context_relevance(self, query_context):
        """Get relevance of stored context to query"""
        # Implementation would go here
        return 0.8


class ConfidenceFuser:
    def __init__(self):
        self.weight_factors = {
            'scene': 0.4,
            'command': 0.3,
            'temporal': 0.2,
            'spatial': 0.1
        }

    def fuse_confidence(self, confidences):
        """Fuse confidence scores from different sources"""
        weighted_confidence = 0.0
        total_weight = 0.0

        for source, confidence in confidences.items():
            weight = self.weight_factors.get(source, 0.0)
            weighted_confidence += confidence * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return 0.5  # Default confidence


def main(args=None):
    rclpy.init(args=args)
    fusion_node = FusionNode()

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

### 6.2 Create Fusion Launch File

Create `~/capstone_project/src/capstone_fusion/launch/fusion_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_fusion',
            executable='fusion_node',
            name='fusion_node',
            output='screen'
        )
    ])
```

### 6.3 Update Setup.py for Fusion Package

Update `~/capstone_project/src/capstone_fusion/setup.py`:

```python
from setuptools import setup

package_name = 'capstone_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/fusion_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='capstone@todo.todo',
    description='Capstone Project Multimodal Fusion Module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_node = capstone_fusion.fusion_node:main',
        ],
    },
)
```

## Step 7: Implement Safety Module

### 7.1 Create Safety Node

Create `~/capstone_project/src/capstone_safety/capstone_safety/safety_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Pose, Twist
from capstone_safety.msg import SafetyStatus
from std_msgs.msg import Bool
import numpy as np
import time
from typing import Dict, Any, List


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')

        # Publishers
        self.safety_status_pub = self.create_publisher(SafetyStatus, 'safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)
        self.velocity_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/current_pose', self.pose_callback, 10)

        # Internal state
        self.current_pose = Pose()
        self.current_velocity = None
        self.obstacle_distances = []
        self.human_proximity = False
        self.risk_level = 0.0

        # Safety parameters
        self.safety_buffer = 0.5  # meters
        self.emergency_stop_distance = 0.3  # meters
        self.human_detection_threshold = 1.0  # meters
        self.max_velocity = 0.5  # m/s

        # Timers
        self.safety_check_timer = self.create_timer(0.1, self.safety_check)  # 10Hz safety checks

        self.get_logger().info('Safety node initialized')

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Get distances within 90 degrees in front of robot
        front_distances = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Sample points in front of robot (±45 degrees)
        front_start_idx = int((np.pi/4 - angle_min) / angle_increment)  # 45 degrees
        front_end_idx = int((3*np.pi/4 - angle_min) / angle_increment)  # 135 degrees

        if 0 <= front_start_idx < len(msg.ranges) and 0 <= front_end_idx < len(msg.ranges):
            front_distances = [d for d in msg.ranges[front_start_idx:front_end_idx]
                             if not np.isnan(d) and d > 0]

        self.obstacle_distances = front_distances

    def pointcloud_callback(self, msg):
        """Process point cloud data for human detection"""
        # In a real implementation, this would use more sophisticated
        # point cloud processing to detect humans
        # For simulation, we'll use a simplified approach

        # Convert point cloud to numpy array (simplified)
        # In practice, use PCL or similar libraries
        self.human_proximity = self.detect_humans_in_pointcloud(msg)

    def velocity_callback(self, msg):
        """Monitor current velocity for safety"""
        self.current_velocity = msg

    def pose_callback(self, msg):
        """Update current pose"""
        self.current_pose = msg

    def detect_humans_in_pointcloud(self, pointcloud_msg) -> bool:
        """Detect humans in point cloud (simplified simulation)"""
        # For simulation, randomly detect humans occasionally
        return np.random.random() < 0.1  # 10% chance of detecting human

    def safety_check(self):
        """Perform safety check and publish status"""
        # Calculate risk based on obstacles
        min_distance = min(self.obstacle_distances) if self.obstacle_distances else float('inf')

        # Calculate risk level (0.0 to 1.0)
        if min_distance <= self.emergency_stop_distance:
            risk_level = 1.0  # Emergency
        elif min_distance <= self.safety_buffer:
            risk_level = 0.8  # High risk
        elif min_distance <= self.safety_buffer * 2:
            risk_level = 0.5  # Medium risk
        else:
            risk_level = 0.1  # Low risk

        # Increase risk if humans detected nearby
        if self.human_proximity:
            risk_level = max(risk_level, 0.7)

        # Check velocity safety
        if self.current_velocity:
            speed = np.sqrt(
                self.current_velocity.linear.x**2 +
                self.current_velocity.linear.y**2 +
                self.current_velocity.linear.z**2
            )

            if speed > self.max_velocity:
                risk_level = max(risk_level, 0.6)

        # Determine if safe to proceed
        safe_to_proceed = risk_level < 0.8

        # Publish safety status
        safety_msg = SafetyStatus()
        safety_msg.safe_to_proceed = safe_to_proceed
        safety_msg.risk_level = risk_level
        safety_msg.min_obstacle_distance = min_distance if self.obstacle_distances else float('inf')
        safety_msg.human_detected = self.human_proximity
        safety_msg.timestamp = time.time()

        # Add detected hazards
        hazards = []
        if min_distance <= self.emergency_stop_distance:
            hazards.append('obstacle_too_close')
        if self.human_proximity:
            hazards.append('human_in_proximity')
        if self.current_velocity and speed > self.max_velocity:
            hazards.append('excessive_speed')

        safety_msg.hazards = hazards

        self.safety_status_pub.publish(safety_msg)

        # Trigger emergency stop if necessary
        if not safe_to_proceed:
            self.trigger_emergency_stop()

        # Update internal risk level
        self.risk_level = risk_level

        self.get_logger().info(f'Safety check - Risk: {risk_level:.2f}, Safe: {safe_to_proceed}')

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
        self.get_logger().warn('EMERGENCY STOP TRIGGERED!')

    def is_safe_to_move(self) -> bool:
        """Check if it's safe to move"""
        return self.risk_level < 0.8


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        safety_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 7.2 Create Safety Message Definition

Create `~/capstone_project/src/capstone_safety/msg/SafetyStatus.msg`:

```
# SafetyStatus.msg
bool safe_to_proceed
float64 risk_level
float64 min_obstacle_distance
bool human_detected
string[] hazards
float64 timestamp
```

### 7.3 Create Safety Launch File

Create `~/capstone_project/src/capstone_safety/launch/safety_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_safety',
            executable='safety_node',
            name='safety_node',
            output='screen'
        )
    ])
```

### 7.4 Update Setup.py for Safety Package

Update `~/capstone_project/src/capstone_safety/setup.py`:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'capstone_safety'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='capstone@todo.todo',
    description='Capstone Project Safety Monitoring Module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_node = capstone_safety.safety_node:main',
        ],
    },
)
```

## Step 8: Create Main Integration Node

### 8.1 Create Main Node

Create `~/capstone_project/src/capstone_main/capstone_main/main_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_perception.msg import SceneDescription
from capstone_language.msg import Response
from capstone_action.msg import ActionResult
from capstone_fusion.msg import Decision
from capstone_safety.msg import SafetyStatus
import time
from typing import Dict, Any


class MainNode(Node):
    def __init__(self):
        super().__init__('capstone_main')

        # Publishers
        self.system_status_pub = self.create_publisher(String, 'system_status', 10)
        self.user_feedback_pub = self.create_publisher(String, 'user_feedback', 10)

        # Subscribers
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)
        self.response_sub = self.create_subscription(
            Response, 'responses', self.response_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)
        self.decision_sub = self.create_subscription(
            String, 'multimodal_decision', self.decision_callback, 10)
        self.safety_sub = self.create_subscription(
            SafetyStatus, 'safety_status', self.safety_callback, 10)

        # System state
        self.system_state = {
            'initialized': True,
            'active_modules': [],
            'performance_metrics': {},
            'error_count': 0,
            'last_update': time.time()
        }

        # Initialize system
        self.initialize_system()

        # Timers
        self.status_timer = self.create_timer(1.0, self.publish_system_status)

        self.get_logger().info('Capstone Main Node Initialized')

    def initialize_system(self):
        """Initialize the integrated system"""
        self.get_logger().info('Initializing capstone system...')

        # Check if all required modules are available
        required_modules = [
            'perception_node',
            'language_node',
            'action_node',
            'fusion_node',
            'safety_node'
        ]

        self.system_state['active_modules'] = required_modules
        self.system_state['initialized'] = True

        self.get_logger().info(f'System initialized with {len(required_modules)} modules')

    def scene_callback(self, msg):
        """Handle scene updates"""
        self.get_logger().info(f'Scene update: {msg.room_type} with {len(msg.objects)} objects')

    def response_callback(self, msg):
        """Handle system responses"""
        self.get_logger().info(f'System response: {msg.text}')

    def action_result_callback(self, msg):
        """Handle action results"""
        status = "SUCCESS" if msg.success else "FAILED"
        self.get_logger().info(f'Action result: {msg.action_type} - {status}')

    def decision_callback(self, msg):
        """Handle multimodal decisions"""
        self.get_logger().info(f'Multimodal decision: {msg.data}')

    def safety_callback(self, msg):
        """Handle safety status updates"""
        status = "SAFE" if msg.safe_to_proceed else "UNSAFE"
        self.get_logger().info(f'Safety status: {status}, Risk: {msg.risk_level:.2f}')

    def publish_system_status(self):
        """Publish overall system status"""
        status_msg = String()
        status_msg.data = f"System Status - Modules: {len(self.system_state['active_modules'])}, " \
                         f"Errors: {self.system_state['error_count']}, " \
                         f"Last Update: {time.time() - self.system_state['last_update']:.1f}s ago"

        self.system_status_pub.publish(status_msg)
        self.system_state['last_update'] = time.time()

    def shutdown_system(self):
        """Graceful system shutdown"""
        self.get_logger().info('Shutting down capstone system...')
        # Add cleanup code here if needed


def main(args=None):
    rclpy.init(args=args)
    main_node = MainNode()

    try:
        rclpy.spin(main_node)
    except KeyboardInterrupt:
        main_node.get_logger().info('Interrupted, shutting down...')
    finally:
        main_node.shutdown_system()
        main_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 8.2 Create Main Launch File

Create `~/capstone_project/src/capstone_main/launch/capstone_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    return LaunchDescription([
        # Launch all capstone modules
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_perception'),
                    'launch',
                    'perception_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_language'),
                    'launch',
                    'language_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_action'),
                    'launch',
                    'action_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_fusion'),
                    'launch',
                    'fusion_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_safety'),
                    'launch',
                    'safety_launch.py'
                ])
            )
        ),
        # Launch main integration node
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_main'),
                    'launch',
                    'main_launch.py'
                ])
            )
        ),
    ])
```

### 8.3 Create Main Launch File

Create `~/capstone_project/src/capstone_main/launch/main_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_main',
            executable='main_node',
            name='capstone_main',
            output='screen'
        )
    ])
```

### 8.4 Update Setup.py for Main Package

Update `~/capstone_project/src/capstone_main/setup.py`:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'capstone_main'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='capstone@todo.todo',
    description='Capstone Project Main Integration Module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_node = capstone_main.main_node:main',
        ],
    },
)
```

## Step 9: Build and Test the System

### 9.1 Build the Workspace

```bash
cd ~/capstone_project
colcon build --packages-select capstone_perception capstone_language capstone_action capstone_fusion capstone_safety capstone_main
source install/setup.bash
```

### 9.2 Create a Test Script

Create `~/capstone_project/test_capstone.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import time


class CapstoneTester(Node):
    def __init__(self):
        super().__init__('capstone_tester')

        # Publishers to simulate inputs
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)
        self.pose_pub = self.create_publisher(Pose, '/current_pose', 10)

        # Start testing after a delay to allow system to initialize
        self.timer = self.create_timer(3.0, self.run_tests)

        self.test_counter = 0
        self.get_logger().info('Capstone tester initialized')

    def run_tests(self):
        """Run a series of tests"""
        tests = [
            self.test_navigation_command,
            self.test_manipulation_command,
            self.test_information_request,
        ]

        if self.test_counter < len(tests):
            test_func = tests[self.test_counter]
            test_func()
            self.test_counter += 1

            # Schedule next test
            self.timer = self.create_timer(5.0, self.run_tests)
        else:
            self.get_logger().info('All tests completed')

    def test_navigation_command(self):
        """Test navigation command"""
        self.get_logger().info('Testing navigation command: "Go to kitchen"')

        speech_msg = String()
        speech_msg.data = 'Go to kitchen'
        self.speech_pub.publish(speech_msg)

    def test_manipulation_command(self):
        """Test manipulation command"""
        self.get_logger().info('Testing manipulation command: "Get me the cup"')

        speech_msg = String()
        speech_msg.data = 'Get me the cup'
        self.speech_pub.publish(speech_msg)

    def test_information_request(self):
        """Test information request"""
        self.get_logger().info('Testing information request: "What do you see?"')

        speech_msg = String()
        speech_msg.data = 'What do you see?'
        self.speech_pub.publish(speech_msg)


def main(args=None):
    rclpy.init(args=args)
    tester = CapstoneTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 9.3 Update Main Package Setup.py for Test Script

Add to `~/capstone_project/src/capstone_main/setup.py`:

```python
entry_points={
    'console_scripts': [
        'main_node = capstone_main.main_node:main',
        'capstone_tester = capstone_main.tester:main',  # Add this line
    ],
},
```

## Step 10: System Integration and Validation

### 10.1 Create Integration Test Node

Create `~/capstone_project/src/capstone_main/capstone_main/integration_test.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_action.msg import ActionResult
from capstone_language.msg import Response
from capstone_perception.msg import SceneDescription
import time
from typing import Dict, List


class IntegrationTestNode(Node):
    def __init__(self):
        super().__init__('integration_test')

        # Subscribers to monitor system responses
        self.response_sub = self.create_subscription(
            Response, 'responses', self.response_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)

        # Publisher for test commands
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)

        # Test state
        self.test_results = []
        self.current_test = None
        self.responses_received = []
        self.action_results_received = []

        # Start test sequence
        self.timer = self.create_timer(2.0, self.run_next_test)
        self.test_index = 0

        self.get_logger().info('Integration test node started')

    def response_callback(self, msg):
        """Record system responses"""
        self.responses_received.append({
            'text': msg.text,
            'intent': msg.intent,
            'timestamp': time.time()
        })

    def action_result_callback(self, msg):
        """Record action results"""
        self.action_results_received.append({
            'action_type': msg.action_type,
            'success': msg.success,
            'timestamp': time.time()
        })

    def scene_callback(self, msg):
        """Record scene updates"""
        self.get_logger().info(f'Scene updated: {msg.room_type} with {len(msg.objects)} objects')

    def run_next_test(self):
        """Run the next test in sequence"""
        test_commands = [
            ('Navigation Test', 'Go to kitchen'),
            ('Manipulation Test', 'Get me the cup'),
            ('Information Test', 'What do you see?'),
            ('Social Test', 'Hello how are you?')
        ]

        if self.test_index < len(test_commands):
            test_name, command = test_commands[self.test_index]

            self.get_logger().info(f'Running {test_name}: {command}')

            # Clear previous results
            self.responses_received.clear()
            self.action_results_received.clear()

            # Send command
            speech_msg = String()
            speech_msg.data = command
            self.speech_pub.publish(speech_msg)

            # Schedule result checking
            self.current_test = test_name
            self.create_timer(3.0, self.check_test_results)

            self.test_index += 1
        else:
            # All tests completed
            self.run_final_validation()

    def check_test_results(self):
        """Check results for current test"""
        if self.current_test:
            result = {
                'test': self.current_test,
                'responses': len(self.responses_received),
                'actions': len(self.action_results_received),
                'success': len(self.responses_received) > 0 or len(self.action_results_received) > 0
            }

            self.test_results.append(result)

            status = "PASSED" if result['success'] else "FAILED"
            self.get_logger().info(f'Test {self.current_test}: {status} '
                                 f'({result["responses"]} responses, {result["actions"]} actions)')

    def run_final_validation(self):
        """Run final validation and report"""
        self.get_logger().info('\n' + '='*50)
        self.get_logger().info('INTEGRATION TEST RESULTS')
        self.get_logger().info('='*50)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        for result in self.test_results:
            status = "PASS" if result['success'] else "FAIL"
            self.get_logger().info(f'{result["test"]}: {status}')

        self.get_logger().info('-' * 30)
        self.get_logger().info(f'Total Tests: {total_tests}')
        self.get_logger().info(f'Passed: {passed_tests}')
        self.get_logger().info(f'Success Rate: {success_rate:.1%}')

        if success_rate >= 0.75:
            self.get_logger().info('🎉 INTEGRATION TESTS PASSED!')
        else:
            self.get_logger().info('❌ INTEGRATION TESTS FAILED - Check system configuration')

        # Stop the node after tests
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    test_node = IntegrationTestNode()

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 10.2 Add Integration Test to Setup.py

Add to the entry_points in `~/capstone_project/src/capstone_main/setup.py`:

```python
entry_points={
    'console_scripts': [
        'main_node = capstone_main.main_node:main',
        'integration_test = capstone_main.integration_test:main',
    ],
},
```

## Step 11: Final System Launch and Documentation

### 11.1 Create Comprehensive Launch File

Create `~/capstone_project/launch_all.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # Launch all capstone modules
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_perception'),
                    'launch',
                    'perception_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_language'),
                    'launch',
                    'language_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_action'),
                    'launch',
                    'action_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_fusion'),
                    'launch',
                    'fusion_launch.py'
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_safety'),
                    'launch',
                    'safety_launch.py'
                ])
            )
        ),
        # Launch main integration node
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('capstone_main'),
                    'launch',
                    'main_launch.py'
                ])
            )
        ),
    ])
```

This completes the implementation of the Capstone Project Integration System. The system now includes all necessary modules for voice-controlled navigation, manipulation, and interaction with proper safety monitoring and multimodal fusion.

The implementation follows the step-by-step guide and creates a fully integrated humanoid robotics system that demonstrates all the concepts learned throughout the book.