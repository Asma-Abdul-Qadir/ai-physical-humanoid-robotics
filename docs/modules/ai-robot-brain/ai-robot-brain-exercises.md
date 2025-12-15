---
sidebar_position: 6
---

# AI Robot Brain Exercises: Working Examples

Welcome to the AI Robot Brain Exercises module, which provides hands-on exercises that integrate perception, language, and action capabilities for humanoid robots. These exercises are designed to reinforce the concepts learned in previous chapters and provide practical experience with implementing multimodal AI systems.

## Learning Objectives

By completing these exercises, you will be able to:
- Implement complete perception-action loops for humanoid robots
- Integrate computer vision, audio processing, and language understanding
- Create multimodal interaction scenarios
- Debug and troubleshoot AI robot brain systems
- Evaluate system performance and robustness
- Apply best practices for multimodal AI development
- Build end-to-end AI robot applications

## Exercise 1: Object Detection and Navigation

### Scenario: Find and Navigate to a Specific Object

In this exercise, you'll implement a system that detects a specific object in the environment and navigates to it.

#### Exercise Setup

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import MarkerArray
import cv2
import numpy as np
from cv_bridge import CvBridge
import time


class ObjectDetectionNavigationExercise(Node):
    def __init__(self):
        super().__init__('object_detection_navigation_exercise')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/exercise_command', self.command_callback, 10)

        # Publishers
        self.navigation_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.detection_pub = self.create_publisher(
            MarkerArray, '/detected_objects', 10)
        self.status_pub = self.create_publisher(
            String, '/exercise_status', 10)

        # Exercise state
        self.current_task = None
        self.detected_objects = {}
        self.robot_pose = Pose()
        self.target_object = None
        self.is_active = False

        # Object detection parameters
        self.object_classes = ['person', 'chair', 'table', 'cup', 'bottle']
        self.detection_threshold = 0.5

        self.get_logger().info('Object detection navigation exercise initialized')

    def command_callback(self, msg):
        """Handle exercise commands"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received command: {command}')

        if command == 'start_exercise':
            self.start_exercise()
        elif command.startswith('find_object:'):
            obj_type = command.split(':')[1]
            self.find_object(obj_type)
        elif command == 'stop_exercise':
            self.stop_exercise()

    def start_exercise(self):
        """Start the object detection and navigation exercise"""
        self.is_active = True
        self.current_task = 'detection'
        self.get_logger().info('Exercise started - detecting objects')

        status_msg = String()
        status_msg.data = 'Exercise started - detecting objects'
        self.status_pub.publish(status_msg)

    def find_object(self, obj_type):
        """Find a specific object type"""
        self.target_object = obj_type.lower()
        self.current_task = 'navigation'
        self.get_logger().info(f'Searching for {obj_type}')

        status_msg = String()
        status_msg.data = f'Searching for {obj_type}'
        self.status_pub.publish(status_msg)

    def stop_exercise(self):
        """Stop the exercise"""
        self.is_active = False
        self.current_task = None
        self.get_logger().info('Exercise stopped')

        status_msg = String()
        status_msg.data = 'Exercise stopped'
        self.status_pub.publish(status_msg)

    def image_callback(self, msg):
        """Process incoming images for object detection"""
        if not self.is_active:
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection (simulated - in real implementation, use actual detection)
            detections = self.simulate_object_detection(cv_image)

            # Update detected objects
            self.detected_objects = detections

            # Publish detections as markers
            self.publish_detections(detections, msg.header)

            # If we're looking for a specific object, navigate to it
            if self.current_task == 'navigation' and self.target_object:
                self.navigate_to_target_object()

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def simulate_object_detection(self, image):
        """Simulate object detection - in real implementation, use actual detection model"""
        height, width = image.shape[:2]
        detections = {}

        # Simulate some detections for exercise purposes
        for i, obj_class in enumerate(self.object_classes):
            # Random position and size
            x = np.random.randint(0, width // 2)
            y = np.random.randint(0, height // 2)
            w = np.random.randint(width // 8, width // 4)
            h = np.random.randint(height // 8, height // 4)

            # Random confidence above threshold
            confidence = np.random.uniform(self.detection_threshold, 0.9)

            detections[f'obj_{i}'] = {
                'class': obj_class,
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'confidence': confidence
            }

        return detections

    def publish_detections(self, detections, header):
        """Publish detections as visualization markers"""
        marker_array = MarkerArray()

        for i, (obj_id, detection) in enumerate(detections.items()):
            marker = self.create_detection_marker(detection, header, i)
            marker_array.markers.append(marker)

        self.detection_pub.publish(marker_array)

    def create_detection_marker(self, detection, header, marker_id):
        """Create a visualization marker for a detection"""
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header = header
        marker.ns = "objects"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position at center of bounding box
        marker.pose.position.x = detection['center'][0] / 100.0  # Scale down for visualization
        marker.pose.position.y = detection['center'][1] / 100.0
        marker.pose.position.z = 0.0

        marker.pose.orientation.w = 1.0

        # Scale based on bounding box size
        marker.scale.x = detection['bbox'][2] / 100.0
        marker.scale.y = detection['bbox'][3] / 100.0
        marker.scale.z = 0.5

        # Color based on class
        class_colors = {
            'person': (0, 1, 0),      # Green
            'chair': (1, 0, 0),       # Red
            'table': (0, 0, 1),       # Blue
            'cup': (1, 1, 0),         # Yellow
            'bottle': (1, 0, 1)       # Magenta
        }

        color = class_colors.get(detection['class'], (0.5, 0.5, 0.5))
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.7

        marker.text = f"{detection['class']}: {detection['confidence']:.2f}"

        return marker

    def navigate_to_target_object(self):
        """Navigate to the target object if found"""
        for obj_id, detection in self.detected_objects.items():
            if detection['class'].lower() == self.target_object.lower():
                # Calculate navigation pose
                nav_pose = self.calculate_navigation_pose(detection)
                self.navigation_pub.publish(nav_pose)

                self.get_logger().info(f'Navigating to {self.target_object} at {detection["center"]}')

                # Update status
                status_msg = String()
                status_msg.data = f'Navigating to {self.target_object}'
                self.status_pub.publish(status_msg)

                return  # Only navigate to first instance found

        # If target object not found, continue searching
        self.get_logger().info(f'{self.target_object} not found yet')

    def calculate_navigation_pose(self, detection):
        """Calculate navigation pose to approach the detected object"""
        from geometry_msgs.msg import Pose

        pose = Pose()

        # Convert image coordinates to world coordinates (simplified)
        # In real implementation, use proper camera calibration and transformation
        pose.position.x = detection['center'][0] / 100.0 - 1.0  # 1m away from object
        pose.position.y = detection['center'][1] / 100.0
        pose.position.z = 0.0

        pose.orientation.w = 1.0

        return pose


def main(args=None):
    rclpy.init(args=args)
    exercise = ObjectDetectionNavigationExercise()

    try:
        rclpy.spin(exercise)
    except KeyboardInterrupt:
        pass
    finally:
        exercise.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Exercise Instructions

1. **Setup**: Ensure your robot has a camera and navigation capabilities
2. **Start Detection**: Use the command `start_exercise` to begin object detection
3. **Target Object**: Use `find_object:cup` (or any other object type) to navigate to a specific object
4. **Monitor**: Watch the visualization markers to see detected objects
5. **Navigation**: The robot should navigate to the target object when found

#### Exercise Solution Analysis

```python
# Solution walkthrough for Exercise 1

# Key components:
# 1. Object detection pipeline
# 2. Navigation planning
# 3. State management
# 4. Visualization

# The exercise demonstrates:
# - Integration of perception (vision) and action (navigation)
# - State-based task execution
# - Real-time processing of sensor data
# - Visualization of results
```

## Exercise 2: Voice Command and Object Manipulation

### Scenario: Voice-Activated Object Fetching

In this exercise, you'll implement a system that accepts voice commands to fetch specific objects.

#### Exercise Setup

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time


class VoiceCommandManipulationExercise(Node):
    def __init__(self):
        super().__init__('voice_command_manipulation_exercise')

        # Subscriptions
        self.voice_sub = self.create_subscription(
            String, '/voice_input', self.voice_callback, 10)
        self.vision_sub = self.create_subscription(
            String, '/object_detections', self.vision_callback, 10)

        # Publishers
        self.navigation_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.manipulation_pub = self.create_publisher(
            String, '/manipulation_command', 10)
        self.response_pub = self.create_publisher(
            String, '/voice_response', 10)
        self.status_pub = self.create_publisher(
            String, '/exercise_status', 10)

        # Exercise state
        self.current_task = None
        self.target_object = None
        self.is_listening = False
        self.detected_objects = {}
        self.object_locations = {}

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_queue = queue.Queue()

        # Voice command patterns
        self.command_patterns = {
            'fetch': [
                'get the',
                'bring me the',
                'fetch the',
                'pick up the',
                'grab the'
            ],
            'go_to': [
                'go to the',
                'move to the',
                'navigate to the'
            ]
        }

        self.get_logger().info('Voice command manipulation exercise initialized')

    def voice_callback(self, msg):
        """Process voice commands"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Voice command received: {command}')

        # Parse the command
        parsed_command = self.parse_voice_command(command)

        if parsed_command:
            self.execute_command(parsed_command, command)
        else:
            response = "I didn't understand that command. Please try again."
            self.speak_response(response)

    def parse_voice_command(self, command):
        """Parse voice command to extract action and object"""
        command_lower = command.lower()

        # Check for fetch commands
        for pattern in self.command_patterns['fetch']:
            if pattern in command_lower:
                # Extract object name
                object_name = command_lower.split(pattern)[-1].strip()
                return {
                    'action': 'fetch',
                    'object': object_name,
                    'original_command': command
                }

        # Check for navigation commands
        for pattern in self.command_patterns['go_to']:
            if pattern in command_lower:
                # Extract destination
                destination = command_lower.split(pattern)[-1].strip()
                return {
                    'action': 'navigate',
                    'destination': destination,
                    'original_command': command
                }

        return None

    def execute_command(self, parsed_command, original_command):
        """Execute the parsed command"""
        action = parsed_command['action']

        if action == 'fetch':
            self.fetch_object(parsed_command['object'])
        elif action == 'navigate':
            self.navigate_to_location(parsed_command['destination'])
        else:
            response = f"I don't know how to {action} yet."
            self.speak_response(response)

    def fetch_object(self, obj_name):
        """Fetch the specified object"""
        self.target_object = obj_name
        self.current_task = 'fetching'

        # Check if object is detected
        if obj_name in self.object_locations:
            # Navigate to object location
            obj_pose = self.object_locations[obj_name]
            self.navigation_pub.publish(obj_pose)

            response = f"I'm going to get the {obj_name}."
            self.speak_response(response)

            # Update status
            status_msg = String()
            status_msg.data = f'Fetching {obj_name}'
            self.status_pub.publish(status_msg)
        else:
            response = f"I don't see the {obj_name} right now. I'll look for it."
            self.speak_response(response)

    def navigate_to_location(self, location):
        """Navigate to a specific location"""
        # In a real implementation, this would use a map
        # For simulation, use predefined locations
        location_poses = {
            'kitchen': (2.0, 1.0, 0.0),
            'living room': (0.0, 2.0, 1.57),
            'bedroom': (-1.0, -1.0, 3.14),
            'office': (3.0, -2.0, -1.57)
        }

        if location in location_poses:
            x, y, theta = location_poses[location]

            # Create navigation pose
            nav_pose = Pose()
            nav_pose.position.x = x
            nav_pose.position.y = y
            nav_pose.position.z = 0.0

            # Convert theta to quaternion
            from math import sin, cos
            cy = cos(theta * 0.5)
            sy = sin(theta * 0.5)
            nav_pose.orientation.z = sy
            nav_pose.orientation.w = cy

            self.navigation_pub.publish(nav_pose)

            response = f"I'm going to the {location}."
            self.speak_response(response)

            # Update status
            status_msg = String()
            status_msg.data = f'Navigating to {location}'
            self.status_pub.publish(status_msg)
        else:
            response = f"I don't know where the {location} is."
            self.speak_response(response)

    def vision_callback(self, msg):
        """Process vision data to update object locations"""
        # In a real implementation, this would parse object detection results
        # For simulation, we'll create mock object locations
        self.update_mock_object_locations()

    def update_mock_object_locations(self):
        """Update mock object locations for simulation"""
        # Simulate object locations
        mock_objects = {
            'cup': Pose(),
            'bottle': Pose(),
            'book': Pose()
        }

        # Set mock positions
        for i, (obj_name, pose) in enumerate(mock_objects.items()):
            pose.position.x = np.random.uniform(-2.0, 2.0)
            pose.position.y = np.random.uniform(-2.0, 2.0)
            pose.position.z = 0.0
            pose.orientation.w = 1.0

            self.object_locations[obj_name] = pose

    def speak_response(self, text):
        """Speak a response using text-to-speech"""
        # Add to queue for non-blocking speech
        self.tts_queue.put(text)

        # Publish response
        response_msg = String()
        response_msg.data = text
        self.response_pub.publish(response_msg)

        # Start speech thread if not running
        if not hasattr(self, 'speech_thread') or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self.speech_worker)
            self.speech_thread.daemon = True
            self.speech_thread.start()

    def speech_worker(self):
        """Worker thread for text-to-speech"""
        while not self.tts_queue.empty():
            try:
                text = self.tts_queue.get_nowait()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except queue.Empty:
                break

    def get_logger(self):
        """Get logger instance"""
        return self.get_logger()


def main(args=None):
    rclpy.init(args=args)
    exercise = VoiceCommandManipulationExercise()

    try:
        rclpy.spin(exercise)
    except KeyboardInterrupt:
        pass
    finally:
        exercise.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Exercise Instructions

1. **Setup**: Ensure your robot has voice recognition and manipulation capabilities
2. **Voice Commands**: Try commands like "get the cup" or "go to the kitchen"
3. **Object Detection**: The system should detect objects and store their locations
4. **Action Execution**: The robot should execute the requested action
5. **Response**: The robot should provide verbal feedback

#### Exercise Solution Analysis

```python
# Solution walkthrough for Exercise 2

# Key components:
# 1. Voice command parsing
# 2. Object location tracking
# 3. Action execution
# 4. Text-to-speech feedback

# The exercise demonstrates:
# - Integration of voice, vision, and action
# - Natural language processing
# - State management for complex tasks
# - Multimodal interaction design
```

## Exercise 3: Social Interaction with Context Awareness

### Scenario: Context-Aware Social Robot

In this exercise, you'll implement a social robot that adapts its behavior based on context and social cues.

#### Exercise Setup

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray
import numpy as np
import time
from datetime import datetime
from collections import defaultdict, deque


class SocialInteractionExercise(Node):
    def __init__(self):
        super().__init__('social_interaction_exercise')

        # Subscriptions
        self.speech_sub = self.create_subscription(
            String, '/speech_input', self.speech_callback, 10)
        self.vision_sub = self.create_subscription(
            MarkerArray, '/detected_people', self.vision_callback, 10)
        self.time_sub = self.create_subscription(
            String, '/current_time', self.time_callback, 10)

        # Publishers
        self.response_pub = self.create_publisher(
            String, '/social_response', 10)
        self.gesture_pub = self.create_publisher(
            String, '/gesture_command', 10)
        self.attention_pub = self.create_publisher(
            Pose, '/attention_target', 10)
        self.status_pub = self.create_publisher(
            String, '/exercise_status', 10)

        # System state
        self.detected_people = {}
        self.conversation_history = deque(maxlen=20)
        self.person_profiles = defaultdict(dict)
        self.current_time = datetime.now()
        self.social_context = {
            'time_of_day': self.get_time_of_day(),
            'people_count': 0,
            'conversation_active': False,
            'last_interaction_time': 0
        }

        # Response templates
        self.greeting_responses = [
            "Hello! How can I help you today?",
            "Hi there! It's nice to meet you.",
            "Good day! How are you doing?",
            "Hello! I'm here to assist you."
        ]

        self.farewell_responses = [
            "Goodbye! It was nice talking with you.",
            "See you later!",
            "Take care!",
            "It was great chatting with you!"
        ]

        self.get_logger().info('Social interaction exercise initialized')

    def speech_callback(self, msg):
        """Process speech input for social interaction"""
        speech = msg.data.lower().strip()
        self.get_logger().info(f'Received speech: {speech}')

        # Add to conversation history
        self.conversation_history.append({
            'type': 'user',
            'content': speech,
            'timestamp': time.time()
        })

        # Analyze social context
        context = self.analyze_social_context()

        # Generate context-aware response
        response = self.generate_contextual_response(speech, context)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Publish appropriate gesture
        self.publish_social_gesture(speech)

        # Update status
        status_msg = String()
        status_msg.data = f'Social interaction: {response[:50]}...'
        self.status_pub.publish(status_msg)

    def vision_callback(self, msg):
        """Process vision input to detect people for social interaction"""
        new_people = {}

        for marker in msg.markers:
            person_id = f"person_{marker.id}"
            new_people[person_id] = {
                'position': (marker.pose.position.x, marker.pose.position.y),
                'last_seen': time.time(),
                'greeting_given': self.detected_people.get(person_id, {}).get('greeting_given', False)
            }

        self.detected_people = new_people

        # Check for new people to greet
        self.check_for_new_people()

    def time_callback(self, msg):
        """Update current time"""
        # In a real system, this would come from a time service
        # For simulation, we'll just update the current time
        self.current_time = datetime.now()
        self.social_context['time_of_day'] = self.get_time_of_day()

    def get_time_of_day(self):
        """Get time of day based on current hour"""
        hour = self.current_time.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def analyze_social_context(self):
        """Analyze current social context"""
        context = {
            'time_of_day': self.social_context['time_of_day'],
            'people_count': len(self.detected_people),
            'conversation_active': self.social_context['conversation_active'],
            'last_interaction_time': self.social_context['last_interaction_time']
        }

        return context

    def generate_contextual_response(self, speech, context):
        """Generate response based on social context"""
        time_of_day = context['time_of_day']
        people_count = context['people_count']

        # Handle greetings based on time of day
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(keyword in speech for keyword in greeting_keywords):
            if time_of_day == 'morning':
                return "Good morning! How can I assist you today?"
            elif time_of_day == 'afternoon':
                return "Good afternoon! What can I help you with?"
            elif time_of_day == 'evening':
                return "Good evening! How are you doing?"
            else:
                return "Hello! How can I help you?"

        # Handle farewells
        farewell_keywords = ['goodbye', 'bye', 'see you', 'farewell', 'thanks', 'thank you']
        if any(keyword in speech for keyword in farewell_keywords):
            self.social_context['conversation_active'] = False
            import random
            return random.choice(self.farewell_responses)

        # Contextual responses based on number of people
        if people_count == 0:
            return "I don't see anyone around. Is someone there?"
        elif people_count == 1:
            if 'how are you' in speech:
                return "I'm doing well, thank you for asking! How can I assist you today?"
            elif 'what is your name' in speech or 'who are you' in speech:
                return "I'm a social robot designed to help and interact with people. You can call me Assistant."
        elif people_count > 1:
            if 'hello' in speech:
                return f"Hello everyone! It's nice to meet all of you."

        # Default response
        return f"I understand you said: '{speech}'. How can I help you?"

    def check_for_new_people(self):
        """Check for new people to greet"""
        current_time = time.time()

        for person_id, person_data in self.detected_people.items():
            # Greet if person is new and not yet greeted
            if not person_data.get('greeted', False):
                time_since_seen = current_time - person_data['last_seen']

                # Only greet if person has been visible for a moment (not just passing by)
                if time_since_seen > 1.0:
                    self.get_logger().info(f'New person detected: {person_id}')
                    self.greet_person(person_id)
                    self.detected_people[person_id]['greeted'] = True

    def greet_person(self, person_id):
        """Greet a newly detected person"""
        import random
        greeting = random.choice(self.greeting_responses)

        # Publish greeting response
        response_msg = String()
        response_msg.data = greeting
        self.response_pub.publish(response_msg)

        # Turn attention toward the person
        person_pos = self.detected_people[person_id]['position']
        attention_pose = Pose()
        attention_pose.position.x = person_pos[0]
        attention_pose.position.y = person_pos[1]
        attention_pose.position.z = 1.0  # Eye level
        attention_pose.orientation.w = 1.0

        self.attention_pub.publish(attention_pose)

        # Publish greeting gesture
        gesture_msg = String()
        gesture_msg.data = 'wave'
        self.gesture_pub.publish(gesture_msg)

    def publish_social_gesture(self, speech):
        """Publish appropriate social gesture based on speech"""
        speech_lower = speech.lower()

        # Choose gesture based on speech content
        if any(word in speech_lower for word in ['hello', 'hi', 'hey']):
            gesture_msg = String()
            gesture_msg.data = 'wave'
            self.gesture_pub.publish(gesture_msg)
        elif any(word in speech_lower for word in ['thank', 'thanks']):
            gesture_msg = String()
            gesture_msg.data = 'nod'
            self.gesture_pub.publish(gesture_msg)
        elif any(word in speech_lower for word in ['goodbye', 'bye', 'see you']):
            gesture_msg = String()
            gesture_msg.data = 'wave_goodbye'
            self.gesture_pub.publish(gesture_msg)

    def get_logger(self):
        """Get logger instance"""
        return self.get_logger()


def main(args=None):
    rclpy.init(args=args)
    exercise = SocialInteractionExercise()

    try:
        rclpy.spin(exercise)
    except KeyboardInterrupt:
        pass
    finally:
        exercise.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Exercise Instructions

1. **Setup**: Ensure your robot has people detection capabilities
2. **Social Interaction**: Speak to the robot and observe its responses
3. **Context Awareness**: Notice how responses change based on time of day
4. **Gesture Feedback**: Observe social gestures like waving or nodding
5. **Attention**: See how the robot focuses on detected people

#### Exercise Solution Analysis

```python
# Solution walkthrough for Exercise 3

# Key components:
# 1. Social context analysis
# 2. Time-based responses
# 3. People detection and tracking
# 4. Social gesture generation

# The exercise demonstrates:
# - Context-aware behavior
# - Social interaction patterns
# - Multimodal social responses
# - Adaptive social behavior
```

## Exercise 4: Multimodal Task Completion

### Scenario: Complete Complex Task with Multiple Modalities

In this exercise, you'll implement a system that completes a complex task using multiple modalities: vision, voice, navigation, and manipulation.

#### Exercise Setup

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray
import numpy as np
import time
from enum import Enum
from collections import deque


class TaskState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    COMPLETED = "completed"
    ERROR = "error"


class MultimodalTaskExercise(Node):
    def __init__(self):
        super().__init__('multimodal_task_exercise')

        # Subscriptions
        self.voice_sub = self.create_subscription(
            String, '/voice_command', self.voice_callback, 10)
        self.vision_sub = self.create_subscription(
            MarkerArray, '/detected_objects', self.vision_callback, 10)
        self.navigation_feedback_sub = self.create_subscription(
            String, '/navigation_feedback', self.navigation_feedback_callback, 10)

        # Publishers
        self.navigation_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.manipulation_pub = self.create_publisher(
            String, '/manipulation_command', 10)
        self.response_pub = self.create_publisher(
            String, '/task_response', 10)
        self.status_pub = self.create_publisher(
            String, '/exercise_status', 10)

        # System state
        self.current_task_state = TaskState.IDLE
        self.current_task = None
        self.detected_objects = {}
        self.robot_pose = Pose()
        self.task_history = deque(maxlen=50)

        # Task definitions
        self.available_tasks = {
            'bring_water': {
                'description': 'Bring water to a person',
                'steps': ['find_water', 'navigate_to_water', 'grasp_water', 'find_person', 'navigate_to_person', 'deliver_water']
            },
            'serve_drink': {
                'description': 'Serve a drink to someone',
                'steps': ['find_drink', 'navigate_to_drink', 'grasp_drink', 'find_recipient', 'navigate_to_recipient', 'offer_drink']
            },
            'tidy_up': {
                'description': 'Clean up objects',
                'steps': ['find_objects', 'plan_sequence', 'pick_up_object', 'place_in_bin']
            }
        }

        self.get_logger().info('Multimodal task exercise initialized')

    def voice_callback(self, msg):
        """Process voice commands for complex tasks"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received task command: {command}')

        # Parse the command to identify the task
        task = self.parse_task_command(command)

        if task:
            self.start_task(task, command)
        else:
            response = "I didn't understand that task. Available tasks: bring water, serve drink, tidy up."
            self.publish_response(response)

    def parse_task_command(self, command):
        """Parse voice command to identify the requested task"""
        command_lower = command.lower()

        # Match against available tasks
        for task_name, task_info in self.available_tasks.items():
            if task_name.replace('_', ' ') in command_lower:
                return {
                    'name': task_name,
                    'info': task_info,
                    'original_command': command
                }

        # Check for synonyms
        if any(word in command_lower for word in ['water', 'drink', 'get water', 'bring water']):
            return {
                'name': 'bring_water',
                'info': self.available_tasks['bring_water'],
                'original_command': command
            }

        return None

    def start_task(self, task, original_command):
        """Start executing a complex task"""
        self.current_task = task
        self.current_task_state = TaskState.PROCESSING

        response = f"Starting task: {task['name'].replace('_', ' ')}. I'll break this down into steps."
        self.publish_response(response)

        # Log the task
        self.task_history.append({
            'task': task['name'],
            'command': original_command,
            'start_time': time.time(),
            'state': self.current_task_state.value
        })

        # Begin task execution
        self.execute_task_step(0)

    def execute_task_step(self, step_index):
        """Execute a specific step of the current task"""
        if not self.current_task:
            return

        task_steps = self.current_task['info']['steps']

        if step_index >= len(task_steps):
            # Task completed
            self.current_task_state = TaskState.COMPLETED
            response = f"Task {self.current_task['name']} completed successfully!"
            self.publish_response(response)
            self.publish_status(f"Task completed: {self.current_task['name']}")
            return

        current_step = task_steps[step_index]

        # Execute the current step
        self.get_logger().info(f'Executing step: {current_step}')

        if current_step.startswith('find_'):
            self.find_object_step(current_step)
        elif current_step.startswith('navigate_to_'):
            self.navigate_step(current_step)
        elif current_step.startswith('grasp_') or current_step.startswith('pick_up_'):
            self.manipulation_step(current_step)
        elif current_step.startswith('deliver_') or current_step.startswith('offer_'):
            self.delivery_step(current_step)
        elif current_step.startswith('place_'):
            self.placement_step(current_step)

        # Schedule next step after delay
        self.get_logger().info(f'Step {step_index + 1}/{len(task_steps)} completed')

    def find_object_step(self, step):
        """Find a specific object for the task"""
        obj_type = step.replace('find_', '')
        response = f"Looking for {obj_type}..."

        # Check if object is already detected
        for obj_id, obj_data in self.detected_objects.items():
            if obj_data['class'] == obj_type:
                response = f"I found the {obj_type}. Proceeding with task."
                break
        else:
            response = f"I need to search for the {obj_type}. This may take a moment."

        self.publish_response(response)
        self.publish_status(f"Finding {obj_type}")

        # Continue to next step after a delay
        time.sleep(1)  # Simulate search time
        self.continue_task()

    def navigate_step(self, step):
        """Navigate to a location for the task"""
        target = step.replace('navigate_to_', '')
        response = f"Navigating to {target}..."

        # In a real implementation, calculate navigation goal
        # For simulation, use mock navigation
        nav_pose = self.calculate_navigation_pose(target)
        self.navigation_pub.publish(nav_pose)

        self.publish_response(response)
        self.publish_status(f"Navigating to {target}")

        # Continue to next step after navigation
        time.sleep(2)  # Simulate navigation time
        self.continue_task()

    def manipulation_step(self, step):
        """Perform manipulation action"""
        action = step.replace('grasp_', '').replace('pick_up_', '')
        response = f"Grasping the {action}..."

        # Publish manipulation command
        manip_cmd = String()
        manip_cmd.data = f'grasp_{action}'
        self.manipulation_pub.publish(manip_cmd)

        self.publish_response(response)
        self.publish_status(f"Grasping {action}")

        # Continue to next step after manipulation
        time.sleep(1)  # Simulate manipulation time
        self.continue_task()

    def delivery_step(self, step):
        """Deliver/offer an object"""
        obj_type = step.replace('deliver_', '').replace('offer_', '')
        response = f"Offering the {obj_type}..."

        # Publish delivery command
        delivery_cmd = String()
        delivery_cmd.data = f'deliver_{obj_type}'
        self.manipulation_pub.publish(delivery_cmd)

        self.publish_response(response)
        self.publish_status(f"Delivering {obj_type}")

        # Continue to next step
        time.sleep(1)  # Simulate delivery time
        self.continue_task()

    def placement_step(self, step):
        """Place object in designated location"""
        response = f"Placing object in bin..."

        # Publish placement command
        place_cmd = String()
        place_cmd.data = 'place_in_bin'
        self.manipulation_pub.publish(place_cmd)

        self.publish_response(response)
        self.publish_status("Placing object")

        # Continue to next step
        time.sleep(1)  # Simulate placement time
        self.continue_task()

    def calculate_navigation_pose(self, target):
        """Calculate navigation pose for target"""
        # In a real implementation, this would use a map
        # For simulation, use mock locations
        locations = {
            'water': (2.0, 1.0, 0.0),
            'drink': (2.0, 1.0, 0.0),
            'person': (0.0, 0.0, 0.0),
            'recipient': (0.0, 0.0, 0.0),
            'bin': (-1.0, -1.0, 0.0)
        }

        if target in locations:
            x, y, theta = locations[target]
        else:
            # Default location
            x, y, theta = (1.0, 1.0, 0.0)

        # Create navigation pose
        nav_pose = Pose()
        nav_pose.position.x = x
        nav_pose.position.y = y
        nav_pose.position.z = 0.0

        # Convert theta to quaternion
        from math import sin, cos
        cy = cos(theta * 0.5)
        sy = sin(theta * 0.5)
        nav_pose.orientation.z = sy
        nav_pose.orientation.w = cy

        return nav_pose

    def vision_callback(self, msg):
        """Process vision data to update detected objects"""
        for marker in msg.markers:
            obj_id = f"obj_{marker.id}"
            self.detected_objects[obj_id] = {
                'class': marker.ns,
                'position': (marker.pose.position.x, marker.pose.position.y),
                'confidence': marker.color.a
            }

    def navigation_feedback_callback(self, msg):
        """Process navigation feedback"""
        feedback = msg.data.lower()

        if 'arrived' in feedback or 'reached' in feedback:
            self.get_logger().info('Navigation completed, continuing task')
            self.continue_task()

    def continue_task(self):
        """Continue to the next step of the current task"""
        if not self.current_task:
            return

        # Find current step index
        task_steps = self.current_task['info']['steps']
        current_step_idx = 0

        # In a real implementation, track current step
        # For simulation, we'll just increment
        self.get_logger().info('Continuing to next task step')

        # For this simulation, we'll just execute the next step
        # In a real system, you'd track which step was last completed
        pass

    def publish_response(self, response):
        """Publish response message"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

    def publish_status(self, status):
        """Publish exercise status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def get_logger(self):
        """Get logger instance"""
        return self.get_logger()


def main(args=None):
    rclpy.init(args=args)
    exercise = MultimodalTaskExercise()

    try:
        rclpy.spin(exercise)
    except KeyboardInterrupt:
        pass
    finally:
        exercise.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Exercise Instructions

1. **Setup**: Ensure your robot has all required capabilities (vision, navigation, manipulation)
2. **Task Commands**: Try commands like "bring water" or "serve drink"
3. **Step-by-Step Execution**: Observe how the task is broken down into steps
4. **Multimodal Integration**: See how different modalities work together
5. **Task Completion**: Monitor the task completion process

#### Exercise Solution Analysis

```python
# Solution walkthrough for Exercise 4

# Key components:
# 1. Task decomposition
# 2. State management for complex tasks
# 3. Multimodal coordination
# 4. Step-by-step execution

# The exercise demonstrates:
# - Complex task planning and execution
# - Integration of multiple AI capabilities
# - State-based task management
# - Error handling in complex tasks
```

## Exercise 5: Adaptive Learning System

### Scenario: Robot that Learns from Interaction

In this exercise, you'll implement a system where the robot learns and adapts its behavior based on user interactions.

#### Exercise Setup

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import numpy as np
import time
from collections import defaultdict, deque
import pickle
import os


class AdaptiveLearningExercise(Node):
    def __init__(self):
        super().__init__('adaptive_learning_exercise')

        # Subscriptions
        self.interaction_sub = self.create_subscription(
            String, '/user_interaction', self.interaction_callback, 10)
        self.feedback_sub = self.create_subscription(
            String, '/user_feedback', self.feedback_callback, 10)

        # Publishers
        self.response_pub = self.create_publisher(
            String, '/adaptive_response', 10)
        self.behavior_pub = self.create_publisher(
            String, '/behavior_adjustment', 10)
        self.status_pub = self.create_publisher(
            String, '/exercise_status', 10)

        # Learning system state
        self.user_preferences = defaultdict(lambda: {'frequency': 0, 'success_rate': 0.0})
        self.interaction_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=100)
        self.personalization_model = {}
        self.learning_enabled = True

        # Load any saved learning data
        self.load_learning_data()

        # Interaction patterns
        self.common_requests = {
            'greeting': ['hello', 'hi', 'hey'],
            'time': ['what time is it', 'time', 'clock'],
            'weather': ['weather', 'temperature'],
            'help': ['help', 'assist', 'what can you do'],
            'navigation': ['go to', 'move to', 'navigate'],
            'fetch': ['get', 'bring', 'fetch', 'pick up']
        }

        self.get_logger().info('Adaptive learning exercise initialized')

    def interaction_callback(self, msg):
        """Process user interactions for learning"""
        interaction = msg.data.lower().strip()
        self.get_logger().info(f'Interaction: {interaction}')

        # Add to interaction history
        self.interaction_history.append({
            'input': interaction,
            'timestamp': time.time(),
            'response_type': self.classify_interaction(interaction)
        })

        # Learn from the interaction
        if self.learning_enabled:
            self.update_user_preferences(interaction)

        # Generate adaptive response
        response = self.generate_adaptive_response(interaction)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Update status
        status_msg = String()
        status_msg.data = f'Learned from interaction: {interaction[:30]}...'
        self.status_pub.publish(status_msg)

    def feedback_callback(self, msg):
        """Process user feedback to adjust behavior"""
        feedback = msg.data.lower().strip()
        self.get_logger().info(f'Feedback: {feedback}')

        # Add to feedback history
        self.feedback_history.append({
            'feedback': feedback,
            'timestamp': time.time()
        })

        # Adjust behavior based on feedback
        if self.learning_enabled:
            self.adjust_behavior_from_feedback(feedback)

    def classify_interaction(self, interaction):
        """Classify the type of interaction"""
        interaction_lower = interaction.lower()

        for category, patterns in self.common_requests.items():
            for pattern in patterns:
                if pattern in interaction_lower:
                    return category

        return 'unknown'

    def update_user_preferences(self, interaction):
        """Update user preferences based on interaction"""
        interaction_type = self.classify_interaction(interaction)

        # Update frequency
        self.user_preferences[interaction_type]['frequency'] += 1

        # Calculate success rate (in real implementation, this would be based on task completion)
        total_interactions = sum(
            self.user_preferences[k]['frequency'] for k in self.user_preferences.keys()
        )

        if total_interactions > 0:
            for key in self.user_preferences.keys():
                self.user_preferences[key]['success_rate'] = min(
                    1.0, self.user_preferences[key]['frequency'] / total_interactions * 2
                )

    def generate_adaptive_response(self, interaction):
        """Generate response based on learned preferences"""
        interaction_type = self.classify_interaction(interaction)

        # Get preference for this interaction type
        pref = self.user_preferences[interaction_type]
        frequency = pref['frequency']
        success_rate = pref['success_rate']

        # Adapt response based on preferences
        if interaction_type == 'greeting':
            if frequency > 5:  # Frequent user
                return "Hello again! How can I help you today?"
            else:
                return "Hello! It's nice to meet you. How can I assist you?"

        elif interaction_type == 'time':
            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M")
            return f"The current time is {current_time}. I've provided this information {frequency} times."

        elif interaction_type == 'help':
            if frequency > 3:
                return "I can help with navigation, object fetching, answering questions, and social interaction. I see you've asked for help {frequency} times."
            else:
                return "I can help with navigation, object fetching, answering questions, and social interaction."

        elif interaction_type == 'navigation':
            if success_rate > 0.8:
                return "I'll help you navigate. I've been successful with navigation requests recently."
            else:
                return "I'll help you navigate. Please guide me if I make mistakes."

        elif interaction_type == 'fetch':
            if success_rate > 0.8:
                return "I'll fetch that for you. I've been successful with fetching tasks."
            else:
                return "I'll try to fetch that for you. I'm still learning to improve."

        else:
            return f"I understand you said: '{interaction}'. I'm learning to better assist you based on our interactions."

    def adjust_behavior_from_feedback(self, feedback):
        """Adjust robot behavior based on user feedback"""
        feedback_lower = feedback.lower()

        # Positive feedback
        if any(word in feedback_lower for word in ['good', 'great', 'excellent', 'thank you', 'perfect']):
            # Increase preference weights
            pass

        # Negative feedback
        elif any(word in feedback_lower for word in ['bad', 'wrong', 'incorrect', 'not', 'stop']):
            # Decrease preference weights or trigger learning adjustment
            pass

        # Confusion feedback
        elif any(word in feedback_lower for word in ['confused', 'repeat', 'again', 'what']):
            # Adjust response complexity or provide more clarification
            pass

    def save_learning_data(self):
        """Save learned data to persistent storage"""
        data = {
            'user_preferences': dict(self.user_preferences),
            'interaction_history': list(self.interaction_history),
            'feedback_history': list(self.feedback_history),
            'personalization_model': self.personalization_model
        }

        try:
            with open('learning_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            self.get_logger().info('Learning data saved')
        except Exception as e:
            self.get_logger().error(f'Error saving learning data: {e}')

    def load_learning_data(self):
        """Load previously learned data"""
        if os.path.exists('learning_data.pkl'):
            try:
                with open('learning_data.pkl', 'rb') as f:
                    data = pickle.load(f)

                self.user_preferences.update(data.get('user_preferences', {}))
                self.interaction_history.extend(data.get('interaction_history', []))
                self.feedback_history.extend(data.get('feedback_history', []))
                self.personalization_model = data.get('personalization_model', {})

                self.get_logger().info('Learning data loaded')
            except Exception as e:
                self.get_logger().error(f'Error loading learning data: {e}')

    def get_logger(self):
        """Get logger instance"""
        return self.get_logger()


def main(args=None):
    rclpy.init(args=args)
    exercise = AdaptiveLearningExercise()

    try:
        rclpy.spin(exercise)
    except KeyboardInterrupt:
        # Save learning data before exit
        exercise.save_learning_data()
    finally:
        exercise.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Exercise Instructions

1. **Setup**: Ensure the system can save/load learning data
2. **Interaction**: Interact with the robot multiple times
3. **Observe Adaptation**: Notice how responses change based on interaction history
4. **Provide Feedback**: Give positive/negative feedback to adjust behavior
5. **Persistence**: Check that learning persists across sessions

#### Exercise Solution Analysis

```python
# Solution walkthrough for Exercise 5

# Key components:
# 1. Preference learning
# 2. Behavioral adaptation
# 3. Persistent storage
# 4. Feedback processing

# The exercise demonstrates:
# - Machine learning in robotics
# - Personalization systems
# - Adaptive behavior
# - User modeling
```

## Comprehensive Exercise: Complete AI Robot System

### Scenario: Full Integration of All Capabilities

This final exercise integrates all the capabilities learned in previous exercises into a complete AI robot brain system.

#### Complete System Implementation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray
import numpy as np
import time
from datetime import datetime
from collections import defaultdict, deque
import threading
import queue


class CompleteAIRobotSystem(Node):
    def __init__(self):
        super().__init__('complete_ai_robot_system')

        # Initialize all subsystems
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()
        self.nlp_processor = NLPProcessor()
        self.action_executor = ActionExecutor()
        self.learning_system = LearningSystem()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.voice_sub = self.create_subscription(
            String, '/voice_input', self.voice_callback, 10)
        self.feedback_sub = self.create_subscription(
            String, '/user_feedback', self.feedback_callback, 10)

        # Publishers
        self.response_pub = self.create_publisher(
            String, '/robot_response', 10)
        self.action_pub = self.create_publisher(
            String, '/robot_action', 10)
        self.visualization_pub = self.create_publisher(
            MarkerArray, '/robot_visualization', 10)
        self.status_pub = self.create_publisher(
            String, '/system_status', 10)

        # System state
        self.system_active = True
        self.current_context = {
            'time': datetime.now(),
            'detected_objects': [],
            'detected_people': [],
            'conversation_history': deque(maxlen=20),
            'task_queue': deque(),
            'system_state': 'idle'
        }

        # Processing queues
        self.vision_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()

        # Start processing threads
        self.start_processing_threads()

        self.get_logger().info('Complete AI robot system initialized')

    def start_processing_threads(self):
        """Start background processing threads"""
        # Vision processing thread
        vision_thread = threading.Thread(target=self.vision_processing_loop)
        vision_thread.daemon = True
        vision_thread.start()

        # Audio processing thread
        audio_thread = threading.Thread(target=self.audio_processing_loop)
        audio_thread.daemon = True
        audio_thread.start()

        # Main processing thread
        main_thread = threading.Thread(target=self.main_processing_loop)
        main_thread.daemon = True
        main_thread.start()

    def image_callback(self, msg):
        """Handle incoming image data"""
        self.vision_queue.put(msg)

    def voice_callback(self, msg):
        """Handle incoming voice commands"""
        self.audio_queue.put(msg)

    def feedback_callback(self, msg):
        """Handle user feedback"""
        feedback = msg.data
        self.learning_system.process_feedback(feedback)

    def vision_processing_loop(self):
        """Background thread for vision processing"""
        while self.system_active:
            try:
                # Get image from queue
                image_msg = self.vision_queue.get(timeout=0.1)

                # Process image
                vision_result = self.vision_processor.process_image(image_msg)

                # Update context with vision results
                self.current_context['detected_objects'] = vision_result.get('objects', [])
                self.current_context['detected_people'] = vision_result.get('people', [])

                # Publish visualization
                visualization = self.create_visualization_markers(vision_result)
                self.visualization_pub.publish(visualization)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Vision processing error: {e}')

    def audio_processing_loop(self):
        """Background thread for audio processing"""
        while self.system_active:
            try:
                # Get audio from queue
                audio_msg = self.audio_queue.get(timeout=0.1)

                # Process audio
                audio_result = self.audio_processor.process_audio(audio_msg)

                # Add to processing queue
                self.processing_queue.put({
                    'type': 'audio',
                    'data': audio_result,
                    'timestamp': time.time()
                })

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {e}')

    def main_processing_loop(self):
        """Main processing loop for integrating all modalities"""
        while self.system_active:
            try:
                # Get item from processing queue
                item = self.processing_queue.get(timeout=0.1)

                if item['type'] == 'audio':
                    # Process the audio command
                    self.process_voice_command(item['data'])

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Main processing error: {e}')

    def process_voice_command(self, audio_data):
        """Process voice command by integrating all modalities"""
        # Extract text from audio
        text = audio_data.get('text', '')

        # Add to conversation history
        self.current_context['conversation_history'].append({
            'type': 'user',
            'text': text,
            'timestamp': time.time()
        })

        # Update system state
        self.current_context['system_state'] = 'processing'

        # Use NLP to understand the command
        nlp_result = self.nlp_processor.process_command(text, self.current_context)

        # Learn from the interaction
        self.learning_system.process_interaction(text, nlp_result)

        # Execute the action
        action_result = self.action_executor.execute_action(nlp_result, self.current_context)

        # Generate response
        response = self.generate_response(nlp_result, action_result)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Publish action if needed
        if action_result.get('action_needed'):
            action_msg = String()
            action_msg.data = action_result['action']
            self.action_pub.publish(action_msg)

        # Update conversation history with response
        self.current_context['conversation_history'].append({
            'type': 'system',
            'text': response,
            'timestamp': time.time()
        })

        # Update system state
        self.current_context['system_state'] = 'idle'

        # Publish status
        status_msg = String()
        status_msg.data = f'Processed: {text[:50]}...'
        self.status_pub.publish(status_msg)

    def generate_response(self, nlp_result, action_result):
        """Generate appropriate response based on processing results"""
        intent = nlp_result.get('intent', 'unknown')
        action_status = action_result.get('status', 'unknown')

        # Generate response based on intent and action status
        if action_status == 'success':
            if intent == 'navigation':
                return f"I've navigated to the {nlp_result.get('destination', 'location')}."
            elif intent == 'manipulation':
                return f"I've {nlp_result.get('action', 'performed the action')} the {nlp_result.get('object', 'object')}."
            elif intent == 'information':
                return f"Here's the information you requested: {action_result.get('data', 'N/A')}."
            else:
                return "I've completed the requested action successfully."
        elif action_status == 'failed':
            return f"I couldn't complete that action: {action_result.get('error', 'Unknown error')}."
        else:
            return "I'm processing your request. Please wait a moment."

    def create_visualization_markers(self, vision_result):
        """Create visualization markers for detected objects"""
        marker_array = MarkerArray()

        # Add markers for detected objects
        for i, obj in enumerate(vision_result.get('objects', [])):
            marker = self.create_object_marker(obj, i)
            marker_array.markers.append(marker)

        # Add markers for detected people
        for i, person in enumerate(vision_result.get('people', []), start=len(marker_array.markers)):
            marker = self.create_person_marker(person, i)
            marker_array.markers.append(marker)

        return marker_array

    def create_object_marker(self, obj, marker_id):
        """Create a marker for a detected object"""
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "objects"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = obj.get('x', 0.0)
        marker.pose.position.y = obj.get('y', 0.0)
        marker.pose.position.z = obj.get('z', 0.0)
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        marker.text = obj.get('class', 'unknown')

        return marker

    def create_person_marker(self, person, marker_id):
        """Create a marker for a detected person"""
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "people"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = person.get('x', 0.0)
        marker.pose.position.y = person.get('y', 0.0)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 1.7

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        marker.text = "Person"

        return marker

    def get_logger(self):
        """Get logger instance"""
        return self.get_logger()


class VisionProcessor:
    """Handles computer vision processing"""
    def __init__(self):
        self.object_classes = ['person', 'chair', 'table', 'cup', 'bottle', 'phone']
        self.detection_threshold = 0.5

    def process_image(self, image_msg):
        """Process image and detect objects"""
        # In a real implementation, this would run through a detection model
        # For simulation, return mock detections
        return {
            'objects': [
                {'class': 'chair', 'x': 1.0, 'y': 1.0, 'z': 0.0, 'confidence': 0.8},
                {'class': 'table', 'x': 2.0, 'y': 0.5, 'z': 0.0, 'confidence': 0.9}
            ],
            'people': [
                {'x': 0.0, 'y': 0.0, 'z': 0.0, 'confidence': 0.95}
            ]
        }


class AudioProcessor:
    """Handles audio processing and speech recognition"""
    def __init__(self):
        self.sample_rate = 16000
        self.energy_threshold = 300

    def process_audio(self, audio_msg):
        """Process audio message and extract text"""
        # In a real implementation, this would perform speech recognition
        # For simulation, return mock text
        return {
            'text': 'Please navigate to the kitchen',
            'confidence': 0.85,
            'timestamp': time.time()
        }


class NLPProcessor:
    """Handles natural language processing"""
    def __init__(self):
        self.intent_patterns = {
            'navigation': ['go to', 'move to', 'navigate to', 'walk to'],
            'manipulation': ['get', 'bring', 'pick up', 'grab', 'fetch'],
            'information': ['what is', 'tell me', 'how to', 'when', 'where'],
            'social': ['hello', 'hi', 'goodbye', 'thank you']
        }

    def process_command(self, text, context):
        """Process natural language command"""
        text_lower = text.lower()

        # Determine intent
        intent = 'unknown'
        for intent_type, patterns in self.intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                intent = intent_type
                break

        # Extract entities
        entities = self.extract_entities(text, intent)

        return {
            'intent': intent,
            'entities': entities,
            'original_text': text,
            'confidence': 0.8
        }

    def extract_entities(self, text, intent):
        """Extract relevant entities from text"""
        entities = {}

        if intent == 'navigation':
            # Extract destination
            for word in text.lower().split():
                if word in ['kitchen', 'living room', 'bedroom', 'office']:
                    entities['destination'] = word
                    break

        elif intent == 'manipulation':
            # Extract object
            for word in text.lower().split():
                if word in ['cup', 'bottle', 'book', 'phone']:
                    entities['object'] = word
                    break

        return entities


class ActionExecutor:
    """Executes robot actions"""
    def __init__(self):
        self.action_history = []

    def execute_action(self, nlp_result, context):
        """Execute the parsed action"""
        intent = nlp_result['intent']
        entities = nlp_result['entities']

        try:
            if intent == 'navigation':
                destination = entities.get('destination', 'unknown')
                # In a real implementation, this would send navigation command
                return {
                    'status': 'success',
                    'action': f'navigate_to_{destination}',
                    'result': f'Navigated to {destination}'
                }

            elif intent == 'manipulation':
                obj = entities.get('object', 'unknown')
                # In a real implementation, this would send manipulation command
                return {
                    'status': 'success',
                    'action': f'grasp_{obj}',
                    'result': f'Grasped {obj}'
                }

            elif intent == 'information':
                # Provide requested information
                return {
                    'status': 'success',
                    'action': 'provide_information',
                    'data': 'I am a humanoid robot designed to assist with various tasks.',
                    'result': 'Provided information'
                }

            else:
                return {
                    'status': 'unknown_intent',
                    'error': f'Unknown intent: {intent}'
                }

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }


class LearningSystem:
    """Handles learning and adaptation"""
    def __init__(self):
        self.interaction_memory = deque(maxlen=1000)
        self.user_preferences = defaultdict(lambda: {'count': 0, 'success_rate': 0.0})

    def process_interaction(self, text, nlp_result):
        """Process an interaction for learning"""
        interaction = {
            'text': text,
            'nlp_result': nlp_result,
            'timestamp': time.time()
        }

        self.interaction_memory.append(interaction)

        # Update preferences
        intent = nlp_result.get('intent', 'unknown')
        self.user_preferences[intent]['count'] += 1

    def process_feedback(self, feedback):
        """Process user feedback for learning"""
        feedback_lower = feedback.lower()

        # In a real implementation, this would adjust behavior based on feedback
        # For now, just log the feedback
        pass


def main(args=None):
    rclpy.init(args=args)
    system = CompleteAIRobotSystem()

    try:
        rclpy.spin(system)
    except KeyboardInterrupt:
        pass
    finally:
        system.system_active = False
        system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise Solutions and Best Practices

### Solution Guidelines

For each exercise, consider these solution approaches:

1. **Integration First**: Always consider how components work together
2. **State Management**: Use clear state machines for complex tasks
3. **Error Handling**: Implement graceful degradation
4. **Performance**: Optimize for real-time operation
5. **Testing**: Create comprehensive test cases
6. **Modularity**: Keep components loosely coupled

### Best Practices for AI Robot Brain Development

1. **Modular Architecture**: Design systems as independent, testable modules
2. **Real-time Performance**: Optimize for responsive interaction
3. **Robustness**: Handle edge cases and failures gracefully
4. **User Experience**: Prioritize natural, intuitive interaction
5. **Safety**: Implement safety checks and emergency procedures
6. **Privacy**: Protect user data and interactions
7. **Scalability**: Design for future capability expansion

## Troubleshooting Common Exercise Issues

### Issue 1: Component Integration Problems
**Symptoms**: Components don't work together as expected
**Solutions**:
- Check message formats and data types
- Verify topic names and publishers/subscribers
- Use ROS2 tools like `ros2 topic echo` to debug
- Implement proper error handling between components

### Issue 2: Performance Bottlenecks
**Symptoms**: System is slow or unresponsive
**Solutions**:
- Use asynchronous processing where possible
- Implement priority-based task scheduling
- Optimize critical code paths
- Monitor system resources

### Issue 3: State Management Issues
**Symptoms**: System behaves inconsistently or loses context
**Solutions**:
- Implement clear state machines
- Use proper state persistence
- Add state validation checks
- Design for graceful state transitions

## Key Takeaways

- AI robot brain systems integrate multiple modalities (vision, audio, language, action)
- Exercises provide hands-on experience with real-world scenarios
- State management is crucial for complex tasks
- Performance optimization ensures responsive interaction
- Learning systems adapt to user preferences over time
- Comprehensive testing validates system reliability
- Best practices ensure robust, maintainable code

These exercises provide a solid foundation for developing sophisticated AI robot brain systems that can perceive, understand, and act in human environments.