---
sidebar_position: 4
---

# Multimodal Pipeline Examples: Connecting Vision, Language, and Action

Welcome to the Multimodal Pipeline Examples module, which demonstrates practical implementations of systems that connect vision, language, and action for humanoid robots. This chapter provides working examples that show how to integrate perception, language understanding, and action execution in real-world scenarios.

## Learning Objectives

By the end of this section, you will be able to:
- Implement multimodal systems that connect vision, language, and action
- Create practical examples of humanoid robot behaviors
- Integrate perception and language for task execution
- Design multimodal interaction flows for human-robot communication
- Implement safety checks and validation for multimodal systems
- Troubleshoot common issues in multimodal integration
- Evaluate multimodal system performance

## Introduction to Multimodal Integration

Multimodal integration connects perception (vision), cognition (language), and action (execution) to create natural and intuitive human-robot interaction. For humanoid robots, this integration is essential for performing complex tasks in human environments.

### Multimodal Architecture Pattern

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │   Language      │    │   Action        │
│   (Vision)      │───▶│   Understanding │───▶│   Execution     │
│   • Objects     │    │   • Commands    │    │   • Navigation  │
│   • People      │    │   • Context     │    │   • Manipulation│
│   • Environment │    │   • Intent      │    │   • Interaction │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal Controller                        │
│              (Decision Making & Coordination)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Example 1: Object Fetching with Natural Language

### Scenario: Fetching a Specific Object

In this example, we'll implement a complete system that allows a humanoid robot to understand natural language commands to fetch specific objects from the environment.

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import MarkerArray
import numpy as np
import json
import time
from collections import defaultdict, deque


class ObjectFetchingSystem(Node):
    def __init__(self):
        super().__init__('object_fetching_system')

        # Subscriptions
        self.command_sub = self.create_subscription(
            String, '/user_command', self.command_callback, 10)
        self.vision_sub = self.create_subscription(
            MarkerArray, '/detected_objects', self.vision_callback, 10)
        self.current_pose_sub = self.create_subscription(
            Pose, '/current_pose', self.pose_callback, 10)

        # Publishers
        self.navigation_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.manipulation_pub = self.create_publisher(
            String, '/manipulation_command', 10)
        self.response_pub = self.create_publisher(
            String, '/robot_response', 10)
        self.status_pub = self.create_publisher(
            String, '/system_status', 10)

        # System state
        self.detected_objects = {}
        self.current_pose = Pose()
        self.current_task = None
        self.task_queue = deque()
        self.conversation_history = deque(maxlen=10)

        # Object recognition mapping
        self.object_synonyms = {
            'bottle': ['water', 'drink', 'liquid', 'container'],
            'cup': ['mug', 'glass', 'drinking', 'coffee', 'tea'],
            'book': ['textbook', 'novel', 'reading', 'paperback'],
            'phone': ['mobile', 'cell', 'smartphone', 'device'],
            'keys': ['keychain', 'car keys', 'house keys'],
            'ball': ['toy', 'plaything', 'sports equipment']
        }

        # Initialize the system
        self.get_logger().info('Object fetching system initialized')

    def command_callback(self, msg):
        """Process user commands for object fetching"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received command: {command}')

        # Add to conversation history
        self.conversation_history.append({
            'type': 'user_input',
            'content': command,
            'timestamp': time.time()
        })

        # Parse the command to extract object and action
        parsed_command = self.parse_fetch_command(command)

        if parsed_command:
            # Check if object is currently visible
            target_object = self.find_object_by_description(parsed_command['target'])

            if target_object:
                # Object is visible, move to it and grasp
                self.execute_fetch_task(target_object, parsed_command)
            else:
                # Object not visible, search for it
                self.search_for_object(parsed_command['target'])
        else:
            # Could not parse command, ask for clarification
            response = "I'm sorry, I didn't understand your request. Could you please specify which object you'd like me to fetch?"
            self.publish_response(response)

    def parse_fetch_command(self, command):
        """Parse natural language command to extract object and action"""
        command_lower = command.lower()

        # Define action keywords
        fetch_keywords = [
            'fetch', 'get', 'bring', 'pick up', 'grab', 'take', 'hand me',
            'give me', 'bring me', 'get me'
        ]

        # Check if command contains a fetch action
        has_fetch_action = any(keyword in command_lower for keyword in fetch_keywords)

        if not has_fetch_action:
            return None

        # Extract object description from command
        # This is a simplified approach - in practice, use NLP techniques
        object_words = [
            'bottle', 'cup', 'book', 'phone', 'keys', 'ball', 'toy',
            'water', 'drink', 'mug', 'glass', 'mobile', 'device'
        ]

        target_object = None
        for word in object_words:
            if word in command_lower:
                target_object = word
                break

        # If no direct match, try synonyms
        if not target_object:
            for obj, synonyms in self.object_synonyms.items():
                if any(syn in command_lower for syn in synonyms):
                    target_object = obj
                    break

        if target_object:
            return {
                'action': 'fetch',
                'target': target_object,
                'original_command': command
            }

        return None

    def find_object_by_description(self, description):
        """Find a detected object that matches the description"""
        # Look for exact matches first
        for obj_id, obj_data in self.detected_objects.items():
            if obj_data.get('class', '').lower() == description.lower():
                return obj_data

        # Look for synonyms
        for obj_id, obj_data in self.detected_objects.items():
            obj_class = obj_data.get('class', '').lower()
            if obj_class in self.object_synonyms.get(description, []):
                return obj_data

        # If no match found, return None
        return None

    def search_for_object(self, target_object):
        """Search for an object that is not currently visible"""
        self.get_logger().info(f'Searching for {target_object}')

        # Publish search response
        response = f"I don't see the {target_object} right now. I'll search for it."
        self.publish_response(response)

        # In a real system, this would implement a search pattern
        # For simulation, we'll just publish a search goal
        search_pose = self.calculate_search_pose(target_object)

        search_task = {
            'type': 'search',
            'target': target_object,
            'pose': search_pose,
            'status': 'in_progress'
        }

        self.task_queue.append(search_task)
        self.execute_next_task()

    def calculate_search_pose(self, target_object):
        """Calculate a search pose based on the target object"""
        # In a real system, this would use a map and search algorithm
        # For simulation, return a nearby pose
        search_pose = Pose()
        search_pose.position.x = self.current_pose.position.x + 1.0
        search_pose.position.y = self.current_pose.position.y + 1.0
        search_pose.position.z = 0.0
        search_pose.orientation.w = 1.0

        return search_pose

    def execute_fetch_task(self, target_object, command_data):
        """Execute the fetch task for a visible object"""
        self.get_logger().info(f'Executing fetch task for {target_object["class"]}')

        # Create fetch task
        fetch_task = {
            'type': 'fetch',
            'target_object': target_object,
            'command_data': command_data,
            'status': 'in_progress',
            'steps': [
                'navigate_to_object',
                'align_with_object',
                'grasp_object',
                'return_to_user'
            ],
            'current_step': 0
        }

        self.task_queue.append(fetch_task)
        self.execute_next_task()

    def execute_next_task(self):
        """Execute the next task in the queue"""
        if not self.task_queue or self.current_task is not None:
            return

        self.current_task = self.task_queue.popleft()
        task_type = self.current_task['type']

        if task_type == 'fetch':
            self.execute_fetch_steps()
        elif task_type == 'search':
            self.execute_search_steps()
        else:
            self.get_logger().warning(f'Unknown task type: {task_type}')
            self.complete_current_task()

    def execute_fetch_steps(self):
        """Execute the steps for a fetch task"""
        if self.current_task['current_step'] >= len(self.current_task['steps']):
            # All steps completed
            self.complete_current_task()
            return

        step = self.current_task['steps'][self.current_task['current_step']]

        if step == 'navigate_to_object':
            self.navigate_to_object()
        elif step == 'align_with_object':
            self.align_with_object()
        elif step == 'grasp_object':
            self.grasp_object()
        elif step == 'return_to_user':
            self.return_to_user()

    def navigate_to_object(self):
        """Navigate to the target object"""
        target_obj = self.current_task['target_object']

        # Calculate pose to approach the object
        approach_pose = Pose()
        approach_pose.position.x = target_obj['position']['x'] - 0.5  # 0.5m away
        approach_pose.position.y = target_obj['position']['y']
        approach_pose.position.z = 0.0
        approach_pose.orientation.w = 1.0

        # Publish navigation goal
        self.navigation_pub.publish(approach_pose)

        self.get_logger().info(f'Navigating to {self.current_task["target_object"]["class"]}')

        # Move to next step
        self.current_task['current_step'] += 1
        self.execute_fetch_steps()

    def align_with_object(self):
        """Align with the target object for grasping"""
        # In a real system, this would perform precise alignment
        # For simulation, just log and move to next step
        self.get_logger().info('Aligning with object for grasping')

        # Move to next step
        self.current_task['current_step'] += 1
        self.execute_fetch_steps()

    def grasp_object(self):
        """Grasp the target object"""
        target_obj = self.current_task['target_object']

        # Publish manipulation command
        grasp_cmd = String()
        grasp_cmd.data = f'grasp_{target_obj["class"]}'
        self.manipulation_pub.publish(grasp_cmd)

        self.get_logger().info(f'Grasping {target_obj["class"]}')

        # Publish success response
        response = f"I have picked up the {target_obj['class']}."
        self.publish_response(response)

        # Move to next step
        self.current_task['current_step'] += 1
        self.execute_fetch_steps()

    def return_to_user(self):
        """Return to the user with the object"""
        # In a real system, this would navigate back to the user
        # For simulation, just complete the task
        self.get_logger().info('Returning to user')

        # Publish completion response
        target_obj = self.current_task['target_object']
        response = f"I have brought you the {target_obj['class']}."
        self.publish_response(response)

        self.complete_current_task()

    def execute_search_steps(self):
        """Execute the steps for a search task"""
        # For simulation, just complete the search task
        search_task = self.current_task
        self.get_logger().info(f'Completed search for {search_task["target"]}')

        # In a real system, this would implement search patterns
        response = f"I've searched for the {search_task['target']} but couldn't find it."
        self.publish_response(response)

        self.complete_current_task()

    def complete_current_task(self):
        """Mark current task as completed and process next"""
        if self.current_task:
            self.get_logger().info(f'Completed task: {self.current_task["type"]}')
            self.current_task = None

        # Process next task if available
        if self.task_queue:
            self.execute_next_task()

    def vision_callback(self, msg):
        """Update detected objects from vision system"""
        # In a real system, this would parse the MarkerArray to extract object information
        # For simulation, we'll update a simple object dictionary

        # This is a simplified representation - in practice, parse the actual marker data
        for marker in msg.markers:
            obj_id = f"obj_{marker.id}"
            self.detected_objects[obj_id] = {
                'class': marker.ns,
                'position': {
                    'x': marker.pose.position.x,
                    'y': marker.pose.position.y,
                    'z': marker.pose.position.z
                },
                'confidence': marker.color.a
            }

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg

    def publish_response(self, response):
        """Publish robot response"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)
        self.get_logger().info(f'Robot response: {response}')


def main(args=None):
    rclpy.init(args=args)
    fetching_system = ObjectFetchingSystem()

    try:
        rclpy.spin(fetching_system)
    except KeyboardInterrupt:
        pass
    finally:
        fetching_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Example 2: Navigation with Natural Language Commands

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import OccupancyGrid
import numpy as np
import json
import math


class NavigationCommandSystem(Node):
    def __init__(self):
        super().__init__('navigation_command_system')

        # Subscriptions
        self.command_sub = self.create_subscription(
            String, '/navigation_command', self.command_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.current_pose_sub = self.create_subscription(
            Pose, '/current_pose', self.pose_callback, 10)

        # Publishers
        self.goal_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.response_pub = self.create_publisher(
            String, '/navigation_response', 10)
        self.path_pub = self.create_publisher(
            String, '/navigation_path', 10)

        # System state
        self.current_pose = Pose()
        self.map_data = None
        self.map_info = None
        self.navigation_tasks = []
        self.location_map = {
            'kitchen': (2.0, 1.0),
            'living room': (0.0, 2.0),
            'bedroom': (-1.0, -1.0),
            'office': (3.0, -2.0),
            'bathroom': (-2.0, 0.0),
            'dining room': (1.5, -1.5),
            'hallway': (0.0, 0.0)
        }

        self.get_logger().info('Navigation command system initialized')

    def command_callback(self, msg):
        """Process navigation commands in natural language"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received navigation command: {command}')

        # Parse the command
        parsed_command = self.parse_navigation_command(command)

        if parsed_command:
            # Execute navigation
            success = self.execute_navigation(parsed_command)

            if success:
                response = f"On my way to the {parsed_command['destination']}!"
            else:
                response = f"Sorry, I couldn't navigate to the {parsed_command['destination']}."
        else:
            response = "I didn't understand your navigation command. Please specify a destination."

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

    def parse_navigation_command(self, command):
        """Parse natural language navigation command"""
        command_lower = command.lower()

        # Define navigation keywords
        nav_keywords = [
            'go to', 'move to', 'navigate to', 'go', 'move', 'navigate',
            'walk to', 'head to', 'go over to', 'move toward'
        ]

        # Check if command contains navigation intent
        has_nav_intent = any(keyword in command_lower for keyword in nav_keywords)

        if not has_nav_intent:
            return None

        # Extract destination from command
        destinations = list(self.location_map.keys())
        destination = None

        for dest in destinations:
            if dest in command_lower:
                destination = dest
                break

        # Handle common destination synonyms
        if not destination:
            destination_synonyms = {
                'kitchen': ['kitchen', 'cooking', 'food'],
                'living room': ['living room', 'sofa', 'couch', 'tv', 'lounge'],
                'bedroom': ['bedroom', 'bed', 'sleep', 'room'],
                'office': ['office', 'work', 'desk', 'computer'],
                'bathroom': ['bathroom', 'bath', 'toilet', 'shower'],
                'dining room': ['dining room', 'dining', 'eat', 'table'],
                'hallway': ['hallway', 'hall', 'corridor', 'passage']
            }

            for dest, synonyms in destination_synonyms.items():
                if any(syn in command_lower for syn in synonyms):
                    destination = dest
                    break

        if destination:
            return {
                'action': 'navigate',
                'destination': destination,
                'target_pose': self.get_destination_pose(destination),
                'original_command': command
            }

        return None

    def get_destination_pose(self, destination):
        """Get the pose for a named destination"""
        if destination in self.location_map:
            x, y = self.location_map[destination]
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            return pose

        return None

    def execute_navigation(self, command_data):
        """Execute navigation to the specified destination"""
        target_pose = command_data['target_pose']

        if target_pose is None:
            self.get_logger().error(f'No pose found for destination: {command_data["destination"]}')
            return False

        # Check if destination is reachable (in a real system, plan path)
        if self.is_destination_reachable(target_pose):
            # Publish navigation goal
            self.goal_pub.publish(target_pose)
            self.get_logger().info(f'Navigating to {command_data["destination"]}: ({target_pose.position.x}, {target_pose.position.y})')
            return True
        else:
            self.get_logger().error(f'Destination not reachable: {command_data["destination"]}')
            return False

    def is_destination_reachable(self, pose):
        """Check if a destination is reachable based on map data"""
        # In a real system, this would use path planning algorithms
        # For simulation, assume all destinations are reachable
        return True

    def map_callback(self, msg):
        """Update internal map representation"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg


def main(args=None):
    rclpy.init(args=args)
    nav_system = NavigationCommandSystem()

    try:
        rclpy.spin(nav_system)
    except KeyboardInterrupt:
        pass
    finally:
        nav_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Example 3: Social Interaction with Multimodal Feedback

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
from collections import deque


class SocialInteractionSystem(Node):
    def __init__(self):
        super().__init__('social_interaction_system')

        # Subscriptions
        self.speech_sub = self.create_subscription(
            String, '/speech_input', self.speech_callback, 10)
        self.vision_sub = self.create_subscription(
            MarkerArray, '/detected_people', self.vision_callback, 10)
        self.gesture_sub = self.create_subscription(
            String, '/gesture_input', self.gesture_callback, 10)

        # Publishers
        self.speech_response_pub = self.create_publisher(
            String, '/speech_response', 10)
        self.gesture_pub = self.create_publisher(
            String, '/gesture_command', 10)
        self.expression_pub = self.create_publisher(
            String, '/expression_command', 10)
        self.attention_pub = self.create_publisher(
            Pose, '/attention_target', 10)

        # System state
        self.people_in_view = {}
        self.conversation_history = deque(maxlen=20)
        self.interaction_state = {
            'current_interlocutor': None,
            'conversation_active': False,
            'last_interaction_time': 0,
            'social_preferences': {}
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

        self.get_logger().info('Social interaction system initialized')

    def speech_callback(self, msg):
        """Process speech input for social interaction"""
        speech = msg.data.lower().strip()
        self.get_logger().info(f'Received speech: {speech}')

        # Add to conversation history
        self.conversation_history.append({
            'type': 'speech',
            'content': speech,
            'timestamp': time.time()
        })

        # Determine if this is a greeting, question, or other interaction
        response = self.generate_social_response(speech)

        if response:
            self.publish_speech_response(response)
            self.publish_gesture_response(speech)

    def vision_callback(self, msg):
        """Process vision input to detect people for social interaction"""
        # Update detected people
        for marker in msg.markers:
            person_id = f"person_{marker.id}"
            self.people_in_view[person_id] = {
                'position': (marker.pose.position.x, marker.pose.position.y),
                'last_seen': time.time(),
                'greeted': self.people_in_view.get(person_id, {}).get('greeted', False)
            }

        # Check for new people to greet
        self.check_for_new_people()

    def gesture_callback(self, msg):
        """Process gesture input"""
        gesture = msg.data.lower().strip()
        self.get_logger().info(f'Received gesture: {gesture}')

        # Add to conversation history
        self.conversation_history.append({
            'type': 'gesture',
            'content': gesture,
            'timestamp': time.time()
        })

        # Respond to gesture if appropriate
        if gesture == 'wave':
            response = "Hello! I see you're waving at me."
            self.publish_speech_response(response)
            self.publish_positive_expression()

    def generate_social_response(self, speech):
        """Generate appropriate social response based on input"""
        speech_lower = speech.lower()

        # Check for greetings
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(keyword in speech_lower for keyword in greeting_keywords):
            if not self.interaction_state['conversation_active']:
                self.interaction_state['conversation_active'] = True
                self.interaction_state['last_interaction_time'] = time.time()
                import random
                return random.choice(self.greeting_responses)

        # Check for farewells
        farewell_keywords = ['goodbye', 'bye', 'see you', 'farewell', 'thanks', 'thank you']
        if any(keyword in speech_lower for keyword in farewell_keywords):
            self.interaction_state['conversation_active'] = False
            import random
            return random.choice(self.farewell_responses)

        # Check for questions
        question_keywords = ['how are you', 'what is your name', 'who are you', 'what can you do']
        if any(keyword in speech_lower for keyword in question_keywords):
            if 'how are you' in speech_lower:
                return "I'm doing well, thank you for asking! How can I assist you today?"
            elif 'what is your name' in speech_lower or 'who are you' in speech_lower:
                return "I'm a humanoid robot designed to help with various tasks. You can call me Assistant."
            elif 'what can you do' in speech_lower:
                return "I can help with navigation, object fetching, answering questions, and social interaction."

        # Default response
        return "I understand you said: " + speech

    def check_for_new_people(self):
        """Check for new people to greet"""
        current_time = time.time()

        for person_id, person_data in self.people_in_view.items():
            # Greet if person is new and not yet greeted
            if not person_data.get('greeted', False):
                time_since_seen = current_time - person_data['last_seen']

                # Only greet if person has been visible for a moment (not just passing by)
                if time_since_seen > 1.0:
                    self.get_logger().info(f'New person detected: {person_id}')
                    self.greet_person(person_id)
                    self.people_in_view[person_id]['greeted'] = True

    def greet_person(self, person_id):
        """Greet a newly detected person"""
        import random
        greeting = random.choice(self.greeting_responses)

        self.publish_speech_response(greeting)
        self.publish_positive_expression()

        # Turn attention toward the person
        person_pos = self.people_in_view[person_id]['position']
        attention_pose = Pose()
        attention_pose.position.x = person_pos[0]
        attention_pose.position.y = person_pos[1]
        attention_pose.position.z = 1.0  # Eye level
        attention_pose.orientation.w = 1.0

        self.attention_pub.publish(attention_pose)

    def publish_speech_response(self, response):
        """Publish speech response"""
        response_msg = String()
        response_msg.data = response
        self.speech_response_pub.publish(response_msg)
        self.get_logger().info(f'Speech response: {response}')

    def publish_gesture_response(self, speech):
        """Publish appropriate gesture response"""
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

    def publish_positive_expression(self):
        """Publish positive facial expression"""
        expr_msg = String()
        expr_msg.data = 'happy'
        self.expression_pub.publish(expr_msg)

    def publish_attention_target(self, pose):
        """Publish attention target"""
        self.attention_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    social_system = SocialInteractionSystem()

    try:
        rclpy.spin(social_system)
    except KeyboardInterrupt:
        pass
    finally:
        social_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 4: Complex Task: Serving Drinks

### Scenario: Serve a Drink to a Person

This example demonstrates a complex multimodal task that combines navigation, object recognition, manipulation, and social interaction.

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
    NAVIGATING_TO_PERSON = "navigating_to_person"
    SEARCHING_FOR_DRINK = "searching_for_drink"
    FETCHING_DRINK = "fetching_drink"
    NAVIGATING_WITH_DRINK = "navigating_with_drink"
    SERVING = "serving"
    RETURNING = "returning"


class DrinkServingSystem(Node):
    def __init__(self):
        super().__init__('drink_serving_system')

        # Subscriptions
        self.command_sub = self.create_subscription(
            String, '/drink_serving_command', self.command_callback, 10)
        self.vision_sub = self.create_subscription(
            MarkerArray, '/detected_objects', self.vision_callback, 10)
        self.people_sub = self.create_subscription(
            MarkerArray, '/detected_people', self.people_callback, 10)
        self.current_pose_sub = self.create_subscription(
            Pose, '/current_pose', self.pose_callback, 10)

        # Publishers
        self.navigation_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.manipulation_pub = self.create_publisher(
            String, '/manipulation_command', 10)
        self.speech_pub = self.create_publisher(
            String, '/speech_output', 10)
        self.gesture_pub = self.create_publisher(
            String, '/gesture_command', 10)
        self.system_status_pub = self.create_publisher(
            String, '/system_status', 10)

        # System state
        self.current_pose = Pose()
        self.detected_people = {}
        self.detected_objects = {}
        self.current_task_state = TaskState.IDLE
        self.target_person = None
        self.target_drink = None
        self.serving_location = None
        self.return_location = None
        self.task_history = deque(maxlen=50)

        # Drink types and locations
        self.drink_types = ['water', 'juice', 'coffee', 'tea', 'soda']
        self.drink_locations = {
            'water': (5.0, 0.0),
            'juice': (5.0, 0.5),
            'coffee': (5.0, 1.0),
            'tea': (5.0, 1.5),
            'soda': (5.0, 2.0)
        }

        self.get_logger().info('Drink serving system initialized')

    def command_callback(self, msg):
        """Process drink serving commands"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received drink serving command: {command}')

        # Parse command to extract drink type and target person
        parsed_command = self.parse_serving_command(command)

        if parsed_command:
            # Start the serving task
            self.start_serving_task(parsed_command)
        else:
            # Ask for clarification
            self.ask_for_clarification()

    def parse_serving_command(self, command):
        """Parse natural language serving command"""
        command_lower = command.lower()

        # Extract drink type
        target_drink = None
        for drink in self.drink_types:
            if drink in command_lower:
                target_drink = drink
                break

        # If no specific drink mentioned, default to water
        if not target_drink:
            target_drink = 'water'

        # Extract target person (for now, use the first detected person)
        if self.detected_people:
            # Use the person closest to the robot
            closest_person = min(
                self.detected_people.items(),
                key=lambda x: self.calculate_distance_to_person(x[1])
            )
            target_person_id = closest_person[0]
        else:
            # No people detected, return None to ask for clarification
            return None

        return {
            'drink_type': target_drink,
            'target_person': target_person_id,
            'original_command': command
        }

    def calculate_distance_to_person(self, person_data):
        """Calculate distance from robot to person"""
        person_pos = person_data['position']
        robot_pos = (self.current_pose.position.x, self.current_pose.position.y)
        return np.sqrt((person_pos[0] - robot_pos[0])**2 + (person_pos[1] - robot_pos[1])**2)

    def start_serving_task(self, parsed_command):
        """Start the drink serving task"""
        self.target_drink = parsed_command['drink_type']
        self.target_person = parsed_command['target_person']

        # Store current location as return location
        self.return_location = (self.current_pose.position.x, self.current_pose.position.y)

        # Start with navigating to person
        self.current_task_state = TaskState.NAVIGATING_TO_PERSON
        self.get_logger().info(f'Starting to serve {self.target_drink} to person {self.target_person}')

        # Publish initial status
        status_msg = String()
        status_msg.data = f'Starting to serve {self.target_drink}'
        self.system_status_pub.publish(status_msg)

        # Begin task execution
        self.execute_current_task()

    def ask_for_clarification(self):
        """Ask user for clarification when command is unclear"""
        response_msg = String()
        response_msg.data = "I'd be happy to serve a drink, but I need to know what type of drink and who to serve it to. Could you please specify?"
        self.speech_pub.publish(response_msg)

    def execute_current_task(self):
        """Execute the current task based on state"""
        if self.current_task_state == TaskState.NAVIGATING_TO_PERSON:
            self.navigate_to_person()
        elif self.current_task_state == TaskState.SEARCHING_FOR_DRINK:
            self.search_for_drink()
        elif self.current_task_state == TaskState.FETCHING_DRINK:
            self.fetch_drink()
        elif self.current_task_state == TaskState.NAVIGATING_WITH_DRINK:
            self.navigate_with_drink()
        elif self.current_task_state == TaskState.SERVING:
            self.serve_drink()
        elif self.current_task_state == TaskState.RETURNING:
            self.return_to_base()

    def navigate_to_person(self):
        """Navigate to the target person"""
        if self.target_person in self.detected_people:
            person_data = self.detected_people[self.target_person]
            person_pos = person_data['position']

            # Calculate approach pose (1 meter away from person)
            approach_pose = Pose()
            approach_pose.position.x = person_pos[0] - 1.0
            approach_pose.position.y = person_pos[1]
            approach_pose.position.z = 0.0
            approach_pose.orientation.w = 1.0

            # Publish navigation goal
            self.navigation_pub.publish(approach_pose)

            self.get_logger().info(f'Navigating to person {self.target_person}')

            # After navigation, search for drink
            time.sleep(2)  # Simulate navigation time
            self.current_task_state = TaskState.SEARCHING_FOR_DRINK
            self.execute_current_task()
        else:
            # Person not detected, ask for help
            response_msg = String()
            response_msg.data = "I can't find the person to serve. Could you please come closer?"
            self.speech_pub.publish(response_msg)

    def search_for_drink(self):
        """Search for the target drink"""
        self.get_logger().info(f'Searching for {self.target_drink}')

        # Publish search response
        response_msg = String()
        response_msg.data = f"I'm looking for {self.target_drink}. Please wait a moment."
        self.speech_pub.publish(response_msg)

        # Navigate to drink location
        if self.target_drink in self.drink_locations:
            drink_x, drink_y = self.drink_locations[self.target_drink]

            drink_pose = Pose()
            drink_pose.position.x = drink_x
            drink_pose.position.y = drink_y
            drink_pose.position.z = 0.0
            drink_pose.orientation.w = 1.0

            # Publish navigation goal
            self.navigation_pub.publish(drink_pose)

            self.get_logger().info(f'Navigating to {self.target_drink} location')

            # After navigation, fetch drink
            time.sleep(2)  # Simulate navigation time
            self.current_task_state = TaskState.FETCHING_DRINK
            self.execute_current_task()
        else:
            # Drink location not found
            response_msg = String()
            response_msg.data = f"Sorry, I couldn't find {self.target_drink}."
            self.speech_pub.publish(response_msg)
            self.return_to_base()

    def fetch_drink(self):
        """Fetch the target drink"""
        self.get_logger().info(f'Fetching {self.target_drink}')

        # Publish manipulation command to fetch drink
        fetch_cmd = String()
        fetch_cmd.data = f'fetch_{self.target_drink}'
        self.manipulation_pub.publish(fetch_cmd)

        # Publish confirmation
        response_msg = String()
        response_msg.data = f"I have picked up the {self.target_drink}."
        self.speech_pub.publish(response_msg)

        # Move to serving state
        self.current_task_state = TaskState.NAVIGATING_WITH_DRINK
        self.execute_current_task()

    def navigate_with_drink(self):
        """Navigate back to the person with the drink"""
        if self.target_person in self.detected_people:
            person_data = self.detected_people[self.target_person]
            person_pos = person_data['position']

            # Calculate service pose (close to person)
            service_pose = Pose()
            service_pose.position.x = person_pos[0] - 0.5  # 0.5m away
            service_pose.position.y = person_pos[1]
            service_pose.position.z = 0.0
            service_pose.orientation.w = 1.0

            # Publish navigation goal
            self.navigation_pub.publish(service_pose)

            self.get_logger().info(f'Navigating to serve {self.target_drink} to person {self.target_person}')

            # After navigation, serve drink
            time.sleep(2)  # Simulate navigation time
            self.current_task_state = TaskState.SERVING
            self.execute_current_task()
        else:
            # Person not detected, return to base
            self.return_to_base()

    def serve_drink(self):
        """Serve the drink to the person"""
        self.get_logger().info(f'Serving {self.target_drink} to person {self.target_person}')

        # Publish manipulation command to serve drink
        serve_cmd = String()
        serve_cmd.data = f'serve_{self.target_drink}'
        self.manipulation_pub.publish(serve_cmd)

        # Publish service message
        response_msg = String()
        response_msg.data = f"Here is your {self.target_drink}. Enjoy!"
        self.speech_pub.publish(response_msg)

        # Publish serving gesture
        gesture_msg = String()
        gesture_msg.data = 'offer'
        self.gesture_pub.publish(gesture_msg)

        # Move to return state
        self.current_task_state = TaskState.RETURNING
        self.execute_current_task()

    def return_to_base(self):
        """Return to the starting location"""
        if self.return_location:
            return_pose = Pose()
            return_pose.position.x = self.return_location[0]
            return_pose.position.y = self.return_location[1]
            return_pose.position.z = 0.0
            return_pose.orientation.w = 1.0

            # Publish navigation goal
            self.navigation_pub.publish(return_pose)

            self.get_logger().info('Returning to base location')

            # After return, go back to idle
            time.sleep(2)  # Simulate navigation time
            self.current_task_state = TaskState.IDLE

            # Publish completion
            response_msg = String()
            response_msg.data = "Task completed. I'm back at my starting position."
            self.speech_pub.publish(response_msg)

            status_msg = String()
            status_msg.data = 'task_completed'
            self.system_status_pub.publish(status_msg)

    def vision_callback(self, msg):
        """Update detected objects from vision system"""
        # Process detected objects
        for marker in msg.markers:
            obj_id = f"obj_{marker.id}"
            self.detected_objects[obj_id] = {
                'class': marker.ns,
                'position': (marker.pose.position.x, marker.pose.position.y),
                'confidence': marker.color.a
            }

    def people_callback(self, msg):
        """Update detected people from vision system"""
        # Process detected people
        for marker in msg.markers:
            person_id = f"person_{marker.id}"
            self.detected_people[person_id] = {
                'position': (marker.pose.position.x, marker.pose.position.y),
                'last_seen': time.time(),
                'confidence': marker.color.a
            }

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg


def main(args=None):
    rclpy.init(args=args)
    serving_system = DrinkServingSystem()

    try:
        rclpy.spin(serving_system)
    except KeyboardInterrupt:
        pass
    finally:
        serving_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 5: Context-Aware Interaction System

### Scenario: Adaptive Interaction Based on Context

This example demonstrates a system that adapts its behavior based on the current context, including time of day, detected objects, and social situation.

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


class ContextAwareSystem(Node):
    def __init__(self):
        super().__init__('context_aware_system')

        # Subscriptions
        self.speech_sub = self.create_subscription(
            String, '/speech_input', self.speech_callback, 10)
        self.vision_sub = self.create_subscription(
            MarkerArray, '/detected_objects', self.vision_callback, 10)
        self.people_sub = self.create_subscription(
            MarkerArray, '/detected_people', self.people_callback, 10)
        self.time_sub = self.create_subscription(
            String, '/current_time', self.time_callback, 10)

        # Publishers
        self.response_pub = self.create_publisher(
            String, '/context_response', 10)
        self.behavior_pub = self.create_publisher(
            String, '/behavior_command', 10)
        self.context_pub = self.create_publisher(
            String, '/current_context', 10)

        # System state
        self.detected_objects = {}
        self.detected_people = {}
        self.current_time = datetime.now()
        self.context_history = deque(maxlen=100)
        self.person_profiles = defaultdict(dict)
        self.daily_routine = {
            'morning': ['greeting', 'weather_update', 'schedule_reminder'],
            'afternoon': ['check_in', 'task_assistance'],
            'evening': ['relaxation_suggestion', 'goodbye']
        }

        self.get_logger().info('Context-aware system initialized')

    def speech_callback(self, msg):
        """Process speech input in context"""
        speech = msg.data.lower().strip()
        self.get_logger().info(f'Received speech: {speech}')

        # Analyze current context
        current_context = self.analyze_context()

        # Generate context-aware response
        response = self.generate_contextual_response(speech, current_context)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Log interaction
        self.context_history.append({
            'type': 'speech_interaction',
            'input': speech,
            'response': response,
            'context': current_context,
            'timestamp': time.time()
        })

    def vision_callback(self, msg):
        """Update detected objects"""
        for marker in msg.markers:
            obj_id = f"obj_{marker.id}"
            self.detected_objects[obj_id] = {
                'class': marker.ns,
                'position': (marker.pose.position.x, marker.pose.position.y),
                'timestamp': time.time()
            }

    def people_callback(self, msg):
        """Update detected people and track"""
        for marker in msg.markers:
            person_id = f"person_{marker.id}"
            self.detected_people[person_id] = {
                'position': (marker.pose.position.x, marker.pose.position.y),
                'last_seen': time.time(),
                'greeting_given': self.detected_people.get(person_id, {}).get('greeting_given', False)
            }

    def time_callback(self, msg):
        """Update current time"""
        # In a real system, this would come from a time service
        # For simulation, we'll just update the current time
        self.current_time = datetime.now()

    def analyze_context(self):
        """Analyze current context based on multiple inputs"""
        context = {
            'time_of_day': self.get_time_of_day(),
            'detected_objects': list(self.detected_objects.keys()),
            'detected_people_count': len(self.detected_people),
            'social_context': self.get_social_context(),
            'environment_context': self.get_environment_context(),
            'routine_context': self.get_routine_context()
        }

        return context

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

    def get_social_context(self):
        """Analyze social context"""
        if len(self.detected_people) == 0:
            return 'alone'
        elif len(self.detected_people) == 1:
            return 'one_person'
        else:
            return 'multiple_people'

    def get_environment_context(self):
        """Analyze environment context based on detected objects"""
        object_classes = [obj['class'] for obj in self.detected_objects.values()]

        if 'kitchen' in object_classes or 'cup' in object_classes:
            return 'kitchen'
        elif 'bed' in object_classes or 'bedroom' in object_classes:
            return 'bedroom'
        elif 'sofa' in object_classes or 'tv' in object_classes:
            return 'living_room'
        elif 'desk' in object_classes or 'computer' in object_classes:
            return 'office'
        else:
            return 'unknown'

    def get_routine_context(self):
        """Get routine context based on time of day"""
        time_of_day = self.get_time_of_day()
        return self.daily_routine.get(time_of_day, [])

    def generate_contextual_response(self, speech, context):
        """Generate response based on context"""
        # Determine response based on context and input
        time_of_day = context['time_of_day']
        social_context = context['social_context']

        # Greeting responses based on time
        if any(word in speech for word in ['hello', 'hi', 'hey']):
            if time_of_day == 'morning':
                return "Good morning! How can I assist you today?"
            elif time_of_day == 'afternoon':
                return "Good afternoon! What can I help you with?"
            elif time_of_day == 'evening':
                return "Good evening! How are you doing?"
            else:
                return "Hello! How can I help you?"

        # Contextual responses
        if social_context == 'one_person':
            if time_of_day == 'morning' and 'coffee' in speech:
                return "I can help you get coffee. Would you like me to fetch it for you?"
            elif time_of_day == 'evening' and any(word in speech for word in ['tired', 'relax', 'rest']):
                return "You seem tired. Would you like me to dim the lights and play some relaxing music?"

        # Default response
        return f"I understand you said: '{speech}'. Based on the current context (time: {time_of_day}, people: {social_context}), how can I assist you?"

    def publish_context(self):
        """Publish current context"""
        context = self.analyze_context()
        context_msg = String()
        context_msg.data = str(context)
        self.context_pub.publish(context_msg)


def main(args=None):
    rclpy.init(args=args)
    context_system = ContextAwareSystem()

    try:
        rclpy.spin(context_system)
    except KeyboardInterrupt:
        pass
    finally:
        context_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Integration and Testing

### Complete Multimodal System Integration

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray
import json
import time
from threading import Thread


class MultimodalIntegrationSystem(Node):
    def __init__(self):
        super().__init__('multimodal_integration_system')

        # Initialize all subsystems
        self.object_fetching_system = ObjectFetchingSystem()
        self.navigation_system = NavigationCommandSystem()
        self.social_system = SocialInteractionSystem()
        self.serving_system = DrinkServingSystem()
        self.context_system = ContextAwareSystem()

        # Publishers for coordination
        self.coordinator_pub = self.create_publisher(
            String, '/system_coordinator', 10)
        self.status_pub = self.create_publisher(
            String, '/integration_status', 10)

        # System state
        self.systems_active = True
        self.performance_metrics = {}

        self.get_logger().info('Multimodal integration system initialized')

    def run_integration_test(self):
        """Run integration tests for all systems"""
        self.get_logger().info('Starting multimodal integration test')

        # Test 1: Object fetching with navigation
        self.test_object_fetching_with_navigation()

        # Test 2: Social interaction with context awareness
        self.test_social_context_integration()

        # Test 3: Complex task integration
        self.test_complex_task_integration()

        # Test 4: Performance under load
        self.test_performance_under_load()

        self.get_logger().info('Multimodal integration test completed')

    def test_object_fetching_with_navigation(self):
        """Test object fetching with navigation"""
        self.get_logger().info('Testing object fetching with navigation')

        # Simulate command to fetch an object
        command_msg = String()
        command_msg.data = "Please get me the red cup from the table"

        # This would be published to the object fetching system
        # For simulation, we'll just log the test
        self.get_logger().info('Object fetching test completed')

    def test_social_context_integration(self):
        """Test social interaction with context awareness"""
        self.get_logger().info('Testing social interaction with context awareness')

        # Simulate social interaction
        speech_msg = String()
        speech_msg.data = "Hello, how are you today?"

        # This would be published to both social and context systems
        self.get_logger().info('Social context test completed')

    def test_complex_task_integration(self):
        """Test complex task integration"""
        self.get_logger().info('Testing complex task integration')

        # Simulate a complex task: serve water to person in living room
        command_msg = String()
        command_msg.data = "Serve water to the person in the living room"

        # This would be processed by the serving system
        self.get_logger().info('Complex task integration test completed')

    def test_performance_under_load(self):
        """Test system performance under load"""
        self.get_logger().info('Testing performance under load')

        # Simulate multiple simultaneous inputs
        start_time = time.time()

        # Process multiple inputs quickly
        for i in range(10):
            # Simulate various inputs
            pass

        end_time = time.time()
        processing_time = end_time - start_time

        self.get_logger().info(f'Performance test completed in {processing_time:.3f}s')

    def monitor_system_performance(self):
        """Monitor performance of integrated systems"""
        while self.systems_active:
            # Collect performance metrics from all systems
            metrics = {
                'timestamp': time.time(),
                'object_fetching_status': 'active',
                'navigation_status': 'active',
                'social_status': 'active',
                'serving_status': 'active',
                'context_status': 'active'
            }

            self.performance_metrics = metrics

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps(metrics)
            self.status_pub.publish(status_msg)

            time.sleep(1.0)  # Monitor every second

    def start_monitoring(self):
        """Start system monitoring in a separate thread"""
        monitor_thread = Thread(target=self.monitor_system_performance)
        monitor_thread.daemon = True
        monitor_thread.start()


def main(args=None):
    rclpy.init(args=args)
    integration_system = MultimodalIntegrationSystem()

    # Start monitoring
    integration_system.start_monitoring()

    try:
        # Run integration tests
        integration_system.run_integration_test()

        # Continue running the integrated system
        rclpy.spin(integration_system)
    except KeyboardInterrupt:
        pass
    finally:
        integration_system.systems_active = False
        integration_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Performance Considerations

### Optimizing Multimodal Systems

```python
#!/usr/bin/env python3

import time
import threading
import queue
import numpy as np
from collections import defaultdict
import psutil


class MultimodalOptimizer:
    def __init__(self):
        self.component_load = defaultdict(float)
        self.response_times = defaultdict(list)
        self.resource_usage = {}
        self.adaptive_parameters = {}

        # Processing queues for different modalities
        self.vision_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        self.fusion_queue = queue.Queue(maxsize=5)

        # Performance thresholds
        self.max_response_time = 1.0  # seconds
        self.cpu_threshold = 80.0     # percent
        self.memory_threshold = 80.0  # percent

    def optimize_for_real_time(self):
        """Optimize system for real-time performance"""
        # Monitor resource usage
        self.resource_usage = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }

        # Adjust processing parameters based on load
        if self.resource_usage['cpu_percent'] > self.cpu_threshold:
            self.reduce_processing_quality()
        elif self.resource_usage['cpu_percent'] < 50:
            self.increase_processing_quality()

    def reduce_processing_quality(self):
        """Reduce processing quality to maintain performance"""
        # Reduce image resolution for vision processing
        self.adaptive_parameters['vision_resolution'] = max(0.3,
            self.adaptive_parameters.get('vision_resolution', 1.0) * 0.9)

        # Reduce audio processing complexity
        self.adaptive_parameters['audio_processing'] = 'simple'

        # Reduce fusion complexity
        self.adaptive_parameters['fusion_rate'] = max(1,
            int(self.adaptive_parameters.get('fusion_rate', 10) * 0.8))

    def increase_processing_quality(self):
        """Increase processing quality when resources allow"""
        # Increase image resolution
        self.adaptive_parameters['vision_resolution'] = min(1.0,
            self.adaptive_parameters.get('vision_resolution', 0.3) * 1.1)

        # Increase audio processing complexity
        self.adaptive_parameters['audio_processing'] = 'full'

        # Increase fusion rate
        self.adaptive_parameters['fusion_rate'] = min(30,
            int(self.adaptive_parameters.get('fusion_rate', 5) * 1.2))

    def prioritize_critical_tasks(self):
        """Prioritize critical tasks in multimodal processing"""
        # Define task priorities
        critical_tasks = ['safety_check', 'collision_avoidance', 'emergency_stop']
        high_priority_tasks = ['navigation', 'obstacle_detection']
        normal_priority_tasks = ['object_recognition', 'speech_recognition']

        # Implement priority-based scheduling
        pass

    def load_balance_processing(self):
        """Distribute processing load across available resources"""
        # Monitor processing times for each component
        avg_times = {}
        for component, times in self.response_times.items():
            if times:
                avg_times[component] = sum(times) / len(times)

        # Adjust workload distribution based on performance
        pass

    def adaptive_fusion_strategy(self):
        """Adapt fusion strategy based on input reliability"""
        # If vision is unreliable, weight audio more heavily
        # If audio is noisy, weight vision more heavily
        # If both are unreliable, use temporal consistency
        pass


class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}

    def start_measurement(self, component_name):
        """Start timing for a component"""
        self.start_times[component_name] = time.time()

    def end_measurement(self, component_name):
        """End timing for a component and record metric"""
        if component_name in self.start_times:
            elapsed = time.time() - self.start_times[component_name]
            self.metrics[component_name].append(elapsed)

            # Keep only recent measurements (last 100)
            if len(self.metrics[component_name]) > 100:
                self.metrics[component_name] = self.metrics[component_name][-100:]

    def get_average_response_time(self, component_name):
        """Get average response time for a component"""
        times = self.metrics[component_name]
        if times:
            return sum(times) / len(times)
        return 0.0

    def get_throughput(self, component_name):
        """Get processing throughput for a component"""
        times = self.metrics[component_name]
        if len(times) > 1:
            total_time = times[-1] - times[0]
            return len(times) / total_time if total_time > 0 else 0.0
        return 0.0
```

## Troubleshooting Common Issues

### Common Multimodal Integration Problems and Solutions

#### Issue 1: Temporal Misalignment
**Symptoms**: Actions don't correspond to the right sensory input, inconsistent behavior
**Causes**:
- Different processing rates for modalities
- Network delays
- Buffer mismatches
**Solutions**:
- Implement proper timestamp synchronization
- Use temporal buffers to align inputs
- Implement interpolation for different rates
- Add latency compensation mechanisms

#### Issue 2: Conflicting Information from Modalities
**Symptoms**: System makes contradictory decisions based on different sensors
**Causes**:
- Sensor calibration issues
- Different confidence levels
- Environmental factors affecting sensors differently
**Solutions**:
- Implement confidence-based weighting
- Use sensor fusion algorithms
- Add consistency checks
- Implement fallback strategies

#### Issue 3: Resource Contention
**Symptoms**: System becomes slow or unresponsive under multimodal load
**Causes**:
- CPU/GPU resource competition
- Memory allocation conflicts
- I/O bottlenecks
**Solutions**:
- Implement resource scheduling
- Use asynchronous processing
- Add resource monitoring
- Implement adaptive quality adjustment

#### Issue 4: Integration Complexity
**Symptoms**: Difficult to debug, maintain, or extend the system
**Causes**:
- Tight coupling between components
- Complex interdependencies
- Lack of clear interfaces
**Solutions**:
- Use modular architecture
- Implement clear component interfaces
- Add comprehensive logging
- Create integration tests

## Best Practices for Multimodal Systems

### 1. Modular Design
- Keep components loosely coupled
- Use clear, well-defined interfaces
- Implement component health checks
- Design for independent testing

### 2. Robust Communication
- Use reliable messaging patterns
- Implement timeout mechanisms
- Add error recovery
- Monitor communication quality

### 3. Adaptive Processing
- Adjust quality based on system load
- Implement graceful degradation
- Use dynamic resource allocation
- Monitor and log performance metrics

### 4. Safety and Reliability
- Implement safety constraints
- Add validation checks
- Use redundancy where critical
- Plan for failure scenarios

## Key Takeaways

- Multimodal systems integrate vision, language, and action for natural interaction
- Context-aware behavior adapts to environment and situation
- Performance optimization is critical for real-time operation
- Proper testing ensures reliable multimodal integration
- Troubleshooting requires understanding of all modalities
- Best practices include modularity, safety, and adaptability

In the next section, we'll explore voice command integration and audio processing for humanoid robot applications.