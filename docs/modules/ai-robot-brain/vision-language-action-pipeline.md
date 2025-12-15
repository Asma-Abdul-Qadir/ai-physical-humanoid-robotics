---
sidebar_position: 2
---

# Vision-Language-Action Pipeline: Integrating Perception, Language, and Control

Welcome to the Vision-Language-Action (VLA) pipeline module, which focuses on integrating computer vision, natural language processing, and robotic action execution for humanoid robots. This chapter covers the implementation of systems that can perceive their environment, understand natural language commands, and execute appropriate actions.

## Learning Objectives

By the end of this section, you will be able to:
- Understand the architecture of Vision-Language-Action systems
- Implement perception pipelines using computer vision techniques
- Integrate speech recognition with Whisper for voice command processing
- Connect language models (LLMs) for natural language understanding
- Create action execution systems that respond to perceptual and linguistic inputs
- Design multimodal fusion mechanisms for decision making
- Evaluate VLA system performance and robustness
- Troubleshoot common issues in multimodal robotics systems

## Introduction to Vision-Language-Action Systems

Vision-Language-Action (VLA) systems represent the next generation of intelligent robotics, where robots can perceive their environment, understand natural language commands, and execute appropriate actions. For humanoid robots, this creates more natural and intuitive human-robot interaction.

### Key Components of VLA Systems

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │   Language      │    │   Action        │
│   (Vision)      │───▶│   Understanding │───▶│   Execution     │
│   • Cameras     │    │   • Whisper     │    │   • Navigation  │
│   • LIDAR       │    │   • LLMs        │    │   • Manipulation│
│   • Depth       │    │   • NLU         │    │   • Speech      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌───────────────────────────────┼─────────────────────────────────┐
│                         Multimodal Fusion                       │
│                     (Decision Making Layer)                     │
└─────────────────────────────────────────────────────────────────┘
```

### VLA Architecture for Humanoid Robots

Humanoid robots present unique challenges for VLA systems:
- **Embodied Interaction**: Actions must consider physical constraints
- **Real-time Processing**: Perception and action must happen in real-time
- **Safety Critical**: Actions must be safe for humans and environment
- **Context Awareness**: Understanding depends on physical context

## Computer Vision Pipeline

### Camera Systems and Image Acquisition

Humanoid robots typically employ multiple cameras for comprehensive perception:

```python
#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


class VisionPipeline(Node):
    def __init__(self):
        super().__init__('vision_pipeline')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera subscriptions
        self.front_camera_sub = self.create_subscription(
            Image, '/camera/front/image_raw', self.front_camera_callback, 10)
        self.head_camera_sub = self.create_subscription(
            Image, '/camera/head/image_raw', self.head_camera_callback, 10)

        # Publishers
        self.object_detection_pub = self.create_publisher(
            MarkerArray, '/detected_objects', 10)
        self.feature_map_pub = self.create_publisher(
            Image, '/feature_maps', 10)

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Vision processing parameters
        self.detection_threshold = 0.5
        self.class_names = [
            'person', 'chair', 'table', 'door', 'obstacle',
            'robot', 'cup', 'phone', 'book', 'plant'
        ]

        # Processing pipeline components
        self.object_detector = self.initialize_object_detector()
        self.feature_extractor = self.initialize_feature_extractor()
        self.scene_analyzer = SceneAnalyzer()

        self.get_logger().info('Vision pipeline initialized')

    def initialize_object_detector(self):
        """Initialize object detection model"""
        # In a real implementation, this would load a YOLO, SSD, or similar model
        # For this example, we'll simulate detection
        return ObjectDetector()

    def initialize_feature_extractor(self):
        """Initialize feature extraction model"""
        # This could be a CNN, ViT, or other feature extractor
        return FeatureExtractor()

    def front_camera_callback(self, msg):
        """Process front-facing camera feed"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Run object detection
            detections = self.object_detector.detect(cv_image, self.detection_threshold)

            # Extract features
            features = self.feature_extractor.extract(cv_image)

            # Analyze scene
            scene_description = self.scene_analyzer.analyze(cv_image, detections)

            # Publish results
            self.publish_detections(detections, msg.header)
            self.publish_features(features, msg)

            # Log scene description
            self.get_logger().info(f'Scene: {scene_description}')

        except Exception as e:
            self.get_logger().error(f'Front camera processing error: {e}')

    def head_camera_callback(self, msg):
        """Process head-mounted camera feed for face detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Face detection for social interaction
            faces = self.detect_faces(cv_image)

            # Person identification
            identities = self.identify_people(faces, cv_image)

            # Update interaction context
            self.update_interaction_context(identities, msg.header)

        except Exception as e:
            self.get_logger().error(f'Head camera processing error: {e}')

    def detect_faces(self, image):
        """Detect faces in image for social interaction"""
        # Use OpenCV Haar cascades or DNN face detector
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return faces

    def identify_people(self, faces, image):
        """Identify people in the scene"""
        identities = []

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]

            # In a real system, this would use face recognition
            # For simulation, assign temporary IDs
            identity = {
                'bbox': (x, y, w, h),
                'confidence': 0.9,
                'person_id': f'person_{len(identities)}'
            }
            identities.append(identity)

        return identities

    def publish_detections(self, detections, header):
        """Publish detected objects as visualization markers"""
        marker_array = MarkerArray()

        for i, detection in enumerate(detections):
            marker = Marker()
            marker.header = header
            marker.ns = "objects"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position in camera frame (would need to transform to world)
            marker.pose.position.x = detection['center'][0]
            marker.pose.position.y = detection['center'][1]
            marker.pose.position.z = 0.0  # Simplified

            marker.pose.orientation.w = 1.0
            marker.scale.x = detection['bbox'][2] / 100.0  # Scale down
            marker.scale.y = detection['bbox'][3] / 100.0
            marker.scale.z = 0.5  # Height of marker

            # Color based on class
            class_colors = {
                'person': (0, 1, 0),      # Green
                'chair': (1, 0, 0),       # Red
                'table': (0, 0, 1),       # Blue
                'door': (1, 1, 0),        # Yellow
                'obstacle': (1, 0, 1)     # Magenta
            }

            color = class_colors.get(detection['class'], (0.5, 0.5, 0.5))
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.7

            marker.text = f"{detection['class']}: {detection['confidence']:.2f}"

            marker_array.markers.append(marker)

        self.object_detection_pub.publish(marker_array)

    def publish_features(self, features, original_msg):
        """Publish feature maps for downstream processing"""
        # Convert features back to image format for visualization
        if features is not None:
            # This is a simplified representation
            feature_image = (features * 255).astype(np.uint8)

            # Publish as grayscale image
            feature_msg = self.bridge.cv2_to_imgmsg(feature_image, "mono8")
            feature_msg.header = original_msg.header

            self.feature_map_pub.publish(feature_msg)


class ObjectDetector:
    def __init__(self):
        # In a real implementation, load a pre-trained model
        # e.g., YOLOv8, SSD MobileNet, etc.
        pass

    def detect(self, image, threshold=0.5):
        """Detect objects in image"""
        # Simulate detection results
        h, w = image.shape[:2]

        # Generate some mock detections
        detections = []

        # Simulate detecting a few objects
        if np.random.random() > 0.3:  # 70% chance of detecting something
            for _ in range(np.random.randint(1, 4)):  # 1-3 objects
                if np.random.random() > 0.5:  # 50% chance of person
                    class_name = 'person'
                else:
                    class_name = np.random.choice(['chair', 'table', 'door', 'obstacle'])

                # Random bounding box
                x = np.random.randint(0, w//2)
                y = np.random.randint(0, h//2)
                width = np.random.randint(w//8, w//4)
                height = np.random.randint(h//8, h//4)

                detection = {
                    'class': class_name,
                    'bbox': (x, y, width, height),
                    'center': (x + width//2, y + height//2),
                    'confidence': np.random.uniform(threshold, 1.0)
                }

                detections.append(detection)

        return detections


class FeatureExtractor:
    def __init__(self):
        # In a real implementation, this would be a CNN or transformer
        pass

    def extract(self, image):
        """Extract features from image"""
        # Simulate feature extraction
        # In reality, this would run through a neural network
        h, w = image.shape[:2]

        # Return a simplified feature representation
        # Shape would typically be (channels, height, width) or (features,)
        return np.random.rand(h//16, w//16, 256).astype(np.float32)


class SceneAnalyzer:
    def __init__(self):
        self.context = {}

    def analyze(self, image, detections):
        """Analyze scene and generate description"""
        if not detections:
            return "Empty scene detected"

        # Count objects by class
        object_counts = {}
        for det in detections:
            cls = det['class']
            object_counts[cls] = object_counts.get(cls, 0) + 1

        # Generate scene description
        description_parts = []
        for obj_class, count in object_counts.items():
            if count == 1:
                description_parts.append(f"1 {obj_class}")
            else:
                description_parts.append(f"{count} {obj_class}s")

        scene_description = f"Scene contains: {', '.join(description_parts)}"
        return scene_description


def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionPipeline()

    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Depth Perception and 3D Understanding

For humanoid robots, depth perception is crucial for safe navigation and interaction:

```python
#!/usr/bin/env python3

import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray
import rclpy
from rclpy.node import Node


class DepthPerception(Node):
    def __init__(self):
        super().__init__('depth_perception')

        # Point cloud subscription
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)

        # Publishers
        self.obstacle_pub = self.create_publisher(
            MarkerArray, '/depth_obstacles', 10)
        self.surface_pub = self.create_publisher(
            MarkerArray, '/surfaces', 10)

        # Processing parameters
        self.ground_plane_threshold = 0.05
        self.obstacle_height_threshold = 0.1
        self.min_cluster_size = 10

        # Ground plane model (will be updated from TF or calibration)
        self.ground_plane = [0, 0, 1, 0]  # ax + by + cz + d = 0

        self.get_logger().info('Depth perception initialized')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        try:
            # Convert PointCloud2 to numpy array
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])

            if not points_list:
                return

            points = np.array(points_list)

            # Filter out ground plane
            filtered_points, ground_points = self.separate_ground(points)

            # Detect obstacles
            obstacles = self.detect_obstacles(filtered_points)

            # Detect surfaces (tables, walls, etc.)
            surfaces = self.detect_surfaces(filtered_points)

            # Publish results
            self.publish_obstacles(obstacles, msg.header)
            self.publish_surfaces(surfaces, msg.header)

        except Exception as e:
            self.get_logger().error(f'Point cloud processing error: {e}')

    def separate_ground(self, points):
        """Separate ground points from obstacles"""
        # Simple ground plane detection using RANSAC-like approach
        # In practice, use PCL or Open3D for robust plane fitting

        # Assume ground is roughly horizontal (z ~ 0)
        ground_mask = np.abs(points[:, 2]) < self.ground_plane_threshold
        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]

        return obstacle_points, ground_points

    def detect_obstacles(self, points):
        """Detect obstacles above ground plane"""
        if len(points) == 0:
            return []

        # Cluster points that are close together
        clusters = self.cluster_points(points)

        obstacles = []
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                # Calculate centroid and bounding box
                centroid = np.mean(cluster, axis=0)
                min_pt = np.min(cluster, axis=0)
                max_pt = np.max(cluster, axis=0)

                # Only consider objects above certain height
                if centroid[2] > self.obstacle_height_threshold:
                    obstacle = {
                        'centroid': centroid,
                        'bbox_min': min_pt,
                        'bbox_max': max_pt,
                        'points': cluster,
                        'size': len(cluster)
                    }
                    obstacles.append(obstacle)

        return obstacles

    def cluster_points(self, points, distance_threshold=0.1):
        """Simple clustering of 3D points"""
        if len(points) == 0:
            return []

        clusters = []
        visited = set()

        for i, pt in enumerate(points):
            if i in visited:
                continue

            cluster = [pt]
            to_visit = [i]
            visited.add(i)

            while to_visit:
                current_idx = to_visit.pop()
                current_pt = points[current_idx]

                # Find neighbors within threshold
                for j, other_pt in enumerate(points):
                    if j in visited:
                        continue

                    dist = np.linalg.norm(current_pt - other_pt)
                    if dist < distance_threshold:
                        cluster.append(other_pt)
                        to_visit.append(j)
                        visited.add(j)

            if len(cluster) > 0:
                clusters.append(np.array(cluster))

        return clusters

    def detect_surfaces(self, points):
        """Detect planar surfaces (tables, walls, etc.)"""
        if len(points) < 10:
            return []

        # In a real implementation, use RANSAC for plane detection
        # For simulation, detect horizontal and vertical surfaces
        surfaces = []

        # Look for horizontal surfaces (tables, floors)
        z_values = points[:, 2]
        unique_z, counts = np.unique(z_values, return_counts=True)

        for z, count in zip(unique_z, counts):
            if count > 20:  # Significant number of points at same height
                surface = {
                    'type': 'horizontal',
                    'z_level': z,
                    'support_area': count
                }
                surfaces.append(surface)

        return surfaces

    def publish_obstacles(self, obstacles, header):
        """Publish obstacle markers"""
        marker_array = MarkerArray()

        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header = header
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position at centroid
            marker.pose.position.x = obstacle['centroid'][0]
            marker.pose.position.y = obstacle['centroid'][1]
            marker.pose.position.z = obstacle['centroid'][2]

            marker.pose.orientation.w = 1.0

            # Size based on bounding box
            size_x = obstacle['bbox_max'][0] - obstacle['bbox_min'][0]
            size_y = obstacle['bbox_max'][1] - obstacle['bbox_min'][1]
            size_z = obstacle['bbox_max'][2] - obstacle['bbox_min'][2]

            marker.scale.x = max(size_x, 0.1)
            marker.scale.y = max(size_y, 0.1)
            marker.scale.z = max(size_z, 0.1)

            # Red color for obstacles
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.obstacle_pub.publish(marker_array)

    def publish_surfaces(self, surfaces, header):
        """Publish surface markers"""
        marker_array = MarkerArray()

        for i, surface in enumerate(surfaces):
            marker = Marker()
            marker.header = header
            marker.ns = "surfaces"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position at surface level
            marker.pose.position.x = 0.0  # Will be updated based on actual detection
            marker.pose.position.y = 0.0
            marker.pose.position.z = surface['z_level']

            marker.pose.orientation.w = 1.0

            # Large flat surface
            marker.scale.x = 2.0
            marker.scale.y = 2.0
            marker.scale.z = 0.01  # Thin surface

            # Blue color for surfaces
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.5

            marker_array.markers.append(marker)

        self.surface_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    depth_node = DepthPerception()

    try:
        rclpy.spin(depth_node)
    except KeyboardInterrupt:
        pass
    finally:
        depth_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Natural Language Processing Pipeline

### Speech Recognition with Whisper

Whisper is a powerful speech recognition model that can be used for voice command processing:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import numpy as np
import torch
import whisper
from transformers import pipeline
import threading
import queue
import wave
import io


class WhisperSpeechProcessor(Node):
    def __init__(self):
        super().__init__('whisper_speech_processor')

        # Audio input subscription
        self.audio_sub = self.create_subscription(
            AudioData, '/audio_input', self.audio_callback, 10)

        # Text output publisher
        self.transcript_pub = self.create_publisher(
            String, '/speech_transcript', 10)
        self.command_pub = self.create_publisher(
            String, '/voice_command', 10)

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        try:
            # Use 'tiny' or 'base' for faster inference, 'small' or 'medium' for better accuracy
            self.model = whisper.load_model("base")
            self.get_logger().info('Whisper model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.model = None

        # Audio processing parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024  # Process audio in chunks
        self.silence_threshold = 0.01  # Threshold for silence detection
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds

        # Audio buffering
        self.audio_buffer = []
        self.is_recording = False
        self.recording_started = 0

        # Processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Whisper speech processor initialized')

    def audio_callback(self, msg):
        """Process incoming audio data"""
        if self.model is None:
            return

        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
            audio_data /= 32768.0  # Normalize to [-1, 1]

            # Check if audio is loud enough (not silence)
            audio_energy = np.mean(np.abs(audio_data))

            if audio_energy > self.silence_threshold:
                if not self.is_recording:
                    # Start recording
                    self.is_recording = True
                    self.recording_started = self.get_clock().now().nanoseconds
                    self.audio_buffer = []

                # Add to buffer
                self.audio_buffer.extend(audio_data)
            else:
                # Check if we were recording and should process
                if self.is_recording:
                    current_time = self.get_clock().now().nanoseconds
                    duration = (current_time - self.recording_started) / 1e9

                    if duration >= self.min_speech_duration and len(self.audio_buffer) > 0:
                        # Queue audio for processing
                        self.audio_queue.put(np.array(self.audio_buffer))

                    self.is_recording = False

        except Exception as e:
            self.get_logger().error(f'Audio processing error: {e}')

    def process_audio_queue(self):
        """Process audio in separate thread to avoid blocking"""
        while rclpy.ok():
            try:
                # Get audio from queue (non-blocking with timeout)
                audio_data = self.audio_queue.get(timeout=1.0)

                if len(audio_data) > 0:
                    # Process with Whisper
                    transcript = self.transcribe_audio(audio_data)

                    if transcript and transcript.strip():
                        # Publish transcript
                        transcript_msg = String()
                        transcript_msg.data = transcript
                        self.transcript_pub.publish(transcript_msg)

                        # Process as command
                        self.process_command(transcript)

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Queue processing error: {e}')

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        try:
            # Ensure audio is at correct sample rate
            # Whisper expects 16kHz, so if input is different, resample
            # For simplicity, assuming input is already 16kHz

            # Pad or trim audio to minimum length if needed
            if len(audio_data) < 16000:  # Less than 1 second
                pad_length = 16000 - len(audio_data)
                audio_data = np.pad(audio_data, (0, pad_length), mode='constant')

            # Run transcription
            result = self.model.transcribe(audio_data)
            return result['text'].strip()

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
            return ""

    def process_command(self, transcript):
        """Process the transcribed text as a command"""
        if not transcript:
            return

        self.get_logger().info(f'Recognized: "{transcript}"')

        # Simple command parsing (in real system, use NLP/NLU)
        command = self.parse_command(transcript)

        if command:
            # Publish command
            cmd_msg = String()
            cmd_msg.data = command
            self.command_pub.publish(cmd_msg)
            self.get_logger().info(f'Processed command: {command}')

    def parse_command(self, text):
        """Parse recognized text into structured command"""
        text = text.lower().strip()

        # Define command patterns
        command_patterns = {
            # Navigation commands
            'move forward': ['go forward', 'move ahead', 'walk forward', 'go straight'],
            'move backward': ['go back', 'move back', 'walk back'],
            'turn left': ['turn left', 'rotate left', 'go left'],
            'turn right': ['turn right', 'rotate right', 'go right'],
            'stop': ['stop', 'halt', 'freeze', 'pause'],
            'come here': ['come here', 'come to me', 'come over', 'approach'],

            # Interaction commands
            'wave': ['wave', 'say hi', 'hello', 'greet'],
            'follow me': ['follow me', 'follow', 'come with me'],
            'find person': ['find person', 'locate person', 'where is person'],
            'bring object': ['bring', 'fetch', 'get', 'pick up'],

            # Information commands
            'what do you see': ['what do you see', 'describe scene', 'what is around'],
            'who is there': ['who is there', 'who is here', 'anyone there']
        }

        # Match text to command
        for canonical_cmd, patterns in command_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return canonical_cmd

        # If no exact match, return the raw text
        return text if len(text) > 3 else ""  # Only return if meaningful length


def main(args=None):
    rclpy.init(args=args)
    speech_node = WhisperSpeechProcessor()

    try:
        rclpy.spin(speech_node)
    except KeyboardInterrupt:
        pass
    finally:
        speech_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Language Understanding with Large Language Models

Integrating LLMs for more sophisticated language understanding:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from action_msgs.msg import GoalStatus
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


class LanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('language_understanding')

        # Subscriptions
        self.command_sub = self.create_subscription(
            String, '/voice_command', self.command_callback, 10)
        self.text_command_sub = self.create_subscription(
            String, '/text_command', self.text_command_callback, 10)

        # Publishers
        self.action_plan_pub = self.create_publisher(
            String, '/action_plan', 10)
        self.navigation_goal_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.response_pub = self.create_publisher(
            String, '/robot_response', 10)

        # Initialize language model
        self.get_logger().info('Initializing language understanding system...')

        try:
            # Use a local model for privacy and offline capability
            model_name = "microsoft/DialoGPT-medium"  # Or use "gpt2" for general purpose
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.get_logger().info('Language model initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize language model: {e}')
            self.tokenizer = None
            self.model = None

        # For online API (uncomment if you have API access)
        # openai.api_key = "your-api-key-here"

        # Context and state
        self.conversation_history = []
        self.robot_state = {
            'location': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'battery': 100.0,
            'current_task': None,
            'detected_objects': [],
            'people_count': 0
        }

        # Command parsers
        self.command_parsers = [
            self.parse_navigation_command,
            self.parse_manipulation_command,
            self.parse_information_command,
            self.parse_social_command
        ]

        self.get_logger().info('Language understanding system initialized')

    def command_callback(self, msg):
        """Process voice commands"""
        self.process_command(msg.data, source='voice')

    def text_command_callback(self, msg):
        """Process text commands"""
        self.process_command(msg.data, source='text')

    def process_command(self, command_text, source='voice'):
        """Process natural language command"""
        if not command_text:
            return

        self.get_logger().info(f'Received {source} command: "{command_text}"')

        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': command_text})
        if len(self.conversation_history) > 10:  # Keep last 10 exchanges
            self.conversation_history = self.conversation_history[-10:]

        # Parse command using multiple parsers
        action_plan = None
        for parser in self.command_parsers:
            action_plan = parser(command_text)
            if action_plan:
                break

        if action_plan:
            # Publish action plan
            plan_msg = String()
            plan_msg.data = json.dumps(action_plan)
            self.action_plan_pub.publish(plan_msg)

            # Execute specific actions based on plan type
            self.execute_action_plan(action_plan, command_text)
        else:
            # If no specific action parsed, try general understanding
            response = self.general_understanding(command_text)
            if response:
                self.publish_response(response)

    def parse_navigation_command(self, text):
        """Parse navigation-related commands"""
        text_lower = text.lower()

        # Navigation keywords
        nav_keywords = ['go to', 'move to', 'walk to', 'navigate to', 'go', 'move', 'walk', 'navigate',
                       'front', 'back', 'backward', 'forward', 'ahead', 'left', 'right',
                       'turn', 'rotate', 'stop', 'halt', 'approach', 'come']

        # Check if this is a navigation command
        if any(keyword in text_lower for keyword in nav_keywords):
            # Extract destination/location if specified
            destinations = ['kitchen', 'living room', 'bedroom', 'office', 'dining room', 'bathroom', 'hallway']

            for dest in destinations:
                if dest in text_lower:
                    return {
                        'type': 'navigation',
                        'destination': dest,
                        'command': text
                    }

            # Direction-based movement
            directions = {
                'forward': {'x': 1.0, 'y': 0.0, 'theta': 0.0},
                'backward': {'x': -1.0, 'y': 0.0, 'theta': 0.0},
                'left': {'x': 0.0, 'y': 1.0, 'theta': 1.57},  # 90 degrees
                'right': {'x': 0.0, 'y': -1.0, 'theta': -1.57},  # -90 degrees
                'slightly left': {'x': 0.5, 'y': 0.2, 'theta': 0.3},
                'slightly right': {'x': 0.5, 'y': -0.2, 'theta': -0.3}
            }

            for direction, pose in directions.items():
                if direction in text_lower:
                    return {
                        'type': 'navigation',
                        'relative_pose': pose,
                        'command': text
                    }

            # Come here command
            if any(phrase in text_lower for phrase in ['come here', 'come to me', 'come over']):
                return {
                    'type': 'navigation',
                    'relative_pose': {'x': -2.0, 'y': 0.0, 'theta': 0.0},  # Move to user location
                    'command': text
                }

        return None

    def parse_manipulation_command(self, text):
        """Parse manipulation-related commands"""
        text_lower = text.lower()

        # Manipulation keywords
        manipulation_keywords = ['pick', 'grab', 'take', 'lift', 'carry', 'bring', 'get', 'fetch',
                                'hand', 'give', 'put', 'place', 'drop', 'release']

        if any(keyword in text_lower for keyword in manipulation_keywords):
            # Extract object to manipulate
            objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'ball', 'toy', 'plate', 'fork']
            target_object = None

            for obj in objects:
                if obj in text_lower:
                    target_object = obj
                    break

            if target_object:
                return {
                    'type': 'manipulation',
                    'action': 'fetch',
                    'target_object': target_object,
                    'command': text
                }

            # General manipulation command
            return {
                'type': 'manipulation',
                'action': 'unknown',
                'command': text
            }

        return None

    def parse_information_command(self, text):
        """Parse information-seeking commands"""
        text_lower = text.lower()

        # Information keywords
        info_keywords = ['what', 'where', 'who', 'how', 'describe', 'tell', 'explain', 'show']

        if any(keyword in text_lower for keyword in info_keywords):
            if 'see' in text_lower or 'around' in text_lower or 'here' in text_lower:
                return {
                    'type': 'information',
                    'request': 'environment_description',
                    'command': text
                }

            if 'person' in text_lower or 'people' in text_lower or 'human' in text_lower:
                return {
                    'type': 'information',
                    'request': 'people_count',
                    'command': text
                }

            if 'battery' in text_lower or 'power' in text_lower or 'charge' in text_lower:
                return {
                    'type': 'information',
                    'request': 'battery_status',
                    'command': text
                }

            if 'location' in text_lower or 'where' in text_lower and 'am' in text_lower:
                return {
                    'type': 'information',
                    'request': 'current_location',
                    'command': text
                }

        return None

    def parse_social_command(self, text):
        """Parse social interaction commands"""
        text_lower = text.lower()

        social_actions = {
            'wave': ['wave', 'waving', 'hello', 'hi', 'greet'],
            'follow': ['follow', 'follow me', 'come with'],
            'wait': ['wait', 'stay', 'hold on', 'stop'],
            'dance': ['dance', 'dancing', 'party'],
            'pose': ['pose', 'picture', 'photo', 'smile']
        }

        for action, keywords in social_actions.items():
            if any(keyword in text_lower for keyword in keywords):
                return {
                    'type': 'social',
                    'action': action,
                    'command': text
                }

        return None

    def execute_action_plan(self, action_plan, original_command):
        """Execute the parsed action plan"""
        action_type = action_plan['type']

        if action_type == 'navigation':
            self.execute_navigation(action_plan)
        elif action_type == 'manipulation':
            self.execute_manipulation(action_plan)
        elif action_type == 'information':
            self.execute_information_request(action_plan)
        elif action_type == 'social':
            self.execute_social_action(action_plan)
        else:
            # Generate response using language model
            response = self.generate_response(original_command)
            self.publish_response(response)

    def execute_navigation(self, plan):
        """Execute navigation plan"""
        if 'destination' in plan:
            # In a real system, this would look up the location in a map
            # For simulation, we'll just acknowledge
            response = f"Okay, I will navigate to the {plan['destination']}."
            self.publish_response(response)

            # Publish navigation goal (simplified)
            goal = Pose()
            goal.position.x = 1.0  # Example position
            goal.position.y = 1.0
            goal.orientation.z = 0.0
            goal.orientation.w = 1.0

            self.navigation_goal_pub.publish(goal)

        elif 'relative_pose' in plan:
            pose = plan['relative_pose']
            response = f"Moving {plan.get('command', 'as requested')}."
            self.publish_response(response)

    def execute_manipulation(self, plan):
        """Execute manipulation plan"""
        if plan['action'] == 'fetch' and 'target_object' in plan:
            response = f"I will fetch the {plan['target_object']} for you."
        else:
            response = "I understand you want me to manipulate something."

        self.publish_response(response)

    def execute_information_request(self, plan):
        """Execute information request"""
        request = plan['request']

        if request == 'environment_description':
            # This would integrate with perception system
            response = "I see a table with a cup on it, and a chair nearby."
        elif request == 'people_count':
            response = f"I can see {self.robot_state['people_count']} people in the area."
        elif request == 'battery_status':
            response = f"My battery level is at {self.robot_state['battery']:.1f}%."
        elif request == 'current_location':
            loc = self.robot_state['location']
            response = f"I am currently at position ({loc['x']:.2f}, {loc['y']:.2f})."
        else:
            response = "I can help you with that information."

        self.publish_response(response)

    def execute_social_action(self, plan):
        """Execute social action"""
        action = plan['action']

        responses = {
            'wave': "Hello! I'm waving.",
            'follow': "Okay, I'll follow you.",
            'wait': "I'm waiting here.",
            'dance': "I wish I could dance!",
            'pose': "I'm ready for the photo!"
        }

        response = responses.get(action, f"I'll try to {action}.")
        self.publish_response(response)

    def general_understanding(self, text):
        """Handle commands that don't fit specific categories"""
        # Use the language model for general understanding
        return self.generate_response(text)

    def generate_response(self, user_input):
        """Generate contextual response using language model"""
        if self.model is None or self.tokenizer is None:
            return "I understand you said: " + user_input

        try:
            # Prepare context
            context = f"The user said: '{user_input}'. The robot's state is: {self.robot_state}. Respond appropriately."

            # Tokenize input
            inputs = self.tokenizer.encode(context + self.tokenizer.eos_token, return_tensors='pt')

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part (after the input)
            if context in response:
                response = response.split(context)[-1].strip()

            return response if response else "I understand your request."

        except Exception as e:
            self.get_logger().error(f'Language generation error: {e}')
            return f"I understand you said: {user_input}"

    def publish_response(self, response):
        """Publish robot's verbal response"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)
        self.get_logger().info(f'Robot response: "{response}"')


def main(args=None):
    rclpy.init(args=args)
    lang_node = LanguageUnderstandingNode()

    try:
        rclpy.spin(lang_node)
    except KeyboardInterrupt:
        pass
    finally:
        lang_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Action Execution System

### Integrating Perception and Language for Action Planning

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import json
import numpy as np


class ActionExecutionNode(Node):
    def __init__(self):
        super().__init__('action_execution')

        # Subscriptions
        self.action_plan_sub = self.create_subscription(
            String, '/action_plan', self.action_plan_callback, 10)
        self.vision_sub = self.create_subscription(
            String, '/detected_objects', self.vision_callback, 10)
        self.location_sub = self.create_subscription(
            Pose, '/current_pose', self.location_callback, 10)

        # Publishers
        self.navigation_pub = self.create_publisher(
            Pose, '/move_base_simple/goal', 10)
        self.manipulation_pub = self.create_publisher(
            String, '/manipulation_command', 10)
        self.system_status_pub = self.create_publisher(
            String, '/system_status', 10)

        # Initialize components
        self.bridge = CvBridge()
        self.current_pose = Pose()
        self.detected_objects = []
        self.current_task = None
        self.task_queue = []

        # Action executors
        self.action_executors = {
            'navigation': self.execute_navigation,
            'manipulation': self.execute_manipulation,
            'information': self.execute_information,
            'social': self.execute_social
        }

        self.get_logger().info('Action execution system initialized')

    def action_plan_callback(self, msg):
        """Process action plan from language understanding"""
        try:
            plan_data = json.loads(msg.data)
            self.get_logger().info(f'Received action plan: {plan_data}')

            # Add to task queue
            self.task_queue.append(plan_data)

            # Execute next task if idle
            if self.current_task is None:
                self.execute_next_task()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in action plan: {e}')
        except Exception as e:
            self.get_logger().error(f'Action plan processing error: {e}')

    def vision_callback(self, msg):
        """Update with latest vision information"""
        try:
            # Parse detected objects (this would come from vision pipeline)
            # For now, we'll simulate updating the internal state
            pass
        except Exception as e:
            self.get_logger().error(f'Vision callback error: {e}')

    def location_callback(self, msg):
        """Update current robot location"""
        self.current_pose = msg

    def execute_next_task(self):
        """Execute the next task in the queue"""
        if not self.task_queue:
            return

        task = self.task_queue.pop(0)
        self.current_task = task

        action_type = task.get('type', 'unknown')
        executor = self.action_executors.get(action_type)

        if executor:
            self.get_logger().info(f'Executing {action_type} task: {task}')
            success = executor(task)

            if success:
                self.get_logger().info(f'{action_type.capitalize()} task completed successfully')
                self.publish_system_status(f'{action_type} completed')
            else:
                self.get_logger().error(f'{action_type.capitalize()} task failed')
                self.publish_system_status(f'{action_type} failed')
        else:
            self.get_logger().warning(f'Unknown action type: {action_type}')
            self.publish_system_status(f'Unknown action: {action_type}')

        # Mark current task as completed
        self.current_task = None

        # Execute next task if available
        if self.task_queue:
            self.execute_next_task()

    def execute_navigation(self, task):
        """Execute navigation task"""
        try:
            if 'destination' in task:
                # Look up destination coordinates (in real system, from map)
                # For simulation, use example coordinates
                destinations = {
                    'kitchen': (2.0, 1.0, 0.0),
                    'living room': (0.0, 2.0, 1.57),
                    'bedroom': (-1.0, -1.0, 3.14),
                    'office': (3.0, -2.0, -1.57)
                }

                dest_name = task['destination']
                if dest_name in destinations:
                    x, y, theta = destinations[dest_name]

                    goal_pose = Pose()
                    goal_pose.position.x = x
                    goal_pose.position.y = y
                    goal_pose.position.z = 0.0

                    # Convert theta to quaternion
                    from math import sin, cos
                    cy = cos(theta * 0.5)
                    sy = sin(theta * 0.5)
                    goal_pose.orientation.z = sy
                    goal_pose.orientation.w = cy

                    self.navigation_pub.publish(goal_pose)
                    self.get_logger().info(f'Navigating to {dest_name} at ({x}, {y})')
                    return True
                else:
                    self.get_logger().warning(f'Unknown destination: {dest_name}')
                    return False

            elif 'relative_pose' in task:
                rel_pose = task['relative_pose']

                # Calculate absolute goal based on current pose
                goal_pose = Pose()
                goal_pose.position.x = self.current_pose.position.x + rel_pose['x']
                goal_pose.position.y = self.current_pose.position.y + rel_pose['y']
                goal_pose.position.z = self.current_pose.position.z

                # For rotation, we'll use the specified theta directly
                from math import sin, cos
                theta = rel_pose['theta']
                cy = cos(theta * 0.5)
                sy = sin(theta * 0.5)
                goal_pose.orientation.z = sy
                goal_pose.orientation.w = cy

                self.navigation_pub.publish(goal_pose)
                self.get_logger().info(f'Moving relatively by ({rel_pose["x"]}, {rel_pose["y"]})')
                return True

        except Exception as e:
            self.get_logger().error(f'Navigation execution error: {e}')
            return False

    def execute_manipulation(self, task):
        """Execute manipulation task"""
        try:
            if 'target_object' in task:
                obj_name = task['target_object']

                # In a real system, this would:
                # 1. Locate the object using vision
                # 2. Plan grasp trajectory
                # 3. Execute manipulation

                # For simulation, publish command
                cmd_msg = String()
                cmd_msg.data = f'fetch_{obj_name}'
                self.manipulation_pub.publish(cmd_msg)

                self.get_logger().info(f'Attempting to manipulate {obj_name}')
                return True

        except Exception as e:
            self.get_logger().error(f'Manipulation execution error: {e}')
            return False

    def execute_information(self, task):
        """Execute information request task"""
        try:
            # Information tasks are typically handled by querying system state
            # and generating appropriate responses

            request = task.get('request', 'unknown')

            if request == 'environment_description':
                # This would integrate with perception system
                self.get_logger().info('Providing environment description')
                return True
            elif request == 'people_count':
                # Query people detection system
                self.get_logger().info('Reporting people count')
                return True
            else:
                self.get_logger().info(f'Handling information request: {request}')
                return True

        except Exception as e:
            self.get_logger().error(f'Information execution error: {e}')
            return False

    def execute_social(self, task):
        """Execute social action task"""
        try:
            action = task.get('action', 'unknown')

            # Social actions would control robot's expressive behaviors
            # like gestures, facial expressions, etc.

            self.get_logger().info(f'Performing social action: {action}')

            # In a real system, this might trigger:
            # - Head/hand gestures
            # - LED expressions
            # - Sound responses
            # - Facial expressions (if equipped)

            return True

        except Exception as e:
            self.get_logger().error(f'Social action execution error: {e}')
            return False

    def publish_system_status(self, status):
        """Publish system status"""
        status_msg = String()
        status_msg.data = status
        self.system_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    action_node = ActionExecutionNode()

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

## Multimodal Fusion and Decision Making

### Combining Vision, Language, and Action

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray
import json
import numpy as np
from collections import defaultdict, deque


class MultimodalFusionNode(Node):
    def __init__(self):
        super().__init__('multimodal_fusion')

        # Subscriptions
        self.vision_sub = self.create_subscription(
            String, '/detected_objects', self.vision_callback, 10)
        self.language_sub = self.create_subscription(
            String, '/speech_transcript', self.language_callback, 10)
        self.location_sub = self.create_subscription(
            Pose, '/current_pose', self.location_callback, 10)
        self.action_status_sub = self.create_subscription(
            String, '/system_status', self.action_status_callback, 10)

        # Publishers
        self.decision_pub = self.create_publisher(
            String, '/multimodal_decision', 10)
        self.attention_map_pub = self.create_publisher(
            MarkerArray, '/attention_map', 10)
        self.context_pub = self.create_publisher(
            String, '/context_state', 10)

        # Context and state management
        self.context = {
            'objects': {},  # Detected objects with positions
            'people': {},   # Detected people with positions
            'conversations': deque(maxlen=10),  # Recent conversations
            'recent_actions': deque(maxlen=10),  # Recent actions
            'current_pose': Pose(),
            'environment_map': {},
            'intentions': []  # Current intentions/priorities
        }

        # Confidence thresholds
        self.object_confidence_threshold = 0.7
        self.action_confidence_threshold = 0.6

        # Temporal windows
        self.short_term_window = 5.0  # seconds
        self.long_term_window = 60.0  # seconds

        # Decision making parameters
        self.importance_weights = {
            'safety': 10.0,
            'task_relevance': 5.0,
            'social_interaction': 3.0,
            'curiosity': 1.0
        }

        # Initialize timers
        self.decision_timer = self.create_timer(0.5, self.make_decisions)  # 2Hz decision making
        self.context_update_timer = self.create_timer(1.0, self.update_context)  # 1Hz context updates

        self.get_logger().info('Multimodal fusion system initialized')

    def vision_callback(self, msg):
        """Process visual input and update context"""
        try:
            # In a real system, this would come from the vision pipeline
            # For simulation, we'll parse a simple format
            # Real implementation would process MarkerArray or similar

            # Update object detections in context
            # This is a simplified representation
            detected_objects_str = msg.data

            # Parse object detections (in real system, this would be structured data)
            # For now, we'll just log and update context
            self.get_logger().debug(f'Vision input: {detected_objects_str}')

            # In a real system, update self.context['objects'] with new detections
            # including positions, confidences, and timestamps

        except Exception as e:
            self.get_logger().error(f'Vision callback error: {e}')

    def language_callback(self, msg):
        """Process language input and update context"""
        try:
            transcript = msg.data

            if transcript.strip():
                # Add to conversation history
                conversation_entry = {
                    'text': transcript,
                    'timestamp': self.get_clock().now().nanoseconds / 1e9,
                    'type': 'user_input'
                }

                self.context['conversations'].append(conversation_entry)

                # Update intentions based on language input
                self.update_intentions_from_language(transcript)

                self.get_logger().info(f'Language input: {transcript}')

        except Exception as e:
            self.get_logger().error(f'Language callback error: {e}')

    def location_callback(self, msg):
        """Update current location"""
        self.context['current_pose'] = msg

    def action_status_callback(self, msg):
        """Process action status updates"""
        try:
            status = msg.data

            # Add to recent actions
            action_entry = {
                'status': status,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            }

            self.context['recent_actions'].append(action_entry)

            # Update intentions based on action outcomes
            self.update_intentions_from_action(status)

        except Exception as e:
            self.get_logger().error(f'Action status callback error: {e}')

    def update_intentions_from_language(self, text):
        """Update intentions based on language input"""
        text_lower = text.lower()

        # Extract intentions from language
        new_intentions = []

        # Navigation intentions
        navigation_keywords = ['go', 'move', 'walk', 'navigate', 'come', 'follow']
        if any(keyword in text_lower for keyword in navigation_keywords):
            new_intentions.append({
                'type': 'navigation',
                'priority': 8,
                'target': self.extract_target_location(text),
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })

        # Manipulation intentions
        manipulation_keywords = ['get', 'bring', 'fetch', 'pick', 'take', 'hand']
        if any(keyword in text_lower for keyword in manipulation_keywords):
            new_intentions.append({
                'type': 'manipulation',
                'priority': 7,
                'target': self.extract_target_object(text),
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })

        # Information intentions
        info_keywords = ['what', 'where', 'who', 'how', 'describe']
        if any(keyword in text_lower for keyword in info_keywords):
            new_intentions.append({
                'type': 'information',
                'priority': 5,
                'query': text,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })

        # Add new intentions
        self.context['intentions'].extend(new_intentions)

        # Keep only recent intentions (last 30 seconds)
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.context['intentions'] = [
            intent for intent in self.context['intentions']
            if current_time - intent['timestamp'] < 30.0
        ]

    def update_intentions_from_action(self, status):
        """Update intentions based on action outcomes"""
        # Adjust intentions based on action success/failure
        if 'failed' in status.lower():
            # Lower priority of related intentions
            for intent in self.context['intentions']:
                if status.split()[0] in intent.get('type', ''):  # Crude matching
                    intent['priority'] = max(1, intent['priority'] - 2)

    def extract_target_location(self, text):
        """Extract target location from text"""
        # Simple location extraction (in real system, use NER/SLU)
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'dining room', 'bathroom', 'hallway']

        for loc in locations:
            if loc in text.lower():
                return loc

        return 'unknown_location'

    def extract_target_object(self, text):
        """Extract target object from text"""
        # Simple object extraction (in real system, use NER/SLU)
        objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'ball', 'toy', 'plate', 'fork']

        for obj in objects:
            if obj in text.lower():
                return obj

        return 'unknown_object'

    def make_decisions(self):
        """Make decisions based on multimodal context"""
        if not self.context['intentions']:
            return  # No intentions to act on

        # Rank intentions by priority and relevance
        ranked_intentions = self.rank_intentions()

        if ranked_intentions:
            top_intention = ranked_intentions[0]

            # Create decision
            decision = {
                'selected_intention': top_intention,
                'confidence': self.calculate_decision_confidence(top_intention),
                'timestamp': self.get_clock().now().nanoseconds / 1e9,
                'context_snapshot': self.get_context_snapshot()
            }

            # Publish decision
            decision_msg = String()
            decision_msg.data = json.dumps(decision, indent=2)
            self.decision_pub.publish(decision_msg)

            self.get_logger().info(f'Decision made: {top_intention["type"]} to {top_intention.get("target", "unknown")}')

    def rank_intentions(self):
        """Rank intentions by priority and context relevance"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        ranked = []
        for intent in self.context['intentions']:
            # Calculate relevance score
            age_factor = max(0.1, 1.0 - (current_time - intent['timestamp']) / 30.0)  # Decay over 30s
            priority_score = intent['priority'] * age_factor

            ranked.append((priority_score, intent))

        # Sort by score (descending)
        ranked.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in ranked]

    def calculate_decision_confidence(self, intention):
        """Calculate confidence in the decision"""
        # Base confidence on intention priority and supporting evidence
        base_confidence = min(1.0, intention['priority'] / 10.0)

        # Boost confidence if there's visual evidence supporting the intention
        supporting_evidence = self.count_supporting_evidence(intention)
        evidence_boost = min(0.3, supporting_evidence * 0.1)

        confidence = base_confidence + evidence_boost
        return min(1.0, confidence)

    def count_supporting_evidence(self, intention):
        """Count visual/evidential support for an intention"""
        count = 0

        # Check if target object/location is visible
        if intention['type'] == 'manipulation' and 'target' in intention:
            target_obj = intention['target']
            # In real system, check self.context['objects'] for target_obj
            count += 1  # Simulated evidence

        elif intention['type'] == 'navigation' and 'target' in intention:
            target_loc = intention['target']
            # In real system, check if target location is known/reachable
            count += 1  # Simulated evidence

        return count

    def update_context(self):
        """Update context state and publish"""
        # Clean up old context elements
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Publish current context
        context_msg = String()
        context_msg.data = json.dumps(self.get_context_snapshot(), indent=2)
        self.context_pub.publish(context_msg)

    def get_context_snapshot(self):
        """Get a snapshot of current context for decision making"""
        return {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'num_objects': len(self.context['objects']),
            'num_people': len(self.context['people']),
            'num_conversations': len(self.context['conversations']),
            'num_recent_actions': len(self.context['recent_actions']),
            'num_intentions': len(self.context['intentions']),
            'current_pose': {
                'x': self.context['current_pose'].position.x,
                'y': self.context['current_pose'].position.y,
                'z': self.context['current_pose'].position.z
            }
        }

    def publish_attention_map(self):
        """Publish attention map showing focused areas"""
        # This would create visualization markers showing where the robot is focusing
        # based on multimodal inputs
        pass


def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultimodalFusionNode()

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

## Integration Example: Complete VLA System

### Putting it all together

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose
import json


class VLASystemNode(Node):
    def __init__(self):
        super().__init__('vla_system')

        # Initialize all components
        self.vision_component = VisionPipeline()
        self.speech_component = WhisperSpeechProcessor()
        self.language_component = LanguageUnderstandingNode()
        self.action_component = ActionExecutionNode()
        self.fusion_component = MultimodalFusionNode()

        # System monitoring
        self.system_status = {
            'vision_operational': True,
            'speech_operational': True,
            'language_operational': True,
            'action_operational': True,
            'fusion_operational': True
        }

        # Self-monitoring timer
        self.health_check_timer = self.create_timer(5.0, self.system_health_check)

        self.get_logger().info('Complete VLA system initialized')

    def system_health_check(self):
        """Check health of all VLA components"""
        # In a real system, this would check if all components are responsive
        operational_count = sum(self.system_status.values())
        total_components = len(self.system_status)

        self.get_logger().info(f'System health: {operational_count}/{total_components} components operational')

        # Log non-operational components
        for component, operational in self.system_status.items():
            if not operational:
                self.get_logger().warning(f'{component} is not operational')


def main(args=None):
    rclpy.init(args=args)
    vla_system = VLASystemNode()

    try:
        # Since all components are initialized as part of the system,
        # we just spin the main node which will handle all callbacks
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Performance Considerations

### Optimizing VLA Systems

```python
# performance_optimization.py

import time
import threading
import queue
from functools import wraps


def timing_decorator(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


class VLAOptimizer:
    def __init__(self):
        self.component_timings = {}
        self.resource_usage = {}
        self.throughput_stats = {}

    def optimize_processing_pipeline(self):
        """Optimize the processing pipeline for real-time performance"""
        # 1. Asynchronous processing where possible
        # 2. Batch processing for ML models
        # 3. Caching of expensive computations
        # 4. Early termination of low-confidence detections

        # Example: Asynchronous vision processing
        self.vision_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.vision_thread = threading.Thread(target=self.async_vision_process)
        self.vision_thread.daemon = True
        self.vision_thread.start()

    def async_vision_process(self):
        """Process vision data asynchronously"""
        while True:
            try:
                # Get vision data with timeout to avoid blocking
                vision_data = self.vision_queue.get(timeout=1.0)

                # Process with optimized parameters
                result = self.optimized_vision_process(vision_data)

                # Publish result
                self.publish_vision_result(result)

                self.vision_queue.task_done()
            except queue.Empty:
                continue  # Timeout, continue loop

    def optimized_vision_process(self, data):
        """Optimized vision processing with early termination"""
        # 1. Quick preprocessing to eliminate obvious negatives
        # 2. Multi-scale processing starting with coarse
        # 3. Adaptive confidence thresholds
        # 4. Spatial-temporal consistency checks

        # Example optimizations:
        # - Use ROI (Region of Interest) based on previous detections
        # - Lower resolution processing for far objects
        # - Temporal consistency to avoid flickering detections

        return self.process_with_optimizations(data)

    def adaptive_resource_allocation(self):
        """Dynamically allocate resources based on task demands"""
        # Monitor current system load and adjust processing parameters
        # - Reduce image resolution when CPU is overloaded
        # - Lower model complexity when memory is constrained
        # - Adjust update rates based on importance

        pass

    def prioritize_real_time_tasks(self):
        """Ensure critical tasks meet real-time deadlines"""
        # Set thread priorities
        # Use real-time scheduling where available
        # Implement soft real-time guarantees

        pass
```

## Testing and Evaluation

### Evaluating VLA System Performance

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
import time
import statistics


class VLAEvaluationNode(Node):
    def __init__(self):
        super().__init__('vla_evaluation')

        # Subscriptions for monitoring system performance
        self.decision_sub = self.create_subscription(
            String, '/multimodal_decision', self.decision_callback, 10)
        self.response_sub = self.create_subscription(
            String, '/robot_response', self.response_callback, 10)
        self.system_status_sub = self.create_subscription(
            String, '/system_status', self.system_status_callback, 10)

        # Publishers for evaluation metrics
        self.accuracy_pub = self.create_publisher(Float32, '/evaluation/accuracy', 10)
        self.latency_pub = self.create_publisher(Float32, '/evaluation/latency', 10)
        self.throughput_pub = self.create_publisher(Float32, '/evaluation/throughput', 10)

        # Evaluation metrics
        self.command_start_times = {}
        self.evaluation_results = {
            'accuracy': [],
            'latency': [],
            'throughput': [],
            'success_rate': []
        }

        # Evaluation timer
        self.evaluation_timer = self.create_timer(10.0, self.compute_evaluation_metrics)

        self.get_logger().info('VLA evaluation system initialized')

    def decision_callback(self, msg):
        """Track decision-making performance"""
        try:
            decision_data = json.loads(msg.data)
            command_id = decision_data.get('timestamp', time.time())

            # Record decision start time
            self.command_start_times[command_id] = time.time()

        except Exception as e:
            self.get_logger().error(f'Decision callback error: {e}')

    def response_callback(self, msg):
        """Track response performance"""
        # Calculate round-trip time for user satisfaction
        pass

    def system_status_callback(self, msg):
        """Track system status for reliability metrics"""
        status = msg.data
        if 'completed' in status.lower():
            # Calculate completion time
            pass

    def compute_evaluation_metrics(self):
        """Compute and publish evaluation metrics"""
        if not self.evaluation_results['latency']:
            return

        # Calculate average metrics
        avg_latency = statistics.mean(self.evaluation_results['latency'])
        avg_accuracy = statistics.mean(self.evaluation_results['accuracy']) if self.evaluation_results['accuracy'] else 0.0
        avg_throughput = statistics.mean(self.evaluation_results['throughput']) if self.evaluation_results['throughput'] else 0.0

        # Publish metrics
        latency_msg = Float32()
        latency_msg.data = float(avg_latency)
        self.latency_pub.publish(latency_msg)

        accuracy_msg = Float32()
        accuracy_msg.data = float(avg_accuracy)
        self.accuracy_pub.publish(accuracy_msg)

        throughput_msg = Float32()
        throughput_msg.data = float(avg_throughput)
        self.throughput_pub.publish(throughput_msg)

        self.get_logger().info(f'Evaluation - Latency: {avg_latency:.3f}s, Accuracy: {avg_accuracy:.3f}, Throughput: {avg_throughput:.2f} cmds/s')


def main(args=None):
    rclpy.init(args=args)
    eval_node = VLAEvaluationNode()

    try:
        rclpy.spin(eval_node)
    except KeyboardInterrupt:
        pass
    finally:
        eval_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Troubleshooting Common Issues

### Common VLA System Problems and Solutions

#### Issue 1: High Latency in Speech Recognition
**Symptoms**: Long delays between speech input and response
**Causes**:
- Large ML models
- Insufficient computing resources
- Blocking I/O operations
**Solutions**:
- Use smaller, faster models (Whisper tiny/base)
- Implement asynchronous processing
- Optimize audio buffering
- Use GPU acceleration where available

#### Issue 2: Poor Object Detection Accuracy
**Symptoms**: Frequent misidentifications or missed objects
**Causes**:
- Insufficient training data
- Lighting conditions
- Occlusions
**Solutions**:
- Fine-tune models on domain-specific data
- Use multi-camera fusion
- Implement temporal consistency checks
- Add domain randomization during training

#### Issue 3: Inconsistent Action Execution
**Symptoms**: Same command produces different actions
**Causes**:
- Ambiguous language understanding
- Inconsistent context
- State estimation errors
**Solutions**:
- Implement explicit confirmation dialogs
- Maintain consistent context state
- Use probabilistic state estimation
- Add redundancy in critical commands

## Best Practices for VLA Systems

### 1. Modularity and Scalability
- Design components as independent modules
- Use standardized interfaces between components
- Implement proper error handling and isolation
- Design for easy addition of new capabilities

### 2. Safety and Robustness
- Implement safety constraints in action planning
- Use confidence thresholds for uncertain situations
- Design graceful degradation for component failures
- Include human oversight capabilities

### 3. Real-time Performance
- Optimize critical paths for real-time execution
- Use asynchronous processing where appropriate
- Implement priority-based task scheduling
- Monitor system resources continuously

### 4. User Experience
- Provide clear feedback about system state
- Implement natural conversation flows
- Use multimodal feedback (speech, gesture, lights)
- Learn from user interactions over time

## Key Takeaways

- VLA systems integrate perception, language, and action for natural human-robot interaction
- Whisper provides powerful speech recognition for voice commands
- LLMs enable sophisticated language understanding and generation
- Multimodal fusion combines different sensory inputs for better decisions
- Real-time performance optimization is critical for responsive behavior
- Proper evaluation metrics ensure system effectiveness
- Safety and robustness must be prioritized in all components

In the next section, we'll explore how to connect these perception and language capabilities to create comprehensive action execution pipelines that enable humanoid robots to perform complex tasks in real-world environments.